import datetime
from pathlib import Path
import math
import time
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.metrics import MR, minADE, minFDE, brier_minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2
from src.utils.LaplaceNLLLoss import LaplaceNLLLoss
from src.utils.VonMisesNLLLoss import VonMisesNLLLoss
from .model_forecast import ModelForecast

import os
import matplotlib.pyplot as plt
import numpy as np

from typing import List

model_dict = {
    'ModelForecast': ModelForecast,
}


class Trainer(pl.LightningModule):
    def __init__(
        self,
        model: dict,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        ws_offset: List = [0.3, 1.0],
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        self.model_type = model.pop('type')
        assert self.model_type in model_dict
        self.net = model_dict[self.model_type](**model)

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print('Pretrained weights have been loaded.')

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
                "b-minFDE6": brier_minFDE(k=6),
            }
        )
        self.loss_fn_pos = LaplaceNLLLoss(reduction='none')
        self.loss_fn_head = VonMisesNLLLoss(reduction='none')
        self.val_metrics = metrics.clone(prefix="val_")
        self.val_metrics_new = metrics.clone(prefix="val_new_")

        self.ws_offset = ws_offset

        self.total_time = 0
        self.cur_time = 0

    def forward(self, data):
        return self.net(data)

    # ------------------------------------------------------------------ #
    #                      Decoder Monitoring Helpers                      #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _log_decoder_diagnostics(self, out, data, tag="train"):
        """Log diagnostics for the DONUT-Mamba autoregressive decoder."""
        y = data["target"][:, 0]  # [B, 60, 2]
        y_hat = out["y_hat"]      # [B, 6, 60, 2]  proposer
        new_y_hat = out["new_y_hat"]  # [B, 6, 60, 2]  refiner
        pi = out["pi"]            # [B, 6]
        new_pi = out["new_pi"]    # [B, 6]
        scal = out["scal"]        # [B, 6, 60, 2]
        scal_new = out["scal_new"]

        B = y.shape[0]

        # --- 1. Proposer vs Refiner minFDE ---
        prop_fde = torch.norm(y_hat[:, :, -1] - y[:, None, -1], dim=-1)  # [B, 6]
        prop_min_fde = prop_fde.min(dim=1)[0].mean()

        ref_fde = torch.norm(new_y_hat[:, :, -1] - y[:, None, -1], dim=-1)
        ref_min_fde = ref_fde.min(dim=1)[0].mean()

        refiner_improvement = prop_min_fde - ref_min_fde  # positive = refiner helps

        self.log(f"{tag}/prop_minFDE", prop_min_fde, on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{tag}/ref_minFDE", ref_min_fde, on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{tag}/refiner_delta", refiner_improvement, on_step=False, on_epoch=True, batch_size=B)

        # --- 2. Mode usage distribution ---
        # Which mode gets selected as best (by refiner output)
        ref_l2 = torch.norm(new_y_hat - y[:, None], dim=-1).sum(dim=-1)  # [B, 6]
        best_modes = ref_l2.argmin(dim=1)  # [B]
        for m in range(new_y_hat.shape[1]):
            mode_freq = (best_modes == m).float().mean()
            self.log(f"{tag}/mode_{m}_freq", mode_freq, on_step=False, on_epoch=True, batch_size=B)

        # Mode probability entropy (high = diverse, low = collapsed)
        probs = F.softmax(new_pi, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        self.log(f"{tag}/mode_entropy", entropy, on_step=False, on_epoch=True, batch_size=B)

        # --- 3. Scale statistics ---
        self.log(f"{tag}/prop_scale_mean", scal.mean(), on_step=False, on_epoch=True, batch_size=B)
        self.log(f"{tag}/ref_scale_mean", scal_new.mean(), on_step=False, on_epoch=True, batch_size=B)

        # --- 4. Per-token-step ADE (split 60 into 6 groups of 10) ---
        # Use refiner's best mode
        best_ref = new_y_hat[torch.arange(B), best_modes]  # [B, 60, 2]
        err_per_step = torch.norm(best_ref - y, dim=-1)    # [B, 60]
        for s in range(6):
            start = s * 10
            end = start + 10
            step_ade = err_per_step[:, start:end].mean()
            self.log(f"{tag}/step_{s}_ade", step_ade, on_step=False, on_epoch=True, batch_size=B)

        # --- 5. Heading error ---
        if "heading_hat" in out and "target_heading" in data:
            from .layers.coordinate_transforms import wrap_angle
            gt_head = data["target_heading"][:, 0]  # [B, 60]
            pred_head = out["new_heading_hat"][torch.arange(B), best_modes]  # [B, 60]
            head_err = wrap_angle(pred_head - gt_head).abs()
            self.log(f"{tag}/heading_mae_deg", head_err.mean() * 180 / math.pi,
                     on_step=False, on_epoch=True, batch_size=B)

    @torch.no_grad()
    def _grad_sq_norm(self, grad: torch.Tensor) -> torch.Tensor:
        if grad.is_sparse:
            grad = grad.coalesce().values()
        grad = grad.detach().float()
        return torch.sum(grad * grad)

    @torch.no_grad()
    def _log_gradient_norms(self):
        """Log decoder gradient health with a focus on BiMamba and clipping pressure."""
        device = self.device
        global_sq = torch.zeros((), device=device)
        proposer_bimamba_sq = torch.zeros((), device=device)
        refiner_bimamba_sq = torch.zeros((), device=device)

        for name, param in self.net.named_parameters():
            if param.grad is None:
                continue

            grad_sq = self._grad_sq_norm(param.grad)
            global_sq = global_sq + grad_sq

            if name.startswith("time_decoder.proposer.bimamba_"):
                proposer_bimamba_sq = proposer_bimamba_sq + grad_sq
            elif name.startswith("time_decoder.refiner.bimamba_"):
                refiner_bimamba_sq = refiner_bimamba_sq + grad_sq

        global_grad_norm = torch.sqrt(global_sq)
        proposer_bimamba_grad_norm = torch.sqrt(proposer_bimamba_sq)
        refiner_bimamba_grad_norm = torch.sqrt(refiner_bimamba_sq)

        clip_val = getattr(self.trainer, "gradient_clip_val", None)
        if clip_val is None or clip_val <= 0:
            clip_indicator = torch.zeros((), device=device)
            clip_ratio = torch.zeros((), device=device)
        else:
            clip_threshold = torch.tensor(float(clip_val), device=device)
            clip_indicator = (global_grad_norm > clip_threshold).float()
            clip_ratio = global_grad_norm / clip_threshold.clamp_min(1e-12)

        self.log(
            "train/global_grad_norm",
            global_grad_norm,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/proposer_bimamba_grad_norm",
            proposer_bimamba_grad_norm,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/refiner_bimamba_grad_norm",
            refiner_bimamba_grad_norm,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/grad_clip_indicator",
            clip_indicator,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train/grad_clip_ratio",
            clip_ratio,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

    # ------------------------------------------------------------------ #
    #                            Loss Computation                         #
    # ------------------------------------------------------------------ #

    def compute_reg_loss(self, best_pred_pos, best_scale_pos, gt_pos):
        """Laplace NLL on best mode, x/y computed separately (DONUT style)."""
        # x position NLL
        best_x = torch.cat([best_pred_pos[..., :1], best_scale_pos[..., :1]], dim=-1)
        loss = self.loss_fn_pos(best_x, gt_pos[..., :1])  # [B, 60, 1]

        # y position NLL
        best_y = torch.cat([best_pred_pos[..., 1:], best_scale_pos[..., 1:]], dim=-1)
        loss = loss + self.loss_fn_pos(best_y, gt_pos[..., 1:])  # [B, 60, 1]

        return loss.mean()

    def compute_heading_reg_loss(self, best_pred_head, best_conc_head, gt_head):
        """Von Mises NLL on heading for the best mode."""
        pred = torch.stack([best_pred_head, best_conc_head], dim=-1)  # [B, 60, 2]
        target = gt_head.unsqueeze(-1)  # [B, 60, 1]
        loss = self.loss_fn_head(pred, target)  # [B, 60, 1]
        return loss.mean()

    def compute_cls_nll(self, pred_pos, scale_pos, gt_pos,
                        pred_head=None, conc_head=None, gt_head=None):
        """NLL for ALL modes at last timestep (no gradient, for mixture cls loss)."""
        with torch.no_grad():
            # x at last timestep
            last_x = torch.cat([pred_pos[:, :, -1:, :1], scale_pos[:, :, -1:, :1]], dim=-1)
            nll = self.loss_fn_pos(last_x, gt_pos[:, None, -1:, :1])

            # y at last timestep
            last_y = torch.cat([pred_pos[:, :, -1:, 1:], scale_pos[:, :, -1:, 1:]], dim=-1)
            nll = nll + self.loss_fn_pos(last_y, gt_pos[:, None, -1:, 1:])

            # heading at last timestep (if provided)
            if pred_head is not None and conc_head is not None and gt_head is not None:
                last_head = torch.stack([
                    pred_head[:, :, -1:], conc_head[:, :, -1:]
                ], dim=-1)  # [B, 6, 1, 2]
                nll_head = self.loss_fn_head(last_head, gt_head[:, None, -1:, None])
                nll = nll + nll_head

        return nll[:, :, 0, 0].detach()  # [B, 6]

    def cal_loss(self, out, data, tag=''):
        y_hat, pi = out["y_hat"], out["pi"]
        new_y_hat, new_pi = out["new_y_hat"], out["new_pi"]
        scal, scal_new = out["scal"], out["scal_new"]
        y_hat_others = out["y_hat_others"]
        ep_offsets = out.get("ep_offsets", None)
        heading_hat = out["heading_hat"]            # [B, 6, 60]
        conc_hat = out["conc_hat"]                  # [B, 6, 60]
        new_heading_hat = out["new_heading_hat"]    # [B, 6, 60]
        new_conc_hat = out["new_conc_hat"]          # [B, 6, 60]

        y, y_others = data["target"][:, 0], data["target"][:, 1:]
        center = data["x_centers"][:, 0]
        gt_heading = data["target_heading"][:, 0]   # [B, 60]
        B = y.shape[0]
        index0 = torch.arange(B, device=y.device)

        # --- Best mode selection: proposer oracle shared by both stages ---
        best_mode = torch.linalg.norm(y_hat - y[:, None], dim=-1).mean(-1).argmin(dim=-1)  # [B]

        # --- Regression loss: proposer (Laplace NLL) ---
        reg_loss_prop = self.compute_reg_loss(
            y_hat[index0, best_mode], scal[index0, best_mode], y)

        # --- Regression loss: refiner (Laplace NLL) ---
        reg_loss_ref = self.compute_reg_loss(
            new_y_hat[index0, best_mode], scal_new[index0, best_mode], y)

        # --- Heading regression loss: proposer (Von Mises NLL) ---
        heading_reg_prop = self.compute_heading_reg_loss(
            heading_hat[index0, best_mode], conc_hat[index0, best_mode], gt_heading)

        # --- Heading regression loss: refiner ---
        heading_reg_ref = self.compute_heading_reg_loss(
            new_heading_hat[index0, best_mode], new_conc_hat[index0, best_mode], gt_heading)

        # --- Over-prediction loss (shifted GT, shared best mode) ---
        over_loss = torch.tensor(0.0, device=y.device)
        t_per_tok = self.net.t_per_tok if hasattr(self.net, 't_per_tok') else 10
        shift = t_per_tok  # =10
        t_pred = y.shape[1]  # =60
        over_len = t_pred - shift  # =50

        for over_key_prefix, bm in [("", best_mode), ("new_", best_mode)]:
            y_over = out.get(f"{over_key_prefix}y_hat_over")
            s_over = out.get(f"{over_key_prefix}scal_over")
            h_over = out.get(f"{over_key_prefix}heading_over")
            c_over = out.get(f"{over_key_prefix}conc_over")
            if y_over is not None and y_over.shape[2] >= over_len:
                over_loss = over_loss + self.compute_reg_loss(
                    y_over[index0, bm, :over_len],
                    s_over[index0, bm, :over_len],
                    y[:, shift:])
                if h_over is not None and h_over.shape[2] >= over_len:
                    over_loss = over_loss + self.compute_heading_reg_loss(
                        h_over[index0, bm, :over_len],
                        c_over[index0, bm, :over_len],
                        gt_heading[:, shift:])

        # --- Classification loss: proposer and refiner each with own pi/nll ---
        nll_prop = self.compute_cls_nll(y_hat, scal, y, heading_hat, conc_hat, gt_heading)
        cls_loss_prop = -torch.logsumexp(F.log_softmax(pi, dim=-1) - nll_prop, dim=-1).mean()

        nll_ref = self.compute_cls_nll(new_y_hat, scal_new, y, new_heading_hat, new_conc_hat, gt_heading)
        cls_loss_ref = -torch.logsumexp(F.log_softmax(new_pi, dim=-1) - nll_ref, dim=-1).mean()

        cls_loss = cls_loss_prop + cls_loss_ref

        # --- Endpoint offset loss (encoder auxiliary, unchanged) ---
        if ep_offsets is not None:
            gt_offsets = y[:, -1] - center
            ep_reg_loss = 0
            if isinstance(ep_offsets, list):
                for w, pred in zip(self.ws_offset, ep_offsets):
                    ep_reg_loss = ep_reg_loss + w * F.smooth_l1_loss(pred, gt_offsets)
            else:
                ep_reg_loss = F.smooth_l1_loss(ep_offsets, gt_offsets)
        else:
            ep_reg_loss = 0

        # --- Other agents loss (encoder auxiliary, unchanged) ---
        others_reg_mask = data["target_mask"][:, 1:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask])

        # --- Total ---
        loss = (reg_loss_prop + reg_loss_ref + cls_loss
                + heading_reg_prop + heading_reg_ref
                + over_loss
                + others_reg_loss)
        if torch.is_tensor(ep_reg_loss):
            loss = loss + ep_reg_loss

        disp_dict = {
            f"{tag}loss": loss.item(),
            f"{tag}reg_loss_prop": reg_loss_prop.item(),
            f"{tag}reg_loss_ref": reg_loss_ref.item(),
            f"{tag}cls_loss_prop": cls_loss_prop.item(),
            f"{tag}cls_loss_ref": cls_loss_ref.item(),
            f"{tag}heading_reg_prop": heading_reg_prop.item(),
            f"{tag}heading_reg_ref": heading_reg_ref.item(),
            f"{tag}over_loss": over_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
        }
        if ep_offsets is not None and torch.is_tensor(ep_reg_loss):
            disp_dict[f"{tag}ep_reg_loss"] = ep_reg_loss.item()

        return loss, disp_dict

    # ------------------------------------------------------------------ #
    #                         Training / Validation                       #
    # ------------------------------------------------------------------ #

    def training_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]

        out = self(data)
        loss, loss_dict = self.cal_loss(out, data)

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}", v,
                on_step=True, on_epoch=True, prog_bar=False, sync_dist=True,
            )

        # Decoder diagnostics (every 100 steps to avoid overhead)
        if batch_idx % 100 == 0:
            self._log_decoder_diagnostics(out, data, tag="train")

        return loss

    def on_after_backward(self):
        self._log_gradient_norms()

    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        _, loss_dict = self.cal_loss(out, data)

        # Standard metrics on proposer output
        metrics = self.val_metrics(out, data['target'][:, 0])

        # Standard metrics on refiner output
        if out['new_y_hat'] is not None:
            out_new = {**out, 'y_hat': out['new_y_hat'], 'pi': out['new_pi']}
            metrics_new = self.val_metrics_new(out_new, data['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True, on_step=False, on_epoch=True,
            batch_size=1, sync_dist=True,
        )
        if out['new_y_hat'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True, on_step=False, on_epoch=True,
                batch_size=1, sync_dist=True,
            )

        # Decoder diagnostics on validation
        self._log_decoder_diagnostics(out, data, tag="val")

    # ------------------------------------------------------------------ #
    #                              Testing                                #
    # ------------------------------------------------------------------ #

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2(save_dir=save_dir)

    def test_step(self, data, batch_idx) -> None:
        if isinstance(data, list):
            data = data[-1]

        torch.cuda.synchronize()
        start_time = time.time()
        out = self(data)
        torch.cuda.synchronize()
        end_time = time.time()

        self.total_time += end_time - start_time
        self.cur_time += 1

        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    # ------------------------------------------------------------------ #
    #                             Optimizer                               #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.MultiheadAttention, nn.LSTM, nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.SyncBatchNorm, nn.LayerNorm, nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if not param.requires_grad:
                    continue
                if param_name.endswith("bias"):
                    no_decay.add(full_param_name)
                elif param_name.endswith("weight"):
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    else:
                        no_decay.add(full_param_name)
                else:
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        missing_params = set(param_dict.keys()) - union_params
        no_decay.update(missing_params)

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-5,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
