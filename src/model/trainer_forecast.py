import datetime
from pathlib import Path
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
from .model_forecast import ModelForecast

import os
import matplotlib.pyplot as plt
import numpy as np

from typing import List

model_dict = {
    'ModelForecast': ModelForecast,  # only 'FINet'
}


class Trainer(pl.LightningModule):
    def __init__(
        self,
        model: dict,
        pretrained_weights: str = None,
        lr: float = 1.4e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-2,
        ws_offset: List = [0.0, 1.0],
        goal_tau: float = None,
        lambda_goal_cls: float = None,
        lambda_goal_reg: float = None,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        # copy mutable config to avoid side effects
        model = dict(model)
        # optional compatibility: allow putting goal loss hparams under model cfg
        model_goal_tau = model.pop('goal_tau', None)
        model_lambda_goal_cls = model.pop('lambda_goal_cls', None)
        model_lambda_goal_reg = model.pop('lambda_goal_reg', None)

        self.model_type = model.pop('type')
        assert self.model_type in model_dict
        self.net = model_dict[self.model_type](**model)

        # self.net = self.get_model(model_type)(**model)

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
        self.laplace_loss = LaplaceNLLLoss()
        self.val_metrics = metrics.clone(prefix="val_")
        self.val_metrics_new = metrics.clone(prefix="val_new_")
        
        self.ws_offset = ws_offset

        self.total_time=0
        self.cur_time = 0
        self.test_step_count = 0
        self.val_exception_count = 0

        # goal auxiliary loss weights (with safe defaults / fallback)
        self.goal_tau = self._resolve_loss_hparam(
            primary=goal_tau,
            secondary=model_goal_tau,
            default=2.0,
            min_value=1e-3,
        )
        self.lambda_goal_cls = self._resolve_loss_hparam(
            primary=lambda_goal_cls,
            secondary=model_lambda_goal_cls,
            default=0.1,
            min_value=0.0,
        )
        self.lambda_goal_reg = self._resolve_loss_hparam(
            primary=lambda_goal_reg,
            secondary=model_lambda_goal_reg,
            default=0.05,
            min_value=0.0,
        )

        self.count = np.zeros(6)
        self.count_closet = np.zeros(6)
        
    

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        memory_dict = None
        predictions = []
        probs = []
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            memory_dict = out['memory_dict']
            prediction, prob = self.submission_handler.format_data(
                cur_data, out["y_hat"], out["pi"], inference=True)
            predictions.append(prediction)
            probs.append(prob)

        return predictions, probs

    @staticmethod
    def _resolve_loss_hparam(
        primary,
        secondary,
        default: float,
        min_value: float = 0.0,
    ) -> float:
        value = primary if primary is not None else secondary
        if value is None:
            value = default
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float(default)
        return max(value, min_value)

    def get_gt_endpoint(self, y_gt: torch.Tensor, y_mask: torch.Tensor = None):
        """
        Extract GT endpoint from future trajectory:
        1) use last valid future point if mask is provided;
        2) fallback to last timestep otherwise.
        """
        if y_gt is None or not torch.is_tensor(y_gt):
            return None, None

        if y_gt.dim() == 2 and y_gt.size(-1) >= 2:
            gt_endpoint = y_gt[..., :2]
            endpoint_valid = torch.isfinite(gt_endpoint).all(dim=-1)
            return gt_endpoint, endpoint_valid

        if y_gt.dim() < 3 or y_gt.size(1) <= 0:
            return None, None

        fallback_endpoint = y_gt[:, -1, :2]
        bsz = y_gt.size(0)
        device = y_gt.device

        valid_mask = None
        if y_mask is not None and torch.is_tensor(y_mask):
            valid_mask = y_mask.bool()
            future_steps = y_gt.size(1)

            # normalize mask to [B, T] for robust endpoint extraction
            # supports shapes like [B, T], [B, T, C], [B, A, T], [B, A, T, C]
            if valid_mask.dim() == 2:
                if valid_mask.size(1) != future_steps:
                    if (
                        valid_mask.size(0) == future_steps
                        and valid_mask.size(1) == y_gt.size(0)
                    ):
                        valid_mask = valid_mask.transpose(0, 1)
                    else:
                        valid_mask = None
            elif valid_mask.dim() >= 3:
                time_dim = None
                for dim_i in range(1, valid_mask.dim()):
                    if valid_mask.size(dim_i) == future_steps:
                        time_dim = dim_i
                        break
                if time_dim is None:
                    valid_mask = None
                else:
                    if time_dim != 1:
                        perm = [0, time_dim] + [d for d in range(1, valid_mask.dim()) if d != time_dim]
                        valid_mask = valid_mask.permute(*perm)
                    reduce_dims = tuple(range(2, valid_mask.dim()))
                    if len(reduce_dims) > 0:
                        valid_mask = valid_mask.any(dim=reduce_dims)
            else:
                valid_mask = None

        if valid_mask is not None and valid_mask.dim() == 2:
            has_valid = valid_mask.any(dim=-1)
            time_idx = torch.arange(valid_mask.size(1), device=device).unsqueeze(0)
            time_idx = time_idx.expand(valid_mask.size(0), -1)
            masked_idx = torch.where(
                valid_mask,
                time_idx,
                torch.full_like(time_idx, -1),
            )
            last_valid_idx = masked_idx.max(dim=-1).values.clamp_min(0)
            batch_idx = torch.arange(bsz, device=device)
            gathered_endpoint = y_gt[batch_idx, last_valid_idx, :2]
            gt_endpoint = torch.where(
                has_valid.unsqueeze(-1), gathered_endpoint, fallback_endpoint
            )
            endpoint_valid = has_valid
        else:
            gt_endpoint = fallback_endpoint
            endpoint_valid = torch.ones(bsz, dtype=torch.bool, device=device)

        endpoint_valid = endpoint_valid & torch.isfinite(gt_endpoint).all(dim=-1)
        return gt_endpoint, endpoint_valid

    def _compute_goal_cls_loss(
        self,
        goal_xy: torch.Tensor,
        goal_scores: torch.Tensor,
        gt_endpoint: torch.Tensor,
        valid_mask: torch.Tensor = None,
        tau: float = None,
    ):
        """
        Soft responsibility supervision:
          r_i = softmax(-d_i / tau), d_i = ||goal_xy_i - gt_endpoint||_2
          L_goal_cls = -sum_i r_i * log softmax(goal_scores_i)
        """
        if gt_endpoint is None or not torch.is_tensor(gt_endpoint):
            return None, None, None
        zero = gt_endpoint.new_zeros(())

        # fallback when auxiliary outputs are absent / malformed
        if (
            goal_xy is None
            or goal_scores is None
            or (not torch.is_tensor(goal_xy))
            or (not torch.is_tensor(goal_scores))
            or goal_xy.dim() != 3
            or goal_scores.dim() != 2
            or goal_xy.size(0) != gt_endpoint.size(0)
            or goal_scores.size(0) != gt_endpoint.size(0)
        ):
            return zero, None, None

        num_goals = min(goal_xy.size(1), goal_scores.size(1))
        if num_goals <= 0:
            return zero, None, None

        goal_xy = goal_xy[:, :num_goals, :2]
        goal_scores = goal_scores[:, :num_goals]
        gt_endpoint = gt_endpoint[:, :2]

        if valid_mask is None:
            valid_mask = torch.ones(
                gt_endpoint.size(0), dtype=torch.bool, device=gt_endpoint.device
            )
        else:
            valid_mask = valid_mask.bool()

        finite_goal_xy = torch.isfinite(goal_xy).all(dim=-1).all(dim=-1)
        finite_goal_scores = torch.isfinite(goal_scores).all(dim=-1)
        finite_gt = torch.isfinite(gt_endpoint).all(dim=-1)
        valid_mask = valid_mask & finite_goal_xy & finite_goal_scores & finite_gt
        if not valid_mask.any():
            return zero, None, valid_mask

        tau = self.goal_tau if tau is None else max(float(tau), 1e-3)
        dist = torch.norm(goal_xy - gt_endpoint.unsqueeze(1), dim=-1)  # [B, G]
        responsibility = torch.softmax(-dist / tau, dim=-1)  # [B, G]
        log_probs = F.log_softmax(goal_scores, dim=-1)  # [B, G]

        cls_per_sample = -(responsibility * log_probs).sum(dim=-1)  # [B]
        cls_loss = cls_per_sample[valid_mask].mean()
        cls_loss = torch.nan_to_num(cls_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return cls_loss, responsibility, valid_mask

    def _compute_goal_reg_loss(
        self,
        goal_xy: torch.Tensor,
        gt_endpoint: torch.Tensor,
        responsibility: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ):
        """
        Intermediate proposal supervision:
          L_goal_reg = sum_i r_i * smooth_l1(goal_xy_i, gt_endpoint)
        """
        if gt_endpoint is None or not torch.is_tensor(gt_endpoint):
            return None
        zero = gt_endpoint.new_zeros(())

        if (
            goal_xy is None
            or responsibility is None
            or (not torch.is_tensor(goal_xy))
            or (not torch.is_tensor(responsibility))
            or goal_xy.dim() != 3
            or responsibility.dim() != 2
            or goal_xy.size(0) != gt_endpoint.size(0)
            or responsibility.size(0) != gt_endpoint.size(0)
        ):
            return zero

        num_goals = min(goal_xy.size(1), responsibility.size(1))
        if num_goals <= 0:
            return zero
        goal_xy = goal_xy[:, :num_goals, :2]
        responsibility = responsibility[:, :num_goals]

        if valid_mask is None:
            valid_mask = torch.ones(
                gt_endpoint.size(0), dtype=torch.bool, device=gt_endpoint.device
            )
        else:
            valid_mask = valid_mask.bool()

        finite_goal_xy = torch.isfinite(goal_xy).all(dim=-1).all(dim=-1)
        finite_resp = torch.isfinite(responsibility).all(dim=-1)
        finite_gt = torch.isfinite(gt_endpoint).all(dim=-1)
        valid_mask = valid_mask & finite_goal_xy & finite_resp & finite_gt
        if not valid_mask.any():
            return zero

        # smooth_l1 per coordinate -> sum x/y -> weighted sum over goals
        endpoint_expanded = gt_endpoint.unsqueeze(1).expand(-1, num_goals, -1)
        reg_per_coord = F.smooth_l1_loss(goal_xy, endpoint_expanded, reduction='none')
        reg_per_goal = reg_per_coord.sum(dim=-1)  # [B, G]
        reg_per_sample = (responsibility * reg_per_goal).sum(dim=-1)  # [B]
        reg_loss = reg_per_sample[valid_mask].mean()
        reg_loss = torch.nan_to_num(reg_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return reg_loss

    def cal_loss(self, out, data, tag=''):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        scal, scal_new = out["scal"], out["scal_new"]
        new_y_hat = out.get("new_y_hat", None)
        new_pi = out.get("new_pi", None)
        dense_predict = out.get("dense_predict", None)
        ep_offsets = out.get("ep_offsets", None)
        goal_scores = out.get("goal_scores", None)
        goal_xy = out.get("goal_xy", None)
        
        center = data["x_centers"][:,0]

        # gt
        y, y_others = data["target"][:, 0], data["target"][:, 1:]

        # loss for output of state query
        if dense_predict is not None:
            if isinstance(dense_predict, list):
                dense_reg_loss = 0
                for pred in dense_predict:
                    dense_reg_loss = dense_reg_loss + F.smooth_l1_loss(pred, y)
            else:
                dense_reg_loss = F.smooth_l1_loss(dense_predict, y)
        else:
            dense_reg_loss = 0
        
        
        if ep_offsets is not None:
            gt_offsets = y[:,-1] - center
            if isinstance(ep_offsets, list):
                ep_reg_loss = 0
                for w, pred in zip(self.ws_offset, ep_offsets):
                    # ep_reg_loss = dense_reg_loss + w*F.smooth_l1_loss(pred, gt_offsets)
                    ep_reg_loss = ep_reg_loss + w*F.smooth_l1_loss(pred, gt_offsets)
            else:
                ep_reg_loss = F.smooth_l1_loss(ep_offsets, gt_offsets)
        else:
            ep_reg_loss = 0
        # gt_offsets = y[:,-1] - center
        # ep_reg_loss = dense_reg_loss + F.smooth_l1_loss(ep_offsets[-1], gt_offsets)
        

        if y_hat.dim() == 3:
            # loss for output of mode query (endpoint-only, shape [B, K, 2])
            ep = y[:,-1,:2]
            l2_norm = torch.norm(y_hat[..., :2] - ep.unsqueeze(1), dim=-1)
            best_mode = torch.argmin(l2_norm, dim=-1)
            y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
            agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], ep)
            agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)
        else:
            # loss for output of mode query (full trajectory, ADE-based WTA)
            l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode = torch.argmin(l2_norm, dim=-1)
            y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
            agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
            agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)

        # loss for final output (ADE-based WTA)
        if new_y_hat is not None:
            l2_norm_new = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_new = torch.argmin(l2_norm_new, dim=-1)
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode_new]
            new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
        else:
            new_agent_reg_loss = 0
        if new_pi is not None:
            new_pi_reg_loss = F.cross_entropy(new_pi, best_mode_new.detach(), label_smoothing=0.2)
        else:
            new_pi_reg_loss = 0

        # loss for other agents
        others_reg_mask = data["target_mask"][:, 1:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        predictions = {}
        predictions['traj'] = y_hat
        predictions['scale'] = scal
        predictions['probs'] = pi
        laplace_loss = self.laplace_loss.compute(predictions, y)

        predictions['traj'] = new_y_hat
        predictions['scale'] = scal_new
        predictions['probs'] = new_pi
        laplace_loss_new = self.laplace_loss.compute(predictions, y)
        
        # -------- goal auxiliary supervision (minimal intrusive) --------
        # Extract ego GT endpoint with mask-aware fallback:
        # - preferred: last valid future point
        # - fallback: last future timestep
        target_mask = data.get("target_mask", None)
        ego_target_mask = None
        if target_mask is not None and torch.is_tensor(target_mask):
            if target_mask.dim() >= 3:
                ego_target_mask = target_mask[:, 0]
            elif target_mask.dim() == 2:
                # fallback for direct [B, T] mask format
                ego_target_mask = target_mask
        gt_endpoint, endpoint_valid = self.get_gt_endpoint(y, ego_target_mask)

        if gt_endpoint is None:
            # fallback guard: no valid GT endpoint tensor
            goal_cls_loss = y.new_zeros(())
            goal_reg_loss = y.new_zeros(())
        else:
            goal_cls_loss, responsibility, goal_valid_mask = self._compute_goal_cls_loss(
                goal_xy=goal_xy,
                goal_scores=goal_scores,
                gt_endpoint=gt_endpoint,
                valid_mask=endpoint_valid,
                tau=self.goal_tau,
            )
            if goal_cls_loss is None:
                goal_cls_loss = y.new_zeros(())
            goal_reg_loss = self._compute_goal_reg_loss(
                goal_xy=goal_xy,
                gt_endpoint=gt_endpoint,
                responsibility=responsibility,
                valid_mask=goal_valid_mask if goal_valid_mask is not None else endpoint_valid,
            )
            if goal_reg_loss is None:
                goal_reg_loss = y.new_zeros(())

                
        loss = new_agent_reg_loss + new_pi_reg_loss + laplace_loss + laplace_loss_new
        loss = loss + agent_reg_loss + agent_cls_loss + others_reg_loss + dense_reg_loss
        loss = loss + ep_reg_loss
        loss = loss + self.lambda_goal_cls * goal_cls_loss + self.lambda_goal_reg * goal_reg_loss
                


        disp_dict = {
            f"{tag}loss": loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}laplace_loss": laplace_loss.item(),
            f"{tag}laplace_loss_new": laplace_loss_new.item(),
            f"{tag}loss_goal_cls": goal_cls_loss.item(),
            f"{tag}loss_goal_reg": goal_reg_loss.item(),
            f"{tag}loss_goal_cls_w": (self.lambda_goal_cls * goal_cls_loss).item(),
            f"{tag}loss_goal_reg_w": (self.lambda_goal_reg * goal_reg_loss).item(),
        }
        
        if new_y_hat is not None:
            disp_dict[f"{tag}reg_loss_refine"] = new_agent_reg_loss.item()
        if new_pi is not None:
            disp_dict[f"{tag}reg_loss_new_pi"] = new_pi_reg_loss.item()
        if dense_predict is not None:
            disp_dict[f"{tag}reg_loss_dense"] = dense_reg_loss.item()
        if ep_offsets is not None:
            disp_dict[f"{tag}ep_reg_loss"] = ep_reg_loss.item()

        return loss, disp_dict

    def training_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
            
        
        out = self(data)
        loss, loss_dict = self.cal_loss(out, data)

        bs = data["target"].shape[0]
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=bs,
            )

        self._log_fallback_stats(out, tag="train", batch_size=bs)

        return loss

    def _log_fallback_stats(self, out, tag: str, batch_size: int):
        fallback_stats = out.get("fallback_stats", None)
        if fallback_stats is None:
            return

        for k, v in fallback_stats.items():
            if v is None:
                continue
            if not torch.is_tensor(v):
                v = torch.tensor(float(v), device=self.device)
            self.log(
                f"{tag}/fallback/{k}",
                v.detach(),
                on_step=(tag == "train"),
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )

    def vis_ep(self, ep, gt, prediction, save_path="/home/shijie/code/FINet_ICCV2025/FINet/Visual_ep"):
        prediction = prediction.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        ep1 = ep[0].cpu().detach().numpy()[0]
        ep2 = ep[1].cpu().detach().numpy()[0]



        fig, ax = plt.subplots(figsize=(6, 6))
        
        for pred in prediction:
            ax.plot(pred[:, 0], pred[:, 1], color='blue', linewidth=2)
        
        ax.plot(gt[:, 0], gt[:, 1], color='red', linewidth=2)

        plt.scatter(ep1[0], ep1[1], color='green', label='Point (5, 10)', zorder=5)  # Plot the point
        plt.text(ep1[0], ep1[1], f'(Endpoint 1)', fontsize=12, ha='left', va='bottom')  # Optionally add a label

        plt.scatter(ep2[0], ep2[1], color='green', label='Point (5, 10)', zorder=5)  # Plot the point
        plt.text(ep2[0], ep2[1], f'(Endpoint 2)', fontsize=12, ha='left', va='bottom')  # Optionally add a label

        # ax.set_xlim(0, 400)
        # ax.set_ylim(0, 400)
        plt.tight_layout()

        
        self.cur_time = self.cur_time + 1
        save_path = os.path.join(save_path, str(self.cur_time) + ".png")
        print(save_path)
        plt.savefig(save_path, dpi=300, pad_inches=0)

        
    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        try:
            out = self(data)
            _, loss_dict = self.cal_loss(out, data)
            metrics = self.val_metrics(out, data['target'][:, 0])
            if out['new_y_hat'] is not None:
                out['y_hat'] = out['new_y_hat']
                out['pi'] = out['new_pi']
            if out['new_y_hat'] is not None:
                metrics_new = self.val_metrics_new(out, data['target'][:, 0])

            # print(out['ep_offsets'], data["target"][:,0,0], data["target"][:,0,-1])

            # self.vis_ep(out['ep_offsets'], data["target"][0,0], out['new_y_hat'][0])

            # ep = out['new_y_hat'][0,:,-1]
            # ep_gt = data["target"][0,0,-1]


            # self.count[out['new_pi'].argmax().item()]  += 1
            # self.count_closet[((ep - ep_gt)**2).sum(-1).argmin().item()] += 1
        
            # print(self.count)
    
            self.log_dict(
                metrics,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
            if out['new_y_hat'] is not None:
                self.log_dict(
                    metrics_new,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=1,
                    sync_dist=True,
                )
            self._log_fallback_stats(out, tag="val", batch_size=1)

        except Exception as exc:
            self.val_exception_count += 1
            print(
                f"[validation_step] batch_idx={batch_idx}, "
                f"exception_count={self.val_exception_count}, "
                f"type={type(exc).__name__}, message={exc}"
            )
            self.log(
                "val/exception_count",
                float(self.val_exception_count),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

        # print(self.count, self.count_closet)

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2(
            save_dir=save_dir
        )
        self.total_time = 0.0
        self.test_step_count = 0

    def test_step(self, data, batch_idx) -> None:
        if isinstance(data, list):
            data = data[-1]
        
        torch.cuda.synchronize()
        start_time = time.time()

        out = self(data)

        torch.cuda.synchronize()
        end_time = time.time()
        
        latency = end_time - start_time
        self.total_time += latency
        self.test_step_count += 1

        self.cur_time += 1

        # print(self.total_time/self.cur_time)
        
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        avg_latency = self.total_time / max(self.test_step_count, 1)
        print(
            f"[test] steps={self.test_step_count}, "
            f"avg_latency={avg_latency:.6f}s"
        )
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
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
