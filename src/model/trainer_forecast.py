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
        num_modes: int = 6,
        use_soft_multimodal: bool = True,
        top_m: int = 2,
        soft_tau: float = 0.7,
        detach_soft_weight: bool = True,
        use_soft_score: bool = True,
        use_soft_intermediate: bool = True,
        use_endpoint_diversity: bool = True,
        div_margin: float = 2.0,
        lambda_div: float = 0.02,
        lambda_score_soft: float = 1.0,
        use_goal_loss: bool = False,
        lambda_goal: float = 0.2,
        goal_recall_threshold: float = 2.0,
        laplace_scale_min: float = 1e-3,
        laplace_scale_max: float = 10.0,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_modes = num_modes
        self.use_soft_multimodal = use_soft_multimodal
        self.top_m = top_m
        self.soft_tau = soft_tau
        self.detach_soft_weight = detach_soft_weight
        self.use_soft_score = use_soft_score
        self.use_soft_intermediate = use_soft_intermediate
        self.use_endpoint_diversity = use_endpoint_diversity
        self.div_margin = div_margin
        self.lambda_div = lambda_div
        self.lambda_score_soft = lambda_score_soft
        self.use_goal_loss = use_goal_loss
        self.lambda_goal = lambda_goal
        self.goal_recall_threshold = goal_recall_threshold
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        self.model_type = model.pop('type')
        assert self.model_type in model_dict
        self.net = model_dict[self.model_type](**model)

        # self.net = self.get_model(model_type)(**model)

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print('Pretrained weights have been loaded.')

        top_k_eval = min(6, self.num_modes)
        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=top_k_eval),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=top_k_eval),
                "MR": MR(),
                "b-minFDE6": brier_minFDE(k=top_k_eval),
            }
        )
        self.laplace_loss = LaplaceNLLLoss(
            num_modes=self.num_modes,
            top_m=self.top_m,
            tau=self.soft_tau,
            detach_soft_weight=self.detach_soft_weight,
            scale_min=laplace_scale_min,
            scale_max=laplace_scale_max,
        )
        self.val_metrics = metrics.clone(prefix="val_")
        self.val_metrics_new = metrics.clone(prefix="val_new_")
        
        self.ws_offset = ws_offset

        self.total_time=0
        self.cur_time = 0

        self.count = np.zeros(self.num_modes)
        self.count_closet = np.zeros(self.num_modes)
        
    

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

    def _compute_mode_distance(self, pred: torch.Tensor, y: torch.Tensor):
        if pred is None:
            return None
        if pred.dim() == 3:
            pred_end = pred[..., :2]
            gt_end = y[:, -1, :2]
        else:
            pred_end = pred[..., -1, :2]
            gt_end = y[:, -1, :2]
        return torch.norm(pred_end - gt_end.unsqueeze(1), dim=-1)

    def _build_soft_target(self, mode_dist: torch.Tensor):
        top_m = min(self.top_m, mode_dist.size(1))
        top_dist, top_idx = torch.topk(mode_dist, k=top_m, dim=-1, largest=False)
        weights = F.softmax(-top_dist / max(self.soft_tau, 1e-6), dim=-1)
        if self.detach_soft_weight:
            weights = weights.detach()
        soft_target = torch.zeros_like(mode_dist)
        soft_target.scatter_(1, top_idx, weights)
        return soft_target

    def _soft_traj_loss(self, pred: torch.Tensor, y: torch.Tensor, soft_target: torch.Tensor):
        if pred.dim() == 3:
            per_mode = F.smooth_l1_loss(
                pred[..., :2], y[:, -1, :2].unsqueeze(1), reduction="none"
            ).mean(dim=-1)
        else:
            per_mode = F.smooth_l1_loss(
                pred[..., :2], y.unsqueeze(1), reduction="none"
            ).mean(dim=(-1, -2))
        return (soft_target * per_mode).sum(dim=-1).mean()

    def _soft_cls_loss(self, logits: torch.Tensor, soft_target: torch.Tensor):
        return -(soft_target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def _hard_losses(self, pred: torch.Tensor, logits: torch.Tensor, y: torch.Tensor):
        mode_dist = self._compute_mode_distance(pred, y)
        best_mode = torch.argmin(mode_dist, dim=-1)
        batch_idx = torch.arange(pred.shape[0], device=pred.device)
        if pred.dim() == 3:
            pred_best = pred[batch_idx, best_mode]
            reg_loss = F.smooth_l1_loss(pred_best[..., :2], y[:, -1, :2])
        else:
            pred_best = pred[batch_idx, best_mode]
            reg_loss = F.smooth_l1_loss(pred_best[..., :2], y)
        if logits is None:
            cls_loss = reg_loss.new_tensor(0.0)
        else:
            cls_loss = F.cross_entropy(logits, best_mode.detach(), label_smoothing=0.2)
        return reg_loss, cls_loss, best_mode, mode_dist

    def _endpoint_diversity_loss(self, pred: torch.Tensor):
        if pred is None:
            return None
        if pred.dim() == 3:
            endpoints = pred[..., :2]
        else:
            endpoints = pred[..., -1, :2]
        if endpoints.size(1) < 2:
            return endpoints.new_tensor(0.0)
        pair_dist = torch.cdist(endpoints, endpoints, p=2)
        valid_pair = ~torch.eye(
            endpoints.size(1), device=endpoints.device, dtype=torch.bool
        ).unsqueeze(0)
        pair_penalty = F.relu(self.div_margin - pair_dist) * valid_pair.float()
        return pair_penalty.sum() / valid_pair.sum().clamp(min=1)

    def cal_loss(self, out, data, tag=''):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        scal, scal_new = out["scal"], out["scal_new"]
        new_y_hat = out.get("new_y_hat", None)
        new_pi = out.get("new_pi", None)
        dense_predict = out.get("dense_predict", None)
        ep_offsets = out.get("ep_offsets", None)
        
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
        

        soft_target = None
        soft_target_new = None
        if self.use_soft_multimodal and self.use_soft_intermediate:
            mode_dist = self._compute_mode_distance(y_hat, y)
            soft_target = self._build_soft_target(mode_dist)
            agent_reg_loss = self._soft_traj_loss(y_hat, y, soft_target)
            if self.use_soft_score:
                agent_cls_loss = self.lambda_score_soft * self._soft_cls_loss(pi, soft_target)
            else:
                best_mode = torch.argmin(mode_dist, dim=-1)
                agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)
        else:
            agent_reg_loss, agent_cls_loss, _, _ = self._hard_losses(y_hat, pi, y)

        best_mode_new = None
        if new_y_hat is not None:
            if self.use_soft_multimodal:
                mode_dist_new = self._compute_mode_distance(new_y_hat, y)
                soft_target_new = self._build_soft_target(mode_dist_new)
                new_agent_reg_loss = self._soft_traj_loss(new_y_hat, y, soft_target_new)
                best_mode_new = torch.argmin(mode_dist_new, dim=-1)
            else:
                new_agent_reg_loss, _, best_mode_new, _ = self._hard_losses(new_y_hat, new_pi, y)
        else:
            new_agent_reg_loss = y.new_tensor(0.0)
        if new_pi is not None:
            if self.use_soft_multimodal and self.use_soft_score and soft_target_new is not None:
                new_pi_reg_loss = self.lambda_score_soft * self._soft_cls_loss(new_pi, soft_target_new)
            else:
                if best_mode_new is None:
                    mode_dist_new = self._compute_mode_distance(new_y_hat, y)
                    best_mode_new = torch.argmin(mode_dist_new, dim=-1)
                new_pi_reg_loss = F.cross_entropy(new_pi, best_mode_new.detach(), label_smoothing=0.2)
        else:
            new_pi_reg_loss = y.new_tensor(0.0)

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

        div_source = new_y_hat if new_y_hat is not None else y_hat
        if self.use_endpoint_diversity:
            div_loss = self._endpoint_diversity_loss(div_source)
        else:
            div_loss = y.new_tensor(0.0)

        goal_loss = y.new_tensor(0.0)
        goal_recall = y.new_tensor(0.0)
        goal_logits = out.get("goal_logits", None)
        goal_candidate_xy = out.get("goal_candidate_xy", None)
        goal_candidate_mask = out.get("goal_candidate_mask", None)
        if (
            self.use_goal_loss
            and goal_logits is not None
            and goal_candidate_xy is not None
            and goal_candidate_mask is not None
        ):
            gt_endpoint = y[:, -1, :2]
            goal_dist = torch.norm(
                goal_candidate_xy - gt_endpoint.unsqueeze(1), dim=-1
            )
            goal_dist = goal_dist.masked_fill(~goal_candidate_mask, 1e9)
            gt_goal_idx = torch.argmin(goal_dist, dim=-1)
            goal_logits = goal_logits.masked_fill(~goal_candidate_mask, -1e9)
            goal_loss = F.cross_entropy(goal_logits, gt_goal_idx.detach())
            goal_recall = (
                goal_dist.min(dim=-1).values < self.goal_recall_threshold
            ).float().mean()

        loss = new_agent_reg_loss + new_pi_reg_loss + laplace_loss + laplace_loss_new
        loss = loss + agent_reg_loss + agent_cls_loss + others_reg_loss + dense_reg_loss
        loss = loss + ep_reg_loss + self.lambda_div * div_loss + self.lambda_goal * goal_loss

        soft_target_log = soft_target_new if soft_target_new is not None else soft_target
        if soft_target_log is not None and soft_target_log.size(1) > 1:
            top_weights = torch.topk(soft_target_log, k=2, dim=-1).values
            soft_w_top1 = top_weights[:, 0].mean()
            soft_w_top2 = top_weights[:, 1].mean()
        elif soft_target_log is not None:
            soft_w_top1 = soft_target_log.max(dim=-1).values.mean()
            soft_w_top2 = y.new_tensor(0.0)
        else:
            soft_w_top1 = y.new_tensor(0.0)
            soft_w_top2 = y.new_tensor(0.0)


        disp_dict = {
            f"{tag}loss": loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}laplace_loss": laplace_loss.item(),
            f"{tag}laplace_loss_new": laplace_loss_new.item(),
            f"{tag}div_loss": div_loss.item(),
            f"{tag}goal_loss": goal_loss.item(),
            f"{tag}goal_recall": goal_recall.item(),
            f"{tag}soft_w_top1": soft_w_top1.item(),
            f"{tag}soft_w_top2": soft_w_top2.item(),
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

        return loss

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

        except:
            pass

        # print(self.count, self.count_closet)

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2(
            save_dir=save_dir
        )
    
    def on_test_end(self) -> None:

       latency = self.total_time / len(self.test_dataloader().dataset)

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

        self.cur_time += 1

        # print(self.total_time/self.cur_time)
        
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
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
