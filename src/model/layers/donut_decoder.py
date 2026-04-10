import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from ..models_mamba import RMSNorm, layer_norm_fn, rms_norm_fn
from .mamba.vim_mamba import create_block
from .transformer_blocks import Cross_Block
from .coordinate_transforms import wrap_angle, global_to_local, local_to_global
from .fourier_embedding import FourierEmbedding


class SimpleTokenizer(nn.Module):
    """
    Converts `t_per_tok` position+heading steps into one token feature vector.
    8-dim input with Fourier Embedding:
      [delta_x, delta_y, velocity, rel_pos_x, rel_pos_y,
       heading, heading_delta, head_vs_motion]
    """

    def __init__(self, embed_dim=128, t_per_tok=10):
        super().__init__()
        self.step_embed = FourierEmbedding(input_dim=8, hidden_dim=embed_dim)
        self.aggregate = nn.Linear((t_per_tok - 1) * embed_dim, embed_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, positions, headings):
        # positions: [B, M, t_per_tok, 2],  headings: [B, M, t_per_tok]
        deltas = positions[..., 1:, :] - positions[..., :-1, :]      # [*,T-1,2]
        velocity = torch.linalg.norm(deltas, dim=-1, keepdim=True)   # [*,T-1,1]
        rel_pos = positions[..., 1:, :] - positions[..., :1, :]      # [*,T-1,2]

        # heading features
        head_val = headings[..., 1:, None]                           # [*,T-1,1]
        head_delta = wrap_angle(
            headings[..., 1:] - headings[..., :-1]
        ).unsqueeze(-1)                                              # [*,T-1,1]

        # angle between heading direction and motion direction
        head_cos = headings[..., 1:].cos()
        head_sin = headings[..., 1:].sin()
        cross = head_cos * deltas[..., 1] - head_sin * deltas[..., 0]
        dot = head_cos * deltas[..., 0] + head_sin * deltas[..., 1]
        head_vs_mot = torch.atan2(cross, dot).unsqueeze(-1)          # [*,T-1,1]

        features = torch.cat(
            [deltas, velocity, rel_pos, head_val, head_delta, head_vs_mot],
            dim=-1,
        )  # [*, T-1, 8]

        x = self.step_embed(features)
        x = x.flatten(-2, -1)
        x = self.aggregate(x)
        x = self.out_proj(x)
        return x


class SimpleDetokenizer(nn.Module):
    """
    Converts a token feature into position deltas, Laplace scale,
    heading deltas and von Mises concentration.
    Supports over-prediction: outputs (1+over_predict)*t_per_tok timesteps.
    """

    def __init__(self, embed_dim=128, t_per_tok=10, over_predict=1):
        super().__init__()
        self.t_per_tok = t_per_tok
        self.over_predict = over_predict
        self.feat_len = (1 + over_predict) * t_per_tok

        self.shared = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.feat_len * 2),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.feat_len * 2),
        )
        self.heading_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.feat_len),
        )
        self.conc_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.feat_len),
        )

    def forward(self, x):
        x = self.shared(x)

        pos_delta = self.pos_head(x).reshape(*x.shape[:2], self.feat_len, 2)

        scale = (1.0 + F.elu(self.scale_head(x))).clamp(min=1e-6)
        scale = scale.reshape(*x.shape[:2], self.feat_len, 2)

        head_delta = self.heading_head(x).reshape(*x.shape[:2], self.feat_len)

        conc = (1.0 + F.elu(self.conc_head(x))).clamp(min=1e-6)
        conc = conc.reshape(*x.shape[:2], self.feat_len)

        return pos_delta, scale, head_delta, conc


class ModeAttention(nn.Module):
    """
    Self-attention among modes at each autoregressive step.
    """

    def __init__(
        self,
        embed_dim=128,
        num_modes=6,
        num_pred_steps=6,
        num_heads=8,
        drop_path=0.2,
    ):
        super().__init__()
        self.mode_emb = nn.Embedding(num_modes, embed_dim)
        self.time_emb = nn.Embedding(num_pred_steps + 1, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, pred_step):
        mode_ids = torch.arange(x.shape[1], device=x.device)
        time_ids = torch.full((1,), pred_step, device=x.device, dtype=torch.long)

        x = x + self.mode_emb(mode_ids).unsqueeze(0)
        x = x + self.time_emb(time_ids).view(1, 1, -1)

        x_norm = self.norm(x)
        attn_out = self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path2(self.mlp(x))
        return x


class AutoregressiveStage(nn.Module):
    """
    One autoregressive decoding stage: Mamba(T) + CrossAttn(R) + ModeAttn(M).
    With per-token local coordinate transforms, heading prediction,
    and over-prediction.
    """

    def __init__(
        self,
        embed_dim=128,
        t_per_tok=10,
        num_modes=6,
        num_pred_tokens=6,
        num_mamba_layers=4,
        num_heads=8,
        drop_path=0.2,
        is_refiner=False,
        over_predict=1,
    ):
        super().__init__()
        self.t_per_tok = t_per_tok
        self.num_pred_tokens = num_pred_tokens
        self.num_modes = num_modes
        self.is_refiner = is_refiner
        self.over_predict = over_predict

        self.tokenizer = SimpleTokenizer(embed_dim=embed_dim, t_per_tok=t_per_tok)
        self.mamba_blocks = nn.ModuleList(
            [
                create_block(
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=drop_path,
                    bimamba=False,
                    rms_norm=True,
                )
                for i in range(num_mamba_layers)
            ]
        )
        self.mamba_norm = RMSNorm(embed_dim, eps=1e-5)
        self.mamba_drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.cross_attn = Cross_Block(
            dim=embed_dim,
            num_heads=num_heads,
            drop_path=drop_path,
        )
        self.mode_attn = ModeAttention(
            embed_dim=embed_dim,
            num_modes=num_modes,
            num_pred_steps=num_pred_tokens,
            num_heads=num_heads,
            drop_path=drop_path,
        )
        self.detokenizer = SimpleDetokenizer(
            embed_dim=embed_dim, t_per_tok=t_per_tok, over_predict=over_predict,
        )
        self.pi_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        if is_refiner:
            self.feature_fuse = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

    def forward(
        self,
        mode_tokens,
        ego_feat,
        scene_encoding,
        scene_mask,
        proposed_positions=None,
        proposed_headings=None,
        proposer_feats=None,
        init_heading=None,
    ):
        B, M, _ = mode_tokens.shape
        device = mode_tokens.device
        T = self.t_per_tok

        hist_token = ego_feat.unsqueeze(1).unsqueeze(2).expand(B, M, 1, -1)
        token_seq_list = [hist_token]

        all_positions = []
        all_scales = []
        all_headings = []
        all_concs = []
        # Over-prediction collections
        all_positions_over = []
        all_scales_over = []
        all_headings_over = []
        all_concs_over = []
        step_feats = []

        # Initial anchors: ego origin and heading
        anchor_pos = torch.zeros(B, M, 2, device=device)
        if init_heading is not None:
            anchor_head = init_heading[:, None].expand(B, M)  # [B, M]
        else:
            anchor_head = torch.zeros(B, M, device=device)

        prev_positions = None
        prev_headings = None

        for step in range(self.num_pred_tokens):
            # ---- Tokenize ----
            if step == 0:
                tok = mode_tokens
            else:
                local_pos, local_head = global_to_local(
                    prev_positions, prev_headings,
                    anchor_pos.unsqueeze(2), anchor_head.unsqueeze(2),
                )
                tok = self.tokenizer(local_pos, local_head)

            if self.is_refiner and proposer_feats is not None:
                tok = tok + self.feature_fuse(proposer_feats[step])

            # ---- Temporal: Mamba ----
            token_seq_list.append(tok.unsqueeze(2))
            seq = torch.cat(token_seq_list, dim=2)
            seq_flat = seq.reshape(B * M, -1, seq.shape[-1])
            residual = None
            for blk in self.mamba_blocks:
                seq_flat, residual = blk(seq_flat, residual)

            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.mamba_norm, RMSNorm) else layer_norm_fn
            )
            seq_flat = fused_add_norm_fn(
                self.mamba_drop_path(seq_flat),
                self.mamba_norm.weight,
                self.mamba_norm.bias,
                eps=self.mamba_norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=True,
            )
            tok = seq_flat[:, -1].reshape(B, M, -1)

            # ---- Road: Cross-Attention ----
            tok = self.cross_attn(tok, scene_encoding, key_padding_mask=scene_mask)

            # ---- Mode: Self-Attention ----
            tok = self.mode_attn(tok, step + 1)
            step_feats.append(tok)

            # ---- Detokenize ----
            # Output shape: [B, M, feat_len, ...] where feat_len = (1+over_predict)*T
            pos_delta_full, scale_full, head_delta_full, conc_full = self.detokenizer(tok)

            # Split normal and over-prediction
            pos_delta = pos_delta_full[:, :, :T, :]
            head_delta = head_delta_full[:, :, :T]
            scale_norm = scale_full[:, :, :T, :]
            conc_norm = conc_full[:, :, :T]

            if self.over_predict:
                pos_delta_over = pos_delta_full[:, :, T:, :]
                head_delta_over = head_delta_full[:, :, T:]
                scale_over = scale_full[:, :, T:, :]
                conc_over = conc_full[:, :, T:]

            # ---- Normal: position / heading with local frames ----
            if self.is_refiner and proposed_positions is not None:
                start = step * T
                end = start + T
                prop_local_pos, prop_local_head = global_to_local(
                    proposed_positions[:, :, start:end, :],
                    proposed_headings[:, :, start:end],
                    anchor_pos.unsqueeze(2), anchor_head.unsqueeze(2),
                )
                local_pos = prop_local_pos + pos_delta
                local_head = wrap_angle(prop_local_head + head_delta)
            else:
                local_pos = torch.cumsum(pos_delta, dim=2)
                local_head = torch.cumsum(
                    0.3 * torch.tanh(head_delta), dim=2
                )

            positions, headings = local_to_global(
                local_pos, local_head,
                anchor_pos.unsqueeze(2), anchor_head.unsqueeze(2),
            )

            all_positions.append(positions)
            all_scales.append(scale_norm)
            all_headings.append(headings)
            all_concs.append(conc_norm)

            # ---- Over-prediction: extend from normal's last position ----
            if self.over_predict:
                # Over-prediction uses the normal output's last point as anchor
                over_anchor_pos = positions[:, :, -1, :]   # [B, M, 2]
                over_anchor_head = headings[:, :, -1]      # [B, M]

                if self.is_refiner and proposed_positions is not None:
                    # Refiner over: residual on proposed (shifted by T)
                    over_start = start + T
                    over_end = min(over_start + T, proposed_positions.shape[2])
                    actual_len = over_end - over_start
                    if actual_len > 0:
                        prop_over_local_pos, prop_over_local_head = global_to_local(
                            proposed_positions[:, :, over_start:over_end, :],
                            proposed_headings[:, :, over_start:over_end],
                            over_anchor_pos.unsqueeze(2), over_anchor_head.unsqueeze(2),
                        )
                        over_local_pos = prop_over_local_pos + pos_delta_over[:, :, :actual_len, :]
                        over_local_head = wrap_angle(
                            prop_over_local_head + head_delta_over[:, :, :actual_len]
                        )
                    else:
                        over_local_pos = torch.cumsum(pos_delta_over, dim=2)
                        over_local_head = torch.cumsum(
                            0.3 * torch.tanh(head_delta_over), dim=2
                        )
                        actual_len = T
                else:
                    over_local_pos = torch.cumsum(pos_delta_over, dim=2)
                    over_local_head = torch.cumsum(
                        0.3 * torch.tanh(head_delta_over), dim=2
                    )
                    actual_len = T

                positions_over, headings_over = local_to_global(
                    over_local_pos, over_local_head,
                    over_anchor_pos.unsqueeze(2), over_anchor_head.unsqueeze(2),
                )

                all_positions_over.append(positions_over)
                all_scales_over.append(scale_over[:, :, :actual_len, :])
                all_headings_over.append(headings_over)
                all_concs_over.append(conc_over[:, :, :actual_len])

            # Update anchors for next step (from normal output only)
            anchor_pos = positions[:, :, -1, :].detach()
            anchor_head = headings[:, :, -1].detach()
            prev_positions = positions.detach()
            prev_headings = headings.detach()

        # ---- Assemble normal output ----
        y_hat = torch.cat(all_positions, dim=2)       # [B, M, 60, 2]
        scal = torch.cat(all_scales, dim=2)
        scal = 0.1 + torch.cumsum(scal, dim=2)        # [B, M, 60, 2]

        heading_hat = torch.cat(all_headings, dim=2)   # [B, M, 60]
        conc_hat = torch.cat(all_concs, dim=2)
        conc_hat = 1.0 / (0.02 + torch.cumsum(conc_hat, dim=2))  # [B, M, 60]

        feat_stack = torch.stack(step_feats, dim=2)
        feat_pool = feat_stack.max(dim=2)[0]
        pi = self.pi_head(feat_pool).squeeze(-1)       # [B, M]

        # ---- Assemble over-prediction output ----
        if self.over_predict and all_positions_over:
            y_hat_over = torch.cat(all_positions_over, dim=2)       # [B, M, 60, 2]
            scal_over = torch.cat(all_scales_over, dim=2)
            scal_over = 0.1 + scal_over                             # no cumsum

            heading_over = torch.cat(all_headings_over, dim=2)      # [B, M, 60]
            conc_over = torch.cat(all_concs_over, dim=2)
            conc_over = 1.0 / (0.02 + conc_over)                    # no cumsum
        else:
            y_hat_over = heading_over = scal_over = conc_over = None

        return (y_hat, pi, scal, step_feats, heading_hat, conc_hat,
                y_hat_over, scal_over, heading_over, conc_over)


class DonutMambaDecoder(nn.Module):
    """
    DONUT-style autoregressive decoder with a Mamba temporal core.
    Two stages: proposer -> refiner.
    With heading prediction, per-token local coordinate transforms,
    Fourier Embedding tokenizer, and over-prediction.
    """

    def __init__(
        self,
        embed_dim=128,
        t_per_tok=10,
        num_modes=6,
        future_steps=60,
        dec_layer_1=4,
        dec_layer_2=4,
        num_heads=8,
        drop_path=0.2,
        over_predict=1,
    ):
        super().__init__()
        if future_steps % t_per_tok != 0:
            raise ValueError(
                f"future_steps ({future_steps}) must be divisible by t_per_tok ({t_per_tok})"
            )

        self.t_per_tok = t_per_tok
        num_pred_tokens = future_steps // t_per_tok

        self.proposer = AutoregressiveStage(
            embed_dim=embed_dim,
            t_per_tok=t_per_tok,
            num_modes=num_modes,
            num_pred_tokens=num_pred_tokens,
            num_mamba_layers=dec_layer_1,
            num_heads=num_heads,
            drop_path=drop_path,
            is_refiner=False,
            over_predict=over_predict,
        )
        self.refiner = AutoregressiveStage(
            embed_dim=embed_dim,
            t_per_tok=t_per_tok,
            num_modes=num_modes,
            num_pred_tokens=num_pred_tokens,
            num_mamba_layers=dec_layer_2,
            num_heads=num_heads,
            drop_path=drop_path,
            is_refiner=True,
            over_predict=over_predict,
        )

    def forward(self, mode_tokens, ego_feat, scene_encoding, mask=None,
                init_heading=None):
        # Stage 1: Proposer
        (y_hat, pi, scal, proposer_feats, heading_hat, conc_hat,
         y_hat_over, scal_over, heading_over, conc_over) = self.proposer(
            mode_tokens=mode_tokens,
            ego_feat=ego_feat,
            scene_encoding=scene_encoding,
            scene_mask=mask,
            init_heading=init_heading,
        )

        # Stage 2: Refiner
        (new_y_hat, new_pi, scal_new, _, new_heading_hat, new_conc_hat,
         new_y_hat_over, new_scal_over, new_heading_over, new_conc_over) = self.refiner(
            mode_tokens=mode_tokens,
            ego_feat=ego_feat,
            scene_encoding=scene_encoding,
            scene_mask=mask,
            proposed_positions=y_hat.detach(),
            proposed_headings=heading_hat.detach(),
            proposer_feats=proposer_feats,
            init_heading=init_heading,
        )

        dense_pred = None
        mode_features = None
        return {
            "dense_pred": dense_pred,
            "y_hat": y_hat, "pi": pi, "scal": scal,
            "mode_features": mode_features,
            "new_y_hat": new_y_hat, "new_pi": new_pi, "scal_new": scal_new,
            "heading_hat": heading_hat, "conc_hat": conc_hat,
            "new_heading_hat": new_heading_hat, "new_conc_hat": new_conc_hat,
            # Over-prediction
            "y_hat_over": y_hat_over, "scal_over": scal_over,
            "heading_over": heading_over, "conc_over": conc_over,
            "new_y_hat_over": new_y_hat_over, "new_scal_over": new_scal_over,
            "new_heading_over": new_heading_over, "new_conc_over": new_conc_over,
        }
