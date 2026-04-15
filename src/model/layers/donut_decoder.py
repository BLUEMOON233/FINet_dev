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


def _safe_signed_angle(cross: torch.Tensor, dot: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Stable signed angle that avoids atan2(0, 0) and its undefined gradient."""
    safe_cross = torch.where(valid, cross, torch.zeros_like(cross))
    safe_dot = torch.where(valid, dot, torch.ones_like(dot))
    angle = torch.atan2(safe_cross, safe_dot)
    return torch.where(valid, angle, torch.zeros_like(angle))


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
        moving = velocity.squeeze(-1) > 1e-6
        head_vs_mot = _safe_signed_angle(cross, dot, moving).unsqueeze(-1)  # [*,T-1,1]

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


class RelationAwareTemporalAttention(nn.Module):
    """
    Temporal cross-attention from the current token to the token sequence,
    augmented with relative geometry and relative time features.
    """

    def __init__(self, embed_dim=128, num_heads=8, drop_path=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.rel_emb = FourierEmbedding(input_dim=4, hidden_dim=embed_dim)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.to_k_rel = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_v_rel = nn.Linear(embed_dim, embed_dim)
        self.to_bias = nn.Linear(embed_dim, num_heads)
        self.to_s = nn.Linear(embed_dim, embed_dim)
        self.to_g = nn.Linear(embed_dim * 2, embed_dim)
        self.to_out = nn.Linear(embed_dim, embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        query,
        seq,
        query_pos,
        query_head,
        seq_pos,
        seq_head,
        query_time,
        seq_time,
        key_padding_mask=None,
    ):
        B, M, D = query.shape
        S = seq.shape[2]
        BM = B * M

        query_norm = self.norm_q(query).reshape(BM, D)
        seq_norm = self.norm_kv(seq).reshape(BM, S, D)
        query_pos = query_pos.reshape(BM, 2)
        query_head = query_head.reshape(BM)
        seq_pos = seq_pos.reshape(BM, S, 2)
        seq_head = seq_head.reshape(BM, S)
        query_time = query_time.reshape(BM)
        seq_time = seq_time.reshape(BM, S)

        rel_pos = seq_pos - query_pos.unsqueeze(1)
        dist = torch.linalg.norm(rel_pos, dim=-1)
        valid_rel = dist > 1e-6

        query_head_cos = query_head.cos().unsqueeze(1)
        query_head_sin = query_head.sin().unsqueeze(1)
        cross = query_head_cos * rel_pos[..., 1] - query_head_sin * rel_pos[..., 0]
        dot = query_head_cos * rel_pos[..., 0] + query_head_sin * rel_pos[..., 1]
        direction = _safe_signed_angle(cross, dot, valid_rel)
        rel_head = wrap_angle(seq_head - query_head.unsqueeze(1))
        time_rel = (seq_time - query_time.unsqueeze(1)).to(query.dtype)

        rel_feat = torch.stack([dist, direction, rel_head, time_rel], dim=-1)
        rel_emb = self.rel_emb(rel_feat)

        q = self.to_q(query_norm).reshape(BM, self.num_heads, self.head_dim)
        k = self.to_k(seq_norm).reshape(BM, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(seq_norm).reshape(BM, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_rel = self.to_k_rel(rel_emb).reshape(BM, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_rel = self.to_v_rel(rel_emb).reshape(BM, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        bias = self.to_bias(rel_emb).permute(0, 2, 1)

        logits = (q.unsqueeze(2) * (k + k_rel)).sum(dim=-1) * self.scale + bias
        all_masked = None
        if key_padding_mask is not None:
            mask = key_padding_mask.reshape(BM, S).unsqueeze(1)
            all_masked = mask.all(dim=-1, keepdim=True)
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        attn = F.softmax(logits, dim=-1)
        if all_masked is not None:
            attn = torch.where(all_masked, torch.zeros_like(attn), attn)
        attn = F.dropout(attn, p=0.1, training=self.training)

        attn_out = (attn.unsqueeze(-1) * (v + v_rel)).sum(dim=2).reshape(BM, D)
        attn_out = self.to_out(attn_out)
        gate = torch.sigmoid(self.to_g(torch.cat([attn_out, query_norm], dim=-1)))
        attn_out = attn_out + gate * (self.to_s(query_norm) - attn_out)
        attn_out = attn_out.reshape(B, M, D)

        x = query + self.drop_path(attn_out)
        x = x + self.drop_path2(self.mlp(x))
        return x


class ModeAttention(nn.Module):
    """
    Relation-aware self-attention among modes at each autoregressive step.
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
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mode_emb = nn.Embedding(num_modes, embed_dim)
        self.time_emb = nn.Embedding(num_pred_steps + 1, embed_dim)
        self.rel_emb = FourierEmbedding(input_dim=3, hidden_dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.to_q = nn.Linear(embed_dim, embed_dim)
        self.to_k = nn.Linear(embed_dim, embed_dim)
        self.to_v = nn.Linear(embed_dim, embed_dim)
        self.to_k_rel = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_v_rel = nn.Linear(embed_dim, embed_dim)
        self.to_bias = nn.Linear(embed_dim, num_heads)
        self.to_s = nn.Linear(embed_dim, embed_dim)
        self.to_g = nn.Linear(embed_dim * 2, embed_dim)
        self.to_out = nn.Linear(embed_dim, embed_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, pred_step, mode_pos, mode_head):
        mode_ids = torch.arange(x.shape[1], device=x.device)
        time_ids = torch.full((1,), pred_step, device=x.device, dtype=torch.long)

        x = x + self.mode_emb(mode_ids).unsqueeze(0)
        x = x + self.time_emb(time_ids).view(1, 1, -1)

        x_norm = self.norm(x)

        rel_pos = mode_pos[:, None, :, :] - mode_pos[:, :, None, :]  # key - query
        dist = torch.linalg.norm(rel_pos, dim=-1)
        valid_rel = dist > 1e-6

        query_head_cos = mode_head.cos()
        query_head_sin = mode_head.sin()
        cross = (
            query_head_cos[:, :, None] * rel_pos[..., 1]
            - query_head_sin[:, :, None] * rel_pos[..., 0]
        )
        dot = (
            query_head_cos[:, :, None] * rel_pos[..., 0]
            + query_head_sin[:, :, None] * rel_pos[..., 1]
        )
        direction = _safe_signed_angle(cross, dot, valid_rel)
        rel_head = wrap_angle(mode_head[:, None, :] - mode_head[:, :, None])

        rel_feat = torch.stack([dist, direction, rel_head], dim=-1)
        rel_emb = self.rel_emb(rel_feat)

        B, M, D = x_norm.shape
        q = self.to_q(x_norm).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(x_norm).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(x_norm).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        k_rel = self.to_k_rel(rel_emb).reshape(B, M, M, self.num_heads, self.head_dim)
        v_rel = self.to_v_rel(rel_emb).reshape(B, M, M, self.num_heads, self.head_dim)
        bias = self.to_bias(rel_emb).permute(0, 3, 1, 2)

        k_rel = k_rel.permute(0, 3, 1, 2, 4)
        v_rel = v_rel.permute(0, 3, 1, 2, 4)

        logits = (
            q.unsqueeze(3) * (k.unsqueeze(2) + k_rel)
        ).sum(dim=-1) * self.scale + bias
        attn = F.softmax(logits, dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)

        attn_out = (attn.unsqueeze(-1) * (v.unsqueeze(2) + v_rel)).sum(dim=3)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, M, D)
        attn_out = self.to_out(attn_out)
        gate = torch.sigmoid(self.to_g(torch.cat([attn_out, x_norm], dim=-1)))
        attn_out = attn_out + gate * (self.to_s(x_norm) - attn_out)

        x = x + self.drop_path(attn_out)
        x = x + self.drop_path2(self.mlp(x))
        return x


class AutoregressiveStage(nn.Module):
    """
    One autoregressive decoding stage: [T-BiMamba-M] x num_repetitions per AR step.
    T = CrossAttention (tok queries temporal token sequence),
    BiMamba = spatial interaction (sorted scene + modes, update all, FINet-style residual),
    M = ModeAttention.
    With per-token local coordinate transforms, heading prediction,
    and over-prediction.
    """

    def __init__(
        self,
        embed_dim=128,
        t_per_tok=10,
        num_modes=6,
        num_pred_tokens=6,
        num_repetitions=2,
        num_heads=8,
        drop_path=0.2,
        is_refiner=False,
        over_predict=1,
    ):
        super().__init__()
        self.t_per_tok = t_per_tok
        self.num_pred_tokens = num_pred_tokens
        self.num_modes = num_modes
        self.num_repetitions = num_repetitions
        self.is_refiner = is_refiner
        self.over_predict = over_predict

        self.tokenizer = SimpleTokenizer(embed_dim=embed_dim, t_per_tok=t_per_tok)
        self.history_adapter_blocks = nn.ModuleList([
            create_block(
                d_model=embed_dim,
                layer_idx=i,
                drop_path=drop_path,
                bimamba=False,
                rms_norm=True,
            )
            for i in range(1)
        ])
        self.history_adapter_norm = RMSNorm(embed_dim, eps=1e-5)
        self.history_adapter_drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.history_to_mode = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # T: relation-aware attention from current token to temporal token sequence
        self.temporal_attns = nn.ModuleList([
            RelationAwareTemporalAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                drop_path=drop_path,
            )
            for _ in range(num_repetitions)
        ])

        # BiMamba: spatial interaction (scene + mode tokens, bidirectional)
        # FINet-style residual within each block
        self.bimamba_blocks = nn.ModuleList([
            create_block(
                d_model=embed_dim, layer_idx=i,
                drop_path=drop_path, bimamba=True, rms_norm=True,
            )
            for i in range(num_repetitions)
        ])
        self.bimamba_norms = nn.ModuleList([
            RMSNorm(embed_dim, eps=1e-5) for _ in range(num_repetitions)
        ])
        self.bimamba_drop_paths = nn.ModuleList([
            DropPath(drop_path) if drop_path > 0 else nn.Identity()
            for _ in range(num_repetitions)
        ])

        # M: Mode self-attention
        self.mode_attns = nn.ModuleList([
            ModeAttention(
                embed_dim=embed_dim, num_modes=num_modes,
                num_pred_steps=num_pred_tokens, num_heads=num_heads,
                drop_path=drop_path,
            )
            for _ in range(num_repetitions)
        ])

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
            self.future_query_scene_attn = Cross_Block(
                dim=embed_dim,
                num_heads=num_heads,
                drop_path=drop_path,
            )
            self.future_query_mamba = create_block(
                d_model=embed_dim,
                layer_idx=0,
                drop_path=drop_path,
                bimamba=False,
                rms_norm=True,
            )
            self.future_query_norm = RMSNorm(embed_dim, eps=1e-5)
            self.future_query_drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _sort_scene_by_center(self, scene_encoding, x_centers, valid_mask, center):
        """Sort scene tokens by distance to *center*; return sorted tensor + undo indices."""
        dists = ((x_centers - center.unsqueeze(1)) ** 2).sum(-1)  # [B, N]
        dists[~valid_mask] = 40000
        dists[:, 0] = -1  # ego always closest
        _, sort_idx = dists.sort(dim=1, descending=True)

        sorted_scene = torch.gather(
            scene_encoding, 1,
            sort_idx.unsqueeze(-1).expand(-1, -1, scene_encoding.size(2)),
        )
        return sorted_scene, sort_idx

    @staticmethod
    def _unsort_scene(sorted_scene, sort_idx):
        """Restore original token order after BiMamba."""
        return torch.scatter(
            sorted_scene, 1,
            sort_idx.unsqueeze(-1).expand(-1, -1, sorted_scene.size(2)),
            sorted_scene,
        )

    def _encode_history_tokens(
        self,
        history_positions,
        history_headings,
        history_mask,
    ):
        """Tokenize chunked ego history into a temporal token sequence."""
        hist_anchor_pos = history_positions[:, :, -1:, :]
        hist_anchor_head = history_headings[:, :, -1:]
        hist_local_pos, hist_local_head = global_to_local(
            history_positions,
            history_headings,
            hist_anchor_pos,
            hist_anchor_head,
        )
        hist_tokens = self.tokenizer(hist_local_pos, hist_local_head)
        hist_token_mask = history_mask.all(dim=-1)
        hist_tokens = hist_tokens * hist_token_mask.unsqueeze(-1)
        return hist_tokens, hist_token_mask

    def _adapt_history_tokens(self, hist_tokens, hist_token_mask):
        """Build a lightweight, stage-specific history state before rollout."""
        if hist_tokens.size(1) == 0:
            hist_state = hist_tokens.new_zeros(hist_tokens.size(0), hist_tokens.size(-1))
            return hist_tokens, hist_state

        x = hist_tokens
        residual = None
        hist_mask = hist_token_mask.unsqueeze(-1)
        for blk in self.history_adapter_blocks:
            x, residual = blk(x, residual)
            x = x * hist_mask

        fused_fn = rms_norm_fn if isinstance(self.history_adapter_norm, RMSNorm) else layer_norm_fn
        x = fused_fn(
            self.history_adapter_drop_path(x),
            self.history_adapter_norm.weight,
            self.history_adapter_norm.bias,
            eps=self.history_adapter_norm.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )
        x = x * hist_mask

        reversed_mask = torch.flip(hist_token_mask, dims=[1])
        last_valid_from_end = reversed_mask.long().argmax(dim=1)
        valid_counts = hist_token_mask.long().sum(dim=1)
        last_valid_idx = hist_token_mask.size(1) - 1 - last_valid_from_end
        last_valid_idx = torch.where(
            valid_counts > 0,
            last_valid_idx,
            torch.zeros_like(last_valid_idx),
        )
        batch_idx = torch.arange(x.size(0), device=x.device)
        hist_state = x[batch_idx, last_valid_idx]
        hist_state = hist_state * (valid_counts > 0).unsqueeze(-1)
        return x, hist_state

    def _refine_future_tokens(self, feat_stack, scene_encoding, scene_mask):
        """Inject scene memory into the full future-token sequence, then smooth it temporally."""
        B, M, S, D = feat_stack.shape
        future_queries = feat_stack.reshape(B, M * S, D)
        future_queries = self.future_query_scene_attn(
            future_queries,
            scene_encoding,
            key_padding_mask=scene_mask,
        )

        future_seq = future_queries.reshape(B * M, S, D)
        residual = None
        future_seq, residual = self.future_query_mamba(future_seq, residual)
        fused_fn = rms_norm_fn if isinstance(self.future_query_norm, RMSNorm) else layer_norm_fn
        future_seq = fused_fn(
            self.future_query_drop_path(future_seq),
            self.future_query_norm.weight,
            self.future_query_norm.bias,
            eps=self.future_query_norm.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True,
        )
        return future_seq.reshape(B, M, S, D)

    def _decode_token_geometry(
        self,
        tok,
        anchor_pos,
        anchor_head,
        step_idx,
        proposed_positions=None,
        proposed_headings=None,
    ):
        """Decode one token chunk into positions/headings for geometry-aware mode interaction."""
        T = self.t_per_tok
        pos_delta_full, _, head_delta_full, _ = self.detokenizer(tok)
        pos_delta = pos_delta_full[:, :, :T, :]
        head_delta = head_delta_full[:, :, :T]

        if self.is_refiner and proposed_positions is not None:
            start = step_idx * T
            end = start + T
            prop_local_pos, prop_local_head = global_to_local(
                proposed_positions[:, :, start:end, :],
                proposed_headings[:, :, start:end],
                anchor_pos.unsqueeze(2),
                anchor_head.unsqueeze(2),
            )
            local_pos = prop_local_pos + pos_delta
            local_head = wrap_angle(prop_local_head + head_delta)
        else:
            local_pos = torch.cumsum(pos_delta, dim=2)
            local_head = torch.cumsum(0.3 * torch.tanh(head_delta), dim=2)

        positions, headings = local_to_global(
            local_pos,
            local_head,
            anchor_pos.unsqueeze(2),
            anchor_head.unsqueeze(2),
        )
        return positions, headings

    @staticmethod
    def _assemble_over_preds(
        all_positions_over,
        all_scales_over,
        all_headings_over,
        all_concs_over,
        scal,
        conc_hat,
        T,
    ):
        """Assemble over-prediction outputs with per-token cumsum offset.

        Over-prediction uncertainty continues growing from the boundary reached
        at the end of each token's normal window — scale and concentration both
        start where the normal window ended, rather than resetting to a constant.
        """
        B, M = scal.shape[:2]
        S = scal.shape[2] // T

        y_hat_over = torch.cat(all_positions_over, dim=2)
        heading_over = torch.cat(all_headings_over, dim=2)

        # Scale: seed each token's over-prediction cumsum from its normal boundary.
        # scal[:, :, T-1::T, :] is the accumulated scale at each token's last step.
        scal_over_raw = torch.cat(all_scales_over, dim=2).reshape(B, M, S, T, 2)
        offsets_scal = scal[:, :, T - 1::T, :].unsqueeze(3)             # [B, M, S, 1, 2]
        scal_over = (
            offsets_scal + torch.cumsum(scal_over_raw, dim=3)
        ).reshape(B, M, S * T, 2)

        # Concentration: 1/conc_hat recovers the running denominator at each boundary;
        # continue the inverse-conc cumsum from there, then invert back.
        conc_over_raw = torch.cat(all_concs_over, dim=2).reshape(B, M, S, T)
        offsets_conc = (1.0 / conc_hat[:, :, T - 1::T]).unsqueeze(3)   # [B, M, S, 1]
        conc_over = (
            1.0 / (offsets_conc + torch.cumsum(conc_over_raw, dim=3))
        ).reshape(B, M, S * T)

        return y_hat_over, scal_over, heading_over, conc_over

    def _decode_token_sequence(
        self,
        token_stack,
        init_heading=None,
        proposed_positions=None,
        proposed_headings=None,
    ):
        """Decode a future-token sequence back into trajectory distribution parameters."""
        B, M, S, _ = token_stack.shape
        T = self.t_per_tok
        device = token_stack.device

        all_positions = []
        all_scales = []
        all_headings = []
        all_concs = []
        all_positions_over = []
        all_scales_over = []
        all_headings_over = []
        all_concs_over = []

        anchor_pos = torch.zeros(B, M, 2, device=device)
        if init_heading is not None:
            anchor_head = init_heading[:, None].expand(B, M)
        else:
            anchor_head = torch.zeros(B, M, device=device)

        for step in range(S):
            tok = token_stack[:, :, step]
            pos_delta_full, scale_full, head_delta_full, conc_full = self.detokenizer(tok)

            pos_delta = pos_delta_full[:, :, :T, :]
            head_delta = head_delta_full[:, :, :T]
            scale_norm = scale_full[:, :, :T, :]
            conc_norm = 1.0 / conc_full[:, :, :T].clamp_min(1e-6)

            if self.over_predict:
                pos_delta_over = pos_delta_full[:, :, T:, :]
                head_delta_over = head_delta_full[:, :, T:]
                scale_over = scale_full[:, :, T:, :]
                conc_over = 1.0 / conc_full[:, :, T:].clamp_min(1e-6)

            if self.is_refiner and proposed_positions is not None:
                start = step * T
                end = start + T
                prop_local_pos, prop_local_head = global_to_local(
                    proposed_positions[:, :, start:end, :],
                    proposed_headings[:, :, start:end],
                    anchor_pos.unsqueeze(2),
                    anchor_head.unsqueeze(2),
                )
                local_pos = prop_local_pos + pos_delta
                local_head = wrap_angle(prop_local_head + head_delta)
            else:
                local_pos = torch.cumsum(pos_delta, dim=2)
                local_head = torch.cumsum(0.3 * torch.tanh(head_delta), dim=2)

            positions, headings = local_to_global(
                local_pos,
                local_head,
                anchor_pos.unsqueeze(2),
                anchor_head.unsqueeze(2),
            )

            all_positions.append(positions)
            all_scales.append(scale_norm)
            all_headings.append(headings)
            all_concs.append(conc_norm)

            if self.over_predict:
                over_anchor_pos = positions[:, :, -1, :]
                over_anchor_head = headings[:, :, -1]

                if self.is_refiner and proposed_positions is not None:
                    over_start = start + T
                    over_end = min(over_start + T, proposed_positions.shape[2])
                    actual_len = over_end - over_start
                    if actual_len > 0:
                        prop_over_local_pos, prop_over_local_head = global_to_local(
                            proposed_positions[:, :, over_start:over_end, :],
                            proposed_headings[:, :, over_start:over_end],
                            over_anchor_pos.unsqueeze(2),
                            over_anchor_head.unsqueeze(2),
                        )
                        over_local_pos = prop_over_local_pos + pos_delta_over[:, :, :actual_len, :]
                        over_local_head = wrap_angle(
                            prop_over_local_head + head_delta_over[:, :, :actual_len]
                        )
                    else:
                        over_local_pos = torch.cumsum(pos_delta_over, dim=2)
                        over_local_head = torch.cumsum(0.3 * torch.tanh(head_delta_over), dim=2)
                        actual_len = T
                else:
                    over_local_pos = torch.cumsum(pos_delta_over, dim=2)
                    over_local_head = torch.cumsum(0.3 * torch.tanh(head_delta_over), dim=2)
                    actual_len = T

                positions_over, headings_over = local_to_global(
                    over_local_pos,
                    over_local_head,
                    over_anchor_pos.unsqueeze(2),
                    over_anchor_head.unsqueeze(2),
                )

                all_positions_over.append(positions_over)
                all_scales_over.append(scale_over[:, :, :actual_len, :])
                all_headings_over.append(headings_over)
                all_concs_over.append(conc_over[:, :, :actual_len])

            anchor_pos = positions[:, :, -1, :].detach()
            anchor_head = headings[:, :, -1].detach()

        y_hat = torch.cat(all_positions, dim=2)
        scal = torch.cat(all_scales, dim=2)
        scal = 0.1 + torch.cumsum(scal, dim=2)

        heading_hat = torch.cat(all_headings, dim=2)
        conc_hat = torch.cat(all_concs, dim=2)
        conc_hat = 1.0 / (0.02 + torch.cumsum(conc_hat, dim=2))

        if self.over_predict and all_positions_over:
            y_hat_over, scal_over, heading_over, conc_over = self._assemble_over_preds(
                all_positions_over, all_scales_over, all_headings_over, all_concs_over,
                scal, conc_hat, T,
            )
        else:
            y_hat_over = heading_over = scal_over = conc_over = None

        return y_hat, scal, heading_hat, conc_hat, y_hat_over, scal_over, heading_over, conc_over

    def forward(
        self,
        mode_tokens,
        ego_feat,
        scene_encoding,
        scene_mask,
        x_centers,
        valid_mask,
        init_sort_center=None,
        proposed_positions=None,
        proposed_headings=None,
        proposer_feats=None,
        init_heading=None,
        history_positions=None,
        history_headings=None,
        history_mask=None,
    ):
        B, M, D = mode_tokens.shape
        device = mode_tokens.device
        T = self.t_per_tok
        N_scene = scene_encoding.shape[1]

        if history_positions is not None and history_headings is not None and history_mask is not None:
            hist_tokens, hist_token_mask = self._encode_history_tokens(
                history_positions,
                history_headings,
                history_mask,
            )
            hist_tokens, hist_state = self._adapt_history_tokens(
                hist_tokens,
                hist_token_mask,
            )
            hist_seq = hist_tokens.unsqueeze(1).expand(-1, M, -1, -1)
            hist_seq_mask = hist_token_mask.unsqueeze(1).expand(-1, M, -1)
            hist_token_pos = history_positions[:, :, -1, :]
            hist_token_head = history_headings[:, :, -1]
            hist_seq_pos = hist_token_pos.unsqueeze(1).expand(-1, M, -1, -1)
            hist_seq_head = hist_token_head.unsqueeze(1).expand(-1, M, -1)
            hist_seq_time = torch.arange(
                hist_tokens.size(1),
                device=device,
                dtype=torch.long,
            ).view(1, 1, -1).expand(B, M, -1)
            init_tok = mode_tokens + self.history_to_mode(hist_state).unsqueeze(1)
        else:
            hist_seq = ego_feat.unsqueeze(1).unsqueeze(2).expand(B, M, 1, -1)
            hist_seq_mask = torch.ones(B, M, 1, device=device, dtype=torch.bool)
            hist_seq_pos = torch.zeros(B, M, 1, 2, device=device)
            if init_heading is not None:
                hist_seq_head = init_heading[:, None, None].expand(B, M, 1)
            else:
                hist_seq_head = torch.zeros(B, M, 1, device=device)
            hist_seq_time = torch.zeros(B, M, 1, device=device, dtype=torch.long)
            init_tok = mode_tokens

        token_seq_list = [hist_seq]
        token_valid_list = [hist_seq_mask]
        token_pos_list = [hist_seq_pos]
        token_head_list = [hist_seq_head]
        token_time_list = [hist_seq_time]

        all_positions = []
        all_scales = []
        all_headings = []
        all_concs = []
        all_positions_over = []
        all_scales_over = []
        all_headings_over = []
        all_concs_over = []
        step_feats = []

        anchor_pos = torch.zeros(B, M, 2, device=device)
        if init_heading is not None:
            anchor_head = init_heading[:, None].expand(B, M)
        else:
            anchor_head = torch.zeros(B, M, device=device)

        prev_positions = None
        prev_headings = None

        # Sort center: use provided init or default to ego origin
        if init_sort_center is not None:
            sort_center = init_sort_center.clone()
        else:
            sort_center = x_centers[:, 0].clone()  # [B, 2]

        for step in range(self.num_pred_tokens):
            # ---- Tokenize ----
            if step == 0:
                tok = init_tok
            else:
                local_pos, local_head = global_to_local(
                    prev_positions, prev_headings,
                    anchor_pos.unsqueeze(2), anchor_head.unsqueeze(2),
                )
                tok = self.tokenizer(local_pos, local_head)

            if self.is_refiner and proposer_feats is not None:
                tok = tok + self.feature_fuse(proposer_feats[step])

            tok_pos = anchor_pos.unsqueeze(2)
            tok_head = anchor_head.unsqueeze(2)
            tok_time = torch.full(
                (B, M, 1),
                hist_seq.shape[2] + step,
                device=device,
                dtype=torch.long,
            )
            seq = torch.cat(token_seq_list, dim=2)  # [B, M, seq_len, D]
            seq_valid = torch.cat(token_valid_list, dim=2)  # [B, M, seq_len]
            seq_pos = torch.cat(token_pos_list, dim=2)  # [B, M, seq_len, 2]
            seq_head = torch.cat(token_head_list, dim=2)  # [B, M, seq_len]
            seq_time = torch.cat(token_time_list, dim=2)  # [B, M, seq_len]

            # ---- [T → BiMamba → M] × num_repetitions ----
            for rep in range(self.num_repetitions):
                # -- T: tok queries seq with explicit relative geometry/time --
                tok = self.temporal_attns[rep](
                    tok,
                    seq,
                    query_pos=tok_pos.squeeze(2),
                    query_head=tok_head.squeeze(2),
                    seq_pos=seq_pos,
                    seq_head=seq_head,
                    query_time=tok_time.squeeze(2),
                    seq_time=seq_time,
                    key_padding_mask=~seq_valid,
                )

                # -- BiMamba: spatial interaction (sorted scene + modes, update all) --
                sorted_scene, sort_idx = self._sort_scene_by_center(
                    scene_encoding, x_centers, valid_mask, sort_center)
                x_combined = torch.cat([sorted_scene, tok], dim=1)  # [B, N+M, D]

                residual_bi = None
                x_combined, residual_bi = self.bimamba_blocks[rep](x_combined, residual_bi)
                fused_fn = rms_norm_fn if isinstance(self.bimamba_norms[rep], RMSNorm) else layer_norm_fn
                x_combined = fused_fn(
                    self.bimamba_drop_paths[rep](x_combined),
                    self.bimamba_norms[rep].weight,
                    self.bimamba_norms[rep].bias,
                    eps=self.bimamba_norms[rep].eps,
                    residual=residual_bi,
                    prenorm=False,
                    residual_in_fp32=True,
                )

                sorted_scene_out = x_combined[:, :N_scene]
                tok = x_combined[:, N_scene:]  # [B, M, D]
                scene_encoding = self._unsort_scene(sorted_scene_out, sort_idx)

                # -- M: Mode self-attention with relative geometry from the
                # current chunk endpoint prediction.
                # torch.no_grad() is intentional: _decode_token_geometry
                # calls self.detokenizer only to obtain geometric positions
                # for routing — its outputs never enter the loss directly.
                # Without the guard, detokenizer.shared / pos_head /
                # heading_head would receive (R+1) gradient paths per AR
                # step while scale_head / conc_head only get 1, causing a
                # systematic 3:1 imbalance that under-trains uncertainty.
                # ModeAttention spatial params (to_k_rel, to_v_rel, to_bias)
                # still train correctly: PyTorch computes d(loss)/d(weight)
                # even when the input tensor has requires_grad=False.
                with torch.no_grad():
                    mode_positions, mode_headings = self._decode_token_geometry(
                        tok,
                        anchor_pos,
                        anchor_head,
                        step,
                        proposed_positions=proposed_positions,
                        proposed_headings=proposed_headings,
                    )
                tok = self.mode_attns[rep](
                    tok,
                    step + 1,
                    mode_pos=mode_positions[:, :, -1, :],
                    mode_head=mode_headings[:, :, -1],
                )

            # ---- Detokenize ----
            pos_delta_full, scale_full, head_delta_full, conc_full = self.detokenizer(tok)

            pos_delta = pos_delta_full[:, :, :T, :]
            head_delta = head_delta_full[:, :, :T]
            scale_norm = scale_full[:, :, :T, :]
            # Match DONUT's heading uncertainty parameterization:
            # accumulate inverse concentration across AR steps, then invert again.
            conc_norm = 1.0 / conc_full[:, :, :T].clamp_min(1e-6)

            if self.over_predict:
                pos_delta_over = pos_delta_full[:, :, T:, :]
                head_delta_over = head_delta_full[:, :, T:]
                scale_over = scale_full[:, :, T:, :]
                conc_over = 1.0 / conc_full[:, :, T:].clamp_min(1e-6)

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

            mem_tok_pos = positions[:, :, -1:, :].detach()
            mem_tok_head = headings[:, :, -1:].detach()

            token_seq_list.append(tok.unsqueeze(2))
            token_valid_list.append(torch.ones(B, M, 1, device=device, dtype=torch.bool))
            token_pos_list.append(mem_tok_pos)
            token_head_list.append(mem_tok_head)
            token_time_list.append(tok_time)
            step_feats.append(tok)

            # ---- Over-prediction: extend from normal's last position ----
            if self.over_predict:
                over_anchor_pos = positions[:, :, -1, :]
                over_anchor_head = headings[:, :, -1]

                if self.is_refiner and proposed_positions is not None:
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

            # Update anchors for next step
            anchor_pos = mem_tok_pos.squeeze(2)
            anchor_head = mem_tok_head.squeeze(2)
            prev_positions = positions.detach()
            prev_headings = headings.detach()

            # Update sort center with pi-weighted predicted endpoint
            with torch.no_grad():
                pi_now = self.pi_head(tok).squeeze(-1)  # [B, M]
                pi_weights = F.softmax(pi_now, dim=1).unsqueeze(-1)  # [B, M, 1]
                sort_center = (pi_weights * positions[:, :, -1, :]).sum(dim=1)  # [B, 2]

        # ---- Assemble normal output ----
        y_hat = torch.cat(all_positions, dim=2)
        scal = torch.cat(all_scales, dim=2)
        scal = 0.1 + torch.cumsum(scal, dim=2)

        heading_hat = torch.cat(all_headings, dim=2)
        conc_hat = torch.cat(all_concs, dim=2)
        conc_hat = 1.0 / (0.02 + torch.cumsum(conc_hat, dim=2))

        feat_stack = torch.stack(step_feats, dim=2)
        if self.is_refiner:
            feat_stack = self._refine_future_tokens(
                feat_stack,
                scene_encoding,
                scene_mask,
            )
            step_feats = [feat_stack[:, :, i] for i in range(feat_stack.size(2))]
            (
                y_hat,
                scal,
                heading_hat,
                conc_hat,
                y_hat_over,
                scal_over,
                heading_over,
                conc_over,
            ) = self._decode_token_sequence(
                feat_stack,
                init_heading=init_heading,
                proposed_positions=proposed_positions,
                proposed_headings=proposed_headings,
            )
        # Use the last AR token as the trajectory-level feature for pi.
        # Through temporal attention, each token attends to all previous tokens,
        # so the final token implicitly aggregates the full sequence — consistent
        # with DONUT's design of reading pi from the last step's dedicated output.
        feat_pool = feat_stack[:, :, -1, :]
        pi = self.pi_head(feat_pool).squeeze(-1)

        # ---- Assemble over-prediction output ----
        # Refiner: already computed by _decode_token_sequence from globally-refined
        #          tokens — do NOT overwrite with AR-phase results.
        # Proposer: assemble from AR-phase collected lists (no global refinement).
        if not self.is_refiner:
            if self.over_predict and all_positions_over:
                y_hat_over, scal_over, heading_over, conc_over = self._assemble_over_preds(
                    all_positions_over, all_scales_over, all_headings_over, all_concs_over,
                    scal, conc_hat, T,
                )
            else:
                y_hat_over = heading_over = scal_over = conc_over = None

        return (y_hat, pi, scal, step_feats, heading_hat, conc_hat,
                y_hat_over, scal_over, heading_over, conc_over)


class DonutMambaDecoder(nn.Module):
    """
    DONUT-style autoregressive decoder: [T-R-BiMamba-M] x num_repetitions.
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
        dec_layer_1=2,
        dec_layer_2=2,
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
            num_repetitions=dec_layer_1,
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
            num_repetitions=dec_layer_2,
            num_heads=num_heads,
            drop_path=drop_path,
            is_refiner=True,
            over_predict=over_predict,
        )

    # NOTE: forward() is intentionally omitted.
    # model_forecast.py calls self.time_decoder.proposer(...) and
    # self.time_decoder.refiner(...) directly with stage-specific
    # arguments (x_centers, valid_mask, init_sort_center, etc.)
    # that differ between the two stages.
