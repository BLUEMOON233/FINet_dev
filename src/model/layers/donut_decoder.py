import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from ..models_mamba import RMSNorm, layer_norm_fn, rms_norm_fn
from .mamba.vim_mamba import create_block
from .transformer_blocks import Cross_Block


class SimpleTokenizer(nn.Module):
    """
    Converts `t_per_tok` position steps into one token feature vector.
    """

    def __init__(self, embed_dim=128, t_per_tok=10):
        super().__init__()
        self.step_embed = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.aggregate = nn.Linear((t_per_tok - 1) * embed_dim, embed_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, positions):
        deltas = positions[..., 1:, :] - positions[..., :-1, :]
        velocity = torch.linalg.norm(deltas, dim=-1, keepdim=True)
        rel_pos = positions[..., 1:, :] - positions[..., :1, :]
        features = torch.cat([deltas, velocity, rel_pos], dim=-1)

        x = self.step_embed(features)
        x = x.flatten(-2, -1)
        x = self.aggregate(x)
        x = self.out_proj(x)
        return x


class SimpleDetokenizer(nn.Module):
    """
    Converts a token feature into position deltas and Laplace scale.
    """

    def __init__(self, embed_dim=128, t_per_tok=10):
        super().__init__()
        self.t_per_tok = t_per_tok
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, t_per_tok * 2),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, t_per_tok * 2),
        )

    def forward(self, x):
        x = self.shared(x)
        pos_delta = self.pos_head(x).reshape(*x.shape[:2], self.t_per_tok, 2)
        scale = (1.0 + F.elu(self.scale_head(x))).clamp(min=1e-6)
        scale = scale.reshape(*x.shape[:2], self.t_per_tok, 2)
        return pos_delta, scale


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
    ):
        super().__init__()
        self.t_per_tok = t_per_tok
        self.num_pred_tokens = num_pred_tokens
        self.num_modes = num_modes
        self.is_refiner = is_refiner

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
        self.detokenizer = SimpleDetokenizer(embed_dim=embed_dim, t_per_tok=t_per_tok)
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
        proposer_feats=None,
    ):
        batch_size, num_modes, _ = mode_tokens.shape
        hist_token = ego_feat.unsqueeze(1).unsqueeze(2).expand(batch_size, num_modes, 1, -1)
        token_seq_list = [hist_token]

        all_positions = []
        all_scales = []
        step_feats = []
        anchor_pos = None
        prev_positions = None

        for step in range(self.num_pred_tokens):
            if step == 0:
                tok = mode_tokens
            else:
                tok = self.tokenizer(prev_positions)

            if self.is_refiner and proposer_feats is not None:
                tok = tok + self.feature_fuse(proposer_feats[step])

            token_seq_list.append(tok.unsqueeze(2))

            seq = torch.cat(token_seq_list, dim=2)
            seq_flat = seq.reshape(batch_size * num_modes, -1, seq.shape[-1])
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
            tok = seq_flat[:, -1].reshape(batch_size, num_modes, -1)

            tok = self.cross_attn(tok, scene_encoding, key_padding_mask=scene_mask)
            tok = self.mode_attn(tok, step + 1)
            step_feats.append(tok)

            pos_delta, scale = self.detokenizer(tok)

            if self.is_refiner and proposed_positions is not None:
                start = step * self.t_per_tok
                end = start + self.t_per_tok
                positions = proposed_positions[:, :, start:end, :] + pos_delta
            else:
                cum_delta = torch.cumsum(pos_delta, dim=2)
                if anchor_pos is None:
                    positions = cum_delta
                else:
                    positions = anchor_pos.unsqueeze(2) + cum_delta

            all_positions.append(positions)
            all_scales.append(scale)
            anchor_pos = positions[:, :, -1, :].detach()
            prev_positions = positions.detach()

        y_hat = torch.cat(all_positions, dim=2)
        scal = torch.cat(all_scales, dim=2)
        scal = 0.1 + torch.cumsum(scal, dim=2)

        feat_stack = torch.stack(step_feats, dim=2)
        feat_pool = feat_stack.max(dim=2)[0]
        pi = self.pi_head(feat_pool).squeeze(-1)

        return y_hat, pi, scal, step_feats


class DonutMambaDecoder(nn.Module):
    """
    DONUT-style autoregressive decoder with a Mamba temporal core.
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
    ):
        super().__init__()
        if future_steps % t_per_tok != 0:
            raise ValueError(
                f"future_steps ({future_steps}) must be divisible by t_per_tok ({t_per_tok})"
            )

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
        )

    def forward(self, mode_tokens, ego_feat, scene_encoding, mask=None):
        y_hat, pi, scal, proposer_feats = self.proposer(
            mode_tokens=mode_tokens,
            ego_feat=ego_feat,
            scene_encoding=scene_encoding,
            scene_mask=mask,
        )
        new_y_hat, new_pi, scal_new, _ = self.refiner(
            mode_tokens=mode_tokens,
            ego_feat=ego_feat,
            scene_encoding=scene_encoding,
            scene_mask=mask,
            proposed_positions=y_hat.detach(),
            proposer_feats=proposer_feats,
        )

        dense_pred = None
        mode_features = None
        return dense_pred, y_hat, pi, mode_features, new_y_hat, new_pi, scal, scal_new

