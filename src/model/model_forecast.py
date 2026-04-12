from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.donut_decoder import DonutMambaDecoder
from .layers.mamba.vim_mamba import create_block
from timm.models.layers import DropPath
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .models_mamba import RMSNorm, rms_norm_fn

class MultimodalDecoder(nn.Module):
    """A naive MLP-based multimodal decoder"""

    def __init__(self, embed_dim) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.multimodal_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

        self.loc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, x):
        x = self.multimodal_proj(x)
        loc = self.loc(x).view(-1, 2)
        return loc, x

class ModelForecast(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 60,
        enc_layer_1: int = 4,
        enc_layer_2: int = 2,
        dec_layer_1: int = 2,
        dec_layer_2: int = 4,
        t_per_tok: int = 10,
    ) -> None:
        super().__init__()

        self.hist_embed_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

        # Agent Encoding Mamba
        self.hist_embed_mamba = nn.ModuleList(
            [
                create_block(
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=0.2,
                    bimamba=False,
                    rms_norm=True,
                )
                for i in range(4)
            ]
        )
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)
        self.drop_path = DropPath(drop_path)

        self.lane_embed = LaneEmbeddingLayer(3, embed_dim) # same as forecast-mae

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(3, embed_dim))

        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, future_steps * 2)
        )

        self.t_per_tok = t_per_tok
        self.time_decoder = DonutMambaDecoder(
            embed_dim=embed_dim,
            t_per_tok=self.t_per_tok,
            num_modes=6,
            future_steps=future_steps,
            dec_layer_1=dec_layer_1,
            dec_layer_2=dec_layer_2,
            num_heads=num_heads,
            drop_path=drop_path,
        )

        # Endpoint predictor for initial sorting
        self.decoder0 = MultimodalDecoder(embed_dim)

        self.future_steps = future_steps
        self.tokens = nn.Parameter(torch.randn(1, 6, embed_dim))

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    # ------------------------------------------------------------------ #
    #  Forward: Encode → Proposer → Refiner
    # ------------------------------------------------------------------ #

    def forward(self, data):

        ###### Scene context encoding ######
        # agent encoding
        hist_valid_mask = data["x_valid_mask"]
        hist_key_valid_mask = data["x_key_valid_mask"]
        hist_valid_float = hist_valid_mask.to(data["x_angles"].dtype)
        hist_feat = torch.cat(
            [
                data["x_positions_diff"],
                data["x_velocity_diff"][..., None],
                hist_valid_mask[..., None],
                data["x_heading_diff"][..., None],
                (torch.cos(data["x_angles"]) * hist_valid_float)[..., None],
                (torch.sin(data["x_angles"]) * hist_valid_float)[..., None],
            ],
            dim=-1,
        )  # [B, N, 50, 7]

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)

        # unidirectional mamba for agent history
        actor_feat = self.hist_embed_mlp(hist_feat[hist_feat_key_valid].contiguous())
        residual = None
        for blk_mamba in self.hist_embed_mamba:
            actor_feat, residual = blk_mamba(actor_feat, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        actor_feat = fused_add_norm_fn(
            self.drop_path(actor_feat),
            self.norm_f.weight, self.norm_f.bias,
            eps=self.norm_f.eps, residual=residual,
            prenorm=False, residual_in_fp32=True,
        )

        actor_feat = actor_feat[:, -1]
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[hist_feat_key_valid] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        # map encoding
        lane_valid_mask = data["lane_valid_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, lane_valid_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        # type embedding and position embedding
        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, -1], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed[data["lane_attr"][..., 0].long()]
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        # scene context features
        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)  # [B, N+M, 128]
        key_valid_mask = torch.cat(
            [data["x_key_valid_mask"], data["lane_key_valid_mask"]], dim=1
        )
        x_encoder = x_encoder + pos_embed
        x_encoder = self.norm(x_encoder)

        ###### Initialize mode tokens (FINet style) ######
        ep_offset_1, ep_tok_1 = self.decoder0(x_encoder[:, 0])
        # ego token x 6 + learnable per-mode tokens + endpoint token on first mode
        mode_tokens = x_encoder[:, 0:1] + self.tokens  # [B, 1, D] + [1, 6, D] -> [B, 6, D]
        mode_tokens = torch.cat(
            [mode_tokens[:, :1] + ep_tok_1.unsqueeze(1), mode_tokens[:, 1:]], dim=1
        )

        valid_mask = (x_centers.sum(-1) != 0)
        valid_mask[:, 0] = True
        ego_feat = x_encoder[:, 0]
        init_heading = data["x_angles"][:, 0, -1]

        ego_history_positions = data["x_positions"][:, 0]
        ego_history_headings = data["x_angles"][:, 0]
        ego_history_mask = data["x_valid_mask"][:, 0]
        hist_steps = ego_history_positions.shape[1]
        if hist_steps % self.t_per_tok != 0:
            raise ValueError(
                f"ego history length ({hist_steps}) must be divisible by t_per_tok ({self.t_per_tok})"
            )
        num_hist_tokens = hist_steps // self.t_per_tok
        ego_history_positions = ego_history_positions.reshape(B, num_hist_tokens, self.t_per_tok, 2)
        ego_history_headings = ego_history_headings.reshape(B, num_hist_tokens, self.t_per_tok)
        ego_history_mask = ego_history_mask.reshape(B, num_hist_tokens, self.t_per_tok)

        proposer_history_positions = ego_history_positions[:, :-1]
        proposer_history_headings = ego_history_headings[:, :-1]
        proposer_history_mask = ego_history_mask[:, :-1]
        refiner_history_positions = ego_history_positions[:, 1:]
        refiner_history_headings = ego_history_headings[:, 1:]
        refiner_history_mask = ego_history_mask[:, 1:]

        ###### Proposer (init sort by decoder0 endpoint) ######
        proposer_sort_center = x_centers[:, 0] + ep_offset_1.detach()

        (y_hat, pi, scal, proposer_feats, heading_hat, conc_hat,
         y_hat_over, scal_over, heading_over, conc_over) = \
            self.time_decoder.proposer(
                mode_tokens=mode_tokens,
                ego_feat=ego_feat,
                scene_encoding=x_encoder,
                scene_mask=~key_valid_mask,
                x_centers=x_centers,
                valid_mask=valid_mask,
                init_sort_center=proposer_sort_center,
                init_heading=init_heading,
                history_positions=proposer_history_positions,
                history_headings=proposer_history_headings,
                history_mask=proposer_history_mask,
            )

        ###### Pi-weighted endpoint for Refiner sorting (detached) ######
        pi_weights = F.softmax(pi, dim=1).unsqueeze(-1)       # [B, 6, 1]
        proposer_endpoint = (
            pi_weights * y_hat[:, :, -1, :]
        ).sum(dim=1).detach()                                  # [B, 2]

        ###### Other agents (from encoder features) ######
        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(
            B, x_others.size(1), -1, 2
        )

        ###### Refiner (init sort by Proposer endpoint) ######
        (new_y_hat, new_pi, scal_new, _, new_heading_hat, new_conc_hat,
         new_y_hat_over, new_scal_over, new_heading_over, new_conc_over) = \
            self.time_decoder.refiner(
                mode_tokens=mode_tokens,
                ego_feat=ego_feat,
                scene_encoding=x_encoder,
                scene_mask=~key_valid_mask,
                x_centers=x_centers,
                valid_mask=valid_mask,
                init_sort_center=proposer_endpoint,
                proposed_positions=y_hat.detach(),
                proposed_headings=heading_hat.detach(),
                proposer_feats=proposer_feats,
                init_heading=init_heading,
                history_positions=refiner_history_positions,
                history_headings=refiner_history_headings,
                history_mask=refiner_history_mask,
            )

        return {
            "y_hat": y_hat, "pi": pi, "scal": scal,
            "dense_predict": None,
            "ep_offsets": [ep_offset_1],
            "y_hat_others": y_hat_others,
            "new_y_hat": new_y_hat, "new_pi": new_pi, "scal_new": scal_new,
            "heading_hat": heading_hat, "conc_hat": conc_hat,
            "new_heading_hat": new_heading_hat, "new_conc_hat": new_conc_hat,
            "y_hat_over": y_hat_over, "scal_over": scal_over,
            "heading_over": heading_over, "conc_over": conc_over,
            "new_y_hat_over": new_y_hat_over, "new_scal_over": new_scal_over,
            "new_heading_over": new_heading_over, "new_conc_over": new_conc_over,
        }
