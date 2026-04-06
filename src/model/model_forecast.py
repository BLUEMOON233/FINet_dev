from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block, InteractionBlock
from .layers.time_decoder import TimeDecoder
from .layers.mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
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
        num_modes: int = 6,
        use_goal_conditioned_tokens: bool = False,
        num_goal_candidates: int = 16,
        use_topology_features: bool = True,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.use_goal_conditioned_tokens = use_goal_conditioned_tokens
        self.num_goal_candidates = num_goal_candidates
        self.use_topology_features = use_topology_features

        self.hist_embed_mlp = nn.Sequential(
            nn.Linear(4, 64),
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

        self.time_decoder = TimeDecoder(dec_layer_1=dec_layer_1, dec_layer_2=dec_layer_2)
        
        num_layers = 4
        encoder_depth = 4
        bimamba_type="none"
        norm_layer = nn.LayerNorm
        self.samba_blocks1 = nn.ModuleList(
            [
                create_block(  
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(enc_layer_1)
            ]
        )
        self.norm_f_1 = RMSNorm(embed_dim, eps=1e-5)
        self.drop_path_1 = DropPath(drop_path)
        self.decoder1 = MultimodalDecoder(embed_dim)

        # dpr = [x.item() for x in torch.linspace(0, drop_path, sum(encoder_depth))]
        self.samba_blocks2 = nn.ModuleList(
            [
                create_block(  
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(enc_layer_2) #encoder_depth)
            ]
        )
        self.norm_f_2 = RMSNorm(embed_dim, eps=1e-5)
        self.drop_path_2 = DropPath(drop_path)
        # self.fut_tok = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.fut_mlp = nn.Linear(embed_dim, embed_dim)
        
        self.decoder0 = MultimodalDecoder(embed_dim)
        
        self.future_steps = future_steps
        self.tokens = nn.Parameter(torch.randn(1, self.num_modes, embed_dim))
        self.goal_input_dim = 10
        self.goal_encoder = nn.Sequential(
            nn.Linear(self.goal_input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.goal_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

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
            
        # nn.init.constant_(self.fut_mlp.weight, 1)  # Set all weights to 1
        # nn.init.constant_(self.fut_mlp.bias, 0)    # Set all biases to 0

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    @staticmethod
    def _gather_by_index(src: torch.Tensor, idx: torch.Tensor):
        view_shape = list(idx.shape) + [1] * (src.dim() - 2)
        expand_shape = list(idx.shape) + list(src.shape[2:])
        gather_idx = idx.view(*view_shape).expand(*expand_shape)
        return torch.gather(src, dim=1, index=gather_idx)

    def _build_goal_tokens(self, data: dict, ego_feat: torch.Tensor):
        lane_centers = data["lane_centers"]
        lane_angles = data["lane_angles"]
        lane_attr = data["lane_attr"]
        lane_positions = data["lane_positions"]
        lane_mask = data["lane_key_valid_mask"]
        ego_center = data["x_centers"][:, 0]
        ego_heading = data["x_angles"][:, 0, -1]

        batch_size, num_lanes, _ = lane_centers.shape
        num_candidates = min(self.num_goal_candidates, num_lanes)
        if num_candidates <= 0:
            mode_tokens = self.tokens.expand(batch_size, -1, -1)
            return {
                "future_mode_tokens": mode_tokens,
                "goal_logits": None,
                "goal_candidate_xy": None,
                "goal_candidate_mask": None,
                "goal_topk_idx": None,
            }

        dist_to_ego = torch.norm(lane_centers - ego_center.unsqueeze(1), dim=-1)
        heading_align = torch.cos(lane_angles - ego_heading.unsqueeze(1))
        rank_score = heading_align - 0.02 * dist_to_ego
        rank_score = rank_score.masked_fill(~lane_mask, -1e9)
        candidate_idx = torch.topk(rank_score, k=num_candidates, dim=-1).indices

        candidate_xy = self._gather_by_index(lane_centers, candidate_idx)
        candidate_heading = self._gather_by_index(lane_angles.unsqueeze(-1), candidate_idx).squeeze(-1)
        candidate_attr = self._gather_by_index(lane_attr, candidate_idx)
        candidate_pos = self._gather_by_index(lane_positions, candidate_idx)
        candidate_mask = self._gather_by_index(lane_mask.unsqueeze(-1), candidate_idx).squeeze(-1)

        no_valid = ~candidate_mask.any(dim=-1)
        if no_valid.any():
            candidate_mask = candidate_mask.clone()
            candidate_xy = candidate_xy.clone()
            candidate_heading = candidate_heading.clone()
            candidate_attr = candidate_attr.clone()
            candidate_pos = candidate_pos.clone()
            candidate_mask[no_valid, 0] = True
            candidate_xy[no_valid, 0] = ego_center[no_valid]
            candidate_heading[no_valid, 0] = ego_heading[no_valid]
            candidate_attr[no_valid, 0] = 0
            candidate_pos[no_valid, 0] = 0

        curvature = torch.norm(
            candidate_pos[:, :, -1] - 2 * candidate_pos[:, :, candidate_pos.size(2) // 2] + candidate_pos[:, :, 0],
            dim=-1,
        )
        lane_type = candidate_attr[..., 0]
        lane_width = candidate_attr[..., 1]
        is_intersection = candidate_attr[..., 2]
        if not self.use_topology_features:
            lane_type = torch.zeros_like(lane_type)
            is_intersection = torch.zeros_like(is_intersection)

        dist_goal = torch.norm(candidate_xy - ego_center.unsqueeze(1), dim=-1)
        heading_goal_align = torch.cos(candidate_heading - ego_heading.unsqueeze(1))
        goal_feat = torch.stack(
            [
                candidate_xy[..., 0],
                candidate_xy[..., 1],
                torch.sin(candidate_heading),
                torch.cos(candidate_heading),
                dist_goal,
                heading_goal_align,
                curvature,
                lane_width,
                lane_type / 2.0,
                is_intersection,
            ],
            dim=-1,
        )

        goal_embed = self.goal_encoder(goal_feat)
        ego_expand = ego_feat.unsqueeze(1).expand(-1, num_candidates, -1)
        goal_logits = self.goal_scorer(torch.cat([goal_embed, ego_expand], dim=-1)).squeeze(-1)
        goal_logits = goal_logits.masked_fill(~candidate_mask, -1e9)

        select_k = min(self.num_modes, num_candidates)
        goal_topk_idx = torch.topk(goal_logits, k=select_k, dim=-1).indices
        selected_goal_embed = self._gather_by_index(goal_embed, goal_topk_idx)

        if select_k < self.num_modes:
            pad_shape = (batch_size, self.num_modes - select_k, selected_goal_embed.size(-1))
            selected_goal_embed = torch.cat(
                [selected_goal_embed, selected_goal_embed.new_zeros(pad_shape)], dim=1
            )

        mode_tokens = self.tokens.expand(batch_size, -1, -1)
        future_mode_tokens = mode_tokens + selected_goal_embed

        return {
            "future_mode_tokens": future_mode_tokens,
            "goal_logits": goal_logits,
            "goal_candidate_xy": candidate_xy,
            "goal_candidate_mask": candidate_mask,
            "goal_topk_idx": goal_topk_idx,
        }

    def spatial_mamba(self, x_encoder, x_centers, future_mode_tokens=None):
        ep_offset_1, ep_tok_1 = self.decoder0(x_encoder[:,0])
        # ep_offset_1 = ep_offset_1.detach()

        valid_mask = (x_centers.sum(-1) != 0)
        valid_mask[:,0] = True
        
        center = x_centers[:,0] + ep_offset_1
        dists = ((x_centers - center.unsqueeze(1))**2).sum(-1)
        dists[~valid_mask] = 40000  # invalid tokens sort first (before valid, before ego)
        dists[:,0] = -1
        dists_sort, indexes = dists.sort(dim=1, descending=True)

        x_encoder = torch.gather(x_encoder, 1, indexes.unsqueeze(-1).expand(-1, -1, x_encoder.size(2)))
        
        fut_tok = x_encoder[:,-1:].clone()
        if future_mode_tokens is None:
            fut_tok = fut_tok + self.tokens
        else:
            fut_tok = fut_tok + future_mode_tokens
        fut_tok = torch.cat([fut_tok[:,:1] + ep_tok_1.unsqueeze(1), fut_tok[:,1:]], 1)

        x_encoder = torch.cat([x_encoder, fut_tok], 1)

        # apply Samba blocks ----------------
        #! First round: init sort & predict ego endpoint
        residual = None
        for blk in self.samba_blocks1:
            x_encoder, residual = blk(x_encoder, residual)                                      # [bs, N+M, D]
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f_1, RMSNorm) else layer_norm_fn
        x_encoder = fused_add_norm_fn(
            self.drop_path_1(x_encoder),
            self.norm_f_1.weight,
            self.norm_f_1.bias,
            eps=self.norm_f_1.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        ) # [421, 50, 128]

        fut_tok = x_encoder[:, -self.num_modes:]
        x_encoder = x_encoder[:, :-self.num_modes]

        x_encoder = torch.scatter(x_encoder, 1, indexes.unsqueeze(-1).expand(-1, -1, x_encoder.size(2)), x_encoder)

        x_ego = fut_tok[:,0]
        ep_offset_2, ep_tok_2 = self.decoder1(x_ego)
        # ep_offset_2 = ep_offset_2.detach()

        
        # center_init = y_hat_init[..., -1, :]
        fut_tok = torch.cat([fut_tok[:,:1] + ep_tok_2.unsqueeze(1), fut_tok[:,1:]], 1)

        center = x_centers[:,0] + ep_offset_2
        dists = ((x_centers - center.unsqueeze(1))**2).sum(-1)
        dists[~valid_mask] = 40000  # invalid tokens sort first (before valid, before ego)
        dists[:,0] = -1
        dists_sort, indexes = dists.sort(dim=1, descending=True)

        x_encoder = torch.gather(x_encoder, 1, indexes.unsqueeze(-1).expand(-1, -1, x_encoder.size(2)))
        x_encoder = torch.cat([x_encoder, fut_tok], 1)
        #! Second round: resort & predict ego endpoint
        # for blk in self.samba_blocks2:
        #     x_encoder = blk(x_encoder)                                      # [bs, N+M, D]
        residual = None
        for blk in self.samba_blocks2:
            x_encoder, residual = blk(x_encoder, residual)   
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f_2, RMSNorm) else layer_norm_fn
        x_encoder = fused_add_norm_fn(
            self.drop_path_2(x_encoder),
            self.norm_f_2.weight,
            self.norm_f_2.bias,
            eps=self.norm_f_2.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        ) # [421, 50, 128]
        fut_tok = x_encoder[:, -self.num_modes:]
        x_encoder = x_encoder[:, :-self.num_modes]
        
        # #! query-base cross-attention
        x_encoder = torch.scatter(x_encoder, 1, indexes.unsqueeze(-1).expand(-1, -1, x_encoder.size(2)), x_encoder)
        return x_encoder, fut_tok, [ep_offset_1, ep_offset_2]
        # return x_encoder, fut_tok, [ep_offset_2]



    def forward(self, data):
        

        ###### Scene context encoding ###### 
        # agent encoding
        hist_valid_mask = data["x_valid_mask"] # [16, 48, 50]
        hist_key_valid_mask = data["x_key_valid_mask"] # [16, 48]
        hist_feat = torch.cat(
            [
                data["x_positions_diff"],
                data["x_velocity_diff"][..., None],
                hist_valid_mask[..., None],
            ],
            dim=-1,
        ) # [16, 48, 50, 4] different to forecast-mae

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D) # [768, 50, 4]
        hist_feat_key_valid = hist_key_valid_mask.view(B * N) # [768]
        


        # unidirectional mamba
        actor_feat = self.hist_embed_mlp(hist_feat[hist_feat_key_valid].contiguous()) # [421, 50, 128]
        residual = None
        for blk_mamba in self.hist_embed_mamba:
            actor_feat, residual = blk_mamba(actor_feat, residual) # [421, 50, 128], [421, 50, 128]
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        actor_feat = fused_add_norm_fn(
            self.drop_path(actor_feat),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        ) # [421, 50, 128]

        actor_feat = actor_feat[:, -1] # [421, 128]
        # actor_feat = actor_feat[:, 0] # [421, 128]
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        ) # [768, 128]
        actor_feat_tmp[hist_feat_key_valid] = actor_feat # [768, 128]
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1]) # [16, 48, 128]

        # map encoding
        lane_valid_mask = data["lane_valid_mask"] # [16, 125, 20]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2) # [16, 125, 20, 2]
        lane_normalized = torch.cat(
            [lane_normalized, lane_valid_mask[..., None]], dim=-1
        ) # [16, 125, 20, 3]
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous()) # [2000, 128]
        lane_feat = lane_feat.view(B, M, -1) # [16, 125, 128]

        # type embedding and position embedding
        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, -1], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat) # [16, 173, 128]

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()] # [16, 48, 128]
        lane_type_embed = self.lane_type_embed[data["lane_attr"][..., 0].long()] # [16, 125, 128]
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        # scene context features
        x_encoder = torch.cat([actor_feat, lane_feat], dim=1) # [16, 173, 128]
        key_valid_mask = torch.cat(
            [data["x_key_valid_mask"], data["lane_key_valid_mask"]], dim=1
        ) # [16, 173]

        x_encoder = x_encoder + pos_embed # [16, 173, 128]

        goal_context = {
            "goal_logits": None,
            "goal_candidate_xy": None,
            "goal_candidate_mask": None,
            "goal_topk_idx": None,
        }
        future_mode_tokens = None
        if self.use_goal_conditioned_tokens:
            goal_context = self._build_goal_tokens(data, x_encoder[:, 0])
            future_mode_tokens = goal_context["future_mode_tokens"]

        x_encoder, mode, ep_offsets = self.spatial_mamba(
            x_encoder, x_centers, future_mode_tokens=future_mode_tokens
        )
        x_encoder = self.norm(x_encoder) # [16, 173, 128]


        x_others = x_encoder[:, 1:N] # [16, 47, 128]
        y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), -1, 2) # ([16, 47, 60, 2]

        
        ep_embedding = torch.linspace(0, 1, steps=self.future_steps).view(1, 1, -1, 1).to(mode.device) * mode.unsqueeze(2)
        mode = x_encoder[:,:1].unsqueeze(1) + ep_embedding

        # decoder module with decoupled queries
        dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, scal, scal_new = \
        self.time_decoder(mode, x_encoder, mask=~key_valid_mask)

        ret_dict = {
            "y_hat": y_hat,  # trajectory output from mode query
            "pi": pi,  # probability output from mode query
            "scal": scal,  # output for Laplace loss from mode query

            "dense_predict": dense_predict,  # trajectory output from state query
            
            "ep_offsets": ep_offsets,

            "y_hat_others": y_hat_others,  # trajectory of other agents

            "new_y_hat": new_y_hat,  # final trajectory output
            "new_pi": new_pi,  # final probability output     
            "scal_new": scal_new,  # final output for Laplace loss
            "goal_logits": goal_context["goal_logits"],
            "goal_candidate_xy": goal_context["goal_candidate_xy"],
            "goal_candidate_mask": goal_context["goal_candidate_mask"],
            "goal_topk_idx": goal_context["goal_topk_idx"],
        }


        return ret_dict
