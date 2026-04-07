import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _safe_masked_softmax(
    logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1
) -> torch.Tensor:
    if mask is None:
        return torch.softmax(logits, dim=dim)

    mask = mask.bool()
    masked_logits = logits.masked_fill(~mask, -1e4)
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights * mask.float()
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    return weights / denom


def _pad_to_size_dim1(x: torch.Tensor, target: int) -> torch.Tensor:
    cur = x.size(1)
    if cur == target:
        return x
    if cur > target:
        return x[:, :target]
    if cur == 0:
        out_shape = list(x.shape)
        out_shape[1] = target
        return x.new_zeros(out_shape)
    pad = x[:, -1:].expand(-1, target - cur, *x.shape[2:])
    return torch.cat([x, pad], dim=1)


def _batched_gather_2d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # x: [B, M, C], idx: [B, K] -> out: [B, K, C]
    expand_shape = [-1, -1, x.size(-1)]
    return torch.gather(x, dim=1, index=idx.unsqueeze(-1).expand(*expand_shape))


def _batched_gather_3d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # x: [B, N, C], idx: [B, G, K] -> out: [B, G, K, C]
    x_expand = x.unsqueeze(1).expand(-1, idx.size(1), -1, -1)
    return torch.gather(
        x_expand,
        dim=2,
        index=idx.unsqueeze(-1).expand(-1, -1, -1, x.size(-1)),
    )


def _extract_angle(angle_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if angle_tensor is None:
        return None
    if angle_tensor.size(-1) == 2:
        return torch.atan2(angle_tensor[..., 1], angle_tensor[..., 0])
    return angle_tensor


class GoalProposalHead(nn.Module):
    """Generate coarse goal candidates from focal/lane context."""

    def __init__(self, embed_dim: int = 128, top_ng: int = 3) -> None:
        super().__init__()
        self.top_ng = top_ng
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.geo_mlp = nn.Sequential(
            nn.Linear(9, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.goal_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        focal_feat: torch.Tensor,  # [B, D]
        lane_feat: torch.Tensor,  # [B, M, D]
        lane_centers: torch.Tensor,  # [B, M, 2]
        focal_center: torch.Tensor,  # [B, 2]
        lane_angles: Optional[torch.Tensor] = None,  # [B, M]
        focal_angle: Optional[torch.Tensor] = None,  # [B]
        lane_attr: Optional[torch.Tensor] = None,  # [B, M, C_attr]
        lane_valid_mask: Optional[torch.Tensor] = None,  # [B, M]
        bias_center: Optional[torch.Tensor] = None,  # [B, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_lanes, dim = lane_feat.shape
        if num_lanes == 0:
            # Safe fallback when no lane tokens are available in current scene.
            # Keep finite outputs so downstream pipeline can continue training/inference.
            goal_scores = focal_feat.new_zeros(bsz, self.top_ng)
            base_xy = bias_center if bias_center is not None else focal_center
            goal_xy = base_xy.unsqueeze(1).expand(-1, self.top_ng, -1).contiguous()
            goal_lane_ctx = focal_feat.new_zeros(bsz, self.top_ng, dim)
            focal_expand = focal_feat.unsqueeze(1).expand(-1, self.top_ng, -1)
            goal_feat = self.goal_proj(torch.cat([goal_lane_ctx, focal_expand], dim=-1))
            return goal_feat, goal_scores, goal_xy

        top_k = max(1, min(self.top_ng, num_lanes))

        if lane_valid_mask is None:
            lane_valid_mask = lane_centers.abs().sum(dim=-1) > 0

        rel = lane_centers - focal_center.unsqueeze(1)  # [B, M, 2]
        dist = torch.norm(rel, dim=-1, keepdim=True)  # [B, M, 1]

        if lane_angles is not None and focal_angle is not None:
            angle_diff = lane_angles - focal_angle.unsqueeze(1)
            cos_diff = torch.cos(angle_diff).unsqueeze(-1)
            sin_diff = torch.sin(angle_diff).unsqueeze(-1)
        else:
            cos_diff = lane_centers.new_zeros(bsz, num_lanes, 1)
            sin_diff = lane_centers.new_zeros(bsz, num_lanes, 1)

        if lane_attr is not None and lane_attr.size(-1) >= 3:
            lane_type = lane_attr[..., 0:1] / 10.0
            lane_width = lane_attr[..., 1:2] / 10.0
            lane_intersection = lane_attr[..., 2:3]
        else:
            lane_type = lane_centers.new_zeros(bsz, num_lanes, 1)
            lane_width = lane_centers.new_zeros(bsz, num_lanes, 1)
            lane_intersection = lane_centers.new_zeros(bsz, num_lanes, 1)

        if bias_center is not None:
            bias_dist = torch.norm(
                lane_centers - bias_center.unsqueeze(1), dim=-1, keepdim=True
            )
        else:
            bias_dist = lane_centers.new_zeros(bsz, num_lanes, 1)

        geo_feat = torch.cat(
            [rel, dist, cos_diff, sin_diff, lane_type, lane_width, lane_intersection, bias_dist],
            dim=-1,
        )  # [B, M, 9]

        lane_ctx = lane_feat + self.geo_mlp(geo_feat)  # [B, M, D]
        query = self.query_proj(focal_feat).unsqueeze(1)  # [B, 1, D]

        score_dot = (query * lane_ctx).sum(dim=-1) / math.sqrt(dim)  # [B, M]
        score_mlp = self.score_mlp(
            torch.cat([lane_ctx, query.expand_as(lane_ctx)], dim=-1)
        ).squeeze(-1)  # [B, M]
        logits = score_dot + score_mlp - 0.10 * bias_dist.squeeze(-1)

        logits = logits.masked_fill(~lane_valid_mask.bool(), -1e4)
        goal_scores, goal_idx = logits.topk(k=top_k, dim=1, largest=True, sorted=True)

        goal_scores = _pad_to_size_dim1(goal_scores, self.top_ng)  # [B, top_ng]
        goal_idx = _pad_to_size_dim1(goal_idx, self.top_ng)  # [B, top_ng]

        goal_lane_ctx = _batched_gather_2d(lane_ctx, goal_idx)  # [B, top_ng, D]
        goal_xy = _batched_gather_2d(lane_centers, goal_idx)  # [B, top_ng, 2]
        goal_feat = self.goal_proj(
            torch.cat(
                [
                    goal_lane_ctx,
                    focal_feat.unsqueeze(1).expand(-1, goal_lane_ctx.size(1), -1),
                ],
                dim=-1,
            )
        )  # [B, top_ng, D]

        return goal_feat, goal_scores, goal_xy


class BranchPooler(nn.Module):
    """Pool lane context around each goal as branch embedding."""

    def __init__(self, embed_dim: int = 128) -> None:
        super().__init__()
        self.goal_query = nn.Linear(embed_dim, embed_dim)
        self.geo_mlp = nn.Sequential(
            nn.Linear(8, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.branch_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        goal_feat: torch.Tensor,  # [B, G, D]
        goal_xy: torch.Tensor,  # [B, G, 2]
        lane_feat: torch.Tensor,  # [B, M, D]
        lane_centers: torch.Tensor,  # [B, M, 2]
        focal_center: torch.Tensor,  # [B, 2]
        lane_angles: Optional[torch.Tensor] = None,  # [B, M]
        lane_attr: Optional[torch.Tensor] = None,  # [B, M, C_attr]
        lane_valid_mask: Optional[torch.Tensor] = None,  # [B, M]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_goals, dim = goal_feat.shape
        num_lanes = lane_feat.size(1)
        if num_lanes == 0:
            # Safe fallback for lane-empty scenes.
            branch_ctx = goal_feat.new_zeros(bsz, num_goals, dim)
            branch_feat = self.branch_proj(torch.cat([goal_feat, branch_ctx], dim=-1))
            branch_scores = goal_feat.new_zeros(bsz, num_goals)
            return branch_feat, branch_scores

        if lane_valid_mask is None:
            lane_valid_mask = lane_centers.abs().sum(dim=-1) > 0

        rel_goal = lane_centers.unsqueeze(1) - goal_xy.unsqueeze(2)  # [B, G, M, 2]
        dist_goal = torch.norm(rel_goal, dim=-1, keepdim=True)  # [B, G, M, 1]

        if lane_angles is not None:
            goal_vec = goal_xy - focal_center.unsqueeze(1)  # [B, G, 2]
            goal_dir = torch.atan2(goal_vec[..., 1], goal_vec[..., 0])  # [B, G]
            angle_diff = lane_angles.unsqueeze(1) - goal_dir.unsqueeze(-1)  # [B, G, M]
            cos_diff = torch.cos(angle_diff).unsqueeze(-1)
            sin_diff = torch.sin(angle_diff).unsqueeze(-1)
        else:
            cos_diff = lane_centers.new_zeros(bsz, num_goals, num_lanes, 1)
            sin_diff = lane_centers.new_zeros(bsz, num_goals, num_lanes, 1)

        if lane_attr is not None and lane_attr.size(-1) >= 3:
            lane_type = lane_attr[..., 0:1].unsqueeze(1).expand(-1, num_goals, -1, -1) / 10.0
            lane_width = lane_attr[..., 1:2].unsqueeze(1).expand(-1, num_goals, -1, -1) / 10.0
            lane_intersection = lane_attr[..., 2:3].unsqueeze(1).expand(-1, num_goals, -1, -1)
        else:
            lane_type = lane_centers.new_zeros(bsz, num_goals, num_lanes, 1)
            lane_width = lane_centers.new_zeros(bsz, num_goals, num_lanes, 1)
            lane_intersection = lane_centers.new_zeros(bsz, num_goals, num_lanes, 1)

        geo_feat = torch.cat(
            [rel_goal, dist_goal, cos_diff, sin_diff, lane_type, lane_width, lane_intersection],
            dim=-1,
        )  # [B, G, M, 8]
        lane_ctx = lane_feat.unsqueeze(1) + self.geo_mlp(geo_feat)  # [B, G, M, D]

        query = self.goal_query(goal_feat).unsqueeze(2)  # [B, G, 1, D]
        score_dot = (query * lane_ctx).sum(dim=-1) / math.sqrt(dim)  # [B, G, M]
        score_mlp = self.score_mlp(
            torch.cat([lane_ctx, query.expand_as(lane_ctx)], dim=-1)
        ).squeeze(-1)  # [B, G, M]
        branch_logits = score_dot + score_mlp

        lane_mask = lane_valid_mask.bool().unsqueeze(1).expand(-1, num_goals, -1)  # [B, G, M]
        branch_weights = _safe_masked_softmax(branch_logits, lane_mask, dim=-1)  # [B, G, M]

        branch_ctx = (branch_weights.unsqueeze(-1) * lane_ctx).sum(dim=2)  # [B, G, D]
        branch_feat = self.branch_proj(torch.cat([goal_feat, branch_ctx], dim=-1))  # [B, G, D]
        branch_scores = branch_weights.max(dim=-1).values  # [B, G]

        return branch_feat, branch_scores


class SocialResponseHead(nn.Module):
    """Generate per-goal social response embeddings (2 branches)."""

    def __init__(self, embed_dim: int = 128, topk_neighbors: int = 8, num_social: int = 2) -> None:
        super().__init__()
        self.topk_neighbors = topk_neighbors
        self.num_social = num_social

        self.speed_proj = nn.Linear(embed_dim, 1)
        self.pair_mlp = nn.Sequential(
            nn.Linear(12, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.query_proj = nn.Linear(embed_dim * 3, embed_dim)
        self.social_fuse = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.response_tokens = nn.Parameter(torch.randn(1, 1, num_social, embed_dim) * 0.02)
        self.response_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.social_score = nn.Linear(embed_dim, 1)

    def forward(
        self,
        focal_feat: torch.Tensor,  # [B, D]
        agent_feat: torch.Tensor,  # [B, N, D]
        agent_centers: torch.Tensor,  # [B, N, 2]
        goal_xy: torch.Tensor,  # [B, G, 2]
        goal_feat: torch.Tensor,  # [B, G, D]
        branch_feat: torch.Tensor,  # [B, G, D]
        x_angles: Optional[torch.Tensor] = None,  # [B, N]
        agent_valid_mask: Optional[torch.Tensor] = None,  # [B, N]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_goals, dim = goal_feat.shape
        num_agents = agent_feat.size(1)
        topk = max(1, min(self.topk_neighbors, num_agents))

        if agent_valid_mask is None:
            agent_valid_mask = agent_centers.abs().sum(dim=-1) > 0

        neighbor_mask = agent_valid_mask.clone().bool()
        if neighbor_mask.size(1) > 0:
            neighbor_mask[:, 0] = False  # exclude focal itself

        dist_to_goal = torch.norm(
            agent_centers.unsqueeze(1) - goal_xy.unsqueeze(2), dim=-1
        )  # [B, G, N]
        dist_rank = dist_to_goal + (~neighbor_mask.unsqueeze(1)).float() * 1e4
        _, top_idx = dist_rank.topk(k=topk, dim=-1, largest=False)  # [B, G, K]

        neigh_feat = _batched_gather_3d(agent_feat, top_idx)  # [B, G, K, D]
        neigh_ctr = _batched_gather_3d(agent_centers, top_idx)  # [B, G, K, 2]

        valid_expand = neighbor_mask.unsqueeze(1).expand(-1, num_goals, -1)
        neigh_valid = torch.gather(valid_expand, dim=2, index=top_idx)  # [B, G, K]

        focal_center = agent_centers[:, 0]  # [B, 2]
        rel_to_focal = neigh_ctr - focal_center.unsqueeze(1).unsqueeze(1)  # [B, G, K, 2]
        rel_to_goal = goal_xy.unsqueeze(2) - neigh_ctr  # [B, G, K, 2]

        dist_focal = torch.norm(rel_to_focal, dim=-1, keepdim=True)  # [B, G, K, 1]
        dist_goal = torch.norm(rel_to_goal, dim=-1, keepdim=True)  # [B, G, K, 1]
        same_region = ((dist_focal < 8.0) & (dist_goal < 8.0)).float()  # [B, G, K, 1]

        focal_speed = self.speed_proj(focal_feat).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 1]
        neigh_speed = self.speed_proj(neigh_feat)  # [B, G, K, 1]
        speed_diff = neigh_speed - focal_speed  # [B, G, K, 1]

        x_angles = _extract_angle(x_angles)
        if x_angles is not None:
            neigh_ang = _batched_gather_3d(x_angles.unsqueeze(-1), top_idx).squeeze(-1)
            focal_ang = x_angles[:, 0].unsqueeze(1).unsqueeze(1)  # [B, 1, 1]
            goal_vec = goal_xy - focal_center.unsqueeze(1)  # [B, G, 2]
            goal_dir = torch.atan2(goal_vec[..., 1], goal_vec[..., 0]).unsqueeze(-1)  # [B, G, 1]

            diff_focal = neigh_ang - focal_ang
            diff_goal = neigh_ang - goal_dir
            cos_focal = torch.cos(diff_focal).unsqueeze(-1)
            sin_focal = torch.sin(diff_focal).unsqueeze(-1)
            cos_goal = torch.cos(diff_goal).unsqueeze(-1)
            sin_goal = torch.sin(diff_goal).unsqueeze(-1)
        else:
            cos_focal = agent_centers.new_zeros(bsz, num_goals, topk, 1)
            sin_focal = agent_centers.new_zeros(bsz, num_goals, topk, 1)
            cos_goal = agent_centers.new_zeros(bsz, num_goals, topk, 1)
            sin_goal = agent_centers.new_zeros(bsz, num_goals, topk, 1)

        pair_feat = torch.cat(
            [
                rel_to_focal,
                rel_to_goal,
                dist_focal,
                dist_goal,
                cos_focal,
                sin_focal,
                cos_goal,
                sin_goal,
                same_region,
                speed_diff,
            ],
            dim=-1,
        )  # [B, G, K, 12]
        neigh_ctx = neigh_feat + self.pair_mlp(pair_feat)  # [B, G, K, D]

        q_input = torch.cat(
            [
                focal_feat.unsqueeze(1).expand(-1, num_goals, -1),
                goal_feat,
                branch_feat,
            ],
            dim=-1,
        )  # [B, G, 3D]
        query = self.query_proj(q_input).unsqueeze(2)  # [B, G, 1, D]
        attn_logits = (query * neigh_ctx).sum(dim=-1) / math.sqrt(dim)  # [B, G, K]
        attn_weights = _safe_masked_softmax(attn_logits, neigh_valid, dim=-1)  # [B, G, K]

        social_ctx = (attn_weights.unsqueeze(-1) * neigh_ctx).sum(dim=2)  # [B, G, D]
        social_base = self.social_fuse(
            torch.cat(
                [
                    focal_feat.unsqueeze(1).expand(-1, num_goals, -1),
                    goal_feat,
                    branch_feat,
                    social_ctx,
                ],
                dim=-1,
            )
        )  # [B, G, D]

        response_tokens = self.response_tokens.expand(bsz, num_goals, -1, -1)  # [B, G, 2, D]
        social_embed = self.response_proj(
            torch.cat(
                [
                    social_base.unsqueeze(2).expand(-1, -1, self.num_social, -1),
                    response_tokens,
                ],
                dim=-1,
            )
        )  # [B, G, 2, D]
        social_logits = self.social_score(social_embed).squeeze(-1)  # [B, G, 2]

        return social_embed, social_logits


class StructuredFutureModeGenerator(nn.Module):
    """Generate K mode tokens with goal/branch/social structure.

    q_m = proj([h_f, g_m, b_m, s_m])
    """

    def __init__(
        self,
        embed_dim: int = 128,
        top_ng: int = 3,
        num_social: int = 2,
        num_modes: int = 6,
        social_topk: int = 8,
    ) -> None:
        super().__init__()
        self.top_ng = top_ng
        self.num_social = num_social
        self.num_modes = num_modes

        self.goal_head = GoalProposalHead(embed_dim=embed_dim, top_ng=top_ng)
        self.branch_pooler = BranchPooler(embed_dim=embed_dim)
        self.social_head = SocialResponseHead(
            embed_dim=embed_dim,
            topk_neighbors=social_topk,
            num_social=num_social,
        )
        self.mode_proj = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mode_norm = nn.LayerNorm(embed_dim)

        # fallback monitor buffers (running statistics)
        self._fallback_keys = [
            "lane_valid_mask_missing",
            "agent_valid_mask_missing",
            "lane_attr_missing",
            "lane_angles_missing",
            "x_angles_missing",
            "goal_padding_used",
            "social_topk_truncated",
            "no_valid_neighbors",
        ]
        self.register_buffer("fallback_total_samples", torch.zeros(1), persistent=False)
        for key in self._fallback_keys:
            self.register_buffer(f"fallback_count_{key}", torch.zeros(1), persistent=False)

    def _update_fallback_stats(self, batch_size: int, batch_flags: dict) -> dict:
        self.fallback_total_samples += float(batch_size)
        total = self.fallback_total_samples.clamp_min(1.0)

        stats = {}
        for key in self._fallback_keys:
            flag = batch_flags.get(key, False)
            if isinstance(flag, torch.Tensor):
                if flag.numel() == 1:
                    count = flag.float().view(1) * float(batch_size)
                else:
                    count = flag.float().reshape(-1).sum().view(1)
            else:
                count = total.new_tensor([float(bool(flag)) * float(batch_size)])

            counter = getattr(self, f"fallback_count_{key}")
            counter += count

            batch_prob = count / float(max(batch_size, 1))
            running_prob = counter / total
            stats[f"batch_{key}_prob"] = batch_prob.squeeze(0)
            stats[f"running_{key}_prob"] = running_prob.squeeze(0)

        return stats

    def forward(
        self,
        focal_feat: torch.Tensor,  # [B, D]
        agent_feat: torch.Tensor,  # [B, N, D]
        lane_feat: torch.Tensor,  # [B, M, D]
        x_centers: torch.Tensor,  # [B, N, 2]
        lane_centers: torch.Tensor,  # [B, M, 2]
        x_angles: Optional[torch.Tensor] = None,  # [B, N]
        lane_angles: Optional[torch.Tensor] = None,  # [B, M]
        lane_attr: Optional[torch.Tensor] = None,  # [B, M, C_attr]
        agent_valid_mask: Optional[torch.Tensor] = None,  # [B, N]
        lane_valid_mask: Optional[torch.Tensor] = None,  # [B, M]
        goal_bias: Optional[torch.Tensor] = None,  # [B, 2], optional endpoint offset bias
    ) -> Tuple[torch.Tensor, dict]:
        bsz = focal_feat.size(0)

        lane_valid_missing = lane_valid_mask is None
        agent_valid_missing = agent_valid_mask is None
        lane_attr_missing = (lane_attr is None) or (lane_attr.size(-1) < 3 if lane_attr is not None else True)
        lane_angles_missing = lane_angles is None
        x_angles_missing = x_angles is None
        goal_padding_used = lane_feat.size(1) < self.top_ng
        social_topk_truncated = agent_feat.size(1) < self.social_head.topk_neighbors

        if agent_valid_mask is None:
            agent_valid_mask_for_monitor = x_centers.abs().sum(dim=-1) > 0
        else:
            agent_valid_mask_for_monitor = agent_valid_mask.bool()
        if agent_valid_mask_for_monitor.size(1) > 0:
            neighbor_valid = agent_valid_mask_for_monitor.clone()
            neighbor_valid[:, 0] = False
        else:
            neighbor_valid = agent_valid_mask_for_monitor
        no_valid_neighbors = neighbor_valid.sum(dim=-1) == 0

        focal_center = x_centers[:, 0]  # [B, 2]
        x_angles = _extract_angle(x_angles)
        focal_angle = x_angles[:, 0] if x_angles is not None else None
        bias_center = focal_center + goal_bias if goal_bias is not None else None

        # goal_feat: [B, G, D], goal_scores: [B, G], goal_xy: [B, G, 2]
        goal_feat, goal_scores, goal_xy = self.goal_head(
            focal_feat=focal_feat,
            lane_feat=lane_feat,
            lane_centers=lane_centers,
            focal_center=focal_center,
            lane_angles=lane_angles,
            focal_angle=focal_angle,
            lane_attr=lane_attr,
            lane_valid_mask=lane_valid_mask,
            bias_center=bias_center,
        )

        # branch_feat: [B, G, D], branch_scores: [B, G]
        branch_feat, branch_scores = self.branch_pooler(
            goal_feat=goal_feat,
            goal_xy=goal_xy,
            lane_feat=lane_feat,
            lane_centers=lane_centers,
            focal_center=focal_center,
            lane_angles=lane_angles,
            lane_attr=lane_attr,
            lane_valid_mask=lane_valid_mask,
        )

        # social_embed: [B, G, 2, D], social_logits: [B, G, 2]
        social_embed, social_logits = self.social_head(
            focal_feat=focal_feat,
            agent_feat=agent_feat,
            agent_centers=x_centers,
            goal_xy=goal_xy,
            goal_feat=goal_feat,
            branch_feat=branch_feat,
            x_angles=x_angles,
            agent_valid_mask=agent_valid_mask,
        )

        num_goals = goal_feat.size(1)
        num_social = social_embed.size(2)
        focal_expand = focal_feat.unsqueeze(1).unsqueeze(2).expand(-1, num_goals, num_social, -1)
        goal_expand = goal_feat.unsqueeze(2).expand(-1, -1, num_social, -1)
        branch_expand = branch_feat.unsqueeze(2).expand(-1, -1, num_social, -1)

        mode_tokens = self.mode_proj(
            torch.cat([focal_expand, goal_expand, branch_expand, social_embed], dim=-1)
        )  # [B, G, 2, D]
        mode_tokens = self.mode_norm(mode_tokens).reshape(mode_tokens.size(0), -1, mode_tokens.size(-1))
        mode_tokens = _pad_to_size_dim1(mode_tokens, self.num_modes)  # [B, 6, D]

        fallback_stats = self._update_fallback_stats(
            batch_size=bsz,
            batch_flags={
                "lane_valid_mask_missing": lane_valid_missing,
                "agent_valid_mask_missing": agent_valid_missing,
                "lane_attr_missing": lane_attr_missing,
                "lane_angles_missing": lane_angles_missing,
                "x_angles_missing": x_angles_missing,
                "goal_padding_used": goal_padding_used,
                "social_topk_truncated": social_topk_truncated,
                "no_valid_neighbors": no_valid_neighbors,
            },
        )

        aux_dict = {
            "goal_scores": goal_scores,  # [B, G]
            "goal_xy": goal_xy,  # [B, G, 2]
            "branch_scores": branch_scores,  # [B, G]
            "social_logits": social_logits,  # [B, G, 2]
            "fallback_stats": fallback_stats,
        }
        return mode_tokens, aux_dict
