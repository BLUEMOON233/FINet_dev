import math
import torch


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def global_to_local(
    x_pos: torch.Tensor,
    x_head: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_head: torch.Tensor,
) -> tuple:
    """Transform positions and headings from global to local frame.

    The local frame is centered at anchor_pos and rotated by anchor_head.

    Args:
        x_pos:       [..., 2]  positions in global frame
        x_head:      [...]     headings in global frame
        anchor_pos:  [..., 2]  anchor position (broadcastable)
        anchor_head: [...]     anchor heading  (broadcastable)

    Returns:
        local_pos:   [..., 2]
        local_head:  [...]
    """
    x_pos = x_pos - anchor_pos

    cos = anchor_head.cos()
    sin = anchor_head.sin()
    rot = torch.stack([
        torch.stack([cos, -sin], dim=-1),
        torch.stack([sin,  cos], dim=-1),
    ], dim=-2)  # [..., 2, 2]
    x_pos = torch.einsum("...i,...ij->...j", x_pos, rot)

    x_head = wrap_angle(x_head - anchor_head)
    return x_pos, x_head


def local_to_global(
    x_pos: torch.Tensor,
    x_head: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_head: torch.Tensor,
) -> tuple:
    """Transform positions and headings from local to global frame.

    Inverse of global_to_local.
    """
    cos = anchor_head.cos()
    sin = anchor_head.sin()
    rot = torch.stack([
        torch.stack([ cos, sin], dim=-1),
        torch.stack([-sin, cos], dim=-1),
    ], dim=-2)  # [..., 2, 2]
    x_pos = torch.einsum("...i,...ij->...j", x_pos, rot)

    x_pos = x_pos + anchor_pos
    x_head = wrap_angle(x_head + anchor_head)
    return x_pos, x_head
