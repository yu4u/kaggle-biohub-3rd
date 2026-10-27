"""Coordinate-safe 3-D warping and point sampling utilities.

All public coordinates and flow channels use ``(z, y, x)`` voxel order.
``warp(source, flow)`` samples source at ``p + flow(p)`` for every output p.
Thus ``warp(frame_b, forward_flow)`` reconstructs frame_a when forward_flow
maps locations in a to their corresponding locations in b.
"""
from functools import lru_cache
import torch
import torch.nn.functional as F


@lru_cache(maxsize=16)
def identity_grid(shape, device=None, dtype=None):
    z, y, x = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(z, device=device, dtype=dtype),
        torch.arange(y, device=device, dtype=dtype),
        torch.arange(x, device=device, dtype=dtype), indexing="ij")
    return torch.stack((zz, yy, xx), 0)


def _normalize_grid(grid, source_shape=None):
    """B,3,Z,Y,X zyx voxel grid -> B,Z,Y,X,3 normalized xyz grid."""
    z, y, x = source_shape or grid.shape[-3:]
    scale = grid.new_tensor([max(z - 1, 1), max(y - 1, 1), max(x - 1, 1)])
    norm = 2.0 * grid / scale.view(1, 3, 1, 1, 1) - 1.0
    return norm[:, [2, 1, 0]].permute(0, 2, 3, 4, 1)


def warp(source, flow, padding_mode="border"):
    """Backward-warp ``source`` using a zyx displacement defined on output."""
    if source.shape[0] != flow.shape[0] or source.shape[-3:] != flow.shape[-3:]:
        raise ValueError("source and flow must have matching batch and spatial shapes")
    base = identity_grid(tuple(flow.shape[-3:]), flow.device, flow.dtype)[None]
    return F.grid_sample(source, _normalize_grid(base + flow, source.shape[-3:]), mode="bilinear",
                         padding_mode=padding_mode, align_corners=True)


def sample_volume(volume, points_zyx):
    """Trilinearly sample B,C,Z,Y,X at B,N,3 (or N,3 for B=1) points."""
    if points_zyx.ndim == 2:
        points_zyx = points_zyx[None]
    b, n, _ = points_zyx.shape
    if b != volume.shape[0]:
        raise ValueError("point and volume batch sizes differ")
    shape = volume.shape[-3:]
    scale = points_zyx.new_tensor([max(v - 1, 1) for v in shape])
    grid = (2.0 * points_zyx / scale - 1.0)[..., [2, 1, 0]]
    grid = grid.reshape(b, n, 1, 1, 3)
    out = F.grid_sample(volume, grid, mode="bilinear", padding_mode="border",
                        align_corners=True)
    return out[:, :, :, 0, 0].transpose(1, 2)


def forward_backward_mask(flow_ab=None, flow_ba=None, threshold=1.5, residual=None):
    """Soft validity mask from forward/backward consistency in voxels."""
    if residual is None:
        residual = flow_ab + warp(flow_ba, flow_ab)
    error = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
    return (error <= threshold).to(residual.dtype).detach()
