"""Checkpoint loading and APIs for direct matching / matcher priors."""
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

from biohub046.core.dataset import to_ds
from .data import downsample_mean, normalize
from .model import build_flow_model
from .ops import sample_volume


def load_model(checkpoint, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    model = build_flow_model(OmegaConf.to_container(cfg.model)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, cfg, device


@torch.no_grad()
def infer_pair(model, frame_a, frame_b, device=None):
    """Infer dense downsample-grid flow and descriptors from normalized arrays."""
    device = device or next(model.parameters()).device
    a = torch.as_tensor(frame_a, dtype=torch.float32, device=device)[None, None]
    b = torch.as_tensor(frame_b, dtype=torch.float32, device=device)[None, None]
    return model(a, b, return_features=True)


@torch.no_grad()
def point_priors(result, points_zyx, downsample=(1, 1, 1)):
    """Return point flow, warped positions, and descriptors in original voxels.

    ``points_zyx`` and returned coordinates are original-volume coordinates;
    model tensors live on the mean-pooled grid.
    """
    device = result["flow"].device
    ds = torch.tensor(downsample, device=device, dtype=torch.float32)
    pts = torch.as_tensor(points_zyx, device=device, dtype=torch.float32)
    grid_pts = torch.as_tensor(to_ds(pts.cpu().numpy(), downsample), device=device,
                               dtype=torch.float32)
    flow_grid = sample_volume(result["flow"], grid_pts)[0]
    descriptors = torch.nn.functional.normalize(
        sample_volume(result["features_a"], grid_pts)[0], dim=1)
    flow_orig = flow_grid * ds
    return {"flow_zyx": flow_orig.cpu().numpy(),
            "warped_zyx": (pts + flow_orig).cpu().numpy(),
            "descriptor": descriptors.cpu().numpy()}


@torch.no_grad()
def point_descriptors(result, points_zyx, frame="a", downsample=(1, 1, 1)):
    """Sample normalized descriptors for detections in frame ``a`` or ``b``."""
    if frame not in {"a", "b"}:
        raise ValueError("frame must be 'a' or 'b'")
    volume = result[f"features_{frame}"]
    grid_pts = torch.as_tensor(to_ds(points_zyx, downsample), device=volume.device,
                               dtype=torch.float32)
    descriptors = sample_volume(volume, grid_pts)[0]
    return torch.nn.functional.normalize(descriptors, dim=1).cpu().numpy()


def prepare_zarr_frame(array, t, quantiles, downsample):
    qlo, qhi = float(quantiles["0.001"]), float(quantiles["0.999"])
    return downsample_mean(normalize(array[t], qlo, qhi), downsample)
