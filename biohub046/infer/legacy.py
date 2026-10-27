"""Checkpoint loading and the detector forward helpers of the submission inference."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from biohub046 import config as C
from biohub046.core.dataset import get_downsample, to_model_input
from biohub046.core.metric import extract_peaks
from biohub046.core.model import get_model_from_cfg

CONFIG_PATH = C.PACKAGE / 'core/config.yaml'
# 009/006: the matcher's candidate ball at inference is 12 um (trained at 16); 029: trunk fp16 autocast
EMB_OVERRIDES = {'model': {'pretrained': False}, 'loss': {'embed_radius_um': 12.0},
                 'test': {'frames_per_batch': 4, 'amp': True}}
# 031: the detector operating point is pinned here, not read from the checkpoint
DET_OVERRIDES = {'model': {'pretrained': False, 'det_threshold': C.DET_THRESHOLD, 'nms_kernel': C.NMS_KERNEL},
                 'test': {'frames_per_batch': 1}}
# 006/009 operating point of the production Hungarian cell (link_embed) and the second round
MAX_LINK_UM = 5.5
DRIFT_ITERS = 2
LAMBDA = 12.0
GATE_EXT_UM = 12.0
GATE_PTH = 0.5
R2_TOPK = 5
DET_BLUR_SIGMA = C.DET_BLUR_SIGMA
TTA_VARIANTS = {'none': [(0, 0)], 'flip2': [(0, 0), (0, 1)],
                'rot4': [(k, 0) for k in range(4)],
                'tta8': [(k, f) for k in range(4) for f in (0, 1)]}


def restore_cfg(cfg, overrides=None):
    """Config with model.* restored from the checkpoint's hyper_parameters (21_test.restore_cfg);
    precedence overrides > checkpoint > yaml."""
    ckpt = torch.load(str(cfg.model.resume_path), map_location='cpu', weights_only=False)
    hp_cfg = ckpt.get('hyper_parameters', {}).get('cfg')
    if hp_cfg is None:
        return cfg
    resume = cfg.model.resume_path
    hp = OmegaConf.create(hp_cfg)
    cfg.model = OmegaConf.merge(cfg.model, hp.model)
    cfg.data.downsample = hp.data.get('downsample', [1, 1, 1])
    cfg.data.local_contrast = hp.data.get('local_contrast', False)
    cfg.model.temporal = hp.model.get('temporal', 'none')
    if hp.model.get('matcher_drift_frame') is None:
        cfg.model.matcher_drift_frame = False
    if overrides is not None:
        cfg = OmegaConf.merge(cfg, overrides)
    cfg.model.resume_path = resume
    assert str(cfg.model.get('arch', 'unet25d')) != 'unet25d_coarse'
    return cfg


def build(ckpt, overrides, device):
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.model.resume_path = str(ckpt)
    cfg = restore_cfg(cfg, OmegaConf.create(overrides))
    return cfg, get_model_from_cfg(cfg, cfg.model.resume_path).eval().to(device)


def load_detector(ckpt, device):
    """A full-resolution single-frame detector checkpoint -> (cfg, model); EMA weights win (core.model)."""
    cfg, model = build(ckpt, DET_OVERRIDES, device)
    if not np.array_equal(get_downsample(cfg), np.ones(3, int)) or cfg.model.get('temporal', 'none') != 'none' \
            or cfg.data.get('local_contrast', False):
        raise ValueError(f'{ckpt}: not a full-resolution single-frame detector')
    return cfg, model


def load_matcher(ckpt, device):
    cfg, model = build(ckpt, EMB_OVERRIDES, device)
    if str(cfg.model.matcher) == 'none' or not cfg.model.matcher_drift_frame:
        raise ValueError(f'{ckpt}: not a drift-frame matcher checkpoint')
    return cfg, model


def check_device(device):
    if device != 'cuda':
        raise SystemExit('no CUDA device: the pipeline needs a GPU')
    major, minor = torch.cuda.get_device_capability()
    print(f'[biohub046] {torch.cuda.get_device_name()} (sm_{major}{minor}), torch {torch.__version__}', flush=True)


@torch.no_grad()
def forward_tta(model, xt, mode, amp=False):
    """Average heatmaps over YX-symmetry TTA (21_test.forward_tta). xt: (B,1,Z,Y,X).
    none | flip2 (id + x-flip) | flip4 | rot4. The forward runs under fp16 autocast when `amp`."""
    autocast = torch.autocast(xt.device.type, dtype=torch.float16, enabled=amp)

    def model_hm(v):
        with autocast:
            o = model(v)
        return (o[0] if isinstance(o, tuple) else o).float()

    out = model_hm(xt)
    n = 1
    if mode in ('flip2', 'flip4'):
        for dims in ([(4,)] if mode == 'flip2' else [(3,), (4,), (3, 4)]):
            out = out + model_hm(xt.flip(dims)).flip(dims)
            n += 1
    elif mode == 'rot4':
        for k in (1, 2, 3):
            out = out + torch.rot90(model_hm(torch.rot90(xt, k, (3, 4))), -k, (3, 4))
            n += 1
    elif mode != 'none':
        raise ValueError(f'unknown tta: {mode}')
    return out / n


def blur3d(v, sigma):
    """Separable isotropic Gaussian on a (Z,Y,X) fp32 heatmap; sigma in voxels (016.blur3d)."""
    if sigma <= 0:
        return v
    r = max(1, int(round(3 * sigma)))
    x = torch.arange(-r, r + 1, device=v.device, dtype=v.dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    out = v[None, None]
    for d in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + d] = -1
        pad = [0, 0, 0]
        pad[d] = r
        out = F.conv3d(out, k.view(shape), padding=tuple(pad))
    return out[0, 0]


@torch.no_grad()
def detect_tta(models, store, sid, device, cfg, tta=C.DET_TTA):
    """031.detect_tta: per member forward_tta (fp16) -> nan_to_num -> weighted sum -> blur 0.7 -> peaks.
    models: [(model, weight)]; returns (peaks per frame (N,3) int32, blurred heatmap value at each peak)."""
    peaks, confidence = [], []
    batch = int(cfg.test.frames_per_batch)
    for t0 in range(0, store.shape[sid][0], batch):
        frames = [store.frame(sid, t) for t in range(t0, min(t0 + batch, store.shape[sid][0]))]
        x = torch.stack([to_model_input(frame, False) for frame in frames]).to(device)
        heatmap = None
        for model, weight in models:
            value = forward_tta(model, x, tta, True)[:, 0]
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)   # 026: fp16 NaN defence
            value = value * weight
            heatmap = value if heatmap is None else heatmap + value
        for value in heatmap:
            value = blur3d(value, DET_BLUR_SIGMA)
            pk = extract_peaks(value, cfg.model.det_threshold, cfg.model.nms_kernel)
            peaks.append(pk.cpu().numpy().astype(np.int32))
            confidence.append(value[pk[:, 0], pk[:, 1], pk[:, 2]].cpu().numpy())
    return peaks, confidence
