"""Zarr volume access, GT graphs, and train/val datasets for heatmap detection.

Train item: a random annotated frame; a random GT node picks the z crop
(node z + jitter, exact final depth) BEFORE the albumentations pipeline; the
full-YX volume goes through the pipeline (gain/flips/rotation/scaling), and
the final YX crop is taken AFTER it, centered on the chosen node's
TRANSFORMED position + jitter (same clip rule as the classic GT-centered
scheme, but in output coordinates). The anisotropic-Gaussian heatmap target
is rendered afterwards at the transformed annotated-node positions; under
rotation, affine border fill is excluded from the loss via a transformed
validity mask.

Sparse labels: only ~2.8% of true cells are annotated, so the crop
background is full of unlabeled true cells. The loss is therefore MASKED.
Two mask modes (loss.mask_mode):
  "carved"  (default): supervise the WHOLE crop except ellipsoids of
      carve_radius_um around unmatched low-threshold DoG detections
      (= suspected unlabeled cells, precomputed).
      Measured on train: GT has another
      cell within 6 µm ~5% of the time, while 78-88% of the volume is >6 µm
      from every detection -> carving supervises that safe background
      instead of throwing it away.
  "support": only the ~3-sigma Gaussian support of annotated nodes
      (smallest, safest, no det cache needed).

Val item: one full (64,256,256) frame + all annotated nodes at that t, used for
heatmap val_loss and detection recall/precision/F1 (Hungarian, <=7 µm).
"""
import csv
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import zarr
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset

from .metric import COARSE_C, COARSE_STRIDE as _COARSE_STRIDE

SCALE = np.array([1.625, 0.40625, 0.40625])  # (z, y, x) µm/voxel
MASK_THRESH = float(np.exp(-0.5 * 3.0**2))   # Gaussian value at 3 sigma -> loss mask support
RING_HI = float(np.exp(-0.5))                # 1 sigma: upper edge of the mid ring (dl_176)


def get_downsample(cfg):
    """data.downsample as an int array (z,y,x); [1,1,1] when absent."""
    return np.array(list(cfg.data.get("downsample", [1, 1, 1])), int)


def to_ds(coords, ds):
    """Original-voxel zyx -> downsampled-grid coords (block centers).
    Block b of factor d covers original voxels [d*b, d*b+d-1], center d*b+(d-1)/2."""
    return (np.asarray(coords, np.float64) - (np.asarray(ds) - 1) / 2) / np.asarray(ds)


def to_orig(coords, ds):
    """Downsampled-grid coords -> original-voxel zyx (block centers)."""
    return np.asarray(coords, np.float64) * np.asarray(ds) + (np.asarray(ds) - 1) / 2


class FlowCache:
    """Per-node flow from the dense flow cache:
    <sid>.npy = (T-1, 3, Z, Y, X) float16 on the FLOW model's own ds grid, in
    grid-voxel units. sample_um() reproduces the point flow cache's
    dense resampling exactly (same to_ds + sample_volume + unit chain),
    so values match the production point cache bit-for-bit up to fp32 rounding.
    The flow grid is FIXED at [1,4,4] (flow_16x's native grid) and independent
    of data.downsample — the caller passes original-voxel coords."""

    FLOW_DS = np.array([1, 4, 4], np.float64)

    def __init__(self, root):
        self.root = Path(root)
        self._mm = {}   # sid -> np.memmap (per worker, lazy)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = {}   # memmaps are re-opened per worker
        return state

    def sample_um(self, sid, t, pts_orig):
        """Flow at boundary t (frame t -> t+1) for (N,3) original-voxel zyx
        points; returns (N,3) float32 displacement in MICRONS."""
        from biohub046.ssl_flow.ops import sample_volume
        pts_orig = np.asarray(pts_orig, np.float64).reshape(-1, 3)
        if len(pts_orig) == 0:
            return np.zeros((0, 3), np.float32)
        mm = self._mm.get(sid)
        if mm is None:
            mm = np.load(self.root / f"{sid}.npy", mmap_mode="r")
            self._mm[sid] = mm
        t = min(int(t), mm.shape[0] - 1)
        vol = torch.from_numpy(np.asarray(mm[t], np.float32))[None]
        gp = torch.as_tensor(to_ds(pts_orig, self.FLOW_DS), dtype=torch.float32)
        flow = sample_volume(vol, gp)[0].numpy() * self.FLOW_DS * SCALE
        return flow.astype(np.float32)


def local_background(x, ds=8, w=3, pct=20):
    """Low-frequency floor under each voxel of a normalized (Z,Y,X) frame.

    dl_155: the nuclei the detector misses are not dim in absolute terms, they
    sit on a bright local floor (bg_local 0.433 vs 0.276 for TPs at the same
    peak intensity), and the frame-level quantile normalization gives the CNN
    no way to see that. This is the floor it measured -- the 20th percentile of
    a +-(1 z, 12 yx) voxel box -- approximated on a YX-mean-pooled grid so it
    costs 3.2 ms per (48,192,192) crop instead of 600 ms (corr 0.984
    against the exact per-voxel filter). ds=8 with w=3 puts one coarse
    voxel of reach on top of the 8x8 pooling window; INTER_LINEAR blends
    two coarse voxels, so the real YX support runs to +-16-20 fine voxels
    rather than the +-12 dl_155 measured -- same order, slightly wider.
    """
    Z, Y, X = x.shape
    cy, cx = max(1, Y // ds), max(1, X // ds)
    c = np.stack([cv2.resize(sl, (cx, cy), interpolation=cv2.INTER_AREA) for sl in x])
    r = w // 2
    win = sliding_window_view(np.pad(c, r, mode="edge"), (w, w, w))
    k = int(round(pct / 100 * (w ** 3 - 1)))
    b = np.partition(win.reshape(c.shape + (w ** 3,)), k, axis=-1)[..., k]
    return np.stack([cv2.resize(sl, (X, Y), interpolation=cv2.INTER_LINEAR) for sl in b])


def local_floor_aug(x, s):
    """dl_165: rescale the local floor and the signal above it with two
    INDEPENDENT smooth random gain fields.

    dl_155: what separates the detector's misses from its hits at equal peak
    intensity is peak - floor contrast (bg_local 0.433 vs 0.276), and the data
    only ever shows the natural correlation between the two (bright cells sit
    near bright floors), so the CNN never gets pressure to subtract the floor.
    Decompose the frame with the same calibrated +-4.9 um p20 floor estimate,
    then draw one coarse 4x4 log-uniform-around-1 gain field per component
    (YX-smooth ~64-voxel correlation length, shared across z) so every
    neighborhood sees its own floor/signal combination -- all four quadrants
    of (peak, floor) instead of the natural diagonal. Floor amplitude 1.5*s
    reaches past the miss regime (TP median bg 0.276 x 2.5 > FN median 0.433);
    signal amplitude 0.5*s stays moderate so the unchanged Gaussian target
    never claims an invisible cell (the contradictory-teacher failure dl_153
    rejected). Must run on the intact pre-pipeline plane: local_background
    would pull affine zero-fill into the floor estimate (see the
    local_contrast assert in CellTrainDataset)."""
    bg = local_background(x)
    g = []
    for a in (1.5 * s, 0.5 * s):
        c = np.random.uniform(-1.0, 1.0, (4, 4)).astype(np.float32)
        f = cv2.resize(c, (x.shape[2], x.shape[1]), interpolation=cv2.INTER_CUBIC)
        g.append(np.exp(f * np.float32(np.log1p(a)))[None])
    return g[0] * bg + g[1] * (x - bg)


def dense_floor_aug(x, s):
    """dl_190: the external confocal tiles have a ~zero background, the
    competition frames sit on a glow (floor 0.16-0.26 after normalization;
    the cells the detector misses sit
    on an even brighter local floor, dl_155). Add a smooth random floor
    a * field, a ~ U[0.05, 0.35] * s, field = 1 + 0.5 * coarse 4x4 noise
    (same shape as IntensityAug's illumination term), so "background = 0"
    does not become the model's prior for nuclei-free regions."""
    rng = np.random
    a = np.float32(rng.uniform(0.05, 0.35) * s)
    coarse = rng.normal(0.0, 1.0, size=(4, 4)).astype(np.float32)
    field = cv2.resize(coarse, (x.shape[2], x.shape[1]), interpolation=cv2.INTER_CUBIC)
    field = np.clip(1.0 + 0.5 * field, 0.3, None)
    return x + a * field[None]


def to_model_input(x, local_contrast):
    """(Z,Y,X) normalized frame -> (C,Z,Y,X) tensor the detector consumes."""
    t = torch.from_numpy(x)[None]
    if local_contrast:
        t = torch.cat([t, torch.from_numpy(x - local_background(x))[None]], 0)
    return t


class RandomGain(A.ImageOnlyTransform):
    """Multiplicative brightness gain s ~ logU[1/(1+a), 1+a] on the normalized
    crop (intensity non-stationarity). Custom transform: albumentations'
    own brightness ops clip float input to [0,1] but our normalization keeps
    values > 1."""

    def __init__(self, a, p=1.0):
        super().__init__(p=p)
        self.hi = float(np.log1p(a))

    def get_params(self):
        return {"gain": float(np.exp(self.py_random.uniform(-self.hi, self.hi)))}

    def apply(self, img, gain, **kw):
        return img * np.float32(gain)

    def apply_to_volume(self, volume, gain, **kw):
        return volume * np.float32(gain)


class IntensityAug(A.ImageOnlyTransform):
    """Imaging-condition intensity variation (intensity
    group). Operates on the percentile-normalized volume (background ~0,
    GT-node signal ~0.5-1.0). Each component fires independently with p=0.5;
    ranges calibrated from train statistics (2026-08-19): per-slice p99 gain
    CV p50=0.18, background-noise sigma p50=0.009/p90=0.116, node signal
    p10-p90 = 0.48-0.96. s scales every amplitude (1.0 = calibrated)."""

    def __init__(self, s, p=1.0):
        super().__init__(p=p)
        self.s = float(s)

    def get_params(self):
        r, s = self.py_random, self.s
        on = lambda: r.random() < 0.5
        return {
            "gamma": float(np.exp(r.uniform(-1, 1) * np.log1p(0.25 * s)))
            if on() else 1.0,
            "offset": r.uniform(-0.08, 0.08) * s if on() else 0.0,
            "noise_a": r.uniform(0.0, 0.10) * s if on() else 0.0,   # additive
            "noise_s": r.uniform(0.0, 0.08) * s if on() else 0.0,   # ~Poisson
            "illum_amp": r.uniform(0.0, 0.20) * s if on() else 0.0,
            "slice_sigma": r.uniform(0.0, 0.15) * s if on() else 0.0,
            "seed": r.randrange(1 << 31),
        }

    def apply(self, img, **kw):
        return self.apply_to_volume(img[None], **kw)[0]

    def apply_to_volume(self, volume, gamma, offset, noise_a, noise_s,
                        illum_amp, slice_sigma, seed, **kw):
        v = volume.astype(np.float32)
        rng = np.random.default_rng(seed)
        if gamma != 1.0:
            v = np.power(np.clip(v, 0, None), np.float32(gamma))
        if slice_sigma > 0:      # per-z gain jitter (depth-varying response)
            g = rng.normal(1.0, slice_sigma, size=(v.shape[0], 1, 1))
            v = v * np.clip(g, 0.5, 1.5).astype(np.float32)
        if illum_amp > 0:        # low-frequency in-plane illumination field
            coarse = rng.normal(0.0, 1.0, size=(4, 4)).astype(np.float32)
            field = cv2.resize(coarse, (v.shape[2], v.shape[1]),
                               interpolation=cv2.INTER_CUBIC)
            v = v * (1.0 + illum_amp * field)[None]
        if noise_a > 0 or noise_s > 0:
            sig = np.sqrt(np.clip(v, 0, None))
            v = v + rng.standard_normal(v.shape).astype(np.float32) \
                * (noise_a + noise_s * sig)
        if offset != 0.0:
            v = v + np.float32(offset)
        return v


class OpticsAug(A.ImageOnlyTransform):
    """PSF variation (optics group): anisotropic Gaussian
    blur with INDEPENDENT z and YX sigmas (p=0.4), or a mild unsharp mask
    (p=0.2, mutually exclusive), else identity. Sigmas in voxels on the
    training grid. s scales the amplitudes (1.0 = calibrated)."""

    def __init__(self, s, p=1.0):
        super().__init__(p=p)
        self.s = float(s)

    def get_params(self):
        r, s = self.py_random, self.s
        u = r.random()
        if u < 0.4:
            return {"sz": r.uniform(0.0, 0.8) * s,
                    "syx": r.uniform(0.0, 1.2) * s, "amount": 0.0}
        if u < 0.6:
            return {"sz": 0.0, "syx": 1.0,
                    "amount": r.uniform(0.2, 0.7) * s}
        return {"sz": 0.0, "syx": 0.0, "amount": 0.0}

    def apply(self, img, sz, syx, amount, **kw):
        return self.apply_to_volume(img[None], sz, syx, amount, **kw)[0]

    def apply_to_volume(self, volume, sz, syx, amount, **kw):
        from scipy.ndimage import gaussian_filter
        v = volume.astype(np.float32)
        if amount > 0:           # unsharp: v + a * (v - blur(v))
            return v + np.float32(amount) * (v - gaussian_filter(v, (0, syx, syx)))
        if sz > 0 or syx > 0:
            return gaussian_filter(v, (sz, syx, syx))
        return v


def build_train_aug(cfg):
    """Train-time pipeline over (volume=(Z,Y,X) z-cropped frame, keypoints=XY,
    mask=2D YX validity). 2D transforms apply to volumes slice-wise in
    YX with shared params; the final YX crop happens OUTSIDE the pipeline
    (jitter around the transformed chosen node, dataset.__getitem__), and the
    heatmap/mask are rendered after it from the transformed keypoints, so
    target sigma and carve radii stay exact under scaling."""
    tfs = []
    if float(cfg.data.get("optics_aug", 0.0)) > 0:
        tfs.append(OpticsAug(float(cfg.data.optics_aug)))
    if float(cfg.data.get("intensity_aug", 0.0)) > 0:
        tfs.append(IntensityAug(float(cfg.data.intensity_aug)))
    if cfg.data.gain_aug > 0:
        tfs.append(RandomGain(cfg.data.gain_aug))
    if cfg.data.flip_aug:
        # uniform D4 in YX (square plane -> free flips/90° rotations). No z
        # flip: z has orientation-consistent optics.
        tfs.append(A.D4(p=1.0))
    if cfg.data.rot_scale_aug:
        # keep_ratio: one isotropic YX factor (default samples x/y scales
        # INDEPENDENTLY); fit_output: canvas grows to keep all content, so a
        # border node's transformed position stays in-canvas and croppable;
        # bilinear mask: nearest would keep the 1-px fill-blended ring at
        # rotated content edges "valid" (see the > 0.99 threshold below)
        s = float(cfg.data.rot_scale_s)
        rot = (-180, 180) if bool(cfg.data.get("rot_scale_rot", True)) else (0, 0)
        tfs.append(A.Affine(rotate=rot, scale=(1 - s, 1 + s),
                            keep_ratio=True, fit_output=True, fill=0.0,
                            mask_interpolation=cv2.INTER_LINEAR, p=1.0))
        # fit_output SHRINKS the canvas below the crop when scale < crop/plane
        # (reachable at rot_scale_s > 0.5); pad back so the YX crop always fits
        _, cy, cx = cfg.data.crop_size
        tfs.append(A.PadIfNeeded(min_height=int(cy), min_width=int(cx),
                                 border_mode=cv2.BORDER_CONSTANT, fill=0.0,
                                 fill_mask=0.0, p=1.0))
    return A.Compose(tfs, keypoint_params=A.KeypointParams(
        format="xy", remove_invisible=False), save_applied_params=True)


def load_gt(geff_path):
    """GT graph from a .geff dir -> (ids, t, zyx, edges, n_total). Same as exp_001."""
    g = zarr.open(str(geff_path), mode="r")
    ids = np.asarray(g["nodes/ids"])
    t = np.asarray(g["nodes/props/t/values"])
    zyx = np.stack([np.asarray(g["nodes/props/z/values"]),
                    np.asarray(g["nodes/props/y/values"]),
                    np.asarray(g["nodes/props/x/values"])], axis=1)
    edges = np.asarray(g["edges/ids"]) if "edges/ids" in g else np.zeros((0, 2), np.int64)
    n_total = g.attrs.asdict().get("geff", {}).get("extra", {}).get("estimated_number_of_nodes")
    return ids, t, zyx, edges, (float(n_total) if n_total else None)


class CellStore:
    """Per-sample zarr volumes (lazy, worker-safe) + eagerly loaded GT graphs."""

    def __init__(self, data_dir, ids, with_gt=True, det_dir=None, downsample=(1, 1, 1),
                 raw_cache=None):
        self.data_dir = Path(data_dir)
        # optional {(sid,t): raw frame} shared with another CellStore over the
        # same volumes at a different downsample, so the zarr chunk is
        # decompressed once instead of once per store. Caller owns eviction.
        self._raw = raw_cache
        self.det_dir = Path(det_dir) if det_dir else None
        self.ds = np.array(list(downsample), int)  # (z,y,x) mean-pool factors
        self._det = {}  # sid -> {t: (det (N,3) int, matched (N,) bool)}, per process
        self.ids = list(ids)
        self.norm = {}        # sid -> (ql, qh) from zarr attrs image_statistics
        self.shape = {}       # sid -> (T, Z, Y, X)
        self.gt = {}          # sid -> (ids, t, zyx, edges, n_total)
        self.nodes_by_t = {}  # sid -> {t: (M,3) int zyx}
        self._arr = {}        # sid -> open zarr array (per process, not pickled)
        self._dir = {}        # sid -> directory holding {sid}.zarr / .geff
        # dl_190: densely labelled external samples (every nucleus annotated).
        # They live in another directory, have no
        # DoG det cache and need none: with no unmatched detections the carved
        # mask is all ones, i.e. the whole crop is supervised.
        self.dense = set()
        self._load(self.data_dir, self.ids, with_gt)

    def _load(self, data_dir, ids, with_gt):
        for sid in ids:
            self._dir[sid] = Path(data_dir)
            zg = zarr.open(str(Path(data_dir) / f"{sid}.zarr"), mode="r")
            q = dict(zg.attrs)["image_statistics"]["quantiles"]
            self.norm[sid] = (float(q["0.001"]), float(q["0.999"]))
            self.shape[sid] = tuple(zg["0"].shape)
            if with_gt:
                gt = load_gt(Path(data_dir) / f"{sid}.geff")
                self.gt[sid] = gt
                by_t = {}
                for tt, zyx in zip(gt[1], gt[2]):
                    by_t.setdefault(int(tt), []).append(zyx)
                self.nodes_by_t[sid] = {tt: np.array(v, np.int32) for tt, v in by_t.items()}

    def add_dense(self, data_dir, ids):
        """Register densely labelled external samples (see `dense`)."""
        self._load(data_dir, ids, with_gt=True)
        self.ids += list(ids)
        self.dense |= set(ids)

    def frame_slab(self, sid, t, z0, z1):
        """Dense samples only: normalized float32 (z1-z0, Y, X) slab read from the
        memmapped {sid}.npy (uint16 (Z,Y,X), T=1).
        Reading the whole volume through zarr (one 34-150 MB zstd chunk) took up
        to 11 s per item inside forked dataloader workers and starved the GPU."""
        assert t == 0 and (self.ds == 1).all()
        arr = self._arr.get(sid)
        if arr is None:
            arr = np.load(str(self._dir[sid] / f"{sid}.npy"), mmap_mode="r")
            self._arr[sid] = arr
        ql, qh = self.norm[sid]
        f = np.asarray(arr[z0:z1], dtype=np.float32)
        return np.clip((f - ql) / (qh - ql + 1e-6), 0, None)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arr"] = {}  # zarr handles are re-opened per worker
        state["_det"] = {}
        state["_raw"] = None
        return state

    def dets(self, sid, t):
        """Cached low-threshold detections at (sid, t) -> (coords (N,3), matched (N,))."""
        if sid in self.dense:
            return np.zeros((0, 3), np.int32), np.zeros(0, bool)
        d = self._det.get(sid)
        if d is None:
            path = self.det_dir / f"{sid}.npz"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing")
            z = np.load(path)
            d = {int(k[1:-4]): (z[k].astype(np.int32), z[f"{k[:-4]}_match"])
                 for k in z.files if k.endswith("_det")}
            self._det[sid] = d
        return d[int(t)]

    def frame(self, sid, t):
        """Normalized float32 (Z,Y,X) frame, same normalization as exp_001.
        With downsample > 1 the raw frame is mean-pooled per axis first
        (anti-aliased 1/ds grid; remainder voxels are dropped)."""
        raw = None if self._raw is None else self._raw.get((sid, t))
        if raw is None:
            arr = self._arr.get(sid)
            if arr is None:
                arr = zarr.open(str(self._dir[sid] / f"{sid}.zarr"), mode="r")["0"]
                self._arr[sid] = arr
            raw = arr[t]
            if self._raw is not None:
                self._raw[(sid, t)] = raw
        ql, qh = self.norm[sid]
        f = raw.astype(np.float32)
        if (self.ds > 1).any():
            dz, dy, dx = self.ds
            Z, Y, X = f.shape
            f = f[:Z // dz * dz, :Y // dy * dy, :X // dx * dx]
            f = f.reshape(Z // dz, dz, Y // dy, dy, X // dx, dx).mean(axis=(1, 3, 5))
        return np.clip((f - ql) / (qh - ql + 1e-6), 0, None)


T15 = float(np.exp(-0.5 * 1.5 ** 2))  # Gaussian value at 1.5 sigma (flow core)


def render_flow(shape, src, dzyx, sigma_vox):
    """Flow regression targets around linked source nodes.

    Within the <=1.5 sigma core of each node the target is the node's constant
    (dz, dy, dx) displacement and the loss weight is the node's Gaussian value.
    Overlaps are resolved by keeping the larger weight (same max-composition
    as gaussian_heatmap). NOTE divisions: two links from one source render
    identical-center cores, and the strict `g > w` keeps only the FIRST
    child's displacement (GT edge order) — the second child is not
    represented in the flow target.
    Returns (flow (3,*shape) [dz, dy, dx], w (*shape))."""
    flow = np.zeros((3,) + tuple(shape), np.float32)
    w = np.zeros(shape, np.float32)
    shape = np.array(shape)
    rad = np.maximum(1, np.ceil(1.5 * sigma_vox)).astype(int)
    for c, d in zip(src, dzyx):
        lo = np.maximum(c - rad, 0)
        hi = np.minimum(c + rad + 1, shape)
        if np.any(lo >= hi):
            continue
        zz = ((np.arange(lo[0], hi[0]) - c[0]) / sigma_vox[0]) ** 2
        yy = ((np.arange(lo[1], hi[1]) - c[1]) / sigma_vox[1]) ** 2
        xx = ((np.arange(lo[2], hi[2]) - c[2]) / sigma_vox[2]) ** 2
        g = np.exp(-0.5 * (zz[:, None, None] + yy[None, :, None] + xx[None, None, :]),
                   dtype=np.float32)
        g[g < T15] = 0.0
        sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        upd = g > w[sl]
        w[sl][upd] = g[upd]
        for k in range(3):
            flow[k][sl][upd] = np.float32(d[k])
    return flow, w


def load_anomaly_links(cfg):
    """{sid: {(source_id, target_id), ...}} to mask from TRAIN link supervision,
    or {} when data.drop_anomaly_links is off. Anomalous GT links (identity
    switches, spike nodes' edges, the 6bba_f20478e9 acquisition jump). Val
    datasets keep the full link set so metrics stay comparable across runs."""
    if not bool(cfg.data.get("drop_anomaly_links", False)):
        return {}
    path = Path(cfg.data.get("anomaly_links_path", "data/gt_anomaly_links.csv"))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing")
    drop = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            drop.setdefault(row["sample"], set()).add(
                (int(row["source_id"]), int(row["target_id"])))
    print(f"anomaly link mask: dropping {sum(len(v) for v in drop.values())} "
          f"GT links over {len(drop)} samples ({path})")
    return drop


def build_links(store, sid, drop=None):
    """GT edges of one sample -> {t: [(idx in nodes_by_t[t], idx in
    nodes_by_t[t+1])]} for consecutive-frame links. Indices follow the GT row
    order, matching how CellStore builds nodes_by_t. drop: optional set of
    (source_id, target_id) GT id pairs to exclude (load_anomaly_links)."""
    gids, gt_t, _, gedges, _ = store.gt[sid]
    id2row = {int(g): k for k, g in enumerate(gids)}
    order, counters = {}, {}
    for k in range(len(gids)):
        tt = int(gt_t[k])
        order[k] = counters.get(tt, 0)
        counters[tt] = order[k] + 1
    links = {}
    for u, v in gedges:
        if drop and (int(u), int(v)) in drop:
            continue
        ku, kv = id2row[int(u)], id2row[int(v)]
        tu, tv = int(gt_t[ku]), int(gt_t[kv])
        if tv == tu + 1:
            links.setdefault(tu, []).append((order[ku], order[kv]))
    return links


DIV_POS, DIV_NEG, DIV_IGN = 1, 0, -1   # daughter-channel node classes (dl_195)
DIV_PAR = 2                            # division parent (matcher node-cls only)


def build_div_classes(store, sid, drop=None, parent_class=False,
                      t2_daughter_neg=False):
    """Daughter-channel supervision class per GT node
    -> {t: (M,) int8 aligned with nodes_by_t[t]}: DIV_POS = daughter at its
    birth frame, DIV_NEG = clear negative, DIV_IGN = excluded from the loss.

    Parenthood uses dt=1 GT edges only (build_links convention); drop is the
    anomaly-edge set. Priority: daughter POS > division-parent NEG (the mother
    one frame before the split, suppressing early fork proposals) > daughter at
    t+2 IGN (near-identical appearance to t+1; a contradictory teacher would
    flatten the peak) > unknown-parent IGN (sparse GT: could be an unlabeled
    division's daughter) > continuation NEG.

    parent_class=True (matcher node-cls loss, loss.matcher_cls_weight):
    division parents become their own class DIV_PAR instead of DIV_NEG — same
    priority order, so the labels only split the binary NEG class.

    t2_daughter_neg=True (div3f parent classifier): the t+2 IGN above exists
    to avoid a contradictory teacher against the POSITIVE t+1 daughter of the
    daughter-channel task; in a task where the t+1 daughter is itself a
    NEGATIVE there is no contradiction, so the t+2 daughter becomes DIV_NEG.
    """
    gids, gt_t, _, gedges, _ = store.gt[sid]
    id2row = {int(g): k for k, g in enumerate(gids)}
    order, counters = {}, {}
    for k in range(len(gids)):
        tt = int(gt_t[k])
        order[k] = counters.get(tt, 0)
        counters[tt] = order[k] + 1
    parent = {}                              # child row -> parent row
    nkids = np.zeros(len(gids), np.int64)
    for u, v in gedges:
        if drop and (int(u), int(v)) in drop:
            continue
        ku, kv = id2row[int(u)], id2row[int(v)]
        if int(gt_t[kv]) == int(gt_t[ku]) + 1:
            parent[kv] = ku
            nkids[ku] += 1
    pos = {k for k, p in parent.items() if nkids[p] >= 2}
    cls = {tt: np.zeros(n, np.int8) for tt, n in counters.items()}
    for k in range(len(gids)):
        if k in pos:
            c = DIV_POS
        elif nkids[k] >= 2:
            c = DIV_PAR if parent_class else DIV_NEG
        elif parent.get(k) in pos:
            c = DIV_NEG if t2_daughter_neg else DIV_IGN
        elif k not in parent:
            c = DIV_IGN
        else:
            c = DIV_NEG
        cls[int(gt_t[k])][order[k]] = c
    return cls


def gaussian_heatmap(shape, coords, sigma_vox, yx_scale=None):
    """Max-composited anisotropic Gaussians (peak 1) at integer zyx coords.

    yx_scale: optional (N,) per-node factor on sigma_y/sigma_x (dl_175:
    crowding-adaptive targets — a node with a detection nearby gets a sharper
    in-plane Gaussian so the target keeps a valley between the pair; sigma_z
    stays fixed, the z pitch (1.625 µm) is too coarse to shrink safely)."""
    hm = np.zeros(shape, np.float32)
    shape = np.array(shape)
    for i, c in enumerate(coords):
        s = np.array(sigma_vox, np.float64)
        if yx_scale is not None:
            s[1:] *= yx_scale[i]
        rad = np.maximum(1, np.ceil(3.0 * s)).astype(int)
        lo = np.maximum(c - rad, 0)
        hi = np.minimum(c + rad + 1, shape)
        if np.any(lo >= hi):
            continue
        zz = ((np.arange(lo[0], hi[0]) - c[0]) / s[0]) ** 2
        yy = ((np.arange(lo[1], hi[1]) - c[1]) / s[1]) ** 2
        xx = ((np.arange(lo[2], hi[2]) - c[2]) / s[2]) ** 2
        g = np.exp(-0.5 * (zz[:, None, None] + yy[None, :, None] + xx[None, None, :]))
        region = hm[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        np.maximum(region, g.astype(np.float32), out=region)
    return hm


COARSE_STRIDE = np.asarray(_COARSE_STRIDE)   # (z,y,x); single source: core/metric.py


def build_coarse_targets(shape, nodes_f, mask, target="binary", sigma_cells=None):
    """[1,4,4]-grid targets for arch=unet25d_coarse (v2 near-block scheme).

    Per node, the ambiguous cells are the NEAR-SIDE 1x2x2 block: the node's
    own cell plus the yx neighbours on the side its continuous position is
    closer to (same z slice — z sits exactly at its cell centre under the
    c=0 convention, so there is no z near side; z+-1 stays a plain negative,
    which also keeps every block cell within Chebyshev-1 of the others so a
    3^3 NMS structurally keeps exactly ONE detection per node — a symmetric
    ring's opposite cells are 2 apart and can BOTH survive, the dl_297 v1
    duplicate/position failure). Block cells other than the centre get NO cls
    loss but DO get offset supervision pointing at the node (|off| up to 1.0
    cell), so whichever block cell wins NMS decodes to the true position.

    nodes_f: (N,3) float crop-local zyx (may lie 1 cell outside the crop —
    the in-bounds part of the block is still stamped). mask: (Z,Y,X)
    full-res loss weights with carve/validity applied.

    Returns cls (1,Z,Yc,Xc) {0,1}, off (3,Z,Yc,Xc) cell-unit offsets
    (c=0 for z / 0.5 for yx, matches metric.decode_coarse), cmask
    (Z,Yc,Xc) cls weights = min-pooled mask with near cells zeroed at
    non-positive cells, offw (Z,Yc,Xc) offset weights (block cells with
    surviving validity). Same-cell collisions overwrite deterministically;
    positive centres are re-stamped LAST so a neighbour's near-block write
    can never leave a positive cell's offset pointing at the other node.
    """
    Z, Y, X = shape
    gshape = np.array([Z, Y // COARSE_STRIDE[1], X // COARSE_STRIDE[2]])
    pooled = mask.reshape(Z, gshape[1], COARSE_STRIDE[1],
                          gshape[2], COARSE_STRIDE[2]).min(axis=(2, 4))
    cls = np.zeros(tuple(gshape), np.float32)
    off = np.zeros((3, *gshape), np.float32)
    offw = np.zeros(tuple(gshape), np.float32)
    gauss = target == "gauss"
    grid = np.asarray(nodes_f, np.float64) / COARSE_STRIDE
    idx = np.floor(grid).astype(int)
    c = np.asarray(COARSE_C)
    for gi, gf in zip(idx, grid):
        frac = gf - gi
        ny = 1 if frac[1] >= 0.5 else -1   # near-side yx neighbours
        nx = 1 if frac[2] >= 0.5 else -1
        for dy in (0, ny):
            for dx in (0, nx):
                j = gi + np.array([0, dy, dx])
                if (j < 0).any() or (j >= gshape).any():
                    continue
                off[:, j[0], j[1], j[2]] = gf - j - c
                offw[tuple(j)] = 1.0
        if not gauss and (gi >= 0).all() and (gi < gshape).all():
            cls[tuple(gi)] = 1.0
    # centres LAST: a neighbour's near-block write above must not leave a
    # positive cell's offset pointing at the other node (the winning positive
    # would then decode onto its neighbour -> duplicate there + a miss here)
    for gi, gf in zip(idx, grid):
        if (gi >= 0).all() and (gi < gshape).all():
            off[:, gi[0], gi[1], gi[2]] = gf - gi - c
    if gauss:
        # coarse_target=gauss: the continuous Gaussian teacher replaces the
        # binary centre + ignore ring — the tail supervises "where the peak
        # is not" densely (the property that makes the full-res MSE line
        # strong; dl_297/299 showed the binary scheme's unsupervised holes
        # and hard in-nucleus negatives are exactly what it loses). Peaks
        # stamp at integer cell indices (same quantization philosophy as the
        # full-res teacher); the offset head carries the exact position.
        inb = idx[((idx >= 0) & (idx < gshape)).all(axis=1)]
        cls = gaussian_heatmap(tuple(gshape), inb, np.asarray(sigma_cells))
        cmask = pooled.astype(np.float32)          # no ignore ring
    else:
        cmask = (pooled * ~((offw > 0) & (cls < 0.5))).astype(np.float32)
    offw = (offw * (pooled > 0)).astype(np.float32)
    return cls[None], off, cmask, offw


def filter_unmatched(det, matched, nodes, dup_um):
    """Unmatched detections (suspected unlabeled cells) minus those within
    dup_um µm of an annotated node (duplicate detections of a labeled cell).
    Original-voxel coords in and out (float64)."""
    unmatched = det[~matched].astype(np.float64)
    if len(unmatched) and nodes is not None and len(nodes):
        d = np.linalg.norm((unmatched[:, None] - nodes[None].astype(np.float64))
                           * SCALE, axis=2)
        unmatched = unmatched[d.min(axis=1) > dup_um]
    return unmatched


def carve_ellipsoid(mask, center, rad_vox, value=0.0):
    """Set mask=value inside an anisotropic ellipsoid (bounds-safe, in place)."""
    r = np.ceil(rad_vox).astype(int)
    lo = np.maximum(center - r, 0)
    hi = np.minimum(center + r + 1, mask.shape)
    if np.any(lo >= hi):
        return
    zz = ((np.arange(lo[0], hi[0]) - center[0]) / rad_vox[0]) ** 2
    yy = ((np.arange(lo[1], hi[1]) - center[1]) / rad_vox[1]) ** 2
    xx = ((np.arange(lo[2], hi[2]) - center[2]) / rad_vox[2]) ** 2
    ell = zz[:, None, None] + yy[None, :, None] + xx[None, None, :] <= 1.0
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sub[ell] = np.minimum(sub[ell], value)


def build_mask(shape, hm, mask_mode, dets=None, origin=None, carve_rad_vox=None,
               nodes=None, core_thresh=None, dup_um=2.0, ds=None, carve_weight=0.0):
    """Loss WEIGHT map (float32). "support": the 3-sigma Gaussian support only.

    "carved": everything except ellipsoids around unmatched detections
    (suspected unlabeled cells), where carving takes priority over the
    annotated support EXCEPT for the inner core (target >= core_thresh,
    ~1.5 sigma): there the target value is dominated by the annotated cell's
    own Gaussian and is correct regardless of neighbors, so it is always kept.
    Unmatched detections within dup_um of an annotated node are treated as
    duplicate detections of that cell and do not carve.

    carve_weight is the weight left INSIDE a carve (0 = the original hard
    carve; 1 = no carve at all, i.e. suspected unlabeled cells are ordinary
    background).  Intermediate values make them weak negatives, which is what
    teaches the detector WHICH cells get annotated (the public pilkwang
    detector trains every non-GT voxel as a 0.01-weight negative and thereby
    spends its node budget on annotation-like cells).

    Loss regions follow from (weight, target): voxels split into positive
    (target >= MASK_THRESH) and background, each averaged with these weights.
    """
    if mask_mode == "support":
        return (hm >= MASK_THRESH).astype(np.float32)
    mask = np.ones(shape, np.float32)
    det, matched = dets
    unmatched = filter_unmatched(det, matched, nodes, dup_um)
    if ds is not None:
        # dup filtering above runs in original coords (exact µm); carving
        # happens on the downsampled grid
        unmatched = np.rint(to_ds(unmatched, ds))
    for dpt in unmatched:
        carve_ellipsoid(mask, dpt.astype(int) - origin, carve_rad_vox, carve_weight)
    return np.maximum(mask, (hm >= core_thresh).astype(np.float32))


class CellTrainDataset(Dataset):
    """One item = one VIDEO (sample); each fetch draws a random annotated frame
    and a random GT node in it. The node's z (+ jitter) picks the z crop before
    the aug pipeline; the final YX crop is centered on the node's TRANSFORMED
    position (+ jitter) after the pipeline — the classic GT-centered-with-jitter
    sampling, with the jitter applied in output coordinates. An epoch therefore
    iterates the videos once (one crop per video per epoch).

    data.drop_dup_frames removes frames that are byte-copies of the previous
    frame from the candidate pool (frozen frames, 6bba only, ~7.4% of its
    frames; zero information — GT is static there).
    """

    def __init__(self, cfg, store, ids):
        self.store = store
        dup = {}
        if cfg.data.drop_dup_frames:
            path = Path(cfg.data.dup_frames_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} missing")
            import json
            dup = {sid: set(ts) for sid, ts in json.loads(path.read_text()).items()}
        self.frames = {}   # sid -> eligible annotated frames
        self.items = []    # one entry per video
        for sid in ids:
            ts = [t for t in sorted(store.nodes_by_t[sid]) if t not in dup.get(sid, ())]
            if ts:
                self.frames[sid] = ts
                self.items.append(sid)
        # dl_622: a soft teacher supervises EVERY frame and every voxel, so the
        # crop can be drawn uniformly from the volume instead of around a GT
        # node. The GT-centered draw (both the annotated-frames-only pool and
        # the node-centered window) is what keeps regions holding only
        # unlabeled cells out of training; it exists because GT covers 2.8% of
        # the cells, which is not a constraint the teacher has.
        self.random_crop = bool(cfg.data.get("random_crop_sampling", False))
        self.all_frames = {}
        if self.random_crop:
            for sid in ids:
                self.all_frames[sid] = [t for t in range(int(store.shape[sid][0]))
                                        if t not in dup.get(sid, ())]
        self.crop = np.array(cfg.data.crop_size, int)
        self.ds = get_downsample(cfg)          # crop/jitter are in this grid
        scale_ds = SCALE * self.ds             # µm per downsampled voxel
        self.sigma_vox = np.asarray(cfg.model.heatmap_sigma_um,
                                    np.float64) / scale_ds
        j = int(cfg.data.jitter_yx)
        self.jitter = np.array([max(1, j // 4), j, j])
        self.aug = build_train_aug(cfg)
        self._aug_pid = None  # per-worker reseed marker (see __getitem__)
        self.mask_mode = str(cfg.loss.mask_mode)
        self.carve_rad_vox = float(cfg.loss.carve_radius_um) / scale_ds
        self.carve_weight = float(cfg.loss.get("carve_weight", 0.0))
        self.gt_loss_radius_um = float(cfg.loss.get("gt_loss_radius_um", 0.0))
        self.gt_region_rad_vox = self.gt_loss_radius_um / scale_ds
        self.core_thresh = float(np.exp(-0.5 * float(cfg.loss.core_sigma) ** 2))
        self.dup_um = float(cfg.loss.dup_radius_um)
        self.local_contrast = bool(cfg.data.get("local_contrast", False))
        # local_background's ds=8 window is calibrated on the original
        # 0.40625 um YX grid; on a pooled grid the same box silently
        # covers 4x the tissue dl_155 measured, so refuse rather than
        # quietly feed a different feature than the diagnostic found
        assert not self.local_contrast or (self.ds == 1).all(), (
            f"data.local_contrast with downsample={self.ds.tolist()}: "
            "the +-4.9 um background window is only calibrated at ds 1")
        # Affine/PadIfNeeded fill zeros, and the background window pulls that
        # zero in up to ~11 voxels PAST what the validity mask drops: measured
        # +0.11 contrast bias there, the size of the whole TP-vs-FN separation
        # dl_155 found (0.276 vs 0.433). Teaches "dark border => high contrast",
        # which inference never shows.
        assert not self.local_contrast or not cfg.data.rot_scale_aug, (
            "data.local_contrast with rot_scale_aug: affine zero-fill biases "
            "the contrast plane inside the supervised region")
        self.dense_floor_s = float(cfg.data.get("extra_floor_aug", 0.0))  # dl_190
        self.floor_aug_s = float(cfg.data.get("local_floor_aug", 0.0))
        # same window-calibration constraint as local_contrast above
        assert self.floor_aug_s == 0 or (self.ds == 1).all(), (
            f"data.local_floor_aug with downsample={self.ds.tolist()}: "
            "the +-4.9 um background window is only calibrated at ds 1")
        # dl_169/dl_173: teacher detections as pseudo-positive targets (GT is
        # ~2.8% of true cells; this densifies positive supervision ~30x).
        # conf >= pseudo_pos_th -> positive; [pseudo_min_th, pseudo_pos_th) ->
        # the teacher's uncertain band, excluded from the loss like an
        # unmatched DoG det (dl_172: the failure lives in the low-confidence
        # ordering; do not teach that band as background OR as cells)
        self.pseudo_dir = cfg.data.get("pseudo_det_dir")
        self.pseudo_pos_th = float(cfg.data.get("pseudo_pos_th", 0.2))
        self.pseudo_min_th = float(cfg.data.get("pseudo_min_th", 0.05))
        # dl_177: pair-local pseudo — keep only strong teacher dets within
        # this distance of a GT node as positives (the absorbed-pair
        # mechanism, dl_150/171: the missing training signal is a SECOND peak
        # right next to an annotated cell; near-GT detections are almost
        # surely real cells, so unlike the closed global-pseudo axis the FP
        # cost is tiny). Farther strong dets carve like uncertain dets.
        # 0 = off (all strong dets are positives, dl_169/173 behaviour).
        self.pseudo_pair_um = float(cfg.data.get("pseudo_pair_um", 0.0))
        self._pseudo = {}   # sid -> open NpzFile (per worker, lazy)
        # dl_622: SOFT heatmap distillation. The teacher's continuous heatmap
        # rides through the aug
        # pipeline as mask3d, which only the geometric transforms touch (D4 /
        # Affine; the intensity transforms are ImageOnlyTransform), so image
        # and target stay aligned. data.distill_mode picks what to do with it:
        #   replace  the teacher heatmap IS the target; the GT Gaussian and
        #            every GT-derived weight (carve, ring, ...) are unused.
        #   combine  the GT path is left exactly as it was and the teacher
        #            heatmap comes along as an extra target, summed in the
        #            loss (L_gt + L_ps). The two disagree at unlabeled cells --
        #            GT says background, the teacher says cell -- and nothing
        #            arbitrates that by hand: MyLoss normalizes the GT positive
        #            and background regions separately, which already leaves a
        #            GT voxel ~500x the per-voxel weight of a teacher one and
        #            an unlabeled cell ~2x the other way.
        self.distill_dir = cfg.data.get("distill_target_dir")
        self.distill_mode = str(cfg.data.get("distill_mode", "replace"))
        self._distill = {}  # sid -> open memmap (per worker, lazy)
        if self.distill_dir:
            assert self.distill_mode in ("replace", "combine"), self.distill_mode
            assert (self.ds == 1).all(), (
                f"distill_target_dir needs data.downsample [1,1,1], got "
                f"{self.ds.tolist()} — the targets were saved on the 1x grid")
            assert str(cfg.model.get("arch", "unet25d")) != "unet25d_coarse", \
                "distill_target_dir: coarse arch unsupported"
            # replace: two target sources. combine: MyLoss has no pseudo+distill
            # form. Either way it cannot run, so say so at construction rather
            # than on the first training step.
            assert not self.pseudo_dir, \
                "distill_target_dir and pseudo_det_dir are both teacher sources"
        if self.distill_dir and self.distill_mode == "replace":
            # these shape the target/weight map from the GT nodes, which
            # replace does not build; fail loudly rather than ignore them.
            # loss.core_sigma and loss.dup_radius_um are likewise unused here
            # (build_mask / filter_unmatched never run) but have meaningful
            # non-zero defaults, so there is nothing to assert on.
            for k in ("gt_loss_radius_um", "ring_weight", "carve_weight"):
                assert float(cfg.loss.get(k, 0.0)) == 0.0, \
                    f"distill_target_dir=replace: loss.{k} has no meaning on this path"
            assert float(cfg.data.get("adaptive_sigma_k", 0.0)) == 0.0, \
                "distill_target_dir=replace: data.adaptive_sigma_k only shapes GT Gaussians"
        if self.distill_dir and self.distill_mode == "combine":
            # combine keeps the GT term intact, so loss.pooled (which pools the
            # GT positive/background split away) would silently change it
            assert not bool(cfg.loss.get("pooled", False)), \
                "distill_mode=combine: loss.pooled would pool the GT term away"
            # only the plain detection step adds the distill term (pl_module)
            assert int(cfg.model.get("deep_supervision", 0)) == 0, \
                "distill_mode=combine: deep_supervision unsupported"
            assert float(cfg.loss.get("mixup_alpha", 0.0)) == 0.0, \
                "distill_mode=combine: mixup_alpha unsupported"
            assert str(cfg.model.get("temporal", "none")) == "none", \
                "distill_mode=combine: model.temporal must be none"
            # the teacher term is a plain mean over the crop. Only rot/scale
            # pulls zero-fill into the canvas (D4 is an exact permutation and
            # the intensity transforms are ImageOnlyTransform), so with it off
            # there is no invalid region to weight around. Fail loudly instead
            # of silently averaging over fill if it is ever switched on.
            assert not bool(cfg.data.get("rot_scale_aug", False)), \
                "distill_mode=combine: data.rot_scale_aug would put aug fill in the teacher term"
            # a uniformly drawn crop usually holds no GT node, and then the GT
            # term is not "left as it was" but inverted: the target is all
            # zeros while mask_mode=carved leaves the weight ~1 everywhere, so
            # bg_term pulls every teacher-positive voxel toward background.
            # combine is the GT-centered recipe by construction.
            assert not self.random_crop, \
                "distill_mode=combine needs GT-centered sampling (data.random_crop_sampling=false)"
            # only the plain detection branch of pl_module passes the teacher
            # target to the loss; the division branch would drop it silently
            assert float(cfg.loss.get("div_weight", 0.0)) == 0.0, \
                "distill_mode=combine: loss.div_weight takes a branch that ignores the teacher term"
        assert self.distill_dir or not self.random_crop, (
            "data.random_crop_sampling needs data.distill_target_dir — with GT "
            "targets a uniformly drawn crop is mostly unsupervised")
        # dl_175: crowding-adaptive target sharpness. dl_150: 93-95% of misses
        # are absorbed by a neighbour's detection; the global sigma_xy shrink
        # (dl_077/159-162) failed because it also sharpened isolated cells.
        # Here only crowded nodes sharpen: in-plane sigma factor =
        # clip(d_nn / (k * max sigma_um), 0.5, 1) with d_nn the distance to
        # the nearest OTHER object (GT node or low-th DoG det past dup_um,
        # the same crowding proxy dl_150/171 measured). k = 0 disables.
        self.adaptive_sigma_k = float(cfg.data.get("adaptive_sigma_k", 0.0))
        self.sigma_ref_um = float(np.max(np.atleast_1d(np.asarray(
            cfg.model.heatmap_sigma_um, np.float64))))
        # dl_176 candidate: extra loss weight on the target's 3sigma..1sigma
        # mid ring — where a merged two-cell blob's excess mass lives. 0 = off.
        self.ring_weight = float(cfg.loss.get("ring_weight", 0.0))
        # dl_191: synthetic-generator curriculum. A pre-generated pool of
        # FULLY-labelled synthetic volumes (calibrated
        # generator) replaces a train item
        # with probability p_synth, annealed linearly from synth_p_start at
        # epoch 0 to 0 at synth_p_end_frac * max_epochs; the remaining tail
        # trains on real data only. The epoch comes from a training callback
        # (persistent_workers is off on this path so per-epoch
        # worker forks see the updated value). Synthetic items carry a full
        # supervision mask — every cell is labelled, so background is truly
        # background (no carve; the dense-teacher counterpart to the
        # 2.8%-sparse real GT).
        self.synth_files = []
        self.synth_epoch = 0
        pool = cfg.data.get("synth_pool_dir")
        if pool:
            self.synth_files = sorted(Path(pool).glob("vol_*.npz"))
            if not self.synth_files:
                raise FileNotFoundError(
                    f"{pool}: no vol_*.npz")
            assert (self.ds == 1).all(), (
                "synth pool volumes are generated on the native grid; "
                "data.downsample must be [1,1,1]")
            self.synth_p_start = float(cfg.data.get("synth_p_start", 1.0))
            self.synth_end_ep = (float(cfg.data.get("synth_p_end_frac", 0.75))
                                 * int(cfg.trainer.max_epochs))
        # dl_193: dense external curriculum — the store's dense samples form a
        # pool that REPLACES a real item with probability extra_p() (dl_190
        # instead appended them as permanent items). Pool samples need their
        # frame list like any item (T=1 -> frame 0).
        self.extra_pool = []
        self.extra_p_start = float(cfg.data.get("extra_p_start") or 0.0)
        if self.extra_p_start > 0:
            self.extra_pool = sorted(store.dense)
            assert self.extra_pool, "data.extra_p_start set but the store has no dense samples"
            self.extra_end_ep = (float(cfg.data.get("extra_p_end_frac", 0.5))
                                 * int(cfg.trainer.max_epochs))
            for sid in self.extra_pool:
                self.frames.setdefault(sid, sorted(store.nodes_by_t[sid]))
        # dl_195: daughter channel. Per-node
        # supervision classes + birth-event oversampling — uniform frame
        # picking shows a daughter in ~0.2% of items, i.e. no gradient.
        # arch=unet25d_coarse: emit [1,4,4]-grid center/offset targets. v1
        # supports the plain single-frame path only — refuse the exotic
        # combinations instead of silently mis-supervising them.
        self.coarse = str(cfg.model.get("arch", "unet25d")) == "unet25d_coarse"
        self.coarse_target = str(cfg.model.get("coarse_target", "binary"))
        self.coarse_sigma_cells = (np.asarray(cfg.model.heatmap_sigma_um,
                                              np.float64)
                                   / (SCALE * COARSE_STRIDE))
        if self.coarse:
            assert (self.ds == 1).all(), "coarse: data.downsample must be [1,1,1]"
            assert (self.crop[1:] % COARSE_STRIDE[1:] == 0).all(), \
                "coarse: crop YX must be divisible by 4"
            assert not self.local_contrast, "coarse: local_contrast unsupported"
            assert self.pseudo_dir is None, "coarse: pseudo_det_dir unsupported"
            assert self.gt_loss_radius_um == 0, \
                "coarse: gt_loss_radius_um>0 would min-pool away the background"
            assert tuple(int(v) for v in cfg.model.nms_kernel) == (3, 3, 3), \
                "coarse: nms_kernel must be [3,3,3] — the near-block scheme " \
                "keeps one detection per node only under 3^3 coarse-grid NMS"
            assert self.mask_mode == "carved", \
                "coarse: mask_mode=support min-pools background weight to 0"
            assert self.adaptive_sigma_k == 0, \
                "coarse: adaptive_sigma_k shrinks targets below the 4x4 " \
                "cell's core restore — min-pool would zero crowded positives"
            assert self.ring_weight == 0, "coarse: loss.ring_weight unsupported"
            assert float(cfg.loss.get("mixup_alpha", 0.0)) == 0, \
                "coarse: loss.mixup_alpha unsupported"
            assert not self.synth_files and not self.extra_pool, \
                "coarse: synth/dense curriculum paths emit no coarse targets"
            assert float(cfg.loss.get("div_weight", 0.0)) == 0, \
                "coarse: div channel collides with the offset channels"
        self.div_on = float(cfg.loss.get("div_weight", 0.0)) > 0
        self.div_p = float(cfg.data.get("div_oversample", 0.0))
        if self.div_on:
            assert not self.synth_files and not self.extra_pool, (
                "div channel is not wired for the synth/dense curriculum "
                "paths (their items carry no GT graph classes)")
            drop = load_anomaly_links(cfg)
            self.div_cls = {sid: build_div_classes(store, sid, drop.get(sid))
                            for sid in self.items}
            self.div_events = [
                (sid, t, int(i)) for sid in self.items
                for t in self.frames[sid]
                for i in np.flatnonzero(self.div_cls[sid][t] == DIV_POS)]
            assert self.div_events or self.div_p == 0, (
                "data.div_oversample set but no division event in this split")
            # dl_196: unmatched DoG dets (suspected unlabeled cells) as WEAK
            # div negatives at this relative weight (0 = off). P(daughter) for
            # a random cell-frame is ~0.7%, and dl_195's paired diagnostic
            # measured a +0.04..0.06 score lift exactly on these dets
            # (annotation bias: curated lineages skew to clear cells).
            self.div_det_neg = float(cfg.loss.get("div_det_neg", 0.0))

    def set_synth_epoch(self, epoch):
        self.synth_epoch = int(epoch)

    def extra_p(self):
        """dl_193: probability that an item is served from the dense external
        pool (CellStore.dense) instead of its real video — annealed linearly
        from extra_p_start at epoch 0 to 0 at extra_p_end_frac * max_epochs
        (same mechanism as synth_p; the epoch arrives through the same
        callback)."""
        if not self.extra_pool:
            return 0.0
        return self.extra_p_start * max(
            0.0, 1.0 - self.synth_epoch / max(self.extra_end_ep, 1e-9))

    def synth_p(self):
        if not self.synth_files:
            return 0.0
        return self.synth_p_start * max(
            0.0, 1.0 - self.synth_epoch / max(self.synth_end_ep, 1e-9))

    def _synth_item(self):
        """One GT-centered crop from a random pool volume, mirroring the real
        path: same normalization convention as real frames (per-volume
        0.1/99.9 percentiles, the diag_synth_density.infer convention; the
        constants ride in the npz), same aug pipeline, same crop geometry.
        The mask is all-ones x aug validity (see __init__ note)."""
        f = self.synth_files[np.random.randint(len(self.synth_files))]
        with np.load(f) as z:
            vol = z["volume"].astype(np.float32) / 65535.0
            cens = z["centroids"].astype(np.float64)
            ql, qh = float(z["ql"]), float(z["qh"])
        full = np.clip((vol - ql) / (qh - ql + 1e-6), 0, None)
        ci = np.random.randint(len(cens))
        jitter = np.random.randint(-self.jitter, self.jitter + 1)
        assert full.shape[0] >= self.crop[0], (
            f"crop_size z {self.crop[0]} > synth volume z {full.shape[0]}")
        z0 = int(np.clip(int(np.rint(cens[ci, 0])) + jitter[0] - self.crop[0] // 2,
                         0, full.shape[0] - self.crop[0]))
        x = full[z0:z0 + self.crop[0]]
        self._reseed_aug()
        out = self.aug(volume=x, keypoints=cens[:, [2, 1]],
                       mask=np.ones(x.shape[1:], np.float32))
        xt = out["volume"]
        kp = np.asarray(out["keypoints"], np.float64)
        valid = out["mask"] > 0.99
        _, cy, cx = self.crop
        hw = np.array(xt.shape[1:])
        assert (hw >= (cy, cx)).all()
        oy, ox = np.clip(np.rint([kp[ci, 1], kp[ci, 0]]).astype(int)
                         + jitter[1:] - [cy // 2, cx // 2], 0, hw - [cy, cx])
        x = np.ascontiguousarray(xt[:, oy:oy + cy, ox:ox + cx])
        zyx = np.rint(np.stack([cens[:, 0] - z0, kp[:, 1] - oy,
                                kp[:, 0] - ox], 1)).astype(int)
        hm = gaussian_heatmap(x.shape, zyx, self.sigma_vox)
        mask = (np.ones(x.shape, np.float32)
                * valid[oy:oy + cy, ox:ox + cx])
        if os.environ.get("SYNTH_DEBUG"):
            print(f"[synth] epoch={self.synth_epoch} p={self.synth_p():.3f} "
                  f"{f.name} n={len(cens)}", flush=True)
        item = dict(image=to_model_input(x, self.local_contrast),
                    target=torch.from_numpy(hm)[None],
                    mask=torch.from_numpy(mask))
        if self.pseudo_dir:
            item["pseudo"] = torch.from_numpy(np.zeros(x.shape, bool))
        return item

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pseudo"] = {}   # zip handles are re-opened per worker
        state["_distill"] = {}  # memmaps likewise
        return state

    def _replacing(self):
        """True when the teacher heatmap stands in for the GT target."""
        return bool(self.distill_dir) and self.distill_mode == "replace"

    def distill_target(self, sid):
        """-> the teacher heatmap (T,Z,Y,X) float16, memory-mapped."""
        a = self._distill.get(sid)
        if a is None:
            a = np.load(Path(self.distill_dir) / f"{sid}.npy", mmap_mode="r")
            self._distill[sid] = a
        return a

    def pseudo_dets(self, sid, t):
        """-> (coords (N,3) int32, conf (N,) float32). Caches without c-keys
        (operating-point caches, dl_169/170) report conf=inf: everything is a
        positive and the uncertain band is empty."""
        z = self._pseudo.get(sid)
        if z is None:
            z = np.load(Path(self.pseudo_dir) / f"{sid}.npz")
            self._pseudo[sid] = z
        key, ckey = f"t{t:04d}", f"c{t:04d}"
        if key not in z.files:
            return np.zeros((0, 3), np.int32), np.zeros(0, np.float32)
        d = z[key]
        c = z[ckey] if ckey in z.files else np.full(len(d), np.inf, np.float32)
        return d, c

    def __len__(self):
        return len(self.items)

    def _reseed_aug(self):
        if self._aug_pid != os.getpid():
            # Compose RNGs are copied into every dataloader worker (fork /
            # pickle), so all workers would replay identical parameter
            # sequences. Reseed once per process from np.random, which PL's
            # seed_everything(workers=True) seeds distinctly per worker.
            # DISTINCT seed per child: set_random_seed alone gives every child
            # the same seed, lockstep-coupling their parameter streams
            # (measured: gain > 1 iff not flipped, 0% joint occurrence).
            for tr in (self.aug, *self.aug.transforms):
                tr.set_random_seed(int(np.random.randint(0, 2**31 - 1)))
            self._aug_pid = os.getpid()

    def __getitem__(self, i):
        # dl_191: curriculum — serve a fully-labelled synthetic crop instead
        # of a real one with the annealed probability (temporal=none only;
        # the pair/siamese subclasses override __getitem__ and never see this)
        if self.synth_files and np.random.rand() < self.synth_p():
            return self._synth_item()
        sid = self.items[i]
        if self.extra_pool and np.random.rand() < self.extra_p():
            sid = self.extra_pool[np.random.randint(len(self.extra_pool))]  # dl_193
        force_ci = None
        if self.div_on and self.div_events and np.random.rand() < self.div_p:
            # dl_195: serve a crop centered on a daughter at its birth frame
            sid, t, force_ci = self.div_events[
                np.random.randint(len(self.div_events))]
        elif self.random_crop:
            # dl_622: every frame is supervised, not just the annotated ones
            ts = self.all_frames[sid]
            t = ts[np.random.randint(len(ts))]
        else:
            ts = self.frames[sid]
            t = ts[np.random.randint(len(ts))]
        dense = sid in self.store.dense
        frame = None if dense else self.store.frame(sid, t)
        nz = self.store.shape[sid][1] // self.ds[0] if dense else frame.shape[0]
        # random_crop draws the window from the volume, so no GT is consulted
        # (an unannotated frame has no entry in nodes_by_t at all)
        nodes = (np.zeros((0, 3)) if self.random_crop
                 else self.store.nodes_by_t[sid][t])
        # the chosen node's z (+ jitter) picks the z crop BEFORE the pipeline
        # (YX transforms never mix z); its YX drives the post-pipeline crop
        ci = None if self.random_crop else (
            force_ci if force_ci is not None else np.random.randint(len(nodes)))
        jitter = np.random.randint(-self.jitter, self.jitter + 1)
        nodes_ds = to_ds(nodes, self.ds)  # crop geometry lives on the ds grid
        # np.clip(lo=0, hi<0) returns the NEGATIVE bound -> silent wrong-size
        # crops / off-grid keypoints; fail loudly instead
        assert nz >= self.crop[0], (
            f"crop_size z {self.crop[0]} > frame z {nz} "
            f"(shrink data.crop_size for data.downsample={self.ds.tolist()})")
        if self.random_crop:
            # uniform over the z slabs; jitter is subsumed by the uniform draw
            z0 = int(np.random.randint(0, nz - self.crop[0] + 1))
        else:
            z0 = int(np.clip(int(np.rint(nodes_ds[ci, 0])) + jitter[0] - self.crop[0] // 2,
                             0, nz - self.crop[0]))
        if dense:
            # dl_190: memmapped z-slab, no whole-volume read (CellStore.frame_slab)
            x = self.store.frame_slab(sid, t, z0, z0 + self.crop[0])
            if self.dense_floor_s > 0:
                x = dense_floor_aug(x, self.dense_floor_s)
        else:
            x = frame[z0:z0 + self.crop[0]]
        if self.floor_aug_s > 0 and np.random.rand() < 0.5:
            # BEFORE the pipeline: the floor estimate needs the intact plane,
            # and the fields then ride through D4/affine with the image.
            # Returns a new array (x is a view into the store's cached frame).
            x = local_floor_aug(x, self.floor_aug_s)
        tgt3d = None
        if self.distill_dir:
            # dl_622: the same z slab of the teacher heatmap, read from the
            # memmap. It goes through the pipeline with the image below.
            tgt3d = np.asarray(
                self.distill_target(sid)[t, z0:z0 + self.crop[0]], np.float32)
            assert tgt3d.shape == x.shape, (
                f"distill target {sid} t{t}: {tgt3d.shape} != image {x.shape}")

        # dup filtering happens in ORIGINAL coords (exact µm geometry), the
        # surviving suspected-unlabeled-cell points are then transformed along
        # with the GT nodes and carved in the output space
        # pseudo positives: OOF champion dets not explained by a GT node (GT
        # wins within dup_um). They render into the target like GT nodes and
        # protect their carve core; DoG carving below only keeps detections
        # explained by NEITHER.
        pseudo, weak = np.zeros((0, 3)), np.zeros((0, 3))
        if self.pseudo_dir:
            pd, pc = self.pseudo_dets(sid, t)
            strong = pd[pc >= self.pseudo_pos_th]
            band = pd[(pc >= self.pseudo_min_th) & (pc < self.pseudo_pos_th)]
            pseudo = filter_unmatched(strong, np.zeros(len(strong), bool),
                                      nodes, self.dup_um)
            if self.pseudo_pair_um > 0 and len(pseudo):
                dpair = np.linalg.norm(
                    (pseudo[:, None] - nodes.astype(np.float64)[None]) * SCALE,
                    axis=2).min(1)
                # lower bound 3.5 um: a teacher det 2-3.5 um from a GT node is
                # as likely the SAME cell displaced (TP residual median
                # 1.62 um, z quantum 1.625 um — dl_157) as a neighbour;
                # teaching a bimodal target there would split single cells.
                # The absorbed pairs dl_142 measured sit at 4-8 um.
                keep = (dpair > 3.5) & (dpair <= self.pseudo_pair_um)
                weak = np.concatenate([weak, pseudo[~keep]], 0)
                pseudo = pseudo[keep]
            weak = np.concatenate([weak, filter_unmatched(
                band, np.zeros(len(band), bool),
                np.concatenate([nodes.astype(np.float64), pseudo], 0),
                self.dup_um)], 0)
        unmatched, det = np.zeros((0, 3)), None
        # dl_622: replace builds no weight map from GT nodes, so the DoG det
        # cache it would carve with is never read; combine keeps the GT term
        # (and its carve) intact
        if self.mask_mode == "carved" and not self._replacing():
            det, matched = self.store.dets(sid, t)
            targets = np.concatenate([nodes.astype(np.float64), pseudo], 0)
            unmatched = filter_unmatched(det, matched, targets, self.dup_um)
        # the teacher's uncertain band carves exactly like an unmatched DoG det
        unmatched = np.concatenate([unmatched, weak], 0)

        yx_scale = ps_scale = None
        if self.adaptive_sigma_k > 0:
            # pseudo positives sharpen by the same rule: a bimodal pair target
            # needs BOTH components sharp or the fat one swallows the valley
            own = np.concatenate([nodes.astype(np.float64), pseudo], 0)
            others = own
            if det is not None:
                others = np.concatenate([others, det.astype(np.float64)], 0)
            d = np.linalg.norm((own[:, None] - others[None]) * SCALE, axis=2)
            d[d <= self.dup_um] = np.inf  # self + duplicate dets of the same cell
            f = np.clip(d.min(1) / (self.adaptive_sigma_k * self.sigma_ref_um),
                        0.5, 1.0)
            yx_scale, ps_scale = f[:len(nodes)], f[len(nodes):]

        self._reseed_aug()
        pts = to_ds(np.concatenate([nodes.astype(np.float64), pseudo,
                                    unmatched], 0),
                    self.ds)  # zyx on the ds grid (identity when ds = 1)
        # rotation/scaling pull zero-fill into the canvas; the transformed 2D
        # ones-mask marks the surviving true-image region for the loss (all z
        # slices share the same YX transform, so 2D suffices)
        out = self.aug(volume=x, keypoints=pts[:, [2, 1]],
                       mask=np.ones(x.shape[1:], np.float32),
                       **({"mask3d": tgt3d} if tgt3d is not None else {}))
        xt = out["volume"]
        kp = np.asarray(out["keypoints"], np.float64)
        valid = out["mask"] > 0.99  # strict: excludes fill-blended edge pixels
        # isotropic YX factor actually applied by Affine (keep_ratio -> x == y)
        g = next((p["scale"]["x"] for n, p in out["applied_transforms"]
                  if n == "Affine"), 1.0)

        # final YX crop around the transformed chosen node + jitter (classic
        # GT-centered sampling, in output coordinates)
        _, cy, cx = self.crop
        hw = np.array(xt.shape[1:])
        assert (hw >= (cy, cx)).all(), (
            f"crop_size yx {(cy, cx)} > canvas {tuple(hw)} "
            f"(shrink data.crop_size for data.downsample={self.ds.tolist()})")
        if self.random_crop:
            oy, ox = (np.random.randint(0, hw[0] - cy + 1),
                      np.random.randint(0, hw[1] - cx + 1))
        else:
            oy, ox = np.clip(np.rint([kp[ci, 1], kp[ci, 0]]).astype(int)
                             + jitter[1:] - [cy // 2, cx // 2], 0, hw - [cy, cx])
        x = np.ascontiguousarray(xt[:, oy:oy + cy, ox:ox + cx])
        if self.random_crop:
            # no keypoints went through the pipeline; nothing downstream of the
            # distill target reads these (div / coarse / pseudo are all off)
            zyx_f = np.zeros((0, 3))
            lnodes = lpseudo = ldets = np.zeros((0, 3), int)
            n1 = 0
        else:
            zyx_f = np.stack([pts[:, 0] - z0, kp[:, 1] - oy, kp[:, 0] - ox], 1)
            zyx = np.rint(zyx_f).astype(int)
            n1, n2 = len(nodes), len(nodes) + len(pseudo)
            lnodes, lpseudo, ldets = zyx[:n1], zyx[n1:n2], zyx[n2:]

        hm_ps = None
        if self._replacing():
            # dl_622: the teacher heatmap IS the target. It rode through the
            # same geometric transforms as the image, so only the final YX
            # crop is left. Every voxel the teacher covers is supervised at
            # weight 1 — there is no GT/unlabeled distinction to carve around,
            # which is the whole point of training on pseudo labels alone.
            hm = np.ascontiguousarray(out["mask3d"][:, oy:oy + cy, ox:ox + cx],
                                      np.float32)
            mask = np.ones(x.shape, np.float32)
        else:
            # bounds-safe renderers clip: nodes just outside still contribute tails
            hm = gaussian_heatmap(x.shape, lnodes, self.sigma_vox, yx_scale)
            if len(lpseudo):
                hm_ps = gaussian_heatmap(x.shape, lpseudo, self.sigma_vox, ps_scale)
                hm_gt = hm
                # Keep the annotated core unchanged. Outside it, neighbouring
                # cells can contribute their own peak; loss ownership below
                # must follow the source of that value, not support overlap.
                hm = np.where(hm_gt >= self.core_thresh, hm_gt,
                              np.maximum(hm_gt, hm_ps))
            dets = (ldets, np.zeros(len(ldets), bool))  # pre-filtered above
            mask = build_mask(x.shape, hm, self.mask_mode, dets, np.zeros(3, int),
                              self.carve_rad_vox * np.array([1.0, g, g]),
                              None, self.core_thresh, self.dup_um,
                              carve_weight=self.carve_weight)
        if self.gt_loss_radius_um > 0:
            # dl_233: supervise ONLY ellipsoids around GT nodes; the carves
            # (and their preserved cores) inside those regions stay as-is,
            # everything outside gets weight 0 (unlabeled far background is
            # neither positive nor negative)
            keep = np.ones_like(mask)
            grad = self.gt_region_rad_vox * np.array([1.0, g, g])
            for c in lnodes:
                carve_ellipsoid(keep, c, grad, 0.0)
            mask *= (1.0 - keep)
        if self.ring_weight > 0:
            # upweight the 3sigma..1sigma ring of the target: a merged
            # two-cell blob parks its excess mass here, so errors in the ring
            # are exactly the absorption failure mode (dl_150/171)
            ring = (hm >= MASK_THRESH) & (hm < RING_HI)
            mask = (mask * (1.0 + self.ring_weight * ring)).astype(np.float32)
        mask = mask * valid[oy:oy + cy, ox:ox + cx]
        item = dict(image=to_model_input(x, self.local_contrast))
        if self.coarse:
            # the full-res target/mask are DEAD for coarse training (the
            # coarse branch of training_step reads only image + coarse_*);
            # dropping them saves ~14 MB/item of collate + H2D transfer
            ccls, coff, cmask, coffw = build_coarse_targets(
                x.shape, zyx_f[:n1], mask, self.coarse_target,
                self.coarse_sigma_cells)
            item["coarse_cls"] = torch.from_numpy(ccls)
            item["coarse_off"] = torch.from_numpy(coff)
            item["coarse_mask"] = torch.from_numpy(cmask)
            item["coarse_offw"] = torch.from_numpy(coffw)
        else:
            item["target"] = torch.from_numpy(hm)[None]
            item["mask"] = torch.from_numpy(mask)
            if self.distill_dir and self.distill_mode == "combine":
                # the teacher's own target, aug-aligned and cropped like the
                # image. It carries no weight map: the GT carve must NOT apply
                # here (carve zeroes the weight exactly at suspected unlabeled
                # cells, which is where the teacher is the only one with
                # anything to say) and rot_scale_aug is asserted off, so every
                # voxel of the crop is real image.
                item["distill_target"] = torch.from_numpy(
                    np.ascontiguousarray(out["mask3d"][:, oy:oy + cy, ox:ox + cx],
                                         np.float32))[None]
        if self.div_on:
            # daughter channel: target = det
            # -style Gaussians at daughters; weight = the det-channel GT
            # Gaussian over supervised (non-IGN) nodes cut to the <=1.5 sigma
            # core — readout happens at ch1 peaks only, tails carry nothing
            # (render_flow convention)
            cls = self.div_cls[sid][t]
            div_hm = gaussian_heatmap(x.shape, lnodes[cls == DIV_POS],
                                      self.sigma_vox)
            dw = gaussian_heatmap(x.shape, lnodes[cls != DIV_IGN],
                                  self.sigma_vox)
            dw *= dw >= T15
            if self.div_det_neg > 0 and len(ldets):
                # weak negatives on unlabeled dets; dets within 7 um of a
                # daughter stay unsupervised — division neighbourhoods are too
                # precious (302 positives) to risk suppressing. GT supervision
                # wins at overlaps (np.maximum, full-scale GT weight).
                dets_neg = ldets
                pos_nodes = nodes[cls == DIV_POS].astype(np.float64)
                if len(pos_nodes):
                    dd = np.linalg.norm((unmatched[:, None] - pos_nodes[None])
                                        * SCALE, axis=2)
                    dets_neg = ldets[dd.min(1) > 7.0]
                if len(dets_neg):
                    dwd = gaussian_heatmap(x.shape, dets_neg, self.sigma_vox)
                    dwd *= dwd >= T15
                    dw = np.maximum(dw, self.div_det_neg * dwd)
            dw *= valid[oy:oy + cy, ox:ox + cx]
            item["div_target"] = torch.from_numpy(div_hm)[None]
            item["div_w"] = torch.from_numpy(dw)
        if self.pseudo_dir:
            # Attribute each supervised value to its source. A pseudo peak
            # inside a GT tail still receives pseudo_weight, not GT weight.
            # GT owns its protected core and ties; the two losses are disjoint.
            ps = ((hm_ps >= MASK_THRESH) & (hm_ps > hm_gt)
                  & (hm_gt < self.core_thresh)
                  if hm_ps is not None else np.zeros(x.shape, bool))
            item["pseudo"] = torch.from_numpy(ps)
        return item


class CellValDataset(Dataset):
    """Mini official-metric validation: contiguous full-frame windows.

    data.val_max_samples evenly-spaced val samples × data.val_window CONSECUTIVE
    frames each (window centered on the annotated span; edges need t -> t+1).
    Per-frame detections are linked across the window and scored as unadjusted
    edge Jaccard against the windowed GT (pl_module.on_validation_epoch_end);
    node recall/precision and the masked val_loss come from the same frames.
    Variable-length GT coords per item -> use with batch_size=1.
    """

    def __init__(self, cfg, store, ids):
        self.store = store
        self.ds = get_downsample(cfg)
        scale_ds = SCALE * self.ds
        self.sigma_vox = np.asarray(cfg.model.heatmap_sigma_um,
                                    np.float64) / scale_ds
        self.mask_mode = str(cfg.loss.mask_mode)
        self.carve_rad_vox = float(cfg.loss.carve_radius_um) / scale_ds
        self.carve_weight = float(cfg.loss.get("carve_weight", 0.0))
        self.core_thresh = float(np.exp(-0.5 * float(cfg.loss.core_sigma) ** 2))
        self.dup_um = float(cfg.loss.dup_radius_um)
        self.local_contrast = bool(cfg.data.get("local_contrast", False))
        # local_background's ds=8 window is calibrated on the original
        # 0.40625 um YX grid; on a pooled grid the same box silently
        # covers 4x the tissue dl_155 measured, so refuse rather than
        # quietly feed a different feature than the diagnostic found
        assert not self.local_contrast or (self.ds == 1).all(), (
            f"data.local_contrast with downsample={self.ds.tolist()}: "
            "the +-4.9 um background window is only calibrated at ds 1")
        n = min(int(cfg.data.val_max_samples), len(ids))
        self.W = W = int(cfg.data.val_window)
        # evenly spaced over sorted ids -> proportional embryo representation
        pick_ids = [ids[j] for j in np.linspace(0, len(ids) - 1, n).astype(int)]
        self.window = {}   # sid -> t0 (window = [t0, t0+W))
        self.items = []
        self.gt_win = {}   # sid -> (ids, t - t0, zyx, edges, None), window-restricted
        for sid in pick_ids:
            ts = sorted(store.nodes_by_t[sid])
            T = store.shape[sid][0]
            t0 = int(np.clip(ts[len(ts) // 2] - W // 2, 0, T - W))
            self.window[sid] = t0
            self.items += [(sid, t) for t in range(t0, t0 + W)]
            gids, gt, gzyx, gedges, n_total = store.gt[sid]
            sel = (gt >= t0) & (gt < t0 + W)
            keep = set(gids[sel].tolist())
            if len(gedges):
                emask = np.array([int(u) in keep and int(v) in keep
                                  for u, v in gedges], bool)
                gedges = gedges[emask]
            # window-scaled n_total so summarise can apply the official
            # over-detection adjustment on windows (cells/frame ~ constant)
            n_win = n_total * W / T if n_total else None
            self.gt_win[sid] = (gids[sel], gt[sel] - t0, gzyx[sel], gedges, n_win)
        # dl_195: daughter-channel validation.
        # The picked windows hold only a handful of division events per fold,
        # so every val sample's daughter-birth frames are APPENDED as div-only
        # items; pl_module keeps them out of the edge metric via tw = -1.
        self.div_on = float(cfg.loss.get("div_weight", 0.0)) > 0
        if self.div_on:
            drop = load_anomaly_links(cfg)
            self.div_cls = {sid: build_div_classes(store, sid, drop.get(sid))
                            for sid in ids}
            in_win = set(self.items)
            self.items += sorted(
                (sid, t) for sid in ids
                for t, c in self.div_cls[sid].items()
                if (c == DIV_POS).any() and (sid, t) not in in_win)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, t = self.items[i]
        x = self.store.frame(sid, t)
        nodes = self.store.nodes_by_t[sid].get(t, np.zeros((0, 3), np.int32))
        iso = (self.ds > 1).any()
        nodes_ds = np.rint(to_ds(nodes, self.ds)).astype(int) if iso else nodes
        hm = gaussian_heatmap(x.shape, nodes_ds, self.sigma_vox)
        dets = self.store.dets(sid, t) if self.mask_mode == "carved" else None
        mask = build_mask(x.shape, hm, self.mask_mode, dets, np.zeros(3, int),
                          self.carve_rad_vox, nodes, self.core_thresh, self.dup_um,
                          ds=self.ds if iso else None, carve_weight=self.carve_weight)
        w0 = self.window.get(sid)
        tw = t - w0 if w0 is not None and 0 <= t - w0 < self.W else -1
        item = dict(
            image=to_model_input(x, self.local_contrast),
            target=torch.from_numpy(hm)[None],
            mask=torch.from_numpy(mask),
            gt_zyx=torch.from_numpy(nodes.astype(np.float32)),
            sid=sid,
            tw=tw,
        )
        if self.div_on:
            cls = self.div_cls[sid].get(t, np.zeros(0, np.int8))
            item["div_cls"] = torch.from_numpy(cls.astype(np.int64))
        return item


class CellTrainDatasetPair(CellTrainDataset):
    """Two-frame item for the pair model (model.temporal=pair).

    Frames t / t+1 share the z-crop window and are stacked ALONG Z through one
    aug pipeline call, guaranteeing identical 2D transform parameters for both
    frames (the pipeline applies YX transforms slice-wise with shared params);
    they are split and YX-cropped identically afterwards.

    Returned tensors:
      image  (2, Z, Y, X)  frames [t, t+1] (t+1 is reference-only: the model
                           cross-attends to it, predictions are for t alone)
      target (4, Z, Y, X)  [hm_t, flow dz, dy, dx]
      mask   (2, Z, Y, X)  [carved mask_t, flow weight]
    Flow = transformed(t+1 node) - transformed(t node) for GT-linked pairs, in
    output-grid voxels (rotation/flip/scale-consistent because both endpoints
    go through the pipeline as keypoints); supervised on the <=1.5 sigma core
    around the frame-t node with the Gaussian value as weight (render_flow).
    """

    def __init__(self, cfg, store, ids):
        super().__init__(cfg, store, ids)
        # a pair item needs a successor frame
        for sid in list(self.frames):
            ts = [t for t in self.frames[sid] if t + 1 < store.shape[sid][0]]
            if ts:
                self.frames[sid] = ts
            else:
                del self.frames[sid]
        self.items = [sid for sid in self.items if sid in self.frames]
        drop = self.anomaly_drop = load_anomaly_links(cfg)
        self.links = {sid: build_links(store, sid, drop.get(sid))
                      for sid in self.items}
        self.flow_scale = float(cfg.loss.get("flow_scale_vox", 10.0))
        # Division oversampling (data.div_oversample): GT holds 151 divisions
        # across the 199 videos, so uniform frame sampling puts them at the
        # noise floor of the InfoNCE rows — and the matcher measurably cannot
        # place a second daughter (dl_136: reverse probability p50 0.892 for
        # the daughter it links vs 0.065 for the one it drops). div_window
        # widens the pick to the frames around the event so the model also
        # sees the mother just before and the daughters just after.
        self.div_p = float(cfg.data.get("div_oversample", 0.0))
        win = int(cfg.data.get("div_window", 1))
        self.div_frames = {}
        for sid in self.items:
            valid = set(self.frames[sid])
            picks = set()
            for t, ls in self.links[sid].items():
                srcs = [a for a, _ in ls]
                if len(srcs) == len(set(srcs)):
                    continue                       # no source with two children
                picks |= {tt for tt in range(t - win, t + win + 1) if tt in valid}
            if picks:
                self.div_frames[sid] = sorted(picks)

    def __getitem__(self, i):
        sid = self.items[i]
        ts = self.frames[sid]
        dts = self.div_frames.get(sid)
        if dts and np.random.rand() < self.div_p:
            t = dts[np.random.randint(len(dts))]
        else:
            t = ts[np.random.randint(len(ts))]
        fa = self.store.frame(sid, t)
        fb = self.store.frame(sid, t + 1)
        na = self.store.nodes_by_t[sid][t]
        nb = self.store.nodes_by_t[sid].get(t + 1, np.zeros((0, 3), np.int32))
        links = self.links[sid].get(t, [])

        ci = np.random.randint(len(na))
        jitter = np.random.randint(-self.jitter, self.jitter + 1)
        na_ds = to_ds(na, self.ds)
        assert fa.shape[0] >= self.crop[0], (
            f"crop_size z {self.crop[0]} > frame z {fa.shape[0]} "
            f"(shrink data.crop_size for data.downsample={self.ds.tolist()})")
        cz = self.crop[0]
        z0 = int(np.clip(int(np.rint(na_ds[ci, 0])) + jitter[0] - cz // 2,
                         0, fa.shape[0] - cz))
        x2 = np.concatenate([fa[z0:z0 + cz], fb[z0:z0 + cz]], axis=0)  # (2cz,Y,X)

        ua = np.zeros((0, 3))
        if self.mask_mode == "carved":
            det, matched = self.store.dets(sid, t)
            ua = filter_unmatched(det, matched, na, self.dup_um)

        self._reseed_aug()
        pts = to_ds(np.concatenate([na.astype(np.float64),
                                    nb.astype(np.float64), ua], 0), self.ds)
        out = self.aug(volume=x2, keypoints=pts[:, [2, 1]],
                       mask=np.ones(x2.shape[1:], np.float32))
        xt = out["volume"]
        kp = np.asarray(out["keypoints"], np.float64)
        valid = out["mask"] > 0.99
        g = next((p["scale"]["x"] for n, p in out["applied_transforms"]
                  if n == "Affine"), 1.0)

        _, cy, cx = self.crop
        hw = np.array(xt.shape[1:])
        assert (hw >= (cy, cx)).all(), (
            f"crop_size yx {(cy, cx)} > canvas {tuple(hw)} "
            f"(shrink data.crop_size for data.downsample={self.ds.tolist()})")
        oy, ox = np.clip(np.rint([kp[ci, 1], kp[ci, 0]]).astype(int)
                         + jitter[1:] - [cy // 2, cx // 2], 0, hw - [cy, cx])
        xa = np.ascontiguousarray(xt[:cz, oy:oy + cy, ox:ox + cx])
        xb = np.ascontiguousarray(xt[cz:, oy:oy + cy, ox:ox + cx])
        # output-space float positions (z is untransformed; YX from keypoints)
        posf = np.stack([pts[:, 0] - z0, kp[:, 1] - oy, kp[:, 0] - ox], 1)
        zyx = np.rint(posf).astype(int)
        n_a, n_b = len(na), len(nb)
        la = zyx[:n_a]
        da = zyx[n_a + n_b:]

        shape = xa.shape
        hm_a = gaussian_heatmap(shape, la, self.sigma_vox)
        rad = self.carve_rad_vox * np.array([1.0, g, g])
        mask_a = build_mask(shape, hm_a, self.mask_mode,
                            (da, np.zeros(len(da), bool)), np.zeros(3, int),
                            rad, None, self.core_thresh, self.dup_um,
                            carve_weight=self.carve_weight)
        v = valid[oy:oy + cy, ox:ox + cx]
        mask_a = mask_a * v

        flow = np.zeros((3,) + shape, np.float32)
        fw = np.zeros(shape, np.float32)
        if links:
            si = np.array([a for a, _ in links])
            ti = np.array([b for _, b in links])
            d = posf[n_a + ti] - posf[si]                 # (L,3) dz,dy,dx float
            flow, fw = render_flow(shape, la[si], d / self.flow_scale,
                                   self.sigma_vox)
        fw = fw * v  # drop affine border fill from the flow loss too

        return dict(
            image=torch.from_numpy(np.stack([xa, xb])),
            target=torch.from_numpy(np.stack([hm_a, *flow])),
            mask=torch.from_numpy(np.stack([mask_a, fw])),
        )


class CellTrainDatasetSiamese(CellTrainDatasetPair):
    """Two-frame item for the siamese embedding model (model.temporal=siamese).

    Same shared-z-crop / shared-aug construction as the pair dataset, but the
    two frames are symmetric (both are encoded independently and both get
    detection supervision — frame t+1's carved mask uses its own det cache).
    No flow targets; instead the item carries what the embedding loss needs:
    transformed GT node positions of both frames (float, crop coords, possibly
    outside the crop) and the GT link index pairs.

    Returned tensors (collated by siamese_collate — pa/pb/links are variable
    length and stay per-item lists):
      image  (2, Z, Y, X)   frames [t, t+1]
      target (2, Z, Y, X)   [hm_t, hm_t+1]
      mask   (2, Z, Y, X)   [carved mask_t, carved mask_t+1] (bool)
      pa     (Na, 3) float  frame-t GT nodes, transformed crop coords
      pb     (Nb, 3) float  frame-t+1 GT nodes
      links  (L, 2) int     (row in pa, row in pb) GT links t -> t+1
      det_a  (Ma, 3) float  champion detections in the crop (frame t; empty
      det_b  (Mb, 3) float  unless data.champion_det_dir is set) — anchors/
                            candidates for the dedicated embedding model
                            (loss.embed_only): train/inference-consistent
                            points incl. FPs as real negatives. NOTE: ideally
                            these would be on-the-fly predictions under the
                            same aug; the cache is the v1 approximation.
    """

    def __init__(self, cfg, store, ids):
        super().__init__(cfg, store, ids)
        self.champ_dir = cfg.data.get("champion_det_dir")
        self._champ = {}   # sid -> open NpzFile (per worker, lazy)
        fc = cfg.data.get("flow_cache_dir")
        self.flow_cache = FlowCache(fc) if fc else None
        # embedding-only training never consumes target/mask — skip the
        # heatmap render + carve loops (they dominated worker CPU at crop 1.5x)
        self.embed_only = bool(cfg.loss.get("embed_only", False))
        # matcher node-cls loss (loss.matcher_cls_weight): per-GT-node
        # other/daughter/division-parent classes, nodes_by_t row order
        self.node_cls = None
        if float(cfg.loss.get("matcher_cls_weight", 0.0)) > 0:
            drop = self.anomaly_drop   # same mask the parent used for links
            self.node_cls = {sid: build_div_classes(store, sid, drop.get(sid),
                                                    parent_class=True)
                             for sid in self.items}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_champ"] = {}   # zip handles are re-opened per worker
        return state

    def champ_dets(self, sid, t):
        z = self._champ.get(sid)
        if z is None:
            z = np.load(Path(self.champ_dir) / f"{sid}.npz")
            self._champ[sid] = z
        key = f"t{t:04d}"
        return z[key] if key in z.files else np.zeros((0, 3), np.int32)

    def __getitem__(self, i):
        sid = self.items[i]
        ts = self.frames[sid]
        t = ts[np.random.randint(len(ts))]
        fa = self.store.frame(sid, t)
        fb = self.store.frame(sid, t + 1)
        na = self.store.nodes_by_t[sid][t]
        nb = self.store.nodes_by_t[sid].get(t + 1, np.zeros((0, 3), np.int32))
        links = self.links[sid].get(t, [])
        ca = cb = np.zeros((0, 3))
        if self.champ_dir:
            ca = self.champ_dets(sid, t).astype(np.float64)
            cb = self.champ_dets(sid, t + 1).astype(np.float64)

        ci = np.random.randint(len(na))
        jitter = np.random.randint(-self.jitter, self.jitter + 1)
        na_ds = to_ds(na, self.ds)
        assert fa.shape[0] >= self.crop[0], (
            f"crop_size z {self.crop[0]} > frame z {fa.shape[0]} "
            f"(shrink data.crop_size for data.downsample={self.ds.tolist()})")
        cz = self.crop[0]
        z0 = int(np.clip(int(np.rint(na_ds[ci, 0])) + jitter[0] - cz // 2,
                         0, fa.shape[0] - cz))
        x2 = np.concatenate([fa[z0:z0 + cz], fb[z0:z0 + cz]], axis=0)  # (2cz,Y,X)

        ua = np.zeros((0, 3))
        ub = np.zeros((0, 3))
        if self.mask_mode == "carved" and not self.embed_only:
            det, matched = self.store.dets(sid, t)
            ua = filter_unmatched(det, matched, na, self.dup_um)
            det_b, matched_b = self.store.dets(sid, t + 1)
            ub = filter_unmatched(det_b, matched_b, nb, self.dup_um)

        # flow partner points (data.flow_cache_dir): each frame-a node p gets a
        # companion keypoint p+f riding the SAME spatial transform; the per-node
        # offset is recovered afterwards as posf[partner]-posf[p], which keeps
        # the vector consistent under D4/affine without component bookkeeping
        # (z is untouched by the yx-only aug and cancels through z0)
        flow_src = None
        if self.flow_cache is not None:
            src = np.concatenate([na.astype(np.float64), ca], 0)
            f_um = self.flow_cache.sample_um(sid, t, src)
            flow_src = f_um / (SCALE * self.ds)   # data-ds-grid voxels

        self._reseed_aug()
        pts = to_ds(np.concatenate([na.astype(np.float64),
                                    nb.astype(np.float64), ua, ub, ca, cb], 0),
                    self.ds)
        if flow_src is not None:
            pts = np.concatenate([pts, to_ds(src, self.ds) + flow_src], 0)
        out = self.aug(volume=x2, keypoints=pts[:, [2, 1]],
                       mask=np.ones(x2.shape[1:], np.float32))
        xt = out["volume"]
        kp = np.asarray(out["keypoints"], np.float64)
        valid = out["mask"] > 0.99
        g = next((p["scale"]["x"] for n, p in out["applied_transforms"]
                  if n == "Affine"), 1.0)

        _, cy, cx = self.crop
        hw = np.array(xt.shape[1:])
        assert (hw >= (cy, cx)).all(), (
            f"crop_size yx {(cy, cx)} > canvas {tuple(hw)} "
            f"(shrink data.crop_size for data.downsample={self.ds.tolist()})")
        oy, ox = np.clip(np.rint([kp[ci, 1], kp[ci, 0]]).astype(int)
                         + jitter[1:] - [cy // 2, cx // 2], 0, hw - [cy, cx])
        xa = np.ascontiguousarray(xt[:cz, oy:oy + cy, ox:ox + cx])
        xb = np.ascontiguousarray(xt[cz:, oy:oy + cy, ox:ox + cx])
        posf = np.stack([pts[:, 0] - z0, kp[:, 1] - oy, kp[:, 0] - ox], 1)
        zyx = np.rint(posf).astype(int)
        n_a, n_b, n_ua, n_ub = len(na), len(nb), len(ua), len(ub)
        la, lb = zyx[:n_a], zyx[n_a:n_a + n_b]
        da = zyx[n_a + n_b:n_a + n_b + n_ua]
        db = zyx[n_a + n_b + n_ua:n_a + n_b + n_ua + n_ub]
        o = n_a + n_b + n_ua + n_ub
        crop_hi = np.array([cz, cy, cx], np.float64)
        o2 = o + len(ca) + len(cb)
        det_a = posf[o:o + len(ca)]
        det_b = posf[o + len(ca):o2]
        keep_a = ((det_a >= 0) & (det_a < crop_hi)).all(1)
        det_a = det_a[keep_a]
        det_b = det_b[((det_b >= 0) & (det_b < crop_hi)).all(1)]
        flow_pa = flow_det_a = None
        if flow_src is not None:
            part = posf[o2:]
            um_scale = SCALE * self.ds
            flow_pa = (part[:n_a] - posf[:n_a]) * um_scale
            flow_det_a = ((part[n_a:] - posf[o:o + len(ca)]) * um_scale)[keep_a]

        shape = xa.shape
        if self.embed_only:
            hm_a = hm_b = np.zeros((1, 1, 1), np.float32)
            mask_a = mask_b = np.zeros((1, 1, 1), bool)
        else:
            rad = self.carve_rad_vox * np.array([1.0, g, g])
            v = valid[oy:oy + cy, ox:ox + cx]
            hm_a = gaussian_heatmap(shape, la, self.sigma_vox)
            mask_a = build_mask(shape, hm_a, self.mask_mode,
                                (da, np.zeros(len(da), bool)), np.zeros(3, int),
                                rad, None, self.core_thresh, self.dup_um,
                                carve_weight=self.carve_weight) * v
            hm_b = gaussian_heatmap(shape, lb, self.sigma_vox)
            mask_b = build_mask(shape, hm_b, self.mask_mode,
                                (db, np.zeros(len(db), bool)), np.zeros(3, int),
                                rad, None, self.core_thresh, self.dup_um,
                                carve_weight=self.carve_weight) * v

        item = dict(
            image=torch.from_numpy(np.stack([xa, xb])),
            target=torch.from_numpy(np.stack([hm_a, hm_b])),
            mask=torch.from_numpy(np.stack([mask_a, mask_b])),
            pa=torch.from_numpy(posf[:n_a].astype(np.float32)),
            pb=torch.from_numpy(posf[n_a:n_a + n_b].astype(np.float32)),
            links=torch.tensor(links, dtype=torch.int64).reshape(-1, 2),
            det_a=torch.from_numpy(det_a.astype(np.float32)),
            det_b=torch.from_numpy(det_b.astype(np.float32)),
        )
        if flow_pa is not None:
            item["flow_pa"] = torch.from_numpy(flow_pa.astype(np.float32))
            item["flow_det_a"] = torch.from_numpy(flow_det_a.astype(np.float32))
        if self.node_cls is not None:
            nc = self.node_cls[sid]
            item["cls_a"] = torch.from_numpy(
                nc.get(t, np.zeros(0, np.int8)).astype(np.int64))
            item["cls_b"] = torch.from_numpy(
                nc.get(t + 1, np.zeros(0, np.int8)).astype(np.int64))
        return item


def siamese_collate(batch):
    """Stack fixed-shape keys; keep variable-length pa/pb/links as lists."""
    out = {k: torch.stack([b[k] for b in batch])
           for k in ("image", "target", "mask") if k in batch[0]}
    for k in ("pa", "pb", "links", "det_a", "det_b"):
        out[k] = [b[k] for b in batch]
    for k in ("cls_a", "cls_b", "flow_pa", "flow_det_a"):   # optional keys
        if k in batch[0]:
            out[k] = [b[k] for b in batch]
    return out


class CellValDatasetSiameseEmbed(Dataset):
    """Embedding-quality validation for the dedicated embedding model
    (loss.embed_only): DETERMINISTIC full-frame (t, t+1) pairs from val
    samples, champion detections as candidates. No aug, no crop, no positive
    drop (pl_module passes drop_p=0) — measures val_emb_acc for overfit
    monitoring and checkpoint selection.

    Items: data.val_max_samples evenly-spaced val samples x
    data.val_embed_pairs (t, t+1) pairs evenly spaced over annotated frames.
    Batch via siamese_collate (variable-length keys), batch_size 1.
    """

    def __init__(self, cfg, store, ids):
        self.store = store
        self.champ_dir = cfg.data.get("champion_det_dir")
        assert self.champ_dir, "embed val needs data.champion_det_dir"
        self.ds = get_downsample(cfg)   # frames are ds-pooled; GT/champion
        self._champ = {}                # coords arrive in ORIGINAL voxels
        fc = cfg.data.get("flow_cache_dir")
        self.flow_cache = FlowCache(fc) if fc else None
        n = min(int(cfg.data.val_max_samples), len(ids))
        k = int(cfg.data.get("val_embed_pairs", 4))
        pick = [ids[j] for j in np.linspace(0, len(ids) - 1, n).astype(int)]
        self.items = []
        self.links = {}
        # matcher node-cls val (no anomaly drop, matching this dataset's links)
        want_cls = float(cfg.loss.get("matcher_cls_weight", 0.0)) > 0
        self.node_cls = {} if want_cls else None
        for sid in pick:
            T = store.shape[sid][0]
            ts = [t for t in sorted(store.nodes_by_t[sid]) if t + 1 < T]
            if not ts:
                continue
            self.links[sid] = build_links(store, sid)
            if want_cls:
                self.node_cls[sid] = build_div_classes(store, sid,
                                                       parent_class=True)
            for j in np.linspace(0, len(ts) - 1, min(k, len(ts))).astype(int):
                self.items.append((sid, int(ts[j])))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_champ"] = {}
        return state

    def champ_dets(self, sid, t):
        z = self._champ.get(sid)
        if z is None:
            z = np.load(Path(self.champ_dir) / f"{sid}.npz")
            self._champ[sid] = z
        key = f"t{t:04d}"
        return z[key] if key in z.files else np.zeros((0, 3), np.int32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, t = self.items[i]
        xa = self.store.frame(sid, t)
        xb = self.store.frame(sid, t + 1)
        na = self.store.nodes_by_t[sid][t]
        nb = self.store.nodes_by_t[sid].get(t + 1, np.zeros((0, 3), np.int32))
        links = self.links[sid].get(t, [])
        item = dict(
            image=torch.from_numpy(np.stack([xa, xb])),
            pa=torch.from_numpy(to_ds(na, self.ds).astype(np.float32)),
            pb=torch.from_numpy(to_ds(nb, self.ds).astype(np.float32)),
            links=torch.tensor(links, dtype=torch.int64).reshape(-1, 2),
            det_a=torch.from_numpy(
                to_ds(self.champ_dets(sid, t), self.ds).astype(np.float32)),
            det_b=torch.from_numpy(
                to_ds(self.champ_dets(sid, t + 1), self.ds).astype(np.float32)),
        )
        if self.flow_cache is not None:
            item["flow_pa"] = torch.from_numpy(
                self.flow_cache.sample_um(sid, t, na.astype(np.float64)))
            item["flow_det_a"] = torch.from_numpy(self.flow_cache.sample_um(
                sid, t, self.champ_dets(sid, t).astype(np.float64)))
        if self.node_cls is not None:
            nc = self.node_cls[sid]
            item["cls_a"] = torch.from_numpy(
                nc.get(t, np.zeros(0, np.int8)).astype(np.int64))
            item["cls_b"] = torch.from_numpy(
                nc.get(t + 1, np.zeros(0, np.int8)).astype(np.int64))
        return item


class CellValDatasetPair(CellValDataset):
    """Val item for the pair model: frames (t, min(t+1, T-1)) stacked, targets
    and masks in the train pair layout [hm_t, flow] (no aug; the last frame
    pairs with itself, where flow supervision is simply absent). gt_zyx / sid
    / tw are unchanged -- frame-t detection is scored from output channel 0."""

    def __init__(self, cfg, store, ids):
        super().__init__(cfg, store, ids)
        sids = {sid for sid, _ in self.items}
        self.links = {sid: build_links(store, sid) for sid in sids}
        self.flow_scale = float(cfg.loss.get("flow_scale_vox", 10.0))

    def __getitem__(self, i):
        sid, t = self.items[i]
        t1 = min(t + 1, self.store.shape[sid][0] - 1)
        xa = self.store.frame(sid, t)
        xb = self.store.frame(sid, t1)
        na = self.store.nodes_by_t[sid].get(t, np.zeros((0, 3), np.int32))
        nb = self.store.nodes_by_t[sid].get(t1, np.zeros((0, 3), np.int32))
        iso = (self.ds > 1).any()
        na_ds = np.rint(to_ds(na, self.ds)).astype(int) if iso else na
        shape = xa.shape
        hm_a = gaussian_heatmap(shape, na_ds, self.sigma_vox)
        ds_arg = self.ds if iso else None
        det_a = self.store.dets(sid, t) if self.mask_mode == "carved" else None
        mask_a = build_mask(shape, hm_a, self.mask_mode, det_a, np.zeros(3, int),
                            self.carve_rad_vox, na, self.core_thresh,
                            self.dup_um, ds=ds_arg, carve_weight=self.carve_weight)

        flow = np.zeros((3,) + shape, np.float32)
        fw = np.zeros(shape, np.float32)
        links = self.links[sid].get(t, []) if t1 == t + 1 else []
        if links:
            si = np.array([a for a, _ in links])
            ti = np.array([b for _, b in links])
            d = to_ds(nb, self.ds)[ti] - to_ds(na, self.ds)[si]   # (L,3)
            flow, fw = render_flow(shape, na_ds[si], d / self.flow_scale,
                                   self.sigma_vox)

        return dict(
            image=torch.from_numpy(np.stack([xa, xb])),
            target=torch.from_numpy(np.stack([hm_a, *flow])),
            mask=torch.from_numpy(np.stack([mask_a, fw])),
            gt_zyx=torch.from_numpy(na.astype(np.float32)),
            sid=sid,
            tw=t - self.window[sid],
        )
