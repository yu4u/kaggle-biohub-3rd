"""Full-volume 3-frame items with labelled detection points of the centre frame.

Labels follow build_div_classes(parent_class=True) assigned to champion
detections by the <=7 um Hungarian:
DIV_PAR -> 1, DIV_NEG / DIV_POS -> 0, DIV_IGN excluded. Unmatched detections
become weak negatives unless within 7 um of a POS/PAR GT node (embed_loss's
matcher_cls_det_neg rule); the loss applies their down-weight, not the item.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from biohub046.core.dataset import (DIV_IGN, DIV_NEG, DIV_PAR, DIV_POS, SCALE,
                         build_div_classes, load_anomaly_links, to_ds)
from biohub046.core.metric import MAX_MATCH_UM, hungarian_pairs


REVIEWED_NEG = 3  # confirmed pseudo negative; retained through spatial augmentations


def frame_times(t, length, n_frames=3, frame_offset=0):
    """Choose input images without moving the labeled node's time or position."""
    if n_frames < 1 or n_frames % 2 != 1:
        raise ValueError('n_frames must be a positive odd number')
    if not isinstance(frame_offset, (int, np.integer)):
        raise ValueError('frame_offset must be an integer')
    half = n_frames // 2
    return tuple(min(max(t + frame_offset + d, 0), length - 1)
                 for d in range(-half, half + 1))


def pool_heatmap(hm, ds):
    """One (Z, Y, X) frame of a cached 1x det heatmap (f16) onto the ds grid: the same remainder-drop + block
    mean as CellStore.frame, no normalization (raw regression output)."""
    f = np.asarray(hm, np.float32)
    dz, dy, dx = (int(v) for v in ds)
    Z, Y, X = f.shape
    f = f[:Z // dz * dz, :Y // dy * dy, :X // dx * dx]
    return f.reshape(Z // dz, dz, Y // dy, dy, X // dx, dx).mean(axis=(1, 3, 5))


class FlowWarp:
    """Backward-warp the t-1 / t+1 frames onto frame t with the dense flow
    cache (<sid>.npy = (T-1, 3, Zf, Yf, Xf)
    f16, flow(t -> t+1) of boundary t in FLOW-grid voxels on the fixed [1,4,4]
    grid). A voxel v of frame t (ds grid) samples the next frame at v + f_t(v)
    and the previous frame at v - f_{t-1}(v): the cache is forward-only, so
    the previous frame uses the t-1 -> t field evaluated at the destination
    instead of at its source (exact up to the field's variation over one
    displacement, ~1-2 um on a field smooth over tens of um)."""

    FLOW_DS = np.array([1, 4, 4], np.float64)

    def __init__(self, root, ds, device="cpu", keep_raw=False):
        self.root, self.ds = Path(root), np.asarray(ds, np.float64)
        # div_762: also keep the raw neighbours -> 5 inputs
        # [raw t-1, warped t-1, t, warped t+1, raw t+1] (model.n_frames=5)
        self.keep_raw = bool(keep_raw)
        self.device = torch.device(device)
        self._mm = {}    # sid -> memmap (per worker, lazy)
        self._grid = {}  # (Z, Y, X) -> (ds-grid coords (Z,Y,X,3), flow-grid coords)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = {}
        return state

    def _coords(self, shape):
        g = self._grid.get(shape)
        if g is None:
            v = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float64) for n in shape],
                                           indexing="ij"), -1)
            orig = v * torch.from_numpy(self.ds) + torch.from_numpy((self.ds - 1) / 2)
            fg = (orig - torch.from_numpy((self.FLOW_DS - 1) / 2)) / torch.from_numpy(self.FLOW_DS)
            g = (v.float().to(self.device), fg.float().to(self.device))
            self._grid[shape] = g
        return g

    def warp(self, sid, t, frames, times):
        """frames (n, Z, Y, X) at `times` around centre t -> the same with every
        frame at t-1 / t+1 warped onto t (replicated boundary frames = t stay)."""
        from biohub046.ssl_flow.ops import sample_volume
        mm = self._mm.get(sid)
        if mm is None:
            mm = np.load(self.root / f"{sid}.npy", mmap_mode="r")
            # the cache must sit on the assumed FLOW_DS grid of the same volume:
            # a cache built on another grid would warp silently misaligned
            expect = tuple(int(n) * int(d) // int(f) for n, d, f in
                           zip(frames.shape[-3:], self.ds, self.FLOW_DS))
            if tuple(mm.shape[-3:]) != expect or mm.shape[1] != 3:
                raise ValueError(f"{self.root / sid}.npy: flow grid {mm.shape} does not match "
                                 f"FLOW_DS {self.FLOW_DS.tolist()} for frames {frames.shape[-3:]} "
                                 f"at ds {self.ds.tolist()} (expected (T-1, 3, *{expect}))")
            self._mm[sid] = mm
        out = frames.copy()
        v, fg = self._coords(frames.shape[-3:])
        for i, u in enumerate(times):
            if u == t:
                continue
            assert abs(u - t) == 1, "warp supports the t-1 / t+1 neighbours only"
            b = t if u > t else t - 1
            flow = torch.from_numpy(np.asarray(mm[b], np.float32))[None].to(self.device)
            d = sample_volume(flow, fg.reshape(-1, 3))[0]          # flow-grid voxels
            d = d * torch.from_numpy(self.FLOW_DS / self.ds).float().to(self.device)  # -> ds-grid voxels
            src = (v.reshape(-1, 3) + (d if u > t else -d)).reshape(*v.shape)
            vol = torch.from_numpy(np.ascontiguousarray(frames[i], np.float32))[None, None].to(self.device)
            out[i] = sample_volume(vol, src.reshape(-1, 3))[0, :, 0].reshape(v.shape[:3]).cpu().numpy()
        if self.keep_raw:
            assert len(times) == 3
            return np.stack([frames[0], out[0], out[1], out[2], frames[2]])
        return out


def apply_cube(vol, perm, flips):
    """One of the 48 cube symmetries (signed axis permutation) on the last
    three axes of ``vol``. "Pick one of 6 faces up, then D4 on it" is exactly
    the full symmetry group of the cube (6 x 8 = 48), which is the group of
    signed permutations: permute the spatial axes, then flip a subset. The ds
    [1,4,4] grid is ~1.625 um isotropic, so treating all axes alike is
    geometrically consistent."""
    nd = vol.ndim
    vol = np.transpose(vol, tuple(range(nd - 3)) + tuple(nd - 3 + int(p) for p in perm))
    for a in range(3):
        if flips[a]:
            vol = np.flip(vol, a - 3)
    return np.ascontiguousarray(vol)


def apply_cube_points(pts, perm, flips, size):
    """Same signed permutation on (N, 3) zyx grid coords (cubic volume)."""
    q = pts[:, perm].astype(np.float64, copy=True)
    s = size - 1
    for a in range(3):
        if flips[a]:
            q[:, a] = s - q[:, a]
    return q


def apply_floor(frames, rng, s):
    """dl_165's local floor/signal independent gain fields for the 3-frame
    stack (div_328): decompose each frame with the p20 local floor and rescale
    floor and signal with two SMOOTH random gain fields (amplitudes 1.5s /
    0.5s, log-uniform around 1). div_327: the div3f misses are DIM parents
    (median brightness 0.43 vs 0.59) — the same mechanism dl_155 measured for
    the detector. The gain fields are drawn ONCE and shared by all three
    frames (an independent draw would fabricate temporal change = a fake
    division cue); the floor estimate stays per-frame."""
    import cv2
    from biohub046.core.dataset import local_background
    h, w = frames.shape[-2:]
    gains = []
    for a in (1.5 * s, 0.5 * s):
        c = rng.uniform(-1.0, 1.0, (4, 4)).astype(np.float32)
        f = cv2.resize(c, (w, h), interpolation=cv2.INTER_CUBIC)
        gains.append(np.exp(f * np.float32(np.log1p(a)))[None])
    out = np.empty_like(frames)
    for i in range(frames.shape[0]):
        bg = local_background(frames[i])
        out[i] = gains[0] * bg + gains[1] * (frames[i] - bg)
    return out


def apply_optics(frames, rng, s, syx_scale):
    """OpticsAug (core/dataset.py, the one aug the detector A/B kept: dl_079)
    with sigmas recalibrated to this grid: they are defined in det-grid voxels
    (0.40625 um yx), so the YX sigma shrinks by the ds factor (x0.25 on
    [1,4,4]); z is ds=1 and stays. One draw, applied to all three frames."""
    from scipy.ndimage import gaussian_filter
    u = rng.random()
    if u < 0.4:
        sz = rng.uniform(0.0, 0.8) * s
        syx = rng.uniform(0.0, 1.2) * s * syx_scale
        return gaussian_filter(frames, (0, sz, syx, syx))
    if u < 0.6:
        amount = np.float32(rng.uniform(0.2, 0.7) * s)
        syx = 1.0 * syx_scale
        return frames + amount * (frames - gaussian_filter(frames, (0, 0, syx, syx)))
    return frames


def apply_rot_scale(frames, pts, rng, s):
    """Detector's rot_scale_aug (core/dataset.py A.Affine) ported to full-volume
    items (div_331): one YX rotation (uniform 360 deg) + isotropic YX scale
    (1 +- s) about the frame centre. Drawn ONCE and shared by the 3 frames (an
    independent draw would fabricate motion = a fake division cue); z is
    untouched (anisotropic optics). Returns the transformed frames/points and
    the keep mask for points that left the canvas (crop_yx rule)."""
    import cv2
    H, W = frames.shape[-2:]
    ang = rng.uniform(0.0, 360.0)
    sc = rng.uniform(1.0 - s, 1.0 + s)
    M = cv2.getRotationMatrix2D(((W - 1) / 2.0, (H - 1) / 2.0), ang, sc)
    flat = frames.reshape(-1, H, W)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        cv2.warpAffine(flat[i], M, (W, H), dst=out[i],
                       flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    q = pts.astype(np.float64, copy=True)
    xy = q[:, [2, 1]] @ M[:, :2].T + M[:, 2]
    q[:, 2], q[:, 1] = xy[:, 0], xy[:, 1]
    keep = ((q[:, 1] >= 0) & (q[:, 1] <= H - 1)
            & (q[:, 2] >= 0) & (q[:, 2] <= W - 1))
    return out.reshape(frames.shape), q[keep].astype(np.float32), keep


def apply_d4(vol, k, flip, zflip=0):
    """rot90 x k + optional x-flip in the yx plane of a (..., Z, Y, X) volume,
    plus an optional z-flip. The detector avoids z-flip (orientation-consistent
    optics), but this model classifies already-detected cells, so
    the optical asymmetry is a nuisance variable rather than a detection cue."""
    vol = np.rot90(vol, k, axes=(-2, -1))
    if flip:
        vol = np.flip(vol, -1)
    if zflip:
        vol = np.flip(vol, -3)
    return np.ascontiguousarray(vol)


def apply_d4_points(pts, k, flip, shape, zflip=0):
    """Same D4 (+z-flip) element on (N, 3) zyx grid coords; shape = spatial
    (Z, Y, X) of the volume (Y == X for the rot90 to be shape-preserving).
    np.rot90 maps index i to S-1-i, whose continuous extension on the
    block-center grid is v -> S-1-v; one rot90 sends (y, x) -> (S-1-x, y)."""
    z, y, x = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()
    s = shape[-1] - 1
    for _ in range(k % 4):
        y, x = s - x, y
    if flip:
        x = s - x
    if zflip:
        z = (shape[0] - 1) - z
    return np.stack([z, y, x], 1)


def apply_shift(frames, pts, rng, pad):
    """div_506: whole-volume translation with border fill. Pad each spatial
    axis by pad[a] (constant 0, like apply_rot_scale) and take an equally
    sized window at a random offset in [-pad, +pad], so the content moves by
    `shift` and the vacated border is background. Drawn ONCE and shared by
    the n frames (an independent draw would fabricate motion). Points move
    with the content; the keep mask drops those that left the canvas
    (crop_yx rule). pad is (z, y, x) on the ds grid; 0 disables that axis."""
    pad = np.asarray(pad, int)
    shift = np.array([rng.integers(-p, p + 1) if p > 0 else 0 for p in pad])
    shape = np.asarray(frames.shape[-3:])
    padded = np.pad(frames, ((0, 0),) * (frames.ndim - 3) + tuple((p, p) for p in pad),
                    mode="constant", constant_values=0.0)
    z0, y0, x0 = pad - shift
    out = padded[..., z0:z0 + shape[0], y0:y0 + shape[1], x0:x0 + shape[2]]
    q = pts.astype(np.float32) + shift.astype(np.float32)
    keep = np.all((q >= 0) & (q <= shape - 1), axis=1)
    return np.ascontiguousarray(out), q[keep], keep, shift


class DivPointDataset(Dataset):
    """One item = (sid, t): frames (3, Z, Y, X) [t-1, t, t+1] (edge frames
    replicated at the boundaries) + variable-length labelled points."""

    def __init__(self, cfg, store, ids, train):
        self.cfg, self.store, self.train = cfg, store, bool(train)
        self.ds = np.array(list(cfg.data.downsample), int)
        self.champ_dir = Path(cfg.data.champion_det_dir)
        self._champ = {}   # sid -> open NpzFile (per worker, lazy)
        self._cache = {}   # (sid, t) -> ds frame, val only (num_workers=0)
        # det heatmap as a 2nd input plane per frame: train samples read the
        # fold's leak-free in-sample teacher (_f{fold}), val samples the OOF
        # cache (= what the detector gives at test time)
        hm_dir = cfg.data.get("det_heatmap_train_dir" if self.train
                              else "det_heatmap_val_dir")
        self.hm_dir = (Path(str(hm_dir).format(fold=int(cfg.data.fold_id)))
                       if hm_dir else None)
        self._hm = {}      # sid -> open memmap (per worker, lazy)
        self._hm_cache = {}  # (sid, t) -> ds heatmap frame, val only
        # div_759: t-1 / t+1 flow-warped onto t (dense flow cache)
        flow_dir = cfg.data.get("warp_flow_dir")
        # val runs in the main process (num_workers=0): warp on the GPU there
        keep_raw = bool(cfg.data.get("warp_keep_raw", False))
        self.warp = (FlowWarp(flow_dir, self.ds, "cuda" if not self.train and torch.cuda.is_available()
                              else "cpu", keep_raw) if flow_dir else None)
        if self.warp is not None:
            assert (int(cfg.model.get("n_frames", 3)) == (5 if keep_raw else 3)
                    and int(cfg.data.get("frame_offset", 0)) == 0 and self.hm_dir is None), (
                "warp_flow_dir: model.n_frames 3 (5 with warp_keep_raw), offset 0, no heatmap plane")
        self._warp_cache = {}  # (sid, t) -> warped (3, Z, Y, X) f16, val only
        drop = load_anomaly_links(cfg)
        gt_fill = self.train and bool(cfg.data.get("gt_fill", True))
        # (sid, t) -> (pts_ds (N,3) f32, label (N,) f32, weak (N,) bool, cls (N,) i8)
        self.points = {}
        target = str(cfg.data.get("target", "parent"))
        if target not in ("parent", "pre", "post"):
            raise ValueError("target must be parent, pre or post")
        self.n_par_missed = 0
        self.frame_items, self.div_items = [], []
        for sid in ids:
            # t2_daughter_neg: the dl_195 rule "daughter at t+2 = IGN" avoids
            # contradicting the POSITIVE t+1 daughter of that task; here the
            # t+1 daughter is itself a negative, so excluding t+2 was an
            # inherited inconsistency — it is a regular (hard) negative
            cls_map = build_div_classes(store, sid, drop.get(sid),
                                        parent_class=True, t2_daughter_neg=True)
            if target != "parent":
                from .phase import phase_labels
                column = 0 if target == "pre" else 1
                phase_gt = phase_labels(store.gt[sid], drop.get(sid))
                cls_map = {t: np.where(p[:, column] < 0, DIV_IGN,
                    np.where(p[:, column] == 1, DIV_PAR, DIV_NEG)).astype(np.int8)
                    for t, p in phase_gt.items()}
            for t, gt in sorted(store.nodes_by_t[sid].items()):
                cls_t = cls_map[t]
                det = self.champ_dets(sid, t).astype(np.float64)
                pairs = hungarian_pairs(det, gt.astype(np.float64))
                # gtp = the matched GT node position per point (own position
                # when there is none) for the data.gt_swap_p augmentation
                pts, gtp, lab, weak, cls = [], [], [], [], []
                for d, g in pairs:
                    c = int(cls_t[g])
                    if c == DIV_IGN:
                        continue
                    pts.append(det[d])
                    gtp.append(gt[g].astype(np.float64))
                    lab.append(1.0 if c == DIV_PAR else 0.0)
                    weak.append(False)
                    cls.append(c)
                matched_d = {d for d, _ in pairs}
                un = np.array([d for d in range(len(det)) if d not in matched_d])
                if len(un):
                    pr = np.flatnonzero((cls_t == DIV_POS) | (cls_t == DIV_PAR))
                    keep = np.ones(len(un), bool)
                    if len(pr):
                        d_um = np.linalg.norm((det[un][:, None] - gt[pr][None])
                                              * SCALE, axis=2)
                        keep = d_um.min(1) > MAX_MATCH_UM
                    for d in un[keep]:
                        pts.append(det[d])
                        gtp.append(det[d])
                        lab.append(0.0)
                        weak.append(True)
                        cls.append(DIV_NEG)
                matched_g = {g for _, g in pairs}
                for g in np.flatnonzero(cls_t == DIV_PAR):
                    if g not in matched_g:
                        self.n_par_missed += 1
                        if gt_fill:
                            pts.append(gt[g].astype(np.float64))
                            gtp.append(gt[g].astype(np.float64))
                            lab.append(1.0)
                            weak.append(False)
                            cls.append(DIV_PAR)
                if not pts:
                    continue
                self.points[(sid, t)] = (
                    to_ds(np.array(pts), self.ds).astype(np.float32),
                    to_ds(np.array(gtp), self.ds).astype(np.float32),
                    np.array(lab, np.float32), np.array(weak, bool),
                    np.array(cls, np.int8))
                self.frame_items.append((sid, t))
                if (self.points[(sid, t)][4] == DIV_PAR).any():
                    self.div_items.append((sid, t))
        # learning-curve ablation (div_372/373): keep a random pos_frac of the
        # parent events; the dropped events' parent points leave the loss
        # (not relabelled) and their frames stay as ordinary frames
        self.pseudo_parents = []
        pseudo_path = cfg.data.get("pseudo_parent_path")
        if self.train and pseudo_path:
            if target != "parent" or float(cfg.data.get("pos_frac", 1.0)) != 1:
                raise ValueError("pseudo parents require target=parent and pos_frac=1")
            self.add_pseudo_parents(pseudo_path, ids)
        self.reviewed_negatives = []
        negative_path = cfg.data.get("reviewed_negative_path")
        self.reviewed_negative_items = []
        if self.train and negative_path:
            if target != "parent":
                raise ValueError("reviewed negatives require target=parent")
            self.add_reviewed_negatives(negative_path, ids)
        if self.train:
            rate = float(cfg.data.get("reviewed_negative_oversample", 0.0))
            total = rate + float(cfg.data.get("div_oversample", 0.0)) + float(cfg.data.get("hard_neg_oversample", 0.0))
            if rate < 0 or total > 1 or (rate > 0 and not self.reviewed_negative_items):
                raise ValueError("invalid reviewed-negative sampling pool or probability")
        pos_frac = float(cfg.data.get("pos_frac", 1.0)) if self.train else 1.0
        if pos_frac < 1.0:
            rng = np.random.default_rng(int(cfg.seed))
            keep = set(rng.choice(len(self.div_items),
                                  int(round(pos_frac * len(self.div_items))),
                                  replace=False).tolist())
            for i, key in enumerate(self.div_items):
                if i in keep:
                    continue
                pts, gtp, lab, weak, cls = self.points[key]
                m = lab == 0
                self.points[key] = (pts[m], gtp[m], lab[m], weak[m], cls[m])
            print(f"pos_frac {pos_frac}: keeping {len(keep)} of "
                  f"{len(self.div_items)} parent events", flush=True)
            self.div_items = [key for i, key in enumerate(self.div_items)
                              if i in keep]
        # hard-negative frames (div_351, data.hard_neg_oversample): the
        # daughters' birth frame t+1 and the mother's frame t-1 of every parent
        # frame. Their DIV_POS / pre-division-mother points look like a
        # division but are labelled 0; without oversampling they only enter
        # via the uniform draw (~0.8% of items) and are averaged into the
        # ~100k other GT-matched negatives.
        self.hardneg_items = [(sid, tt) for sid, t in self.div_items
                              for tt in (t - 1, t + 1) if (sid, tt) in self.points]
        self.pseudo_sampling_probability = None
        if self.train and bool(cfg.data.get("pseudo_to_gt_sampling", False)):
            if not pseudo_path or target != "parent" or pos_frac != 1:
                raise ValueError("pseudo-to-GT sampling requires parent pseudo labels and pos_frac=1")
            if int(cfg.train.epochs) < 2 or float(cfg.data.div_oversample) <= 0:
                raise ValueError("pseudo-to-GT sampling requires at least two epochs and positive oversampling")
            horizon = int(cfg.data.get("pseudo_to_gt_epochs") or cfg.train.epochs)
            if not 2 <= horizon <= int(cfg.train.epochs):
                raise ValueError("pseudo-to-GT horizon must be between two and total epochs")
            self.gt_div_items = [k for k in self.div_items
                                 if ((self.points[k][2] == 1) & ~self.points[k][3]).any()]
            self.pseudo_div_items = [k for k in self.div_items
                                     if ((self.points[k][2] == 1) & self.points[k][3]).any()]
            if not self.gt_div_items or not self.pseudo_div_items:
                raise ValueError("pseudo-to-GT sampling requires nonempty GT and pseudo frame pools")
            self.set_epoch(1)
        if not self.train:
            stride = int(cfg.eval.get("neg_frame_stride", 4))
            div = set(self.div_items)
            others = [it for it in self.frame_items if it not in div]
            self.items = self.div_items + others[::stride]

    def set_epoch(self, epoch):
        """Set the positive-frame mixture before creating this epoch's workers."""
        if self.train and bool(self.cfg.data.get("pseudo_to_gt_sampling", False)):
            epochs = int(self.cfg.train.epochs)
            if not 1 <= epoch <= epochs:
                raise ValueError("epoch outside the configured training schedule")
            horizon = int(self.cfg.data.get("pseudo_to_gt_epochs") or epochs)
            self.pseudo_sampling_probability = max(0., (horizon - epoch) / (horizon - 1))

    def add_pseudo_parents(self, path, ids):
        """Promote reviewed unmatched detections; weak=True marks pseudo origin.

        Keep GT labels intact and exclude whole crop components touching val.
        Positive weak points retain provenance; they never enter weak-negative BCE.
        """
        manifest = json.loads(Path(path).read_text())
        folds, groups = manifest["folds"], manifest["overlap_groups"]
        if set(folds) != set(self.store.gt) or set(groups) != set(folds):
            raise ValueError("pseudo manifest universe differs from training store")
        fold = int(self.cfg.data.fold_id)
        forbidden = {groups[s] for s, f in folds.items() if f == fold}
        seen = set()
        for r in manifest["rows"]:
            sid, t = r["sid"], int(r["t"])
            if r["status"] != "accept":
                raise ValueError("unaccepted pseudo parent")
            if sid not in ids or groups[sid] in forbidden:
                continue
            key = (sid, t)
            pts, gtp, lab, weak, cls = self.points[key]
            pos = to_ds(np.array([r["det_xyz"]], float), self.ds)[0]
            matches = np.flatnonzero(np.linalg.norm(pts - pos, axis=1) < 1e-4)
            if len(matches) != 1:
                raise ValueError(f"pseudo detection missing or ambiguous: {key}")
            i = int(matches[0])
            if (sid, t, i) in seen or not weak[i] or lab[i] != 0:
                raise ValueError(f"pseudo parent must be an unused weak negative: {key}")
            seen.add((sid, t, i))
            lab, cls = lab.copy(), cls.copy()
            lab[i], cls[i] = 1, DIV_PAR
            self.points[key] = pts, gtp, lab, weak, cls
            self.pseudo_parents.append((sid, t, i))
            if key not in self.div_items:
                self.div_items.append(key)

    def add_reviewed_negatives(self, path, ids):
        """Supervise only accepted unmatched points outside validation components."""
        manifest = json.loads(Path(path).read_text())
        folds, groups = manifest["folds"], manifest["overlap_groups"]
        if set(folds) != set(self.store.gt) or set(groups) != set(folds):
            raise ValueError("reviewed-negative manifest universe differs from store")
        fold = int(self.cfg.data.fold_id)
        forbidden = {groups[s] for s, f in folds.items() if f == fold}
        seen = set()
        for row in manifest["rows"]:
            sid, t = row["sid"], int(row["t"])
            if row["status"] != "accept" or row["label"] != 0:
                raise ValueError("unaccepted reviewed negative")
            if sid not in ids or folds[sid] == fold or groups[sid] in forbidden:
                continue
            key = (sid, t)
            pts, gtp, lab, weak, cls = self.points[key]
            pos = to_ds(np.array([row["det_xyz"]], float), self.ds)[0]
            matches = np.flatnonzero(np.linalg.norm(pts - pos, axis=1) < 1e-4)
            if len(matches) != 1:
                raise ValueError(f"reviewed negative missing or ambiguous: {key}")
            i = int(matches[0])
            if (sid, t, i) in seen or not weak[i] or lab[i] != 0:
                raise ValueError(f"reviewed negative must be an unused weak negative: {key}")
            seen.add((sid, t, i))
            cls = cls.copy()
            cls[i] = REVIEWED_NEG
            self.points[key] = pts, gtp, lab, weak, cls
            self.reviewed_negatives.append(dict(sid=sid, t=t, point_index=i, case=row["case"]))
            if key not in self.reviewed_negative_items:
                self.reviewed_negative_items.append(key)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_champ"] = {}   # zip handles are re-opened per worker
        state["_hm"] = {}      # memmaps likewise
        return state

    def champ_dets(self, sid, t):
        z = self._champ.get(sid)
        if z is None:
            z = np.load(self.champ_dir / f"{sid}.npz")
            self._champ[sid] = z
        key = f"t{t:04d}"
        return z[key] if key in z.files else np.zeros((0, 3), np.int32)

    def _hm_frame(self, sid, t):
        f = self._hm_cache.get((sid, t))
        if f is not None:
            return f.astype(np.float32) if f.dtype == np.float16 else f
        a = self._hm.get(sid)
        if a is None:
            a = np.load(self.hm_dir / f"{sid}.npy", mmap_mode="r")
            self._hm[sid] = a
        f = pool_heatmap(a[t], self.ds)
        if not self.train:
            # same f16 rule as the image cache (_frame): full-res frames
            # would add another ~48 GB of f32
            self._hm_cache[(sid, t)] = (f.astype(np.float16)
                                        if f.nbytes > 8_000_000 else f)
        return f

    def _frame(self, sid, t):
        if self.train:
            return self.store.frame(sid, t)
        f = self._cache.get((sid, t))
        if f is None:
            f = self.store.frame(sid, t)
            # full-res frames (16 MB f32) would put the val cache at ~48 GB;
            # store those as f16 (values ~[0, 2] after quantile norm)
            self._cache[(sid, t)] = (f.astype(np.float16)
                                     if f.nbytes > 8_000_000 else f)
            return f
        return f.astype(np.float32) if f.dtype == np.float16 else f

    def __len__(self):
        return int(self.cfg.data.items_per_epoch) if self.train else len(self.items)

    def __getitem__(self, index):
        if self.train:
            worker = get_worker_info()
            rng = np.random.default_rng(
                (worker.seed if worker else int(self.cfg.seed)) + index)
            r = rng.random()
            p_div = float(self.cfg.data.div_oversample)
            p_hn = float(self.cfg.data.get("hard_neg_oversample", 0.0))
            if self.div_items and r < p_div:
                pool = self.div_items
                if self.pseudo_sampling_probability is not None:
                    pool = (self.pseudo_div_items if r < p_div * self.pseudo_sampling_probability
                            else self.gt_div_items)
                sid, t = pool[rng.integers(len(pool))]
            elif self.hardneg_items and r < p_div + p_hn:
                sid, t = self.hardneg_items[rng.integers(len(self.hardneg_items))]
            elif self.reviewed_negative_items and r < p_div + p_hn + float(
                    self.cfg.data.get("reviewed_negative_oversample", 0.0)):
                sid, t = self.reviewed_negative_items[rng.integers(len(self.reviewed_negative_items))]
            else:
                sid, t = self.frame_items[rng.integers(len(self.frame_items))]
        else:
            sid, t = self.items[index]
        T = self.store.shape[sid][0]
        # with warp the temporal window is always t-1..t+1 (n_frames 5 = raw + warped)
        tts = frame_times(t, T, 3 if self.warp is not None else int(self.cfg.model.get("n_frames", 3)),
                          self.cfg.data.get("frame_offset", 0))
        if self.warp is not None:
            frames = self._warp_cache.get((sid, t))
            if frames is None:
                frames = self.warp.warp(sid, t, np.stack([self._frame(sid, tt) for tt in tts]), tts)
                if not self.train:
                    self._warp_cache[(sid, t)] = frames.astype(np.float16)
            frames = frames.astype(np.float32)
        else:
            frames = np.stack([self._frame(sid, tt) for tt in tts])
        if self.hm_dir is not None:
            # (n, 2, Z, Y, X): spatial augs act on the last three axes of both
            # planes alike; intensity augs below touch plane 0 only
            frames = np.stack([frames, np.stack([self._hm_frame(sid, tt)
                                                 for tt in tts])], 1)
        pts, gtp, lab, weak, cls = self.points[(sid, t)]
        if self.train:
            swap_p = float(self.cfg.data.get("gt_swap_p", 0.0))
            if swap_p > 0:
                m = rng.random(len(pts)) < swap_p
                pts = np.where(m[:, None], gtp, pts)
                jit_um = float(self.cfg.data.get("gt_swap_jitter_um", 0.0))
                if jit_um > 0 and m.any():
                    # div_377: a swapped point does not sit exactly on the GT
                    # node but at a random offset within a jit_um-radius ball
                    # (uniform in volume, isotropic in µm) -> the position
                    # around the cell varies instead of collapsing onto the
                    # annotation. Points are on the ds grid: µm -> ds voxels
                    # per axis via SCALE * ds.
                    k = int(m.sum())
                    v = rng.normal(size=(k, 3))
                    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
                    v *= jit_um * rng.random((k, 1)) ** (1.0 / 3.0)
                    pts[m] = pts[m] + (v / (SCALE * self.ds)).astype(np.float32)
            crop = int(self.cfg.data.get("crop_yx", 0))
            if crop:
                # detector-style yx crop (z stays full): centre on a positive
                # when the frame has one (the div_oversample draw), else on a
                # random labelled point, + jitter; drop points outside
                pool = np.flatnonzero(lab == 1)
                if len(pool) == 0:
                    pool = np.arange(len(pts))
                c = pts[pool[rng.integers(len(pool))]]
                H, W = frames.shape[-2:]
                jy, jx = rng.integers(-(crop // 4), crop // 4 + 1, size=2)
                y0 = int(np.clip(round(float(c[1])) + jy - crop // 2, 0, H - crop))
                x0 = int(np.clip(round(float(c[2])) + jx - crop // 2, 0, W - crop))
                frames = frames[..., y0:y0 + crop, x0:x0 + crop]
                pts = pts - np.array([0.0, y0, x0], np.float32)
                keep = ((pts[:, 1] >= 0) & (pts[:, 1] < crop)
                        & (pts[:, 2] >= 0) & (pts[:, 2] < crop))
                pts, lab, weak, cls = pts[keep], lab[keep], weak[keep], cls[keep]
            pad = [int(v) for v in self.cfg.data.get("pad_jitter_zyx", [0, 0, 0])]
            if any(pad):
                frames, pts, keep, _ = apply_shift(frames, pts, rng, pad)
                lab, weak, cls = lab[keep], weak[keep], cls[keep]
            img = frames if self.hm_dir is None else frames[:, 0]
            floor_aug = float(self.cfg.data.get("floor_aug", 0.0))
            if floor_aug > 0:
                img = apply_floor(img, rng, floor_aug)
            optics_aug = float(self.cfg.data.get("optics_aug", 0.0))
            if optics_aug > 0:
                img = apply_optics(img, rng, optics_aug,
                                   1.0 / float(self.ds[1]))
            gain_aug = float(self.cfg.data.get("gain_aug", 0.0))
            if gain_aug > 0:
                img = img * np.float32(np.exp(rng.uniform(
                    -np.log1p(gain_aug), np.log1p(gain_aug))))
            if self.hm_dir is None:
                frames = img
            else:
                frames = np.stack([img, frames[:, 1]], 1)
            if bool(self.cfg.data.get("cube_aug", False)):
                perm = rng.permutation(3)
                flips = rng.integers(2, size=3)
                frames = apply_cube(frames, perm, flips)
                pts = apply_cube_points(pts, perm, flips, frames.shape[-1])
            elif bool(self.cfg.data.get("d4_aug", False)):
                k, flip = int(rng.integers(4)), int(rng.integers(2))
                zflip = (int(rng.integers(2))
                         if bool(self.cfg.data.get("zflip_aug", False)) else 0)
                frames = apply_d4(frames, k, flip, zflip)
                pts = apply_d4_points(pts, k, flip, frames.shape[-3:], zflip)
            rot_scale = float(self.cfg.data.get("rot_scale_aug", 0.0))
            # div_444: rotation only (scale fixed at 1) — the scale part of
            # div_441 flattened the score scale
            if rot_scale > 0 or bool(self.cfg.data.get("rot_aug", False)):
                frames, pts, keep = apply_rot_scale(frames, pts, rng, rot_scale)
                lab, weak, cls = lab[keep], weak[keep], cls[keep]
        return dict(frames=torch.from_numpy(np.ascontiguousarray(frames)),
                    points=torch.from_numpy(pts.astype(np.float32)),
                    labels=torch.from_numpy(lab),
                    weak=torch.from_numpy(weak),
                    cls=torch.from_numpy(cls),
                    sid=sid, t=t)


def div_collate(batch):
    """Stack frames; keep variable-length point fields as per-item lists."""
    out = {"frames": torch.stack([b["frames"] for b in batch])}
    for k in ("points", "labels", "weak", "cls", "sid", "t"):
        out[k] = [b[k] for b in batch]
    return out
