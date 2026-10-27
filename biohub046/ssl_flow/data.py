"""Unlabelled consecutive-frame crops from Biohub zarr stores."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info
import zarr


def augment_pair(a, b, rng, gain_aug=0.0, optics_aug=0.0, d4_aug=False):
    """Apply correspondence-preserving detector/matcher augmentations."""
    if optics_aug > 0:
        from scipy.ndimage import gaussian_filter
        u = rng.random()
        if u < 0.4:
            sz = rng.uniform(0.0, 0.8) * optics_aug
            syx = rng.uniform(0.0, 1.2) * optics_aug
            a, b = gaussian_filter(a, (sz, syx, syx)), gaussian_filter(b, (sz, syx, syx))
        elif u < 0.6:
            amount = rng.uniform(0.2, 0.7) * optics_aug
            a = a + amount * (a - gaussian_filter(a, (0, 1, 1)))
            b = b + amount * (b - gaussian_filter(b, (0, 1, 1)))
    if gain_aug > 0:
        gain = np.exp(rng.uniform(-np.log1p(gain_aug), np.log1p(gain_aug)))
        a, b = a * gain, b * gain
    if d4_aug:
        k = int(rng.integers(4))
        a, b = np.rot90(a, k, axes=(-2, -1)), np.rot90(b, k, axes=(-2, -1))
        if rng.integers(2):
            a, b = np.flip(a, -1), np.flip(b, -1)
    return np.ascontiguousarray(a, dtype=np.float32), np.ascontiguousarray(b, dtype=np.float32)


def discover_ids(data_dir):
    return sorted(p.stem for p in Path(data_dir).glob("*.zarr"))


def split_ids(ids, val_fraction, seed):
    ids = list(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = max(1, round(len(ids) * val_fraction)) if len(ids) > 1 else len(ids)
    return ids[n:], ids[:n]


def normalize(frame, qlo, qhi):
    return np.clip((frame.astype(np.float32) - qlo) / (qhi - qlo + 1e-6), 0, None)


def downsample_mean(x, factors):
    dz, dy, dx = factors
    z, y, xx = x.shape
    x = x[:z // dz * dz, :y // dy * dy, :xx // dx * dx]
    return x.reshape(z // dz, dz, y // dy, dy, xx // dx, dx).mean((1, 3, 5))


class VolumePairDataset(Dataset):
    def __init__(self, data_dir, ids, crop_size, downsample=(1, 1, 1),
                 length=None, random=True, seed=42, gain_aug=0.0,
                 optics_aug=0.0, d4_aug=False):
        self.root, self.ids = Path(data_dir), list(ids)
        self.crop = np.asarray(crop_size, int)
        self.ds = tuple(int(v) for v in downsample)
        self.random, self.seed = random, seed
        self.gain_aug, self.optics_aug = float(gain_aug), float(optics_aug)
        self.d4_aug = bool(d4_aug)
        self.meta, self._arrays = {}, {}
        self.pairs = []
        for sid in self.ids:
            g = zarr.open(str(self.root / f"{sid}.zarr"), mode="r")
            arr = g["0"]
            q = dict(g.attrs)["image_statistics"]["quantiles"]
            self.meta[sid] = (arr.shape, float(q["0.001"]), float(q["0.999"]))
            self.pairs.extend((sid, t) for t in range(arr.shape[0] - 1))
        if not self.random:
            # Interleave samples by time so a capped validation set represents
            # every held-out sample instead of exhausting the first stores.
            self.pairs.sort(key=lambda pair: (pair[1], pair[0]))
        if not self.pairs:
            raise ValueError(f"no consecutive frame pairs under {self.root}")
        self.length = int(length) if length is not None else len(self.pairs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    def __len__(self):
        return self.length

    def _array(self, sid):
        if sid not in self._arrays:
            self._arrays[sid] = zarr.open(str(self.root / f"{sid}.zarr"), mode="r")["0"]
        return self._arrays[sid]

    def __getitem__(self, index):
        worker = get_worker_info()
        rng_seed = worker.seed + index if self.random and worker else self.seed + index
        rng = np.random.default_rng(rng_seed)
        sid, t = self.pairs[rng.integers(len(self.pairs))] if self.random else self.pairs[index % len(self.pairs)]
        shape, qlo, qhi = self.meta[sid]
        a = downsample_mean(normalize(self._array(sid)[t], qlo, qhi), self.ds)
        b = downsample_mean(normalize(self._array(sid)[t + 1], qlo, qhi), self.ds)
        if np.any(np.asarray(a.shape) < self.crop):
            raise ValueError(f"crop {tuple(self.crop)} exceeds downsampled shape {a.shape}")
        high = np.asarray(a.shape) - self.crop
        origin = np.array([rng.integers(v + 1) for v in high]) if self.random else high // 2
        sl = tuple(slice(o, o + c) for o, c in zip(origin, self.crop))
        a, b = np.ascontiguousarray(a[sl]), np.ascontiguousarray(b[sl])
        if self.random:
            a, b = augment_pair(a, b, rng, self.gain_aug, self.optics_aug,
                                self.d4_aug)
        return {"a": torch.from_numpy(a)[None], "b": torch.from_numpy(b)[None],
                "sid": sid, "t": t, "origin": torch.from_numpy(origin)}
