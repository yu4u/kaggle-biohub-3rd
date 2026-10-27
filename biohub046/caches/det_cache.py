"""Low-threshold DoG detections per frame for the detection-carved loss mask.

For every frame: blob_dog on the isotropic [1,4,4] mean-pooled, quantile-normalised volume at a
LOW threshold (recall first), matched to GT nodes by a <= 7 um Hungarian. Unmatched detections
are suspected unlabeled cells and are carved out of the detector loss (core/dataset.py).
Output: <cache>/det_cache/<sid>.npz with t{T}_det (N,3) int16 zyx and t{T}_match (N,) bool.
"""
from __future__ import annotations

import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment
from skimage.feature import blob_dog

from biohub046 import config as C

SCALE = np.array([1.625, 0.40625, 0.40625])
DOWN = np.array([1, 4, 4])
MAX_MATCH_UM = 7.0
THRESHOLD = 0.015
MIN_SIGMA, MAX_SIGMA, SIGMA_RATIO, OVERLAP = 1.0, 4.0, 1.6, 0.5   # exp_001.Params

OUT = C.CACHE_DIR / 'det_cache'


def _downsample(frame):
    z, y, x = frame.shape
    return frame.reshape(z, y // 4, 4, x // 4, 4).mean(axis=(2, 4))


def one(sid):
    out_path = OUT / f'{sid}.npz'
    if out_path.exists():
        return sid, -1
    from biohub046.core.dataset import load_gt
    zg = zarr.open(str(C.TRAIN_DIR / f'{sid}.zarr'), mode='r')
    arr = zg['0']
    q = dict(zg.attrs)['image_statistics']['quantiles']
    ql, qh = float(q['0.001']), float(q['0.999'])
    _, gt_t, gt_zyx, _, _ = load_gt(C.TRAIN_DIR / f'{sid}.geff')
    data = {}
    for t in range(arr.shape[0]):
        f = _downsample(np.asarray(arr[int(t)]).astype(np.float32))
        fn = np.clip((f - ql) / (qh - ql + 1e-6), 0, None)
        b = blob_dog(fn, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA, sigma_ratio=SIGMA_RATIO,
                     threshold=THRESHOLD, overlap=OVERLAP)
        if len(b) == 0:
            det = np.zeros((0, 3), np.int16)
            matched = np.zeros(0, bool)
        else:
            det = np.rint(b[:, :3] * DOWN).astype(np.int16)
            gk = np.where(gt_t == t)[0]
            matched = np.zeros(len(det), bool)
            if len(gk):
                d = np.linalg.norm(det.astype(float)[:, None] * SCALE - gt_zyx[gk].astype(float)[None] * SCALE, axis=2)
                ri, ci = linear_sum_assignment(np.where(d > MAX_MATCH_UM, 1e6, d))
                for i, j in zip(ri, ci):
                    if d[i, j] <= MAX_MATCH_UM:
                        matched[i] = True
        data[f't{int(t)}_det'] = det
        data[f't{int(t)}_match'] = matched
    tmp = out_path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **data)
    tmp.replace(out_path)
    return sid, sum(len(v) for k, v in data.items() if k.endswith('_det'))


def build(ids=None, workers=16):
    OUT.mkdir(parents=True, exist_ok=True)
    ids = list(ids or C.discover_ids())
    todo = [s for s in ids if not (OUT / f'{s}.npz').exists()]
    if not todo:
        print(f'[det_cache] skip ({len(ids)} cached)', flush=True)
        return
    with Pool(min(workers, len(todo))) as pool:
        for k, (sid, n) in enumerate(pool.imap_unordered(one, todo), 1):
            print(f'[det_cache] {k}/{len(todo)} {sid}: {n} detections', flush=True)
