"""Dense flow_164 field per movie: <sid>.npy (T-1, 3, Z, Y, X) float16
on the model's [1,4,4] grid in GRID-VOXEL units. Consumed by the matcher training (per-node candidate
offsets) and by div_762 (warping t-1 / t+1 onto t). ~156 MB per movie, ~31 GB for 199.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import zarr

from biohub046 import config as C

DENSE_FLOW_DIR = C.CACHE_DIR / 'dense_flow' / 'flow_164'


def build(ids=None):
    from biohub046.ssl_flow.inference import load_model, prepare_zarr_frame
    from biohub046.train.flow import ckpt_path
    DENSE_FLOW_DIR.mkdir(parents=True, exist_ok=True)
    ids = list(ids or C.discover_ids())
    todo = [s for s in ids if not (DENSE_FLOW_DIR / f'{s}.npy').exists()]
    if not todo:
        print(f'[dense_flow] skip ({len(ids)} cached)', flush=True)
        return
    model, cfg, device = load_model(str(ckpt_path()))
    t0 = time.time()
    for k, sid in enumerate(todo, 1):
        zg = zarr.open(str(C.TRAIN_DIR / f'{sid}.zarr'), mode='r')
        arr, q = zg['0'], dict(zg.attrs)['image_statistics']['quantiles']
        flows, prev = [], None
        for t in range(arr.shape[0] - 1):
            a = prev if prev is not None else prepare_zarr_frame(arr, t, q, cfg.data.downsample)
            b = prepare_zarr_frame(arr, t + 1, q, cfg.data.downsample)
            prev = b
            with torch.no_grad():
                res = model(torch.from_numpy(a)[None, None].to(device), torch.from_numpy(b)[None, None].to(device),
                            return_features=True)
            flows.append(res['flow'][0].cpu().numpy().astype(np.float16))
        tmp = DENSE_FLOW_DIR / f'{sid}.part.npy'
        np.save(tmp, np.stack(flows))
        tmp.rename(DENSE_FLOW_DIR / f'{sid}.npy')
        print(f'[dense_flow] {k}/{len(todo)} {sid} ({time.time() - t0:.0f}s)', flush=True)
