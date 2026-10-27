"""Fold-matched OOF detection cache of the two-member ensemble used to TRAIN the matcher and div3f.

Members dl_284 / dl_275, weights 0.6 / 0.4, th 0.2 (no TTA, no blur): fold f's dl_284 / dl_275 checkpoints run on fold f's validation movies,
heatmaps averaged 0.6 / 0.4, peaks = exact local maxima above 0.2 within NMS [3,7,7].
Layout per movie: t0000.. (N,3) int32 peak zyx (original voxels), c0000.. (N,) float16 heatmap value.
"""
from __future__ import annotations

import time

import numpy as np

from biohub046 import config as C

CHAMPION_DIR = C.CACHE_DIR / 'champion_dets' / 'ens_dl284_dl275_oof_last'
MEMBERS = [('dl_284', 0.6), ('dl_275', 0.4)]
TH = 0.2


def build(folds=None):
    import torch
    from biohub046.core.dataset import CellStore, to_model_input
    from biohub046.core.metric import extract_peaks
    from biohub046.infer.legacy import load_detector
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    assignments = C.fold_assignments()
    device = 'cuda'
    t0 = time.time()
    for fold in folds if folds is not None else range(C.FOLD_NUM):
        ids = sorted(s for s, f in assignments.items() if f == fold)
        todo = [s for s in ids if not (CHAMPION_DIR / f'{s}.npz').exists()]
        if not todo:
            print(f'[champion_dets] fold {fold}: skip ({len(ids)} cached)', flush=True)
            continue
        models = [(load_detector(C.ARTIFACTS_DIR / 'models' / name / f'{name}_fold{fold}_last.ckpt', device)[1], w)
                  for name, w in MEMBERS]
        store = CellStore(C.TRAIN_DIR, todo, with_gt=False, downsample=np.ones(3, int))
        with torch.no_grad():
            for k, sid in enumerate(todo, 1):
                T = store.shape[sid][0]
                coords, confs = [], []
                for t in range(T):
                    x = to_model_input(store.frame(sid, t), False)[None].to(device)
                    acc = None
                    for model, w in models:
                        out = model(x)
                        out = (out[0] if isinstance(out, tuple) else out).float()[:, 0]
                        acc = out * w if acc is None else acc + out * w
                    pk = extract_peaks(acc[0], TH, C.NMS_KERNEL)
                    coords.append(pk.cpu().numpy().astype(np.int32))
                    confs.append(acc[0][pk[:, 0], pk[:, 1], pk[:, 2]].cpu().numpy().astype(np.float16))
                tmp = CHAMPION_DIR / f'{sid}.tmp.npz'
                np.savez_compressed(tmp, **{f't{i:04d}': c for i, c in enumerate(coords)},
                                    **{f'c{i:04d}': c for i, c in enumerate(confs)})
                tmp.replace(CHAMPION_DIR / f'{sid}.npz')
                print(f'[champion_dets] fold{fold} {k}/{len(todo)} {sid}: {sum(len(c) for c in coords)} dets '
                      f'({time.time() - t0:.0f}s)', flush=True)
        del models
        torch.cuda.empty_cache()
