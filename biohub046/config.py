"""Paths, fold split and the fixed constants of the 046 v4 stack.

Every location is taken from the environment so the same code runs inside the
container (compose sets the three directories) and directly with `uv run`.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SOLUTION = PACKAGE.parent

INPUT_DIR = Path(os.environ.get('BIOHUB_INPUT_DIR', SOLUTION / 'input'))
ARTIFACTS_DIR = Path(os.environ.get('BIOHUB_ARTIFACTS_DIR', SOLUTION / 'artifacts'))
CACHE_DIR = Path(os.environ.get('BIOHUB_CACHE_DIR', SOLUTION / 'cache'))
TRAIN_DIR = Path(os.environ.get('BIOHUB_TRAIN_DIR', INPUT_DIR / 'train'))   # --smoke points this at a subset
SMOKE = os.environ.get('BIOHUB_SMOKE', '') == '1'                          # same as --smoke
SMOKE_ISOLATED = os.environ.get('BIOHUB_SMOKE_ISOLATED', '') == '1'        # set by pipeline.reexec_smoke only

FOLD_NUM = 5
SEED = 42

# packaged data files read by the training datasets (frozen frames, anomalous GT links)
DUP_FRAMES = PACKAGE / 'data/dup_frames.json'
ANOMALY_LINKS = PACKAGE / 'data/gt_anomaly_links.csv'

# ---- stage ① detection (033 operating point) -------------------------------------------------
DETECTORS = [('dl_284', 0.3), ('dl_275', 0.3), ('dl_492', 0.4)]
DET_THRESHOLD = 0.2
NMS_KERNEL = [3, 7, 7]
DET_BLUR_SIGMA = 0.7
DET_TTA = 'flip2'

# ---- stage ④ division models ------------------------------------------------------------------
DIV_ROLES = ('parent', 'pre', 'post', 'native_parent')
DIV_TTA = 'rot4'

# ---- lineage optimization (046 v4) ------------------------------------------------------------
LP_ARGS = ['--include-divisions', '--node-adjustment', 'clamped', '--division-postprocess', 'legacy',
           '--gap-lmax', '3', '--rounds', '3', '--time-limit', '120']

REFERENCE_SCORE = 0.977800677463061   # 046 v4 stack, all 199 train movies OOF


def model_paths():
    """Every checkpoint the OOF inference needs, keyed by file name, in `artifacts/models`."""
    m = ARTIFACTS_DIR / 'models'
    paths = {}
    for name, _ in DETECTORS:
        for f in range(FOLD_NUM):
            paths[f'{name}_fold{f}_last.ckpt'] = m / name / f'{name}_fold{f}_last.ckpt'
    for f in range(FOLD_NUM):
        paths[f'dl_401_fold{f}_last.ckpt'] = m / 'dl_401' / f'dl_401_fold{f}_last.ckpt'
        paths[f'div_415_f{f}_calpick.pt'] = m / 'div3f' / f'div_415_f{f}_calpick.pt'
        paths[f'div_478_pre_fold{f}_calpick.pt'] = m / 'div3f' / f'div_478_pre_fold{f}_calpick.pt'
        paths[f'div_478_post_fold{f}_calpick.pt'] = m / 'div3f' / f'div_478_post_fold{f}_calpick.pt'
        paths[f'div_762_f{f}_calpick.pt'] = m / 'div3f' / f'div_762_f{f}_calpick.pt'
    paths['ssl_flow_flow_164_ep48.pt'] = m / 'flow_164' / 'ssl_flow_flow_164_ep48.pt'
    return paths


def discover_ids(data_dir=None):
    return sorted(p.stem for p in Path(data_dir or TRAIN_DIR).glob('*.zarr'))


def fold_assignments(ids=None):
    """{sid: fold}: deterministic round-robin within each embryo (core.folds.fold_split)."""
    from biohub046.core.folds import fold_assignments as _assign
    return _assign(list(ids or discover_ids()), FOLD_NUM)


def write_folds(path=None):
    path = Path(path or CACHE_DIR / 'folds.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    folds = fold_assignments()
    path.write_text(json.dumps(folds, indent=2) + '\n')
    return folds


def publish(src, target):
    """Copy a finished model to its deployed path atomically: the training stages skip a model whose
    target exists, so a partial copy must never appear there."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.tmp')
    shutil.copyfile(src, tmp)
    os.replace(tmp, target)


def thread_env(n=1):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS'):
        os.environ.setdefault(key, str(n))
    os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')


def make_subset(n, out):
    """Symlink the first n movies (zarr + geff) of INPUT_DIR/train into `out` (the --smoke universe)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in (INPUT_DIR / 'train').glob('*.zarr'))[:n]
    for sid in ids:
        for suffix in ('.zarr', '.geff'):
            link = out / f'{sid}{suffix}'
            if not link.exists():
                link.symlink_to((INPUT_DIR / 'train' / f'{sid}{suffix}').resolve())
    return ids
