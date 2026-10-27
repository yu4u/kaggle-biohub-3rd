"""Stages ⑧ / ⑨: calibrated predictions -> pooled LP (797 seed, 3 rounds) -> gap closing / L6 -> affine A6 + A7.

The LP driver is the deployed one (lineage.div846_lineage `predict` / `optimize`, run as subprocesses so the
single-thread BLAS environment they set applies to them only), with the fold-matched calibrator bundles.
All movies form ONE pool, as in the 046 v4 OOF evaluation (exp_846).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from biohub046 import config as C

LINEAGE_DIR = C.CACHE_DIR / 'lineage'
FINAL_DIR = C.CACHE_DIR / 'final_graphs'


def _run(args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'[optimize] {" ".join(map(str, args))}\n[optimize] log -> {log}', flush=True)
    with log.open('a') as stream:
        subprocess.run([sys.executable, '-m', 'biohub046.lineage.div846_lineage', *map(str, args)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def predict(ids, jobs=4):
    from biohub046.infer.oof_infer import OOF_DIR
    out = LINEAGE_DIR / 'calibration'
    if all((out / 'predictions' / f'{s}.npz').exists() for s in ids):
        print('[optimize] skip predict (predictions exist)', flush=True)
        return out / 'predictions'
    folds = C.CACHE_DIR / 'folds.json'
    if not folds.exists():
        C.write_folds(folds)
    _run(['predict', '--graphs', OOF_DIR / 'graphs', '--confidence', OOF_DIR / 'confidence',
          '--div-scores', OOF_DIR / 'div_scores', '--bundle-root', C.ARTIFACTS_DIR / 'calibrators',
          '--folds', folds, '--ids', LINEAGE_DIR / 'ids.json', '--out', out, '--jobs', jobs],
         C.CACHE_DIR / 'logs' / 'predict.log')
    return out / 'predictions'


def optimize(ids, predictions, jobs=4):
    from biohub046.infer.oof_infer import OOF_DIR
    out = LINEAGE_DIR / 'joint'
    if (out / 'optimization.json').exists() and all((out / 'final' / f'{s}.npz').exists() for s in ids):
        print('[optimize] skip LP (final graphs exist)', flush=True)
        return out
    _run(['optimize', '--graphs', OOF_DIR / 'graphs', '--predictions', predictions, '--div-scores', OOF_DIR / 'div_scores',
          '--ids', LINEAGE_DIR / 'ids.json', '--out', out, *C.LP_ARGS, '--jobs', jobs], C.CACHE_DIR / 'logs' / 'optimize.log')
    summary = json.loads((out / 'optimization.json').read_text())
    print(f'[optimize] chosen round {summary["chosen_round"]}; pooled surrogate per round: '
          + ', '.join(f'{h["round"]}: {h["pooled"]["score"]:.6f}' for h in summary['history']), flush=True)
    return out


def postprocess(ids, predictions, joint):
    """A6 (q-weighted affine on the real detections) + A7 (affine bridge on the gap nodes) per movie."""
    from biohub046.lineage.affine_refinement import refine_final
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    todo = [s for s in ids if not (FINAL_DIR / f'{s}.npz').exists()]
    if not todo:
        print('[postprocess] skip (final graphs exist)', flush=True)
        return
    started = time.perf_counter()
    rows = []
    for k, sid in enumerate(todo, 1):
        shape = json.loads((C.TRAIN_DIR / f'{sid}.zarr/0/zarr.json').read_text())['shape'][1:]
        row = refine_final(predictions / f'{sid}.npz', joint / 'final' / f'{sid}.npz', shape,
                           FINAL_DIR / f'{sid}.npz', gap_affine=True)
        rows.append(row)
        if k % 25 == 0 or k == len(todo):
            print(f'[postprocess] {k}/{len(todo)} {sid}: moved {row["moved"]}/{row["nodes"]} real, '
                  f'{row["gap"].get("moved", 0)}/{row["gap"].get("synthetic", 0)} gap', flush=True)
    (LINEAGE_DIR / 'affine_report.json').write_text(json.dumps(dict(
        method='post_611 q-weighted affine strength 2 + post_838 affine gap bridge', samples=len(rows),
        wall_seconds=time.perf_counter() - started, per_sample=rows), indent=2) + '\n')


def run(ids=None, jobs=4):
    ids = sorted(ids or C.discover_ids())
    manifest = LINEAGE_DIR / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['ids'] != ids:
        print('[optimize] movie set changed: discarding the previous predictions / LP / final graphs', flush=True)
        shutil.rmtree(LINEAGE_DIR, ignore_errors=True)
        shutil.rmtree(FINAL_DIR, ignore_errors=True)
    LINEAGE_DIR.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(dict(ids=ids), indent=2) + '\n')
    (LINEAGE_DIR / 'ids.json').write_text(json.dumps(ids) + '\n')   # the movie set of predict and the LP
    predictions = predict(ids, jobs)
    joint = optimize(ids, predictions, jobs)
    postprocess(ids, predictions, joint)
