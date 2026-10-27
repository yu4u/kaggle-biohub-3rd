"""Run the 046 v4 reproduction end to end: `python -m biohub046.pipeline all`.

Stages (each skipped when its outputs already exist):
  prepare        fold split, DoG detection cache for the carved detector loss
  train_det      dl_284 / dl_275 / dl_492 x 5 folds            -> artifacts/models/<name>/
  train_flow     flow_164                                      -> artifacts/models/flow_164/
  train_caches   two-member OOF detection cache, dense flow cache (inputs of the matcher / div3f training)
  train_matcher  dl_401 x 5 folds                              -> artifacts/models/dl_401/
  train_div3f    div_415, div_478 pre / post, div_762 x 5 folds -> artifacts/models/div3f/
  oof_infer      fold-matched inference of every train movie   -> cache/oof/
  calibrate      ordinary / native / wide calibrators per fold and on all movies -> artifacts/calibrators/
  optimize       calibrated predictions -> pooled LP -> gap closing / L6 -> affine A6 + A7 -> cache/final_graphs/
  score          official metric, pooled over all movies       -> cache/score.json (+ printed)
"""
from __future__ import annotations

import argparse
import os
import time

from biohub046 import config as C

STAGES = ['prepare', 'train_det', 'train_flow', 'train_caches', 'train_matcher', 'train_div3f',
          'oof_infer', 'calibrate', 'optimize', 'score']


def run_stage(name, args):
    ids = C.discover_ids()[:args.limit] if args.limit else None
    if name == 'prepare':
        folds = C.write_folds()
        print(f'[prepare] {len(folds)} movies, folds '
              f'{[sum(1 for f in folds.values() if f == k) for k in range(C.FOLD_NUM)]}', flush=True)
        from biohub046.caches import det_cache
        det_cache.build(workers=args.jobs if args.jobs > 4 else os.cpu_count() or 4)
    elif name == 'train_det':
        from biohub046.train import det
        det.train(smoke=args.smoke)
    elif name == 'train_flow':
        from biohub046.train import flow
        flow.train(smoke=args.smoke)
    elif name == 'train_caches':
        from biohub046.caches import champion_dets, dense_flow
        from biohub046.train import div3f, matcher
        needed = any(not matcher.ckpt_path(f).exists() for f in range(C.FOLD_NUM)) or any(
            not div3f.ckpt_path(n, f).exists() for n in div3f.MODELS for f in range(C.FOLD_NUM))
        if not needed:
            print('[train_caches] skip (every matcher / div3f checkpoint exists)', flush=True)
            return
        champion_dets.build()
        dense_flow.build()
    elif name == 'train_matcher':
        from biohub046.train import matcher
        matcher.train(smoke=args.smoke)
    elif name == 'train_div3f':
        from biohub046.train import div3f
        div3f.train(smoke=args.smoke)
    elif name == 'oof_infer':
        from biohub046.infer import oof_infer
        oof_infer.main(['--limit', str(args.limit)] if args.limit else [])
    elif name == 'calibrate':
        from biohub046 import calibrate
        calibrate.run(ids, jobs=args.jobs)
    elif name == 'optimize':
        from biohub046.lineage import optimize
        optimize.run(ids, jobs=args.jobs)
    elif name == 'score':
        from biohub046 import score
        score.run(ids, jobs=args.jobs)
    else:
        raise ValueError(name)


SMOKE_MOVIES = 12


def reexec_smoke(argv):
    """--smoke: a 12-movie universe with its own artifacts / cache under <cache>/smoke, so nothing a smoke run
    produces can be mistaken for a trained model. The directories are fixed through the environment before
    any module computes a path from them, hence the re-exec."""
    import subprocess
    import sys
    smoke_root = C.CACHE_DIR / 'smoke'
    subset = smoke_root / 'train'
    C.make_subset(SMOKE_MOVIES, subset)
    env = dict(os.environ, BIOHUB_SMOKE='1', BIOHUB_SMOKE_ISOLATED='1', BIOHUB_TRAIN_DIR=str(subset),
               BIOHUB_ARTIFACTS_DIR=str(smoke_root / 'artifacts'), BIOHUB_CACHE_DIR=str(smoke_root / 'cache'))
    print(f'[pipeline] smoke: {SMOKE_MOVIES} movies in {subset}; artifacts / cache under {smoke_root}', flush=True)
    raise SystemExit(subprocess.call([sys.executable, '-m', 'biohub046.pipeline', *argv], env=env))


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', nargs='?', default='all', choices=['all', *STAGES])
    p.add_argument('--from', dest='start', choices=STAGES, default=None, help='run from this stage onwards')
    p.add_argument('--limit', type=int, default=int(os.environ.get('BIOHUB_LIMIT', '0')) or None,
                   help='restrict oof_infer / calibrate / optimize / score to the first N movies '
                        '(the training stages and their caches always use every movie)')
    p.add_argument('--smoke', action='store_true', default=C.SMOKE,
                   help=f'{SMOKE_MOVIES}-movie universe, 1-2 training epochs: validates the code path, not the models')
    p.add_argument('--jobs', type=int, default=int(os.environ.get('BIOHUB_JOBS', '4')), help='CPU workers')
    args = p.parse_args(argv)
    if args.smoke and not C.SMOKE_ISOLATED:
        reexec_smoke([a for a in argv if a != '--smoke'])
    stages = STAGES if args.stage == 'all' else [args.stage]
    if args.start:
        stages = STAGES[STAGES.index(args.start):]
    print(f'[pipeline] input {C.INPUT_DIR} | artifacts {C.ARTIFACTS_DIR} | cache {C.CACHE_DIR} | '
          f'stages {stages} | limit {args.limit} | smoke {args.smoke}', flush=True)
    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in stages:
        started = time.monotonic()
        print(f'\n[pipeline] ===== {name} =====', flush=True)
        run_stage(name, args)
        print(f'[pipeline] {name} done in {time.monotonic() - started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
