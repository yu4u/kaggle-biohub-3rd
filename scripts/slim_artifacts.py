"""Copy artifacts/ to a distribution directory with inference-only Lightning checkpoints.

A training checkpoint (detectors, matcher) carries the optimizer state and, with EMA, both weight
sets. Inference reads the EMA weights when present, the raw weights otherwise (core/model.py
_resume_state), plus hyper_parameters. This keeps exactly those; every other file is copied as is.

    uv run python scripts/slim_artifacts.py --out /path/to/biohub-sub046-artifacts
"""
import argparse
import shutil
from pathlib import Path

import torch

SOLUTION = Path(__file__).resolve().parents[1]


def slim(src, dst):
    ckpt = torch.load(src, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']
    ema = {k: v for k, v in state.items() if k.startswith('model_ema.module.')}
    keep = ema or {k: v for k, v in state.items() if k.startswith('model.')}
    torch.save(dict(state_dict=keep, hyper_parameters=ckpt['hyper_parameters'],
                    epoch=ckpt.get('epoch'), global_step=ckpt.get('global_step')), dst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, default=SOLUTION / 'artifacts')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    for src in sorted(f for f in args.artifacts.rglob('*') if f.is_file() and f.name != '.gitkeep'):
        dst = args.out / src.relative_to(args.artifacts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == '.ckpt':
            slim(src, dst)
        else:
            shutil.copyfile(src, dst)
        print(f'{src.relative_to(args.artifacts)}: {src.stat().st_size / 2**20:.0f} -> {dst.stat().st_size / 2**20:.0f} MiB', flush=True)


if __name__ == '__main__':
    main()
