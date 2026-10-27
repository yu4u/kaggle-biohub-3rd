"""Stage ② dense flow flow_164 (ssl_flow FlowNet3D, base 32 / feature 32, 48 epochs).

Recipe: lr 5e-4,
warmup 6, base_channels / feature_dim 32, 48 epochs). The deployed checkpoint is the last epoch.
"""
from __future__ import annotations

from biohub046 import config as C

FLOW_NAME = 'ssl_flow_flow_164_ep48.pt'
OVERRIDES = ['data.crop_size=[64,64,64]', 'data.gain_aug=0', 'data.optics_aug=0', 'data.d4_aug=false',
             'data.pairs_per_epoch=10000', 'train.batch_size=4', 'train.epochs=48', 'train.lr=0.0005',
             'train.warmup_epochs=6', 'train.scheduler=cosine', 'train.min_lr=1e-6',
             'model.architecture=3d', 'model.base_channels=32', 'model.feature_dim=32']


def ckpt_path():
    return C.ARTIFACTS_DIR / 'models' / 'flow_164' / FLOW_NAME


def train(smoke=False):
    target = ckpt_path()
    if target.exists():
        print(f'[train_flow] skip {target.name} (exists)', flush=True)
        return
    from biohub046.ssl_flow.train import main as flow_main
    work = C.CACHE_DIR / 'train' / 'flow_164'
    argv = OVERRIDES + [f'data.train_dir={C.TRAIN_DIR}', f'train.output_dir={work}', f'seed={C.SEED}']
    if smoke:
        argv += ['train.epochs=1', 'data.pairs_per_epoch=64', 'eval.max_pairs=8']
    print(f'[train_flow] flow_164 -> {target}', flush=True)
    flow_main(argv)
    C.publish(work / 'last.pt', target)
