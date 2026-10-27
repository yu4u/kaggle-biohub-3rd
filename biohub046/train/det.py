"""Stage ① detectors: dl_284 (2.5D EfficientNetV2-L), dl_275 (3D SegResNet), dl_492 (2.5D EfficientNet-B7).

The overrides are the original training commands (dl_284 / dl_330, dl_275 / dl_282, dl_492), checked against the configs saved inside the adopted checkpoints.
"""
from __future__ import annotations

from biohub046 import config as C
from biohub046.train.lightning import train_fold

_UNET25D = {
    'data': {'batch_size': 8, 'crop_size': [48, 192, 192], 'optics_aug': 1.0},
    'loss': {'bg_weight': 0.5},
    'model': {'ema': True, 'ema_decay': 0.995, 'in_channels': 3},
    'trainer': {'max_epochs': 800},
}

RECIPES = {
    'dl_284': {**_UNET25D, 'model': {**_UNET25D['model'], 'encoder_name': 'tf_efficientnetv2_l.in21k_ft_in1k'}},
    'dl_275': {
        'model': {'arch': 'segresnet', 'ema': True, 'ema_decay': 0.999},
        'data': {'batch_size': 4, 'crop_size': [48, 192, 192]},
        'loss': {'bg_weight': 0.5},
        'opt': {'lr': 3e-4, 'weight_decay': 1e-4},
        'scheduler': {'warmup_epochs': 10},
        'trainer': {'max_epochs': 400, 'gradient_clip_val': 0.5},
    },
    'dl_492': {**_UNET25D, 'model': {**_UNET25D['model'], 'encoder_name': 'tf_efficientnet_b7.ns_jft_in1k',
                                     'encoder_grad_ckpt': True},
               'data': {**_UNET25D['data'], 'num_workers': 16}},
}


def ckpt_path(name, fold):
    return C.ARTIFACTS_DIR / 'models' / name / f'{name}_fold{fold}_last.ckpt'


def train(names=None, folds=None, smoke=False):
    for name in names or [n for n, _ in C.DETECTORS]:
        for fold in folds if folds is not None else range(C.FOLD_NUM):
            target = ckpt_path(name, fold)
            if target.exists():
                print(f'[train_det] skip {target.name} (exists)', flush=True)
                continue
            print(f'[train_det] {name} fold {fold} -> {target}', flush=True)
            train_fold(name, fold, RECIPES[name], target, smoke=smoke)
