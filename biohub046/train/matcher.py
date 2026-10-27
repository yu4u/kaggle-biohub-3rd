"""Stage ② matcher dl_401: siamese EfficientNetV2-S 2.5D trunk + NodeMatcher (xattn_cos, role embedding).

Anchors / candidates come from the fold-matched two-member
detection cache (dl_284 0.6 / dl_275 0.4, th 0.2) and the matcher's candidate frame from the
flow_164 dense field, both built by caches/.
"""
from __future__ import annotations

from biohub046 import config as C
from biohub046.caches.champion_dets import CHAMPION_DIR
from biohub046.caches.dense_flow import DENSE_FLOW_DIR
from biohub046.train.lightning import train_fold

RECIPE = {
    'model': {'temporal': 'siamese', 'matcher': 'xattn_cos', 'in_channels': 3, 'matcher_role_embed': True,
              'embed_head_arch': 'none', 'embed_tap': -2, 'decoder_channels': [512, 256, 128, 64, 32]},
    'data': {'downsample': [1, 4, 4], 'crop_size': [64, 64, 64], 'batch_size': 8, 'num_workers': 8,
             'champion_det_dir': str(CHAMPION_DIR), 'flow_cache_dir': str(DENSE_FLOW_DIR)},
    'loss': {'embed_only': True, 'embed_gt_fill': True, 'embed_max_peaks': 1024, 'embed_radius_um': 16,
             'embed_warmup': [0, 0]},
    'trainer': {'max_epochs': 200},
}


def ckpt_path(fold):
    return C.ARTIFACTS_DIR / 'models' / 'dl_401' / f'dl_401_fold{fold}_last.ckpt'


def train(folds=None, smoke=False):
    for fold in folds if folds is not None else range(C.FOLD_NUM):
        target = ckpt_path(fold)
        if target.exists():
            print(f'[train_matcher] skip {target.name} (exists)', flush=True)
            continue
        print(f'[train_matcher] dl_401 fold {fold} -> {target}', flush=True)
        train_fold('dl_401', fold, RECIPE, target, smoke=smoke)
