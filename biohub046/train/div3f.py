"""Stage ④ division models: div_415 (parent), div_478 (pre / post), div_762 (native parent, warp5 input).

Shared recipe (the div_762 checkpoint config): 2.5D EfficientNetV2-S, ds [1,2,2], decoder GroupNorm, out_tap -2, det_neg_weight 0,
div / hard-neg oversample .25, 20 epochs, bs 8, lr 1e-4, bf16, every eval epoch saved, then
"calpick" = the highest val_auc among epochs with val_median_neg < .05 and val_median_weak < .15.
"""
from __future__ import annotations

import json

from biohub046 import config as C
from biohub046.caches.champion_dets import CHAMPION_DIR
from biohub046.caches.dense_flow import DENSE_FLOW_DIR

BASE = ['model.arch=25d', 'data.downsample=[1,2,2]', 'loss.det_neg_weight=0', 'train.save_every_eval=true',
        'train.lr=1e-4', 'train.epochs=20', 'train.min_lr=5e-5', 'data.div_oversample=0.25',
        'data.hard_neg_oversample=0.25', 'model.out_tap=-2', 'model.decoder_norm=gn']

MODELS = {
    # name -> (target, extra overrides, artifact file name pattern)
    'div_415': ('parent', [], 'div_415_f{f}_calpick.pt'),
    'div_478_pre': ('pre', [], 'div_478_pre_fold{f}_calpick.pt'),
    'div_478_post': ('post', [], 'div_478_post_fold{f}_calpick.pt'),
    'div_762': ('parent', ['train.ema_decay=0.995', 'train.min_lr=0.000001', 'data.warp_keep_raw=true',
                           'model.n_frames=5', 'data.num_workers=12', f'data.warp_flow_dir={DENSE_FLOW_DIR}'],
                'div_762_f{f}_calpick.pt'),
}


def ckpt_path(name, fold):
    return C.ARTIFACTS_DIR / 'models' / 'div3f' / MODELS[name][2].format(f=fold)


def calpick(history_path):
    rows = [json.loads(line) for line in history_path.read_text().splitlines()]
    evaluated = [r for r in rows if 'val_auc' in r]
    eligible = [r for r in evaluated if r['val_median_neg'] < .05 and r['val_median_weak'] < .15] or evaluated
    return max(eligible, key=lambda r: r['val_auc'])


def train(names=None, folds=None, smoke=False):
    from biohub046.div3f.train import main as div3f_main
    for name in names or list(MODELS):
        target_role, extra, _ = MODELS[name]
        for fold in folds if folds is not None else range(C.FOLD_NUM):
            target = ckpt_path(name, fold)
            if target.exists():
                print(f'[train_div3f] skip {target.name} (exists)', flush=True)
                continue
            work = C.CACHE_DIR / 'train' / f'{name}_f{fold}'
            argv = BASE + extra + [f'name={name}_f{fold}', f'data.fold_id={fold}', f'data.target={target_role}',
                                   f'data.train_dir={C.TRAIN_DIR}', f'data.champion_det_dir={CHAMPION_DIR}',
                                   f'data.anomaly_links_path={C.ANOMALY_LINKS}', f'train.output_dir={work}',
                                   f'seed={C.SEED}']
            if smoke:
                argv += ['train.epochs=2', 'data.items_per_epoch=16', 'eval.neg_frame_stride=64',
                         'data.num_workers=2']
            print(f'[train_div3f] {name} fold {fold} -> {target}', flush=True)
            div3f_main(argv)
            chosen = calpick(work / 'history.jsonl')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.with_suffix('.json').write_text(json.dumps(chosen, indent=2) + '\n')
            C.publish(work / f"ep{chosen['epoch']:03d}.pt", target)
            print(f'[train_div3f] {name} fold {fold}: calpick epoch {chosen["epoch"]} '
                  f'(val_auc {chosen["val_auc"]:.4f})', flush=True)
