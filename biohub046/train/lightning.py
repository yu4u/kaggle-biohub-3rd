"""Lightning training of the detectors and the matcher (without wandb).

The recipe dictionaries in det.py / matcher.py are merged onto core/config.yaml
exactly like the original CLI dotlist overrides; the paths that config.yaml
expresses relative to the old repository root are replaced here.
"""
from __future__ import annotations

from omegaconf import OmegaConf

from biohub046 import config as C


def base_config():
    cfg = OmegaConf.load(C.PACKAGE / 'core/config.yaml')
    cfg.data.train_dir = str(C.TRAIN_DIR)
    cfg.data.det_cache_dir = str(C.CACHE_DIR / 'det_cache')
    cfg.data.dup_frames_path = str(C.DUP_FRAMES)
    cfg.data.anomaly_links_path = str(C.ANOMALY_LINKS)
    cfg.wandb.name = None
    return cfg


def train_fold(run_name, fold, overrides, target, smoke=False):
    """Train one fold and publish its final checkpoint at `target` (returned).

    Training checkpoints go to <cache>/train/<run_name>/ and an interrupted fold resumes from its
    `_last.ckpt` there; `target` is written only after the scheduled epochs finish, so its existence
    marks a complete model."""
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from biohub046.core.datamodule import CellDataModule
    from biohub046.core.pl_module import MyModel

    cfg = OmegaConf.merge(base_config(), OmegaConf.create(overrides))
    cfg.data.fold_id = int(fold)
    if smoke:
        # a few optimizer steps per model: validates the data path, the loss and the checkpoint format
        cfg.trainer.max_epochs = 2
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.data.val_max_samples = 2
        cfg.data.val_window = 8
        cfg.data.val_embed_pairs = 1
        cfg.data.num_workers = min(int(cfg.data.num_workers), 4)
        # an untrained detector fires on every voxel; linking 100k peaks per frame in the mini val
        # metric would take minutes, so the smoke validates one batch only
        cfg.trainer.limit_val_batches = 1
    seed_everything(cfg.task.seed, workers=True)
    out_dir = C.CACHE_DIR / 'train' / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = MyModel(cfg, mode='train')
    print(OmegaConf.to_yaml(cfg))
    dm = CellDataModule(cfg)
    train_args = dict(cfg.trainer)
    train_args['logger'] = CSVLogger(str(C.CACHE_DIR / 'logs' / run_name), name=f'fold{fold}')
    monitor = str(cfg.task.get('ckpt_monitor', 'val_edge'))
    if bool(cfg.loss.get('embed_only', False)) and monitor == 'val_edge':
        monitor = 'val_emb_acc'
    filename = f'{run_name}_fold{fold}'
    callbacks = [
        LearningRateMonitor(),
        ModelCheckpoint(dirpath=str(out_dir), monitor=monitor, mode='max', save_last=False,
                        filename=filename + '_{epoch:03d}_{' + monitor + ':.4f}',
                        save_weights_only=False, save_top_k=cfg.task.save_top_k),
        # the deployed checkpoint: the newest epoch, saved every epoch (monitor=None -> latest wins)
        ModelCheckpoint(dirpath=str(out_dir), filename=filename + '_last', save_top_k=1,
                        monitor=None, save_weights_only=False),
    ]
    ckpt = out_dir / f'{filename}_last.ckpt'
    resume = ckpt if ckpt.exists() else None
    if resume:
        print(f'[train] resume {run_name} fold {fold} from {resume}', flush=True)
    trainer = Trainer(**train_args, callbacks=callbacks)
    # the checkpoint holds the OmegaConf hparams, which torch>=2.6's weights_only load rejects; it is our own file
    trainer.fit(model, datamodule=dm, ckpt_path=str(resume) if resume else None, weights_only=False)
    if not ckpt.exists():
        raise FileNotFoundError(f'training finished without {ckpt}')
    # the top-k checkpoint is a by-product of the original recipe; the deployed one is `_last`
    for extra in out_dir.glob(f'{filename}_epoch=*.ckpt'):
        extra.unlink()
    C.publish(ckpt, target)
    ckpt.unlink()                                   # the published copy is the complete model
    return target
