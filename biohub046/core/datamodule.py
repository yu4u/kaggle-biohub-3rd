"""DataModule: sample-level, embryo-stratified fold split over the 199 train crops.

CV design is an OPEN question: with only 2 embryos this
split is a model-selection device for DL training, NOT the main CV number.
The main CV stays the official metric over all 199 samples
(the full out-of-fold evaluation).
"""
from pathlib import Path

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .dataset import (CellStore, CellTrainDataset, CellTrainDatasetPair,
                      CellTrainDatasetSiamese, CellValDataset, CellValDatasetPair,
                      CellValDatasetSiameseEmbed, get_downsample, siamese_collate)


def discover_ids(data_dir):
    return sorted(p.stem for p in Path(data_dir).glob("*.zarr"))


from .folds import fold_split


class CellDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        cfg = self.cfg
        ids = discover_ids(cfg.data.train_dir)
        if int(cfg.data.fold_id) == -2:
            # all-data training (dl_353): every sample trains; the fold-0 val
            # split stays as a LEAKY monitoring set (never select on it — the
            # deliverable is last.ckpt)
            train_ids, val_ids = list(ids), fold_split(
                ids, cfg.data.fold_num, 0)[1]
        else:
            train_ids, val_ids = fold_split(ids, cfg.data.fold_num,
                                            cfg.data.fold_id)
        det_dir = cfg.data.det_cache_dir if cfg.loss.mask_mode == "carved" else None
        store = CellStore(cfg.data.train_dir, ids, with_gt=True, det_dir=det_dir,
                          downsample=get_downsample(cfg))
        extra_dir = cfg.data.get("extra_train_dir")
        if extra_dir:
            # dl_190: densely labelled external samples join the TRAIN list only
            # (never val, never the fold split); see CellStore.add_dense
            extra_ids = discover_ids(extra_dir)
            store.add_dense(extra_dir, extra_ids)
            if cfg.data.get("extra_p_start"):
                # dl_193: curriculum — the dense samples form a replacement pool
                # (CellTrainDataset.extra_p), the item list stays real-only
                print(f"extra_train_dir {extra_dir}: {len(extra_ids)} dense samples as a "
                      f"curriculum pool (p {cfg.data.extra_p_start} -> 0 at "
                      f"{cfg.data.get('extra_p_end_frac', 0.5)} x max_epochs)")
            else:
                train_ids = list(train_ids) + extra_ids
                print(f"extra_train_dir {extra_dir}: +{len(extra_ids)} dense samples "
                      f"(train only)")
        if str(cfg.model.get("temporal", "none")) == "siamese":
            self.train_dataset = CellTrainDatasetSiamese(cfg, store, train_ids)
            if bool(cfg.loss.get("embed_only", False)):
                # dedicated embedding model: detection val is meaningless (and
                # its garbage heatmaps explode peak matching) — validate the
                # embedding itself (val_emb_acc)
                self.val_dataset = CellValDatasetSiameseEmbed(cfg, store, val_ids)
            else:
                # joint model: val stays single-frame detection (comparable to
                # the dl_011 line); embedding is evaluated offline
                self.val_dataset = CellValDataset(cfg, store, val_ids)
        elif str(cfg.model.get("temporal", "none")) == "pair":
            self.train_dataset = CellTrainDatasetPair(cfg, store, train_ids)
            self.val_dataset = CellValDatasetPair(cfg, store, val_ids)
        else:
            self.train_dataset = CellTrainDataset(cfg, store, train_ids)
            self.val_dataset = CellValDataset(cfg, store, val_ids)
        print(f"fold {cfg.data.fold_id}: train {len(self.train_dataset)} videos "
              f"(1 random annotated frame each per epoch) / val "
              f"{len(self.val_dataset)} frames")

    def train_dataloader(self):
        siamese = str(self.cfg.model.get("temporal", "none")) == "siamese"
        # dl_191: the synth-curriculum epoch counter is an attribute a callback
        # sets on the MAIN-process dataset each epoch; persistent workers would
        # keep stale forks alive across epochs, so they are respawned per epoch
        # on this path (fork picks up the updated value; ~1 s/epoch)
        synth = bool(self.cfg.data.get("synth_pool_dir")) or \
            bool(self.cfg.data.get("extra_p_start"))   # dl_193 shares the mechanism
        return DataLoader(
            self.train_dataset, batch_size=self.cfg.data.batch_size, shuffle=True,
            drop_last=True, num_workers=self.cfg.data.num_workers, pin_memory=True,
            persistent_workers=self.cfg.data.num_workers > 0 and not synth,
            collate_fn=siamese_collate if siamese else None,
        )

    def val_dataloader(self):
        # batch_size=1: full frames with variable-length GT coords per item
        embed_val = isinstance(self.val_dataset, CellValDatasetSiameseEmbed)
        return DataLoader(
            self.val_dataset, batch_size=1, shuffle=False,
            num_workers=min(2, self.cfg.data.num_workers), pin_memory=True,
            collate_fn=siamese_collate if embed_val else None,
        )
