"""Train with: uv run python -m div3f.train [key=value ...]."""
import hashlib
import json
import sys
import time
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from timm.scheduler import create_scheduler_v2
from torch.utils.data import DataLoader

from biohub046.core.dataset import DIV_PAR, DIV_POS, CellStore, get_downsample
from biohub046.core.datamodule import discover_ids, fold_split
from biohub046.core.metric import binary_auc_ap
from biohub046.core.division_teacher import make_teacher, teacher_kl
from .data import DivPointDataset, div_collate, REVIEWED_NEG
from .model import build_model, sample_logits


def load_config(argv=None):
    cfg = OmegaConf.load(Path(__file__).with_name("config.yaml"))
    return OmegaConf.merge(cfg, OmegaConf.from_cli(argv))


def initialize_from_checkpoint(model, cfg):
    """Warm start model weights only, requiring the same fold and target."""
    template = cfg.train.get("init_checkpoint")
    if not template:
        return None
    path = Path(str(template).format(fold=int(cfg.data.fold_id)))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("fold_id", "fold_num", "target"):
        # Checkpoints predating independent phase heads were parent-only.
        previous = checkpoint["config"]["data"].get(key, "parent" if key == "target" else None)
        if previous != cfg.data[key]:
            raise ValueError(f"Initialization checkpoint has a different {key}: {path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                epoch=checkpoint["epoch"], fold=int(cfg.data.fold_id))


def gather_point_logits(logits, batch, device):
    """Concatenate per-item sampled logits + labels over the batch."""
    lg, lab, weak, cls = [], [], [], []
    for i, pts in enumerate(batch["points"]):
        if len(pts) == 0:
            continue
        lg.append(sample_logits(logits[i:i + 1], pts.to(device)))
        lab.append(batch["labels"][i].to(device))
        weak.append(batch["weak"][i].to(device))
        cls.append(batch["cls"][i])
    return (torch.cat(lg), torch.cat(lab), torch.cat(weak),
            torch.cat(cls).numpy())


def div_bce_loss(lg, lab, weak, neg_weight, det_neg_weight,
                 label_smoothing=0.0, pseudo_pos_weight=0.0, pseudo_as_gt=False,
                 reviewed_negative=None):
    """Per-class separate means (embed_loss cls pattern): the ~150 positives
    must not drown in the ~400-per-frame weak negatives."""
    if pseudo_as_gt:
        assert pseudo_pos_weight == 0, 'Merged positives cannot also use separate pseudo loss'
        weak = weak & (lab == 0)
    tgt = lab
    if label_smoothing > 0:
        # div_378: soft targets y(1-e)+e/2; class means below still use lab
        tgt = lab * (1.0 - label_smoothing) + label_smoothing / 2.0
    ce = F.binary_cross_entropy_with_logits(lg.float(), tgt, reduction="none")
    zero = lg.sum() * 0.0

    def cmean(sel):
        return ce[sel].mean() if bool(sel.any()) else zero

    reviewed = torch.zeros_like(weak) if reviewed_negative is None else reviewed_negative
    result = (cmean((lab == 1) & ~weak)
            + neg_weight * cmean((lab == 0) & ~weak)
            + det_neg_weight * cmean((lab == 0) & weak & ~reviewed)
            + pseudo_pos_weight * cmean((lab == 1) & weak))
    return result if reviewed_negative is None else result + neg_weight * cmean(reviewed)


@torch.no_grad()
def validate(model, loader, device, amp, amp_dtype=torch.float16):
    model.eval()
    scores, labels, weaks, clss = [], [], [], []
    for batch in loader:
        frames = batch["frames"].to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            logits = model(frames)
        lg, lab, weak, cls = gather_point_logits(logits.float(), batch, device)
        scores.append(torch.sigmoid(lg).cpu().numpy())
        labels.append(lab.cpu().numpy())
        weaks.append(weak.cpu().numpy())
        clss.append(cls)
    s, lab = np.concatenate(scores), np.concatenate(labels)
    weak, cls = np.concatenate(weaks), np.concatenate(clss)
    m = ~weak
    auc, ap = binary_auc_ap(s[m], lab[m])   # GT-matched only: comparable with
    out = {"auc": auc, "ap": ap,            # the dl_195/197 protocol
           "n_pos": int(lab[m].sum()), "n_matched": int(m.sum())}
    pd = m & ((cls == DIV_PAR) | (cls == DIV_POS))
    if (lab[pd] == 1).any() and (lab[pd] == 0).any():
        out["auc_par_vs_dau"] = binary_auc_ap(s[pd], lab[pd])[0]
    for key, sel in (("median_pos", m & (lab == 1)),
                     ("median_dau", m & (cls == DIV_POS)),
                     ("median_neg", m & (lab == 0)),
                     ("median_weak", weak)):
        if sel.any():
            out[key] = float(np.median(s[sel]))
    return out


def main(argv=None):
    cfg = load_config(argv)
    has_hm = [bool(cfg.data.get(k)) for k in ("det_heatmap_train_dir",
                                              "det_heatmap_val_dir")]
    assert has_hm[0] == has_hm[1], "det_heatmap_train_dir/val_dir: set both or neither"
    # the image + det heatmap planes; kept in the saved config for export
    cfg.model.in_planes = 2 if has_hm[0] else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    ids = discover_ids(cfg.data.train_dir)
    train_ids, val_ids = fold_split(ids, cfg.data.fold_num, cfg.data.fold_id)
    store = CellStore(cfg.data.train_dir, ids, with_gt=True,
                      downsample=get_downsample(cfg))
    train_ds = DivPointDataset(cfg, store, train_ids, train=True)
    val_ds = DivPointDataset(cfg, store, val_ids, train=False)
    print(f"train: {len(train_ds.frame_items)} frames "
          f"({len(train_ds.div_items)} with a parent, "
          f"{train_ds.n_par_missed} parents undetected) | "
          f"val: {len(val_ds.items)} items "
          f"({len(val_ds.div_items)} with a parent, "
          f"{val_ds.n_par_missed} parents undetected)")
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=False, num_workers=cfg.data.num_workers,
                              pin_memory=True, collate_fn=div_collate,
                              generator=torch.Generator().manual_seed(cfg.seed))
    # num_workers=0 so the val frame cache lives (and persists) in this process
    val_loader = DataLoader(val_ds, batch_size=cfg.eval.batch_size,
                            shuffle=False, num_workers=0,
                            collate_fn=div_collate)
    model = build_model(cfg.model).to(device)
    initialization = initialize_from_checkpoint(model, cfg)
    teacher_weight = float(cfg.loss.get("teacher_weight", 0.0))
    teacher = make_teacher(model, initialization, teacher_weight)
    enc_mult = float(cfg.train.get("encoder_lr_mult", 1.0))
    if enc_mult != 1.0 and hasattr(model, "encoder"):
        # div_378: discriminative lr — the pretrained encoder at lr*mult
        enc_ids = {id(p) for p in model.encoder.parameters()}
        groups = [{"params": [p for p in model.parameters() if id(p) in enc_ids],
                   "lr": cfg.train.lr * enc_mult},
                  {"params": [p for p in model.parameters() if id(p) not in enc_ids],
                   "lr": cfg.train.lr}]
        opt = torch.optim.AdamW(groups, lr=cfg.train.lr,
                                weight_decay=cfg.train.weight_decay)
    else:
        # freeze_encoder_epochs > 0 (div_402): the encoder is frozen only for the
        # first N epochs and released afterwards, so its params must be in the
        # optimizer from the start (AdamW skips params whose grad is None)
        unfreeze_at = int(cfg.train.get("freeze_encoder_epochs", 0))
        params = (list(model.parameters()) if unfreeze_at > 0
                  else [p for p in model.parameters() if p.requires_grad])
        opt = torch.optim.AdamW(params, lr=cfg.train.lr,
                                weight_decay=cfg.train.weight_decay)
    ema_decay = float(cfg.train.get("ema_decay", 0.0))
    ema = None
    if ema_decay > 0:
        # div_378: weight EMA (det line's ModelEmaV3); val + saved ckpts use
        # the EMA weights, like the detector's mini val
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=ema_decay, update_after_step=0)
        ema.requires_grad_(False)
    eval_model = ema.module if ema is not None else model
    label_smoothing = float(cfg.loss.get("label_smoothing", 0.0))
    # div_447: train.sched_epochs = the cosine horizon (defaults to the
    # training length); with sched_epochs > epochs the run stops before the
    # decay completes (the old 50-epoch recipe's lr trajectory, cut at ep20)
    scheduler, _ = create_scheduler_v2(
        optimizer=opt, sched="cosine",
        num_epochs=int(cfg.train.get("sched_epochs") or cfg.train.epochs),
        min_lr=float(cfg.train.min_lr),
        warmup_epochs=int(cfg.train.warmup_epochs), warmup_lr=0,
        step_on_epochs=False, updates_per_epoch=max(len(train_loader), 1))
    # bf16: same fp32 dynamic range, no GradScaler — the norm-free Block hit
    # transient fp16 activation overflows (NaN at ep27 of the first div_303
    # run, grad_norm 4-7 sustained) that damaged val_auc 0.907 -> 0.74
    precision = str(cfg.train.precision)
    amp = device.type == "cuda" and precision in ("amp", "bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and precision == "amp")
    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if initialization is not None:
        (out / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")
    history_path = out / "history.jsonl"
    history_path.write_text("")
    neg_w = float(cfg.loss.neg_weight)
    det_neg_w = float(cfg.loss.det_neg_weight)
    pseudo_w = float(cfg.loss.get("pseudo_pos_weight", 0.0))
    pseudo_as_gt = bool(cfg.loss.get("pseudo_as_gt", False))
    every = int(cfg.eval.get("every_n_epochs", 2))
    history, best, global_step = [], -1.0, 0
    unfreeze_at = int(cfg.train.get("freeze_encoder_epochs", 0))
    for epoch in range(1, int(cfg.train.epochs) + 1):
        # DataLoader recreates its workers each epoch (persistent_workers=False).
        train_ds.set_epoch(epoch)
        epoch_start = time.perf_counter()
        if unfreeze_at > 0 and epoch == unfreeze_at + 1 and getattr(model, "freeze_encoder", False):
            # div_402: release the encoder after the warm-up epochs
            model.freeze_encoder = False
            for p in model.encoder.parameters():
                p.requires_grad_(True)
            print(f"epoch {epoch}: encoder unfrozen", flush=True)
        model.train()
        running, grad_norms, n_skipped = 0.0, [], 0
        gt_presentations, pseudo_presentations = 0, 0
        hard_running, teacher_running, teacher_points = 0.0, 0.0, 0
        teacher_positive_points = 0
        reviewed_presentations, reviewed_running = 0, 0.0
        reviewed_by_frame = {}
        for batch in train_loader:
            frames = batch["frames"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                if teacher is not None:
                    with torch.no_grad():
                        teacher_lg = gather_point_logits(teacher(frames), batch, device)[0]
                logits = model(frames)
                lg, lab, weak, cls = gather_point_logits(logits, batch, device)
                reviewed = (torch.as_tensor(cls == REVIEWED_NEG, device=device)
                            if cfg.data.get("reviewed_negative_path") else None)
                loss = div_bce_loss(lg, lab, weak, neg_w, det_neg_w,
                                    label_smoothing, pseudo_w, pseudo_as_gt, reviewed)
                hard_loss = loss
                if teacher is not None:
                    consistency = teacher_kl(lg, teacher_lg,
                        class_balanced=bool(cfg.loss.get("teacher_class_balanced", False)))
                    loss = loss + teacher_weight * consistency
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            gt_presentations += int(((lab == 1) & ~weak).sum())
            pseudo_presentations += int(((lab == 1) & weak).sum())
            if reviewed is not None:
                reviewed_presentations += int(reviewed.sum())
                if bool(reviewed.any()):
                    reviewed_running += neg_w * float(F.binary_cross_entropy_with_logits(
                        lg[reviewed].detach().float(), torch.full_like(lg[reviewed].float(), label_smoothing / 2)))
                for sid, t, item_cls in zip(batch["sid"], batch["t"], batch["cls"]):
                    count = int((item_cls == REVIEWED_NEG).sum())
                    if count:
                        key = f"{sid}:{t}"
                        reviewed_by_frame[key] = reviewed_by_frame.get(key, 0) + count
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)))
            scaler.step(opt)
            scaler.update()
            global_step += 1
            if ema is not None:
                ema.update(model, step=global_step)
            scheduler.step_update(global_step)
            running += float(loss.detach())
            if teacher is not None:
                hard_running += float(hard_loss.detach())
                teacher_running += float(consistency.detach())
                teacher_points += len(lg)
                teacher_positive_points += int((teacher_lg.detach().float().sigmoid() >= .5).sum())
        n_done = max(len(train_loader) - n_skipped, 1)
        record = {"epoch": epoch, "train_loss": running / n_done,
                  "gt_positive_presentations": gt_presentations,
                  "pseudo_positive_presentations": pseudo_presentations,
                  "grad_norm": float(np.median(grad_norms)) if grad_norms else 0.0,
                  "n_skipped": n_skipped, "lr": opt.param_groups[0]["lr"]}
        if cfg.data.get("reviewed_negative_path"):
            record.update(reviewed_negative_presentations=reviewed_presentations,
                          reviewed_negative_loss=reviewed_running / n_done,
                          reviewed_negative_by_frame=reviewed_by_frame)
        if teacher is not None:
            record.update(hard_train_loss=hard_running / n_done,
                          teacher_kl=teacher_running / n_done, teacher_points=teacher_points,
                          teacher_positive_points=teacher_positive_points)
        if train_ds.pseudo_sampling_probability is not None:
            record["pseudo_sampling_probability"] = train_ds.pseudo_sampling_probability
        if epoch % every == 0 or epoch == int(cfg.train.epochs):
            metrics = validate(eval_model, val_loader, device, amp, amp_dtype)
            record.update({f"val_{k}": v for k, v in metrics.items()})
            state = {"model": eval_model.state_dict(),
                     "config": OmegaConf.to_container(cfg),
                     "epoch": epoch, "val": metrics}
            torch.save(state, out / "last.pt")
            if bool(cfg.train.get("save_every_eval", False)):
                torch.save(state, out / f"ep{epoch:03d}.pt")
            if metrics["auc"] > best:
                best = metrics["auc"]
                torch.save(state, out / "best.pt")
        record["epoch_seconds"] = time.perf_counter() - epoch_start
        if device.type == "cuda":
            record["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        history.append(record)
        with history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(" ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in record.items()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
