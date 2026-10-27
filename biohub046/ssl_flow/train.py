"""Train with: uv run python -m ssl_flow.train [key=value ...]."""
import sys
import os
import json
import time
from pathlib import Path
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
from timm.scheduler import create_scheduler_v2

from biohub046.core.dataset import load_gt, SCALE, to_ds
from .data import VolumePairDataset, discover_ids, split_ids
from .losses import self_supervised_loss, synthetic_equivariance_loss
from .model import build_flow_model
from .ops import sample_volume


class TrainingObjective(torch.nn.Module):
    """Run all repeated flow forwards inside one DDP forward graph."""

    def __init__(self, flow_model):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, a, b, loss_cfg):
        ab = self.flow_model(a, b, return_features=True)
        ba = self.flow_model(b, a, return_features=True)
        loss, _ = self_supervised_loss(
            a, b, ab["flow"], ba["flow"], loss_cfg,
            (ab["features_a"], ab["features_b"]))
        synthetic = synthetic_equivariance_loss(
            self.flow_model, a, loss_cfg.synthetic_max_shift,
            smooth_amplitude=loss_cfg.synthetic_smooth_amplitude)
        return loss + float(loss_cfg.synthetic) * synthetic


def load_config(argv=None):
    cfg = OmegaConf.load(Path(__file__).with_name("config.yaml"))
    return OmegaConf.merge(cfg, OmegaConf.from_cli(argv))


def sparse_edge_cache(data_dir, ids):
    """Load normal dt=1 GT edges for validation-only model selection."""
    cache = {}
    for sid in ids:
        path = Path(data_dir) / f"{sid}.geff"
        if not path.exists():
            continue
        node_ids, times, coords, edges, _ = load_gt(path)
        row = {int(node_id): i for i, node_id in enumerate(node_ids)}
        by_time = {}
        for u, v in edges:
            a, b = row[int(u)], row[int(v)]
            if times[b] == times[a] + 1:
                by_time.setdefault(int(times[a]), []).append((coords[a], coords[b]))
        cache[sid] = by_time
    return cache


@torch.no_grad()
def validate(model, loader, cfg, device, gt_cache=None):
    model.eval()
    sums, count, gt_errors = {}, 0, []
    for batch in loader:
        a, b = batch["a"].to(device), batch["b"].to(device)
        ab, ba = model(a, b, return_features=True), model(b, a, return_features=True)
        loss, metrics = self_supervised_loss(a, b, ab["flow"], ba["flow"], cfg.loss,
                                             (ab["features_a"], ab["features_b"]))
        synthetic = synthetic_equivariance_loss(
            model, a, cfg.loss.synthetic_max_shift, shift=(1.0, 2.0, -2.0),
            smooth_amplitude=cfg.loss.synthetic_smooth_amplitude)
        metrics["synthetic"] = synthetic
        values = {"loss": loss, **metrics}
        for k, value in values.items():
            sums[k] = sums.get(k, 0.0) + float(value)
        if gt_cache:
            ds = np.asarray(cfg.data.downsample, dtype=np.float32)
            for i, sid in enumerate(batch["sid"]):
                edges = gt_cache.get(sid, {}).get(int(batch["t"][i]), [])
                if not edges:
                    continue
                src = np.asarray([edge[0] for edge in edges], dtype=np.float32)
                dst = np.asarray([edge[1] for edge in edges], dtype=np.float32)
                origin = batch["origin"][i].numpy()
                points = to_ds(src, ds) - origin
                shape = np.asarray(ab["flow"].shape[-3:])
                keep = np.all((points >= -0.5) & (points <= shape - 0.5), axis=1)
                if not np.any(keep):
                    continue
                points_t = torch.as_tensor(
                    points[keep], device=device, dtype=ab["flow"].dtype)
                flow = sample_volume(ab["flow"][i:i + 1], points_t)[0]
                pred = src[keep] + flow.cpu().numpy() * ds
                error = np.linalg.norm((pred - dst[keep]) * SCALE, axis=1)
                gt_errors.extend(error.tolist())
        count += 1
    result = {k: v / max(count, 1) for k, v in sums.items()}
    if gt_errors:
        result["gt_epe_um"] = float(np.mean(gt_errors))
        result["gt_recall"] = float(
            np.mean(np.asarray(gt_errors) <= float(cfg.eval.match_radius_um)))
        result["gt_edges"] = len(gt_errors)
    return result


def main(argv=None):
    cfg = load_config(argv)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, local_rank = 0, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    torch.manual_seed(cfg.seed)
    ids = discover_ids(cfg.data.train_dir)
    train_ids, val_ids = split_ids(ids, cfg.data.val_fraction, cfg.seed)
    common = dict(data_dir=cfg.data.train_dir, crop_size=cfg.data.crop_size,
                  downsample=cfg.data.downsample, seed=cfg.seed,
                  gain_aug=cfg.data.get("gain_aug", 0.0),
                  optics_aug=cfg.data.get("optics_aug", 0.0),
                  d4_aug=cfg.data.get("d4_aug", False))
    train_ds = VolumePairDataset(ids=train_ids, length=cfg.data.pairs_per_epoch,
                                 random=True, **common)
    val_ds = VolumePairDataset(ids=val_ids, length=min(cfg.eval.max_pairs, 10_000),
                               random=False, **common)
    train_sampler = DistributedSampler(
        train_ds, shuffle=False, drop_last=False) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=False,
                              sampler=train_sampler,
                              num_workers=cfg.data.num_workers, pin_memory=True,
                              generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = DataLoader(val_ds, batch_size=cfg.eval.batch_size, shuffle=False,
                            num_workers=min(2, cfg.data.num_workers), pin_memory=True)
    gt_cache = sparse_edge_cache(cfg.data.train_dir, val_ids)
    flow_model = build_flow_model(OmegaConf.to_container(cfg.model)).to(device)
    if cfg.train.get("resume_path"):
        resume = torch.load(str(cfg.train.resume_path), map_location=device,
                            weights_only=False)
        flow_model.load_state_dict(resume["model"])
        if rank == 0:
            print(f"loaded model weights from {cfg.train.resume_path}")
    model = TrainingObjective(flow_model)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    scheduler = None
    if str(cfg.train.get("scheduler", "none")).lower() == "cosine":
        scheduler, _ = create_scheduler_v2(
            optimizer=opt, sched="cosine", num_epochs=int(cfg.train.epochs),
            min_lr=float(cfg.train.get("min_lr", 1e-6)),
            warmup_epochs=int(cfg.train.get("warmup_epochs", 0)), warmup_lr=0,
            step_on_epochs=False, updates_per_epoch=max(len(train_loader), 1))
    elif str(cfg.train.get("scheduler", "none")).lower() == "delayed_cosine":
        total = max(int(cfg.train.epochs) * len(train_loader), 1)
        start = int(cfg.train.get("decay_start_epoch", 0)) * len(train_loader)
        floor = float(cfg.train.get("min_lr", 1e-6)) / float(cfg.train.lr)

        def lr_factor(step):
            if step <= start:
                return 1.0
            progress = min(max((step - start) / max(total - start, 1), 0.0), 1.0)
            return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.train.precision == "amp")
    out = Path(cfg.train.output_dir)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    history_path = out / "history.jsonl"
    if rank == 0:
        history_path.write_text("")
    history = []
    best, best_gt = float("inf"), float("inf")
    global_step = 0
    for epoch in range(1, int(cfg.train.epochs) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_start = time.perf_counter()
        model.train(); running = 0.0
        for batch in train_loader:
            a, b = batch["a"].to(device), batch["b"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                loss = model(a, b, cfg.loss)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            global_step += 1
            if scheduler is not None:
                if hasattr(scheduler, "step_update"):
                    scheduler.step_update(global_step)
                else:
                    scheduler.step()
            running += float(loss.detach())
        if distributed:
            running_tensor = torch.tensor(running, device=device)
            dist.all_reduce(running_tensor, op=dist.ReduceOp.SUM)
            running = float(running_tensor) / dist.get_world_size()
            dist.barrier()
        if rank != 0:
            dist.barrier()
            continue
        eval_model = (model.module.flow_model if distributed
                      else model.flow_model)
        metrics = validate(eval_model, val_loader, cfg, device, gt_cache)
        train_loss = running / len(train_loader)
        selection = metrics["loss"] + float(cfg.loss.synthetic) * metrics["synthetic"]
        record = {"epoch": epoch, "train_loss": train_loss,
                  **{f"val_{k}": v for k, v in metrics.items()},
                  "selection": selection, "lr": opt.param_groups[0]["lr"],
                  "epoch_seconds": time.perf_counter() - epoch_start}
        history.append(record)
        with history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"epoch={epoch} train={running / len(train_loader):.5f} " +
              " ".join(f"val_{k}={v:.5f}" for k, v in metrics.items()))
        state = {"model": eval_model.state_dict(), "config": OmegaConf.to_container(cfg),
                 "epoch": epoch, "val": metrics, "history": history}
        torch.save(state, out / "last.pt")
        every = int(cfg.train.get("checkpoint_every", 0))
        if every > 0 and epoch % every == 0:
            torch.save(state, out / f"epoch_{epoch:03d}.pt")
        if selection < best:
            best = selection; torch.save(state, out / "best.pt")
        if metrics.get("gt_epe_um", float("inf")) < best_gt:
            best_gt = metrics["gt_epe_um"]
            torch.save(state, out / "best_gt.pt")
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
