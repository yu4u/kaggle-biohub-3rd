import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from pytorch_lightning.core.module import LightningModule
from timm.utils import ModelEmaV3
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from .model import get_model_from_cfg
from .loss import DivLoss, get_loss
from .dataset import DIV_IGN, DIV_PAR, DIV_POS, get_downsample, to_ds, to_orig
from .metric import (extract_peaks, refine_peaks, refine_rad, hungarian_pairs,
                     build_graph, filter_short_tracks, eval_sample, summarise,
                     binary_auc_ap, decode_coarse)


def _gather(obj):
    """all_gather a python object across ranks -> flat list (or [obj] if not distributed)."""
    if dist.is_available() and dist.is_initialized():
        out = [None] * dist.get_world_size()
        dist.all_gather_object(out, obj)
        return out
    return [obj]


class MyModel(LightningModule):
    def __init__(self, cfg, mode="train"):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = get_model_from_cfg(cfg, cfg.model.resume_path)
        if hasattr(self.model, "embed_only_forward") and bool(
                cfg.data.get("champion_det_dir")) and (
                bool(cfg.model.get("freeze_trunk", False))
                or bool(cfg.loss.get("embed_only", False))):
            # no det loss and candidates come from the cache: the heatmap is
            # never read, so skip the stride-1 decoder block (before the EMA
            # copy so the eval model inherits the flag)
            self.model.embed_only_forward = True
        if mode != "test" and cfg.model.ema:
            self.model_ema = ModelEmaV3(
                self.model, decay=cfg.model.ema_decay,
                update_after_step=cfg.model.ema_update_after_step,
            )
            self.model_ema.requires_grad_(False)
        self.loss = get_loss(cfg)
        # dl_195: daughter channel
        self.div_weight = float(cfg.loss.get("div_weight", 0.0))
        if self.div_weight > 0:
            assert str(cfg.model.get("temporal", "none")) == "none", (
                "div channel is only wired for the single-frame detector")
            assert bool(cfg.model.get("div_head", False)), (
                "loss.div_weight > 0 needs model.div_head=true")
            assert float(cfg.loss.get("mixup_alpha", 0.0)) == 0, (
                "div channel is not wired through the mixup path")
            self.div_loss = DivLoss(cfg)
        # matcher node-cls loss: other/daughter/division-parent CE on the
        # matcher's final cross-attended features (core/embed_loss.py)
        self.matcher_cls_weight = float(cfg.loss.get("matcher_cls_weight", 0.0))
        if self.matcher_cls_weight > 0:
            assert str(cfg.model.get("temporal", "none")) == "siamese" \
                and str(cfg.model.get("matcher", "none")) != "none" \
                and bool(cfg.model.get("matcher_cls_head", False)), (
                    "loss.matcher_cls_weight > 0 needs model.temporal=siamese, "
                    "model.matcher=xattn|xattn_cos and "
                    "model.matcher_cls_head=true")
        # frozen dense-flow prior for the matcher's candidate frame; registered
        # as a plain attribute (not a submodule) so it stays out of the
        # optimizer, the checkpoint and EMA
        self._flow_model = None
        if cfg.model.get("flow_ckpt"):
            from biohub046.ssl_flow.inference import load_model
            fm, _, _ = load_model(cfg.model.flow_ckpt, device="cpu")
            fm.requires_grad_(False).eval()
            object.__setattr__(self, "_flow_model", fm)

    def _flow(self):
        """The flow prior on the current device, or None when unconfigured."""
        if self._flow_model is None:
            return None
        return self._flow_model.to(self.device).eval()

    def forward(self, x):
        return self.model(x)

    # ---- train ----
    def _embed_weight(self):
        """Embedding-loss warmup: 0 until embed_warmup[0] of training, linear
        to 1 at embed_warmup[1], then 1 (user-specified schedule)."""
        lo, hi = (float(v) for v in self.cfg.loss.embed_warmup)
        frac = self.current_epoch / max(1, self.cfg.trainer.max_epochs)
        if frac < lo:
            return 0.0
        if frac >= hi:
            return 1.0
        return (frac - lo) / max(hi - lo, 1e-9)

    def training_step(self, batch, batch_idx):
        if str(self.cfg.model.get("temporal", "none")) == "siamese":
            return self._training_step_siamese(batch)
        x, ps = batch["image"], batch.get("pseudo")
        if str(self.cfg.model.get("arch", "unet25d")) == "unet25d_coarse":
            out = self.model(x)
            ld = self.loss(out, batch["coarse_cls"], batch["coarse_off"],
                           batch["coarse_mask"], batch["coarse_offw"])
            for k in ("cls_pos", "cls_bg", "off"):
                self.log(k, ld[k], on_epoch=True, sync_dist=True,
                         batch_size=len(x))
            self.log("loss", ld["loss"], on_epoch=True, prog_bar=True,
                     sync_dist=True, batch_size=len(x))
            return ld["loss"]
        a = float(self.cfg.loss.get("mixup_alpha", 0.0))
        p = float(self.cfg.loss.get("mixup_p", 1.0))
        if a > 0 and len(x) > 1 and np.random.rand() < p:
            # input mixup, two-target loss (dl_088/dl_179): predictions on the
            # mixed input are scored against BOTH original targets separately
            # and combined as lam * L(t1, m1) + (1-lam) * L(t2, m2). Targets/
            # masks are NOT linearly mixed — the carved masks (and their
            # pos/bg normalization) have no meaningful linear combination.
            lam = float(np.random.beta(a, a))
            perm = torch.randperm(len(x), device=x.device)
            out = self.model(lam * x + (1.0 - lam) * x[perm])
            loss = lam * self.loss(out, batch["target"], batch["mask"], ps)["loss"] \
                + (1.0 - lam) * self.loss(out, batch["target"][perm],
                                          batch["mask"][perm],
                                          None if ps is None else ps[perm])["loss"]
        elif self.div_weight > 0:
            out = self.model(x)
            dd = self.div_loss(out[:, 1:2], batch["div_target"], batch["div_w"])
            self.log("div_pos", dd["pos"], on_epoch=True, sync_dist=True,
                     batch_size=len(x))
            self.log("div_neg", dd["neg"], on_epoch=True, sync_dist=True,
                     batch_size=len(x))
            if self._trunk_frozen():
                # det channel is a constant on a frozen trunk — skip its loss
                loss = self.div_weight * dd["loss"]
            else:
                loss = self.loss(out[:, :1], batch["target"], batch["mask"],
                                 ps)["loss"] + self.div_weight * dd["loss"]
        elif int(self.cfg.model.get("deep_supervision", 0)) > 0:
            # multi-scale deep supervision: model returns (main, aux_s2, ...).
            # Aux targets/masks are pooled ON GPU per level k (factor 2^(k+1)):
            # max-pool keeps the Gaussian peak amplitude, min-pool keeps the
            # carve conservative (same choice as the coarse-line min-pool).
            assert ps is None, "deep_supervision + pseudo labels unsupported"
            outs = self.model(x)
            ld = self.loss(outs[0], batch["target"], batch["mask"])
            loss = ld["loss"]
            w = [float(v) for v in self.cfg.loss.get("ds_weights", [0.5, 0.25])]
            assert len(w) == len(outs) - 1
            for k, (aux, wk) in enumerate(zip(outs[1:], w)):
                f = 2 ** (k + 1)
                tgt = torch.nn.functional.max_pool3d(batch["target"], f)
                msk = -torch.nn.functional.max_pool3d(
                    -batch["mask"].unsqueeze(1), f)[:, 0]
                la = self.loss(aux, tgt, msk)["loss"]
                self.log(f"ds{f}", la, on_epoch=True, sync_dist=True,
                         batch_size=len(x))
                loss = loss + wk * la
        else:
            out = self.model(x)
            # dl_622 data.distill_mode=combine: the soft teacher target rides
            # in the batch and is summed into the loss (train only; the val
            # dataset has no teacher target)
            ld = self.loss(out, batch["target"], batch["mask"], ps,
                           batch.get("distill_target"))
            loss = ld["loss"]
            if "ps" in ld:
                self.log("ps_loss", ld["ps"], on_epoch=True, sync_dist=True,
                         batch_size=len(x))
            if "distill" in ld:
                self.log("distill", ld["distill"], on_epoch=True,
                         sync_dist=True, batch_size=len(x))
        self.log("loss", loss, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=len(x))
        return loss

    def _training_step_siamese(self, batch):
        from .embed_loss import embed_link_loss
        img = batch["image"]                       # (B, 2, Z, Y, X)
        B = img.shape[0]
        hm, emb = self.model(img.reshape(B * 2, 1, *img.shape[2:]))
        freeze = bool(self.cfg.model.get("freeze_trunk", False)) \
            or bool(self.cfg.loss.get("embed_only", False))
        if freeze:
            # embedding-only training: frozen trunk (det loss constant) or a
            # dedicated embedding model (loss.embed_only, no det objective)
            det = None
            loss = 0.0
        else:
            # detection: both frames supervised, folded into the batch dim
            det = self.loss(hm, batch["target"].reshape(B * 2, 1, *img.shape[2:]),
                            batch["mask"].reshape(B * 2, *img.shape[2:]))
            loss = det["loss"]
        w = self._embed_weight()
        self.log("emb_w", w, on_epoch=True, sync_dist=True, batch_size=B)
        if w > 0:
            hm2 = None if hm is None else hm.detach().reshape(B, 2, *img.shape[2:])
            emb2 = emb.reshape(B, 2, emb.shape[1], *emb.shape[2:])
            eld = embed_link_loss(
                self.model, hm2, emb2, batch, self.cfg,
                gt_fill=bool(self.cfg.loss.get("embed_gt_fill", False)),
                flow_model=self._flow())
            loss = loss + w * float(self.cfg.loss.embed_weight) * eld["loss"]
            self.log("emb_loss", eld["loss"], on_epoch=True, sync_dist=True,
                     batch_size=B)
            self.log("emb_acc", eld["acc"], on_epoch=True, prog_bar=True,
                     sync_dist=True, batch_size=B)
            self.log("emb_null", eld["null_frac"], on_epoch=True, sync_dist=True,
                     batch_size=B)
            self.log("emb_n", float(eld["n"]), on_epoch=True, sync_dist=True,
                     batch_size=B)
            if self.matcher_cls_weight > 0:
                loss = loss + w * self.matcher_cls_weight * eld["cls_loss"]
                self.log("cls_loss", eld["cls_loss"], on_epoch=True,
                         sync_dist=True, batch_size=B)
        elif det is None:
            # loss is a plain float 0.0 here: no det objective and no embed
            # weight yet -> every step would be dead compute (and Lightning
            # rejects a non-tensor training_step output). Fail loudly.
            raise RuntimeError(
                "no trainable loss: freeze_trunk/embed_only with embed weight 0 "
                "(set loss.embed_warmup=[0, 0])")
        self.log("loss", loss, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=B)
        if det is not None:
            self.log("det_loss", det["loss"], on_epoch=True, sync_dist=True,
                     batch_size=B)
        return loss

    def _trunk_frozen(self):
        return bool(self.cfg.model.get("freeze_trunk", False)) \
            and str(self.cfg.model.get("temporal", "none")) == "none"

    def on_train_epoch_start(self):
        if self._trunk_frozen():
            # Lightning flips the whole module to train() before each epoch;
            # keep the trunk (incl. BN running stats) bit-identical to the
            # resume ckpt — only the div head actually trains (dl_195)
            self.model.eval()
            self.model.div_head.train()

    def on_train_batch_end(self, out, batch, batch_idx):
        if self.cfg.model.ema:
            self.model_ema.update(self.model, step=self.global_step)

    # ---- validation: heatmap loss + node recall/precision + mini edge metric ----
    # Contiguous val windows (CellValDataset): per-frame detections are collected,
    # linked per sample, and scored as UNADJUSTED edge Jaccard vs the windowed GT.
    def _embed_val(self):
        return str(self.cfg.model.get("temporal", "none")) == "siamese" \
            and bool(self.cfg.loss.get("embed_only", False))

    def on_validation_epoch_start(self):
        self.val_stats = np.zeros(3, np.int64)  # matched, n_pred, n_gt
        self.val_dets = []                      # (sid, tw, (M,3) int zyx)
        self.val_emb = np.zeros(4, np.float64)  # loss*n, correct, null*n, n
        self.val_div = []                       # (scores, classes) per frame
        self.val_cls = []                       # (probs (N,3), labels) per pair

    def _eval_model(self):
        return self.model_ema.module if self.cfg.model.ema else self.model

    def validation_step(self, batch, batch_idx):
        # fp32 regardless of trainer.precision. Under "16-mixed" the val
        # forward runs in autocast, and detection val is a DISCRETE decision on
        # top of it (threshold + local-max NMS), which the training loss never
        # makes: fp16 rounding near det_threshold manufactures peaks. Measured
        # on dl_079 fold0 last: val_edge 0.8682 (fp16) vs 0.8957 (fp32),
        # n_pred +3.2%. Worse, the bias is not uniform across checkpoints — on
        # dl_027 it FLIPPED the order of ep089 and last, which is exactly a
        # -0.017 ckpt-selection miss.
        # NOTE: this shifts val_edge ~+0.02 versus every value logged before
        # 2026-08-22; compare across that boundary with care.
        with torch.autocast(self.device.type, enabled=False):
            return self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx):
        if self._embed_val():
            return self._validation_step_embed(batch)
        out = self._eval_model()(batch["image"])
        if isinstance(out, tuple):                 # siamese: (heatmap, embedding)
            out = out[0]
        ds = get_downsample(self.cfg)
        if self.div_weight > 0:
            # daughter scores at GT node positions (+-1 voxel max, the same
            # tolerance inference uses at ch1 peaks); IGN nodes are dropped
            dm = torch.nn.functional.max_pool3d(out[:, 1:2].float(), 3, 1, 1)[:, 0]
            shp = np.array(dm.shape[1:])
            for b in range(len(dm)):
                cls = batch["div_cls"][b].cpu().numpy()
                keep = cls != DIV_IGN
                if not keep.any():
                    continue
                pts = np.rint(to_ds(batch["gt_zyx"][b].cpu().numpy(),
                                    ds)).astype(int)
                pts = np.clip(pts, 0, shp - 1)[keep]
                sc = dm[b][pts[:, 0], pts[:, 1], pts[:, 2]].cpu().numpy()
                self.val_div.append((sc, cls[keep]))
        # div-only extra frames (tw = -1) feed ONLY the daughter metric — keep
        # them out of val_loss and the window edge metric
        if int(batch["tw"][0]) < 0:
            return
        coarse = str(self.cfg.model.get("arch", "unet25d")) == "unet25d_coarse"
        if not coarse:
            # coarse: CellValDataset still yields full-res heatmap targets,
            # which don't apply — val_edge (the monitor) needs no val_loss
            ld = self.loss(out[:, :1] if self.div_weight > 0 else out,
                           batch["target"], batch["mask"])
            bs = len(batch["image"])
            self.log("val_loss", ld["loss"], on_epoch=True, sync_dist=True, batch_size=bs)
            self.log("val_loss_pos", ld["pos"], on_epoch=True, sync_dist=True, batch_size=bs)
            self.log("val_loss_bg", ld["bg"], on_epoch=True, sync_dist=True, batch_size=bs)
            if "flow" in ld:
                self.log("val_loss_flow", ld["flow"], on_epoch=True, sync_dist=True,
                         batch_size=bs)
        for b in range(len(batch["image"])):
            if coarse:
                peaks = decode_coarse(
                    out[b].float(), self.cfg.model.det_threshold,
                    self.cfg.model.nms_kernel,
                    logit_space=str(self.cfg.model.get(
                        "coarse_target", "binary")) != "gauss")
            else:
                peaks = extract_peaks(out[b, 0].float(), self.cfg.model.det_threshold,
                                      self.cfg.model.nms_kernel)
                if self.cfg.test.subvoxel:
                    peaks = refine_peaks(out[b, 0].float(), peaks, refine_rad(ds))
            peaks = peaks.cpu().numpy()
            if (ds > 1).any():
                # heatmap lives on the downsampled grid; matching/linking and
                # gt_zyx are in ORIGINAL voxels
                peaks = to_orig(peaks, ds)
            gt = batch["gt_zyx"][b].cpu().numpy()
            pairs = hungarian_pairs(peaks.astype(np.float64), gt.astype(np.float64))
            self.val_stats += (len(pairs), len(peaks), len(gt))
            self.val_dets.append((batch["sid"][b], int(batch["tw"][b]),
                                  peaks.astype(np.float64)))

    def _validation_step_embed(self, batch):
        """val_emb_acc: embedding argmax accuracy on deterministic full-frame
        val pairs with champion candidates (drop_p=0, no candidate cap)."""
        from .embed_loss import embed_link_loss
        img = batch["image"]                       # (1, 2, Z, Y, X)
        B = img.shape[0]
        # fp32: full-frame (~6k token) matcher attention NaN'd under the val
        # autocast (dl_063/067); offline eval runs fp32 too, so this also
        # keeps val_emb_acc consistent with it
        with torch.autocast("cuda", enabled=False):
            m = self._eval_model()   # same weights for forward AND matcher/
            hm, emb = m(img.reshape(B * 2, 1, *img.shape[2:]))
            eld = embed_link_loss(   # logit_scale/null (EMA-consistent)
                m, None if hm is None else hm.detach().reshape(B, 2, *img.shape[2:]),
                emb.reshape(B, 2, emb.shape[1], *emb.shape[2:]), batch, self.cfg,
                drop_p=0.0, max_peaks=1 << 30, flow_model=self._flow())
        n = eld["n"]
        self.val_emb += (float(eld["loss"]) * n, eld["acc"] * n,
                         eld["null_frac"] * n, n)
        if self.matcher_cls_weight > 0:
            self.val_cls.append((eld["cls_probs"], eld["cls_true"]))

    def on_validation_epoch_end(self):
        if self._embed_val():
            s = np.sum(_gather(self.val_emb), axis=0)
            n = max(s[3], 1.0)
            self.log("val_emb_acc", s[1] / n, prog_bar=True, sync_dist=False)
            self.log("val_emb_loss", s[0] / n, sync_dist=False)
            self.log("val_emb_null", s[2] / n, sync_dist=False)
            if self.trainer.is_global_zero:
                print(f"[val] epoch {self.current_epoch} "
                      f"val_emb_acc={s[1]/n:.4f} val_emb_loss={s[0]/n:.4f} "
                      f"null={s[2]/n:.4f} (n={int(s[3])})")
            if self.matcher_cls_weight > 0:
                parts = [p for part in _gather(self.val_cls) for p in part]
                pr = (np.concatenate([p for p, _ in parts])
                      if parts else np.zeros((0, 3)))
                tg = (np.concatenate([t for _, t in parts])
                      if parts else np.zeros(0, np.int64))
                msg = []
                for c, name in ((DIV_POS, "dau"), (DIV_PAR, "par")):
                    auc, ap = binary_auc_ap(pr[:, c], (tg == c).astype(int))
                    self.log(f"val_cls_{name}_auc", auc, sync_dist=False)
                    self.log(f"val_cls_{name}_ap", ap, sync_dist=False)
                    msg.append(f"{name}_auc={auc:.4f} ap={ap:.4f} "
                               f"pos={int((tg == c).sum())}")
                if self.trainer.is_global_zero:
                    print(f"[val] epoch {self.current_epoch} cls "
                          + " ".join(msg) + f" (n={len(tg)})")
            return
        stats = np.sum(_gather(self.val_stats), axis=0)
        matched, n_pred, n_gt = (int(v) for v in stats)
        recall = matched / n_gt if n_gt else 0.0
        prec = matched / n_pred if n_pred else 0.0
        # precision is deflated by sparse GT (~97% of true cells unlabeled)
        # — meaningful only as a relative signal across epochs.
        self.log("val_recall", recall, sync_dist=False)
        self.log("val_prec", prec, sync_dist=False)

        # mini edge metric: link detections per sample window, score vs windowed GT
        dets = [d for part in _gather(self.val_dets) for d in part]
        ds = self.trainer.datamodule.val_dataset
        by_sid = {}
        for sid, tw, pk in dets:
            by_sid.setdefault(sid, {})[tw] = pk
        rows = []
        for sid, frames in by_sid.items():
            per_t = [frames.get(tw, np.zeros((0, 3), np.int32)) for tw in range(ds.W)]
            nodes, edges = build_graph(per_t, self.cfg.test.max_link_um,
                                       self.cfg.test.drift_comp,
                                       self.cfg.test.drift_iters)
            nodes, edges = filter_short_tracks(nodes, edges,
                                               self.cfg.test.min_track_len)
            rows.append(eval_sample(nodes, edges, ds.gt_win[sid]))
        # same aggregation as the full eval (adjusted, per-sample weighted);
        # only the sample/frame coverage differs from the full evaluation
        s = summarise(rows) if rows else None
        val_edge = 0.0
        if s is not None:
            for key in ("score", "edge_jaccard"):
                if np.isfinite(s[key]):
                    val_edge = float(s[key])
                    break
        self.log("val_edge", val_edge, prog_bar=True, sync_dist=False)
        if self.trainer.is_global_zero and s is not None:
            print(f"[val] epoch {self.current_epoch} score={val_edge:.4f} "
                  f"(raw edge={s['edge_jaccard']:.4f} TP/FP/FN={s['tp']}/{s['fp']}/{s['fn']}) "
                  f"recall={recall:.4f} prec={prec:.4f} (pred={n_pred:,} gt={n_gt:,})")
        if self.div_weight > 0:
            frames = [f for part in _gather(self.val_div) for f in part]
            sc = np.concatenate([s_ for s_, _ in frames]) if frames else np.zeros(0)
            cl = np.concatenate([c for _, c in frames]) if frames else np.zeros(0)
            auc, ap = binary_auc_ap(sc, cl)
            self.log("val_div_auc", auc, sync_dist=False)
            self.log("val_div_ap", ap, prog_bar=True, sync_dist=False)
            if self.trainer.is_global_zero:
                print(f"[val] epoch {self.current_epoch} div_auc={auc:.4f} "
                      f"div_ap={ap:.4f} (pos={int((cl == 1).sum())} n={len(cl)})")

    # ---- optimizer / scheduler (verbatim from kaggle-rogii template) ----
    @staticmethod
    def split_decay_groups(model: nn.Module):
        norm_modules = (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
            nn.LayerNorm, nn.GroupNorm,
            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
        )
        decay, no_decay = [], []
        for m in model.modules():
            for name, p in m.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if (name == "bias") or (isinstance(m, norm_modules) and name in ("weight", "bias")):
                    no_decay.append(p)
                else:
                    decay.append(p)
        return decay, no_decay

    def configure_optimizers(self):
        decay, no_decay = self.split_decay_groups(self.model)
        opt_kwargs = dict(self.cfg.opt)
        matcher_lr = opt_kwargs.pop("matcher_lr", None)
        param_groups = [
            dict(params=decay, weight_decay=self.cfg.opt.weight_decay),
            dict(params=no_decay, weight_decay=0.0),
        ]
        if matcher_lr is not None:
            # dl_421: the scratch-initialised head (NodeMatcher + the InfoNCE
            # scalars) trains at its own lr while a pretrained trunk keeps
            # opt.lr. timm's schedulers scale every group from its own base lr.
            head = {id(p) for n, p in self.model.named_parameters()
                    if n.startswith("matcher.") or n.startswith("embed_")}
            split = []
            for g in param_groups:
                trunk = [p for p in g["params"] if id(p) not in head]
                hp = [p for p in g["params"] if id(p) in head]
                if trunk:
                    split.append(dict(g, params=trunk))
                if hp:
                    split.append(dict(g, params=hp, lr=float(matcher_lr)))
            param_groups = split
            print(f"opt.matcher_lr={matcher_lr}: {sum(len(g['params']) for g in param_groups if 'lr' in g)} "
                  f"head tensors at {matcher_lr}, trunk at {self.cfg.opt.lr}", flush=True)
        optimizer = create_optimizer_v2(model_or_params=param_groups, **opt_kwargs)
        batch_size = self.cfg.data.batch_size
        updates_per_epoch = (len(self.trainer.datamodule.train_dataset) // batch_size
                             // self.trainer.num_devices // self.cfg.trainer.accumulate_grad_batches)
        if updates_per_epoch == 0:
            updates_per_epoch = 1
            print("Warning: updates_per_epoch is set to 1 to avoid zero division.")
        scheduler, _ = create_scheduler_v2(
            optimizer=optimizer, num_epochs=self.cfg.trainer.max_epochs,
            warmup_lr=0, **self.cfg.scheduler,
            step_on_epochs=False, updates_per_epoch=updates_per_epoch,
        )
        lr_dict = dict(scheduler=scheduler, interval="step", frequency=1)
        return dict(optimizer=optimizer, lr_scheduler=lr_dict)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step_update(num_updates=self.global_step)
