"""InfoNCE + null-class embedding loss for the siamese linking model.

Design:
- Anchors/candidates are the model's OWN detections (threshold + NMS on the
  predicted heatmaps), assigned to GT nodes by <=7 µm Hungarian matching —
  the official matching radius; wider only adds wrong-cell positives.
- Per anchor with a GT successor, an InfoNCE row over: cross-frame candidates
  within embed_radius_um, same-frame neighbours within the same radius as
  extra negatives, and a learnable null ("no link") logit.
    * successor detected            -> target = its detection
    * successor GT outside the crop -> target = null (candidate set genuinely
      lacks the answer; same situation as a volume-boundary exit at test time)
    * successor GT inside but undetected -> anchor skipped (positive absent)
    * with prob embed_drop_p the positive is masked out and target = null
      (simulates a missed detection: "prefer null over the second-best")
- Divisions: one row per child; the sibling positive is masked out of the
  row's negatives (multi-positive handling).
- Bidirectional: the same construction runs frame1 -> frame0.
- Embeddings are L2-normalized in float32; logits = exp(logit_scale) * cos,
  null logit unscaled (both learnable, owned by the model).
"""
import numpy as np
import torch
import torch.nn.functional as F

from .dataset import DIV_IGN, DIV_NEG, DIV_PAR, DIV_POS, get_downsample
from .metric import (SCALE, estimate_collective_drift, extract_peaks,
                     hungarian_pairs)
from biohub046.ssl_flow.ops import sample_volume

NEG_INF = -1e4
MAX_PEAKS = 384   # per frame per crop; top-by-heatmap beyond this (safety cap)


def sample_embeddings(emb, pos, crop_shape):
    """Trilinear-sample embedding vectors at model-input-grid coords.

    emb: (C, z', y', x') map on a strided grid covering the same crop window;
    pos: (N, 3) float zyx on the SAME grid as crop_shape — the model-input
    grid, i.e. the ds grid when data.downsample is active (orig-voxel coords
    must be to_ds-converted first); crop_shape: (Z, Y, X) of that grid.
    align_corners=False + voxel-center normalization against the crop_shape
    extent maps both grids onto the same physical window."""
    if len(pos) == 0:
        return emb.new_zeros((0, emb.shape[0]))
    sz = torch.tensor(crop_shape, device=emb.device, dtype=torch.float32)
    g = (2.0 * (pos.to(emb.device).float() + 0.5) / sz - 1.0).flip(-1)  # zyx->xyz
    out = F.grid_sample(emb[None].float(), g[None, :, None, None, :],
                        align_corners=False)                # (1,C,N,1,1)
    return out[0, :, :, 0, 0].T                             # (N, C)


def _detect_and_match(hm, gt_pos, threshold, nms_kernel, provided=None,
                      max_peaks=MAX_PEAKS, *, ds):
    """Candidate points of one crop + <=7 µm Hungarian assignment to GT rows.
    Candidates are the crop heatmap's own peaks, or `provided` float coords
    (champion detections, loss.embed_only). ds (required): downsample factors
    — coords on a ds grid are rescaled to original-voxel units for the µm
    matching gate (hungarian_pairs applies full-res SCALE internally).
    Returns (points (M,3) float tensor, {gt_row: det_idx})."""
    if provided is not None:
        peaks = provided
        if len(peaks) > max_peaks:
            peaks = peaks[torch.randperm(len(peaks))[:max_peaks]]
    else:
        peaks = extract_peaks(hm.detach().float(), threshold, nms_kernel)
        if len(peaks) > max_peaks:
            v = hm.detach().float()[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
            peaks = peaks[torch.topk(v, max_peaks).indices]
    peaks = peaks.float()
    if len(peaks) == 0 or len(gt_pos) == 0:
        return peaks, {}
    f = np.asarray(ds, np.float64)
    pairs = hungarian_pairs(peaks.cpu().numpy().astype(np.float64) * f,
                            gt_pos.cpu().numpy().astype(np.float64) * f)
    return peaks, {int(g): int(d) for d, g in pairs}


def _gt_fill(peaks, det_of, gt_pos, rows, crop_shape):
    """Revive skipped positives (loss.embed_gt_fill): GT nodes that take part
    in a link but got no detection within 7 µm are appended as extra candidate
    points at their GT coordinate (train-time positive-pair augmentation; the
    detector misses ~2.3% of GT, so ~4.6% of links lose an endpoint and are
    silently skipped). Injected rows flow through the normal row construction,
    so embed_drop_p null-teaching applies to them like any other positive.
    In-crop test uses the block-center window (cf. the boundary fix).
    Also returns the list of appended GT rows (in append order) so callers can
    align per-node side data (e.g. flow offsets) with the extra candidates."""
    lo = -0.5
    hi = torch.tensor(crop_shape, dtype=torch.float32,
                      device=gt_pos.device) - 0.5
    extra, added = [], []
    for r in rows:
        if r in det_of:
            continue
        p = gt_pos[r]
        if bool((p >= lo).all()) and bool((p < hi).all()):
            det_of[r] = len(peaks) + len(extra)
            extra.append(p)
            added.append(r)
    if extra:
        peaks = torch.cat([peaks, torch.stack(extra).to(peaks.device).float()])
    return peaks, det_of, added


def _rows_one_direction(logits_x, logits_same, null_vec, links_st,
                        det_of_src, det_of_tgt, outside_tgt, drop_p,
                        pos_far=None):
    """InfoNCE rows for one direction. links_st: list of (src_row, tgt_row);
    outside_tgt: python list of bools (tgt GT outside the crop), precomputed —
    per-row GPU syncs here were the training-throughput killer.
    null_vec: (N_src,) per-anchor null logits (matcher) or an expanded scalar
    (cosine mode). Row layout: [cross candidates | same-frame negatives | null].
    Returns (rows (R,K) tensor or None, targets tensor, n_null)."""
    null_idx = logits_x.shape[1] + logits_same.shape[1]
    by_src = {}
    for s, t in links_st:
        by_src.setdefault(s, []).append(t)
    das, targets, kill = [], [], []        # kill: (row, col) -> NEG_INF
    n_null = 0
    for s, t in links_st:
        da = det_of_src.get(s)
        if da is None:
            continue
        db = det_of_tgt.get(t)
        if db is None and not outside_tgt[t]:
            continue                       # successor undetected inside crop
        r = len(das)
        for sib in by_src[s]:              # mask sibling positives (divisions)
            if sib != t and det_of_tgt.get(sib) is not None:
                kill.append((r, det_of_tgt[sib]))
        if db is None:                     # crop-boundary exit
            target = null_idx
        elif pos_far is not None and bool(pos_far[da, db]):
            # true successor beyond the candidate radius (large drift):
            # within the candidate universe the truth IS "no link" — teaching
            # the (NEG_INF-masked) positive exploded the CE (dl_063 train-loss
            # spikes of 250-450 and the offline<<val discrepancy)
            target = null_idx
        elif np.random.rand() < drop_p:    # simulated missed detection
            kill.append((r, db))
            target = null_idx
        else:
            target = db
        n_null += target == null_idx
        das.append(da)
        targets.append(target)
    if not das:
        return None, None, 0
    di = torch.tensor(das, device=logits_x.device)
    rows = torch.cat([logits_x[di], logits_same[di], null_vec[di][:, None]],
                     dim=1)
    if kill:
        ki = torch.tensor(kill, device=rows.device)
        rows[ki[:, 0], ki[:, 1]] = NEG_INF
    return rows, torch.tensor(targets, device=rows.device), n_null


def embed_link_loss(model, hm, emb, batch, cfg, drop_p=None, max_peaks=None,
                    gt_fill=False, flow_model=None):
    """hm: (B, 2, Z, Y, X) predicted heatmaps (grad not needed; None with
    model.embed_only_forward — champion candidates only), emb:
    (B, 2, C, z', y', x'). batch carries per-item pa/pb (transformed GT node
    positions, crop coords) and links (L, 2) row-index pairs.
    drop_p / max_peaks overrides: validation uses drop_p=0 (deterministic,
    no simulated misses) and an effectively unlimited candidate cap.
    gt_fill: TRAIN-only positive-pair augmentation — see _gt_fill (validation
    keeps pure detections for comparability with inference).
    Returns dict(loss, acc, null_frac, n) — zeros when no rows materialize."""
    lc = cfg.loss
    radius = float(lc.embed_radius_um)
    if max_peaks is None:
        # train-time candidate cap (feat/embed-max-peaks, dl_127/128). The 384
        # default silently thins the DENSEST frames (p90 count 658, max 865 ->
        # ~20-28% of items get a random subsample), a train/inference mismatch
        # concentrated exactly on crowded frames; loss.embed_max_peaks=1024
        # disables it in practice (the dl_128 base recipe)
        max_peaks = int(lc.get("embed_max_peaks", MAX_PEAKS))
    if drop_p is None:
        drop_p = float(lc.embed_drop_p)
    th = float(cfg.model.det_threshold)
    nms = cfg.model.nms_kernel
    champion = bool(cfg.data.get("champion_det_dir"))   # candidates from cache
    # crop coords live on the (possibly downsampled) grid: µm per ds voxel is
    # SCALE * ds, and the 7 µm hungarian gate gets coords rescaled to
    # original-voxel units (relative distances are offset-free)
    ds = get_downsample(cfg)
    scale = torch.tensor(SCALE * ds, dtype=torch.float32)
    logit_scale = model.embed_logit_scale.exp().clamp(max=100.0)
    null_logit = model.embed_null

    # matcher node-cls loss (loss.matcher_cls_weight): 3-way per-node CE
    # (other/daughter/parent) on the matcher's final cross-attended features,
    # supervised ONLY at detections matched to a GT node (the same <=7 µm
    # assignment as the InfoNCE anchors) — unmatched detections carry no
    # teacher under sparse GT, IGN rows skipped
    want_cls = (getattr(model, "matcher", None) is not None
                and getattr(model.matcher, "cls_head", None) is not None
                and "cls_a" in batch)
    cls_logits, cls_targets = [], []
    # loss.matcher_cls_det_neg > 0: UNMATCHED candidates (unlabeled-cell
    # suspects) join as weak "other" teachers (dl_196 pattern; dl_197
    # measured unmatched detections floating toward daughter, p_dau median
    # 0.142 vs 0.058 matched), sparing dets within 7 µm of a daughter/parent
    # GT node (duplicate/split detections of the positive itself)
    cls_weak = []
    cls_det_neg = float(lc.get("matcher_cls_det_neg", 0.0))

    total, correct, n_rows, n_null = 0.0, 0, 0, 0
    r2_total, r2_rows, r2_correct = 0.0, 0, 0
    r2w = float(lc.get("r2_sim_weight", 0.0))
    img = batch["image"]                       # (B, 2, Z, Y, X) = hm shape
    for i in range(img.shape[0]):
        crop_shape = tuple(img.shape[2:])
        pa, pb = batch["pa"][i], batch["pb"][i]
        links = batch["links"][i]
        if len(links) == 0:
            continue
        prov_a = batch["det_a"][i] if champion else None
        prov_b = batch["det_b"][i] if champion else None
        # data.flow_cache_dir: per-node flow (µm) aligned with det_a / pa rows;
        # subsample prov_a HERE so the flow rows follow the same permutation
        # (with the recipe's max_peaks=1024 this practically never triggers)
        flow_da = batch["flow_det_a"][i] if "flow_det_a" in batch else None
        flow_pa_gt = batch["flow_pa"][i] if "flow_pa" in batch else None
        if champion and flow_da is not None and len(prov_a) > max_peaks:
            sel = torch.randperm(len(prov_a))[:max_peaks]
            prov_a, flow_da = prov_a[sel], flow_da[sel]
        hm_a = None if hm is None else hm[i, 0]
        hm_b = None if hm is None else hm[i, 1]
        pk_a, a_of = _detect_and_match(hm_a, pa, th, nms, prov_a, max_peaks,
                                       ds=ds)
        pk_b, b_of = _detect_and_match(hm_b, pb, th, nms, prov_b, max_peaks,
                                       ds=ds)
        add_a = []
        if gt_fill and len(links):
            li = links.cpu().numpy()
            pk_a, a_of, add_a = _gt_fill(pk_a, a_of, pa,
                                         set(int(v) for v in li[:, 0]),
                                         crop_shape)
            pk_b, b_of, _ = _gt_fill(pk_b, b_of, pb,
                                     set(int(v) for v in li[:, 1]),
                                     crop_shape)
        if len(pk_a) == 0 or len(pk_b) == 0:
            continue
        raw_a = sample_embeddings(emb[i, 0], pk_a.float(), crop_shape)
        raw_b = sample_embeddings(emb[i, 1], pk_b.float(), crop_shape)
        um_a = pk_a.float().cpu() * scale
        um_b = pk_b.float().cpu() * scale
        drift_frame = bool(getattr(model, "matcher_drift_frame", False))
        drift_cpu = torch.zeros(3, dtype=torch.float32)
        offset_dev = None
        if drift_frame and champion and flow_da is not None:
            # cached per-node flow (dense_flow_cache): already µm, already
            # aug-consistent (keypoint-pair transform in the dataset); append
            # the gt_fill rows' flow in the same order _gt_fill added them
            flow_cand = flow_da
            if add_a:
                flow_cand = torch.cat([flow_cand, flow_pa_gt[
                    torch.tensor(add_a, dtype=torch.long)]])
            assert len(flow_cand) == len(pk_a), (len(flow_cand), len(pk_a))
            drift_cpu = flow_cand.float().cpu()
            offset_dev = drift_cpu.to(emb.device)
        elif drift_frame and flow_model is not None:
            # dense per-node frame (flow_147: better than the collective drift
            # at inference).  The flow runs on THIS crop's own two frames, so
            # the D4 augmentation and the crop origin are carried implicitly
            # and no coordinate bookkeeping is needed; pk_a is already in crop
            # grid coords, which is the flow model's input grid.
            im = batch["image"][i]
            with torch.autocast(emb.device.type, enabled=False):
                with torch.no_grad():
                    fl = flow_model(im[0][None, None].float(),
                                    im[1][None, None].float())
            offset_dev = (sample_volume(fl, pk_a.float().to(fl.device))[0]
                          * scale.to(fl.device))
            drift_cpu = offset_dev.cpu()          # (N,3), broadcasts row-wise
        elif drift_frame:
            drift_cpu = torch.from_numpy(estimate_collective_drift(
                um_a.numpy(), um_b.numpy(),
                max_link_um=float(cfg.test.max_link_um),
                drift_iters=int(cfg.test.drift_iters))).float()
        d_ab_cpu = torch.cdist(um_a + drift_cpu, um_b)
        pos_far = (d_ab_cpu > radius).numpy()
        if getattr(model, "matcher", None) is not None:
            # end-to-end node matcher: logits from cross-attended features +
            # sparse pair MLP; per-anchor null heads; no same-frame negatives
            # (self-attention already conditions on intra-frame context)
            ga, gb = um_a.to(emb.device), um_b.to(emb.device)
            offset = ((offset_dev if offset_dev is not None
                       else drift_cpu.to(emb.device).expand(len(pk_a), 3))
                      if drift_frame else None)
            l_ab, null_a, null_b = model.matcher(
                raw_a, ga, raw_b, gb, radius, candidate_offset_a=offset)
            if want_cls:
                # gt_fill-injected candidates are in a_of/b_of too and get
                # supervised like any matched detection
                for cls_v, of, lg, gt_p, um_c in (
                        (batch["cls_a"][i], a_of, model.matcher.cls_logits_a,
                         pa, um_a),
                        (batch["cls_b"][i], b_of, model.matcher.cls_logits_b,
                         pb, um_b)):
                    cv = cls_v.tolist()  # one sync per item
                    rows = [(dv, cv[r]) for r, dv in of.items()
                            if cv[r] != DIV_IGN]
                    if rows:
                        di = torch.tensor([dv for dv, _ in rows],
                                          device=lg.device)
                        cls_logits.append(lg[di])
                        cls_targets.append(torch.tensor(
                            [c for _, c in rows], device=lg.device))
                    if cls_det_neg > 0 and len(lg) > len(of):
                        un = torch.tensor(
                            [d for d in range(len(lg))
                             if d not in set(of.values())])
                        pr = [r for r in range(len(cv))
                              if cv[r] in (DIV_POS, DIV_PAR)]
                        if pr:
                            d_pos = torch.cdist(
                                um_c[un], gt_p[pr].float().cpu() * scale)
                            un = un[d_pos.min(dim=1).values > 7.0]
                        if len(un):
                            cls_weak.append(lg[un.to(lg.device)])
                # invalidate the stash: a read after any later matcher call
                # must fail loudly rather than see this pair's logits
                model.matcher.cls_logits_a = model.matcher.cls_logits_b = None
            l_ba = l_ab.T
            l_aa = raw_a.new_zeros((len(pk_a), 0))
            l_bb = raw_b.new_zeros((len(pk_b), 0))
        else:
            ea = F.normalize(raw_a, dim=1)
            eb = F.normalize(raw_b, dim=1)
            d_ab = d_ab_cpu.to(emb.device)
            d_aa = torch.cdist(um_a, um_a).to(emb.device)
            d_bb = torch.cdist(um_b, um_b).to(emb.device)
            far_ab = (d_ab > radius) * NEG_INF
            l_ab = logit_scale * (ea @ eb.T) + far_ab
            l_ba = logit_scale * (eb @ ea.T) + far_ab.T
            l_aa = logit_scale * (ea @ ea.T) + (d_aa > radius) * NEG_INF \
                + torch.eye(len(pk_a), device=emb.device) * NEG_INF
            l_bb = logit_scale * (eb @ eb.T) + (d_bb > radius) * NEG_INF \
                + torch.eye(len(pk_b), device=emb.device) * NEG_INF
            null_a = null_logit.expand(len(pk_a))
            null_b = null_logit.expand(len(pk_b))

        links_st = [(int(s), int(t)) for s, t in links]
        # crop window in grid coords is [-0.5, hi-0.5) per axis (voxel i covers
        # [i-0.5, i+0.5)); with ds > 1 in-frame low-border GT maps to small
        # NEGATIVE ds coords (to_ds(0) = -(ds-1)/(2 ds)), so a 0 lower bound
        # would mint spurious null targets for frame-edge cells
        hi_t = torch.tensor(crop_shape, dtype=torch.float32, device=pa.device)
        out_a = ((pa < -0.5) | (pa >= hi_t - 0.5)).any(1).tolist()  # one sync per item
        out_b = ((pb < -0.5) | (pb >= hi_t - 0.5)).any(1).tolist()
        rows_f, tg_f, nn_f = _rows_one_direction(
            l_ab, l_aa, null_a, links_st, a_of, b_of, out_b, drop_p, pos_far)
        rows_r, tg_r, nn_r = _rows_one_direction(
            l_ba, l_bb, null_b, [(t, s) for s, t in links_st], b_of, a_of,
            out_a, drop_p, pos_far.T)
        for lg, tt in ((rows_f, tg_f), (rows_r, tg_r)):
            if lg is None:
                continue
            total = total + F.cross_entropy(lg.float(), tt, reduction="sum")
            correct += int((lg.argmax(1) == tt).sum())
            n_rows += len(tt)
        n_null += nn_f + nn_r

        # r2 simulation (dl_344, loss.r2_sim_weight): inference's second round
        # re-runs the matcher with the wiring-claimed frame-b nodes REMOVED
        # FROM ITS INPUT (export_oof_graphs.second_round) — a condition round-1
        # training never sees, yet its output ranks the division daughters.
        # Simulate it: per linked anchor mark one detected true successor as
        # claimed, drop the claimed set from the b input, and supervise the
        # re-run — a division parent's target becomes its surviving in-radius
        # sibling daughter, everyone else's becomes null.
        if r2w > 0 and getattr(model, "matcher", None) is not None:
            by_src = {}
            for s, t in links_st:
                if a_of.get(s) is not None and b_of.get(t) is not None:
                    by_src.setdefault(s, []).append(t)
            claimed = set()
            for s, ts in by_src.items():
                pick = ts[np.random.randint(len(ts))] if len(ts) > 1 else ts[0]
                claimed.add(b_of[pick])
            keep = [j for j in range(len(pk_b)) if j not in claimed]
            if by_src and keep:
                new_idx = {j: k for k, j in enumerate(keep)}
                ki = torch.tensor(keep, device=raw_b.device)
                l2, n2a, _ = model.matcher(raw_a, ga, raw_b[ki], gb[ki],
                                           radius, candidate_offset_a=offset)
                if want_cls:   # keep the stash-invalidation contract intact
                    model.matcher.cls_logits_a = None
                    model.matcher.cls_logits_b = None
                null_col = l2.shape[1]
                das, tgs = [], []
                for s, ts in by_src.items():
                    da = a_of[s]
                    tgt = null_col
                    for t in ts:
                        db = b_of[t]
                        # surviving sibling only counts if inside the radius —
                        # a NEG_INF-masked CE target explodes (dl_063 lesson)
                        if db not in claimed and not bool(pos_far[da, db]):
                            tgt = new_idx[db]
                            break
                    das.append(da)
                    tgs.append(tgt)
                di = torch.tensor(das, device=l2.device)
                rows2 = torch.cat([l2[di], n2a[di][:, None]], 1)
                tt2 = torch.tensor(tgs, device=l2.device)
                r2_total = r2_total + F.cross_entropy(rows2.float(), tt2,
                                                      reduction="sum")
                r2_correct += int((rows2.argmax(1) == tt2).sum())
                r2_rows += len(tt2)

    if n_rows == 0:
        # graph-connected zero (rare path; keeps DDP/backward happy)
        res = dict(loss=emb.sum() * 0.0, acc=0.0, null_frac=0.0, n=0)
    else:
        res = dict(loss=total / n_rows, acc=correct / n_rows,
                   null_frac=n_null / n_rows, n=n_rows)
    if r2w > 0 and r2_rows > 0:
        res["loss"] = res["loss"] + r2w * (r2_total / r2_rows)
        res["r2_acc"] = r2_correct / r2_rows
    if want_cls:
        # per-class separate means (DivLoss pattern): ~300 daughters / ~150
        # parents must not drown in the ~133k continuation negatives.
        # cls_probs/cls_true feed only the val AUC/AP; training reads just
        # cls_loss, so skip the softmax + GPU->CPU sync when grad is on
        want_probs = not torch.is_grad_enabled()
        if cls_logits:
            lg = torch.cat(cls_logits).float()
            tg = torch.cat(cls_targets)
            ce = F.cross_entropy(lg, tg, reduction="none")
            w_other = float(lc.get("matcher_cls_other_weight", 1.0))
            cl = lg.sum() * 0.0
            for c, wc in ((DIV_POS, 1.0), (DIV_PAR, 1.0), (DIV_NEG, w_other)):
                sel = tg == c
                if bool(sel.any()):
                    cl = cl + wc * ce[sel].mean()
            if cls_weak:
                lg_w = torch.cat(cls_weak).float()
                cl = cl + cls_det_neg * F.cross_entropy(
                    lg_w, torch.full((len(lg_w),), DIV_NEG, dtype=torch.long,
                                     device=lg_w.device))
            res["cls_loss"] = cl
            if want_probs:
                res.update(cls_probs=torch.softmax(lg.detach(), 1)
                           .cpu().numpy(),
                           cls_true=tg.cpu().numpy())
        else:
            # graph-connected zero THROUGH the cls head (not the trunk) so
            # DDP's reducer still gets grads for its params on empty steps
            res["cls_loss"] = sum(
                p.sum() for p in model.matcher.cls_head.parameters()) * 0.0
            if want_probs:
                res.update(cls_probs=np.zeros((0, 3), np.float32),
                           cls_true=np.zeros(0, np.int64))
    return res
