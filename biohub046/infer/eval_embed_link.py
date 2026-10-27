"""Offline evaluation of the siamese embedding linker (dl_057+).

Runs full-frame inference on the val fold (detections + per-peak embeddings),
then reports:

A. Link-accuracy proxy on normal both-detected GT edges:
     - embedding argmax over candidates within embed_radius_um
     - drift-compensated NN baseline (the 98.25% oracle to beat)
     - the production Hungarian (97.18% on dl_028)
   plus accuracy on the subset where the NN is WRONG (the 449-case pool the
   embedding must rescue).

B. Official metric with the cost-additive wiring (design decision: floor-
   preserving): Hungarian on cost = dist_drift + lambda * (1 - cos), gate on
   the distance term only, drift iterations unchanged. lambda = 0 reproduces
   the production linker; sweep over test.embed_lambdas.

Run:
  python -m biohub046.infer.eval_embed_link model.resume_path=<ckpt>
Optional: --lambdas "0,0.5,1,2,4" --limit N

Hybrid wiring (--det_ckpt): detections come from a SEPARATE detector's
heatmaps (e.g. the dl_028 champion) and the siamese model only supplies
embeddings sampled at those peaks — sidesteps the joint-training detection
degradation. lambda=0 then reproduces the detector's own production score.
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment


from biohub046.core.dataset import CellStore, get_downsample, to_ds
from biohub046.core.embed_loss import sample_embeddings
from biohub046.core.matcher import NEG_INF
from biohub046.core.metric import (SCALE, estimate_collective_drift, eval_sample,
                        extract_peaks, filter_short_tracks, hungarian_pairs,
                        summarise)
from biohub046.core.model import get_model_from_cfg


def restore_cfg(base_config, opts, ckpt_path):
    """Config with the model section restored from a checkpoint (same
    precedence as 21_test: CLI > checkpoint > yaml)."""
    cfg = OmegaConf.load(base_config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(opts))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp_cfg = ckpt.get("hyper_parameters", {}).get("cfg")
    if hp_cfg is not None:
        hp = OmegaConf.create(hp_cfg)
        cfg.model = OmegaConf.merge(cfg.model, hp.model)
        cfg.data.downsample = hp.data.get("downsample", [1, 1, 1])
        cfg.model.temporal = hp.model.get("temporal", "none")
        if hp.model.get("matcher_drift_frame") is None:
            # ckpt predates the key = raw-frame training; the default-true
            # yaml must not evaluate it in the drift frame (CLI still wins)
            cfg.model.matcher_drift_frame = False
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(opts))
    cfg.model.resume_path = ckpt_path
    return cfg


@torch.no_grad()
def detect_sample_embed(model, store, sid, cfg, device, det_model=None,
                        det_peaks=None):
    """Full-frame inference -> per-t (peaks (M,3) int, emb (M,C) unit float32).
    det_model (hybrid wiring): peaks come from its heatmap instead of the
    siamese model's own; embeddings are sampled at those peaks.
    det_peaks: precomputed per-t peak arrays (champion_dets cache) —
    skips the detector forward entirely (halves hybrid-eval inference)."""
    # ds != 1: frames (and the embedding map) live on the ds grid, while
    # champion peaks / GT / linking stay in ORIGINAL voxels — peaks are
    # converted with to_ds only for embedding sampling. Own-peak extraction
    # is not ds-threaded (would need to_orig), so the cache path is required.
    ds = get_downsample(cfg)
    ds_on = bool((ds != 1).any())
    assert not ds_on or det_peaks is not None, \
        "downsample!=1 eval needs the champion det cache (peaks in orig voxels)"
    T = store.shape[sid][0]
    bs = int(cfg.test.frames_per_batch)
    th = float(cfg.model.det_threshold)
    nms = cfg.model.nms_kernel
    if hasattr(model, "embed_only_forward"):
        # det cache: the heatmap is never read -> decoder stops at the tap
        model.embed_only_forward = det_peaks is not None
    out = []
    for t0 in range(0, T, bs):
        frames = [store.frame(sid, t) for t in range(t0, min(T, t0 + bs))]
        xt = torch.from_numpy(np.stack(frames))[:, None].to(device)
        with torch.autocast(device.type if hasattr(device, "type") else device,
                            dtype=torch.float16,
                            enabled=bool(cfg.test.get("amp", False))):
            hm, emb = model(xt)
        emb = emb.float()
        if det_peaks is None and det_model is not None:
            hm = det_model(xt)
            if isinstance(hm, tuple):
                hm = hm[0]
        if hm is not None:
            hm = hm.float()
        for b in range(xt.shape[0]):
            if det_peaks is not None:
                pk_np = det_peaks[t0 + b].astype(np.int32)
                # ds_on: cached orig-voxel peaks -> ds grid for sampling only
                # (the assert above makes this the only ds_on branch)
                pos_np = to_ds(pk_np, ds) if ds_on else pk_np
                pos = torch.from_numpy(
                    pos_np.astype(np.float32)).to(device)
            else:
                peaks = extract_peaks(hm[b, 0], th, nms)
                pk_np = peaks.cpu().numpy().astype(np.int32)
                pos = peaks.float()
            # cache RAW embeddings: the matcher consumes raw features; the
            # cosine paths normalize at use time (idempotent on old caches)
            ev = sample_embeddings(emb[b], pos, tuple(xt.shape[2:]))
            out.append((pk_np, ev.cpu().numpy().astype(np.float32)))
    return out


def estimate_boundary_drifts(per_t, max_link_um, drift_iters):
    """GT-free production-style collective drift for every frame boundary."""
    pos = [np.asarray(p, np.float64).reshape(-1, 3) * SCALE for p, _ in per_t]
    return [estimate_collective_drift(ap, bp, max_link_um, drift_iters)
            for ap, bp in zip(pos[:-1], pos[1:])]


@torch.no_grad()
def build_matcher_scores(per_t, model, cfg, device, candidate_drifts=None):
    """Raw pair/null logits for each frame boundary.

    Unlike the old probability-only bridge, this preserves both null heads so
    inference can make an explicit birth/death decision. candidate_drifts, if
    supplied, is a list of physical (µm) frame-a offsets and changes both the
    matcher candidate ball and its relative-position features.
    """
    radius = float(cfg.loss.embed_radius_um)
    scale = torch.tensor(SCALE, dtype=torch.float32, device=device)
    scores = []
    for t in range(len(per_t) - 1):
        pa, ea = per_t[t]
        pb, eb = per_t[t + 1]
        if len(pa) == 0 or len(pb) == 0:
            scores.append(dict(
                logits=np.full((len(pa), len(pb)), NEG_INF, np.float32),
                null_a=np.zeros(len(pa), np.float32),
                null_b=np.zeros(len(pb), np.float32)))
            continue
        fa = torch.from_numpy(ea).to(device)
        fb = torch.from_numpy(eb).to(device)
        ga = torch.from_numpy(pa.astype(np.float32)).to(device) * scale
        gb = torch.from_numpy(pb.astype(np.float32)).to(device) * scale
        offset = None
        if candidate_drifts is not None:
            offset = torch.as_tensor(candidate_drifts[t], device=device,
                                     dtype=torch.float32).expand(len(pa), 3)
        # the matcher stays fp32 even under test.amp: autocast casts per op,
        # so MHA would run its QK^T in fp16 and overflow to NaN whatever dtype
        # the inputs are (core/matcher.py, dl_067 ep39)
        with torch.autocast(fa.device.type, enabled=False):
            logits, null_a, null_b = model.matcher(
                fa.float(), ga, fb.float(), gb, radius,
                candidate_offset_a=offset)
        scores.append(dict(
            logits=logits.cpu().numpy().astype(np.float32),
            null_a=null_a.cpu().numpy().astype(np.float32),
            null_b=null_b.cpu().numpy().astype(np.float32)))
    return scores


def matcher_probs(scores, no_null=False):
    """Backward-compatible source-row probabilities for legacy wiring."""
    sims = []
    for s in scores:
        logits = torch.from_numpy(s["logits"])
        if not len(logits):
            sims.append(np.zeros_like(s["logits"], dtype=np.float32))
        elif no_null:
            sims.append(torch.softmax(logits, 1).numpy().astype(np.float32))
        else:
            null = torch.from_numpy(s["null_a"])[:, None]
            sims.append(torch.softmax(torch.cat([logits, null], 1), 1)[:, :-1]
                        .numpy().astype(np.float32))
    return sims


def matcher_bidir_probs(scores):
    """Direction-averaged link probability: mean of the a->b row softmax
    (competing against a's candidates + null_a) and the b->a column softmax
    (competing against b's candidates + null_b).  Unlike matcher_sym_probs
    this KEEPS the competition normalization in both directions and just
    fuses the two views — a pair is confident only if it wins its race from
    BOTH endpoints."""
    sims = []
    for s in scores:
        logits = torch.from_numpy(s["logits"])
        if not logits.numel():
            sims.append(np.zeros_like(s["logits"], dtype=np.float32))
            continue
        na = torch.from_numpy(s["null_a"])[:, None]
        nb = torch.from_numpy(s["null_b"])[None, :]
        p_ab = torch.softmax(torch.cat([logits, na], 1), 1)[:, :-1]
        p_ba = torch.softmax(torch.cat([logits, nb], 0), 0)[:-1, :]
        sims.append(((p_ab + p_ba) / 2).numpy().astype(np.float32))
    return sims


def matcher_rev_probs(scores):
    """Column softmax only: for each frame-b node, its distribution over
    candidate PARENTS (competing with null_b).

    The a->b row softmax is the wrong direction for divisions — a dividing
    parent's two daughters split its row mass, so the second child is
    structurally suppressed (measured: p50 0.18 on the parent->daughter pairs
    the one-to-one wiring had to drop). Each daughter, however, has exactly one
    parent, so the reverse direction does not split.
    """
    sims = []
    for s in scores:
        logits = torch.from_numpy(s["logits"])
        if not logits.numel():
            sims.append(np.zeros_like(s["logits"], dtype=np.float32))
            continue
        nb = torch.from_numpy(s["null_b"])[None, :]
        sims.append(torch.softmax(torch.cat([logits, nb], 0), 0)[:-1, :]
                    .numpy().astype(np.float32))
    return sims


def matcher_sym_probs(scores):
    """Symmetric two-direction link probability (issue: gate uses the a->b
    row-softmax only, although training supervises BOTH directions).  The
    pair scalar combined with both null heads gives the direction-symmetric
    log-odds `s = 2*pair_logit - null_a - null_b` (same quantity
    partial_pairs assigns on); sigmoid(s/2) maps it to a [0,1] score whose
    0.5 point is "pair logit beats the average null" — row-softmax
    normalizers (candidate-count dependence) drop out entirely."""
    sims = []
    for s in scores:
        logits = s["logits"].astype(np.float64)
        if not logits.size:
            sims.append(np.zeros_like(s["logits"], dtype=np.float32))
            continue
        sym = 2.0 * logits - s["null_a"][:, None] - s["null_b"][None, :]
        sym = np.clip(sym / 2.0, -60.0, 60.0)
        sims.append((1.0 / (1.0 + np.exp(-sym))).astype(np.float32))
    return sims


def partial_pairs(ap, bp, score, max_link_um, motion_alpha, link_bias, drift,
                  sim=None, ext_um=0.0, p_th=1.0):
    """One-to-one partial assignment with individual birth/death dummies.

    The pair scalar is supervised once in each direction.  Combining those
    two directional log-odds gives
    `2*pair_logit - source_null - target_null`; hence the -2 multiplier below.
    This is invariant to the row-softmax normalizers and uses both null heads.
    """
    na, nb = len(ap), len(bp)
    if not na or not nb:
        return []
    d = np.linalg.norm((ap + drift)[:, None, :] - bp[None, :, :], axis=2)
    logits = score["logits"].astype(np.float64)
    # NEG_INF/2 recognizes the matcher's out-of-radius sentinel (real logits
    # are tanh-bounded to |30|). NOTE: with raw-frame scores the sentinel was
    # set on RAW distances, so pairs with drift-residual <= max_link_um but
    # raw distance > embed_radius_um stay unreachable (see main()'s warning).
    valid = (d <= max_link_um) & (logits > NEG_INF / 2)
    if ext_um > max_link_um and sim is not None:
        # same confidence-gated long links the legacy wiring gets (+0.0080 on
        # the 5-fold OOF, dl_134): without this, partial is capped at the 7 um
        # gate and cannot reach the band where the matcher's exclusive wins are
        valid |= (d <= ext_um) & (sim >= p_th) & (logits > NEG_INF / 2)
    huge = 1e6
    cost = np.full((na + nb, nb + na), huge, np.float64)
    cost[:na, :nb] = np.where(
        valid, -2.0 * logits - link_bias + motion_alpha * d, huge)
    # One private death slot per source and one private birth slot per target.
    cost[np.arange(na), nb + np.arange(na)] = -score["null_a"]
    cost[na + np.arange(nb), np.arange(nb)] = -score["null_b"]
    # Dummy birth rows may absorb unused death columns at zero cost.
    cost[na:, nb:] = 0.0
    ri, ci = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(ri, ci)
            if i < na and j < nb and valid[i, j]]


def build_nodes(pos):
    """(t, z, y, x) int32 node table + per-frame offsets from float zyx
    frames (submission coords are rounded; distances elsewhere stay float)."""
    nodes, offsets = [], [0]
    for t, coords in enumerate(pos):
        for zyx in np.rint(coords).astype(np.int32):
            nodes.append((t, int(zyx[0]), int(zyx[1]), int(zyx[2])))
        offsets.append(len(nodes))
    nodes = np.array(nodes, dtype=np.int32) if nodes else np.zeros((0, 4), np.int32)
    return nodes, offsets


def link_partial(per_t, scores, max_link_um, motion_alpha, link_bias, drifts,
                 sims=None, ext_um=0.0, p_th=1.0):
    pos = [np.asarray(p, np.float64).reshape(-1, 3) for p, _ in per_t]
    nodes, offsets = build_nodes(pos)
    edges = []
    for t, (ap, bp) in enumerate(zip(pos[:-1], pos[1:])):
        pairs = partial_pairs(ap * SCALE, bp * SCALE, scores[t], max_link_um,
                              motion_alpha, link_bias, drifts[t],
                              sim=None if sims is None else sims[t],
                              ext_um=ext_um, p_th=p_th)
        edges.extend((offsets[t] + i, offsets[t + 1] + j) for i, j in pairs)
    return nodes, edges


def load_cached(model_fn, store, sid, cfg, device, cache_dir, det_model_fn=None,
                det_cache_dir=None, strict=False):
    f = cache_dir / f"{sid}.npz"
    if f.exists():
        z = np.load(f)
        T = len([k for k in z.files if k.startswith("p")])
        return [(z[f"p{t:04d}"], z[f"e{t:04d}"]) for t in range(T)]
    if strict:
        # --embed_cache_dir promises "read directly": never fall back to
        # inference (which could silently mix detection sources or crash in
        # savez after minutes of compute on eval-only machines)
        raise FileNotFoundError(f"embed cache missing (strict mode): {f}")
    det_peaks = None
    if det_cache_dir is not None:
        dc = det_cache_dir / f"{sid}.npz"
        if dc.exists():
            z = np.load(dc)
            # detection caches carry per-peak
            # confidences under c#### keys next to the t#### coordinates
            det_peaks = [z[k] for k in sorted(k for k in z.files
                                              if k.startswith("t"))]
        elif det_model_fn is None:
            raise FileNotFoundError(f"det cache missing (strict mode): {dc}")
    per_t = detect_sample_embed(
        model_fn(), store, sid, cfg, device,
        det_model_fn() if (det_model_fn and det_peaks is None) else None,
        det_peaks)
    np.savez_compressed(f, **{f"p{t:04d}": p for t, (p, _) in enumerate(per_t)},
                        **{f"e{t:04d}": e for t, (_, e) in enumerate(per_t)})
    return per_t


def link_embed(per_t, max_link_um, lam, drift_iters, sims=None,
               ext_um=0.0, p_th=1.0, flow=None, flow_clip_um=None):
    """Production linking with cost = drift-compensated distance +
    lam * (1 - sim); the GATE stays on the distance term (floor-preserving:
    lam = 0 is exactly build_graph). sim = normalized-embedding cosine, or
    matcher link probability when `sims` (from build_sims) is given.
    ext_um/p_th (confidence-gated long links): pairs with
    max_link_um < d <= ext_um are additionally admitted when sim >= p_th —
    the matcher is trained/scored out to 16 um but the 7 um acceptance gate
    discarded that band (55% of its exclusive wins sit just outside it).
    ext_um <= max_link_um (or a scalar sim) keeps the original gate exactly.

    flow: optional list over boundaries of (M,3) per-node displacements in MICRONS
    for the frame-t detections.  Applied to the
    frame-a positions BEFORE the drift loop, so the existing collective
    estimator keeps running on top and absorbs whatever global bias the field
    has left; flow=None reproduces the previous graph exactly.  Unlike
    build_graph's flow argument this is NOT deviation-only: flow_145 measured
    the deviation-only form (1.6627 um) as clearly worse than the raw field
    (1.4555 um), because the weak term is the collective estimate, not the local
    one.  flow_clip_um zeroes vectors longer than the clip — a guard for the
    field's far tail (flow_145: recall@7 0.9952 vs 0.9984 for a pure translation).
    """
    pos = [np.asarray(p, np.float64).reshape(-1, 3) for p, _ in per_t]
    embs = [e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
            for _, e in per_t]
    nodes, offsets = build_nodes(pos)

    def assign(ap, bp, sim):
        d = np.linalg.norm(ap[:, None, :] - bp[None, :, :], axis=2)
        ok = d <= max_link_um
        if ext_um > max_link_um and isinstance(sim, np.ndarray):
            ok |= (d <= ext_um) & (sim >= p_th)
        cost = np.where(ok, d + lam * (1.0 - sim), 1e6)
        ri, ci = linear_sum_assignment(cost)
        return [(i, j) for i, j in zip(ri, ci) if ok[i, j]]

    edges = []
    for t in range(len(pos) - 1):
        a0, b0 = offsets[t], offsets[t + 1]
        if not len(pos[t]) or not len(pos[t + 1]):
            continue
        ap = pos[t] * SCALE
        bp = pos[t + 1] * SCALE
        if flow is not None:
            fa = np.asarray(flow[t], np.float64)
            # a silent length mismatch would misalign every offset and still
            # produce a plausible score — fail loudly instead
            assert fa.shape == ap.shape, \
                f"flow[{t}] {fa.shape} does not match detections {ap.shape}"
            if flow_clip_um:
                fa = np.where(
                    (np.linalg.norm(fa, axis=1) > float(flow_clip_um))[:, None],
                    0.0, fa)
            ap = ap + fa
        if sims is not None:
            sim = sims[t]              # needed for the gate even at lam = 0
        elif lam <= 0:
            sim = 0.0
        else:
            sim = embs[t] @ embs[t + 1].T
        pairs = assign(ap, bp, sim)
        for _ in range(int(drift_iters)):
            if len(pairs) < 5:
                break
            drift = np.median(np.stack([bp[j] - ap[i] for i, j in pairs]), axis=0)
            pairs = assign(ap + drift, bp, sim)
        edges += [(a0 + i, b0 + j) for i, j in pairs]
    return nodes, edges


def link_accuracy(per_t, gt, radius_um, sims=None, cand_drifts=None):
    """Proxy A on one sample: per normal both-detected GT edge, does the
    embedding argmax (within radius) pick the true successor? vs the
    drift-compensated NN. Matching/drift construction mirrors
    analyze_mislink.py (GT-derived drift). sims: matcher probabilities
    replace embedding cosine when given. cand_drifts: the production
    (GT-free) per-boundary drifts the matcher gated with — pass them iff the
    sims were built in the drift frame, so the argmax candidate set matches
    the matcher's own candidate ball."""
    ids_gt, gt_t, gt_zyx, gt_edges, _ = gt
    peak_um = [p.astype(np.float64) * SCALE for p, _ in per_t]
    unit = [e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)
            for _, e in per_t]
    det_of = {}                                     # gt row -> (t, det idx)
    by_t = defaultdict(list)
    for k in range(len(ids_gt)):
        by_t[int(gt_t[k])].append(k)
    for tt, gks in by_t.items():
        if tt >= len(per_t) or not len(per_t[tt][0]):
            continue
        for i, j in hungarian_pairs(per_t[tt][0].astype(np.float64),
                                    gt_zyx[gks].astype(np.float64)):
            det_of[int(ids_gt[gks[j]])] = (tt, int(i))

    gt_out = defaultdict(int)
    for s, tg in gt_edges:
        gt_out[int(s)] += 1

    disp = defaultdict(list)
    for s, tg in gt_edges:
        a, b = det_of.get(int(s)), det_of.get(int(tg))
        if a and b:
            disp[a[0]].append(peak_um[b[0]][b[1]] - peak_um[a[0]][a[1]])
    drift = {t: (np.median(np.stack(v), 0) if len(v) >= 3 else np.zeros(3))
             for t, v in disp.items()}

    c = defaultdict(int)
    for s, tg in gt_edges:
        s, tg = int(s), int(tg)
        if gt_out[s] >= 2:
            continue
        a, b = det_of.get(s), det_of.get(tg)
        if not a or not b or b[0] != a[0] + 1:
            continue
        t, da = a
        db = b[1]
        # NN uses (GT-derived) drift-compensated distances; the candidate set
        # for the similarity argmax must mirror the frame the matcher masked
        # in — raw distances for a raw-frame matcher, production-drift-shifted
        # distances for a drift-frame matcher (mixing the framings
        # mismeasured dl_063)
        cand_d = np.linalg.norm(
            peak_um[t + 1] - (peak_um[t][da] + drift.get(t, np.zeros(3))), axis=1)
        if sims is not None:
            shift = cand_drifts[t] if cand_drifts is not None else np.zeros(3)
            shift = np.asarray(shift)
            if shift.ndim == 2:       # per-node offsets (dense flow frame)
                shift = shift[da]
            gate_d = np.linalg.norm(
                peak_um[t + 1] - (peak_um[t][da] + shift), axis=1)
            near = np.nonzero(gate_d <= radius_um)[0]
        else:
            near = np.nonzero(cand_d <= radius_um)[0]
        c["n"] += 1
        nn = int(np.argmin(cand_d))
        c["nn_ok"] += nn == db
        if len(near):
            if sims is not None:
                sim = sims[t][da, near]
            else:
                sim = unit[t][da] @ unit[t + 1][near].T
            emb_pick = int(near[int(np.argmax(sim))])
            c["emb_ok"] += emb_pick == db
            if nn != db:
                c["nn_bad"] += 1
                c["nn_bad_emb_ok"] += emb_pick == db
        else:
            c["no_cand"] += 1
    return c
