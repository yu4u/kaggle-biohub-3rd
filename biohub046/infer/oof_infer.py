"""Stage ⑥: per-movie inference of the 046 v4 stack with FOLD-MATCHED models (out-of-fold).

This flattens the per-clip chain of submission 046 (metric_inference.infer_clip wrapped by 030 /
031 / 037 / 041 / 042) into one function. For every train movie the fold whose validation set
contains it supplies one checkpoint per model (the deployed submission averaged the five folds
because the test embryos are disjoint). Per movie it writes

    cache/oof/graphs/<sid>.npz          nodes, LP edges, radius-12 top-5 candidates, second-round top-5
    cache/oof/confidence/<sid>.npz      blurred detector heatmap value at every node
    cache/oof/div_scores/<sid>.npz      score (div_415 parent), pre, post (div_478), native_parent (div_762)
    cache/oof/division_graphs/<sid>.npz the radius-16 extra-candidate graph (div797_features.radius16_graph)

Stages: ① detection (3 members, flip2 TTA, fp16, blur 0.7, th 0.2) -> ② matcher trunk embeddings at the
detections, flow_164 dense field + per-node offsets -> production Hungarian (link_embed) -> second
round -> global LP wiring -> ④ div3f parent / pre / post (rot4, encoder reuse) + native762 (warp5, rot4).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from biohub046 import config as C

OOF_DIR = C.CACHE_DIR / 'oof'
SUBDIRS = ('graphs', 'confidence', 'div_scores', 'division_graphs')


def done(sid, out=OOF_DIR):
    return all((out / d / f'{sid}.npz').exists() for d in SUBDIRS)


# ---- ② matcher trunk: one decoded batch per frame (030.ensemble_detect_sample_embed) ----------------
def detect_sample_embed(models, store, sid, cfg, device, det_peaks):
    """Per frame: (peaks, concat of each model's sampled embeddings along axis 1)."""
    import torch
    from biohub046.core.dataset import get_downsample, to_ds
    from biohub046.core.embed_loss import sample_embeddings
    ds = get_downsample(cfg)
    ds_on = bool((ds != 1).any())
    T = store.shape[sid][0]
    bs = int(cfg.test.frames_per_batch)
    amp = bool(cfg.test.get('amp', False))
    dev = device.type if hasattr(device, 'type') else str(device).split(':')[0]
    for model in models:
        if hasattr(model, 'embed_only_forward'):
            model.embed_only_forward = True
    out = []
    with torch.no_grad():
        for t0 in range(0, T, bs):
            frames = [store.frame(sid, t) for t in range(t0, min(T, t0 + bs))]
            xt = torch.from_numpy(np.stack(frames))[:, None].to(device)
            embs = []
            for model in models:
                with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                    _, emb = model(xt)
                embs.append(emb.float())
            for b in range(xt.shape[0]):
                pk_np = det_peaks[t0 + b].astype(np.int32)
                pos_np = to_ds(pk_np, ds) if ds_on else pk_np
                pos = torch.from_numpy(pos_np.astype(np.float32)).to(device)
                ev = [sample_embeddings(emb[b], pos, tuple(xt.shape[2:])).cpu().numpy().astype(np.float32)
                      for emb in embs]
                out.append((pk_np, np.concatenate(ev, axis=1)))
    return out


# ---- ④ div3f parent / pre / post with rot4 TTA and encoder reuse (037.div3f_scores_roles_reuse) -----
def div3f_scores_roles_reuse(roles, store, sid, nodes, device, batch=8):
    import torch
    from biohub046.core.dataset import to_ds
    from biohub046.div3f.data import apply_d4_points
    from biohub046.div3f.model import sample_logits
    from biohub046.infer.legacy import TTA_VARIANTS
    variants = TTA_VARIANTS[C.DIV_TTA]
    by_t = {}
    for i, node in enumerate(nodes):
        by_t.setdefault(int(node[0]), []).append(i)
    times = sorted(by_t)
    T = store.shape[sid][0]
    frames = torch.stack([torch.from_numpy(store.frame(sid, t)) for t in range(T)]).to(device)
    shape = tuple(frames.shape[1:])                     # (Z, Y, X) on the div3f grid; Y == X
    windows = {t: (max(t - 1, 0), t, min(t + 1, T - 1)) for t in times}
    pts = {t: to_ds(nodes[by_t[t], 1:].astype(np.float64), store.ds) for t in times}
    scores = {role: np.zeros(len(nodes), np.float32) for role in roles}
    dev = str(device).split(':')[0]
    denominators = {role: len(models) * len(variants) for role, models in roles.items()}
    with torch.no_grad():
        for k, flip in variants:
            view = torch.rot90(frames, k, dims=(-2, -1))
            if flip:
                view = torch.flip(view, dims=(-1,))
            moved = {t: torch.from_numpy(apply_d4_points(p, k, flip, shape).astype(np.float32)).to(device)
                     for t, p in pts.items()}
            for role, models in roles.items():
                for model in models:
                    cache = {}                          # frame index -> per-stage (1, C, ...) features
                    for start in range(0, len(times), batch):
                        chunk = times[start:start + batch]
                        new = sorted({u for t in chunk for u in windows[t]} - cache.keys())
                        with torch.autocast(dev, dtype=torch.float16, enabled=dev.startswith('cuda')):
                            if new:
                                feats = model.encode(view[new].unsqueeze(1))
                                for i, u in enumerate(new):
                                    cache[u] = [f[i:i + 1] for f in feats]
                            stacked = [torch.cat([torch.cat([cache[u][s] for u in windows[t]], 1) for t in chunk])
                                       for s in range(len(cache[chunk[0]]))]
                            logits = model.decode(stacked, shape)
                        for b, t in enumerate(chunk):
                            scores[role][by_t[t]] += (torch.sigmoid(sample_logits(logits[b:b + 1].float(), moved[t]))
                                                      .cpu().numpy() / denominators[role])
                        for u in [u for u in cache if u < chunk[-1]]:   # later windows start at >= chunk[-1]
                            del cache[u]
    return scores


def aligned_confidence(nodes, peaks, confidences):
    """metric_inference.aligned_confidence: fail if the LP reordered the detector nodes."""
    if len(peaks) != len(confidences):
        raise ValueError('confidence frame count differs from detections')
    expected, values = [], []
    for t, (p, c) in enumerate(zip(peaks, confidences)):
        p, c = np.asarray(p).reshape(-1, 3), np.asarray(c)
        if c.shape != (len(p),) or not np.isfinite(c).all():
            raise ValueError('invalid detector confidence')
        expected.append(np.column_stack([np.full(len(p), t), p]))
        values.append(c.astype(np.float16).astype(np.float32))   # precision of the original caches
    expected = np.concatenate(expected) if expected else np.empty((0, 4))
    if not np.array_equal(nodes, expected):
        raise ValueError('matcher nodes do not align with detector confidence')
    return np.concatenate(values) if values else np.empty(0, np.float32)


class CachedZarrArray:
    """`array[t]` from the CellStore raw cache (filled by the detector pass), zarr on a miss (030)."""

    def __init__(self, array, raw_cache, sid):
        self.array, self.raw, self.sid = array, raw_cache, sid
        self.shape = array.shape

    def __getitem__(self, t):
        key = (self.sid, int(t))
        raw = None if self.raw is None else self.raw.get(key)
        if raw is None:
            raw = self.array[t]
            if self.raw is not None:
                self.raw[key] = raw
        return raw


def load_models(fold, device):
    """Fold-matched single checkpoints of every model."""
    import torch
    from biohub046.div3f.model import build_model
    from biohub046.infer.legacy import load_detector, load_matcher
    from biohub046.ssl_flow.inference import load_model as load_flow_model
    paths = C.model_paths()
    det = [(load_detector(paths[f'{name}_fold{fold}_last.ckpt'], device)[1], weight) for name, weight in C.DETECTORS]
    cfg_emb, emb = load_matcher(paths[f'dl_401_fold{fold}_last.ckpt'], device)
    flow, cfg_flow, _ = load_flow_model(str(paths['ssl_flow_flow_164_ep48.pt']), device)
    roles, grids = {}, set()
    for role, name in [('parent', f'div_415_f{fold}_calpick.pt'), ('pre', f'div_478_pre_fold{fold}_calpick.pt'),
                       ('post', f'div_478_post_fold{fold}_calpick.pt'), ('native_parent', f'div_762_f{fold}_calpick.pt')]:
        state = torch.load(paths[name], map_location='cpu', weights_only=False)
        data, mcfg = state['config']['data'], state['config']['model']
        if role in ('pre', 'post') and data['target'] != role:
            raise ValueError(f'{name}: checkpoint target {data["target"]} != {role}')
        if role == 'native_parent':
            assert data['target'] == 'parent' and data['frame_offset'] == 0 and data['warp_keep_raw'] \
                and data['warp_flow_dir'] and mcfg['n_frames'] == 5, name
        if int(data['fold_id']) != fold:
            raise ValueError(f'{name}: checkpoint fold {data["fold_id"]} != {fold}')
        model = build_model(dict(mcfg, pretrained=False)).to(device).eval()
        model.load_state_dict(state['model'])
        roles[role] = [model]
        grids.add(tuple(data['downsample']))
    if grids != {(1, 2, 2)}:
        raise ValueError(f'div3f checkpoints have unexpected grids: {grids}')
    return det, (cfg_emb, emb), (cfg_flow, flow), roles, np.array([1, 2, 2])


def infer_clip(sid, models, stores, device, out, raw_cache):
    import torch
    import zarr
    from omegaconf import OmegaConf
    from biohub046.infer import legacy
    from biohub046.infer.div797_parent import flow_with_dense
    from biohub046.infer.div797_parent_reuse import parent_scores_reuse
    from biohub046.infer.division import second_round
    from biohub046.infer.lp_wiring import lp_wiring, top_candidates
    from biohub046.infer import eval_embed_link as eel
    from biohub046.lineage.div797_features import radius16_graph
    det, (cfg_emb, emb), (cfg_flow, flow), roles, _ = models
    store_det, store_emb, store_div = stores
    cfg_det = OmegaConf.create({'model': {'det_threshold': C.DET_THRESHOLD, 'nms_kernel': C.NMS_KERNEL},
                                'test': {'frames_per_batch': 1}})
    times = {}
    tic = time.monotonic()
    # ① detection
    peaks, conf = legacy.detect_tta(det, store_det, sid, device, cfg_det)
    torch.cuda.synchronize()
    times['det'] = time.monotonic() - tic
    tic = time.monotonic()
    # ② matcher embeddings, flow, scores, production cell, second round, LP wiring
    per_t = detect_sample_embed([emb], store_emb, sid, cfg_emb, device, peaks)
    z = zarr.open(str(C.TRAIN_DIR / f'{sid}.zarr'), mode='r')
    array = CachedZarrArray(z['0'], raw_cache, sid)
    offsets, dense = flow_with_dense(array, dict(z.attrs)['image_statistics']['quantiles'], peaks, flow,
                                     cfg_flow.data.downsample, device)
    scores = eel.build_matcher_scores(per_t, emb, cfg_emb, device, offsets)
    sims, revs, syms = (f(scores) for f in [eel.matcher_bidir_probs, eel.matcher_rev_probs, eel.matcher_sym_probs])
    _, initial = eel.link_embed(per_t, legacy.MAX_LINK_UM, legacy.LAMBDA, legacy.DRIFT_ITERS, sims,
                                ext_um=legacy.GATE_EXT_UM, p_th=legacy.GATE_PTH, flow=offsets)   # 022: FLOW_TO_WIRING
    r2i, r2p = second_round(per_t, initial, emb, cfg_emb, device, offsets, k=legacy.R2_TOPK)
    if r2i.shape != r2p.shape or r2i.ndim != 2 or r2i.shape[1] != legacy.R2_TOPK:
        raise ValueError(f'r2 cache requires width {legacy.R2_TOPK}')
    nodes, edges = lp_wiring(per_t, sims, revs, syms, offsets, r2i, r2p)
    confidence = aligned_confidence(nodes, peaks, conf)
    ci, cp, cr, cs, cd = top_candidates(per_t, sims, revs, syms, offsets)
    torch.cuda.synchronize()
    times['matcher_flow_lp'] = time.monotonic() - tic
    tic = time.monotonic()
    # ④ division scores
    phase = div3f_scores_roles_reuse({k: roles[k] for k in ('parent', 'pre', 'post')}, store_div, sid, nodes, device)
    phase = dict(score=phase['parent'], pre=phase['pre'], post=phase['post'])
    phase['native_parent'] = parent_scores_reuse(roles['native_parent'], store_div, sid, nodes, device, dense)
    del dense
    for v in phase.values():
        if v.shape != (len(nodes),) or not np.isfinite(v).all() or np.any((v < 0) | (v > 1)):
            raise ValueError(f'{sid}: invalid phase scores')
    torch.cuda.synchronize()
    times['div3f'] = time.monotonic() - tic
    tic = time.monotonic()
    g = dict(nodes=nodes, edges=np.asarray(edges, np.int64).reshape(-1, 2), cand_idx=ci, cand_prob=cp, cand_rev=cr,
             cand_sym=cs, cand_dist=cd, r2_idx=r2i, r2_prob=r2p)
    wide = radius16_graph(g, phase, per_t, offsets, emb.matcher, device)
    times['radius16'] = time.monotonic() - tic
    for name in SUBDIRS:
        (out / name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'graphs' / f'{sid}.npz', **g)
    np.savez_compressed(out / 'confidence' / f'{sid}.npz', nodes=nodes, confidence=confidence)
    np.savez_compressed(out / 'div_scores' / f'{sid}.npz', nodes=nodes, **phase)
    np.savez_compressed(out / 'division_graphs' / f'{sid}.npz', **wide)
    return len(nodes), times


def run(ids, out=OOF_DIR, shard=None):
    import torch
    from biohub046.core.dataset import CellStore, get_downsample
    from biohub046.infer.legacy import check_device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    check_device(device)
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folds = C.fold_assignments()
    todo = [s for s in ids if not done(s, out)]
    tag = f'[oof_infer{"" if shard is None else f" shard {shard}"}]'
    print(f'{tag} {len(todo)} of {len(ids)} movies to infer', flush=True)
    started = time.monotonic()
    k = 0
    for fold in range(C.FOLD_NUM):
        members = [s for s in todo if folds[s] == fold]
        if not members:
            continue
        models = load_models(fold, device)
        raw_cache = {}
        stores = [CellStore(C.TRAIN_DIR, members, with_gt=False, downsample=ds, raw_cache=raw_cache)
                  for ds in [np.ones(3, int), get_downsample(models[1][0]), models[4]]]
        for sid in members:
            k += 1
            count, times = infer_clip(sid, models, stores, device, out, raw_cache)
            raw_cache.clear()
            print(f'{tag} {k}/{len(todo)} {sid} (fold {fold}): {count} nodes, '
                  + ', '.join(f'{n} {v:.1f}s' for n, v in times.items())
                  + f', elapsed {time.monotonic() - started:.0f}s', flush=True)
        del models
        torch.cuda.empty_cache()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit', type=int, default=None, help='first N movies (sorted ids)')
    p.add_argument('--shards', type=int, default=None, help='GPU workers; default = visible GPUs')
    p.add_argument('--shard', type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    ids = C.discover_ids()[:args.limit]
    import torch
    shards = min(args.shards or max(1, torch.cuda.device_count()), len(ids))
    if args.shard is not None:
        run(ids[args.shard::shards], shard=args.shard)
        return
    if shards <= 1:
        run(ids)
    else:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        devices = visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
        procs = []
        for i in range(shards):
            cmd = [sys.executable, '-m', 'biohub046.infer.oof_infer', '--shards', str(shards), '--shard', str(i)]
            if args.limit:
                cmd += ['--limit', str(args.limit)]
            procs.append(subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices[i])))
        codes = [proc.wait() for proc in procs]
        if any(codes):
            raise RuntimeError(f'inference worker exit codes: {codes}')
    missing = [s for s in ids if not done(s)]
    if missing:
        raise RuntimeError(f'incomplete OOF caches: {missing[:5]} ...')


if __name__ == '__main__':
    main()
