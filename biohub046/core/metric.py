"""Peak extraction, node matching, linking, and the official edge metric.

build_graph / eval_sample / summarise reproduce the official edge metric.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

SCALE = np.array([1.625, 0.40625, 0.40625])  # (z, y, x) µm/voxel
MAX_MATCH_UM = 7.0                            # official node-matching threshold


def estimate_collective_drift(pos_a_um, pos_b_um, max_link_um=7.0,
                              drift_iters=2, min_pairs=5, clip_um=15.0):
    """GT-free robust translation from two detection sets in physical units.

    This is deliberately shared by matcher training and inference so both see
    the identical frame.  Production-style loop (gated Hungarian,
    coordinate-wise median displacement, rematch in the translated frame), but
    NOT numerically identical to the linker in build_graph/link_embed: it runs
    drift_iters + 1 estimation rounds (>= 1 even at drift_iters=0, one more
    refinement than the linker's in-loop estimate), clips the vector norm to
    clip_um, and falls back to ZERO — not the previous estimate — whenever
    gated matches drop below min_pairs.  The returned vector maps frame-a
    coordinates into frame b; it is never differentiated through.
    """
    ap = np.asarray(pos_a_um, np.float64).reshape(-1, 3)
    bp = np.asarray(pos_b_um, np.float64).reshape(-1, 3)
    drift = np.zeros(3, np.float64)
    if len(ap) < min_pairs or len(bp) < min_pairs:
        return drift
    for _ in range(max(1, int(drift_iters) + 1)):
        d = np.linalg.norm((ap + drift)[:, None, :] - bp[None, :, :], axis=2)
        ri, ci = linear_sum_assignment(np.where(d > max_link_um, 1e6, d))
        pairs = [(i, j) for i, j in zip(ri, ci) if d[i, j] <= max_link_um]
        if len(pairs) < min_pairs:
            return np.zeros(3, np.float64)
        drift = np.median(np.stack([bp[j] - ap[i] for i, j in pairs]), axis=0)
        norm = np.linalg.norm(drift)
        if norm > clip_um:
            drift *= clip_um / norm
    return drift


def extract_peaks(hm, threshold, kernel):
    """Local maxima of a (D,H,W) heatmap tensor -> (N,3) int64 zyx tensor."""
    k = tuple(int(v) for v in kernel)
    pad = tuple(v // 2 for v in k)
    mp = F.max_pool3d(hm[None, None], kernel_size=k, stride=1, padding=pad)[0, 0]
    return ((hm == mp) & (hm > threshold)).nonzero()


COARSE_STRIDE = (1, 4, 4)   # arch=unet25d_coarse output grid (z, y, x)
COARSE_C = (0.0, 0.5, 0.5)  # cell-centre offset per axis (stride-1 z: the
                            # centre IS the index; strided yx: +0.5) — the
                            # encode side (dataset.build_coarse_targets)
                            # imports both constants from here


def decode_coarse(out, threshold, kernel, topk=4096, return_conf=False,
                  logit_space=True):
    """Peaks from a coarse-head output (4, Z, Yc, Xc) = [cls_logit, dz, dy, dx]
    -> (M, 3) float zyx in ds-grid voxels.

    NMS + threshold run on the RAW logit channel: sigmoid saturates to exactly
    1.0 for confident cells and the `hm == maxpool` equality NMS would then
    keep whole plateaus of tied duplicates. `threshold` is a probability (same
    knob as model.det_threshold) converted to logit space here.

    Cell-center convention: c = 0.5 for the strided yx axes, 0 for the
    stride-1 z axis (a stride-1 cell's center IS the voxel index; a uniform
    +0.5 would push every z target to the -0.5 clamp). Matches the dataset's
    coarse_off targets. topk caps runaway peak counts (untrained/collapsed
    ckpts) before the O(n^3) linking downstream."""
    logit = out[0]
    if logit_space:                       # BCE head: NMS/threshold on raw logits
        p = min(max(float(threshold), 1e-6), 1 - 1e-6)
        th = float(np.log(p / (1 - p)))
    else:                                 # gauss/MSE head: raw regression values
        th = float(threshold)
    idx = extract_peaks(logit, th, kernel)
    if topk and len(idx) > topk:
        conf = logit[idx[:, 0], idx[:, 1], idx[:, 2]]
        idx = idx[conf.topk(topk).indices]
    off = out[1:4, idx[:, 0], idx[:, 1], idx[:, 2]].T.clamp(-1.0, 1.0)
    pos = (idx.float() + out.new_tensor(COARSE_C) + off) \
        * out.new_tensor(COARSE_STRIDE)
    # border cells with saturated offsets decode outside the volume (down to
    # y=-2 / z=-1, past the far edge by +2); clamp like refine_peaks does so
    # graph/submission coordinates stay in-volume
    lim = out.new_tensor([d * v for d, v in zip(out.shape[1:], COARSE_STRIDE)])
    pos = pos.clamp(min=0.0).minimum(lim - 1)
    if return_conf:
        conf = logit[idx[:, 0], idx[:, 1], idx[:, 2]]
        return pos, (conf.sigmoid() if logit_space else conf)
    return pos


def refine_rad(ds, base=(2, 6, 6)):
    """refine_peaks window for a (z,y,x) downsample factor: shrink the base
    window (tuned on the original grid) to keep its µm extent. Identity at
    ds=[1,1,1]."""
    return tuple(max(1, round(r / int(d))) for r, d in zip(base, ds))


def refine_peaks(hm, peaks, rad=(2, 6, 6)):
    """Subvoxel refinement: heatmap-weighted centroid in a small window around
    each integer peak -> float (N,3). Used for LINKING distances only — the
    submission/metric coordinates stay integer voxels (format constraint), so
    this buys inter-frame jitter reduction, not matching accuracy.
    """
    if len(peaks) == 0:
        return peaks.float()
    rz, ry, rx = rad
    pad = F.pad(hm[None, None], (rx, rx, ry, ry, rz, rz))[0, 0]
    oz = torch.arange(-rz, rz + 1, device=hm.device, dtype=torch.float32)
    oy = torch.arange(-ry, ry + 1, device=hm.device, dtype=torch.float32)
    ox = torch.arange(-rx, rx + 1, device=hm.device, dtype=torch.float32)
    zi = peaks[:, 0, None] + oz.long()[None] + rz
    yi = peaks[:, 1, None] + oy.long()[None] + ry
    xi = peaks[:, 2, None] + ox.long()[None] + rx
    w = pad[zi[:, :, None, None], yi[:, None, :, None], xi[:, None, None, :]]
    w = w.clamp_min(0)                                    # regression output can dip <0
    sw = w.sum((1, 2, 3)).clamp_min(1e-6)
    dz = (w.sum((2, 3)) * oz).sum(1) / sw
    dy = (w.sum((1, 3)) * oy).sum(1) / sw
    dx = (w.sum((1, 2)) * ox).sum(1) / sw
    out = peaks.float() + torch.stack([dz, dy, dx], dim=1)
    hi = torch.tensor(hm.shape, device=hm.device, dtype=torch.float32) - 1
    return out.clamp(torch.zeros(3, device=hm.device), hi)


def hungarian_pairs(pred_zyx, gt_zyx):
    """Optimal <=7 µm pairs between voxel coords -> list of (pred_i, gt_j)."""
    if len(pred_zyx) == 0 or len(gt_zyx) == 0:
        return []
    d = np.linalg.norm(pred_zyx[:, None, :] * SCALE - gt_zyx[None, :, :] * SCALE, axis=2)
    cost = np.where(d > MAX_MATCH_UM, 1e6, d)
    ri, ci = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(ri, ci) if d[i, j] <= MAX_MATCH_UM]


def build_graph(per_t, max_link_um, drift_comp=False, drift_iters=1,
                flow=None, flow_blend=1.0, flow_clip_um=None):
    """One-to-one Hungarian linking between consecutive frames (no divisions).

    per_t: list over t of (M,3) int zyx arrays.
    drift_comp: iterative linking — after the plain first pass, estimate the
    global frame-to-frame translation (per-axis median of matched
    displacements; motion is largely collective flow) and redo the
    assignment on drift-compensated distances (gate applied after
    compensation, so links consistent with the flow may exceed the raw gate).
    drift_iters repeats estimate-and-rematch; later iterations see pairs that
    are less censored by the raw gate, so the estimate can self-correct.

    flow: optional list over t of (M,3) per-node predicted displacements in
    ORIGINAL-grid voxels (pair model). Only the per-node DEVIATION from the
    frame's median flow is applied (the collective part stays with the drift
    estimator, whose µm-accuracy the flow head merely matches — dl_042 diag):
        comp_i = flow_blend * clip(flow_i - median(flow), flow_clip_um)
    and linking runs on (pos_i + comp_i) with the usual drift iterations on
    top. flow_clip_um guards the heavy prediction tails (p90 outliers).

    Returns nodes (N,4) int [t,z,y,x] and edges list of (i,j) global indices.
    """
    # per_t may be float (subvoxel-refined): distances use the float positions,
    # node coordinates are rounded to integer voxels (submission format).
    pos = [np.asarray(c, np.float64).reshape(-1, 3) for c in per_t]
    nodes = []
    offsets = [0]
    for t, coords in enumerate(pos):
        for zyx in np.rint(coords).astype(np.int32):
            nodes.append((t, int(zyx[0]), int(zyx[1]), int(zyx[2])))
        offsets.append(len(nodes))
    nodes = np.array(nodes, dtype=np.int32) if nodes else np.zeros((0, 4), np.int32)

    edges = []
    for t in range(len(pos) - 1):
        a0, a1 = offsets[t], offsets[t + 1]
        b0, b1 = offsets[t + 1], offsets[t + 2]
        if a1 == a0 or b1 == b0:
            continue
        ap = pos[t] * SCALE
        bp = pos[t + 1] * SCALE
        if flow is not None and flow[t] is not None and len(flow[t]) == len(ap) \
                and len(ap) >= 5:
            fa = np.asarray(flow[t], np.float64) * SCALE          # µm
            dev = fa - np.median(fa, axis=0)
            if flow_clip_um:
                dev[np.linalg.norm(dev, axis=1) > float(flow_clip_um)] = 0.0
            ap = ap + float(flow_blend) * dev
        d = np.linalg.norm(ap[:, None, :] - bp[None, :, :], axis=2)
        cost = np.where(d > max_link_um, 1e6, d)
        ri, ci = linear_sum_assignment(cost)
        pairs = [(i, j) for i, j in zip(ri, ci) if d[i, j] <= max_link_um]
        if drift_comp:
            for _ in range(int(drift_iters)):
                if len(pairs) < 5:
                    break
                drift = np.median(np.stack([bp[j] - ap[i] for i, j in pairs]), axis=0)
                d2 = np.linalg.norm((ap + drift)[:, None, :] - bp[None, :, :], axis=2)
                cost2 = np.where(d2 > max_link_um, 1e6, d2)
                ri, ci = linear_sum_assignment(cost2)
                pairs = [(i, j) for i, j in zip(ri, ci) if d2[i, j] <= max_link_um]
        edges += [(a0 + i, b0 + j) for i, j in pairs]
    return nodes, edges


def filter_short_tracks(nodes, edges, min_len):
    """Drop connected components (tracks) with fewer than min_len nodes.

    Singletons and very short tracks are predominantly spurious detections
    (low heatmap value, dim, uncorroborated by low-threshold DoG;
    dl_021+diag): removing them cuts the n_pred penalty AND recovers TP lost
    to match-stealing. min_len<=1 is a no-op.
    """
    if min_len <= 1 or not len(nodes):
        return nodes, edges
    parent = list(range(len(nodes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        parent[find(int(i))] = find(int(j))
    roots = np.array([find(i) for i in range(len(nodes))])
    _, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    keep = counts[inv] >= min_len
    remap = -np.ones(len(nodes), np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    nodes2 = nodes[keep]
    edges2 = [(int(remap[i]), int(remap[j])) for i, j in edges if keep[i] and keep[j]]
    return nodes2, edges2


def eval_sample(nodes, edges, gt):
    """Reproduce the official EDGE metric for one sample (no divisions predicted)."""
    ids, gt_t, gt_zyx, gt_edges, n_total = gt

    matched = -np.ones(len(nodes), dtype=np.int64)  # matched GT id per pred node
    gt_by_t, pred_by_t = {}, {}
    for k in range(len(ids)):
        gt_by_t.setdefault(int(gt_t[k]), []).append(k)
    for pi in range(len(nodes)):
        pred_by_t.setdefault(int(nodes[pi, 0]), []).append(pi)

    for tt, gks in gt_by_t.items():
        pis = pred_by_t.get(tt)
        if not pis:
            continue
        for i, j in hungarian_pairs(nodes[pis, 1:].astype(np.float64),
                                    gt_zyx[gks].astype(np.float64)):
            matched[pis[i]] = int(ids[gks[j]])

    gt_out, gt_in, gt_edge_set = {}, {}, set()
    for s, tg in gt_edges:
        gt_out[int(s)] = gt_out.get(int(s), 0) + 1
        gt_in[int(tg)] = gt_in.get(int(tg), 0) + 1
        gt_edge_set.add((int(s), int(tg)))

    tp = fp = 0
    seen = set()
    for i, j in edges:
        if (i, j) in seen:
            continue
        seen.add((i, j))
        ms, mt = matched[i], matched[j]
        out_valid = ms != -1 and gt_out.get(ms, 0) > 0
        in_valid = mt != -1 and gt_in.get(mt, 0) > 0
        if ms != -1 and mt != -1 and (ms, mt) in gt_edge_set:
            tp += 1
        elif out_valid or in_valid:
            fp += 1
    fn = len(gt_edges) - tp

    n_matched = int(np.unique(matched[matched != -1]).size)
    n_gt_div = sum(1 for v in gt_out.values() if v >= 2)
    return dict(tp=tp, fp=fp, fn=fn, n_pred=len(nodes), n_total=n_total,
                n_gt_edges=len(gt_edges), n_gt_nodes=len(ids), n_matched=n_matched,
                n_gt_div=n_gt_div)


def summarise(rows):
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    edge_jacc = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")

    num = den = 0.0
    for r in rows:
        w = r["tp"] + r["fp"] + r["fn"]
        if w == 0 or not r["n_total"]:
            continue
        j = r["tp"] / w
        ratio = (r["n_pred"] - r["n_total"]) / r["n_total"]
        adj = max(0.0, j * (1 - 0.1 * ratio))
        num += w * adj
        den += w
    adj_edge = num / den if den else float("nan")

    gt_div = sum(r["n_gt_div"] for r in rows)
    div_jacc = 0.0 if gt_div > 0 else float("nan")   # no predicted divisions
    score = adj_edge + 0.1 * div_jacc if gt_div > 0 else adj_edge

    recall = np.mean([r["n_matched"] / r["n_gt_nodes"] for r in rows if r["n_gt_nodes"]])
    return dict(score=score, edge_jaccard=edge_jacc, adj_edge_jaccard=adj_edge,
                division_jaccard=div_jacc, node_recall=float(recall),
                n=len(rows), tp=tp, fp=fp, fn=fn)


def binary_auc_ap(scores, labels):
    """ROC-AUC (midrank Mann-Whitney) and average precision for a small 1-D
    scoring problem (dl_195 daughter channel; no sklearn dependency).
    labels: 1 = positive, anything else = negative. -> (auc, ap), (0, 0) when
    either class is empty."""
    scores = np.asarray(scores, np.float64)
    p = np.asarray(labels) == 1
    n_pos, n_neg = int(p.sum()), int((~p).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    auc = (r[p].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    desc = np.argsort(-scores, kind="mergesort")
    lp = p[desc]
    prec = np.cumsum(lp) / np.arange(1, len(lp) + 1)
    return float(auc), float(prec[lp].mean())
