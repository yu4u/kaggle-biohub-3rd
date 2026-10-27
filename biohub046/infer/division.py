"""Division attachment (stage 4) for submission 006 onwards.

The production wiring is strictly one-to-one, so every GT division loses an
edge AND the whole `0.1 * division_jaccard` term. This adds a second child to
parents that look like mothers, using the rule dl_136 arrived at on the 5-fold
OOF (199 samples, 151 GT divisions, scored with the competition's own
division metric):

    5-fold OOF   0.8980 -> 0.9116   (adj_edge unchanged at 0.8980,
                                     div_j 0 -> 0.1359, TP/FN/FP 28/123/55)

The matcher is NOT the detector here — measured, it cannot place a second
daughter (reverse probability p50 0.892 for the daughter it links vs 0.065 for
the one it drops). It contributes only as a filter: the daughter must be the
parent's argmax when the matcher is re-run with the already-claimed nodes
REMOVED FROM ITS INPUT (renormalising the softmax afterwards is too late — the
cross-attention has already conditioned on the taken child).
"""
import numpy as np
import torch

SCALE = np.array([1.625, 0.40625, 0.40625])   # (z, y, x) µm/voxel

# dl_136's operating point, picked from the plateau (d 8-9 x lineage>=7 x
# daughters 12-15 all score 0.9109-0.9116; with only 151 GT divisions a div_j
# step of 0.01 is ~1.5 events, so the peak is not worth chasing)
D_MAX_UM = 8.0        # daughter within this of the drift-predicted mother
RES_MIN_UM = 2.5      # the existing child must itself have moved off the drift
COS_MAX = -0.6        # the two daughters open at >=127 degrees
LINEAGE_MIN = 7       # the mother already existed this long
DAUGHTER_MIN = 15     # both daughters persist this long


@torch.no_grad()
def second_round(per_t, edges, model, cfg, device, drifts, k=1):
    """Re-run the matcher per boundary with the claimed frame-b nodes removed
    from its INPUT; return each node's top-k successor among what remains."""
    pos = [np.asarray(p, np.float64).reshape(-1, 3) for p, _ in per_t]
    offs, n = [0], 0
    for p in pos:
        n += len(p)
        offs.append(n)
    claimed = np.zeros(n, bool)
    for _, j in edges:
        claimed[int(j)] = True

    idx = -np.ones((n, k), np.int64)
    prob = np.zeros((n, k), np.float32)
    radius = float(cfg.loss.embed_radius_um)
    scale = torch.tensor(SCALE, dtype=torch.float32, device=device)
    for t in range(len(pos) - 1):
        pa, ea = per_t[t]
        pb, eb = per_t[t + 1]
        free = np.nonzero(~claimed[offs[t + 1]:offs[t + 2]])[0]
        if not len(pa) or not len(free):
            continue
        fa = torch.from_numpy(ea).to(device)
        fb = torch.from_numpy(eb[free]).to(device)
        ga = torch.from_numpy(pa.astype(np.float32)).to(device) * scale
        gb = torch.from_numpy(pb[free].astype(np.float32)).to(device) * scale
        offset = torch.as_tensor(drifts[t], device=device,
                                 dtype=torch.float32).expand(len(pa), 3)
        logits, null_a, _ = model.matcher(fa, ga, fb, gb, radius,
                                          candidate_offset_a=offset)
        p_ab = torch.softmax(torch.cat([logits, null_a[:, None]], 1), 1)[:, :-1]
        p_ab = p_ab.float().cpu().numpy()
        kk = min(k, p_ab.shape[1])
        order = np.argsort(-p_ab, axis=1)[:, :kk]
        rows = np.arange(len(pa))[:, None]
        idx[offs[t]:offs[t + 1], :kk] = free[order] + offs[t + 1]
        prob[offs[t]:offs[t + 1], :kk] = p_ab[rows, order]
    return idx, prob


def keep_mask(nodes, edges, min_len):
    """The survivors of filter_short_tracks, as a mask over the input nodes."""
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
    return counts[inv] >= min_len


def _forward_len(edges, n):
    nxt = {}
    for i, j in edges:
        nxt.setdefault(int(i), []).append(int(j))
    length = np.ones(n, np.int32)
    for i in range(n - 1, -1, -1):
        for j in nxt.get(i, ()):
            if j > i:
                length[i] = max(length[i], 1 + length[j])
    return length


def _lineage_len(edges, n):
    prev = {}
    for i, j in edges:
        prev.setdefault(int(j), []).append(int(i))
    length = np.ones(n, np.int32)
    for j in range(n):
        for i in prev.get(j, ()):
            if i < j:
                length[j] = max(length[j], 1 + length[i])
    return length


def _frame_drift(nodes, edges, pos_um):
    by_t = {}
    for i, j in edges:
        by_t.setdefault(int(nodes[i, 0]), []).append(pos_um[j] - pos_um[i])
    return {tt: np.median(np.stack(v), axis=0) for tt, v in by_t.items() if len(v) >= 5}


def add_divisions(nodes, edges, keep, r2_idx, r2_prob, r2_top1=True,
                  d_max_um=D_MAX_UM, cos_max=COS_MAX,
                  daughter_min=DAUGHTER_MIN, par_score=None, par_min=0.0):
    """Return `edges` plus one second child for each accepted mother.

    Operates in PRE-filter index space with the keep mask computed from the
    original wiring, so a division link can never resurrect a node the
    short-track filter had already removed.

    `r2_top1` requires the daughter to be the second-round matcher's own top
    candidate. That gate was fitted on the per-boundary Hungarian wiring; on the
    global-LP wiring it turns redundant and only rejects true divisions
    (dl_296: div_j 0.0960 -> 0.1435, +0.0048 on the 199-sample OOF), so the LP
    submission passes False and every unclaimed track start at t+1 is
    considered, ranked by its second-round probability.

    `par_score` (div_325, submission 013): per-node pre-division probability
    from the div3f 3-frame classifier. When given, mothers must clear
    `par_score[p] >= par_min` and proposals are RANKED by it instead of the
    second-round probability. The geometry is then relaxed to the
    pre-registered arm (d<=12 um / cos<=0 / both daughters >=10) — measured
    0.9254 -> 0.9346 on the 199-sample OOF full chain (div tp 30 -> 42,
    fp 58 -> 28)."""
    edges = list(edges)
    n = len(nodes)
    out_deg = np.zeros(n, np.int32)
    in_deg = np.zeros(n, np.int32)
    child_of = {}
    for i, j in edges:
        out_deg[int(i)] += 1
        in_deg[int(j)] += 1
        child_of.setdefault(int(i), []).append(int(j))

    t = nodes[:, 0]
    pos_um = nodes[:, 1:].astype(np.float64) * SCALE
    fwd = _forward_len(edges, n)
    lin = _lineage_len(edges, n)
    drift = _frame_drift(nodes, edges, pos_um)

    proposals = []
    for p in np.nonzero((out_deg == 1) & keep)[0]:
        c1 = child_of[int(p)][0]
        if not keep[c1] or lin[p] < LINEAGE_MIN or fwd[c1] < daughter_min:
            continue
        if par_score is not None and float(par_score[p]) < par_min:
            continue
        tt = int(t[p])
        step = drift.get(tt, np.zeros(3))
        v1 = pos_um[c1] - pos_um[p] - step
        if np.linalg.norm(v1) < RES_MIN_UM:
            continue
        cands = ([int(r2_idx[p, 0])] if r2_top1
                 else np.nonzero((t == tt + 1) & (in_deg == 0) & keep)[0])
        for s2 in cands:
            s2 = int(s2)
            if s2 < 0 or s2 == c1 or not keep[s2] or in_deg[s2] != 0:
                continue
            if int(t[s2]) != tt + 1 or fwd[s2] < daughter_min:
                continue
            v2 = pos_um[s2] - pos_um[p] - step
            d = float(np.linalg.norm(v2))
            n1 = float(np.linalg.norm(v1))
            if d > d_max_um or n1 < 1e-6 or d < 1e-6:
                continue
            if float(v1 @ v2) / (n1 * d) > cos_max:
                continue
            if par_score is not None:
                score = float(par_score[p])
            else:
                hit = np.nonzero(r2_idx[p] == s2)[0]
                score = float(r2_prob[p, hit[0]]) if len(hit) else 0.0
            proposals.append((score, int(p), s2))

    proposals.sort(reverse=True)
    used_p, used_s = set(), set()
    for _, p, s in proposals:
        if p in used_p or s in used_s:
            continue
        used_p.add(p)
        used_s.add(s)
        edges.append((p, s))
    return edges


def apply_keep(nodes, edges, keep):
    remap = -np.ones(len(nodes), np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    edges2 = [(int(remap[i]), int(remap[j])) for i, j in edges if keep[i] and keep[j]]
    return nodes[keep], edges2
