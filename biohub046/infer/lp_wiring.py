"""Stage-③ wiring as one global optimisation (dl_296), packaged for submissions.

The per-boundary Hungarian is replaced by a single LP over the whole movie whose
objective is derived from the competition metric:

    score = [Σ p_a x_a / (K + Σ f_a x_a)] · (1 − 0.1 (Σ y_i − n_total)/n_total)

Linearising the ratio (Dinkelbach) gives `max Σ[p_a − J f_a] x_a − q Σ y_i`, and
the constraints `out(i) ≤ y_i`, `in(j) ≤ y_j` form a network-flow matrix, so the
LP relaxation is integral — HiGHS returns the exact global optimum.

Measured on the 199-sample OOF (dl_296): wiring 0.9062 → 0.9080 with the
ensemble detections, 0.9006 → 0.9022 with dl_079's, i.e. the gain does not
depend on the detector. Node pruning is left to the existing L6 filter: the
LP's node price prunes whole tracks rather than junk (q x10 collapses to 0.8355).

`p_a` is a calibrated P[true link] fitted on all 199 OOF samples
(train-fit AUC 0.9951); the fold-wise
run of the same fit reports held-out AUC 0.9936-0.9970.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

SCALE = np.array([1.625, 0.40625, 0.40625])   # (z, y, x) µm/voxel
TOPK = 5
ANNOT_RATE = 0.028          # GT annotates ~2.8% of cells
CAND_UM = 12.0              # candidate radius for the LP (matcher's own gate)

FEATS = ("d", "d2", "logd", "p", "rev", "sym", "r2", "top1", "rank",
         "d_rel", "p_rel", "ncand", "bias")
W = np.array([2.366962e+01, -1.139539e+01, -1.783497e+01, 5.689715e+00,
              -2.913233e+00, -1.239305e+00, 5.546593e-03, 9.698265e-02,
              -6.312752e-02, 1.493242e-01, 2.040867e-01, 0.0, 0.0])
MU = np.array([2.531014e+01, 1.080983e+02, 2.714330e+00, 1.964808e-01,
               1.978971e-01, 2.454469e-01, 5.960438e-03, 4.602382e-02,
               2.0, 2.424668e+01, -7.569327e-01, 5.0, 1.0])
SD = np.array([2.098522e+01, 1.156074e+02, 1.245531e+00, 3.831569e-01,
               3.848117e-01, 3.947036e-01, 6.109830e-02, 2.095367e-01,
               1.414214e+00, 2.110053e+01, 3.937005e-01, 1e-9, 1e-9])


def top_candidates(per_t, sims, revs, syms, offsets, k=TOPK):
    """Per node, its k best successors by matcher probability, with the
    flow-compensated distance alongside ."""
    pos = [np.asarray(p, np.float64).reshape(-1, 3) for p, _ in per_t]
    offs, n = [0], 0
    for p in pos:
        n += len(p)
        offs.append(n)
    idx = -np.ones((n, k), np.int64)
    prob = np.zeros((n, k), np.float32)
    rev = np.zeros((n, k), np.float32)
    sym = np.zeros((n, k), np.float32)
    dist = np.full((n, k), np.inf, np.float32)
    for t in range(len(pos) - 1):
        a, b = pos[t] * SCALE, pos[t + 1] * SCALE
        if not len(a) or not len(b):
            continue
        s = sims[t]
        d = np.linalg.norm((a + offsets[t])[:, None, :] - b[None, :, :], axis=2)
        kk = min(k, s.shape[1])
        order = np.argsort(-s, axis=1)[:, :kk]
        rows = np.arange(len(a))[:, None]
        idx[offs[t]:offs[t + 1], :kk] = order + offs[t + 1]
        prob[offs[t]:offs[t + 1], :kk] = s[rows, order]
        rev[offs[t]:offs[t + 1], :kk] = revs[t][rows, order]
        sym[offs[t]:offs[t + 1], :kk] = syms[t][rows, order]
        dist[offs[t]:offs[t + 1], :kk] = d[rows, order]
    return idx, prob, rev, sym, dist


def calibrated_prob(ci, cp, crev, csym, cd, r2i, r2p):
    """P[candidate is the true successor] — the number the LP objective needs
    to be comparable across frame boundaries."""
    n, k = ci.shape
    valid = ci >= 0
    d = np.where(valid, np.minimum(cd, 50.0), 0.0)
    r2 = np.zeros_like(cp)
    for col in range(r2i.shape[1]):
        hit = (r2i[:, col][:, None] == ci) & valid
        r2 = np.where(hit, r2p[:, col][:, None], r2)
    top1 = ((r2i[:, 0][:, None] == ci) & valid).astype(np.float64)
    dmin = np.where(valid, d, np.inf).min(1, keepdims=True)
    dmin[~np.isfinite(dmin)] = 0.0
    pmax = np.where(valid, cp, -np.inf).max(1, keepdims=True)
    pmax[~np.isfinite(pmax)] = 0.0
    ncand = valid.sum(1, keepdims=True).astype(np.float64)
    f = np.stack([
        d, d ** 2 / 10.0, np.log1p(d), cp, crev, csym, r2, top1,
        np.tile(np.arange(k, dtype=np.float64), (n, 1)),
        d - dmin, cp - pmax, np.tile(ncand, (1, k)), np.ones((n, k)),
    ], axis=2) * valid[:, :, None]
    z = ((f - MU) / SD) @ W
    return np.where(valid, 1.0 / (1.0 + np.exp(-z)), 0.0)


def solve(n_nodes, arcs, p, f, n_total, k_gt, iters=2):
    """Dinkelbach on J = Σp x / (K + Σf x) with the linear n_pred term."""
    m = len(arcs)
    if not m:
        return np.zeros(0, bool)
    src, dst = arcs[:, 0], arcs[:, 1]
    rows = np.concatenate([src, np.arange(n_nodes),
                           n_nodes + dst, n_nodes + np.arange(n_nodes)])
    cols = np.concatenate([np.arange(m), m + np.arange(n_nodes),
                           np.arange(m), m + np.arange(n_nodes)])
    vals = np.concatenate([np.ones(m), -np.ones(n_nodes),
                           np.ones(m), -np.ones(n_nodes)])
    A = coo_matrix((vals, (rows, cols)), shape=(2 * n_nodes, m + n_nodes)).tocsr()
    b = np.zeros(2 * n_nodes)
    J, u, fx, x = 0.9, 1.0, float(k_gt), np.zeros(m)
    for _ in range(iters):
        q = 0.1 * J * (k_gt + fx) / max(u, 1e-6) / max(n_total, 1.0)
        cost = np.concatenate([-(p - J * f), q * np.ones(n_nodes)])
        res = linprog(cost, A_ub=A, b_ub=b, bounds=(0, 1), method="highs")
        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")
        x, y = res.x[:m], res.x[m:]
        tp, fx = float(p @ x), float(f @ x)
        J = tp / max(k_gt + fx, 1e-9)
        u = 1.0 - 0.1 * (y.sum() - n_total) / max(n_total, 1.0)
    return x > 0.5


def lp_wiring(per_t, sims, revs, syms, offsets, r2_idx, r2_prob):
    """-> (nodes (N,4) t,z,y,x int, edges [(i,j)]) from the global LP."""
    nodes = np.concatenate([
        np.column_stack([np.full(len(np.asarray(p).reshape(-1, 3)), t),
                         np.rint(np.asarray(p).reshape(-1, 3)).astype(np.int64)])
        for t, (p, _) in enumerate(per_t)]).astype(np.int64)
    ci, cp, crev, csym, cd = top_candidates(per_t, sims, revs, syms, offsets)
    pcal = calibrated_prob(ci, cp, crev, csym, cd, r2_idx, r2_prob)
    ok = ((ci >= 0) & (cd <= CAND_UM)).ravel()
    src = np.repeat(np.arange(len(nodes)), ci.shape[1])[ok]
    dst = ci.ravel()[ok]
    arcs = np.stack([src, dst], 1).astype(np.int64)
    pr = pcal.ravel()[ok].astype(np.float64)
    sel = solve(len(nodes), arcs, pr * ANNOT_RATE, (1.0 - pr) * ANNOT_RATE,
                n_total=float(len(nodes)),
                k_gt=float(len(nodes) * ANNOT_RATE))
    return nodes, [(int(i), int(j)) for i, j in arcs[sel]]
