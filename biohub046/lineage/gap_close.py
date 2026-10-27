"""Multi-frame gap closing (④ graph repair, inference only, no GPU).

Verbatim port of the functions used by submission 046 from the
gap-closing experiment (dl_182): track ends (out-degree 0) and track
heads (in-degree 0) are matched per frame boundary for gaps g = 1..lmax by a
Hungarian assignment on drift-compensated distance with gate g1 * sqrt(g + 1);
matched pairs get g linearly interpolated synthetic nodes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

SCALE = np.array([1.625, 0.40625, 0.40625])  # (z, y, x) µm/voxel


def apply_keep(nodes, edges, keep):
    remap = -np.ones(len(nodes), np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    edges2 = [(int(remap[i]), int(remap[j])) for i, j in edges if keep[i] and keep[j]]
    return nodes[keep], edges2


def comp_sizes(n: int, edges) -> np.ndarray:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        parent[find(int(i))] = find(int(j))
    roots = np.array([find(i) for i in range(n)])
    _, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    return counts[inv]


def gap_close(nodes, edges, lmax: int, g1: float, minfrag: int = 1):
    """Return the graph with interpolated nodes inserted: (nodes2, edges2, n_bridge, n_insert).

    minfrag: minimum connected-component size of both fragments being bridged.
    """
    n = len(nodes)
    t = nodes[:, 0].astype(np.int64)
    pos_um = nodes[:, 1:].astype(np.float64) * SCALE
    out_deg = np.zeros(n, np.int64)
    in_deg = np.zeros(n, np.int64)
    for i, j in edges:
        out_deg[int(i)] += 1
        in_deg[int(j)] += 1
    csize = comp_sizes(n, edges) if minfrag > 1 else np.ones(n, np.int64)

    # per-boundary drift (median displacement of the existing links) and its cumulative sum
    t_max = int(t.max()) if n else 0
    drift = np.zeros((t_max + 1, 3))
    by_t: dict[int, list[np.ndarray]] = {}
    for i, j in edges:
        by_t.setdefault(int(t[int(i)]), []).append(pos_um[int(j)] - pos_um[int(i)])
    for tt, v in by_t.items():
        if len(v) >= 5:
            drift[tt] = np.median(np.stack(v), axis=0)
    cum = np.vstack([np.zeros(3), np.cumsum(drift, axis=0)])   # cum[b]-cum[a] = Σ_{a..b-1}

    ends_at: dict[int, list[int]] = {}
    starts_at: dict[int, list[int]] = {}
    for k in range(n):
        if csize[k] < minfrag:
            continue
        if out_deg[k] == 0 and t[k] < t_max:
            ends_at.setdefault(int(t[k]), []).append(k)
        if in_deg[k] == 0 and t[k] > 0:
            starts_at.setdefault(int(t[k]), []).append(k)

    used_e: set[int] = set()
    used_s: set[int] = set()
    new_nodes: list[np.ndarray] = []
    new_edges: list[tuple[int, int]] = []
    n_bridge = 0
    for g in range(1, lmax + 1):                    # shorter gaps are settled first
        gate = g1 * np.sqrt(g + 1)
        for t_e, E0 in ends_at.items():
            t_s = t_e + g + 1
            if t_s > t_max:
                continue
            E = [e for e in E0 if e not in used_e]
            S = [s for s in starts_at.get(t_s, []) if s not in used_s]
            if not E or not S:
                continue
            exp = pos_um[E] + (cum[t_s] - cum[t_e])            # extrapolate with the cumulative drift
            d = np.linalg.norm(exp[:, None, :] - pos_um[S][None, :, :], axis=2)
            cost = np.where(d > gate, 1e6, d)
            ri, ci = linear_sum_assignment(cost)
            for a, b in zip(ri, ci):
                if d[a, b] > gate:
                    continue
                e, s = E[a], S[b]
                used_e.add(e)
                used_s.add(s)
                n_bridge += 1
                prev = e
                pe, ps = nodes[e, 1:].astype(np.float64), nodes[s, 1:].astype(np.float64)
                for m in range(1, g + 1):
                    zyx = np.rint(pe + (ps - pe) * (m / (g + 1))).astype(np.int64)
                    new_nodes.append(np.array([t_e + m, *zyx], np.int64))
                    cur = n + len(new_nodes) - 1
                    new_edges.append((prev, cur))
                    prev = cur
                new_edges.append((prev, s))

    if not new_nodes:
        return nodes, list(map(tuple, edges)), 0, 0
    nodes2 = np.vstack([nodes, np.stack(new_nodes)])
    edges2 = [(int(i), int(j)) for i, j in edges] + new_edges
    return nodes2, edges2, n_bridge, len(new_nodes)
