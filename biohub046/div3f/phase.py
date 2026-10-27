"""Lineage-supported labels for independent pre-parent and newborn-daughter models."""
import numpy as np


def phase_labels(gt, drop=None):
    """Return {t: (N_t, 2)} in GT row order; -1 means insufficient evidence.

    pre: a single outgoing edge leads to a parent splitting one frame later.
    post: the incoming parent split in the immediately preceding frame.
    Incomplete track ends and anomalous/gapped neighborhoods stay unknown.
    """
    ids, ts, _, edges, _ = gt
    lookup = {int(g): i for i, g in enumerate(ids)}
    children = [[] for _ in ids]
    parents = [[] for _ in ids]
    bad_out, bad_in = set(), set()
    for u, v in edges:
        i, j = lookup[int(u)], lookup[int(v)]
        children[i].append(j)
        parents[j].append(i)
        if ts[j] != ts[i] + 1 or (drop and (int(u), int(v)) in drop):
            bad_out.add(i)
            bad_in.add(j)

    def known_out(i):
        return i not in bad_out and len(children[i]) in (1, 2)

    labels = np.full((len(ids), 2), -1, np.float32)
    for i in range(len(ids)):
        if known_out(i):
            if len(children[i]) == 2:
                labels[i, 0] = 0
            else:
                j = children[i][0]
                if len(parents[j]) == 1 and known_out(j):
                    labels[i, 0] = float(len(children[j]) == 2)
        if i not in bad_in and len(parents[i]) == 1:
            p = parents[i][0]
            if known_out(p):
                labels[i, 1] = float(len(children[p]) == 2)
    return {int(t): labels[ts == t] for t in np.unique(ts)}
