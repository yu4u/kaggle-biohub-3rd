"""Shared pooled count algebra for optimization and reported metric variants."""
import numpy as np


def node_factor(nodes, total, clamp=False):
    factor = 1.1 - .1*np.asarray(nodes)/np.asarray(total)
    return np.minimum(factor, 1.) if clamp else factor


def pooled_components(tp, fp, nodes, dtp, dfp, total, gt_edges, gt_divisions, clamp=False):
    de = float(np.sum(gt_edges)+np.sum(fp))
    dd = float(np.sum(gt_divisions)+np.sum(dfp))
    num = float(np.maximum(0., node_factor(nodes, total, clamp)*np.asarray(tp)).sum())
    je = num/de if de else 0.
    jd = float(np.sum(dtp))/dd if dd else 0.
    return dict(edge_den=de, division_den=dd, edge_jaccard=je,
                division_jaccard=jd, score=je+.1*jd)
