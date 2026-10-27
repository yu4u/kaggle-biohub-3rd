"""Fixed affine motion correction after lineage selection and graph pruning."""
import numpy as np
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import spsolve

SCALE = np.array([1.625, .40625, .40625])


def ordinary_edges_and_anchors(edges, n_nodes, n_real):
    """Keep real ordinary edges in input order; anchor divisions and gap ends."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    branching = np.bincount(edges[:, 0], minlength=n_nodes) > 1
    division = branching[edges[:, 0]]
    real = (edges < n_real).all(axis=1)
    fixed = np.zeros(n_real, dtype=bool)
    anchors = edges[division | ((edges[:, 0] < n_real) != (edges[:, 1] < n_real))]
    fixed[anchors[anchors < n_real]] = True
    return edges[real & ~division], fixed


def refine_coordinates(nodes, edges, n_real, ordinary_q, shape):
    """Return integer coordinates corrected with fixed strength 2.

    Inputs are the final graph after gap insertion and short-track pruning.
    Real detections occupy the first ``n_real`` rows; synthetic gap nodes follow.
    ``ordinary_q`` is the existing calibrated conditional probability aligned to
    real-to-real nonbranching edges in their input order. ``shape`` is (Z,Y,X).
    Node order, times, synthetic nodes, division nodes and gap ends are preserved.
    The caller retains the original edges. No models, GT or caches are read.
    """
    pairs, fixed = ordinary_edges_and_anchors(edges, len(nodes), n_real)
    q = np.asarray(ordinary_q, dtype=float)
    if q.shape != (len(pairs),) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("ordinary_q must contain one finite nonnegative value per ordinary edge")
    if not len(pairs) or fixed.all():
        return nodes.copy()
    steps = (nodes[pairs[:, 1], 1:] - nodes[pairs[:, 0], 1:]) * SCALE
    times = nodes[pairs[:, 0], 0]
    motion = np.zeros_like(steps)
    for t in np.unique(times):
        on = times == t
        positions = nodes[pairs[on, 0], 1:] * SCALE
        w, d = q[on], steps[on]
        total = float(w.sum())
        if total <= 0:
            motion[on] = np.median(d, axis=0)
            continue
        center = np.sum(w[:, None] * positions, axis=0) / total
        translation = np.sum(w[:, None] * d, axis=0) / total
        spread = np.sqrt(np.sum(w[:, None] * (positions - center)**2, axis=0) / total)
        design = np.column_stack([np.ones(len(w)), (positions - center) / np.where(spread > 0, spread, 1.)])
        coefficients, _, rank, _ = np.linalg.lstsq(design * np.sqrt(w[:, None]), d * np.sqrt(w[:, None]), rcond=None)
        motion[on] = design @ coefficients if rank == 4 else translation

    y = nodes[:n_real, 1:].astype(float) * SCALE
    b = coo_matrix((np.tile([-1., 1.], len(pairs)),
                    (np.repeat(np.arange(len(pairs)), 2), pairs.ravel())),
                   shape=(len(pairs), n_real)).tocsr()
    free = np.flatnonzero(~fixed)
    a = b[:, free]
    matrix = (eye(len(free), format='csc') + 2. * a.T @ a.multiply(q[:, None])).tocsc()
    y[free] += spsolve(matrix, 2. * a.T @ (q[:, None] * (motion - b @ y)))
    points = np.rint(y / SCALE).astype(int)

    result = nodes.copy()
    occupied = set(map(tuple, nodes))
    destinations = set()
    for i, point in enumerate(points):
        target = nodes[i].copy()
        target[1:] = point
        if np.array_equal(target, nodes[i]):
            continue
        key = tuple(target)
        if np.any(point < 0) or np.any(point >= shape) or key in occupied or key in destinations:
            continue
        result[i] = target
        destinations.add(key)
    return result
