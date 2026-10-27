"""Adopted post_838/840 affine interpolation for synthetic gap nodes only.

The numerical operations and collision fallback match exp_838 exactly. Motion
is estimated from pre-refinement real ordinary edges, while gap coordinates are
updated after the existing real-coordinate refinement.
"""
import numpy as np

from biohub046.core.coordinate_refinement import SCALE


def affine_fields(nodes, pairs, q):
    """Same weighted affine fit and rank fallback as existing coordinate correction."""
    fields = {}
    times = nodes[pairs[:, 0], 0]
    for t in np.unique(times):
        on = times == t
        positions = nodes[pairs[on, 0], 1:] * SCALE
        d = (nodes[pairs[on, 1], 1:] - nodes[pairs[on, 0], 1:]) * SCALE
        w = q[on]
        total = float(w.sum())
        if total <= 0:
            fields[int(t)] = (np.zeros((3, 3)), np.median(d, axis=0))
            continue
        center = np.sum(w[:, None] * positions, axis=0) / total
        translation = np.sum(w[:, None] * d, axis=0) / total
        spread = np.sqrt(np.sum(w[:, None] * (positions - center)**2, axis=0) / total)
        spread = np.where(spread > 0, spread, 1.)
        design = np.column_stack([np.ones(len(w)), (positions - center) / spread])
        coefficients, _, rank, _ = np.linalg.lstsq(design * np.sqrt(w[:, None]), d * np.sqrt(w[:, None]), rcond=None)
        matrix = coefficients[1:] / spread[:, None] if rank == 4 else np.zeros((3, 3))
        offset = coefficients[0] - center @ matrix if rank == 4 else translation
        fields[int(t)] = (matrix, offset)
    return fields


def bridge(start, end, t, duration, fields):
    points = [start.copy()]
    for tt in range(t, t + duration):
        matrix, offset = fields[tt]
        points.append(points[-1] + points[-1] @ matrix + offset)
    values = np.asarray(points)
    values += np.arange(duration + 1)[:, None] / duration * (end - values[-1])
    np.testing.assert_allclose(values[0], start, atol=1e-10)
    np.testing.assert_allclose(values[-1], end, atol=1e-10)
    return values


def refine_gaps(nodes, edges, nr, fields, shape):
    result = nodes.copy()
    successor = np.full(len(nodes), -1, int)
    successor[edges[:, 0]] = edges[:, 1]
    indegree = np.bincount(edges[:, 1], minlength=len(nodes))
    outdegree = np.bincount(edges[:, 0], minlength=len(nodes))
    assert np.all(indegree[nr:] == 1) and np.all(outdegree[nr:] == 1)
    starts = edges[(edges[:, 0] < nr) & (edges[:, 1] >= nr)]
    occupied = set(map(tuple, nodes))
    destinations = set()
    visited = set()
    rejected = missing = 0
    for start, first in starts:
        chain = [int(start), int(first)]
        while chain[-1] >= nr:
            chain.append(int(successor[chain[-1]]))
        synthetic = chain[1:-1]
        assert not visited.intersection(synthetic)
        visited.update(synthetic)
        t = int(nodes[start, 0])
        duration = len(chain) - 1
        assert duration in (2, 3, 4)
        np.testing.assert_array_equal(nodes[chain, 0], np.arange(t, t + duration + 1))
        if any(tt not in fields for tt in range(t, t + duration)):
            missing += 1
            continue
        points = bridge(nodes[start, 1:] * SCALE, nodes[chain[-1], 1:] * SCALE, t, duration, fields)
        for i, point in zip(synthetic, np.rint(points[1:-1] / SCALE).astype(int)):
            target = nodes[i].copy()
            target[1:] = point
            if np.array_equal(target, nodes[i]):
                continue
            key = tuple(target)
            if np.any(point < 0) or np.any(point >= shape) or key in occupied or key in destinations:
                rejected += 1
                continue
            result[i] = target
            destinations.add(key)
    assert visited == set(range(nr, len(nodes)))
    return result, dict(gaps=len(starts), synthetic=len(nodes) - nr, rejected=rejected, missing_fields=missing)
