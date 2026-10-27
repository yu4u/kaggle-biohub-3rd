"""Remove strictly losing division variables under ordinary-only context."""
import copy
import numpy as np
from scipy.sparse import coo_matrix, vstack


def prune_negative_divisions(problem, values):
    """Return a reduced problem, prices and mapping to the original variables.

    Requires the standard MetricEventProblem node rows, nonpositive node prices,
    and no negative division coefficients outside the no-isolate node rows.
    Removing a negative division then frees capacity without removing support;
    deactivating newly isolated nodes cannot decrease the objective.
    """
    values = np.asarray(values, float)
    n, m = problem.n, problem.m
    if values.shape != (m + n,) or not np.isfinite(values).all():
        raise ValueError('invalid objective vector')
    if np.any(values[m:] > 0):
        raise ValueError('positive node prices are unsupported')
    division = np.fromiter((len(event.targets) == 2 for event in problem.events), bool, count=m)
    drop = np.flatnonzero(division & (values[:m] < 0))
    audit = dict(divisions=int(division.sum()), removed=int(len(drop)),
                 original_rows=problem.matrix.shape[0], original_columns=m + n)
    if not len(drop):
        audit.update(rows=problem.matrix.shape[0], columns=m + n)
        return problem, values, np.arange(m + n), audit
    # No additional constraint may depend on node activation.
    if problem.matrix[3 * n:, m:].nnz:
        raise ValueError('context depends on node activation')
    node = np.arange(n)
    expected = coo_matrix((np.tile([-1., -1., 1.], n),
        (np.arange(3 * n), np.repeat(node, 3))), shape=(3 * n, n)).tocsc()
    if (problem.matrix[:3 * n, m:] != expected).nnz or np.any(problem.ub[:3 * n] != 0):
        raise ValueError('nonstandard node activation rows')
    removed = problem.matrix[:, drop]
    isolate = (removed.indices < 3 * n) & (removed.indices % 3 == 2)
    if np.any((removed.data < 0) & ~isolate):
        raise ValueError('a removed division supports another constraint')
    del removed
    keep = np.ones(m, bool)
    keep[drop] = False
    events = np.flatnonzero(keep)
    columns = np.r_[events, m + node]
    matrix = problem.matrix[:, columns]
    context = matrix[3 * n:].tocsr()
    positive = context.copy()
    positive.data = positive.data > 0
    positive.eliminate_zeros()
    # For x >= 0, an all-nonpositive row with upper bound >= 0 is redundant.
    rows = (np.diff(positive.indptr) > 0) | (problem.ub[3 * n:] < 0)
    reduced = copy.copy(problem)
    reduced.m = len(events)
    reduced.events = [problem.events[j] for j in events]
    for name in ('dataset', 'tp', 'fp', 'dtp', 'dfp', 'extra'):
        setattr(reduced, name, getattr(problem, name)[keep])
    reduced.matrix = vstack([matrix[:3 * n], context[rows]], format='csc')
    reduced.ub = np.r_[problem.ub[:3 * n], problem.ub[3 * n:][rows]]
    audit.update(rows=reduced.matrix.shape[0], columns=reduced.matrix.shape[1])
    return reduced, values[columns], columns, audit


def restrict_incumbent(reduced, original, columns):
    """Drop eliminated events and clear their now-isolated node activations."""
    result = np.asarray(original, float)[columns].copy()
    result[reduced.m:] = 0
    for j in np.flatnonzero(result[:reduced.m]):
        event = reduced.events[j]
        result[reduced.m + event.source] = 1
        result[reduced.m + np.asarray(event.targets)] = 1
    if not np.all(reduced.matrix @ result <= reduced.ub + 1e-6):
        raise ValueError('invalid restricted incumbent')
    return result
