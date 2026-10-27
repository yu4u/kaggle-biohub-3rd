"""Portable div778 context compaction and div793 objective pruning."""
import copy
import time
import numpy as np
from scipy.sparse import coo_matrix,vstack
from biohub046.core.division_objective_pruning import prune_negative_divisions,restrict_incumbent

def aggregate_context(problem):
    """In/out capacity is <=1, so context can be summed per node for binary x."""
    counts=np.array([len(e.targets) for e in problem.events],int)
    sources=np.array([e.source for e in problem.events],int);events=np.arange(problem.m)
    target_events=np.repeat(events,counts);targets=np.array([t for e in problem.events for t in e.targets],int)
    div=np.flatnonzero(counts==2);daughters=targets[np.repeat(counts==2,counts)]
    parents=np.unique(sources[div]);children=np.unique(daughters)
    parent_row=np.full(problem.n,-1,int);parent_row[parents]=np.arange(len(parents))
    child_row=np.full(problem.n,-1,int);child_row[children]=len(parents)+np.arange(len(children))
    incoming=parent_row[targets]>=0;outgoing=child_row[sources]>=0
    rr=np.r_[parent_row[sources[div]],parent_row[targets[incoming]],child_row[daughters],child_row[sources[outgoing]]]
    cc=np.r_[div,target_events[incoming],np.repeat(div,2),events[outgoing]]
    vv=np.r_[np.ones(len(div)),-np.ones(incoming.sum()),np.ones(len(daughters)),-np.ones(outgoing.sum())]
    context=coo_matrix((vv,(rr,cc)),shape=(len(parents)+len(children),problem.m+problem.n)).tocsc()
    end=3*problem.n+3*len(div)
    assert end<=len(problem.ub) and np.all(problem.ub[:end]==0)
    before=dict(rows=problem.matrix.shape[0],nonzeros=problem.matrix.nnz,context_rows=3*len(div))
    # The first 3*n rows and any appended exclusion rows retain their order.
    problem.matrix=vstack([problem.matrix[:3*problem.n],context,problem.matrix[end:]],format='csc')
    problem.ub=np.r_[problem.ub[:3*problem.n],np.zeros(context.shape[0]),problem.ub[end:]]
    return dict(before=before,after=dict(rows=problem.matrix.shape[0],nonzeros=problem.matrix.nnz,context_rows=context.shape[0]))

def compact(problem):
    counts = np.array([len(e.targets) for e in problem.events])
    division = np.flatnonzero(counts == 2)
    parents = np.array([problem.events[j].source for j in division])
    daughters = np.array([problem.events[j].targets for j in division], int).reshape(-1, 2)
    both = np.intersect1d(parents, daughters.ravel())
    row = np.full(problem.n, -1, int)
    row[both] = np.arange(len(both))
    nodes = np.r_[parents, daughters.ravel()]
    columns = np.r_[division, np.repeat(division, 2)]
    keep = row[nodes] >= 0
    expected = coo_matrix((np.ones(keep.sum()), (row[nodes[keep]], columns[keep])),
                                  shape=(len(both), problem.m + problem.n)).tocsc()
    end = 3 * problem.n + 3 * len(division)
    # This strengthening requires precisely the existing adjacent-fork exclusion.
    assert problem.matrix[end:].shape == expected.shape
    assert (problem.matrix[end:] != expected).nnz == 0
    np.testing.assert_array_equal(problem.ub[end:], np.ones(len(both)))
    sizes = aggregate_context(problem)
    end = problem.matrix.shape[0] - len(both)
    context = problem.matrix[3 * problem.n:end].tocoo()
    is_division = np.r_[counts == 2, np.zeros(problem.n, bool)]
    keep = ~((context.data < 0) & is_division[context.col])
    reduced = coo_matrix((context.data[keep], (context.row[keep], context.col[keep])),
                                 shape=context.shape).tocsc()
    problem.matrix = vstack([problem.matrix[:3 * problem.n], reduced, problem.matrix[end:]], format='csc')
    sizes['ordinary_only'] = dict(rows=problem.matrix.shape[0], nonzeros=problem.matrix.nnz,
                                  context_rows=reduced.shape[0])
    return sizes

def prune_unused_negative_ordinary(problem, values):
    """Only remove ordinary events absent from every surviving context row."""
    values = np.asarray(values, float)
    n, m = problem.n, problem.m
    if values.shape != (m + n,) or not np.isfinite(values).all() or np.any(values[m:] > 0):
        raise ValueError('requires finite objective and nonpositive node prices')
    node = np.arange(n)
    expected = coo_matrix((np.tile([-1., -1., 1.], n),
        (np.arange(3 * n), np.repeat(node, 3))), shape=(3 * n, n)).tocsc()
    if ((problem.matrix[:3 * n, m:] != expected).nnz or np.any(problem.ub[:3 * n] != 0)
            or problem.matrix[3 * n:, m:].nnz):
        raise ValueError('nonstandard node activation constraints')
    ordinary = np.fromiter((len(event.targets) == 1 for event in problem.events), bool, count=m)
    degree = np.asarray(problem.matrix[3 * n:, :m].getnnz(axis=0)).ravel()
    drop = ordinary & (values[:m] < 0) & (degree == 0)
    audit = dict(ordinary=int(ordinary.sum()), removed=int(drop.sum()),
        supported_negative=int((ordinary & (values[:m] < 0) & (degree > 0)).sum()))
    if not drop.any():
        return problem, values, np.arange(m + n), audit
    events = np.flatnonzero(~drop)
    columns = np.r_[events, m + node]
    reduced = copy.copy(problem)
    reduced.m = len(events)
    reduced.events = [problem.events[j] for j in events]
    for name in ('dataset', 'tp', 'fp', 'dtp', 'dfp', 'extra'):
        setattr(reduced, name, getattr(problem, name)[~drop])
    reduced.matrix = problem.matrix[:, columns]
    return reduced, values[columns], columns, audit

def combined_solve(problem, values, current, time_limit=120.):
    start = time.monotonic()
    first, prices, columns, division = prune_negative_divisions(problem, values)
    reduced, prices, second_columns, ordinary = prune_unused_negative_ordinary(first, prices)
    columns = columns[second_columns]
    incumbent = restrict_incumbent(reduced, current, columns)
    reduction_seconds = time.monotonic() - start
    start = time.monotonic()
    decision, info = reduced.linear_solve(prices, time_limit=time_limit, incumbent=incumbent)
    solve_seconds = time.monotonic() - start
    lifted = np.zeros(problem.m + problem.n)
    lifted[columns] = decision
    assert np.all(problem.matrix @ lifted <= problem.ub + 1e-6)
    np.testing.assert_allclose(values @ lifted, prices @ decision, rtol=0, atol=1e-12)
    return lifted, dict(division=division, ordinary=ordinary, columns=reduced.m + reduced.n,
        reduction_seconds=reduction_seconds, solve_seconds=solve_seconds,
        combined_seconds=reduction_seconds + solve_seconds, solver=info)
