"""Free-node event selection with the competition's pooled count objective.

This is a solver core, not a probability model. Inputs are expected TP/FP counts
for continuation, gap and TWO-daughter division events. They must be estimated
without the test GT. The optimized ratio-of-expected-counts is a surrogate for
expected official score; matching and division-component dependencies are not
represented by independent event counts.
"""
from dataclasses import dataclass
from collections import deque

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, linprog
from scipy.sparse import coo_matrix
from biohub046.core.metric_counts import pooled_components, node_factor


@dataclass(frozen=True)
class MetricEvent:
    source: int
    targets: tuple[int, ...]
    edge_tp: float
    edge_fp: float
    division_tp: float = 0.
    division_fp: float = 0.
    missing_nodes: int = 0


@dataclass(frozen=True)
class CountPrior:
    estimated_nodes: float
    gt_edges: float
    gt_divisions: float


def pooled_score(tp, fp, node_count, div_tp, div_fp, priors, clamp=False):
    return pooled_components(tp, fp, node_count, div_tp, div_fp,
        [p.estimated_nodes for p in priors], [p.gt_edges for p in priors],
        [p.gt_divisions for p in priors], clamp)['score']


class MetricEventProblem:
    """One global problem, optionally containing multiple independent videos.

    Any original link may be replaced; there are no immutable tracklets.
    A division is one event that consumes the parent's outgoing choice and
    BOTH daughters' incoming slots. Gap events consume the same slots.
    """
    def __init__(self, node_dataset, events, priors, *, require_division_context=True,
                 node_times=None, clamp=False):
        self.clamp = clamp
        self.node_dataset = np.asarray(node_dataset, int)
        self.events = list(events)
        self.priors = list(priors)
        self.n, self.m = len(self.node_dataset), len(self.events)
        self.s = len(self.priors)
        if not self.s or self.node_dataset.ndim != 1:
            raise ValueError('provide a prior per dataset and a node dataset vector')
        if self.n and (self.node_dataset.min() < 0 or self.node_dataset.max() >= self.s):
            raise ValueError('invalid node dataset')
        for p in self.priors:
            if (not np.isfinite([p.estimated_nodes, p.gt_edges, p.gt_divisions]).all()
                    or p.estimated_nodes <= 0 or p.gt_edges < 0 or p.gt_divisions < 0):
                raise ValueError('invalid count prior')
        incoming, outgoing = [[] for _ in range(self.n)], [[] for _ in range(self.n)]
        times = None if node_times is None else np.asarray(node_times)
        if times is not None and (times.shape != (self.n,) or not np.isfinite(times).all()
                                  or np.any(times < 0) or np.any(times != np.rint(times))):
            raise ValueError('invalid node timepoints')
        counts = np.array([len(e.targets) for e in self.events], int)
        if np.any((counts<1)|(counts>2)):
            raise ValueError('invalid event endpoints')
        sources = np.array([e.source for e in self.events], int)
        target_event = np.repeat(np.arange(self.m), counts)
        targets = np.array([t for e in self.events for t in e.targets], int)
        numbers = np.array([[e.edge_tp,e.edge_fp,e.division_tp,e.division_fp,e.missing_nodes]
                            for e in self.events], float).reshape(-1,5)
        if (np.any(sources<0) or np.any(sources>=self.n) or np.any(targets<0)
                or np.any(targets>=self.n) or np.any(sources[target_event]==targets)):
            raise ValueError('invalid event endpoints')
        if np.any(self.node_dataset[sources[target_event]]!=self.node_dataset[targets]):
            raise ValueError('cross-dataset event')
        keys = np.array([(e.source,*sorted(e.targets),-1)[:3] if len(e.targets)==2
                         else (e.source,e.targets[0],-1) for e in self.events],int).reshape(-1,3)
        if (len(np.unique(keys,axis=0))!=self.m
                or any(len(e.targets)==2 and e.targets[0]==e.targets[1] for e in self.events)):
            raise ValueError('duplicate event or endpoints')
        if (not np.isfinite(numbers).all() or np.any(numbers<0)
                or np.any(numbers[:,4]!=np.rint(numbers[:,4]))
                or np.any(numbers[:,0]+numbers[:,1]>counts+numbers[:,4]+1e-8)
                or np.any(numbers[:,2]+numbers[:,3]>1+1e-8)
                or np.any(numbers[counts==1,2:4]!=0) or np.any(numbers[counts==2,4]!=0)):
            raise ValueError('invalid expected counts')
        if times is not None and np.any(times[targets]-times[sources[target_event]]!=numbers[target_event,4]+1):
            raise ValueError('event timepoints disagree with missing node count')
        for j,e in enumerate(self.events):
            outgoing[e.source].append(j)
            for t in e.targets:incoming[t].append(j)
        # Valid consecutive timestamps already prove acyclicity.
        if times is None:
            degree = np.array([len(v) for v in incoming])
            queue = deque(np.flatnonzero(degree == 0).tolist())
            visited = 0
            while queue:
                node = queue.popleft()
                visited += 1
                for j in outgoing[node]:
                    for target in self.events[j].targets:
                        degree[target] -= 1
                        if degree[target] == 0:
                            queue.append(target)
            if visited != self.n:
                raise ValueError('candidate event graph must be acyclic')
        self.dataset = np.array([self.node_dataset[e.source] for e in self.events], int)
        self.tp = np.array([e.edge_tp for e in self.events])
        self.fp = np.array([e.edge_fp for e in self.events])
        self.dtp = np.array([e.division_tp for e in self.events])
        self.dfp = np.array([e.division_fp for e in self.events])
        self.extra = np.array([e.missing_nodes for e in self.events], int)
        # Vectorized in/out capacity and node-incidence constraints.
        node = np.arange(self.n)
        event = np.arange(self.m)
        rr = [3*targets,3*sources+1,3*targets+2,3*sources+2,3*node,3*node+1,3*node+2]
        cc = [target_event,event,target_event,event,self.m+node,self.m+node,self.m+node]
        vv = [np.ones(len(targets)),np.ones(self.m),-np.ones(len(targets)),-np.ones(self.m),
              -np.ones(self.n),-np.ones(self.n),np.ones(self.n)]
        r_context=[];c_context=[];v_context=[];row=3*self.n
        if require_division_context:
            for j in np.flatnonzero(counts==2):
                e=self.events[j]
                for support in [incoming[e.source],*[outgoing[t] for t in e.targets]]:
                    r_context.extend([row]*(1+len(support)))
                    c_context.extend([j,*support]);v_context.extend([1.,*[-1.]*len(support)])
                    row+=1
        rr.append(np.asarray(r_context,int));cc.append(np.asarray(c_context,int));vv.append(np.asarray(v_context,float))
        self.matrix=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),
                              shape=(row,self.m+self.n)).tocsc()
        self.ub=np.zeros(row)

    def counts(self, decision):
        x, y = decision[:self.m], decision[self.m:]
        by_event = lambda v: np.bincount(self.dataset, weights=v*x, minlength=self.s)
        nodes = np.bincount(self.node_dataset, weights=y, minlength=self.s) + by_event(self.extra)
        return by_event(self.tp), by_event(self.fp), nodes, by_event(self.dtp), by_event(self.dfp)

    def objective(self, decision):
        return pooled_score(*self.counts(decision), self.priors, clamp=self.clamp)

    def gradient(self, decision, pooled=None):
        """Marginal prices from the POOLED metric, not per-sample Jaccards."""
        tp, fp, nodes, dtp, dfp = self.counts(decision)
        totals = np.array([p.estimated_nodes for p in self.priors])
        raw_u = node_factor(nodes, totals)
        u = node_factor(nodes, totals, self.clamp)
        if pooled is None:
            pooled = pooled_components(tp, fp, nodes, dtp, dfp, totals,
                [p.gt_edges for p in self.priors], [p.gt_divisions for p in self.priors], self.clamp)
        de = max(pooled['edge_den'], 1e-12)
        dd = pooled['division_den']
        je, jd = pooled['edge_jaccard'], pooled['division_jaccard']
        active = (raw_u > 0) & ((raw_u < 1) if self.clamp else True)
        node_price = np.where(active, .1*tp/(totals*de), 0.)
        event_value = (np.maximum(0., u[self.dataset])*self.tp-je*self.fp)/de
        if dd > 0:
            event_value += .1*(self.dtp-jd*self.dfp)/dd
        event_value -= self.extra*node_price[self.dataset]
        return np.r_[event_value, -node_price[self.node_dataset]]

    def linear_solve(self, values, *, time_limit=30., incumbent=None):
        if not self.n:
            return np.zeros(0), {'status': 0, 'mip_gap': 0.}
        values = np.asarray(values, float)
        # Metric derivatives are ~1/N. Normalize to avoid solver feasibility /
        # objective tolerances dwarfing the score differences being optimized.
        cost = -values/max(float(np.max(np.abs(values))), 1e-12)
        nonpositive_nodes = np.all(values[self.m:] <= 0)
        if nonpositive_nodes:
            # In/out capacity + node activation is a network matrix. The
            # no-isolate rows are unnecessary with nonpositive node rewards;
            # remove unused activations after solving the integral relaxation.
            rows = np.array([r for r in range(len(self.ub)) if r >= 3*self.n or r % 3 != 2])
            matrix, upper = self.matrix[rows], self.ub[rows]
        else:
            matrix, upper = self.matrix, self.ub
        relaxed = linprog(cost, A_ub=matrix, b_ub=upper, bounds=(0., 1.), method='highs',
                          options={'time_limit': time_limit})
        if relaxed.success and np.max(np.abs(relaxed.x-np.rint(relaxed.x))) <= 1e-6:
            x = np.rint(relaxed.x)
            if nonpositive_nodes:
                x[self.m:] = 0
                for active, event in zip(x[:self.m], self.events):
                    if active:
                        x[self.m+event.source] = 1
                        x[self.m+np.array(event.targets)] = 1
            if np.all(self.matrix @ x <= self.ub+1e-6):
                return x, {'status': 0, 'mip_gap': 0., 'solver': 'integral LP'}
        integrality = np.ones(self.m+self.n)
        if nonpositive_nodes:
            # Conditional on binary division choices, the remaining in/out
            # capacities and context requirements are an integral flow problem.
            integrality[:] = 0
            integrality[:self.m] = [len(e.targets) == 2 for e in self.events]
        result = milp(cost, integrality=integrality,
                      bounds=Bounds(0., 1.),
                      constraints=LinearConstraint(matrix, -np.inf, upper),
                      options={'time_limit': time_limit, 'mip_rel_gap': 0.})
        if result.x is None:
            if incumbent is not None:
                x = np.asarray(incumbent, float)
                if (x.shape == (self.m+self.n,) and np.all((x == 0) | (x == 1))
                        and np.all(self.matrix @ x <= self.ub+1e-6)):
                    return x.copy(), {'status': int(result.status), 'mip_gap': None,
                                      'solver': 'retained incumbent', 'message': result.message}
            raise RuntimeError(f'no feasible solution: {result.message}')
        x = np.rint(result.x)
        if nonpositive_nodes and np.max(np.abs(x-result.x)) <= 1e-5:
            x[self.m:] = 0
            for active, event in zip(x[:self.m], self.events):
                if active:
                    x[self.m+event.source] = 1
                    x[self.m+np.array(event.targets)] = 1
        # Removing zero-price isolated nodes may change y by one legitimately.
        if (np.max(np.abs(np.rint(result.x)-result.x)) > 1e-5 or np.any(x < 0) or np.any(x > 1)
                or np.any(self.matrix @ x > self.ub+1e-6)):
            raise RuntimeError('invalid integer incumbent')
        return x, {'status': int(result.status), 'mip_gap': float(result.mip_gap), 'solver': 'MILP'}


def save_problem(problem, path):
    """Store validated immutable numeric structure; no executable pickle payload."""
    p = problem
    np.savez(path, node_dataset=p.node_dataset, dataset=p.dataset,
             source=[e.source for e in p.events],
             targets=[[e.targets[0], e.targets[1] if len(e.targets)==2 else -1] for e in p.events],
             tp=p.tp, fp=p.fp, dtp=p.dtp, dfp=p.dfp, extra=p.extra,
             priors=[[v.estimated_nodes,v.gt_edges,v.gt_divisions] for v in p.priors],
             clamp=p.clamp, ub=p.ub, data=p.matrix.data, indices=p.matrix.indices,
             indptr=p.matrix.indptr, shape=p.matrix.shape)


def restore_problem(path):
    """Restore the immutable structure written during this run's initialization."""
    from scipy.sparse import csc_matrix
    p = MetricEventProblem.__new__(MetricEventProblem)
    with np.load(path, allow_pickle=False) as z:
        for name in ['node_dataset','dataset','tp','fp','dtp','dfp','extra','ub']:
            setattr(p,name,z[name])
        p.clamp=bool(z['clamp'])
        p.priors=[CountPrior(*v) for v in z['priors']]
        p.n,p.m,p.s=len(p.node_dataset),len(p.tp),len(p.priors)
        p.events=[MetricEvent(int(i),tuple(int(t) for t in ts if t>=0),tp,fp,dtp,dfp,int(g))
                  for i,ts,tp,fp,dtp,dfp,g in zip(z['source'],z['targets'],p.tp,p.fp,p.dtp,p.dfp,p.extra)]
        p.matrix=csc_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']))
    return p
