"""Joint tracklet selection, gap assignment and division attachment from caches.

No models, ground truth, or dataset identity enter inference. Existing one-to-one
links define atomic tracklets; this deliberately does not cut confident links.
A sparse binary program chooses tracklets and mutually compatible events. The
objective is a tunable surrogate, NOT the (unobservable) official test metric.
"""
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

SCALE = np.array([1.625, .40625, .40625])


@dataclass(frozen=True)
class OptimizerConfig:
    track_cost: float = 5.5
    max_gap: int = 2
    gap_radius: float = 3.0
    gap_distance_cost: float = 1.0
    synthetic_node_cost: float = .25
    division_threshold: float = .7
    division_weight: float = 4.0
    division_radius: float = 12.0
    division_cos: float = 0.0
    division_residual: float = 2.5
    parent_length: int = 7
    daughter_length: int = 10
    division_geometry_cost: float = .1
    time_limit: float = 60.0
    relative_gap: float = 0.0
    confidence_weight: float = 0.0
    confidence_pivot: float = 0.5
    confidence_bonus: bool = False
    # "joint" compares gaps and divisions in the same optimization. "division"
    # accepts an already gap-closed graph and only optimizes division assignment.
    mode: str = "joint"

    def __post_init__(self):
        if self.mode not in ("joint", "division"):
            raise ValueError("mode must be joint or division")
        if not 0 < self.division_threshold < 1:
            raise ValueError("division_threshold must be in (0,1)")
        if (self.max_gap < 0 or self.gap_radius <= 0 or self.track_cost < 0
                or self.time_limit <= 0 or self.relative_gap < 0
                or self.division_radius <= 0 or self.division_weight < 0
                or self.parent_length < 1 or self.daughter_length < 1
                or not -1 <= self.division_cos <= 1
                or self.gap_distance_cost < 0 or self.division_geometry_cost < 0
                or self.synthetic_node_cost < 0
                or not 0 <= self.confidence_weight < 1 or self.confidence_pivot <= 0):
            raise ValueError("invalid optimizer configuration")
        if not all(np.isfinite(v) for v in asdict(self).values()
                   if isinstance(v, (int, float))):
            raise ValueError("configuration must be finite")


@dataclass(frozen=True)
class Event:
    source: int
    target: int
    kind: str
    reward: float


def validate_graph(nodes, edges, *, branching=False):
    nodes = np.asarray(nodes)
    raw = np.asarray(edges)
    if nodes.ndim != 2 or nodes.shape[1] != 4:
        raise ValueError("nodes must have shape (N,4), t,z,y,x in native voxels")
    if not np.isfinite(nodes).all() or np.any(nodes < 0) or np.any(nodes != np.rint(nodes)):
        raise ValueError("nodes must be finite, nonnegative integer voxel coordinates")
    if raw.size and (raw.ndim != 2 or raw.shape[1] != 2):
        raise ValueError("edges must have shape (M,2)")
    if not np.isfinite(raw).all() or np.any(raw != np.rint(raw)):
        raise ValueError("edge indices must be finite integers")
    edges = raw.astype(np.int64).reshape(-1, 2)
    n = len(nodes)
    if edges.size:
        if edges.min() < 0 or edges.max() >= n:
            raise ValueError("edge index outside node array")
        if len(np.unique(edges, axis=0)) != len(edges):
            raise ValueError("duplicate edges")
        if np.any(nodes[edges[:, 1], 0] - nodes[edges[:, 0], 0] != 1):
            raise ValueError("links must connect consecutive frames")
        if (np.bincount(edges[:, 1], minlength=n).max() > 1
                or np.bincount(edges[:, 0], minlength=n).max() > (2 if branching else 1)):
            raise ValueError("input must be one-to-one" if not branching else "invalid lineage degree")
    return nodes.astype(np.int64), edges


def tracklets(nodes, edges):
    """Works with unsorted node rows, including appended gap interpolation nodes."""
    n = len(nodes)
    prev, nxt = np.full(n, -1, int), np.full(n, -1, int)
    if len(edges):
        nxt[edges[:, 0]] = edges[:, 1]
        prev[edges[:, 1]] = edges[:, 0]
    groups = []
    component, age, remaining = np.full(n, -1, int), np.zeros(n, int), np.zeros(n, int)
    for start in np.flatnonzero(prev < 0):
        path, i = [], int(start)
        while i >= 0:
            path.append(i)
            i = int(nxt[i])
        component[path] = len(groups)
        age[path] = np.arange(1, len(path) + 1)
        remaining[path] = np.arange(len(path), 0, -1)
        groups.append(np.array(path, dtype=int))
    return groups, component, age, remaining, nxt


def propose_events(nodes, edges, division_score, cfg):
    groups, component, age, remaining, nxt = tracklets(nodes, edges)
    n = len(nodes)
    if not n:
        return groups, component, []
    xyz, ts = nodes[:, 1:] * SCALE, nodes[:, 0]
    drift = np.zeros((int(ts.max()) + 1, 3))
    by_frame = defaultdict(list)
    for a, b in edges:
        by_frame[int(ts[a])].append(xyz[b] - xyz[a])
    for t, steps in by_frame.items():
        if len(steps) >= 5:
            drift[t] = np.median(steps, axis=0)
    cumulative = np.vstack([np.zeros(3), np.cumsum(drift, axis=0)])
    starts = defaultdict(list)
    for group in groups:
        starts[int(ts[group[0]])].append(int(group[0]))
    trees = {t: cKDTree(xyz[ids]) for t, ids in starts.items()}
    events = []
    if cfg.mode == "joint":
        for group in groups:
            end = int(group[-1])
            t = int(ts[end])
            for gap in range(1, cfg.max_gap + 1):
                target_t = t + gap + 1
                if target_t not in trees:
                    continue
                gate = cfg.gap_radius * np.sqrt(gap + 1)
                expected = xyz[end] + cumulative[target_t] - cumulative[t]
                for j in sorted(trees[target_t].query_ball_point(expected, gate)):
                    start = starts[target_t][j]
                    residual = np.linalg.norm(xyz[start] - expected) / gate
                    # Combining two tracklets refunds one component activation
                    # cost. Synthesized nodes incur a cost, not detection evidence.
                    reward = (cfg.track_cost - cfg.synthetic_node_cost * gap
                              - cfg.gap_distance_cost * residual**2)
                    events.append(Event(end, start, "gap", float(reward)))
    threshold_logit = np.log(cfg.division_threshold / (1 - cfg.division_threshold))
    for p in np.flatnonzero((division_score >= cfg.division_threshold)
                            & (nxt >= 0) & (age >= cfg.parent_length)):
        c1 = int(nxt[p])
        target_t = int(ts[p]) + 1
        if remaining[c1] < cfg.daughter_length or target_t not in trees:
            continue
        expected = xyz[p] + drift[int(ts[p])]
        v1 = xyz[c1] - expected
        n1 = np.linalg.norm(v1)
        if n1 < max(cfg.division_residual, 1e-9):
            continue
        for j in sorted(trees[target_t].query_ball_point(expected, cfg.division_radius)):
            c2 = starts[target_t][j]
            if remaining[c2] < cfg.daughter_length or component[c2] == component[p]:
                continue
            v2 = xyz[c2] - expected
            n2 = np.linalg.norm(v2)
            if n2 < 1e-9:
                continue
            cosine = float(v1 @ v2 / (n1 * n2))
            if cosine > cfg.division_cos:
                continue
            q = np.clip(division_score[p], 1e-6, 1 - 1e-6)
            reward = cfg.division_weight * (np.log(q / (1 - q)) - threshold_logit)
            reward -= cfg.division_geometry_cost * (n2 / cfg.division_radius)**2
            events.append(Event(int(p), c2, "division", float(reward)))
    return groups, component, events


def detector_node_rewards(confidence, cfg):
    """Bounded heatmap evidence; not a calibrated probability or GT-derived fit.

    At weight zero each observed node contributes exactly one, as before.
    By default only below-pivot confidence is discounted. Enabling bonus also
    rewards above-pivot confidence, allowing strong short tracks to survive.
    """
    confidence = np.asarray(confidence, float)
    if confidence.ndim != 1 or not np.isfinite(confidence).all() or np.any(confidence < 0):
        raise ValueError("detector confidence must be a finite nonnegative vector")
    evidence = np.clip(np.log2(np.maximum(confidence, 1e-12) / cfg.confidence_pivot),
                       -1., 1. if cfg.confidence_bonus else 0.)
    return 1. + cfg.confidence_weight * evidence


def select_events(groups, component, events, cfg, node_rewards=None):
    """Exact sparse set packing with tracklet activation and shared capacities.

    x_e <= y_source, x_e <= y_target; sum incoming events <= y_target;
    sum outgoing events at each source node <= y_source. Divisions and gaps
    therefore compete for the SAME daughter, even across different frames.
    """
    k, m = len(groups), len(events)
    if not k:
        return np.zeros(0, bool), np.zeros(m, bool), dict(status="empty", optimal=True)
    rewards = np.array([(len(g) if node_rewards is None else float(node_rewards[g].sum()))
                        - cfg.track_cost for g in groups]
                       + [e.reward for e in events], float)
    lower = np.zeros(k + m)
    upper = np.ones(k + m)
    if cfg.mode == "division":
        # Preserve the caller's filtered graph exactly in division-only mode.
        lower[:k] = 1
    rows, cols, values, ub = [], [], [], []

    def constraint(coeff):
        r = len(ub)
        for c, v in coeff:
            rows.append(r); cols.append(c); values.append(v)
        ub.append(0.)

    incoming, outgoing = defaultdict(list), defaultdict(list)
    for j, e in enumerate(events):
        a, b = int(component[e.source]), int(component[e.target])
        constraint([(k + j, 1.), (a, -1.)])
        constraint([(k + j, 1.), (b, -1.)])
        incoming[b].append(k + j)
        outgoing[e.source].append(k + j)
    for b, js in incoming.items():
        constraint([(j, 1.) for j in js] + [(b, -1.)])
    for a, js in outgoing.items():
        constraint([(j, 1.) for j in js] + [(int(component[a]), -1.)])
    matrix = coo_matrix((values, (rows, cols)), shape=(len(ub), k + m)).tocsc()
    result = milp(-rewards, integrality=np.ones(k + m), bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix, -np.inf, np.array(ub)),
                  options={"time_limit": cfg.time_limit, "mip_rel_gap": cfg.relative_gap})
    if result.x is None:
        raise RuntimeError(f"No feasible optimizer incumbent: {result.message}")
    x = np.rint(result.x)
    if (np.max(np.abs(result.x - x)) > 1e-5 or np.any(x < lower - 1e-6)
            or np.any(x > upper + 1e-6) or np.any(matrix @ x > 1e-6)):
        raise RuntimeError("Solver returned an invalid incumbent")
    return x[:k].astype(bool), x[k:].astype(bool), dict(
        status=int(result.status), optimal=bool(result.status == 0 and
                                               getattr(result, "mip_gap", 0.) <= 1e-9),
        objective=float(rewards @ x), mip_gap=float(getattr(result, "mip_gap", 0.)),
        message=result.message)


def optimize_lineage(nodes, edges, division_score, config=None, *, detector_confidence=None):
    """Return compact integer nodes, consecutive-frame edges, and diagnostics.

    division_score is the div3f `score` vector in the ORIGINAL cached node row
    order. When gap nodes already exist, the caller must explicitly pad scores
    (zero means no learned evidence for division at an interpolated point).
    """
    cfg = config or OptimizerConfig()
    nodes, edges = validate_graph(nodes, edges)
    score = np.asarray(division_score, float)
    if score.shape != (len(nodes),) or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError("division scores must be aligned finite (N,) probabilities")
    groups, component, events = propose_events(nodes, edges, score, cfg)
    node_rewards = None
    if detector_confidence is not None:
        if np.asarray(detector_confidence).shape != (len(nodes),):
            raise ValueError("detector confidence must align with input nodes")
        node_rewards = detector_node_rewards(detector_confidence, cfg)
    elif cfg.confidence_weight:
        raise ValueError("nonzero confidence_weight requires detector_confidence")
    keep, chosen, info = select_events(groups, component, events, cfg, node_rewards)
    mask = keep[component] if len(nodes) else np.zeros(0, bool)
    original_indices = np.flatnonzero(mask)
    remap = np.full(len(nodes), -1, int)
    remap[mask] = np.arange(mask.sum())
    new_nodes = list(nodes[mask])
    new_edges = [(int(remap[a]), int(remap[b])) for a, b in edges if mask[a] and mask[b]]
    selected = [e for e, yes in zip(events, chosen) if yes]
    for event in selected:
        a, b = event.source, event.target
        prev = int(remap[a])
        dt = int(nodes[b, 0] - nodes[a, 0])
        for step in range(1, dt):
            pos = np.rint(nodes[a, 1:] + (nodes[b, 1:] - nodes[a, 1:]) * step / dt).astype(int)
            cur = len(new_nodes)
            new_nodes.append(np.array([nodes[a, 0] + step, *pos]))
            new_edges.append((prev, cur))
            prev = cur
        new_edges.append((prev, int(remap[b])))
    out_nodes = np.asarray(new_nodes, np.int64).reshape(-1, 4)
    out_edges = np.asarray(new_edges, np.int64).reshape(-1, 2)
    validate_graph(out_nodes, out_edges, branching=True)
    info.update(config=asdict(cfg), tracklets=len(groups), candidate_events=len(events),
                selected_gaps=sum(e.kind == "gap" for e in selected),
                selected_divisions=sum(e.kind == "division" for e in selected),
                retained_original_nodes=int(mask.sum()), synthetic_nodes=len(out_nodes) - int(mask.sum()),
                original_indices=original_indices)
    return out_nodes, out_edges, info
