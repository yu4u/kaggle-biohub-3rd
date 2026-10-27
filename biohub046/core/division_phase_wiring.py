"""Use phase-head scores on actual candidate predecessors and daughters."""
import numpy as np

from biohub046.core.metric_calibration import division_features, DIV_FEATURES, EDGE_FEATURES

PHASE_DIV_FEATURES = DIV_FEATURES + ['parent_pre', 'parent_post', 'predecessor_pre_support',
                                   'daughter_post_min', 'daughter_post_max']
PHASE_EDGE_FEATURES = EDGE_FEATURES + ['source_pre', 'target_pre', 'source_post', 'target_post',
                                     'pre_to_parent', 'parent_to_post']


def phase_edge_features(x, src, dst, parent, pre, post):
    """State compatibility on directed candidate edges, using predictions only."""
    return np.column_stack([x, pre[src], pre[dst], post[src], post[dst],
                            pre[src] * parent[dst], parent[src] * post[dst]]).astype(np.float32)


def phase_division_features(graph, confidence, score, pre, post, src, dst, slot,
                            inmax, outmax):
    n = len(graph['nodes'])
    for values in [score, pre, post]:
        if values.shape != (n,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('invalid phase score alignment or range')
    # Incoming candidates u -> p, not nearby points or an already chosen edge.
    support = np.zeros(n)
    np.maximum.at(support, dst, pre[src] * graph['cand_prob'][src, slot])
    parents, children, pairs, features = division_features(
        graph, confidence, score, src, dst, slot, inmax, outmax)
    lo = post[children].min(axis=1)
    hi = post[children].max(axis=1)
    extra = np.column_stack([pre[parents], post[parents], support[parents], lo, hi])
    return parents, children, pairs, np.column_stack([features, extra]).astype(np.float32)
