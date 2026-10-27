"""Light unit tests of the ported components (no data, no GPU)."""
import json

import numpy as np
import torch

from biohub046 import config as C
from biohub046.core.folds import fold_assignments, fold_split
from biohub046.core.metric import extract_peaks
from biohub046.core.gap_coordinate_refinement import bridge, refine_gaps, affine_fields
from biohub046.lineage.gap_close import apply_keep, comp_sizes, gap_close
from biohub046 import score


def test_fold_split_round_robin_within_embryo():
    ids = [f'44b6_{i:03d}' for i in range(71)] + [f'6bba_{i:03d}' for i in range(128)]
    folds = fold_assignments(ids, 5)
    sizes = [sum(1 for f in folds.values() if f == k) for k in range(5)]
    assert sizes == [41, 40, 40, 39, 39]
    train, val = fold_split(ids, 5, 0)
    assert val[0] == '44b6_000' and len(train) + len(val) == 199 and not set(train) & set(val)


def test_extract_peaks_exact_local_maxima():
    hm = torch.zeros(8, 16, 16)
    hm[3, 5, 5] = 0.9
    hm[3, 5, 6] = 0.5     # inside the NMS window of the first peak
    hm[6, 12, 12] = 0.3
    hm[1, 1, 1] = 0.1     # below threshold
    peaks = extract_peaks(hm, 0.2, [3, 7, 7]).numpy()
    assert sorted(map(tuple, peaks)) == [(3, 5, 5), (6, 12, 12)]


def test_gap_close_inserts_linear_nodes():
    # one track with frames 0,1 then 4,5 -> gap of 2 missing frames
    nodes = np.array([[0, 10, 100, 100], [1, 10, 102, 102], [4, 10, 108, 108], [5, 10, 110, 110]])
    edges = [(0, 1), (2, 3)]
    n2, e2, n_bridge, n_insert = gap_close(nodes, edges, lmax=3, g1=3.)
    assert (n_bridge, n_insert) == (1, 2)
    assert len(n2) == 6 and len(e2) == 5
    assert n2[4].tolist() == [2, 10, 104, 104] and n2[5].tolist() == [3, 10, 106, 106]
    assert comp_sizes(len(n2), e2).tolist() == [6] * 6
    keep = comp_sizes(len(n2), e2) >= 6
    kept_nodes, kept_edges = apply_keep(n2, e2, keep)
    assert len(kept_nodes) == 6 and len(kept_edges) == 5


def test_gap_bridge_geometry():
    # exp_838.verify_geometry: zero field -> linear interpolation; affine field -> exact propagation
    start, end = np.array([2., 3., 4.]), np.array([8., 6., 13.])
    constant = {t: (np.zeros((3, 3)), np.array([1., 2., 3.])) for t in range(4)}
    expected = start + np.arange(5)[:, None] / 4 * (end - start)
    np.testing.assert_allclose(bridge(start, end, 0, 4, constant), expected)
    fields = {t: (np.diag([.01, -.02, .03]), np.array([1., 2., 3.])) for t in range(4)}
    propagated = start.copy()
    for t in range(4):
        a, b = fields[t]
        propagated += propagated @ a + b
    trajectory = bridge(start, propagated, 0, 4, fields)
    for t in range(4):
        a, b = fields[t]
        np.testing.assert_allclose(trajectory[t + 1], trajectory[t] + trajectory[t] @ a + b)


def test_refine_gaps_moves_only_synthetic_nodes():
    # real nodes 0..3 (two tracks), synthetic nodes 4,5 bridging node 1 (t=1) to node 2 (t=4)
    nodes = np.array([[0, 10, 100, 100], [1, 10, 102, 102], [4, 10, 120, 120], [5, 10, 122, 122],
                      [2, 10, 104, 104], [3, 10, 106, 106]])
    edges = np.array([[0, 1], [1, 4], [4, 5], [5, 2], [2, 3]])
    pairs = np.array([[0, 1], [2, 3]])
    fields = affine_fields(nodes, pairs, np.ones(2))
    for t in (1, 2, 3):
        fields.setdefault(t, fields[0])
    refined, stats = refine_gaps(nodes, edges, 4, fields, np.array([64, 256, 256]))
    np.testing.assert_array_equal(refined[:4], nodes[:4])
    assert stats['gaps'] == 1 and stats['synthetic'] == 2


def test_pooled_score_formula():
    rows = [dict(tp=90, fp=5, fn=5, div_tp=1, div_fp=0, div_fn=1, n_pred=1000, n_total=1000.),
            dict(tp=40, fp=10, fn=0, div_tp=0, div_fp=1, div_fn=0, n_pred=1200, n_total=1000.)]
    r = score.pooled(rows)
    # edge: (90*1.0 + 40*(1.1-0.12)) / (100 + 50); division: 1 / 3
    assert abs(r['adj_edge'] - (90 + 40 * 0.98) / 150) < 1e-12
    assert abs(r['div_j'] - 1 / 3) < 1e-12
    assert abs(r['score'] - (r['adj_edge'] + 0.1 / 3)) < 1e-12
    assert score.pooled(rows, clamp=True)['adj_edge'] == (90 + 40 * 0.98) / 150


def test_config_model_paths_cover_every_checkpoint():
    paths = C.model_paths()
    assert len(paths) == 3 * 5 + 5 + 4 * 5 + 1
    assert all(p.suffix in ('.ckpt', '.pt') for p in paths.values())
    assert C.LP_ARGS[-2:] == ['--time-limit', '120'] and json.dumps(C.LP_ARGS)
