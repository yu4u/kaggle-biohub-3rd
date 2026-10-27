"""Align selected calibrated edges to final graphs, then apply the post_611 correction."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from biohub046.lineage.gap_close import apply_keep, comp_sizes, gap_close
from biohub046.core.coordinate_refinement import ordinary_edges_and_anchors, refine_coordinates
from biohub046.core.lineage_optimizer import validate_graph


def read_npz(path):
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refine_final(prediction, final, shape, destination, *, gap_affine=False):
    """Use saved decisions, including the source pool, to recover exact q alignment.

    Replaying gap/L6 recovers node provenance without guessing from coordinates:
    a synthetic node can share coordinates with a discarded real detection.
    The replay must reproduce the saved graph before refinement can proceed.
    """
    started = time.perf_counter()
    prediction, final, destination = map(Path, (prediction, final, destination))
    joint = final.resolve().parent.parent
    sid = final.stem
    iteration = json.loads((joint / 'optimization.json').read_text())['chosen_round']
    decision_path = joint / f'round_{iteration}' / f'{sid}.npz'
    v = read_npz(prediction)
    n, o = len(v['nodes']), len(v['src'])
    node_offset = o + len(v['parents'])
    d = read_npz(decision_path)['decision'].astype(bool)
    if len(d) != node_offset + n:
        raise ValueError(f'{sid}: decision length does not match calibrated events')
    selected_nodes = d[node_offset:]
    raw = read_npz(joint / 'raw' / f'{sid}.npz')
    np.testing.assert_array_equal(raw['nodes'], v['nodes'][selected_nodes])
    original_edges = list(zip(v['src'][d[:o]], v['dst'][d[:o]]))
    for parent, daughters in zip(v['parents'][d[o:node_offset]], v['daughters'][d[o:node_offset]]):
        original_edges.extend((parent, child) for child in daughters)
    remap = np.full(n, -1, int)
    remap[selected_nodes] = np.arange(selected_nodes.sum())
    np.testing.assert_array_equal(raw['edges'], remap[np.asarray(original_edges, int).reshape(-1, 2)])
    ng, eg, _, _ = gap_close(raw['nodes'], raw['edges'], lmax=3, g1=3.)
    keep = comp_sizes(len(ng), eg) >= 6
    nodes, edges = apply_keep(ng, eg, keep)
    edges = np.asarray(edges, int).reshape(-1, 2)
    saved = read_npz(final)
    np.testing.assert_array_equal(nodes, saved['nodes'])
    np.testing.assert_array_equal(edges, saved['edges'])
    real = np.flatnonzero(selected_nodes)[keep[:len(raw['nodes'])]]
    nr = len(real)
    mapping = np.full(n, -1, int)
    mapping[real] = np.arange(nr)
    np.testing.assert_array_equal(nodes[:nr], v['nodes'][real])
    selected = np.flatnonzero(d[:o] & (mapping[v['src']] >= 0) & (mapping[v['dst']] >= 0))
    pairs = np.column_stack([mapping[v['src'][selected]], mapping[v['dst'][selected]]])
    ordinary, fixed = ordinary_edges_and_anchors(edges, len(nodes), nr)
    np.testing.assert_array_equal(ordinary, pairs)
    q = v['conditional'][selected].astype(float)
    prepared = time.perf_counter()
    refined = refine_coordinates(nodes, edges, nr, q, np.asarray(shape))
    corrected = time.perf_counter()
    np.testing.assert_array_equal(refined[:, 0], nodes[:, 0])
    np.testing.assert_array_equal(refined[nr:], nodes[nr:])
    np.testing.assert_array_equal(refined[:nr][fixed], nodes[:nr][fixed])
    gap_stats = {}
    if gap_affine:
        from biohub046.core.gap_coordinate_refinement import affine_fields, refine_gaps
        gap_started = time.perf_counter()
        fields = affine_fields(nodes, ordinary, q)
        gap_refined, gap_stats = refine_gaps(refined, edges, nr, fields, np.asarray(shape))
        np.testing.assert_array_equal(gap_refined[:nr], refined[:nr])
        np.testing.assert_array_equal(gap_refined[:, 0], refined[:, 0])
        gap_stats['moved'] = int(np.any(gap_refined != refined, axis=1).sum())
        gap_stats['seconds'] = time.perf_counter() - gap_started
        refined = gap_refined
    validate_graph(refined, edges, branching=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, nodes=refined, edges=edges)
    return dict(sid=sid, nodes=len(nodes), edges=len(edges), real_nodes=nr,
                ordinary_edges=len(q), fixed_real=int(fixed.sum()),
                moved=int(np.any(refined != nodes, axis=1).sum()),
                preparation_seconds=prepared-started, correction_seconds=corrected-prepared,
                total_seconds=time.perf_counter()-started,
                prediction_sha256=digest(prediction), decision_sha256=digest(decision_path),
                baseline_sha256=digest(final), output_sha256=digest(destination),
                gap_affine=gap_affine, gap=gap_stats)
