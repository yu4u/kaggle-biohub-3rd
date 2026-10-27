"""Stage ⑩: official metric on the final OOF graphs of all movies.

Per movie the organisers' implementation (third_party/tracking_cellmot, royerlab
kaggle-cell-tracking-competition commit 016845f) returns edge / division TP, FP, FN and the node
count. The pooled score follows the competition definition: the adjusted edge Jaccard is the
TP+FP+FN-weighted mean of the per-movie `max(0, J (1 - 0.1 (n_pred - n_total) / n_total))`
and the division Jaccard is micro-averaged; score = adj_edge + 0.1 * div_jaccard.
The clamped variant (node factor capped at 1) is reported
as a diagnostic.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from biohub046 import config as C

THIRD_PARTY = C.SOLUTION / 'third_party'
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

SCALE = (1.625, 0.40625, 0.40625)


def pred_graph(nodes, edges):
    """(N,4) t,z,y,x int + (M,2) index pairs -> tracksdata graph."""
    import polars as pl
    import tracksdata as td
    g = td.graph.RustWorkXGraph()
    for key in ('z', 'y', 'x'):
        g.add_node_attr_key(key, pl.Int64, 0)
    ids = [g.add_node({'t': int(t), 'z': int(z), 'y': int(y), 'x': int(x)}) for t, z, y, x in np.asarray(nodes)]
    for i, j in np.asarray(edges).reshape(-1, 2):
        g.add_edge(ids[int(i)], ids[int(j)], {})
    return g


def counts(sid, graph_path):
    from tracking_cellmot.metrics import evaluate
    from biohub046.lineage.lineage_evaluation import load_gt_sync
    ids, ts, xyz, gt_edges, n_total = load_gt_sync(C.TRAIN_DIR / f'{sid}.geff')
    index = {int(g): i for i, g in enumerate(ids)}
    truth = pred_graph(np.column_stack([ts, xyz]), [(index[int(a)], index[int(b)]) for a, b in gt_edges])
    with np.load(graph_path, allow_pickle=False) as z:
        nodes, edges = z['nodes'], z['edges']
    r = evaluate(pred_graph(nodes, edges), truth, scale=SCALE, max_distance=7.)
    return dict(sid=sid, tp=int(r.edge_tp), fp=int(r.edge_fp), fn=int(r.edge_fn), div_tp=int(r.division_tp),
                div_fp=int(r.division_fp), div_fn=int(r.division_fn), n_pred=int(r.num_pred_nodes), n_total=n_total)


def node_factor(nodes, total, clamp=False):
    factor = 1.1 - .1 * np.asarray(nodes, float) / np.asarray(total, float)
    return np.minimum(factor, 1.) if clamp else factor


def pooled(rows, clamp=False):
    num = den = 0.
    for r in rows:
        weight = r['tp'] + r['fp'] + r['fn']
        num += max(0., r['tp'] * float(node_factor(r['n_pred'], r['n_total'], clamp)))
        den += weight
    div_den = sum(r['div_tp'] + r['div_fn'] + r['div_fp'] for r in rows)
    adj_edge = num / den if den else 0.
    div_j = sum(r['div_tp'] for r in rows) / div_den if div_den else 0.
    return dict(score=adj_edge + .1 * div_j, adj_edge=adj_edge, div_j=div_j, samples=len(rows),
                **{k: int(sum(r[k] for r in rows)) for k in ('tp', 'fp', 'fn', 'div_tp', 'div_fn', 'div_fp', 'n_pred')})


def run(ids=None, jobs=4, graphs=None):
    from joblib import Parallel, delayed
    graphs = Path(graphs or C.CACHE_DIR / 'final_graphs')
    ids = sorted(ids or C.discover_ids())
    folds = C.fold_assignments()
    started = time.monotonic()
    rows = {}
    for i, row in enumerate(Parallel(n_jobs=jobs, return_as='generator_unordered')(
            delayed(counts)(s, graphs / f'{s}.npz') for s in ids), 1):
        rows[row['sid']] = row
        if i % 25 == 0 or i == len(ids):
            print(f'[score] {i}/{len(ids)}', flush=True)
    cohorts = {'all': ids}
    for embryo in sorted({s.split('_')[0] for s in ids}):
        cohorts[embryo] = [s for s in ids if s.startswith(embryo + '_')]
    for f in range(C.FOLD_NUM):
        cohorts[f'fold{f}'] = [s for s in ids if folds[s] == f]
    result = {name: pooled([rows[s] for s in members]) for name, members in cohorts.items() if members}
    result['all']['score_clamped'] = pooled([rows[s] for s in ids], clamp=True)['score']
    report = dict(reference_046v4_all199=C.REFERENCE_SCORE, results=result, per_sample=rows,
                  seconds=time.monotonic() - started)
    (C.CACHE_DIR / 'score.json').write_text(json.dumps(report, indent=2) + '\n')
    print()
    print('=' * 72)
    print(f'  sub 046 v4 reproduction — official metric, {len(ids)} train movies, out-of-fold')
    print('=' * 72)
    for name, r in result.items():
        print(f'  {name:>8}: score {r["score"]:.6f}  (adj_edge {r["adj_edge"]:.6f}, div_j {r["div_j"]:.4f}, '
              f'edge TP/FP/FN {r["tp"]}/{r["fp"]}/{r["fn"]}, div TP/FP/FN {r["div_tp"]}/{r["div_fp"]}/{r["div_fn"]}, '
              f'n={r["samples"]})')
    print(f'  clamped node factor (diagnostic): {result["all"]["score_clamped"]:.6f}')
    print(f'  reference (046 v4, all 199 OOF):  {C.REFERENCE_SCORE:.6f}')
    print('=' * 72, flush=True)
    return report
