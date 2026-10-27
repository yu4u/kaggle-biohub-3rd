"""Stage ⑦: the calibrators of the 046 v4 stack, fit per fold (OOF evaluation) and on all movies (deployment).

Per movie the rows are built from the OOF caches exactly as inference consumes them
(lineage.div797_lineage.features), labelled against the GT graph with the official 7 um matching
(lineage.train_metric_calibration.labels):

* ordinary links (model.json): the 26 phase-edge features of the radius-12 candidates and the 19-feature
  parent-415 division candidates (run_division_phase_wiring.py: exposed rows + a deterministic 1 % of the
  unexposed with weight 100; logistic conditional / exposure heads and the count priors);
* native divisions (div797 native): radius-12 parents with native762 score >= .1; 19-feature logistic
  conditional (exposed rows, count-scaled L2) and exposure, then a 100-round LightGBM residual on the
  34 features (init_score = logistic logit) and the 846 XGBoost on the same rows (base_margin = logit);
* wide divisions (div797 wide): the radius-16 extra pairs plus the parent-415 candidates re-featured on the
  radius-16 graph (exp_758); logistic edge heads on the radius-16 candidates, logistic division heads,
  LightGBM and XGBoost residuals as above.

Deviation from the original (user decision 2026-10-01): the rows are the inference-time candidate sets;
the original 797 rows came from the experiment chain and carried 9 legacy seed exceptions.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from biohub046 import config as C

ROWS_DIR = C.CACHE_DIR / 'calib' / 'rows'
ORDINARY_DIR = C.CACHE_DIR / 'calib' / 'ordinary'
BUNDLES = C.ARTIFACTS_DIR / 'calibrators'
BUNDLE_FILES = ('model.json', 'div797.json', 'native.txt', 'wide.txt', 'div846.json', 'native.ubj', 'wide.ubj')

# exp_687.PARAMS (+ num_threads 8 at fit time) and exp_823.PARAMS
LGB_PARAMS = dict(objective='binary', num_leaves=4, max_depth=2, learning_rate=.05, min_data_in_leaf=100,
                  lambda_l2=1., verbosity=-1, num_threads=8, seed=687, deterministic=True, force_col_wise=True)
XGB_PARAMS = dict(objective='binary:logistic', eval_metric='logloss', booster='gbtree', tree_method='hist',
                  device='cpu', grow_policy='depthwise', max_depth=2, max_leaves=4, eta=.05, min_child_weight=1.,
                  reg_lambda=1., reg_alpha=0., gamma=0., max_delta_step=0., subsample=1., colsample_bytree=1.,
                  scale_pos_weight=1., max_bin=255, nthread=8, seed=687)
ROUNDS = 100


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundle_dir(fold):
    return BUNDLES / ('full' if fold < 0 else f'fold{fold}')


def bundle_complete(fold):
    return all((bundle_dir(fold) / name).exists() for name in BUNDLE_FILES)


def prepare_rows(sid):
    """Training rows of one movie -> cache/calib/rows/<sid>.npz (+ the ordinary rows for fit_fold)."""
    from biohub046.core.division_phase_wiring import phase_division_features, phase_edge_features
    from biohub046.core.metric_calibration import candidate_features
    from biohub046.lineage import div688_lineage as legacy
    from biohub046.lineage.div797_features import motion_features
    from biohub046.lineage.div797_lineage import features
    from biohub046.lineage.lineage_evaluation import load_gt_sync
    from biohub046.lineage.train_metric_calibration import labels
    from biohub046.infer.oof_infer import OOF_DIR
    out = ROWS_DIR / f'{sid}.npz'
    ordinary_out = ORDINARY_DIR / 'training' / f'{sid}.npz'
    stats_out = ORDINARY_DIR / 'stats' / f'{sid}.json'
    if out.exists() and ordinary_out.exists() and stats_out.exists():
        return sid, json.loads(stats_out.read_text())
    g = read(OOF_DIR / 'graphs' / f'{sid}.npz')
    conf = read(OOF_DIR / 'confidence' / f'{sid}.npz')
    phase = read(OOF_DIR / 'div_scores' / f'{sid}.npz')
    w = read(OOF_DIR / 'division_graphs' / f'{sid}.npz')
    for other in (conf, phase, w):
        np.testing.assert_array_equal(g['nodes'], other['nodes'])
    conf = conf['confidence']
    gt = load_gt_sync(C.TRAIN_DIR / f'{sid}.geff')
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(sid.encode()).digest()[:8], 'little'))
    # ---- ordinary links + parent-415 division candidates (run_division_phase_wiring.prepare) ----
    src, dst, slot, x, im, om = candidate_features(g, conf, phase['score'], radius_um=12.)
    par, dau, pair, dx = phase_division_features(g, conf, phase['score'], phase['pre'], phase['post'], src, dst, slot, im, om)
    x = phase_edge_features(x, src, dst, phase['score'], phase['pre'], phase['post'])
    tp, exposed, dtp, dexposed, stats = labels(g['nodes'], gt, src, dst, par, dau)
    keep = exposed | (rng.random(len(src)) < .01)
    ordinary_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ordinary_out, x=x[keep], tp=tp[keep], exposed=exposed[keep],
                        weight=np.where(exposed[keep], 1., 100.).astype(np.float32), dx=dx, dtp=dtp, dexposed=dexposed)
    stats.update(div_candidates=len(par), div_exposed=int(dexposed.sum()), div_positive=int(dtp.sum()))
    # ---- 797 candidate sets as inference builds them ----
    f = features(g, w, conf, phase)
    p = f['p']
    _, _, ny, nknown, _ = labels(g['nodes'], gt, p['src'], p['dst'], p['parents'], p['daughters'])
    # ---- wide rows (exp_758): the parent-415 floor-.1 candidates (688 / legacy.extract, 19 raw features)
    #      re-featured on the radius-16 graph + the extra pairs ----
    p415, _, raw415, _, _ = legacy.extract(g, conf, phase)
    ws, wd, wslot = f['ws'], f['wd'], f['wslot']
    lookup = np.full(w['cand_idx'].shape, -1, int)
    lookup[ws, wslot] = np.arange(len(ws))
    old_edges = lookup[p415['src'], p415['slot']]
    assert np.all(old_edges >= 0), 'radius-12 candidates must be radius-16 candidates'
    old_dx = raw415.copy()
    op = old_edges[p415['arc_pairs']]
    q = w['cand_prob'][ws, wslot][op]
    rev = w['cand_rev'][ws, wslot][op]
    sym = w['cand_sym'][ws, wslot][op]
    old_dx[:, 6] = q.min(1); old_dx[:, 7] = q.max(1); old_dx[:, 8] = rev.min(1); old_dx[:, 9] = sym.min(1)
    n_old = len(p415['parents'])
    all_par = np.r_[p415['parents'], f['par']]
    all_dau = np.concatenate([p415['daughters'], f['dau']])
    all_dx = np.concatenate([old_dx, f['extra_dx']]).astype(np.float32)
    wtp, wexposed, wdtp, wde, _ = labels(g['nodes'], gt, ws, wd, all_par, all_dau)
    keep_w = wexposed | (rng.random(len(ws)) < .01)
    dk = wde | (rng.random(len(all_par)) < .01)
    aux, rich = legacy.incoming_features(phase['pre'], p['src'], p['dst'], f['rawq'])
    motion = motion_features(g['nodes'], p['src'], p['dst'], f['rawq'], all_par, all_dau)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
                        native_x=f['dx'].astype(np.float32), native_tree=f['tx'].astype(np.float32),
                        native_y=ny, native_known=nknown,
                        wide_x=f['wx'][keep_w], wide_tp=wtp[keep_w], wide_exposed=wexposed[keep_w],
                        wide_weight=np.where(wexposed[keep_w], 1., 100.).astype(np.float32),
                        wide_dx=all_dx[dk], wide_dtp=wdtp[dk], wide_dexposed=wde[dk],
                        wide_dweight=np.where(wde[dk], 1., 100.).astype(np.float32),
                        wide_aux=aux[all_par[dk]].astype(np.float32), wide_rich=rich[all_par[dk]].astype(np.float32),
                        wide_motion=motion[dk].astype(np.float32), wide_added=np.flatnonzero(dk) >= n_old)
    stats.update(native_candidates=int(len(ny)), native_known=int(nknown.sum()), native_positive=int(ny.sum()),
                 wide_candidates=int(len(all_par)), wide_known=int(wde.sum()), wide_positive=int(wdtp.sum()))
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(stats) + '\n')
    return sid, stats


def _concat(blocks, key):
    return np.concatenate([b[key] for b in blocks])


def fit_bundle(fold, sids, folds, stats):
    """Write artifacts/calibrators/{fold<f>|full}/: model.json, div797.json + LightGBM, div846.json + XGBoost."""
    import lightgbm as lgb
    import xgboost as xgb
    from biohub046.core.division_phase_wiring import PHASE_DIV_FEATURES, PHASE_EDGE_FEATURES
    from biohub046.core import metric_calibration as mc
    from biohub046.core.metric_calibration import bounded_logit, predict_logistic
    fit_logistic = mc.fit_logistic
    from biohub046.lineage.div797_lineage import FEATURES
    from biohub046.lineage.train_metric_calibration import fit_fold
    out = bundle_dir(fold)
    out.mkdir(parents=True, exist_ok=True)
    train = [s for s in sids if fold < 0 or folds[s] != fold]
    started = time.monotonic()
    # ---- ordinary links: model.json (the 027/031 phase26/19 calibrator) ----
    args = SimpleNamespace(out=ORDINARY_DIR, clip_density=None, target_block=None, density_priors=None)
    _, ordinary = fit_fold(fold, sids, folds, args, stats, save=False)
    ordinary.update(edge_features=PHASE_EDGE_FEATURES, division_features=PHASE_DIV_FEATURES,
                    division_phase_expand=False, candidate_radius_um=12.)
    (out / 'model.json').write_text(json.dumps(ordinary, indent=2) + '\n')
    blocks = [read(ROWS_DIR / f'{s}.npz') for s in train]
    meta = dict(fold=fold, training_ids=train, features=FEATURES, parent_floor=.1, rounds=ROUNDS,
                params=LGB_PARAMS, lightgbm_version=lgb.__version__)
    xgb_meta = dict(fold=fold, training_ids=train, features=FEATURES, xgboost_version=xgb.__version__,
                    params=XGB_PARAMS, rounds=ROUNDS, ensemble='probability50:50 afterfloat32',
                    seed='LightGBM native seed (797)', models={})
    # ---- native (762) divisions ----
    x = _concat(blocks, 'native_x'); y = _concat(blocks, 'native_y'); known = _concat(blocks, 'native_known').astype(bool)
    tree = _concat(blocks, 'native_tree')
    old_exposed = sum(stats[s]['div_exposed'] for s in train)
    old_candidates = sum(stats[s]['div_candidates'] for s in train)
    native = dict(division_conditional=fit_logistic(x[known], y[known], l2=.001 * old_exposed / max(int(known.sum()), 1)),
                  division_exposure=fit_logistic(x, known, l2=.001 * old_candidates / max(len(x), 1)))
    rows = {'native': (tree[known], y[known], native)}
    # ---- wide (radius-16 extra) divisions ----
    wx = _concat(blocks, 'wide_x'); wtp = _concat(blocks, 'wide_tp'); wexp = _concat(blocks, 'wide_exposed').astype(bool)
    wweight = _concat(blocks, 'wide_weight')
    dx = _concat(blocks, 'wide_dx'); dtp = _concat(blocks, 'wide_dtp'); dk = _concat(blocks, 'wide_dexposed').astype(bool)
    dweight = _concat(blocks, 'wide_dweight')
    lin = dx.copy(); lin[:, [14, 16]] = 0
    wide = dict(edge_conditional=fit_logistic(wx[wexp], wtp[wexp]), edge_exposure=fit_logistic(wx, wexp, wweight),
                division_conditional=fit_logistic(lin[dk], dtp[dk], l2=1e-3),
                division_exposure=fit_logistic(dx, dk, dweight, l2=1e-3))
    wtree = np.column_stack([lin[dk], _concat(blocks, 'wide_aux')[dk], _concat(blocks, 'wide_rich')[dk],
                             _concat(blocks, 'wide_motion')[dk]]).astype(np.float32)
    rows['wide'] = (wtree, dtp[dk], wide)
    report = {}
    for kind, (t, label, model) in rows.items():
        assert t.shape[1] == len(FEATURES) == 34, t.shape
        init = bounded_logit(predict_logistic(model['division_conditional'], t[:, :19]))
        booster = lgb.train(LGB_PARAMS, lgb.Dataset(t, label=label, init_score=init), num_boost_round=ROUNDS)
        booster.save_model(str(out / f'{kind}.txt'))
        xtree = xgb.train(XGB_PARAMS, xgb.DMatrix(t, label=label, base_margin=init, nthread=8), num_boost_round=ROUNDS)
        xtree.save_model(out / f'{kind}.ubj')
        meta[kind] = model
        xgb_meta['models'][kind] = dict(sha256=digest(out / f'{kind}.ubj'), rows=int(len(label)), positive=int(label.sum()))
        report[kind] = dict(rows=int(len(label)), positive=int(label.sum()))
    meta['files'] = {name: digest(out / name) for name in ('model.json', 'native.txt', 'wide.txt')}
    (out / 'div797.json').write_text(json.dumps(meta, indent=2) + '\n')
    xgb_meta['source_div797_sha256'] = digest(out / 'div797.json')
    (out / 'div846.json').write_text(json.dumps(xgb_meta, indent=2) + '\n')
    from biohub046.lineage.div846_lineage import check_bundle
    check_bundle(out)
    print(f'[calibrate] {out.name}: {len(train)} movies, {report}, {time.monotonic() - started:.0f}s', flush=True)


def _tolerant_fits():
    """--smoke only: untrained models leave some calibrator heads without rows; fit a constant instead of failing."""
    from biohub046.core import metric_calibration as mc
    from biohub046.lineage import train_metric_calibration as tmc
    original = mc.fit_logistic

    def fit_logistic(x, y, weights=None, l2=1e-4):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if len(y) and (weights is None or np.all(np.asarray(weights) > 0)) and 0 < y.sum() < len(y):
            return original(x, y, weights, l2)
        rate = float(np.clip(y.mean() if len(y) else .5, 1e-3, 1 - 1e-3))
        print(f'[calibrate] smoke: constant head ({len(y)} rows, {int(y.sum()) if len(y) else 0} positive)', flush=True)
        return dict(mean=[0.] * x.shape[1], scale=[1.] * x.shape[1], beta=[float(np.log(rate / (1 - rate)))] + [0.] * x.shape[1],
                    samples=int(len(y)), positive=int(y.sum()) if len(y) else 0, l2=l2, loss=0.)
    mc.fit_logistic = tmc.fit_logistic = fit_logistic
    globals()['fit_logistic_override'] = fit_logistic


def run(ids=None, jobs=4):
    from joblib import Parallel, delayed
    if C.SMOKE:
        _tolerant_fits()
    folds = C.fold_assignments()
    sids = sorted(ids or folds)
    manifest = BUNDLES / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['ids'] != sids:
        print(f'[calibrate] movie set changed: refitting every bundle', flush=True)
        for fold in [*range(C.FOLD_NUM), -1]:
            shutil.rmtree(bundle_dir(fold), ignore_errors=True)
    targets = [f for f in [*range(C.FOLD_NUM), -1] if not bundle_complete(f)]
    if not targets:
        print('[calibrate] skip (all bundles exist)', flush=True)
        return
    stats = {}
    for i, (sid, row) in enumerate(Parallel(n_jobs=jobs, return_as='generator')(delayed(prepare_rows)(s) for s in sids), 1):
        stats[sid] = row
        if i % 25 == 0 or i == len(sids):
            print(f'[calibrate] rows {i}/{len(sids)}', flush=True)
    (ORDINARY_DIR / 'manifest.json').write_text(json.dumps(dict(folds={s: folds[s] for s in sids}, stats=stats), indent=2) + '\n')
    for fold in targets:
        fit_bundle(fold, sids, folds, stats)
    BUNDLES.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(dict(ids=sids, folds={s: folds[s] for s in sids}), indent=2) + '\n')
