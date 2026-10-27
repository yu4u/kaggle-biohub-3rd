"""Control-flow tests of the pipeline (no data, no GPU, no real training): resuming an interrupted
training fold, the movie set given to the calibrated prediction / LP drivers, and smoke isolation."""
import json
import sys

import joblib
import pytest
import pytorch_lightning

from biohub046 import config as C
from biohub046 import pipeline
from biohub046.train import det, lightning


class FakeTrainer:
    """Stands in for pytorch_lightning.Trainer: records the resume checkpoint and writes `_last.ckpt`
    (optionally failing afterwards, as an interrupted run would)."""
    calls = []
    fail = False

    def __init__(self, callbacks=None, **kwargs):
        self.out_dir = callbacks[-1].dirpath
        self.filename = callbacks[-1].filename

    def fit(self, model, datamodule=None, ckpt_path=None, weights_only=None):
        FakeTrainer.calls.append(ckpt_path)
        progress = (int(open(ckpt_path).read()) if ckpt_path else 0) + 1
        with open(f'{self.out_dir}/{self.filename}.ckpt', 'w') as f:
            f.write(str(progress))
        if FakeTrainer.fail:
            raise KeyboardInterrupt


@pytest.fixture
def fake_training(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'ARTIFACTS_DIR', tmp_path / 'artifacts')
    monkeypatch.setattr(C, 'CACHE_DIR', tmp_path / 'cache')
    monkeypatch.setattr(pytorch_lightning, 'Trainer', FakeTrainer)
    monkeypatch.setattr('biohub046.core.pl_module.MyModel', lambda cfg, mode: None)
    monkeypatch.setattr('biohub046.core.datamodule.CellDataModule', lambda cfg: None)
    FakeTrainer.calls, FakeTrainer.fail = [], False
    return tmp_path


def test_interrupted_fold_resumes_and_publishes_only_when_complete(fake_training):
    target = det.ckpt_path('dl_284', 0)
    FakeTrainer.fail = True
    with pytest.raises(KeyboardInterrupt):
        det.train(['dl_284'], [0])
    assert not target.exists()                     # an in-progress checkpoint is never the deployed one
    FakeTrainer.fail = False
    det.train(['dl_284'], [0])
    resumed = FakeTrainer.calls[-1]
    assert FakeTrainer.calls[0] is None and resumed.endswith('dl_284_fold0_last.ckpt')
    assert target.read_text() == '2'               # trained on from the interrupted state
    det.train(['dl_284'], [0])
    assert len(FakeTrainer.calls) == 2             # complete: skipped


def test_distributed_checkpoint_without_training_state_is_skipped(fake_training):
    target = det.ckpt_path('dl_284', 0)
    target.parent.mkdir(parents=True)
    target.write_text('weights only')
    det.train(['dl_284'], [0])
    assert FakeTrainer.calls == []


class StopDriver(Exception):
    pass


def capture_parallel(seen):
    """joblib.Parallel stand-in that records the first argument of every delayed call, then stops."""
    def parallel(*args, **kwargs):
        def run(tasks):
            seen.extend(t[1][0] for t in tasks)
            raise StopDriver
        return run
    return parallel


@pytest.mark.parametrize('module, flag, folder', [
    ('biohub046.lineage.div846_lineage', 'predict', 'graphs'),
    ('biohub046.lineage.run_metric_event_pipeline', None, 'predictions'),
])
def test_drivers_process_only_the_given_movies(tmp_path, monkeypatch, module, flag, folder):
    import importlib
    main = importlib.import_module(module).main
    for sid in ('movie_a', 'movie_b'):             # movie_b: a stale cache outside the selection
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / f'{sid}.npz').touch()
    ids = tmp_path / 'ids.json'
    ids.write_text(json.dumps(['movie_a']))
    common = ['--graphs', tmp_path / 'graphs', '--div-scores', tmp_path, '--out', tmp_path / 'out']
    argv = (['predict', *common, '--confidence', tmp_path, '--model', tmp_path / 'model.json'] if flag
            else [*common, '--predictions', tmp_path / 'predictions'])
    for extra, expected in (([], ['movie_a', 'movie_b']), (['--ids', ids], ['movie_a'])):
        seen = []
        monkeypatch.setattr(joblib, 'Parallel', capture_parallel(seen))
        monkeypatch.setattr(sys, 'argv', ['driver', *map(str, argv + extra)])
        with pytest.raises(StopDriver):
            main()
        assert seen == expected


def test_optimize_passes_the_selected_movies_to_predict_and_lp(tmp_path, monkeypatch):
    from biohub046.lineage import optimize
    monkeypatch.setattr(C, 'CACHE_DIR', tmp_path / 'cache')
    monkeypatch.setattr(optimize, 'LINEAGE_DIR', tmp_path / 'lineage')
    monkeypatch.setattr(optimize, 'postprocess', lambda *a: None)
    calls = []

    def fake_run(args, log):
        calls.append([str(a) for a in args])
        out = optimize.LINEAGE_DIR / 'joint'
        out.mkdir(parents=True, exist_ok=True)
        (out / 'optimization.json').write_text(json.dumps(dict(chosen_round=0, history=[])))
    monkeypatch.setattr(optimize, '_run', fake_run)
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache' / 'folds.json').write_text('{}')
    optimize.run(['movie_b', 'movie_a'])
    assert [c[0] for c in calls] == ['predict', 'optimize']
    for c in calls:
        assert json.loads(open(c[c.index('--ids') + 1]).read()) == ['movie_a', 'movie_b']


@pytest.mark.parametrize('argv, env_smoke, isolated, reexec', [
    (['all', '--smoke'], False, False, True),      # CLI
    (['all'], True, False, True),                  # BIOHUB_SMOKE=1 (compose)
    (['all'], True, True, False),                  # the isolated child itself
    (['all'], False, False, False),                # a normal run
])
def test_every_smoke_request_runs_isolated(monkeypatch, tmp_path, argv, env_smoke, isolated, reexec):
    monkeypatch.setattr(C, 'SMOKE', env_smoke)
    monkeypatch.setattr(C, 'SMOKE_ISOLATED', isolated)
    monkeypatch.setattr(C, 'ARTIFACTS_DIR', tmp_path / 'artifacts')
    monkeypatch.setattr(C, 'CACHE_DIR', tmp_path / 'cache')
    events = []

    def fake_reexec(child_argv):
        events.append(('reexec', child_argv))
        raise SystemExit(0)
    monkeypatch.setattr(pipeline, 'reexec_smoke', fake_reexec)
    monkeypatch.setattr(pipeline, 'run_stage', lambda name, args: events.append(('stage', args.smoke)))
    try:
        pipeline.main(argv)
    except SystemExit:
        pass
    if reexec:
        assert events == [('reexec', ['all'])]
    else:
        assert events and all(e == ('stage', env_smoke) for e in events)


@pytest.mark.parametrize('existing_dir', [False, True])
def test_division_fold_publishes_into_empty_or_existing_artifacts(tmp_path, monkeypatch, existing_dir):
    from biohub046.train import div3f
    monkeypatch.setattr(C, 'ARTIFACTS_DIR', tmp_path / 'artifacts')
    monkeypatch.setattr(C, 'CACHE_DIR', tmp_path / 'cache')
    runs = []

    def fake_main(argv):
        out = next(a.split('=', 1)[1] for a in argv if a.startswith('train.output_dir='))
        runs.append(out)
        (tmp_path / out).mkdir(parents=True, exist_ok=True)
        (tmp_path / out / 'ep001.pt').write_text('weights')
        (tmp_path / out / 'history.jsonl').write_text(json.dumps(
            dict(epoch=1, val_auc=.9, val_median_neg=.01, val_median_weak=.01)) + '\n')
        if FakeTrainer.fail:
            raise KeyboardInterrupt
    monkeypatch.setattr('biohub046.div3f.train.main', fake_main)
    target = div3f.ckpt_path('div_415', 0)
    if existing_dir:
        target.parent.mkdir(parents=True)
    FakeTrainer.fail = True
    with pytest.raises(KeyboardInterrupt):
        div3f.train(['div_415'], [0])
    assert not target.exists()                     # no deployed model before training completes
    FakeTrainer.fail = False
    div3f.train(['div_415'], [0])
    assert target.read_text() == 'weights'
    assert json.loads(target.with_suffix('.json').read_text())['epoch'] == 1
    div3f.train(['div_415'], [0])
    assert len(runs) == 2                          # complete: skipped
