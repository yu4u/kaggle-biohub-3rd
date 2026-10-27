"""Deployment of div797 without experimental caches or GT dependencies."""
import argparse
import json
import os
import sys
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
from scipy.special import expit
from biohub046.lineage import div688_lineage as legacy
from biohub046.lineage.div797_features import motion_features,extra_pairs
from biohub046.lineage.div797_solver import compact,combined_solve
from biohub046.core.metric_calibration import candidate_features,bounded_logit,predict_logistic
from biohub046.core.division_phase_wiring import phase_edge_features

pipeline=legacy.pipeline
read,digest=legacy.read,legacy.digest
FEATURES=legacy.FEATURES+['midpoint_velocity_error','daughter_velocity_error_min','daughter_velocity_error_max',
    'predicted_separation_change','incoming_velocity_spread','outgoing_velocity_spread_max']


def features(g,wide_graph,confidence,phase):
    """GT-free candidate sets and feature matrices shared by inference and calibrator training.

    native: the radius-12 candidates of parents with native762 score >= .1 (legacy.extract), 19-feature
    `dx` (pre-support columns zeroed) and the 34-feature tree matrix `tx`; wide: the radius-16 extra pairs
    (`extra_pairs`) with their 19-feature `extra_dx` and 34-feature `wt`; `ex` / `wx` are the 26-feature
    ordinary-edge matrices of the radius-12 / radius-16 candidates.
    """
    native=np.asarray(phase['native_parent']);np.testing.assert_equal(native.shape,phase['score'].shape)
    p,_,_,dx,tx=legacy.extract(g,confidence,dict(phase,score=native))
    rawq=g['cand_prob'][p['src'],p['slot']]
    tx=np.column_stack([tx,motion_features(p['nodes'],p['src'],p['dst'],rawq,p['parents'],p['daughters'])])
    _,_,_,ex,_,_=candidate_features(g,confidence,phase['score'],radius_um=12.)
    ex=phase_edge_features(ex,p['src'],p['dst'],phase['score'],phase['pre'],phase['post'])
    ws,wd,slot,wx,_,_=candidate_features(wide_graph,confidence,phase['score'],radius_um=16.)
    wx=phase_edge_features(wx,ws,wd,phase['score'],phase['pre'],phase['post'])
    par,dau,pairs,extra_dx,anchor=extra_pairs(wide_graph,p,phase,confidence,ws,wd,slot,g)
    aux,rich=legacy.incoming_features(phase['pre'],p['src'],p['dst'],rawq)
    extra_x=extra_dx.copy();extra_x[:,[14,16]]=0
    wt=np.column_stack([extra_x,aux[par],rich[par],motion_features(p['nodes'],p['src'],p['dst'],rawq,par,dau)])
    return dict(p=p,native=native,rawq=rawq,dx=dx,tx=tx,ex=ex,ws=ws,wd=wd,wslot=slot,wx=wx,
        par=par,dau=dau,pairs=pairs,extra_dx=extra_dx,wt=wt,anchor=anchor)


def predict(g,wide_graph,confidence,phase,old,models,trees):
    f=features(g,wide_graph,confidence,phase)
    p,native,rawq,dx,tx,ex,wx=f['p'],f['native'],f['rawq'],f['dx'],f['tx'],f['ex'],f['wx']
    par,dau,pairs,extra_dx,wt,anchor=f['par'],f['dau'],f['pairs'],f['extra_dx'],f['wt'],f['anchor']
    p.update(conditional=predict_logistic(old['edge_conditional'],ex).astype(np.float32),
        exposure=predict_logistic(old['edge_exposure'],ex).astype(np.float32),
        prior=np.array([old['priors'][k] for k in ('node_ratio','edge_rate','division_rate','edge_exposure')]),
        fold=np.int32(old['fold']))
    q=[];e=[]
    for kind,x,t in [('native',dx,tx),('wide',extra_dx,wt)]:
        m=models[kind]
        init=bounded_logit(predict_logistic(m['division_conditional'],t[:,:19]))
        q.append(expit(init+trees[kind].predict(t,raw_score=True,num_threads=1)) if len(t) else np.empty(0))
        e.append(predict_logistic(m['division_exposure'],x))
    for key in ('conditional','exposure'):
        extra_edge=np.column_stack([p[key][anchor],predict_logistic(models['wide']['edge_'+key],wx[pairs[:,1]])])
        p['division_edge_'+key]=np.concatenate([p[key][p['arc_pairs']],extra_edge]).astype(np.float32)
    p['native_count']=np.int64(len(p['parents']))
    p['native_parent_score']=native;p['raw_matcher_q']=rawq
    p['parents']=np.r_[p['parents'],par];p['daughters']=np.concatenate([p['daughters'],dau])
    p['arc_pairs']=np.concatenate([p['arc_pairs'],np.full((len(par),2),-1,np.int64)])
    p['division_conditional']=np.concatenate(q).astype(np.float32)
    p['division_exposure']=np.concatenate(e).astype(np.float32)
    return p


def seed_decision(p,edges):
    count=int(p['native_count']);scores=np.zeros_like(p['division_conditional'])
    scores[:count]=p['division_conditional'][:count]
    ix=np.flatnonzero(p['native_parent_score'][p['parents'][:count]]>=.5)
    seed=dict(p,conditional=p['raw_matcher_q'],division_conditional=scores,
        seed_division_indices=ix,seed_division_conditional=scores[ix])
    return legacy.seed_decision(seed,edges)


def initialize(sid,args):
    _,edges,p,problem=pipeline.load_problem(sid,args)
    legacy.constrain(problem,p);sizes=compact(problem)
    decision,removed=seed_decision(p,edges)
    assert np.all(problem.matrix@decision.astype(float)<=problem.ub+1e-9)
    pipeline.save_problem(problem,args.out/'problems'/f'{sid}.npz')
    np.savez_compressed(args.out/'round_0'/f'{sid}.npz',decision=decision)
    return dict(sid=sid,counts=pipeline.record(problem,decision),matrix=sizes,seed_demotions=removed)


def step(sid,args,iteration,pooled,base_iteration=None):
    start=time.monotonic();problem=pipeline.restore_problem(args.out/'problems'/f'{sid}.npz')
    previous=iteration-1 if base_iteration is None else base_iteration
    current=read(args.out/f'round_{previous}'/f'{sid}.npz')['decision'].astype(float)
    values=problem.gradient(current,pooled)
    candidate,audit=combined_solve(problem,values,current,time_limit=args.time_limit)
    candidate,info=legacy.keep_better_incumbent(problem,values,candidate,audit['solver'],current)
    np.savez_compressed(args.out/f'round_{iteration}'/f'{sid}.npz',decision=candidate.astype(bool))
    return dict(sid=sid,counts=pipeline.record(problem,candidate),solver=info,pruning=audit,
        seconds=time.monotonic()-start,changed_events=int(np.count_nonzero(current[:problem.m]!=candidate[:problem.m])))


def check_bundle(path):
    import lightgbm as lgb
    path=Path(path);meta=json.loads((path/'div797.json').read_text())
    assert isinstance(meta['fold'],int) and meta['features']==FEATURES and meta['parent_floor']==.1
    assert meta['lightgbm_version']==lgb.__version__
    for name,h in meta['files'].items():assert digest(path/name)==h,name
    trees={k:lgb.Booster(model_file=str(path/f'{k}.txt')) for k in ('native','wide')}
    return meta,trees


def export_one(sid,args):
    import resource
    start=time.monotonic();models,trees=check_bundle(args.model.parent)
    old=json.loads(args.model.read_text());g=read(args.graphs/f'{sid}.npz');c=read(args.confidence/f'{sid}.npz')
    np.testing.assert_array_equal(g['nodes'],c['nodes'])
    w=read(args.graphs.parent/'division_graphs'/f'{sid}.npz')
    np.testing.assert_array_equal(g['nodes'],w['nodes'])
    p=predict(g,w,c['confidence'],read(args.div_scores/f'{sid}.npz'),old,models,trees)
    np.savez_compressed(args.out/'predictions'/f'{sid}.npz',**p)
    return dict(sid=sid,nodes=len(p['nodes']),native=int(p['native_count']),candidates=len(p['parents']),
        seconds=time.monotonic()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    mode=sys.argv.pop(1)
    if mode=='optimize':
        pipeline.initialize=initialize;pipeline.step=step;pipeline.main();return
    assert mode=='predict'
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('graphs','confidence','div-scores','model','out'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=4);args=parser.parse_args()
    (args.out/'predictions').mkdir(parents=True,exist_ok=True)
    from joblib import Parallel,delayed
    ids=sorted(p.stem for p in args.graphs.glob('*.npz'));assert ids
    records=[]
    for r in Parallel(n_jobs=args.jobs,return_as='generator')(delayed(export_one)(s,args) for s in ids):
        records.append(r);print('div797 predict',r,flush=True)
    (args.out/'prediction_manifest.json').write_text(json.dumps(dict(samples=records,model=str(args.model),
        base_sha256=digest(args.model),division_sha256=digest(args.model.parent/'div797.json')),indent=2)+'\n')


if __name__=='__main__':main()
