"""Deployment of div688 calibration and div763 exclusion; no GT in inference."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ.setdefault(key,'1')
import numpy as np
from scipy.special import expit
from biohub046.core.metric_calibration import candidate_features,bounded_logit,predict_logistic
from biohub046.core.division_phase_wiring import phase_division_features,phase_edge_features,PHASE_DIV_FEATURES
from biohub046.lineage.div763_constraint import constrain,repair_seed
from biohub046.core.lp_incumbent_guard import keep_better_incumbent
from biohub046.lineage import run_metric_event_pipeline as pipeline

FEATURES=PHASE_DIV_FEATURES+['incoming_count','incoming_mass','incoming_max',
    'pre_parent','pre_support','pre_top1','pre_top2','pre_mean','pre_max']


def read(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def incoming_features(pre,src,dst,q):
    n=len(pre);count=np.bincount(dst,minlength=n);mass=np.bincount(dst,weights=q,minlength=n)
    weighted=np.bincount(dst,weights=q*pre[src],minlength=n)
    maxq,maximum,support=[np.zeros(n) for _ in range(3)]
    np.maximum.at(maxq,dst,q);np.maximum.at(maximum,dst,pre[src]);np.maximum.at(support,dst,pre[src]*q)
    order=np.lexsort((np.arange(len(src)),-q,dst));target=dst[order]
    first=np.r_[True,target[1:]!=target[:-1]] if len(order) else np.zeros(0,bool)
    ix=np.flatnonzero(first);top1=np.zeros(n);top2=np.zeros(n)
    top1[target[ix]]=pre[src[order[ix]]]
    second=ix[count[target[ix]]>1]+1;top2[target[second]]=pre[src[order[second]]]
    aux=np.column_stack([np.log1p(count),mass,maxq]).astype(np.float32)
    rich=np.column_stack([pre,support,top1,top2,
        np.divide(weighted,mass,out=np.zeros(n),where=mass>0),maximum]).astype(np.float32)
    return aux,rich


def extract(g,confidence,phase):
    np.testing.assert_array_equal(g['nodes'],phase['nodes'])
    score=phase['score'];src,dst,slot,x,im,om=candidate_features(g,confidence,score,radius_um=12.)
    eligible=np.where(score>=.1,np.maximum(score,.5),score)
    par,dau,pair,dx=phase_division_features(g,confidence,eligible,phase['pre'],phase['post'],src,dst,slot,im,om)
    dx[:,0]=bounded_logit(score[par])
    ex=phase_edge_features(x,src,dst,score,phase['pre'],phase['post'])
    aux,rich=incoming_features(phase['pre'],src,dst,g['cand_prob'][src,slot])
    raw=dx.copy();dx[:,[14,16]]=0
    tree=np.column_stack([dx,aux[par],rich[par]])
    return dict(nodes=g['nodes'],src=src,dst=dst,slot=slot,parents=par,daughters=dau,arc_pairs=pair),ex,raw,dx,tree


def predict(g,confidence,phase,old,linear,tree):
    p,ex,raw,dx,tx=extract(g,confidence,phase)
    p.update(conditional=predict_logistic(old['edge_conditional'],ex).astype(np.float32),
        exposure=predict_logistic(old['edge_exposure'],ex).astype(np.float32),
        division_exposure=predict_logistic(linear['division_exposure'],dx).astype(np.float32),
        division_conditional=expit(bounded_logit(predict_logistic(linear['division_conditional'],dx))+
            tree.predict(tx,raw_score=True,num_threads=1)).astype(np.float32),
        prior=np.array([old['priors'][k] for k in ('node_ratio','edge_rate','division_rate','edge_exposure')]),
        fold=np.int32(old['fold']))
    ix=np.flatnonzero(phase['score'][p['parents']]>=.5)
    p['seed_division_indices']=ix
    p['seed_division_conditional']=predict_logistic(old['division_conditional'],raw[ix]).astype(np.float32)
    return p


def seed_decision(p,edges):
    n=len(p['nodes']);o=len(p['src']);m=o+len(p['parents']);decision=np.zeros(m+n,bool)
    selected=np.isin(p['src']*n+p['dst'],edges[:,0]*n+edges[:,1])
    decision[:o]=selected;decision[m+p['src'][selected]]=True;decision[m+p['dst'][selected]]=True
    incoming=np.full(n,-1,int);outgoing=np.full(n,-1,int);active=np.flatnonzero(selected)
    incoming[p['dst'][active]]=p['src'][active];outgoing[p['src'][active]]=active
    for k in np.argsort(-p['seed_division_conditional'],kind='stable'):
        if p['seed_division_conditional'][k]<.25:break
        j=p['seed_division_indices'][k];parent=p['parents'][j];dau=p['daughters'][j];a=outgoing[parent]
        if incoming[parent]<0 or a<0 or p['dst'][a] not in dau:continue
        if any(incoming[d]>=0 and incoming[d]!=parent for d in dau):continue
        if any(outgoing[d]==-1 for d in dau):continue
        decision[a]=False;decision[o+j]=True;outgoing[parent]=-2;incoming[dau]=parent;decision[m+dau]=True
    return repair_seed(p,decision)


def initialize(sid,args):
    _,edges,p,problem=pipeline.load_problem(sid,args)
    added=constrain(problem,p);decision,removed=seed_decision(p,edges)
    assert np.all(problem.matrix@decision.astype(float)<=problem.ub+1e-9)
    pipeline.save_problem(problem,args.out/'problems'/f'{sid}.npz')
    np.savez_compressed(args.out/'round_0'/f'{sid}.npz',decision=decision)
    return dict(sid=sid,counts=pipeline.record(problem,decision),exclusion_rows=added,seed_demotions=removed)


def step(sid,args,iteration,pooled,base_iteration=None):
    start=time.monotonic();problem=pipeline.restore_problem(args.out/'problems'/f'{sid}.npz')
    previous=iteration-1 if base_iteration is None else base_iteration
    current=read(args.out/f'round_{previous}'/f'{sid}.npz')['decision'].astype(float)
    values=problem.gradient(current,pooled)
    candidate,info=problem.linear_solve(values,time_limit=args.time_limit,incumbent=current)
    candidate,info=keep_better_incumbent(problem,values,candidate,info,current)
    np.savez_compressed(args.out/f'round_{iteration}'/f'{sid}.npz',decision=candidate.astype(bool))
    return dict(sid=sid,counts=pipeline.record(problem,candidate),solver=info,
        seconds=time.monotonic()-start,changed_events=int(np.count_nonzero(current[:problem.m]!=candidate[:problem.m])))


def check_bundle(path):
    import lightgbm as lgb
    path=Path(path);meta=json.loads((path/'div688.json').read_text())
    assert isinstance(meta['fold'],int) and meta['features']==FEATURES and meta['parent_floor']==.1
    assert meta['lightgbm_version']==lgb.__version__,(meta['lightgbm_version'],lgb.__version__)
    assert digest(path/'div688.txt')==meta['tree_sha256']
    assert digest(path/'model_fold-1.json')==meta['base_sha256']
    return meta,lgb.Booster(model_file=str(path/'div688.txt'))


def export_one(sid,args):
    import resource
    start=time.monotonic();linear,tree=check_bundle(args.model.parent)
    old=json.loads(args.model.read_text());g=read(args.graphs/f'{sid}.npz');c=read(args.confidence/f'{sid}.npz')
    np.testing.assert_array_equal(g['nodes'],c['nodes'])
    p=predict(g,c['confidence'],read(args.div_scores/f'{sid}.npz'),old,linear,tree)
    np.savez_compressed(args.out/'predictions'/f'{sid}.npz',**p)
    return dict(sid=sid,nodes=len(p['nodes']),candidates=len(p['parents']),seconds=time.monotonic()-start,
        maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


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
        records.append(r);print('div688 predict',r,flush=True)
    (args.out/'prediction_manifest.json').write_text(json.dumps(dict(samples=records,model=str(args.model),
        base_sha256=digest(args.model),division_sha256=digest(args.model.parent/'div688.json')),indent=2)+'\n')


if __name__=='__main__':main()
