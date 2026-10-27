"""Build five-fold count calibrators from fixed detection/matcher/div3f caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','POLARS_MAX_THREADS']:
    os.environ.setdefault(key,'1')
import numpy as np
from biohub046.core.metric_calibration import (candidate_features,division_features,fit_logistic,predict_logistic,
                                    EDGE_FEATURES,DIV_FEATURES,CLIP_DENSITY_FEATURE,append_clip_density,clip_density)
from biohub046.core.metric import hungarian_pairs
from biohub046.core.folds import fold_assignments
from omegaconf import OmegaConf
from biohub046.lineage.lineage_evaluation import load_gt_sync


def read_sample(sid,args):
    with np.load(args.graphs/f'{sid}.npz') as z:g={k:z[k] for k in z.files}
    with np.load(args.confidence/f'{sid}.npz') as z:
        if not np.array_equal(g['nodes'],z['nodes']):raise ValueError(f'{sid}: confidence mismatch')
        conf=z['confidence']
    with np.load(args.div_scores/f'{sid}.npz') as z:div=z['score']
    return g,conf,div


def labels(nodes,gt,src,dst,parents,daughters):
    ids,ts,xyz,ge,total=gt
    matched=np.full(len(nodes),-1,int)
    for t in np.unique(ts):
        pi=np.flatnonzero(nodes[:,0]==t);gi=np.flatnonzero(ts==t)
        for a,b in hungarian_pairs(nodes[pi,1:].astype(float),xyz[gi].astype(float)):
            matched[pi[a]]=gi[b]
    lookup={int(g):i for i,g in enumerate(ids)}
    gedge=np.array([(lookup[int(a)],lookup[int(b)]) for a,b in ge],int).reshape(-1,2)
    radix=len(ids)+1
    truth=(gedge[:,0]+1)*radix+gedge[:,1]+1
    codes=(matched[src]+1)*radix+matched[dst]+1
    tp=np.isin(codes,truth)
    gout=np.bincount(gedge[:,0],minlength=len(ids));gin=np.bincount(gedge[:,1],minlength=len(ids))
    so=np.zeros(len(nodes),bool);ti=so.copy()
    valid=matched>=0;so[valid]=gout[matched[valid]]>0;ti[valid]=gin[matched[valid]]>0
    exposed=so[src]|ti[dst]
    d_exposed=so[parents]
    d_tp=np.zeros(len(parents),bool)
    if len(parents):
        c1=(matched[parents]+1)*radix+matched[daughters[:,0]]+1
        c2=(matched[parents]+1)*radix+matched[daughters[:,1]]+1
        d_tp=np.isin(c1,truth)&np.isin(c2,truth)
    return tp,exposed,d_tp,d_exposed,dict(nodes=len(nodes),gt_edges=len(ge),
          gt_divisions=int((gout>=2).sum()),total_nodes=total,arcs=len(src),exposed=int(exposed.sum()))


# dl_494 modes: 'both' = density column in the edge and division calibrators, 'edge' = edge
# calibrators only, 'edge-exposure' = edge exposure only (edge conditional drops the column)
DENSITY_MODES=['both','edge','edge-exposure']


def features(g,conf,div,clip_density):
    src,dst,slot,x,im,om=candidate_features(g,conf,div)
    par,dau,pair,dx=division_features(g,conf,div,src,dst,slot,im,om)
    if clip_density:
        x=append_clip_density(x,g)
        if clip_density=='both':dx=append_clip_density(dx,g)
    return src,dst,slot,x,par,dau,pair,dx


def feature_names(clip_density):
    extra=[CLIP_DENSITY_FEATURE] if clip_density else []
    return EDGE_FEATURES+extra,DIV_FEATURES+(extra if clip_density=='both' else [])


def edge_conditional_columns(x,clip_density):
    return x[:,:-1] if clip_density=='edge-exposure' else x


def prepare(sid,args):
    g,conf,div=read_sample(sid,args)
    src,dst,slot,x,par,dau,pair,dx=features(g,conf,div,args.clip_density)
    tp,exposed,dtp,dexposed,stats=labels(g['nodes'],load_gt_sync(args.train/f'{sid}.geff'),src,dst,par,dau)
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(sid.encode()).digest()[:8],'little'))
    keep=exposed|(rng.random(len(src))<.01)
    weight=np.where(exposed[keep],1.,100.).astype(np.float32)
    np.savez_compressed(args.out/'training'/f'{sid}.npz',x=x[keep],tp=tp[keep],exposed=exposed[keep],weight=weight,
                        dx=dx,dtp=dtp,dexposed=dexposed)
    stats.update(div_candidates=len(par),div_exposed=int(dexposed.sum()),div_positive=int(dtp.sum()),
                 log_density=clip_density(g))
    return sid,stats


# dl_495: the LP prices a node at .1*tp_i/n_total_i with n_total_i = nodes_i * node_ratio,
# and a single train-wide node_ratio (0.843, dominated by the dense clips) over-prices nodes in sparse
# clips whose true ratio is ~1.0 (levels 0-1: 0.97-1.00, levels 5-6: 0.82). Per-clip priors instead:
# node-weighted mean of log(ratio) per probe density level of the TRAINING clips, interpolated linearly
# in log density between the level centres (clamped at the ends). 'nodes' = node_ratio only;
# 'all' = also edge_rate and division_rate (they only scale the pooled denominators).
DENSITY_PRIOR_MODES=['nodes','all']
DENSITY_LEVEL_BINS=[60,100,160,260,420,680]


def fit_density_priors(stats,train):
    import bisect
    lvl=np.array([bisect.bisect(DENSITY_LEVEL_BINS,np.exp(stats[s]['log_density'])) for s in train])
    ld=np.array([stats[s]['log_density'] for s in train]);w=np.array([stats[s]['nodes'] for s in train],float)
    out={'centers':[],'log_node_ratio':[],'log_edge_rate':[],'log_division_rate':[]}
    for k in range(len(DENSITY_LEVEL_BINS)+1):
        m=lvl==k
        if m.sum()<3:continue
        out['centers'].append(float(np.average(ld[m],weights=w[m])))
        for key,num in [('log_node_ratio','total_nodes'),('log_edge_rate','gt_edges'),('log_division_rate','gt_divisions')]:
            ratio=sum(stats[s][num] for s in np.array(train)[m])/w[m].sum()   # pooled within the level
            out[key].append(float(np.log(max(ratio,1e-6))))
    return out


def density_prior(model,log_density):
    dp=model['density_priors'];c=np.array(dp['centers'])
    f=lambda key:float(np.exp(np.interp(log_density,c,np.array(dp[key]))))
    pri=dict(model['priors'])
    pri['node_ratio']=f('log_node_ratio')
    if model.get('density_prior_mode')=='all':
        pri['edge_rate']=f('log_edge_rate');pri['division_rate']=f('log_division_rate')
    return pri


def clip_weights(train,args):
    return {s:1. for s in train}                 # uniform: every train clip weighs the same


def fit_fold(fold,sids,folds,args,stats,*,save=True):
    train=[s for s in sids if fold<0 or folds[s]!=fold]
    cw=clip_weights(train,args)
    blocks=[]
    for sid in train:
        with np.load(args.out/'training'/f'{sid}.npz') as z:b={k:z[k] for k in z.files}
        b['cw']=np.full(len(b['x']),cw[sid],np.float32);b['dcw']=np.full(len(b['dx']),cw[sid],np.float32)
        blocks.append(b)
    data={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    expose=data['exposed'].astype(bool);de=data['dexposed'].astype(bool)
    xc=edge_conditional_columns(data['x'],args.clip_density)
    model={'edge_conditional':fit_logistic(xc[expose],data['tp'][expose],data['cw'][expose]),
           'edge_exposure':fit_logistic(data['x'],expose,data['weight']*data['cw']),
           'division_conditional':fit_logistic(data['dx'][de],data['dtp'][de],data['dcw'][de],l2=1e-3),
           'division_exposure':fit_logistic(data['dx'],de,data['dcw'],l2=1e-3)}
    sums={k:sum(stats[s][k]*cw[s] for s in train) for k in ['nodes','gt_edges','gt_divisions','total_nodes','arcs','exposed']}
    model.update(priors={'node_ratio':sums['total_nodes']/sums['nodes'],
                         'edge_rate':sums['gt_edges']/sums['nodes'],
                         'division_rate':sums['gt_divisions']/sums['nodes'],
                         'edge_exposure':sums['exposed']/sums['arcs']},
                 fold=fold,training_ids=train,clip_density=args.clip_density or False,
                 target_block=getattr(args,'target_block',None) or False,
                 density_prior_mode=getattr(args,'density_priors',None) or False,
                 density_priors=fit_density_priors(stats,train) if getattr(args,'density_priors',None) else None,
                 edge_features=feature_names(args.clip_density)[0],division_features=feature_names(args.clip_density)[1])
    if save:(args.out/f'model_fold{fold}.json').write_text(json.dumps(model,indent=2)+'\n')
    return fold,model


def export(sid,model,args):
    g,conf,div=read_sample(sid,args)
    # the model decides the schema (predict_metric_calibration passes deployment models)
    mode=model.get('clip_density',False) or None
    src,dst,slot,x,par,dau,pair,dx=features(g,conf,div,mode)
    q=predict_logistic(model['edge_conditional'],edge_conditional_columns(x,mode));a=predict_logistic(model['edge_exposure'],x)
    dq=predict_logistic(model['division_conditional'],dx);da=predict_logistic(model['division_exposure'],dx)
    priors=density_prior(model,clip_density(g)) if model.get('density_prior_mode') else model['priors']
    np.savez_compressed(args.out/'predictions'/f'{sid}.npz',nodes=g['nodes'],src=src,dst=dst,slot=slot,
                        conditional=q.astype(np.float32),exposure=a.astype(np.float32),
                        parents=par,daughters=dau,arc_pairs=pair,division_conditional=dq.astype(np.float32),
                        division_exposure=da.astype(np.float32),fold=np.int32(model['fold']),
                        prior=np.array([priors[k] for k in ['node_ratio','edge_rate','division_rate','edge_exposure']]))
    return sid
