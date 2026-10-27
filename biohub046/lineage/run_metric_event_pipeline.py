"""Pooled-metric free rewiring using OOF count predictions, with final evaluation.

Optimization never reads GT. Shared metric prices are broadcast to independent
sample subproblems. Optional GT is read only after all optimization rounds end.
"""
import argparse
import json
import os
from pathlib import Path
import time

for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','POLARS_MAX_THREADS']:
    os.environ.setdefault(key,'1')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
import numpy as np
from biohub046.core.metric_event_optimizer import MetricEvent,CountPrior,MetricEventProblem,save_problem,restore_problem
from biohub046.core.metric_counts import pooled_components
from biohub046.core.lineage_optimizer import validate_graph,OptimizerConfig,optimize_lineage


def load_problem(sid,args):
    node_adjustment=getattr(args,'node_adjustment','unclamped')
    with np.load(args.predictions/f'{sid}.npz') as z:v={k:z[k] for k in z.files}
    with np.load(args.graphs/f'{sid}.npz') as z:
        nodes,edges=z['nodes'],z['edges']
        if not np.array_equal(nodes,v['nodes']):raise ValueError(f'{sid}: prediction nodes mismatch')
        if not np.array_equal(z['cand_idx'][v['src'],v['slot']],v['dst']):raise ValueError('candidate mismatch')
    n=len(nodes);q=v['conditional'].astype(float)
    exposure=v['exposure'].astype(float) if args.exposure=='learned' else np.full(len(q),v['prior'][3])
    tp,fp=q*exposure,(1-q)*exposure
    events=[MetricEvent(int(i),(int(j),),float(t),float(f)) for i,j,t,f in zip(v['src'],v['dst'],tp,fp)]
    if args.include_divisions:
        if 'division_edge_conditional' in v:
            # A division may use edges that are not standalone continuation candidates.
            qd=v['division_edge_conditional'].astype(float)
            ed=v['division_edge_exposure'].astype(float)
            if qd.shape!=(len(v['parents']),2) or ed.shape!=qd.shape:
                raise ValueError('division edge prediction shape mismatch')
            if args.exposure!='learned':
                if 'division_edge_exposure_prior' not in v:
                    raise ValueError('constant exposure requires division_edge_exposure_prior for separate division edges')
                ed=np.full(qd.shape,v['division_edge_exposure_prior'])
            dtp,dfp=(qd*ed).sum(1),((1-qd)*ed).sum(1)
        else:
            dtp,dfp=tp[v['arc_pairs']].sum(1),fp[v['arc_pairs']].sum(1)
        for i,dau,etp,efp,dq,de in zip(v['parents'],v['daughters'],dtp,dfp,v['division_conditional'],v['division_exposure']):
            events.append(MetricEvent(int(i),tuple(map(int,dau)),float(etp),float(efp),
                                      float(dq*de),float((1-dq)*de)))
    prior=CountPrior(float(n*v['prior'][0]),float(n*v['prior'][1]),float(n*v['prior'][2]))
    problem=MetricEventProblem(np.zeros(n,int),events,[prior],node_times=nodes[:,0], clamp=node_adjustment=='clamped')
    return nodes,edges,v,problem


def record(problem,decision):
    tp,fp,nn,dtp,dfp=(float(v[0]) for v in problem.counts(decision))
    prior=problem.priors[0]
    return dict(tp=tp,fp=fp,nodes=nn,dtp=dtp,dfp=dfp,total=prior.estimated_nodes,
                gt_edges=prior.gt_edges,gt_divisions=prior.gt_divisions)


def aggregate(records, clamp=False):
    return pooled_components(*[[r[k] for r in records] for k in
        ['tp','fp','nodes','dtp','dfp','total','gt_edges','gt_divisions']], clamp=clamp)


def initialize(sid,args):
    nodes,edges,v,problem=load_problem(sid,args)
    n=len(nodes);m=len(v['src'])+(len(v['parents']) if args.include_divisions else 0)
    decision=np.zeros(m+n)
    selected=np.isin(v['src']*n+v['dst'],edges[:,0]*n+edges[:,1])
    decision[:len(selected)]=selected
    decision[m+v['src'][selected]]=1;decision[m+v['dst'][selected]]=1
    if args.include_divisions:
        incoming=np.full(n,-1,int);outgoing=np.full(n,-1,int)
        active=np.flatnonzero(selected)
        incoming[v['dst'][active]]=v['src'][active];outgoing[v['src'][active]]=active
        # Feasible warm start avoids a zero-division ratio whose first tangent
        # has no FP penalty. This is initialization only: every choice is free
        # to change in subsequent pooled optimization rounds.
        for j in np.argsort(-v['division_conditional'],kind='stable'):
            if v['division_conditional'][j]<.25:break
            parent=int(v['parents'][j]);dau=v['daughters'][j];a=int(outgoing[parent])
            if incoming[parent]<0 or a<0 or v['dst'][a] not in dau:continue
            if any(incoming[d]>=0 and incoming[d]!=parent for d in dau):continue
            if any(outgoing[d]==-1 for d in dau):continue
            decision[a]=0;decision[len(selected)+j]=1
            outgoing[parent]=-2  # selected division, also a valid context edge
            incoming[dau]=parent;decision[m+dau]=1
    r=record(problem,decision)
    save_problem(problem, args.out/'problems'/f'{sid}.npz')
    np.savez_compressed(args.out/'round_0'/f'{sid}.npz',decision=decision.astype(bool))
    return dict(sid=sid,counts=r)


def step(sid,args,iteration,pooled,base_iteration=None):
    start=time.monotonic();problem=restore_problem(args.out/'problems'/f'{sid}.npz')
    with np.load(args.out/f'round_{iteration-1 if base_iteration is None else base_iteration}'/f'{sid}.npz') as z:current=z['decision'].astype(float)
    candidate,info=problem.linear_solve(problem.gradient(current,pooled),time_limit=args.time_limit,incumbent=current)
    np.savez_compressed(args.out/f'round_{iteration}'/f'{sid}.npz',decision=candidate.astype(bool))
    return dict(sid=sid,counts=record(problem,candidate),solver=info,seconds=time.monotonic()-start,
                changed_events=int(np.count_nonzero(current[:problem.m]!=candidate[:problem.m])))


def final_graph(sid,args,iteration):
    with np.load(args.predictions/f'{sid}.npz') as z:v={k:z[k] for k in z.files}
    with np.load(args.out/f'round_{iteration}'/f'{sid}.npz') as z:decision=z['decision']
    nodes=v['nodes'];m=len(decision)-len(nodes);keep=decision[m:]
    selected=decision[:len(v['src'])]
    edges=list(zip(v['src'][selected].tolist(),v['dst'][selected].tolist()))
    if args.include_divisions:
        for active,p,ds in zip(decision[len(v['src']):m],v['parents'],v['daughters']):
            if active:edges.extend([(int(p),int(ds[0])),(int(p),int(ds[1]))])
    remap=np.full(len(nodes),-1,int);remap[keep]=np.arange(keep.sum())
    e=np.asarray([(remap[a],remap[b]) for a,b in edges],int).reshape(-1,2);n=nodes[keep]
    validate_graph(n,e,branching=True)
    outputs={'raw':(n,e)}
    from biohub046.lineage.gap_close import gap_close,comp_sizes,apply_keep
    ng,eg,_,_=gap_close(n,e,lmax=int(getattr(args,'gap_lmax',2)),g1=3.)
    kg=comp_sizes(len(ng),eg)>=6
    nf,ef=apply_keep(ng,eg,kg)
    if not args.include_divisions:
        with np.load(args.div_scores/f'{sid}.npz') as z:div=z['score'][keep]
        div=np.pad(div,(0,len(ng)-len(n)))[kg]
        nf,ef,_=optimize_lineage(nf,ef,div,OptimizerConfig(mode='division'))
    if args.include_divisions and getattr(args,'division_postprocess','legacy')=='independent':
        from biohub046.core.lineage_postprocess import filter_division_tracks
        nf,ef=filter_division_tracks(ng,eg)
    validate_graph(nf,ef,branching=True)
    outputs['final']=(nf,ef)
    rows={}
    for name,(nn,ee) in outputs.items():
        np.savez_compressed(args.out/name/f'{sid}.npz',nodes=nn,edges=np.asarray(ee,int).reshape(-1,2))
    return dict(sid=sid,results=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['graphs','predictions','div-scores','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--exposure',choices=['learned','constant'],default='learned')
    p.add_argument('--include-divisions',action='store_true')
    p.add_argument('--node-adjustment',choices=['clamped','unclamped'],default='clamped')
    p.add_argument('--division-postprocess',choices=['independent','legacy'],default='legacy',help='legacy: L6 on the selected lineage; independent: optional track L6 and daughter length 10 ablation')
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--time-limit',type=float,default=15.)
    p.add_argument('--gap-lmax',type=int,default=2,help='gap closing: bridge up to this many missing frames (dl_488: 3 is +0.0008 on 017)')
    p.add_argument('--jobs',type=int,default=6);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--ids',type=Path,default=None,help='JSON list of the movies to optimize (default: every prediction)')
    args=p.parse_args()
    if args.rounds<1 or args.jobs<1 or args.time_limit<=0:p.error('invalid runtime configuration')
    sids=sorted(json.loads(args.ids.read_text())) if args.ids else sorted(p.stem for p in args.predictions.glob('*.npz'))
    if args.limit:sids=sids[:args.limit]
    if not sids:p.error('no predictions')
    args.out.mkdir(parents=True,exist_ok=True)
    for name in ['round_0','raw','final','problems']:(args.out/name).mkdir(exist_ok=True)
    from joblib import Parallel,delayed
    rows=Parallel(n_jobs=args.jobs)(delayed(initialize)(s,args) for s in sids)
    pooled=aggregate([r['counts'] for r in rows], args.node_adjustment=='clamped');history=[dict(round=0,pooled=pooled,accepted=True)];chosen=0
    print('initial',pooled,flush=True)
    for iteration in range(1,args.rounds+1):
        (args.out/f'round_{iteration}').mkdir(exist_ok=True)
        prices=pooled.copy()
        attempts=[]
        for attempt in range(3):
            proposals=[]
            gen=Parallel(n_jobs=args.jobs,return_as='generator')(delayed(step)(s,args,iteration,prices,chosen) for s in sids)
            for i,row in enumerate(gen,1):
                proposals.append(row)
                if i%25==0 or i==len(sids):print('round',iteration,'attempt',attempt,i,'/',len(sids),flush=True)
            new=aggregate([r['counts'] for r in proposals],args.node_adjustment=='clamped')
            accepted=new['score']>pooled['score']+1e-10
            attempts.append(dict(prices=prices,pooled=new,accepted=accepted))
            if accepted:break
            # Fractional-programming retry: price proposed FP at its own ratio,
            # keeping the feasible incumbent and local node linearization fixed.
            if abs(new['division_jaccard']-prices['division_jaccard'])<1e-10:break
            prices={**prices,'division_jaccard':new['division_jaccard']}
        history.append(dict(round=iteration,pooled=new,accepted=accepted,attempts=attempts))
        (args.out/f'round_{iteration}.json').write_text(json.dumps(proposals,indent=2)+'\n')
        print('proposal',new,'accepted',accepted,flush=True)
        if not accepted:break
        chosen=iteration;pooled=new
    summary=dict(global_optimum_certified=False,inputs={k:str(v) for k,v in vars(args).items()},chosen_round=chosen,history=history)
    (args.out/'optimization.json').write_text(json.dumps(summary,indent=2)+'\n')
    final=[]
    gen=Parallel(n_jobs=args.jobs,return_as='generator')(delayed(final_graph)(s,args,chosen) for s in sids)
    with (args.out/'per_sample.jsonl').open('w') as stream:
        for i,row in enumerate(gen,1):
            final.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
            if i%25==0 or i==len(sids):print('evaluate',i,'/',len(sids),flush=True)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
