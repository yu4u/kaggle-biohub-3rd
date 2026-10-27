"""GT-free feature and radius16 candidate construction for submission041 (div797)."""
import numpy as np
from biohub046.core.metric_calibration import SCALE,bounded_logit


def radius16_graph(g,phase,per_t,offsets,matcher,device):
    """Reuse cached trunk embeddings; run only the matcher at radius16."""
    nodes=g['nodes'];n=len(nodes)
    valid=(g['cand_idx']>=0)&np.isfinite(g['cand_dist'])&(g['cand_dist']<=12.)
    src,slot=np.nonzero(valid);dst=g['cand_idx'][src,slot];q=g['cand_prob'][src,slot]
    incoming=np.zeros(n);outgoing=np.zeros(n)
    np.maximum.at(incoming,dst,q);np.maximum.at(outgoing,src,q)
    anchor=np.full(n,-1,int);ix=np.argmax(np.where(valid,g['cand_prob'],-1),axis=1)
    active=outgoing>0;anchor[active]=g['cand_idx'][np.flatnonzero(active),ix[active]]
    v={k:a for k,a in g.items() if not k.startswith('cand_')}
    v['cand_idx']=np.full((n,10),-1,np.int64);v['cand_dist']=np.full((n,10),np.inf,np.float32)
    for k in ('cand_prob','cand_rev','cand_sym','cand_rank'):v[k]=np.zeros((n,10),np.float32)
    v['cand_idx'][:,:5]=np.where(valid,g['cand_idx'],-1)
    v['cand_dist'][:,:5]=np.where(valid,g['cand_dist'],np.inf)
    v.update(anchor=anchor,incoming=incoming,outgoing=outgoing)
    for t in sorted(np.unique(nodes[:,0]).astype(int).tolist()):
        ia=np.flatnonzero(nodes[:,0]==t);ib=np.flatnonzero(nodes[:,0]==t+1)
        if not len(ia) or not len(ib):continue
        pa,ea=per_t[t];pb,eb=per_t[t+1];flow=offsets[t]
        np.testing.assert_array_equal(pa,nodes[ia,1:]);np.testing.assert_array_equal(pb,nodes[ib,1:])
        distance=np.linalg.norm((pa*SCALE+flow)[:,None]-pb[None]*SCALE,axis=2)
        raw=g['cand_idx'][ia];rr,cc=np.nonzero(raw>=0);local=np.searchsorted(ib,raw[rr,cc])
        np.testing.assert_allclose(distance[rr,local],g['cand_dist'][ia][rr,cc],rtol=0,atol=2e-5)
        probs=probabilities(matcher,pa,ea,pb,eb,flow,16.,device)
        old=np.where(valid[ia],np.searchsorted(ib,np.maximum(raw,0)),-1)
        eligible=(phase['score'][ia]>=.1)&(incoming[ia]>0)&(anchor[ia]>=0)
        extra=new_neighbors(distance,old,eligible,outgoing[ib]>0,probs[0])
        ar,ac=np.nonzero(extra>=0);target=extra[ar,ac]
        v['cand_idx'][ia[ar],5+ac]=ib[target];v['cand_dist'][ia[ar],5+ac]=distance[ar,target]
        merged=v['cand_idx'][ia];r,c=np.nonzero(merged>=0);target=np.searchsorted(ib,merged[r,c])
        for key,a in zip(('cand_prob','cand_rev','cand_sym'),probs):v[key][ia[r],c]=a[r,target]
        order=np.argsort(-probs[0],axis=1,kind='stable');rank=np.empty_like(order)
        np.put_along_axis(rank,order,np.broadcast_to(np.arange(len(ib)),order.shape),axis=1)
        v['cand_rank'][ia[r],c]=rank[r,target]/4.
    return v

def moments(index,delta,q,n):
    mass=np.bincount(index,weights=q,minlength=n)
    mean=np.column_stack([np.bincount(index,weights=q*delta[:,j],minlength=n) for j in range(3)])
    mean=np.divide(mean,mass[:,None],out=np.full_like(mean,np.nan),where=mass[:,None]>0)
    second=np.bincount(index,weights=q*np.sum(delta*delta,axis=1),minlength=n)
    second=np.divide(second,mass,out=np.full(n,np.nan),where=mass>0)
    return mean,np.sqrt(np.maximum(second-np.sum(mean*mean,axis=1),0))

def motion_features(nodes,src,dst,q,parents,daughters):
    xyz=nodes[:,1:]*SCALE
    assert np.all(nodes[dst,0]-nodes[src,0]==1)
    assert np.isfinite(q).all() and np.all(q>=0)
    delta=xyz[dst]-xyz[src]
    incoming,pre_spread=moments(dst,delta,q,len(nodes))
    outgoing,post_spread=moments(src,delta,q,len(nodes))
    children=xyz[daughters];parent=xyz[parents]
    midpoint=children.mean(axis=1)
    mid_error=np.linalg.norm(midpoint-parent-incoming[parents],axis=1)
    child_error=np.linalg.norm(outgoing[daughters]-(children-parent[:,None,:]),axis=2)
    next_children=children+outgoing[daughters]
    separation=np.linalg.norm(children[:,0]-children[:,1],axis=1)
    change=np.linalg.norm(next_children[:,0]-next_children[:,1],axis=1)-separation
    return np.column_stack([mid_error,child_error.min(axis=1),child_error.max(axis=1),change,
        pre_spread[parents],post_spread[daughters].max(axis=1)]).astype(np.float32)

def probabilities(model,pa,ea,pb,eb,flow,radius,device):
    import torch
    with torch.no_grad():
        args=[torch.as_tensor(v,device=device,dtype=torch.float32) for v in
              (ea,pa*SCALE,eb,pb*SCALE)]
        logits,na,nb=model(*args,radius,candidate_offset_a=torch.as_tensor(flow,device=device,dtype=torch.float32))
        row=torch.softmax(torch.cat([logits,na[:,None]],1),1)[:,:-1]
        rev=torch.softmax(torch.cat([logits,nb[None,:]],0),0)[:-1,:]
        sym=torch.sigmoid((2*logits-na[:,None]-nb[None,:])/2)
        return [a.cpu().numpy() for a in ((row+rev)/2,rev,sym)]

def new_neighbors(distance,old_targets,eligible,target_supported,probability,limit=5):
    allowed=(distance<=16.)&(probability>0)&target_supported[None,:]&eligible[:,None]
    rows,cols=np.nonzero(old_targets>=0)
    allowed[rows,old_targets[rows,cols]]=False
    width=min(limit,distance.shape[1])
    ranked=np.argsort(np.where(allowed,distance,np.inf),axis=1,kind='stable')[:,:width]
    return np.where(np.take_along_axis(allowed,ranked,axis=1),ranked,-1)

def extra_pairs(g,p,phase,conf,src,dst,slot,original_graph):
    n=len(g['nodes']);lookup=np.full(g['cand_idx'].shape,-1,int);lookup[src,slot]=np.arange(len(src))
    parent,col=np.nonzero(g['cand_idx'][:,5:]>=0);col=col+5
    anchor_slot=np.argmax(g['cand_idx'][parent,:5]==g['anchor'][parent,None],axis=1)
    pairs=np.column_stack([lookup[parent,anchor_slot],lookup[parent,col]])
    assert np.all(pairs>=0)
    dau=dst[pairs];xyz=g['nodes'][:,1:]*SCALE
    # Frozen median motion uses exactly the original radius12 top choices.
    motion=np.zeros((int(g['nodes'][:,0].max())+1,3))
    for t in np.unique(g['nodes'][:,0]):
        support=np.flatnonzero((g['nodes'][:,0]==t)&(g['outgoing']>=.5))
        if len(support)>=5:motion[int(t)]=np.median(xyz[g['anchor'][support]]-xyz[support],axis=0)
    v=xyz[dau]-xyz[parent,None]-motion[g['nodes'][parent,0].astype(int),None]
    distance=np.linalg.norm(v,axis=2);cos=np.divide(np.sum(v[:,0]*v[:,1],axis=1),np.prod(distance,axis=1),
        out=np.ones(len(parent)),where=np.min(distance,axis=1)>=1e-6)
    mask=(np.min(distance,axis=1)>=1e-6)&(cos<=.5)&(g['outgoing'][dau].min(1)>0)
    parent,dau,pairs,v,distance,cos,anchor_slot=[a[mask] for a in (parent,dau,pairs,v,distance,cos,anchor_slot)]
    qr=g['cand_prob'][src,slot][pairs];rev=g['cand_rev'][src,slot][pairs];sym=g['cand_sym'][src,slot][pairs]
    support=np.zeros(n);original_q=original_graph['cand_prob'][p['src'],p['slot']]
    np.maximum.at(support,p['dst'],phase['pre'][p['src']]*original_q)
    dx=np.column_stack([bounded_logit(phase['score'][parent]),cos,np.linalg.norm(v.mean(1),axis=1),
        np.linalg.norm(v[:,0]-v[:,1],axis=1),distance.min(1),distance.max(1),qr.min(1),qr.max(1),
        rev.min(1),sym.min(1),conf[parent],conf[dau].min(1),g['incoming'][parent],g['outgoing'][dau].min(1),
        phase['pre'][parent],phase['post'][parent],support[parent],phase['post'][dau].min(1),phase['post'][dau].max(1)]).astype(np.float32)
    old_index=np.full(g['cand_idx'].shape,-1,int);old_index[p['src'],p['slot']]=np.arange(len(p['src']))
    anchor_arc=old_index[parent,anchor_slot];assert np.all(anchor_arc>=0)
    return parent,dau,pairs,dx,anchor_arc
