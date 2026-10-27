"""Flow-warped five-input div762 inference, matching div3f.export_scores."""
import numpy as np
import torch
from biohub046.div3f.data import FlowWarp,apply_d4_points,frame_times
from biohub046.div3f.model import sample_logits
from biohub046.core.dataset import to_ds
from biohub046.core.metric_calibration import SCALE
from biohub046.ssl_flow.inference import point_priors,prepare_zarr_frame


@torch.no_grad()
def flow_with_dense(array,quantiles,peaks,model,downsample,device):
    """One flow forward per boundary, retaining both point and dense outputs."""
    assert tuple(downsample)==(1,4,4)
    offsets=[];dense=[];prev=prepare_zarr_frame(array,0,quantiles,downsample)
    shape=prev.shape
    for t in range(array.shape[0]-1):
        nxt=prepare_zarr_frame(array,t+1,quantiles,downsample)
        result=model(torch.from_numpy(prev)[None,None].to(device),
            torch.from_numpy(nxt)[None,None].to(device),return_features=True)
        points=np.asarray(peaks[t],np.float64)
        offsets.append((point_priors(result,points,downsample)['flow_zyx']*SCALE).astype(np.float32)
            if len(points) else np.zeros((0,3),np.float32))
        dense.append(result['flow'][0].cpu().numpy().astype(np.float16));prev=nxt
    return offsets,np.stack(dense) if dense else np.empty((0,3,*shape),np.float16)


@torch.no_grad()
def parent_scores(models,store,sid,nodes,device,dense):
    """Warp before rotation; average sampled sigmoid across rot4 and folds."""
    device=torch.device(device);T=store.shape[sid][0]
    expected=tuple(int(n)*int(d)//int(f) for n,d,f in zip(store.frame(sid,0).shape,store.ds,FlowWarp.FLOW_DS))
    assert dense.shape==(T-1,3,*expected) and dense.dtype==np.float16
    warp=FlowWarp('.',store.ds,device,keep_raw=True)
    warp._mm[sid]=dense
    frames={};score=np.zeros(len(nodes),np.float32)
    for t in sorted(np.unique(nodes[:,0]).astype(int).tolist()):
        times=frame_times(t,T,3,0)
        for u in times:
            if u not in frames:frames[u]=store.frame(sid,u)
        x=warp.warp(sid,t,np.stack([frames[u] for u in times]),times)
        x=torch.from_numpy(x);views=torch.stack([torch.rot90(x,k,dims=(-2,-1)) for k in range(4)]).to(device)
        ids=np.flatnonzero(nodes[:,0]==t);points=to_ds(nodes[ids,1:].astype(np.float64),store.ds)
        moved=[torch.from_numpy(apply_d4_points(points,k,0,x.shape[-3:]).astype(np.float32)).to(device) for k in range(4)]
        for model in models:
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):logits=model(views)
            prob=np.zeros(len(ids),np.float64)
            for k in range(4):prob+=torch.sigmoid(sample_logits(logits[k:k+1].float(),moved[k])).cpu().numpy()
            score[ids]+=(prob/4).astype(np.float32)/len(models)
        for u in list(frames):
            if u<t:del frames[u]
    return score
