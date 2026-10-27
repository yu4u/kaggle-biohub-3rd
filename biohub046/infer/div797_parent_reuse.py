"""div797_parent.parent_scores with the raw frames encoded once per view (dl_806).

The div_762 window is [raw t-1, warp(t-1 -> t), raw t, warp(t+1 -> t), raw t+1]. Only the two warped
frames depend on the window; the three raw frames are shared with the neighbouring windows, so per
view and fold they are encoded once and cached (037's reuse, applied to the 5-input parent). Per
time point the encoder sees 4 x (new raw frames, usually one) + 8 warped frames instead of 20.
The decoder input, the sampling and the accumulation order are those of parent_scores; the only
difference is the batch composition of the encoder forward (conv rounding under fp16).
"""
import numpy as np
import torch
from biohub046.div3f.data import FlowWarp, apply_d4_points, frame_times
from biohub046.div3f.model import sample_logits
from biohub046.core.dataset import to_ds


@torch.no_grad()
def parent_scores_reuse(models, store, sid, nodes, device, dense):
    """Same signature / result as div797_parent.parent_scores (rot4 x fold mean of sampled sigmoids)."""
    device = torch.device(device)
    T = store.shape[sid][0]
    expected = tuple(int(n) * int(d) // int(f) for n, d, f in zip(store.frame(sid, 0).shape, store.ds, FlowWarp.FLOW_DS))
    assert dense.shape == (T - 1, 3, *expected) and dense.dtype == np.float16
    warp = FlowWarp('.', store.ds, device, keep_raw=False)   # -> [warp t-1, raw t, warp t+1]
    warp._mm[sid] = dense
    amp = torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda')
    frames, raw_views = {}, {}                  # u -> raw frame (numpy) / (4, 1, Z, Y, X) rotated views
    cache = [{} for _ in models]                # per fold: u -> per-stage (4, C, ...) features of the raw views
    score = np.zeros(len(nodes), np.float32)
    for t in sorted(np.unique(nodes[:, 0]).astype(int).tolist()):
        times = frame_times(t, T, 3, 0)
        for u in times:
            if u not in frames:
                frames[u] = store.frame(sid, u)
                rv = torch.from_numpy(frames[u])
                raw_views[u] = torch.stack([torch.rot90(rv, k, dims=(-2, -1)) for k in range(4)]).to(device)[:, None]
        w = warp.warp(sid, t, np.stack([frames[u] for u in times]), times)
        shape = tuple(w.shape[-3:])
        wv = torch.from_numpy(np.stack([w[0], w[2]]))
        warped = torch.stack([torch.rot90(wv, k, dims=(-2, -1)) for k in range(4)]).to(device)   # (4, 2, Z, Y, X)
        ids = np.flatnonzero(nodes[:, 0] == t)
        points = to_ds(nodes[ids, 1:].astype(np.float64), store.ds)
        moved = [torch.from_numpy(apply_d4_points(points, k, 0, shape).astype(np.float32)).to(device) for k in range(4)]
        for m, model in enumerate(models):
            with amp:
                new = [u for u in dict.fromkeys(times) if u not in cache[m]]
                batch = torch.cat([raw_views[u] for u in new] + [warped.reshape(8, 1, *shape)])
                feats = model.encode(batch)
                n_new = 4 * len(new)
                for i, u in enumerate(new):
                    cache[m][u] = [f[4 * i:4 * i + 4] for f in feats]
                wf = [f[n_new:].reshape(4, 2, *f.shape[1:]) for f in feats]
                prev, cur, nxt = (cache[m][u] for u in times)
                stacked = [torch.cat([prev[s], wf[s][:, 0], cur[s], wf[s][:, 1], nxt[s]], 1) for s in range(len(feats))]
                logits = model.decode(stacked, shape)
            prob = np.zeros(len(ids), np.float64)
            for k in range(4):
                prob += torch.sigmoid(sample_logits(logits[k:k + 1].float(), moved[k])).cpu().numpy()
            score[ids] += (prob / 4).astype(np.float32) / len(models)
            for u in [u for u in cache[m] if u < t]:      # later windows start at >= t
                del cache[m][u]
        for u in [u for u in frames if u < t]:
            del frames[u]
            del raw_views[u]
    return score
