"""Length gates on continuation tracks, independent of division connectivity."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def filter_division_tracks(nodes, edges, min_len=6, daughter_min=10):
    nodes=np.asarray(nodes);edges=np.asarray(edges,int).reshape(-1,2)
    if not len(nodes):return nodes,edges
    degree=np.bincount(edges[:,0],minlength=len(nodes))
    is_fork=degree[edges[:,0]]==2
    normal=edges[~is_fork]
    adjacency=coo_matrix((np.ones(len(normal)),(normal[:,0],normal[:,1])),shape=(len(nodes),len(nodes)))
    _,components=connected_components(adjacency,directed=False)
    sizes=np.bincount(components)[components]
    keep=sizes>=min_len
    selected=list(map(tuple,normal[keep[normal].all(axis=1)]))
    for parent in np.flatnonzero(degree==2):
        children=edges[(edges[:,0]==parent),1]
        if not keep[parent]:continue
        if np.all(keep[children]) and np.all(sizes[children]>=daughter_min):
            selected.extend((int(parent),int(d)) for d in children)
        else:
            # Preserve at most one surviving continuation, in cached candidate order.
            surviving=[int(d) for d in children if keep[d]]
            if surviving:selected.append((int(parent),surviving[0]))
    remap=np.full(len(nodes),-1,int);remap[keep]=np.arange(keep.sum())
    return nodes[keep],remap[np.asarray(selected,int).reshape(-1,2)]
