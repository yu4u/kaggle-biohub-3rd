"""Division-only exclusion rows and deterministic repair of a frozen seed."""
import numpy as np
from scipy.sparse import coo_matrix,vstack


def conflicts(parents,daughters,active):
    selected=np.flatnonzero(active)
    outgoing={int(parents[j]):int(j) for j in selected}
    return [(int(j),outgoing[int(d)]) for j in selected for d in daughters[j] if int(d) in outgoing]


def constrain(problem,p):
    start=len(p['src']);parents=p['parents'];daughters=p['daughters']
    both=np.intersect1d(parents,daughters.ravel())
    row=np.full(problem.n,-1,int);row[both]=np.arange(len(both))
    nodes=np.r_[parents,daughters.ravel()]
    events=np.r_[np.arange(len(parents)),np.repeat(np.arange(len(parents)),2)]+start
    keep=row[nodes]>=0
    matrix=coo_matrix((np.ones(keep.sum()),(row[nodes[keep]],events[keep])),
                     shape=(len(both),problem.m+problem.n)).tocsc()
    problem.matrix=vstack([problem.matrix,matrix],format='csc')
    problem.ub=np.r_[problem.ub,np.ones(len(both))]
    return len(both)


def repair_seed(p,decision):
    """Demote low-score conflicting forks; cascade unsupported fork demotions.

    Only remove edges from the original seed. Keep the strongest ordinary
    daughter link when a division is demoted. Rankings always come from 688.
    """
    x=decision.copy();start=len(p['src']);m=start+len(p['parents']);n=len(p['nodes'])
    removed=[]
    def demote(j):
        assert x[start+j]
        arcs=p['arc_pairs'][j]
        assert np.all(arcs>=0)
        assert np.all(p['src'][arcs]==p['parents'][j])
        np.testing.assert_array_equal(p['dst'][arcs],p['daughters'][j])
        x[start+j]=False;x[arcs[np.argmax(p['conditional'][arcs])]]=True
        removed.append(int(j))
    while pairs:=conflicts(p['parents'],p['daughters'],x[start:m]):
        candidates=np.unique(pairs)
        demote(candidates[np.argmin(p['division_conditional'][candidates])])
    while True:
        ordinary=np.flatnonzero(x[:start]);division=np.flatnonzero(x[start:m])
        incoming=np.zeros(n,bool);outgoing=np.zeros(n,bool)
        incoming[p['dst'][ordinary]]=True;outgoing[p['src'][ordinary]]=True
        incoming[p['daughters'][division].ravel()]=True;outgoing[p['parents'][division]]=True
        invalid=division[~incoming[p['parents'][division]]|~outgoing[p['daughters'][division]].all(1)]
        if not len(invalid):break
        for j in invalid:demote(j)
    x[m:]=incoming|outgoing
    return x,removed
