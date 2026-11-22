"""Chunked global UOT: exact global reductions over a fixed sorted support."""

from experiment.efficiency import measured as _eff_measured
import math
from pathlib import Path
import numpy as np
import torch


def _check(callback):
    if callback is not None:callback()


def _validate(rows,cols,cost,shape,chunk,check):
    ns,nt=shape;row_count=torch.zeros(ns,dtype=torch.int64);col_count=torch.zeros(nt,dtype=torch.int64)
    previous=-1
    for start in range(0,len(rows),chunk):
        _check(check);r=rows[start:start+chunk];c=cols[start:start+chunk]
        if (r<0).any() or (r>=ns).any() or (c<0).any() or (c>=nt).any():raise ValueError('Support out of range')
        keys=r*nt+c
        if int(keys[0])<=previous or (keys[1:]<=keys[:-1]).any():raise ValueError('Support must be unique and lexicographically sorted')
        previous=int(keys[-1]);v=cost[start:start+chunk]
        if not torch.isfinite(v).all() or (v<0).any():raise ValueError('Invalid cost')
        row_count.index_add_(0,r,torch.ones_like(r));col_count.index_add_(0,c,torch.ones_like(c))
    if (row_count==0).any() or (col_count==0).any():raise ValueError('Every marginal node requires support')


def _reduce(rows,cols,cost,potential,by_row,size,epsilon,chunk,check):
    maximum=cost.new_full((size,),-torch.inf)
    for start in range(0,len(cost),chunk):
        _check(check);sl=slice(start,start+chunk);ids=rows[sl] if by_row else cols[sl]
        other=cols[sl] if by_row else rows[sl]
        value=-cost[sl]/epsilon+potential[other]
        maximum.scatter_reduce_(0,ids,value,reduce='amax',include_self=True)
    total=cost.new_zeros(size)
    for start in range(0,len(cost),chunk):
        _check(check);sl=slice(start,start+chunk);ids=rows[sl] if by_row else cols[sl]
        other=cols[sl] if by_row else rows[sl]
        value=-cost[sl]/epsilon+potential[other]
        total.index_add_(0,ids,(value-maximum[ids]).exp())
    return maximum+total.log()


@torch.no_grad()
def sparse_sinkhorn(rows,cols,cost,shape,epsilon=.2,tau=1.,iterations=100,
                    source_marginal=None,target_marginal=None,resource_check=None,chunk=262144,
                    balanced=False):
    if rows.device.type!='cpu' or cost.device.type!='cpu':raise ValueError('Chunked backend requires CPU state')
    if rows.dtype!=torch.long or cols.dtype!=torch.long:raise ValueError('COO indices must be int64')
    if rows.shape!=cols.shape or rows.shape!=cost.shape or cost.ndim!=1 or not len(rows):raise ValueError('Unaligned support')
    ns,nt=shape
    if min(ns,nt,epsilon,tau,iterations,chunk)<=0:raise ValueError('Invalid UOT settings')
    _validate(rows,cols,cost,shape,chunk,resource_check)
    def marginal(value,n):
        if value is None:return cost.new_full((n,),-math.log(n))
        if value.shape!=(n,) or not torch.isfinite(value).all() or not (value>0).all():raise ValueError('Invalid marginal')
        return (value/value.sum()).log()
    a,b=marginal(source_marginal,ns),marginal(target_marginal,nt)
    u,v=cost.new_zeros(ns),cost.new_zeros(nt)
    # balanced=True is the tau->inf limit: ordinary OT with hard marginals.
    exponent=1.0 if balanced else tau/(tau+epsilon)
    for _ in range(iterations):
        u=exponent*(a-_reduce(rows,cols,cost,v,True,ns,epsilon,chunk,resource_check))
        v=exponent*(b-_reduce(rows,cols,cost,u,False,nt,epsilon,chunk,resource_check))
    mass=torch.empty_like(cost)
    for start in range(0,len(cost),chunk):
        _check(resource_check);sl=slice(start,start+chunk)
        mass[sl]=(u[rows[sl]]-cost[sl]/epsilon+v[cols[sl]]).clamp(-80.,30.).exp()
    return mass


@_eff_measured('sct_refinement', evaluation=False)
@torch.no_grad()
def sparse_refine(zs,zt,labels,initial_prob,rows,cols,args,classes,exact_scale=False,
                  seed=20261023,initial_q=None,resource_check=None,directory=None,chunk=262144):
    from model.sct import _reliability
    import torch.nn.functional as F
    ns,nt=len(zs),len(zt);zs,zt=F.normalize(zs,dim=1),F.normalize(zt,dim=1)
    generator=torch.Generator().manual_seed(seed)
    if exact_scale:
        if ns*nt>4096**2:raise ValueError('Exact scale is only for small checks')
        sr=torch.arange(ns).repeat_interleave(nt);sc=torch.arange(nt).repeat(ns)
    else:
        sr=torch.randint(ns,(min(ns*nt,1000000),),generator=generator)
        sc=torch.randint(nt,sr.shape,generator=generator)
    def scale(v):
        positive=v[v>0]
        return positive.median().clamp_min(1e-12) if len(positive) else v.new_tensor(1.)
    distances=zs.new_empty(len(sr))
    for start in range(0,len(sr),65536):
        _check(resource_check);sl=slice(start,start+65536)
        distances[sl]=(zs[sr[sl]]-zt[sc[sl]]).square().sum(1)
    feature_scale=scale(distances);del distances
    def vector(name):
        if directory is None:return zs.new_empty(len(rows))
        path=Path(directory);path.mkdir(parents=True,exist_ok=True)
        mapped=np.lib.format.open_memmap(path/f'{name}.npy',mode='w+',dtype=np.float32,shape=(len(rows),))
        return torch.from_numpy(mapped)
    feature=vector('feature_cost');cost=vector('combined_cost')
    for start in range(0,len(rows),65536):
        _check(resource_check);sl=slice(start,start+65536)
        feature[sl]=(zs[rows[sl]]-zt[cols[sl]]).square().sum(1)/feature_scale
    q0=initial_prob.clamp_min(1e-8);q0=q0/q0.sum(1,keepdim=True)
    q=q0.clone() if initial_q is None else initial_q.clamp_min(1e-8)
    q=q/q.sum(1,keepdim=True)
    options=dict(floor=args.joint_reliability_floor,scale_floor=args.joint_reliability_scale_floor,
        scale_multiplier=args.joint_reliability_scale,ema=args.joint_reliability_ema)
    reliability=_reliability(q,**options)
    def solve(q,r):
        label_scale=scale(-q[sc,labels[sr]].clamp_min(1e-8).log())
        for start in range(0,len(rows),chunk):
            _check(resource_check);sl=slice(start,start+chunk)
            semantic=-q[cols[sl],labels[rows[sl]]].clamp_min(1e-8).log()/label_scale
            cost[sl]=args.joint_feature_weight*feature[sl]+args.joint_label_weight*semantic
        return sparse_sinkhorn(rows,cols,cost,(ns,nt),args.joint_epsilon,args.joint_tau,
            args.joint_sinkhorn_iters,target_marginal=r/r.sum(),resource_check=resource_check,chunk=chunk,
            balanced=bool(getattr(args,'joint_balanced',False)))
    history=[]
    for iteration in range(args.joint_refine_rounds):
        mass=solve(q,reliability);received=mass.new_zeros(nt*classes);target_mass=mass.new_zeros(nt)
        for start in range(0,len(rows),chunk):
            _check(resource_check);sl=slice(start,start+chunk)
            received.index_add_(0,cols[sl]*classes+labels[rows[sl]],mass[sl]);target_mass.index_add_(0,cols[sl],mass[sl])
        received=received.reshape(nt,classes);prior=reliability/reliability.sum()
        updated=args.joint_update_weight*received+args.joint_prior_weight*prior[:,None]*q0
        updated/=(args.joint_update_weight*target_mass[:,None]+args.joint_prior_weight*prior[:,None]).clamp_min(1e-12)
        updated=updated.clamp_min(1e-8);updated/=updated.sum(1,keepdim=True)
        nr=_reliability(updated,reliability,**options)
        delta=max(float((updated-q).abs().max()),float((nr-reliability).abs().max()))
        q,reliability=updated,nr;history.append(dict(round=iteration+1,max_change=delta,mass=float(mass.sum())))
        if delta<=args.joint_tolerance:break
    return dict(probabilities=q,mass=solve(q,reliability),reliability=reliability,rows=rows,cols=cols,
        history=history,feature_scale=float(feature_scale),scale_sample_pairs=len(sr),support='global_sparse_chunked')
