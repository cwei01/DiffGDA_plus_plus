"""Global sparse transport sampling; batches never renormalize the coupling."""
from pathlib import Path
import numpy as np


class GlobalAssignment:
    """One global conditional CDF, shared by both endpoint branches.

    COO rows must be sorted, as produced by hnsw_support. Float64 cumulative
    masses avoid losing small late-row probabilities in a million-row plan.
    """
    def __init__(self, rows, cols, mass, nodes, directory, power=1., check=None):
        self.rows=np.asarray(rows);self.cols=np.asarray(cols)
        self.nodes=int(nodes);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        if self.rows.shape!=self.cols.shape or self.rows.shape!=np.shape(mass):
            raise ValueError('Unaligned sparse coupling')
        if power<=0 or not len(self.rows):raise ValueError('Invalid coupling or mass exponent')
        if self.rows[0]<0 or self.rows[-1]>=nodes:raise ValueError('Source row out of range')
        self.cdf=np.lib.format.open_memmap(directory/'global_cdf.npy',mode='w+',dtype=np.float64,shape=(len(mass)+1,))
        self.cdf[0]=0
        for start in range(0,len(mass),1048576):
            if check:check()
            stop=min(start+1048576,len(mass));values=np.asarray(mass[start:stop],dtype=np.float64)
            if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid coupling mass')
            if start and self.rows[start]<self.rows[start-1]:raise ValueError('COO rows must be sorted')
            if (np.diff(self.rows[start:stop])<0).any():raise ValueError('COO rows must be sorted')
            self.cdf[start+1:stop+1]=self.cdf[start]+np.cumsum(values,dtype=np.float64)
        self.cdf.flush()
        self.ptr=np.searchsorted(self.rows,np.arange(nodes+1))
        self.mass=self.cdf[self.ptr[1:]]-self.cdf[self.ptr[:-1]]
        self.valid=self.mass>1e-12
        raw=np.where(self.valid,self.mass**power,0.)
        self.weights=raw/raw.sum() if raw.sum()>0 else raw

    def draw(self, ids, uniforms):
        """Uniforms are indexed by global source ID, reused for node/edge ends."""
        ids=np.asarray(ids,dtype=np.int64);uniforms=np.asarray(uniforms)
        if uniforms.shape!=(self.nodes,) or (ids<0).any() or (ids>=self.nodes).any():
            raise ValueError('Global source identity required')
        u=uniforms[ids]
        if not np.isfinite(u).all() or (u<0).any() or (u>=1).any():raise ValueError('Uniform values must be in [0,1)')
        if not self.valid[ids].all():raise ValueError('Cannot draw an unsupported source row')
        threshold=self.cdf[self.ptr[ids]]+u*self.mass[ids]
        # Keep a rounded threshold strictly inside its row interval.
        threshold=np.minimum(threshold,np.nextafter(self.cdf[self.ptr[ids+1]],-np.inf))
        at=np.searchsorted(self.cdf,threshold,side='right')-1
        at=np.minimum(np.maximum(at,self.ptr[ids]),self.ptr[ids+1]-1)
        return self.cols[at]


def global_edge_weights(pairs, node_weights, valid, directory, check=None):
    """omega_e = w_i*w_j / sum_global_active_edges(w_i*w_j)."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    pairs=np.asarray(pairs);count=pairs.shape[1]
    out=np.lib.format.open_memmap(directory/'global_edge_weights.npy',mode='w+',dtype=np.float64,shape=(count,))
    total=0.
    for start in range(0,count,1048576):
        if check:check()
        stop=min(start+1048576,count);i,k=pairs[:,start:stop]
        value=node_weights[i]*node_weights[k]*valid[i]*valid[k]
        out[start:stop]=value;total+=np.sum(value,dtype=np.float64)
    if total>0:
        for start in range(0,count,1048576):out[start:start+1048576]/=total
    out.flush();return out


def sampled_weighted_loss(per_item, sampled_ids, global_weights, population):
    """Uniform sampling correction: N/B times the global weighted batch sum."""
    import torch
    if len(sampled_ids)==0 or len(per_item)!=len(sampled_ids):raise ValueError('Invalid center batch')
    weights=torch.as_tensor(np.asarray(global_weights)[np.asarray(sampled_ids)].copy(),
                            device=per_item.device,dtype=per_item.dtype)
    return (per_item*weights).sum()*(population/len(sampled_ids))
