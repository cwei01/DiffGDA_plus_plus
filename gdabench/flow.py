"""Experimental equivalent fast indexing for GDABench-sized graphs.

Dense integer lookup arrays replace repeated binary searches; endpoint draws,
message sampling, normalization and losses are unchanged. Scratch memory O(N+E).
Original implementation stays in large_flow_v3.py.

Sampled joint velocity with exact global dynamic degrees at block boundaries.

All incident edges of sampled vertices contribute to their normalization. Only
messages are sampled. Endpoint assignments are shared globally by source ID;
node and edge center losses retain the original global weighting.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from gdabench.sampling import aggregate
from gdabench.assignment import sampled_weighted_loss


def topology(adjacency, pairs):
    cached=getattr(adjacency,'_pp_v3_indexed',None)
    if cached is None:
        m=pairs.shape[1];a,b=pairs
        values=np.arange(1,m+1,dtype=np.int64)
        cached=coo_matrix((np.r_[values,values],(np.r_[a,b],np.r_[b,a])),shape=adjacency.shape).tocsr()
        if not (np.array_equal(cached.indptr,adjacency.indptr) and
                np.array_equal(cached.indices,adjacency.indices)):
            raise ValueError('Flow adjacency must contain exactly the fixed candidate support')
        adjacency._pp_v3_indexed=cached
    return cached


def skeleton(indexed, centers, fanout, rng):
    outputs=np.unique(centers);layers=[]
    for _ in range(2):
        src=[];dst=[];edge=[];correction=[]
        for j,node in enumerate(outputs):
            lo,hi=indexed.indptr[node:node+2]
            neighbors=indexed.indices[lo:hi];indices=indexed.data[lo:hi]-1
            population=len(neighbors)
            if fanout is not None and population>fanout:
                chosen=rng.choice(population,fanout,replace=False)
                neighbors=neighbors[chosen];indices=indices[chosen]
            src.extend(neighbors);dst.extend([j]*len(neighbors));edge.extend(indices)
            correction.extend([population/max(1,len(neighbors))]*len(neighbors))
            src.append(node);dst.append(j);edge.append(-1);correction.append(1.)
        inputs=np.unique(src)
        layers.append((inputs,outputs,np.searchsorted(inputs,src),np.asarray(dst),
                       np.asarray(edge),np.asarray(correction,dtype=np.float32)))
        outputs=inputs
    return outputs,list(reversed(layers))


def flow_step(model,zs,zt,labels,pairs,values,adjacency,plan,edge_weights,
              target_adjacency,target_roots,rng,batch,delta=.01,fanout=10):
    n=len(zs);m=pairs.shape[1];device=next(model.parameters()).device
    nodes=rng.choice(n,min(batch,n),replace=False)
    edges=rng.choice(m,min(batch,m),replace=False)
    centers=np.unique(np.r_[nodes,pairs[:,edges].reshape(-1)])
    # Draw endpoint/time randomness before neighbor sampling so fanout comparisons
    # use the same path, rather than conflating sampling and transport noise.
    uniforms=rng.random(n)
    t=float(rng.random())
    indexed=topology(adjacency,pairs)
    ids,layers=skeleton(indexed,centers,fanout,rng)
    incident=indexed[ids]
    edge_mask=np.zeros(m,dtype=bool);edge_mask[incident.data-1]=True
    edge_ids=np.flatnonzero(edge_mask)
    edge_pos=np.empty(m,dtype=np.int64);edge_pos[edge_ids]=np.arange(len(edge_ids))
    node_pos=np.empty(n,dtype=np.int64);node_pos[ids]=np.arange(len(ids))
    # Include every endpoint affecting a required degree, not just sampled edges.
    needed_mask=np.zeros(n,dtype=bool);needed_mask[ids]=True;needed_mask[pairs[:,edge_ids]]=True
    needed=np.flatnonzero(needed_mask)
    needed_pos=np.empty(n,dtype=np.int64);needed_pos[needed]=np.arange(len(needed))
    assignment=np.zeros(len(needed),dtype=np.int64)
    good=plan.valid[needed]
    assignment[good]=plan.draw(needed[good],uniforms)
    mapped=target_roots[assignment[needed_pos[pairs[:,edge_ids]]]]
    end_values=np.asarray(target_adjacency[mapped[0],mapped[1]]).reshape(-1).astype(np.float32)
    end_values[mapped[0]==mapped[1]]=0
    active=plan.valid[pairs[0,edge_ids]] & plan.valid[pairs[1,edge_ids]]
    b0=torch.logit(delta+(1-2*delta)*torch.from_numpy(np.array(values[edge_ids],copy=True))).numpy()
    b1=torch.logit(delta+(1-2*delta)*torch.from_numpy(end_values)).numpy()
    b1=np.where(active,b1,b0)
    path_b=(1-t)*b0+t*b1
    decoded=np.where(active,torch.from_numpy(path_b).sigmoid().numpy(),values[edge_ids])
    incident_weights=decoded[edge_pos[incident.data-1]]
    row_ids=np.repeat(np.arange(len(ids)),np.diff(incident.indptr))
    degree=(1+np.bincount(row_ids,weights=incident_weights,minlength=len(ids))).astype(np.float32)
    inv=degree**np.float32(-.5)
    block_layers=[]
    for inputs,outputs,src,dst,ei,correction in layers:
        weights=np.ones(len(ei),dtype=np.float32);mask=ei>=0
        weights[mask]=decoded[edge_pos[ei[mask]]]
        weights*=inv[node_pos[inputs[src]]]*inv[node_pos[outputs[dst]]]*correction
        block_layers.append((inputs,outputs,src,dst,weights))
    end=np.array(zt[assignment[needed_pos[ids]]],copy=True)
    start=np.array(zs[ids],copy=True);end[~plan.valid[ids]]=start[~plan.valid[ids]]
    z=torch.as_tensor((1-t)*start+t*end,device=device)
    h=F.silu(aggregate(model.message1.linear(z),block_layers[0])+model.message1.bias)
    h=F.silu(aggregate(model.message2.linear(h),block_layers[1])+model.message2.bias)
    center_pos=node_pos[centers];zc=z[torch.as_tensor(center_pos,device=device)]
    yc=model.label_embedding(torch.as_tensor(labels[centers].astype(np.int64),device=device))
    phase=z.new_tensor(t)*model.frequencies;clock=torch.cat((phase.sin(),phase.cos()))
    ni=torch.as_tensor(np.searchsorted(centers,nodes),device=device)
    vz=model.node_net(torch.cat((zc[ni],h[ni],clock.expand(len(nodes),-1),yc[ni]),1))
    vz=vz*torch.as_tensor(plan.valid[nodes,None],device=device)
    i,k=[torch.as_tensor(x,device=device) for x in np.searchsorted(centers,pairs[:,edges])]
    ep=edge_pos[edges]
    features=torch.cat((zc[i]+zc[k],(zc[i]-zc[k]).abs(),h[i]+h[k],
        torch.as_tensor(path_b[ep,None],device=device),clock.expand(len(edges),-1),yc[i]+yc[k]),1)
    vb=model.edge_net(features).squeeze(1)*torch.as_tensor(active[ep],device=device)
    displacement=torch.as_tensor((end-start)[node_pos[nodes]],device=device)
    edge_delta=torch.as_tensor((b1-b0)[ep],device=device)
    nl=sampled_weighted_loss((vz-displacement).square().mean(1),nodes,plan.weights,n)
    el=sampled_weighted_loss((vb-edge_delta).square(),edges,edge_weights,m)
    original=values[edges]>0
    details=dict(context_nodes=len(ids),degree_boundary_nodes=len(needed),context_edges=len(edge_ids),
        node_centers=len(nodes),edge_centers=len(edges),normalization='global_dynamic_degree_blocks',
        target_edge_endpoint_nonzero_fraction=float((end_values[ep]>0).mean()))
    for name,mask in [('original',original),('knn_added',~original)]:
        details[name+'_sampled_edges']=int(mask.sum())
        details[name+'_endpoint_fraction']=float((end_values[ep][mask]>0).mean()) if mask.any() else None
    return nl,el,details
