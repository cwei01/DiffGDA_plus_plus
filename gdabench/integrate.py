"""CPU/disk joint-velocity inference with synchronous global Heun states."""

from experiment.efficiency import measured as _eff_measured
from pathlib import Path
import hashlib
import json
import copy
import numpy as np
import torch
import torch.nn.functional as F


def array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def blocks(n,size,reverse=False):
    starts=list(range(0,n,size))
    return reversed(starts) if reverse else starts


def normalized_graph(pairs,values,n):
    from scipy.sparse import coo_matrix,diags
    i,k=array(pairs);values=array(values)
    loops=np.arange(n,dtype=np.int64)
    matrix=coo_matrix((np.concatenate((values,values,np.ones(n,dtype=np.float32))),
        (np.concatenate((i,k,loops)),np.concatenate((k,i,loops)))),shape=(n,n)).tocsr()
    degrees=np.asarray(matrix.sum(1)).reshape(-1)
    inv=np.maximum(degrees,1e-12)**np.float32(-.5)
    norm=(diags(inv)@matrix@diags(inv)).tocsr().astype(np.float32)
    norm.sort_indices();return norm


@torch.no_grad()
def convolution(layer,z,normalized,directory,name,block=8192,activation=False,check=None,reverse=False):
    """Full-neighbor weighted convolution. Degree normalization is global."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    n=len(z);width=layer.linear.out_features
    projected=np.lib.format.open_memmap(directory/f'{name}_linear.npy',mode='w+',dtype=np.float32,shape=(n,width))
    for start in blocks(n,block,reverse):
        if check:check()
        x=torch.from_numpy(np.array(z[start:start+block],dtype=np.float32,copy=True))
        projected[start:start+len(x)]=layer.linear(x).numpy()
    projected.flush()
    out=np.lib.format.open_memmap(directory/f'{name}.npy',mode='w+',dtype=np.float32,shape=(n,width))
    for start in blocks(n,block,reverse):
        if check:check()
        values=torch.from_numpy(normalized[start:start+block].dot(projected))+layer.bias
        if activation=='relu':values=F.relu(values)
        elif activation:values=F.silu(values)
        out[start:start+len(values)]=values.numpy()
    out.flush();return out


@torch.no_grad()
def velocity(model,z,b,t,labels,graph,directory,block=8192,edge_block=32768,check=None,reverse=False):
    """Evaluate both branches from exactly the same immutable Z/B snapshot."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    model=copy.deepcopy(model).cpu().eval()
    z,b,labels=array(z),array(b),array(labels)
    pairs=array(graph.pairs);valid=array(graph.valid);active=array(graph.active)
    # Decode on CPU, preserving the same active/identity convention as GraphState.
    values=np.where(active,torch.from_numpy(np.array(b,copy=True)).sigmoid().numpy(),array(graph.source_values))
    normalized=normalized_graph(pairs,values,len(z))
    h1=convolution(model.message1,z,normalized,directory,'message1',block,True,check,reverse)
    h=convolution(model.message2,h1,normalized,directory,'message2',block,True,check,reverse)
    phase=torch.tensor(float(t),dtype=torch.float32)*model.frequencies
    time_encoding=torch.cat((phase.sin(),phase.cos()))
    node=np.lib.format.open_memmap(directory/'node.npy',mode='w+',dtype=np.float32,shape=z.shape)
    for start in blocks(len(z),block,reverse):
        if check:check()
        stop=min(start+block,len(z))
        label=model.label_embedding(torch.as_tensor(labels[start:stop].astype(np.int64)))
        features=torch.cat((torch.from_numpy(np.array(z[start:stop],copy=True)),
            torch.from_numpy(np.array(h[start:stop],copy=True)),time_encoding.expand(stop-start,-1),label),1)
        node[start:stop]=(model.node_net(features)*torch.from_numpy(valid[start:stop,None])).numpy()
    edge=np.lib.format.open_memmap(directory/'edge.npy',mode='w+',dtype=np.float32,shape=b.shape)
    if model.edge_net is None:edge[:]=0
    else:
        for start in blocks(len(b),edge_block,reverse):
            if check:check()
            stop=min(start+edge_block,len(b));i,k=pairs[:,start:stop]
            zi=torch.from_numpy(np.array(z[i],copy=True));zk=torch.from_numpy(np.array(z[k],copy=True))
            hi=torch.from_numpy(np.array(h[i],copy=True));hk=torch.from_numpy(np.array(h[k],copy=True))
            yi=model.label_embedding(torch.as_tensor(labels[i].astype(np.int64)))
            yk=model.label_embedding(torch.as_tensor(labels[k].astype(np.int64)))
            features=torch.cat((zi+zk,(zi-zk).abs(),hi+hk,
                torch.from_numpy(np.array(b[start:stop,None],copy=True)),time_encoding.expand(stop-start,-1),yi+yk),1)
            edge[start:stop]=(model.edge_net(features).squeeze(1)*torch.from_numpy(active[start:stop])).numpy()
    node.flush();edge.flush()
    return node,edge


@_eff_measured('transport', evaluation=False)
@torch.no_grad()
def integrate(model,z,labels,graph,steps,directory,block=8192,edge_block=32768,check=None,reverse=False,compact=False):
    """Block order cannot mutate the state seen by another block at a Heun stage."""
    if steps<1:raise ValueError('Positive Heun steps required')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    z=array(z);b=array(graph.initial_logits);dt=1./steps
    for step in range(steps):
        work=directory/f'step{step:03d}';work.mkdir(exist_ok=True)
        marker=work/'complete.json'
        state_z=work/'state_z.npy';state_b=work/'state_b.npy'
        if marker.exists():
            saved=json.loads(marker.read_text(encoding='utf-8'))
            for path,name in ((state_z,'state_z_sha256'),(state_b,'state_b_sha256')):
                digest=hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(4<<20),b''):digest.update(chunk)
                if saved.get(name)!=digest.hexdigest():raise ValueError(f'Heun checkpoint hash mismatch: {path.name}')
            z=np.load(state_z,mmap_mode='r');b=np.load(state_b,mmap_mode='r')
            continue
        scratch=directory/'scratch' if compact else work
        scratch.mkdir(exist_ok=True)
        vz,vb=velocity(model,z,b,step*dt,labels,graph,scratch/'first',block,edge_block,check,reverse)
        trial_z=np.lib.format.open_memmap(scratch/'trial_z.npy',mode='w+',dtype=np.float32,shape=z.shape)
        trial_b=np.lib.format.open_memmap(scratch/'trial_b.npy',mode='w+',dtype=np.float32,shape=b.shape)
        for start in blocks(len(z),block,reverse):trial_z[start:start+block]=z[start:start+block]+dt*vz[start:start+block]
        for start in blocks(len(b),edge_block,reverse):trial_b[start:start+edge_block]=b[start:start+edge_block]+dt*vb[start:start+edge_block]
        trial_z.flush();trial_b.flush()
        next_z,next_b=velocity(model,trial_z,trial_b,(step+1)*dt,labels,graph,scratch/'second',block,edge_block,check,reverse)
        out_z=np.lib.format.open_memmap(state_z,mode='w+',dtype=np.float32,shape=z.shape)
        out_b=np.lib.format.open_memmap(state_b,mode='w+',dtype=np.float32,shape=b.shape)
        for start in blocks(len(z),block,reverse):
            if check:check()
            out_z[start:start+block]=z[start:start+block]+.5*dt*(vz[start:start+block]+next_z[start:start+block])
        for start in blocks(len(b),edge_block,reverse):
            if check:check()
            out_b[start:start+edge_block]=b[start:start+edge_block]+.5*dt*(vb[start:start+edge_block]+next_b[start:start+edge_block])
        out_z.flush();out_b.flush()
        hashes={}
        for path,key in ((state_z,'state_z_sha256'),(state_b,'state_b_sha256')):
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(4<<20),b''):digest.update(chunk)
            hashes[key]=digest.hexdigest()
        marker.write_text(json.dumps(dict(completed=True,step=step,**hashes),indent=2),encoding='utf-8')
        z,b=out_z,out_b
        # Windows cannot truncate scratch files while the previous iteration's
        # mappings remain open. States live in distinct durable step folders.
        del vz,vb,next_z,next_b,trial_z,trial_b
    return z,b,2*steps
