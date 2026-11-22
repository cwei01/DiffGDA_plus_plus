import copy,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

class ResourceLimit(RuntimeError):
    pass

def sample_graph(adjacency, roots, hops, fanout, rng):
    """Sample incoming neighbors; roots alone contribute to learning losses."""
    roots = np.asarray(roots, dtype=np.int64)
    frontier = np.unique(roots)
    selected = [frontier]
    src = []
    dst = []
    for _ in range(hops):
        next_nodes = []
        for node in frontier:
            neighbors = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]
            if fanout is not None and len(neighbors) > fanout:
                neighbors = neighbors[rng.choice(len(neighbors), fanout, replace=False)]
            if len(neighbors):
                src.append(np.asarray(neighbors, dtype=np.int64))
                dst.append(np.full(len(neighbors), node, dtype=np.int64))
                next_nodes.append(neighbors)
        if not next_nodes:
            break
        frontier = np.unique(np.concatenate(next_nodes))
        selected.append(frontier)
    ids = np.unique(np.concatenate(selected))
    if src:
        a = np.searchsorted(ids, np.concatenate(src))
        b = np.searchsorted(ids, np.concatenate(dst))
        keys = np.unique(a.astype(np.int64) * len(ids) + b)
        edges = np.stack((keys // len(ids), keys % len(ids)))
    else:
        edges = np.empty((2, 0), dtype=np.int64)
    return (ids, edges, np.searchsorted(ids, roots))

def normalized_adjacency(adjacency):
    from scipy.sparse import eye, diags
    matrix = (adjacency + eye(adjacency.shape[0], dtype=np.float32, format='csr')).tocsr()
    degree = np.asarray(matrix.sum(1)).reshape(-1)
    inverse = np.maximum(degree, 1e-12) ** np.float32(-0.5)
    out = (diags(inverse) @ matrix @ diags(inverse)).tocsr().astype(np.float32)
    out.sort_indices()
    return out

@torch.no_grad()
@torch.no_grad()
def encode_full(encoder, features, adjacency, steps, destination, monitor=None, block=8192):
    """Exact-neighbor CPU forward in row blocks; model weights remain shared."""
    n = len(features)
    width = encoder.propagation.lin.weight.shape[0]
    weight = encoder.propagation.lin.weight.detach().cpu()
    bias = encoder.propagation.bias.detach().cpu()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    current = np.lib.format.open_memmap(destination.with_name(destination.stem + '_linear.npy'), mode='w+', dtype=np.float32, shape=(n, width))
    for start in range(0, n, block):
        if monitor:
            monitor.check()
        x = torch.from_numpy(np.array(features[start:start + block], dtype=np.float32, copy=True))
        current[start:start + len(x)] = F.linear(x, weight).numpy()
    current.flush()
    normalized = normalized_adjacency(adjacency) if steps else None
    for step in range(steps):
        nxt = np.lib.format.open_memmap(destination.with_name(destination.stem + f'_prop{step + 1}.npy'), mode='w+', dtype=np.float32, shape=(n, width))
        for start in range(0, n, block):
            if monitor:
                monitor.check()
            nxt[start:start + block] = normalized[start:start + block].dot(current)
        nxt.flush()
        current = nxt
    out = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float32, shape=(n, width))
    for start in range(0, n, block):
        if monitor:
            monitor.check()
        out[start:start + block] = torch.relu(torch.from_numpy(np.array(current[start:start + block], copy=True)) + bias).numpy()
    out.flush()
    return out

@torch.no_grad()
@torch.no_grad()
def predict_root_probabilities(encoder, z, roots, monitor=None):
    classifier = copy.deepcopy(encoder.classifier).cpu().eval()
    probabilities = np.empty((len(roots), classifier.out_features), dtype=np.float32)
    for begin in range(0, len(roots), 8192):
        if monitor:
            monitor.check()
        ids = roots[begin:begin + 8192]
        probabilities[begin:begin + len(ids)] = classifier(torch.from_numpy(np.array(z[ids], copy=True))).softmax(1).numpy()
    return probabilities
