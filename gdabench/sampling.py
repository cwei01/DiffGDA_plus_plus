"""Root blocks with global GCN degrees; no subgraph degree normalization.

Full neighbors reproduce full-graph root outputs. Finite fanout uses uniform
neighbor sampling with degree/sample-count correction at each message layer.
The self-loop is exact and is not included in the neighbor sampling population.
"""
import numpy as np
import torch
import torch.nn.functional as F

def blocks(adjacency, roots, depth, fanout, rng):
    roots = np.asarray(roots, dtype=np.int64)
    if depth < 0 or (fanout is not None and fanout < 1):
        raise ValueError('Invalid sampling settings')
    outputs = np.unique(roots)
    layers = []
    degrees = np.asarray(adjacency.sum(1)).reshape(-1) + 1
    inv = np.maximum(degrees, 1e-12) ** (-0.5)
    for _ in range(depth):
        sources = []
        destinations = []
        weights = []
        for (j, node) in enumerate(outputs):
            (start, stop) = adjacency.indptr[node:node + 2]
            neighbors = adjacency.indices[start:stop]
            values = adjacency.data[start:stop]
            keep = neighbors != node
            neighbors = neighbors[keep]
            values = values[keep]
            population = len(neighbors)
            if fanout is not None and population > fanout:
                selected = rng.choice(population, fanout, replace=False)
                neighbors = neighbors[selected]
                values = values[selected]
            correction = population / max(len(neighbors), 1)
            sources.extend(neighbors.tolist())
            destinations.extend([j] * len(neighbors))
            weights.extend((values * inv[node] * inv[neighbors] * correction).tolist())
            sources.append(int(node))
            destinations.append(j)
            weights.append(float(inv[node] ** 2))
        inputs = np.unique(np.asarray(sources, dtype=np.int64))
        layers.append((inputs, outputs, np.searchsorted(inputs, sources), np.asarray(destinations, dtype=np.int64), np.asarray(weights, dtype=np.float32)))
        outputs = inputs
    return (outputs, list(reversed(layers)), np.searchsorted(np.unique(roots), roots))

def aggregate(features, layer):
    (_, outputs, source, destination, weights) = layer
    device = features.device
    src = torch.as_tensor(source, device=device)
    dst = torch.as_tensor(destination, device=device)
    w = torch.as_tensor(weights, device=device, dtype=features.dtype)
    return features.new_zeros((len(outputs), features.shape[1])).index_add(0, dst, features[src] * w[:, None])

def encode(model, domain, graph, roots, hops, rng, fanout=10):
    expected = model.source_steps if domain == 'source' else model.target_steps
    if hops != expected:
        raise ValueError('Sampler depth and model propagation differ')
    (ids, layers, positions) = blocks(graph['adjacency'], roots, hops, fanout, rng)
    device = next(model.parameters()).device
    x = torch.as_tensor(np.array(graph['x'][ids], copy=True), device=device)
    h = model.propagation.lin(x)
    for layer in layers:
        h = aggregate(h, layer)
    if model.propagation.bias is not None:
        h = h + model.propagation.bias
    h = F.dropout(F.relu(h), p=model.dropout, training=model.training)
    return (h[torch.as_tensor(positions, device=device)], len(ids))

def head_logits(head, z, adjacency, roots, rng, fanout=10):
    (ids, layers, positions) = blocks(adjacency, roots, 2, fanout, rng)
    device = next(head.parameters()).device
    h = torch.as_tensor(np.array(z[ids], copy=True), device=device)
    h = aggregate(head.first.linear(h), layers[0]) + head.first.bias
    h = F.dropout(F.relu(h), p=head.dropout, training=head.training)
    h = aggregate(head.second.linear(h), layers[1]) + head.second.bias
    return (h[torch.as_tensor(positions, device=device)], len(ids))
