"""Sampled classifier-hidden MMD, after joint source graph evolution.

Frozen Z inputs retain gradients through the shared head's first layer.
Target root IDs select adaptation nodes; historical neighbors are context only.
"""
import numpy as np
import torch
import torch.nn.functional as F
from gdabench.sampling import blocks, aggregate
from model.pp_training import rbf_mmd
from experiment.runtime import isolated_rng


def head_logits_features(head, z, adjacency, roots, rng, fanout=10):
    ids, layers, positions = blocks(adjacency, roots, 2, fanout, rng)
    device = next(head.parameters()).device
    x = torch.as_tensor(np.array(z[ids], copy=True), device=device)
    hidden = F.relu(aggregate(head.first.linear(x), layers[0]) + head.first.bias)
    # First block also includes second-layer neighbors. Only center features
    # enter MMD; neighbors must not be counted as additional domain samples.
    center = np.searchsorted(layers[0][1], np.asarray(roots))
    features = hidden[torch.as_tensor(center, device=device)]
    dropped = F.dropout(hidden, head.dropout, head.training)
    logits = aggregate(head.second.linear(dropped), layers[1]) + head.second.bias
    return logits[torch.as_tensor(positions, device=device)], features, len(ids)


def head_features(head, z, adjacency, roots, rng, fanout=10):
    ids, layers, positions = blocks(adjacency, roots, 1, fanout, rng)
    device = next(head.parameters()).device
    x = torch.as_tensor(np.array(z[ids], copy=True), device=device)
    hidden = F.relu(aggregate(head.first.linear(x), layers[0]) + head.first.bias)
    return hidden[torch.as_tensor(positions, device=device)]


def generated_target_mmd(head, generated_features, target_z, target_adj,
                         target_roots, args, cfg, step):
    if args.encoder_mmd_weight <= 0:
        return generated_features.new_zeros(())
    # Step-keyed RNG does not need extra mutable checkpoint state. No labels.
    seed = cfg['seed'] + 400000 + step
    rng = np.random.default_rng(seed)
    count = min(cfg['batch'], args.encoder_mmd_max_samples, len(target_roots))
    chosen = rng.choice(len(target_roots), count, replace=False)
    target_features = head_features(head, target_z, target_adj,
        np.asarray(target_roots)[chosen], rng, cfg['fanout'])
    with isolated_rng(seed):
        return rbf_mmd(generated_features, target_features, args.encoder_mmd_max_samples)
