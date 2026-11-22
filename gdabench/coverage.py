"""Checkpointable one-pass supervision and source-only class weights."""
import numpy as np
import torch
import torch.nn.functional as F


class CoverageOrder:
    def __init__(self, nodes, seed, state=None):
        self.order = np.random.default_rng(seed).permutation(nodes)
        self.cursor = 0
        if state is not None:
            self.order = state['order'].copy()
            self.cursor = state['cursor']
            if not np.array_equal(np.sort(self.order), np.arange(nodes)):
                raise ValueError('Invalid saved supervision permutation')
        if not 0 <= self.cursor <= nodes:
            raise ValueError('Invalid saved supervision cursor')

    def take(self, batch):
        if self.cursor == len(self.order):
            raise ValueError('Supervision pass exhausted')
        end = min(self.cursor + batch, len(self.order))
        roots = self.order[self.cursor:end]
        self.cursor = end
        return roots

    def state(self):
        return dict(order=self.order.copy(), cursor=self.cursor)


def source_weights(labels, classes):
    counts = np.bincount(labels, minlength=classes)
    if np.any(counts == 0):
        raise ValueError('Source class missing')
    weights = np.minimum(10., np.sqrt(counts.max() / counts))
    return (weights / (weights * counts).sum() * len(labels)).astype(np.float32)


def supervised_loss(logits, labels, weights=None):
    loss = F.cross_entropy(logits, labels, reduction='none')
    if weights is not None:
        loss = loss * torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)[labels]
    return loss.mean()
