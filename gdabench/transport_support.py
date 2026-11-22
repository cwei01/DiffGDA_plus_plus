import torch
@torch.no_grad()
def hnsw_support(source, target, k=64, seed=20261023, resource_check=None):
    """Union of bidirectional ANN support; insertion and query order are fixed."""
    from gdabench.faiss_bridge import PYTHON, knn
    import numpy as np
    if min(len(source), len(target)) < 1:
        raise ValueError('Empty domain')
    (forward, _) = knn(target, source, k, True, seed, resource_check=resource_check)
    (reverse, _) = knn(source, target, k, True, seed, resource_check=resource_check)
    if resource_check is not None:
        resource_check()
    nt = len(target)
    keys = np.unique(np.concatenate(((np.arange(len(source), dtype=np.int64)[:, None] * nt + forward).reshape(-1), (reverse * nt + np.arange(nt, dtype=np.int64)[:, None]).reshape(-1))))
    return (keys // nt, keys % nt)
