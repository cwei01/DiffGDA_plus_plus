import numpy as np
from gdabench.common import digest
from gdabench.runtime import save

def structure_support(zs, original, out, monitor, k=10):
    from gdabench.faiss_bridge import knn
    from scipy.sparse import coo_matrix
    n = len(zs)
    monitor.phase = 'structure_raw_l2_hnsw'
    (neighbors, metadata) = knn(zs, zs, k + 1, False, 20261025, resource_check=monitor.check)
    keys = [np.asarray(original[0]) * n + np.asarray(original[1])]
    for start in range(0, n, 8192):
        monitor.check()
        stop = min(start + 8192, n)
        selected = np.asarray(neighbors[start:stop])
        left = np.arange(start, stop)[:, None]
        nonself = selected != left
        keep = nonself & (np.cumsum(nonself, axis=1) <= min(k, n - 1))
        a = np.broadcast_to(left, selected.shape)[keep]
        b = selected[keep]
        keys.append(np.minimum(a, b) * n + np.maximum(a, b))
    monitor.phase = 'structure_support_union'
    merged = np.unique(np.concatenate(keys))
    del keys
    pairs = np.stack((merged // n, merged % n))
    np.save(out / 'pairs.npy', pairs)
    original_keys = np.asarray(original[0]) * n + np.asarray(original[1])
    at = np.searchsorted(original_keys, merged)
    at = np.minimum(at, len(original_keys) - 1)
    values = (original_keys[at] == merged).astype(np.float32)
    np.save(out / 'source_values.npy', values)
    save(out / 'structure_support.json', dict(nodes=n, candidate_edges=len(merged), original_edges=original.shape[1], structure_knn=k, hnsw_metadata=metadata, pairs_sha256=digest(out / 'pairs.npy'), values_sha256=digest(out / 'source_values.npy')))
    return (pairs, values)
