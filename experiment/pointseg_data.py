"""Versioned, label-independent PointSegDA-derived single-graph tasks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

PROTOCOL = 'pointsegda-derived-transductive-v1'
DOMAINS = ('adobe', 'faust', 'mit', 'scape')
CLASSES = tuple(range(8))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def graph_from_array(array, seed, max_points=1024, k=16):
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[1] != 4 or len(array) < 2:
        raise ValueError('Expected finite [N,4] XYZ plus labels, N>=2')
    if not np.isfinite(array).all():
        raise ValueError('Non-finite raw sample')
    labels = array[:, 3]
    if not np.equal(labels, labels.astype(np.int64)).all() or not np.isin(labels, np.arange(1, 9)).all():
        raise ValueError('Official labels must be integers in 1..8')
    if max_points < 2 or k < 1:
        raise ValueError('max_points>=2 and k>=1 required')
    # Normalize the full XYZ, then sample; neither operation reads class labels.
    xyz = array[:, :3].astype(np.float64)
    center = xyz.mean(0)
    radius = np.linalg.norm(xyz - center, axis=1).max()
    if radius <= 0:
        raise ValueError('Degenerate zero-radius point cloud')
    indices = np.sort(np.random.default_rng(seed).choice(len(array), min(len(array), max_points), replace=False))
    x = ((xyz - center) / radius)[indices].astype(np.float32)
    distances = ((x[:, None, :].astype(np.float64) - x[None, :, :]) ** 2).sum(2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1, kind='stable')[:, :min(k, len(x) - 1)]
    rows = np.repeat(np.arange(len(x)), nearest.shape[1])
    cols = nearest.reshape(-1)
    keys = np.unique(np.concatenate((rows * len(x) + cols, cols * len(x) + rows)))
    edges = np.stack((keys // len(x), keys % len(x)))
    graph = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edges).long(),
                 y=torch.from_numpy(labels[indices].astype(np.int64) - 1))
    metadata = {'point_indices': indices.tolist(), 'normalization_center': center.tolist(),
                'normalization_radius': float(radius), 'point_seed': seed,
                'raw_nodes': len(array), 'nodes': len(x), 'directed_edge_entries': len(keys),
                'class_counts': np.bincount(graph.y.numpy(), minlength=8).tolist()}
    return graph, metadata


def build_dataset(raw_root, output, seed=20261019, groups=5, max_points=1024, k=16):
    raw_root, output = Path(raw_root), Path(output)
    if output.exists():
        raise FileExistsError('Use a new derived output directory')
    # Inventory before choosing samples; official train only, never val/test.
    inventory, choices, duplicates = {}, {}, []
    seen = {}
    for domain in DOMAINS:
        files = sorted((raw_root / domain / 'train').glob('*.npy'))
        if len(files) < groups:
            raise ValueError(f'{domain}: need {groups} train samples, got {len(files)}')
        inventory[domain] = {part: len(list((raw_root / domain / part).glob('*.npy')))
                             for part in ('train', 'val', 'test')}
        selected = np.random.default_rng(seed + DOMAINS.index(domain)).choice(len(files), groups, replace=False)
        choices[domain] = [files[i] for i in selected]
    records = []
    output.mkdir(parents=True)
    for domain in DOMAINS:
        for group, path in enumerate(choices[domain]):
            array = np.load(path, allow_pickle=False)
            xyz = array[:, :3].astype('<f8')
            order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
            shape_hash = hashlib.sha256(xyz[order].tobytes()).hexdigest()
            key = f'{domain}_{group}'
            if shape_hash in seen:
                duplicates.append([seen[shape_hash], key])
            seen[shape_hash] = key
            point_seed = seed + 1000 * (1 + DOMAINS.index(domain)) + group
            graph, metadata = graph_from_array(array, point_seed, max_points, k)
            target = output / (key + '.npz')
            np.savez_compressed(target, x=graph.x.numpy(), y=graph.y.numpy(),
                                edge_index=graph.edge_index.numpy())
            records.append({'id': key, 'domain': domain, 'group': group,
                            'raw_path': path.relative_to(raw_root).as_posix(),
                            'raw_sha256': sha256(path), 'xyz_order_invariant_sha256': shape_hash,
                            'file': target.name, 'sha256': sha256(target), **metadata})
    lookup = {r['id']: r for r in records}
    tasks = []
    for group in range(groups):
        for source in DOMAINS:
            for target in DOMAINS:
                if source == target:
                    continue
                s, t = lookup[f'{source}_{group}'], lookup[f'{target}_{group}']
                supported = [c for c in CLASSES if s['class_counts'][c]]
                missing = [c for c in CLASSES if t['class_counts'][c] and c not in supported]
                tasks.append({'source': s['id'], 'target': t['id'], 'group': group,
                              'closed_set': not missing, 'missing_source_classes': missing,
                              'eight_class_source': len(supported) == 8})
    manifest = {'protocol': PROTOCOL, 'name': 'PointSegDA-derived', 'seed': seed,
                'groups': groups, 'max_points': max_points, 'knn': k,
                'class_values': list(CLASSES), 'raw_label_mapping': '1..8 -> 0..7',
                'label_names': 'Anatomical names not verified; retain official numeric IDs',
                'partition': 'official train only; derived transductive evaluation, not official test',
                'inventory': inventory, 'graphs': records, 'tasks': tasks,
                'exact_xyz_duplicates': duplicates,
                'identity_limitation': 'File/XYZ hashes detect exact duplicates, not shared human identity or near duplicates'}
    write_json(output / 'manifest.json', manifest)
    return manifest


def load_graph(directory, graph_id):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['protocol'] != PROTOCOL:
        raise ValueError('Unsupported derived protocol')
    record = next(r for r in manifest['graphs'] if r['id'] == graph_id)
    path = directory / record['file']
    if sha256(path) != record['sha256']:
        raise ValueError('Graph checksum mismatch: ' + graph_id)
    with np.load(path, allow_pickle=False) as data:
        graph = Data(**{name: torch.from_numpy(data[name].copy()) for name in ('x', 'y', 'edge_index')})
    return graph, record


def segmentation_metrics(labels, logits):
    labels = torch.as_tensor(labels).detach().cpu().long()
    logits = torch.as_tensor(logits).detach().cpu()
    if logits.shape != (labels.numel(), 8) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Expected finite [N,8] logits')
    prediction = logits.argmax(1)
    if labels.numel() == 0 or bool(((labels < 0) | (labels > 7)).any()):
        raise ValueError('Expected nonempty labels in 0..7')
    confusion = torch.bincount(labels * 8 + prediction, minlength=64).reshape(8, 8).double()
    tp = confusion.diag()
    true, predicted = confusion.sum(1), confusion.sum(0)
    union = true + predicted - tp
    iou = [float(tp[c] / union[c]) if union[c] else None for c in CLASSES]
    present = [value for value in iou if value is not None]
    return {'micro_f1': float(tp.sum() / labels.numel()),
            'macro_f1': float((2 * tp / (true + predicted).clamp_min(1)).mean()),
            'class_iou': iou, 'miou': sum(present) / len(present),
            'iou_valid_classes': len(present), 'confusion_matrix': confusion.long().tolist()}
