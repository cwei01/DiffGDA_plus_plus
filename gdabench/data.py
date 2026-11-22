import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from gdabench.common import digest
from gdabench.runtime import save
PROJECT=Path(__file__).resolve().parents[1]
BASE=DATA=PROJECT/'data/GDABench-ArxivTemporal/v1'
RESULTS=PROJECT/'runs/gdabench'
BOUNDS={'A':(1950,2016),'B':(2016,2018),'C':(2018,2020)}
EXPECTED={'A':69499,'B':51241,'C':48603}
TASKS={'A_to_B':('A','B'),'A_to_C':('A','C'),'B_to_C':('B','C')}

def pair_path(task, role):
    (src, tgt) = TASKS[task]
    return DATA / (src if role == 'source' else tgt) / 'pairs.npy'

def training_view(task):
    (src, tgt) = TASKS[task]
    protocol = json.loads((DATA / 'protocol.json').read_text())
    for name in (src, tgt):
        files = ['features.npy', 'adjacency.npz', 'pairs.npy', 'ids.npy', 'roots.npy']
        if name == src:
            files.append('source_labels.npy')
        for filename in files:
            key = f'{name}/{filename}'
            if digest(DATA / key) != protocol['hashes'][key]:
                raise ValueError('Input changed: ' + key)

    def graph(name):
        folder = DATA / name
        return dict(x=np.load(folder / 'features.npy', mmap_mode='r'), adjacency=load_npz(folder / 'adjacency.npz'), nodes=protocol['domains'][name]['nodes'], y=None)
    (source, target) = (graph(src), graph(tgt))
    source['y'] = np.load(DATA / src / 'source_labels.npy')
    if not np.array_equal(np.unique(source['y']), np.arange(40)):
        raise ValueError('Source does not cover all 40 classes')
    target['adaptation_roots'] = np.load(DATA / tgt / 'roots.npy')
    return dict(source=source, target=target, classes=40)

def audit():
    protocol = json.loads((DATA / 'protocol.json').read_text())
    for (name, sha) in protocol['hashes'].items():
        if digest(DATA / name) != sha:
            raise ValueError('Frozen data hash mismatch: ' + name)
    allids = []
    for (name, (lower, upper)) in BOUNDS.items():
        ids = np.load(DATA / name / 'ids.npy')
        allids.append(ids)
        years = np.load(DATA / name / 'years.npy')
        pairs = np.load(DATA / name / 'pairs.npy')
        adj = load_npz(DATA / name / 'adjacency.npz')
        assert len(ids) == EXPECTED[name] and np.all((years > lower) & (years <= upper))
        assert (adj != adj.T).nnz == 0 and (not adj.diagonal().any())
        assert np.all(pairs[0] < pairs[1]) and np.unique(pairs, axis=1).shape == pairs.shape
        assert adj.nnz == 2 * pairs.shape[1]
    assert len(np.unique(np.concatenate(allids))) == 169343
    for task in TASKS:
        view = training_view(task)
        assert view['target']['y'] is None
    result = dict(status='passed', domains=protocol['domains'], protocol_sha256=digest(DATA / 'protocol.json'))
    save(DATA / 'audit.json', result)
    return result
