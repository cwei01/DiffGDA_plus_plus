from experiment.efficiency import measured as _eff_measured
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

@_eff_measured('save', evaluation=False)
def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)

@_eff_measured('data', evaluation=False)
def split_graphs(direction, limit=None):
    import torch
    graphs = torch.load(BASE / 'graphs.pt', map_location='cpu', weights_only=False)
    selection = load(BASE / 'selection.json')
    (source_domain, target_domain) = direction.split('_')
    source_ids = {r['graph_id'] for r in selection if r['domain'] == source_domain and r['source_role'] == 'train'}
    valid_ids = {r['graph_id'] for r in selection if r['domain'] == source_domain and r['source_role'] == 'holdout'}
    target_ids = {r['graph_id'] for r in selection if r['domain'] == target_domain}
    assert not source_ids & valid_ids and (not source_ids & target_ids) and (not valid_ids & target_ids)
    lists = []
    for (ids, labeled) in ((source_ids, True), (valid_ids, False), (target_ids, False)):
        subset = [g.clone() for g in graphs if int(g.graph_id) in ids]
        if limit:
            generator = random.Random(SEARCH_SEED)
            subset = sorted(generator.sample(subset, min(limit, len(subset))), key=lambda g: int(g.graph_id))
        if not labeled:
            for g in subset:
                g.y = None
        lists.append(subset)
    valid_gold = {int(g.graph_id): int(g.y) for g in graphs if int(g.graph_id) in valid_ids}
    valid_gold = [valid_gold[int(g.graph_id)] for g in lists[1]]
    return (*lists, valid_gold)

def scores(labels, logits):
    import numpy as np
    from sklearn.metrics import f1_score, confusion_matrix
    pred = logits.argmax(1).cpu().numpy() if hasattr(logits, 'cpu') else np.asarray(logits).argmax(1)
    cm = confusion_matrix(labels, pred, labels=[0, 1]).tolist()
    return dict(micro_f1=float(f1_score(labels, pred, labels=[0, 1], average='micro', zero_division=0)), macro_f1=float(f1_score(labels, pred, labels=[0, 1], average='macro', zero_division=0)), class_f1=f1_score(labels, pred, labels=[0, 1], average=None, zero_division=0).tolist(), confusion_matrix=cm)

def predict(model, graphs, device, batch_size=64, domain='target'):
    import torch
    from torch_geometric.loader import DataLoader
    from experiment.runtime import isolated_rng
    modes = [(m, m.training) for m in model.modules()]
    values = []
    ids = []
    with isolated_rng(), torch.no_grad():
        try:
            model.eval()
            for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
                batch = batch.to(device)
                logits = model(batch)
                assert logits.shape == (batch.num_graphs, 2) and torch.isfinite(logits).all()
                values.append(logits.cpu())
                ids.extend(batch.graph_id.cpu().tolist())
        finally:
            for (m, mode) in modes:
                m.training = mode
    return (torch.cat(values), ids)

class HoldoutObserver:

    def __init__(self, graphs, gold, device, domain='source', selection='source_holdout_macro'):
        (self.graphs, self.gold, self.device) = (graphs, gold, device)
        (self.domain, self.selection) = (domain, selection)
        self.history = []
        self.best = None
        self.best_state = None
        self.final = None

    @_eff_measured('evaluation', evaluation=True)
    def __call__(self, model, epoch):
        (logits, ids) = predict(model, self.graphs, self.device, domain=self.domain)
        current = dict(epoch=epoch, **scores(self.gold, logits))
        self.history.append(current)
        self.final = current
        if self.best is None or (current['micro_f1'] > self.best['micro_f1'] if self.selection == 'target_micro' else rank(current) > rank(self.best)):
            self.best = current
            self.best_state = {k: v.detach().cpu().clone() for (k, v) in model.state_dict().items()}
        print('HOLDOUT', epoch, current['macro_f1'], current['micro_f1'], flush=True)
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/Mutagenicity/three_domains"
SEARCH_SEED=20261022

def load(path):
 return json.loads(Path(path).read_text(encoding="utf8"))
