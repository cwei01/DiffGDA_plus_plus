import json,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score,confusion_matrix


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda : f.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def metrics(labels, logits, classes=165):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits)
    if not len(labels) or (labels < 0).any() or (labels >= classes).any():
        raise ValueError('Invalid evaluator labels')
    if logits.shape != (len(labels), classes) or not np.isfinite(logits).all():
        raise ValueError('Invalid prediction shape or values')
    prediction = logits.argmax(1)
    micro = float(f1_score(labels, prediction, average='micro'))
    accuracy = float((labels == prediction).mean())
    if abs(micro - accuracy) > 1e-12:
        raise ValueError('Micro-F1 != accuracy')
    return dict(macro_f1=float(f1_score(labels, prediction, labels=np.arange(classes), average='macro', zero_division=0)), micro_f1=micro, accuracy=accuracy, present_class_macro_f1=float(f1_score(labels, prediction, labels=np.unique(labels), average='macro', zero_division=0)), per_class_f1=f1_score(labels, prediction, labels=np.arange(classes), average=None, zero_division=0).tolist(), class_support=np.bincount(labels, minlength=classes).tolist(), predicted_counts=np.bincount(prediction, minlength=classes).tolist(), confusion_matrix=confusion_matrix(labels, prediction, labels=np.arange(classes)).tolist())

def args_for(seed, device, epochs=150):
    from experiment.cli import parse_args
    return parse_args(['--method', 'diffgda_pp', '--source', 'ACMv9', '--target', 'Citationv1', '--device', device, '--seed', str(seed), '--deterministic', 'strict', '--artifact-policy', 'full', '--encoder-epochs', str(epochs), '--flow-epochs', str(epochs), '--head-epochs', str(epochs), '--encoder-lr', '.001', '--head-lr', '.001', '--flow-lr', '.001', '--hidden', '64', '--s-prop', '0', '--t-prop', '5', '--transport-strength', '.5', '--head-source-anchor-weight', '0', '--joint-cost-mode', 'legacy_normalized', '--flow-state', 'joint'])

def save(path,value):
    from gdabench.runtime import save as atomic_save
    return atomic_save(path,value)
