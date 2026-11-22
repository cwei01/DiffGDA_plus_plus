import hashlib
import json
import platform
import time
import traceback
from pathlib import Path
import torch
from experiment.pointseg_data import PROTOCOL, load_graph, segmentation_metrics, sha256, write_json
from experiment.runtime import Evaluator, Timings, configure_determinism, seed_everything

class SegmentationEvaluator(Evaluator):

    def __init__(self, labels, timings, num_classes=8):
        super().__init__(labels, timings, num_classes=num_classes)

    def __call__(self, predictor, prediction, round, epoch):
        super().__call__(predictor, prediction, round=round, epoch=epoch)
        extra = segmentation_metrics(self.labels, self.final_logits)
        self.final.update(extra)
        if self.best['evaluation_index'] == self.final['evaluation_index']:
            self.best.update(extra)

def prepare_graphs(directory, source_id, target_id):
    (source, sr) = load_graph(directory, source_id)
    (target, tr) = load_graph(directory, target_id)
    if sr['group'] != tr['group'] or sr['domain'] == tr['domain']:
        raise ValueError('Use different domains in the same fixed graph group')
    if set(source.y.tolist()) != set(range(8)):
        raise ValueError('Source lacks classes required by fixed eight-class protocol; do not resample')
    labels = target.y.clone()
    target.y = None
    return (source, target, labels, sr, tr)
