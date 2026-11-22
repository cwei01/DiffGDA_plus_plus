"""Standard-library-only protocol definitions shared by training and orchestration."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'gda-target-best-v2'
SELECTION = 'target_label_assisted_best_micro'
SELECTION_PROTOCOLS = {
    'diffgda': 'legacy_diffusion_round_target_micro_f1',
    'diffgda_pp': 'diffgda_pp_training_trajectory_target_micro_f1',
}
BASELINE_METHODS = ('gcn', 'gat', 'gin', 'dane', 'udagcn', 'adagcn',
                    'strurw', 'grade', 'pairalign', 'graphalign', 'a2gnn',
                    'ggda', 'tdss', 'dgsda', 'gaa')
METHODS = ('diffgda', 'diffgda_pp')
PAPER_METHODS = (*METHODS, *BASELINE_METHODS)
FAMILIES = {
    'Citation': ('ACMv9', 'Citationv1', 'DBLPv7'),
    'Airport': ('USA', 'BRAZIL', 'EUROPE'),
    'Blog': ('Blog1', 'Blog2'),
}
SCENARIOS = [f'{s}:{t}' for names in FAMILIES.values() for s in names for t in names if s != t]
SEARCH_SEEDS = (1, 2, 3)
FINAL_SEEDS = (4, 5, 6, 7, 8)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def fingerprint():
    digest = hashlib.sha256()
    paths = [ROOT / 'train.py']
    for directory in ('experiment', 'model', 'models', 'utilss', 'config'):
        paths += list((ROOT / directory).rglob('*.py')) + list((ROOT / directory).rglob('*.yaml'))
    for path in sorted(set(paths)):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()

def identity(config, code):
    return hashlib.sha256(canonical({'protocol': VERSION, 'code': code, 'config': config}).encode()).hexdigest()
