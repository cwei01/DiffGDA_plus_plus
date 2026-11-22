"""Complete, resumable sparse PP on the pinned GDABench Arxiv domains.

Training never opens evaluation labels. Only diagnose(), after frozen prediction,
does so. Probe outputs are deliberately not accepted as training checkpoints.
"""

from experiment.efficiency import measured as _eff_measured, stage as _eff_stage
import copy
import gc
import hashlib
import json
import random
import shutil
import time
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from gdabench.data import BASE, DATA, RESULTS, TASKS, EXPECTED, training_view, pair_path
from gdabench.common import digest
from gdabench.runtime import save, Monitor, RunLock, process_identity, l2
from gdabench.common import args_for, metrics
from gdabench.graph_ops import ResourceLimit, sample_graph, encode_full, predict_root_probabilities
from gdabench.structure import structure_support
from gdabench.sampling import encode as sampled_encoding
from gdabench.flow import flow_step as sampled_flow_step
from experiment.runtime import configure_determinism, seed_everything, isolated_rng
from model.pp_components import SharedA2GNN, encode_domain
from gdabench.head_mmd import head_logits_features, generated_target_mmd
from gdabench.coverage import source_weights, supervised_loss
from gdabench.source_order import CoverageOrder
from gdabench.sampling import head_logits
from gdabench.transport_support import hnsw_support
from gdabench.transport import sparse_refine
from gdabench.assignment import GlobalAssignment, global_edge_weights
from model.joint_graph import JointVelocity, GraphState, GraphHead
from gdabench.integrate import integrate, normalized_graph, convolution
ROOT = RESULTS / 'pp_generated_hidden_mmd_v2'
VERSION = 'gdabench-arxiv-pp-block-generated-hidden-mmd-v4'

def configuration(candidate='PP', task='A_to_B'):
    raise RuntimeError('Use run_gdabench.py to load frozen task parameters')

def arguments(cfg, device):
    args = args_for(cfg['seed'], device, cfg['encoder_updates'])
    for key in ('encoder_lr', 'flow_lr', 'head_lr', 's_prop', 't_prop', 'transport_strength'):
        setattr(args, key, cfg[key])
    args.encoder_mmd_weight = cfg['mmd_weight']
    args.joint_encoder_soft_weight = cfg['feedback_weight']
    for key in ('hidden', 'dropout', 'head_source_anchor_weight', 'joint_epsilon', 'joint_tau', 'joint_label_weight', 'joint_mass_power', 'structure_knn', 'structure_loss_weight', 'edge_logit_delta', 'ode_steps'):
        if key in cfg:
            setattr(args, key, cfg[key])
    args.joint_balanced = bool(cfg.get('joint_balanced', False))
    args.flow_epochs = cfg['flow_updates']
    args.head_epochs = cfg['head_updates']
    if args.head_weighting != 'uniform':
        raise ValueError('This backend implements the frozen uniform head / zero anchor protocol')
    return args

def feedback_steps(updates):
    first = 2 * updates // 3
    return (first, first + (updates - first) // 2)

def feedback_loss(logits, q, reliability, global_mean):
    """Uniform root estimator of the globally reliability-normalized loss."""
    return (reliability * -(q * logits.log_softmax(1)).sum(1)).mean() / max(float(global_mean), 1e-12)

def rng_state(rng):
    return dict(numpy=rng.bit_generator.state, torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], python=random.getstate())

def restore_rng(state, rng):
    rng.bit_generator.state = state['numpy']
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    random.setstate(state['python'])

@_eff_measured("save")
def checkpoint(folder, stage, step, model, optimizer, rng, identity, extra):
    """Immutable numbered snapshots and an atomic latest pointer."""
    dest = folder / 'checkpoints' / stage
    dest.mkdir(parents=True, exist_ok=True)
    serial = len(list(dest.glob(f'{step:06d}_*.pt')))
    path = dest / f'{step:06d}_{serial:03d}.pt'
    temporary = path.with_suffix('.pending')
    with temporary.open('wb') as stream:
        torch.save(dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(rng), identity=identity, extra=extra), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    save(dest / 'latest.json', dict(file=path.name, sha256=digest(path)))

def restore(folder, stage, model, optimizer, rng, identity):
    dest = folder / 'checkpoints' / stage
    pointer = dest / 'latest.json'
    if not pointer.exists():
        return (0, {})
    info = json.loads(pointer.read_text(encoding='utf-8'))
    path = dest / info['file']
    if digest(path) != info['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['identity'] != identity:
        raise ValueError('Checkpoint config/data/code identity mismatch')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'], rng)
    return (state['step'], state['extra'])

def coverage_summary(counts, labels, classes):
    return dict(draws=int(counts.sum()), unique_nodes=int(np.count_nonzero(counts)), unique_fraction=float(np.count_nonzero(counts) / len(counts)), draws_per_class=np.bincount(labels, weights=counts, minlength=classes).astype(np.int64).tolist())

@_eff_measured('sct_refinement', evaluation=False)
def refresh(encoder, view, args, folder, name, previous, monitor):
    pointer = folder / f'{name}.json'
    if pointer.exists():
        info = json.loads(pointer.read_text(encoding='utf-8'))
        for (key, value) in info['hashes'].items():
            if digest(folder / info['directory'] / key) != value:
                raise ValueError('Refresh artifact hash mismatch')
        return folder / info['directory']
    attempt = len(list(folder.glob(name + '_attempt*'))) + 1
    out = folder / f'{name}_attempt{attempt}'
    out.mkdir()
    encoder.eval()
    (source, target) = (view['source'], view['target'])
    started = time.perf_counter()
    monitor.phase = name + '_full_encoding'
    zs = encode_full(encoder, source['x'], source['adjacency'], args.s_prop, out / 'source_z.npy', monitor)
    zall = encode_full(encoder, target['x'], target['adjacency'], args.t_prop, out / 'target_z.npy', monitor)
    roots = target['adaptation_roots']
    zt = np.array(zall[roots], copy=True)
    current = predict_root_probabilities(encoder, zall, roots, monitor)
    prior = current if previous is None else 0.5 * np.load(previous / 'prior.npy') + 0.5 * current
    warm = None if previous is None else torch.from_numpy(np.load(previous / 'q.npy'))
    np.save(out / 'prior.npy', prior)
    np.save(out / 'encoder_probabilities.npy', current)
    monitor.phase = name + '_global_hnsw'
    (rows, cols) = hnsw_support(zs, zt, 64, 20261025, resource_check=monitor.check)
    np.save(out / 'rows.npy', rows)
    np.save(out / 'cols.npy', cols)
    monitor.phase = name + '_global_uot'
    result = sparse_refine(torch.from_numpy(np.asarray(zs)), torch.from_numpy(zt), torch.as_tensor(source['y'].astype(np.int64)), torch.from_numpy(prior), torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long), args, view['classes'], seed=20261025, initial_q=warm, resource_check=monitor.check, directory=out / 'uot_workspace')
    for (filename, key) in [('mass.npy', 'mass'), ('q.npy', 'probabilities'), ('reliability.npy', 'reliability')]:
        np.save(out / filename, result[key].numpy())
    names = ['source_z.npy', 'target_z.npy', 'prior.npy', 'encoder_probabilities.npy', 'rows.npy', 'cols.npy', 'mass.npy', 'q.npy', 'reliability.npy']
    save(pointer, dict(directory=out.name, seconds=time.perf_counter() - started, hashes={n: digest(out / n) for n in names}, history=result['history'], feature_scale=result['feature_scale'], target_roots=len(roots), source_nodes=len(zs), candidate_pairs=len(rows), target_labels_loaded=False, previous=None if previous is None else previous.name))
    return out

@_eff_measured('encoder', evaluation=False)
def train_encoder(view, args, cfg, folder, monitor, identity):
    if view['target']['y'] is not None:
        raise ValueError('Target training labels must be None')
    encoder = SharedA2GNN(cfg['input_dim'], cfg['hidden'], cfg['classes'], args.dropout, args.t_prop, source_steps=args.s_prop).to(args.device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(20261025)
    (start, extra) = restore(folder, 'encoder', encoder, optimizer, rng, identity)
    monitor.restored('encoder', start)
    counts = extra.get('coverage', np.zeros(view['source']['nodes'], dtype=np.int32))
    history = extra.get('history', [])
    previous = folder / extra['feedback'] if extra.get('feedback') else None
    order = CoverageOrder(view['source']['nodes'], 2026102501, extra.get('source_order'))
    target_rng = np.random.default_rng(2026102502)
    if 'target_rng' in extra:
        target_rng.bit_generator.state = extra['target_rng']
    supervision_weights = None
    q = rel = None
    if previous is not None:
        q = np.load(previous / 'q.npy')
        rel = np.load(previous / 'reliability.npy')
    roots = view['target']['adaptation_roots']
    n = len(roots)
    (first, second) = feedback_steps(cfg['encoder_updates'])
    panel_rng = np.random.default_rng(20261025)
    panel = np.unique(np.r_[[np.flatnonzero(view['source']['y'] == c)[0] for c in np.unique(view['source']['y'])], panel_rng.choice(view['source']['nodes'], min(2048, view['source']['nodes']), replace=False)])
    np.save(folder / 'source_panel.npy', panel)

    def store(step):
        checkpoint(folder, 'encoder', step, encoder, optimizer, rng, identity, dict(coverage=counts, history=history, feedback=None if previous is None else previous.name, source_order=order.state(), target_rng=target_rng.bit_generator.state))
    for step in range(start, cfg['encoder_updates']):
        if step in (first, second):
            store(step)
            saved_rng = rng_state(rng)
            previous = refresh(encoder, view, args, folder, f'feedback_{step}', previous, monitor)
            restore_rng(saved_rng, rng)
            q = np.load(previous / 'q.npy')
            rel = np.load(previous / 'reliability.npy')
            store(step)
        monitor.phase = f'encoder_{step + 1}'
        monitor.check()
        encoder.train()
        optimizer.zero_grad()
        si = order.take(cfg['batch'])
        ti = target_rng.choice(n, min(cfg['batch'], n), replace=False)
        (zs, scontext) = sampled_encoding(encoder, 'source', view['source'], si, args.s_prop, rng, cfg['fanout'])
        ce = supervised_loss(encoder.classifier(zs), torch.as_tensor(view['source']['y'][si].astype(np.int64), device=args.device), supervision_weights)
        tcontext = 0
        loss = ce
        fb = None
        if q is not None:
            with isolated_rng(cfg['seed'] + 200000 + step):
                (zt, tcontext) = sampled_encoding(encoder, 'target', view['target'], roots[ti], args.t_prop, rng, cfg['fanout'])
        if q is not None:
            fb = feedback_loss(encoder.classifier(zt), torch.as_tensor(q[ti], device=args.device), torch.as_tensor(rel[ti], device=args.device), rel.mean())
            ramp = min(1.0, (step - first + 1) / min(30, cfg['encoder_updates'] - first))
            loss = loss + args.joint_encoder_soft_weight * ramp * fb
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite encoder loss')
        loss.backward()
        optimizer.step()
        np.add.at(counts, si, 1)
        monitor.steps = step + 1
        history.append(dict(update=step + 1, ce=float(ce.detach()), mmd_enabled=False, feedback=None if fb is None else float(fb.detach()), loss=float(loss.detach()), source_context=scontext, target_context=tcontext))
        if (step + 1) % 50 == 0 or step + 1 == cfg['encoder_updates']:
            encoder.eval()
            with torch.no_grad(), isolated_rng(20261025):
                (pz, _) = sampled_encoding(encoder, 'source', view['source'], panel, args.s_prop, np.random.default_rng(20261025), cfg['fanout'])
                pred = encoder.classifier(pz).argmax(1).cpu().numpy()
            history[-1]['source_panel_accuracy'] = float((pred == view['source']['y'][panel]).mean())
            store(step + 1)
            save(folder / 'encoder_history.json', history)
            save(folder / 'encoder_coverage.json', coverage_summary(counts, view['source']['y'], cfg['classes']))
    if not np.all(counts == cfg['source_passes']):
        raise ValueError('Encoder complete-pass coverage failed')
    encoder.eval()
    saved_rng = rng_state(rng)
    final = refresh(encoder, view, args, folder, 'final_coupling', previous, monitor)
    restore_rng(saved_rng, rng)
    return (encoder, final)

def adjacency(pairs, values, n):
    (a, b) = pairs
    return coo_matrix((np.r_[values, values], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()

@_eff_measured('flow_and_head_pipeline', evaluation=False)
def train_remaining(view, args, cfg, folder, final, monitor, identity):
    zs = np.load(final / 'source_z.npy', mmap_mode='r')
    zall = np.load(final / 'target_z.npy', mmap_mode='r')
    roots = view['target']['adaptation_roots']
    zt = np.array(zall[roots], copy=True)
    support = folder / 'structure'
    support.mkdir(exist_ok=True)
    if not (support / 'structure_support.json').exists():
        structure_support(zs, np.load(pair_path(cfg['task'], 'source'), mmap_mode='r'), support, monitor, args.structure_knn)
    pairs = np.load(support / 'pairs.npy', mmap_mode='r')
    values = np.load(support / 'source_values.npy', mmap_mode='r')
    plan = GlobalAssignment(np.load(final / 'rows.npy', mmap_mode='r'), np.load(final / 'cols.npy', mmap_mode='r'), np.load(final / 'mass.npy', mmap_mode='r'), len(zs), folder / 'plan', power=args.joint_mass_power, check=monitor.check)
    ew = global_edge_weights(pairs, plan.weights, plan.valid, folder / 'plan', monitor.check)
    adj = adjacency(pairs, np.ones(len(values), dtype=np.float32), len(zs))
    seed_everything(cfg['seed'] + 300000)
    flow = JointVelocity(cfg['hidden'], args.velocity_width, args.velocity_depth, cfg['classes']).to(args.device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=args.flow_lr, weight_decay=args.flow_weight_decay)
    rng = np.random.default_rng(20261026)
    (start, extra) = restore(folder, 'flow', flow, optimizer, rng, identity)
    history = extra.get('history', [])
    monitor.restored('flow', start)
    with _eff_stage('velocity'):
        for step in range(start, cfg['flow_updates']):
            monitor.phase = f'flow_{step + 1}'
            monitor.check()
            optimizer.zero_grad()
            (nl, el, details) = sampled_flow_step(flow, zs, zt, view['source']['y'], pairs, values, adj, plan, ew, view['target']['adjacency'], roots, rng, cfg['batch'], args.edge_logit_delta, cfg['fanout'])
            loss = nl + args.structure_loss_weight * el
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite flow loss')
            loss.backward()
            optimizer.step()
            monitor.steps = step + 1
            history.append(dict(update=step + 1, node_loss=float(nl.detach()), edge_loss=float(el.detach()), **details))
            if (step + 1) % 50 == 0 or step + 1 == cfg['flow_updates']:
                checkpoint(folder, 'flow', step + 1, flow, optimizer, rng, identity, dict(history=history))
                save(folder / 'flow_history.json', history)
        valid = plan.valid.copy()
        del adj, plan, ew, optimizer
        gc.collect()
    graph = GraphState(torch.from_numpy(np.array(pairs, copy=True)), torch.from_numpy(np.array(values, copy=True)), torch.from_numpy(valid), args.edge_logit_delta)
    monitor.phase = 'heun'
    monitor.check()
    if shutil.disk_usage(folder).free < 16 * 1024 ** 3:
        raise ResourceLimit('Need at least 16 GiB free disk for compact Heun')
    flow.cpu().eval()
    (statez, stateb, nfe) = integrate(flow, zs, view['source']['y'], graph, args.ode_steps, folder / 'integration', check=monitor.check, compact=True)
    moved = np.lib.format.open_memmap(folder / 'moved_z.npy', mode='w+', dtype=np.float32, shape=zs.shape)
    mixed = np.lib.format.open_memmap(folder / 'moved_values.npy', mode='w+', dtype=np.float32, shape=values.shape)
    for begin in range(0, len(zs), 8192):
        monitor.check()
        sl = slice(begin, begin + 8192)
        moved[sl] = zs[sl] + cfg['transport_strength'] * (statez[sl] - zs[sl])
    for begin in range(0, len(values), 32768):
        monitor.check()
        sl = slice(begin, begin + 32768)
        decoded = np.where(graph.active.numpy()[sl], torch.from_numpy(np.array(stateb[sl], copy=True)).sigmoid().numpy(), values[sl])
        mixed[sl] = values[sl] + cfg['transport_strength'] * (decoded - values[sl])
    moved.flush()
    mixed.flush()
    save(folder / 'evolution.json', dict(nfe=nfe, heun_steps=args.ode_steps, nodes=len(zs), candidate_edges=len(values), transport_strength=cfg['transport_strength']))
    node_sq = 0.0
    edge_abs = 0.0
    edge_large = 0
    for start in range(0, len(zs), 8192):
        node_sq += float(np.square(moved[start:start + 8192] - zs[start:start + 8192]).sum(dtype=np.float64))
    for start in range(0, len(values), 32768):
        change = np.abs(mixed[start:start + 32768] - values[start:start + 32768])
        edge_abs += float(change.sum(dtype=np.float64))
        edge_large += int((change > 0.1).sum())
    save(folder / 'evolution_diagnostics.json', dict(node_rms_displacement=(node_sq / zs.size) ** 0.5, mean_abs_edge_change=edge_abs / len(values), fraction_edges_change_over_point1=edge_large / len(values)))
    del graph, flow, statez, stateb
    gc.collect()
    seed_everything(cfg['seed'] + 400000)
    with _eff_stage('classification_and_finalization'):
        head = GraphHead(cfg['hidden'], cfg['classes'], args.dropout).to(args.device)
        optimizer = torch.optim.Adam(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
        adj = adjacency(pairs, mixed, len(zs))
        rng = np.random.default_rng(20262025)
        original_adj = view['source']['adjacency']
        active = np.load(folder / 'plan/valid.npy') if (folder / 'plan/valid.npy').exists() else None
        mass = np.load(final / 'mass.npy', mmap_mode='r')
        effective_anchor = args.head_source_anchor_weight if np.any(mass > 0) else 1.0
        (start, extra) = restore(folder, 'head', head, optimizer, rng, identity)
        monitor.restored('head', start)
        history = extra.get('history', [])
        counts = extra.get('coverage', np.zeros(len(zs), dtype=np.int32))
        order = CoverageOrder(len(zs), 2026102503, extra.get('source_order'))
        weights = source_weights(view['source']['y'], cfg['classes']) if cfg['head_weighting'] == 'inverse_sqrt' else None
        for step in range(start, cfg['head_updates']):
            monitor.phase = f'head_{step + 1}'
            monitor.check()
            head.train()
            optimizer.zero_grad()
            roots_batch = order.take(cfg['batch'])
            np.add.at(counts, roots_batch, 1)
            (logits, generated_features, context) = head_logits_features(head, moved, adj, roots_batch, rng, cfg['fanout'])
            loss = supervised_loss(logits, torch.as_tensor(view['source']['y'][roots_batch].astype(np.int64), device=logits.device), weights)
            if effective_anchor > 0:
                (original_logits, _) = head_logits(head, zs, original_adj, roots_batch, rng, cfg['fanout'])
                original_ce = F.cross_entropy(original_logits, torch.as_tensor(view['source']['y'][roots_batch].astype(np.int64), device=logits.device))
                loss = (1 - effective_anchor) * loss + effective_anchor * original_ce
            details = dict(context_nodes=context, roots=len(roots_batch), normalization='global_degree_layer_blocks', source_anchor=effective_anchor)
            mmd = generated_target_mmd(head, generated_features, zall, view['target']['adjacency'], roots, args, cfg, step)
            loss = loss + args.encoder_mmd_weight * mmd
            details.update(mmd=float(mmd.detach()), mmd_weighted_loss=float((args.encoder_mmd_weight * mmd).detach()), mmd_enabled=args.encoder_mmd_weight > 0, mmd_alignment_stage='classification_generated_vs_target_hidden')
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite head loss')
            loss.backward()
            optimizer.step()
            monitor.steps = step + 1
            history.append(dict(update=step + 1, loss=float(loss.detach()), **details))
            if (step + 1) % 50 == 0 or step + 1 == cfg['head_updates']:
                checkpoint(folder, 'head', step + 1, head, optimizer, rng, identity, dict(history=history, coverage=counts, source_order=order.state()))
                save(folder / 'head_history.json', history)
                save(folder / 'head_coverage.json', coverage_summary(counts, view['source']['y'], cfg['classes']))
        if not np.all(counts == cfg['source_passes']):
            raise ValueError('Head complete-pass coverage failed')
        del adj, optimizer
        gc.collect()
        head.cpu().eval()
        target_pairs = np.load(pair_path(cfg['task'], 'target'), mmap_mode='r')
        norm = normalized_graph(target_pairs, np.ones(target_pairs.shape[1], dtype=np.float32), view['target']['nodes'])
        monitor.phase = 'full_target_prediction'
        h = convolution(head.first, zall, norm, folder / 'prediction', 'h1', activation='relu', check=monitor.check)
        logits = convolution(head.second, h, norm, folder / 'prediction', 'logits', check=monitor.check)
        frozen = np.array(logits[roots], copy=True)
        if frozen.shape != (len(roots), cfg['classes']) or not np.isfinite(frozen).all():
            raise ValueError('Invalid full target logits')
        np.save(folder / 'final_logits.npy', frozen)
        checkhead = GraphHead(cfg['hidden'], cfg['classes'], args.dropout).cpu().eval()
        ptr = json.loads((folder / 'checkpoints' / 'head' / 'latest.json').read_text(encoding='utf-8'))
        restored = torch.load(folder / 'checkpoints' / 'head' / ptr['file'], map_location='cpu', weights_only=False)
        checkhead.load_state_dict(restored['model'])
        for (left, right) in zip(head.state_dict().values(), checkhead.state_dict().values()):
            if not torch.equal(left, right):
                raise ValueError('Restored head weights differ')
        h2 = convolution(checkhead.first, zall, norm, folder / 'replay', 'h1', activation='relu', check=monitor.check)
        l2 = convolution(checkhead.second, h2, norm, folder / 'replay', 'logits', check=monitor.check)
        if not np.allclose(frozen, l2[roots], atol=5e-05, rtol=0.0001) or not np.array_equal(frozen.argmax(1), l2[roots].argmax(1)):
            raise ValueError('Checkpoint prediction replay mismatch')
        save(folder / 'frozen.json', dict(final_logits_sha256=digest(folder / 'final_logits.npy'), target_roots=len(roots), target_context_nodes=view['target']['nodes'] - len(roots), classes=cfg['classes'], target_labels_loaded=False, replay_passed=True, final_coupling=final.name))

@_eff_measured('evaluation', evaluation=False)
def diagnose(folder):
    folder = Path(folder)
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    frozen = json.loads((folder / 'frozen.json').read_text(encoding='utf-8'))
    if digest(folder / 'final_logits.npy') != frozen['final_logits_sha256']:
        raise ValueError('Frozen logits changed')
    dest = folder / 'diagnostics.json'
    if cfg['candidate'] == 'smoke':
        result = dict(final={'macro_f1': None}, evaluation_events=0, note='Smoke logits verified; target labels not evaluated')
        save(dest, result)
        return result
    if dest.exists():
        return json.loads(dest.read_text(encoding='utf-8'))
    labels = np.load(DATA / 'evaluation' / (TASKS[cfg['task']][1] + '_labels.npy'))
    final = folder / frozen['final_coupling']
    source_labels = np.load(DATA / TASKS[cfg['task']][0] / 'source_labels.npy')
    classes = cfg['classes']
    constant = np.full(len(labels), np.bincount(source_labels, minlength=classes).argmax(), dtype=np.int64)
    reference = np.zeros((len(labels), classes), dtype=np.float32)
    reference[np.arange(len(labels)), constant] = 1
    result = dict(final=metrics(labels, np.load(folder / 'final_logits.npy'), classes), encoder=metrics(labels, np.load(final / 'encoder_probabilities.npy'), classes), coupling=metrics(labels, np.load(final / 'q.npy'), classes), constant=metrics(labels, reference, classes), selection='fixed_budget_final_once', evaluation_events=1, note='Encoder and coupling are auxiliary diagnostics; only final is the PP score')
    save(dest, result)
    return result

def elapsed_campaign():
    return sum((json.loads(p.read_text()).get('seconds', 0.0) for p in ROOT.glob('*/sessions/*.json')))

def implementation_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / 'gdabench').glob('*.py')) + [root / 'model/pp_training.py', root / 'model/joint_graph.py', root / 'model/pp_components.py', root / 'model/sct.py', root / 'experiment/cli.py', root / 'experiment/runtime.py']
    return {str(p.relative_to(root)): digest(p) for p in paths}

def estimate_next(cfg):
    return None

def _train(candidate='PP', task='A_to_B', resume=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    validation = ROOT / 'validation.json'
    if not validation.exists() or json.loads(validation.read_text(encoding='utf-8'))['status'] != 'passed':
        raise RuntimeError('Run validate before complete training')
    if json.loads(validation.read_text(encoding='utf-8')).get('code') != implementation_hashes():
        raise RuntimeError('Implementation changed since validation')
    cfg = configuration(candidate, task)
    folder = ROOT / f"{task}_{cfg['candidate']}"
    if folder.exists() and (not resume):
        raise FileExistsError('Existing run preserved; use --resume')
    folder.mkdir(exist_ok=True)
    path = folder / 'config.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != cfg:
        raise ValueError('Frozen configuration mismatch')
    if not path.exists():
        save(path, cfg)
    identity = dict(config=cfg, data=digest(DATA / 'protocol.json'), code=implementation_hashes())
    ip = folder / 'identity.json'
    if ip.exists() and json.loads(ip.read_text(encoding='utf-8')) != identity:
        raise ValueError('Code/data identity changed; use a new run version')
    if not ip.exists():
        save(ip, identity)
    consumed = elapsed_campaign()
    estimate = None
    save(folder / 'budget_estimate.json', dict(consumed_seconds=consumed, estimated_next_seconds=estimate, budget_seconds=None, based_on='new graph; runtime measured independently; no cumulative time cap'))
    configure_determinism('strict')
    torch.set_num_threads(4)
    seed_everything(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    device = cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    args = arguments(cfg, device)
    save(folder / 'effective_arguments.json', vars(args))
    save(folder / 'provenance.json', dict(version=VERSION, training_from_scratch=True, inherited_checkpoints=[], supersedes=None, reason='Generated-source versus target hidden MMD at classifier stage; encoder CE and SCT feedback only'))
    sessions = folder / 'sessions'
    sessions.mkdir(exist_ok=True)
    record_path = sessions / f'{time.time_ns()}.json'
    started = time.perf_counter()
    monitor = Monitor(folder / 'progress.json')
    check = monitor.check
    last_ledger = [0.0]

    def bounded():
        check()
        elapsed = time.perf_counter() - started
        if elapsed - last_ledger[0] > 30:
            save(record_path, dict(status='running', seconds=elapsed, candidate=cfg['candidate'], task=task))
            last_ledger[0] = elapsed
    monitor.check = bounded
    status = dict(status='running', candidate=cfg['candidate'], task=task, target_evaluations=0, **process_identity())
    save(folder / 'status.json', status)
    try:
        view = training_view(task)
        (encoder, final) = train_encoder(view, args, cfg, folder, monitor, identity)
        encoder.cpu()
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        train_remaining(view, args, cfg, folder, final, monitor, identity)
        score = diagnose(folder)
        status.update(status='completed', target_evaluations=score['evaluation_events'], final_macro_f1=score['final']['macro_f1'])
    except (Exception, KeyboardInterrupt) as error:
        import traceback
        status.update(status='resource_stopped' if isinstance(error, ResourceLimit) else 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', reason=str(error), traceback=traceback.format_exc())
    finally:
        monitor.close()
        status.update(seconds=time.perf_counter() - started, resources=monitor.state())
        save(record_path, status)
        save(folder / 'status.json', status)
    return status

def train(candidate='PP', task='A_to_B', resume=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    with RunLock(ROOT / 'campaign.lock'):
        return _train(candidate, task, resume)
