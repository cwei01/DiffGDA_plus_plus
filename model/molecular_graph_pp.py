"""Graph-level PP extension for batches of disconnected molecular graphs.

Classification is performed once per molecule.  SCT couples pooled molecules;
one target molecule is then sampled for each source molecule and atom matching is
restricted to equal atom categories inside that molecule pair.
"""
from __future__ import annotations

from experiment.efficiency import measured as _eff_measured, stage as _eff_stage

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from experiment.runtime import isolated_rng, seed_everything
from model.joint_graph import (GraphHead, GraphState, JointVelocity, candidate_pairs,
                               canonical_pairs, integrate_joint, mix_output,
                               pair_keys, query_edges)
from model.pp_training import rbf_mmd
from model.sct import coupling_weights, diagnostics as sct_diagnostics, finite, refine
from model.transport import robust_scale


class MolecularGIN(nn.Module):
    def __init__(self, input_dim, hidden, classes, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            GINConv(nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden)), eps=0.0, train_eps=False),
            GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden)), eps=0.0, train_eps=False),
        ])
        self.classifier = nn.Linear(hidden, classes)
        self.dropout = dropout

    def atoms(self, data):
        value = data.x
        for conv in self.convs:
            value = F.dropout(F.relu(conv(value, data.edge_index)), self.dropout,
                              self.training)
        return value

    def forward(self, data):
        atoms = self.atoms(data)
        pooled = global_mean_pool(atoms, data.batch)
        return pooled, self.classifier(pooled), atoms


class MolecularPredictor(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.method = "graph_pp"

    def forward(self, data):
        atoms = self.encoder.atoms(data)
        pairs = canonical_pairs(data.edge_index, data.num_nodes)
        atom_logits = self.head(atoms, pairs, atoms.new_ones(pairs.size(1)))
        return global_mean_pool(atom_logits, data.batch)


@torch.no_grad()
def encode_graphs(encoder, graphs, batch_size, device):
    encoder.eval()
    embeddings, logits, ids = [], [], []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        pooled, score, _ = encoder(batch)
        embeddings.append(pooled)
        logits.append(score)
        ids.append(batch.graph_id.reshape(-1))
    return torch.cat(embeddings), torch.cat(logits), torch.cat(ids)


def _ordered_labels(graphs, device):
    return torch.cat([graph.y.reshape(-1) for graph in graphs]).long().to(device)


@_eff_measured('encoder', evaluation=False)
def train_encoder(source_graphs, target_graphs, args, device):
    encoder = MolecularGIN(source_graphs[0].x.size(1), args.hidden, 2,
                           args.dropout).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr,
                                 weight_decay=args.weight_decay)
    source_labels = _ordered_labels(source_graphs, device)
    state = {"q": None, "prior": None}
    feedback = None
    refreshes, history = [], []
    pretrain = 2 * args.encoder_epochs // 3
    second = pretrain + (args.encoder_epochs - pretrain) // 2

    def refresh(epoch):
        zs, source_logits, _ = encode_graphs(encoder, source_graphs, args.batch_size, device)
        zt, target_logits, _ = encode_graphs(encoder, target_graphs, args.batch_size, device)
        current = target_logits.softmax(1)
        prior = current if state["prior"] is None else 0.5 * state["prior"] + 0.5 * current
        result = refine(zs, zt, source_labels, prior, args, 2, initial_q=state["q"])
        state.update(q=result.probabilities.detach(), prior=prior.detach())
        refreshes.append({"epoch_index": epoch, "rounds": len(result.history),
                          "coupling_mass": float(result.coupling.sum())})
        return result.probabilities.detach(), result.reliability.detach()

    for epoch in range(args.encoder_epochs):
        if epoch in (pretrain, second):
            feedback = refresh(epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        source_loader = DataLoader(source_graphs, batch_size=args.batch_size, shuffle=True,
                                   generator=generator)
        target_loader = DataLoader(target_graphs, batch_size=args.batch_size, shuffle=True,
                                   generator=torch.Generator().manual_seed(args.seed + 10000 + epoch))
        target_iterator = iter(target_loader)
        sums = {"ce": 0.0, "mmd": 0.0, "feedback": 0.0, "total": 0.0, "graphs": 0}
        encoder.train()
        for source in source_loader:
            try:
                target = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                target = next(target_iterator)
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad()
            _, source_logits, _ = encoder(source)
            with isolated_rng(args.seed + 200000 + epoch):
                _, target_logits, _ = encoder(target)
            ce = F.cross_entropy(source_logits, source.y.reshape(-1))
            # MMD trains the final classifier on generated-vs-target features.
            mmd = ce.new_zeros(())
            loss = ce
            feedback_loss = loss.new_zeros(())
            if feedback is not None:
                target_indices = target.graph_id.reshape(-1) - 1
                # Dataset graph ids are one-based; q is stored in target_graphs order.
                index_by_id = {int(graph.graph_id): i for i, graph in enumerate(target_graphs)}
                q_indices = torch.tensor([index_by_id[int(value)] for value in target.graph_id],
                                         device=device)
                q, reliability = feedback[0][q_indices], feedback[1][q_indices]
                soft_ce = -(q * F.log_softmax(target_logits, 1)).sum(1)
                feedback_loss = (reliability * soft_ce).sum() / reliability.sum().clamp_min(1e-12)
                ramp = min(1.0, (epoch - pretrain + 1) /
                           min(30, args.encoder_epochs - pretrain))
                loss = loss + args.joint_encoder_soft_weight * ramp * feedback_loss
            finite(loss, "molecular encoder loss").backward()
            optimizer.step()
            count = source.num_graphs
            sums["ce"] += float(ce.detach()) * count
            sums["mmd"] += float(mmd.detach()) * count
            sums["feedback"] += float(feedback_loss.detach()) * count
            sums["total"] += float(loss.detach()) * count
            sums["graphs"] += count
        history.append({"epoch": epoch + 1, "mmd_enabled": False, **{key: value / sums["graphs"]
                                               for key, value in sums.items() if key != "graphs"}})
    encoder.eval().requires_grad_(False)
    zs, source_logits, source_ids = encode_graphs(encoder, source_graphs, args.batch_size, device)
    zt, target_logits, target_ids = encode_graphs(encoder, target_graphs, args.batch_size, device)
    current = target_logits.softmax(1)
    prior = current if state["prior"] is None else 0.5 * state["prior"] + 0.5 * current
    result = refine(zs, zt, source_labels, prior, args, 2, initial_q=state["q"])
    return encoder, zs, zt, source_logits, current, result, history, refreshes, source_ids, target_ids


def _disjoint_support(batch, embeddings, k):
    pair_chunks = []
    original_chunks = []
    for graph_index in range(batch.num_graphs):
        nodes = (batch.batch == graph_index).nonzero(as_tuple=False).reshape(-1)
        start = int(nodes[0])
        local_edges = batch.edge_index[:, batch.batch[batch.edge_index[0]] == graph_index] - start
        original = canonical_pairs(local_edges, len(nodes))
        candidates = candidate_pairs(embeddings[nodes], original, k) + start
        pair_chunks.append(candidates)
        original_chunks.append(original + start)
    pairs = torch.cat(pair_chunks, 1)
    original = torch.cat(original_chunks, 1)
    values = query_edges(pairs, pair_keys(original, batch.num_nodes), batch.num_nodes,
                         embeddings.dtype)
    return pairs, original, values


def _atom_endpoints(source_batch, target_batch, zs, zt, graph, graph_probability,
                    valid_graphs, epsilon, seed):
    """One target molecule per source molecule, then same-category atom matching."""
    device = zs.device
    assignment = torch.full((source_batch.num_nodes,), -1, dtype=torch.long, device=device)
    matched = torch.zeros(source_batch.num_nodes, dtype=torch.bool, device=device)
    with isolated_rng(seed):
        target_graph_for_source = torch.zeros(source_batch.num_graphs, dtype=torch.long,
                                              device=device)
        target_graph_for_source[valid_graphs] = torch.multinomial(
            graph_probability[valid_graphs], 1).reshape(-1)
        for source_graph in range(source_batch.num_graphs):
            if not bool(valid_graphs[source_graph]):
                continue
            source_nodes = (source_batch.batch == source_graph).nonzero(as_tuple=False).reshape(-1)
            target_graph = int(target_graph_for_source[source_graph])
            target_nodes = (target_batch.batch == target_graph).nonzero(as_tuple=False).reshape(-1)
            for atom_type in torch.unique(source_batch.atom_type[source_nodes]):
                left = source_nodes[source_batch.atom_type[source_nodes] == atom_type]
                right = target_nodes[target_batch.atom_type[target_nodes] == atom_type]
                if not right.numel():
                    continue
                cost = torch.cdist(F.normalize(zs[left], dim=1),
                                   F.normalize(zt[right], dim=1)).square()
                scaled = robust_scale(cost)
                probability = torch.softmax(-scaled / epsilon, dim=1)
                chosen = torch.multinomial(probability, 1).reshape(-1)
                assignment[left] = right[chosen]
                matched[left] = True
    end_z = zs.clone()
    end_z[matched] = zt[assignment[matched]]
    end_b = graph.initial_logits.clone()
    matched_edges = matched[graph.pairs[0]] & matched[graph.pairs[1]]
    if bool(matched_edges.any()):
        mapped = assignment[graph.pairs[:, matched_edges]]
        target_pairs = canonical_pairs(target_batch.edge_index, target_batch.num_nodes)
        target_values = query_edges(mapped, pair_keys(target_pairs, target_batch.num_nodes),
                                    target_batch.num_nodes, zt.dtype)
        end_b[matched_edges] = torch.logit(
            graph.delta + (1 - 2 * graph.delta) * target_values)
    return end_z, end_b, matched, target_graph_for_source


def _mass_weights(batch, graph_weights, pairs):
    node_counts = torch.bincount(batch.batch, minlength=batch.num_graphs)
    node_weights = graph_weights[batch.batch] / node_counts[batch.batch]
    pair_graph = batch.batch[pairs[0]]
    if not torch.equal(pair_graph, batch.batch[pairs[1]]):
        raise ValueError("Candidate support contains a cross-molecule edge")
    edge_counts = torch.bincount(pair_graph, minlength=batch.num_graphs)
    edge_weights = graph_weights[pair_graph] / edge_counts[pair_graph].clamp_min(1)
    return node_weights, edge_weights


@_eff_measured('velocity', evaluation=False)
def train_velocity(source_batch, target_batch, zs_atoms, zt_atoms, labels, result,
                   graph, args):
    probability, graph_weights, valid_graphs = coupling_weights(
        result.coupling, args.joint_mass_power)
    node_weights, edge_weights = _mass_weights(source_batch, graph_weights, graph.pairs)
    velocity = JointVelocity(args.hidden, args.velocity_width, args.velocity_depth, 2,
                             structure=True).to(zs_atoms.device)
    optimizer = torch.optim.Adam(velocity.parameters(), lr=args.flow_lr,
                                 weight_decay=args.flow_weight_decay)
    atom_labels = labels[source_batch.batch]
    history = []
    velocity.train()
    for epoch in range(args.flow_epochs):
        end_z, end_b, matched, sampled = _atom_endpoints(
            source_batch, target_batch, zs_atoms, zt_atoms, graph, probability,
            valid_graphs, args.joint_epsilon, args.seed + 300000 + epoch)
        with isolated_rng(args.seed + 400000 + epoch):
            t = torch.rand((), device=zs_atoms.device, dtype=zs_atoms.dtype)
        path_z = (1 - t) * zs_atoms + t * end_z
        path_b = (1 - t) * graph.initial_logits + t * end_b
        optimizer.zero_grad()
        node, edge = velocity(path_z, path_b, t, atom_labels, graph)
        node_loss = (node_weights * (node - (end_z - zs_atoms)).square().mean(1)).sum()
        edge_loss = (edge_weights * (edge - (end_b - graph.initial_logits)).square()).sum()
        loss = finite(node_loss + args.structure_loss_weight * edge_loss,
                      "molecular joint flow loss")
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "node_mse": float(node_loss.detach()),
                        "edge_mse": float(edge_loss.detach()), "total_loss": float(loss.detach()),
                        "matched_atom_fraction": float(matched.float().mean()),
                        "sampled_target_graphs": int(torch.unique(sampled).numel())})
    return velocity.eval(), history, graph_weights, valid_graphs


def _graph_head_loss(head, moved, pairs, adjacency, source_atoms, original_pairs,
                     labels, batch, anchor, return_features=False):
    atom_logits, atom_features = head(moved, pairs, adjacency, return_features=True)
    moved_logits = global_mean_pool(atom_logits, batch.batch)
    moved_loss = F.cross_entropy(moved_logits, labels)
    source_logits = global_mean_pool(
        head(source_atoms, original_pairs, source_atoms.new_ones(original_pairs.size(1))),
        batch.batch)
    source_loss = F.cross_entropy(source_logits, labels)
    result = ((1 - anchor) * moved_loss + anchor * source_loss, moved_logits, {
        "transported_loss": float(((1 - anchor) * moved_loss).detach()),
        "source_loss": float((anchor * source_loss).detach()),
    })
    if return_features:
        return (*result, global_mean_pool(atom_features, batch.batch))
    return result


@dataclass
class MolecularPPArtifacts:
    predictor: MolecularPredictor
    diagnostics: dict
    histories: dict
    evolution: dict


def fit_molecular_pp(source_graphs, target_graphs, args, device, on_head_epoch=None):
    seed_everything(args.seed)
    outputs = train_encoder(source_graphs, target_graphs, args, device)
    (encoder, zs_graph, zt_graph, source_encoder_logits, q0, result,
     encoder_history, refreshes, source_ids, target_ids) = outputs
    source_batch = Batch.from_data_list(source_graphs).to(device)
    target_batch = Batch.from_data_list(target_graphs).to(device)
    encoder.eval()
    with torch.no_grad():
        zs_atoms = encoder.atoms(source_batch).detach()
        zt_atoms = encoder.atoms(target_batch).detach()
    pairs, original_pairs, original_values = _disjoint_support(
        source_batch, zs_atoms, args.structure_knn)
    _, _, valid_graphs = coupling_weights(result.coupling, args.joint_mass_power)
    valid_atoms = valid_graphs[source_batch.batch]
    graph = GraphState(pairs, original_values, valid_atoms, args.edge_logit_delta, joint=True)
    labels = source_batch.y.reshape(-1)
    velocity, flow_history, graph_weights, valid_graphs = train_velocity(
        source_batch, target_batch, zs_atoms, zt_atoms, labels, result, graph, args)
    atom_labels = labels[source_batch.batch]
    flow_z, flow_b, nfe = integrate_joint(velocity, zs_atoms, atom_labels, graph, args.ode_steps)
    moved, adjacency = mix_output(zs_atoms, flow_z, graph, flow_b, args.transport_strength)
    moved, adjacency = moved.detach(), adjacency.detach()
    target_pairs = canonical_pairs(target_batch.edge_index, target_batch.num_nodes)
    target_values = zt_atoms.new_ones(target_pairs.size(1))
    seed_everything(args.seed + 1000)
    with _eff_stage('classification'):
        head = GraphHead(args.hidden, 2, args.dropout).to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=args.head_lr,
                                     weight_decay=args.weight_decay)
        head_history = []
        for epoch in range(args.head_epochs):
            head.train()
            optimizer.zero_grad()
            loss, _, details, generated_features = _graph_head_loss(
                head, moved, pairs, adjacency, zs_atoms, original_pairs, labels,
                source_batch, args.head_source_anchor_weight, return_features=True)
            mmd = loss.new_zeros(())
            if args.encoder_mmd_weight > 0:
                # One row per molecule prevents large molecules dominating alignment.
                # Frozen atom states feed the same trainable, pre-dropout head layer.
                target_features = global_mean_pool(
                    head.features(zt_atoms, target_pairs, target_values), target_batch.batch)
                with isolated_rng(args.seed + 400000 + epoch):
                    mmd = rbf_mmd(generated_features, target_features,
                                  args.encoder_mmd_max_samples)
                loss = loss + args.encoder_mmd_weight * mmd
            details.update(mmd=float(mmd.detach()),
                           mmd_weighted_loss=float((args.encoder_mmd_weight * mmd).detach()),
                           mmd_enabled=args.encoder_mmd_weight > 0)
            finite(loss, "molecular graph-head loss").backward()
            optimizer.step()
            head_history.append({"epoch": epoch + 1, "total_loss": float(loss.detach()), **details})
            # Optional graph-level source-holdout observer. The caller isolates RNG
            # and restores module modes; no scores or labels return to optimization.
            if on_head_epoch is not None:
                on_head_epoch(MolecularPredictor(encoder, head), epoch + 1)
    predictor = MolecularPredictor(encoder.eval(), head.eval())
    diag = sct_diagnostics(result, q0, args.joint_mass_power)
    diag.update({
        "method_revision": "graph-pp-gin-sct-molecule-pair-atom-joint-generated-mmd-v2",
        "mmd_alignment_stage": "classification_generated_vs_target_graph_hidden",
        "encoder_feedback_refreshes": refreshes,
        "source_graphs": source_batch.num_graphs,
        "target_graphs": target_batch.num_graphs,
        "source_atoms": source_batch.num_nodes,
        "target_atoms": target_batch.num_nodes,
        "candidate_edges": pairs.size(1),
        "original_source_edges": original_pairs.size(1),
        "candidate_cross_molecule_edges": int(
            (source_batch.batch[pairs[0]] != source_batch.batch[pairs[1]]).sum()),
        "classification_supervision_units": source_batch.num_graphs,
        "source_encoder_accuracy": float(
            (source_encoder_logits.argmax(1) == labels).float().mean()),
        "nfe": nfe,
        "target_labels_used_for_training": False,
    })
    evolution = {
        "source_z": zs_atoms.detach().cpu(), "labels": atom_labels.detach().cpu(),
        "pairs": pairs.detach().cpu(), "source_values": original_values.detach().cpu(),
        "valid": valid_atoms.detach().cpu(), "edge_logit_delta": args.edge_logit_delta,
        "velocity_state": {k: v.detach().cpu() for k, v in velocity.state_dict().items()},
        "velocity_config": {"hidden": args.hidden, "width": args.velocity_width,
                            "depth": args.velocity_depth, "classes": 2},
        "steps": args.ode_steps, "transport_strength": args.transport_strength,
        "evolved_z": moved.cpu(), "evolved_edge_weights": adjacency.cpu(),
    }
    return MolecularPPArtifacts(
        predictor=predictor,
        diagnostics=diag,
        histories={"encoder": encoder_history, "flow": flow_history, "head": head_history},
        evolution=evolution,
    )


@torch.no_grad()
def predict_graphs(predictor, graphs, batch_size, device):
    predictor.eval()
    logits, ids = [], []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        logits.append(predictor(batch).cpu())
        ids.append(batch.graph_id.reshape(-1).cpu())
    return torch.cat(logits), torch.cat(ids)


def replay_evolution(saved):
    graph = GraphState(saved["pairs"], saved["source_values"], saved["valid"],
                       saved["edge_logit_delta"], joint=True)
    config = saved["velocity_config"]
    velocity = JointVelocity(config["hidden"], config["width"], config["depth"],
                             config["classes"], structure=True)
    velocity.load_state_dict(saved["velocity_state"])
    velocity.eval()
    z, b, _ = integrate_joint(velocity, saved["source_z"], saved["labels"], graph,
                              saved["steps"])
    return mix_output(saved["source_z"], z, graph, b, saved["transport_strength"])

