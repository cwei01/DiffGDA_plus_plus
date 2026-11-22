import copy

import torch
from torch import nn
import torch.nn.functional as F

from experiment.runtime import isolated_rng, seed_everything
from model.pp_components import (encode_domain, LatentHead, integrate_heun,
                                  train_node_velocity)
from model.joint_graph import (canonical_pairs, pair_keys, query_edges, candidate_pairs,
                               GraphState, JointVelocity, UncoupledVelocity, GraphHead,
                               integrate_joint, mix_output)
from model.sct import coupling_weights, diagnostics as sct_diagnostics, finite


class GraphPredictor(nn.Module):
    """把冻结编码器和最终分类头打包，输出 [目标节点数, 类别数] 分数。

    返回的是 logits，取每行最大分数的列号即可得到类别。这里不运行 SCT 或 ODE。
    编码器自带的辅助 classifier 不用于最终预测；head 是另外训练的新分类头。
    """
    def __init__(self, encoder, head, head_kind):
        super().__init__()
        self.encoder, self.head, self.head_kind = encoder, head, head_kind
        self.method = 'diffgda_pp'

    def forward(self, data):
        # 部署路径只需要目标原图。这里重新编码，应与训练时冻结 zt 的预测一致。
        # 不需要源标签条件、运输矩阵或目标 ODE；这些仅用于构造源训练图。
        z = encode_domain(self.encoder, data.x, data.edge_index, 'target')
        if self.head_kind == 'mlp':
            return self.head(z)
        pairs = canonical_pairs(data.edge_index, data.num_nodes)
        return self.head(z, pairs, z.new_ones(pairs.size(1)))


def train_joint_velocity(z_source, z_target, labels, result, graph, target_keys, args, classes):
    """学习“目前这个状态应往哪里、以多快速度变化”。

    z_source/target 是冻结的节点表示，labels 只有源真标签；result 保存 SCT 对应。
    每轮随机抽目标端点及时间，构造中间状态，要求网络同时预测节点/边的位移。
    返回训练好的速度网络、每轮损失和源节点权重；没有有效对应时返回 None 网络。
    这里训练速度网络，不训练最终分类头，也不通过 ODE 反向传播。
    """
    # AFE 的联合条件流匹配训练。输入表示和 Π 已冻结，只优化 velocity.parameters()。
    # probability: [Ns,Nt] 的 P(j|i)；weights: [Ns] 的 w_i；valid: [Ns]。
    # 前者控制目标端点采样，后者控制源节点在回归目标中的相对贡献。
    probability, weights, valid = coupling_weights(result.coupling, args.joint_mass_power)
    if not bool(valid.any()):
        # 无有效运输行：不训练/积分速度模型，主流程保留原源图训练分类器。
        return None, [], weights
    if args.flow_state == 'uncoupled':
        # 消融：节点/边分支各自演化，但不再交换当前状态。
        velocity = UncoupledVelocity(args.hidden, args.velocity_width, args.velocity_depth,
                                     classes, z_source).to(z_source.device)
    else:
        velocity = JointVelocity(args.hidden, args.velocity_width, args.velocity_depth,
                                 classes, structure=args.flow_state == 'joint').to(z_source.device)
    optimizer = torch.optim.Adam(velocity.parameters(), lr=args.flow_lr,
                                 weight_decay=args.flow_weight_decay)
    edge_weights = graph.edge_weights(weights)
    # ω_ik ∝ w_i*w_k，只在双有效端点的活动候选边上归一化，全程固定。
    history = []
    velocity.train()
    samples=int(getattr(args,'pp_flow_samples',1))
    if samples<1:raise ValueError('pp_flow_samples must be positive')
    for epoch in range(args.flow_epochs):
        optimizer.zero_grad()
        node_total=edge_total=loss_total=0.
        for draw in range(samples):
            with isolated_rng(args.seed + 300000 + epoch + draw*1000000):
                assignment = torch.zeros(z_source.size(0), dtype=torch.long, device=z_source.device)
                assignment[valid] = torch.multinomial(probability[valid], 1).squeeze(1)
                t = torch.rand((), device=z_source.device, dtype=z_source.dtype)
                # Stratified times plus an average over strata still estimate
                # the same uniform-time SSJF objective. K=1 is the original path.
                if samples>1:t=(draw+t)/samples
            end_z, end_b = graph.endpoints(assignment, target_keys, z_target.size(0), z_target)
            end_z = torch.where(valid[:, None], end_z, z_source)
            path_z = (1 - t) * z_source + t * end_z
            path_b = (1 - t) * graph.initial_logits + t * end_b
            node, edge = velocity(path_z, path_b, t, labels, graph)
            node_loss = (weights * (node - (end_z - z_source)).square().mean(1)).sum()
            edge_loss = (edge_weights * (edge - (end_b - graph.initial_logits)).square()).sum()
            loss = finite(node_loss + args.structure_loss_weight * edge_loss, 'joint flow loss')
            (loss/samples).backward()
            node_total+=float(node_loss.detach())/samples
            edge_total+=float(edge_loss.detach())/samples
            loss_total+=float(loss.detach())/samples
        # 一次反传共同更新图消息层、节点/边输出分支和条件嵌入。
        optimizer.step()
        history.append({'epoch': epoch + 1, 'node_mse': node_total,
                        'edge_mse': edge_total, 'flow_samples':samples,
                        'weighted_total_loss': loss_total})
    return velocity.eval(), history, weights


class DiffGDAPPMethod:
    """
    完成核心工作，把 PP 方法的几个阶段按顺序接起来：
    读取已经整理好的源域和目标域图，训练节点表示，计算跨域对应关系，再演化源图，最后训练分类器并评估。

    输入：
        source：源域图，含节点特征 source.x、边 source.edge_index 和标签 source.y。
        target：目标域图，含特征和边；目标真标签已经被移出，target.y 应为 None。
        evaluator：评估器，它保存目标标签，用来算分数和选择表现最好的模型。

    """
    def __init__(self, args, num_classes):
        self.args, self.num_classes = args, num_classes

    def fit(self, source, target, evaluator):

        # 取出配置并保存评估器
        args, classes = self.args, self.num_classes
        self.evaluator = evaluator
                   
        # 整理源图和目标图的边
        source_pairs = canonical_pairs(source.edge_index, source.num_nodes) # 源图无序边列表
        target_pairs = canonical_pairs(target.edge_index, target.num_nodes) # 目标图无序边列表
        target_keys = pair_keys(target_pairs, target.num_nodes) # 目标图边的键
        encoder_history = []

        # 调用 prepare，得到节点表示和跨域匹配
        from model.pp_training import prepare
        outputs = prepare(source, target, args, classes, evaluator.timings)
        encoder, zs, zt, q0, result, reliability, encoder_history, training_diag = outputs

        # 按需暂存完整保存材料
        if args.artifact_policy == 'full':
            self.training_artifact = {
                'refined_probabilities': result.probabilities.cpu(),
                'target_reliability': reliability.cpu(),
                'diagnostics': training_diag,
            }

        # 从运输质量得到速读训练要用的权重
        with torch.no_grad():
            _, weights, valid = coupling_weights(result.coupling, args.joint_mass_power)

        # 建立可演化的侯选边集合
        with evaluator.timings.stage('structure_support'):
            # 确定候选边，找出源图中哪些节点对可以被考虑为边，节点表示相近的节点也可以列入候选集合
            pairs = candidate_pairs(zs, source_pairs, args.structure_knn)
            # 检查每条候选边在原始源图中是否存在
            original_values = query_edges(pairs, pair_keys(source_pairs, source.num_nodes),
                                          source.num_nodes, zs.dtype)
            # 建立图状态对象
            graph = GraphState(pairs, original_values, valid, args.edge_logit_delta,
                               joint=args.flow_state in ('joint', 'uncoupled'))
            # 检查是否至少有一个候选边被选中
            if not bool(graph.edge_weights(weights).sum() > 0):
                graph.active.zero_()

        # 训练速度网络
        with evaluator.timings.stage('velocity'):
            # 如果是节点演化，训练节点速度场；默认 joint 会训练联合速度网络，同时预测节点和边变化
            if args.flow_state == 'node':
                velocity, _, flow_history, weights = train_node_velocity(
                    zs, zt, source.y, result.coupling, args, classes)
                flow_history = flow_history or []
            else:
                velocity, flow_history, weights = train_joint_velocity(
                    zs, zt, source.y, result, graph, target_keys, args, classes)

        # 用速度网络演化源图
        flow_z, flow_b, nfe = zs.clone(), graph.initial_logits.clone(), 0
        if velocity is not None:
            with evaluator.timings.stage('transport'):
                if args.flow_state == 'node':
                    flow_z[valid] = integrate_heun(velocity, zs[valid], source.y[valid], args.ode_steps)
                    nfe = 2 * args.ode_steps
                else:
                    flow_z, flow_b, nfe = integrate_joint(velocity, zs, source.y, graph, args.ode_steps)

        # 混合演化前后的状态
        moved, adjacency = mix_output(zs, flow_z, graph, flow_b, args.transport_strength)
        moved, adjacency = moved.detach(), adjacency.detach()

        # 训练分类头
        head_history = []
        with evaluator.timings.stage('classification'), isolated_rng():
            seed_everything(args.seed + 1000)
            head = (GraphHead(args.hidden, classes, args.dropout) if args.head_kind == 'gcn'
                    else LatentHead(args.hidden, classes, args.dropout)).to(zs.device)
            optimizer = torch.optim.Adam(head.parameters(), lr=args.head_lr,
                                         weight_decay=args.weight_decay)
            from model.pp_stability import update_ema
            ema_decay=getattr(args,'pp_ema_decay',0.)
            evaluation_head=copy.deepcopy(head).eval().requires_grad_(False) if ema_decay>0 else head
            predictor = GraphPredictor(encoder, evaluation_head, args.head_kind)
            target_values = zt.new_ones(target_pairs.size(1))
            for epoch in range(args.head_epochs):
                head.train()
                optimizer.zero_grad()
                logits, generated_features = (head(moved, pairs, adjacency, return_features=True)
                    if args.head_kind == 'gcn' else head(moved, return_features=True))
                from model.pp_training import head_loss
                loss, head_details = head_loss(
                    head, logits, source.y, weights, valid, zs, source_pairs,
                    zs.new_ones(source_pairs.size(1)), args)
                # 生成源图与目标原图经同一个可训练分类头提取隐藏表示。
                # 输入Z已冻结，但features保留到head第一层的梯度；不能对moved/zt直接算常量MMD。
                mmd = logits.new_zeros(())
                if args.encoder_mmd_weight > 0:
                    from model.pp_training import rbf_mmd
                    target_features = (head.features(zt, target_pairs, target_values)
                                       if args.head_kind == 'gcn' else head.features(zt))
                    with isolated_rng(args.seed + 400000 + epoch):
                        mmd = rbf_mmd(generated_features, target_features,
                                      args.encoder_mmd_max_samples)
                    loss = loss + args.encoder_mmd_weight * mmd
                head_details.update(mmd=float(mmd.detach()),
                                    mmd_weighted_loss=float((args.encoder_mmd_weight * mmd).detach()),
                                    mmd_enabled=args.encoder_mmd_weight > 0)
                finite(loss, 'classification loss').backward()
                optimizer.step()
                if evaluation_head is not head:update_ema(evaluation_head,head,ema_decay)
                head_history.append({'epoch': epoch + 1, 'total_loss': float(loss.detach()), **head_details})
                # 每轮评估目标图
                evaluator(predictor,lambda: (evaluation_head(zt, target_pairs, target_values) if args.head_kind == 'gcn'
                                   else evaluation_head(zt)), round=1, epoch=epoch + 1)
        evaluator.finish()

        # 汇总诊断并返回
        diag = sct_diagnostics(result, q0, args.joint_mass_power)
        diag.update(training_diag)
        diag['mmd_alignment_stage'] = 'classification_generated_vs_target_hidden'
        possible = max(1, source.num_nodes * (source.num_nodes - 1) // 2)
        diag.update({
            'method_revision': args.method_revision, 'flow_state': args.flow_state,
            'joint_cost_mode': args.joint_cost_mode, 'head_kind': args.head_kind,
            'head_weighting': args.head_weighting, 'joint_mass_power': args.joint_mass_power,
            'candidate_edge_count': pairs.size(1), 'active_edge_count': int(graph.active.sum()),
            'active_edge_fraction': float(graph.active.float().mean()) if pairs.size(1) else 0.0,
            'structure_loss_enabled': bool(graph.active.any()),
            'source_edge_count': source_pairs.size(1),
            'source_edge_density': source_pairs.size(1) / possible,
            'evolved_edge_density': float(adjacency.sum()) / possible,
            'adjacency_mean_absolute_change': float((adjacency - original_values).abs().mean()) if pairs.size(1) else 0.0,
            'source_representation_mean_displacement': float((moved - zs).norm(dim=1).mean()),
            'joint_flow_loss_history': flow_history, 'encoder_history': encoder_history,
            'head_history': head_history, 'nfe': nfe,
            'actual_flow_epochs': args.flow_epochs if velocity is not None else 0,
            'transport_strength': args.transport_strength,
            'transported_source_label_retention': float((encoder.classifier(moved).argmax(1) == source.y).float().mean()),
        })
        if args.artifact_policy == 'full':
            self.evolution_artifact = {
                'method_revision': args.method_revision, 'flow_state': args.flow_state,
                'velocity': copy.deepcopy(velocity).cpu() if velocity is not None else None,
                'graph': copy.deepcopy(graph).cpu(), 'source_z': zs.cpu(),
                'source_labels': source.y.cpu(), 'flow_z': flow_z.cpu(), 'flow_b': flow_b.cpu(),
                'evolved_z': moved.cpu(), 'evolved_edge_weights': adjacency.cpu(),
                'node_weights': weights.cpu(), 'ode_steps': args.ode_steps,
                'transport_strength': args.transport_strength,
            }
        return diag

    def save_artifacts(self, directory):
        """full 保存策略下额外保存演化重放材料和最终 SCT 诊断。

        best.pt/final.pt 由 evaluator 保存；这里保存的不是中断续训所需的优化器状态。
        hasattr 是保存对象的兼容检查，不是在选择基础版或增强版。
        """
        torch.save(self.evolution_artifact, directory / 'evolution.pt')
        if hasattr(self, 'training_artifact'):
            torch.save(self.training_artifact, directory / 'training.pt')
            return ('evolution.pt', 'training.pt')
        return ('evolution.pt',)

    def predict(self, data, state='best'):
        """调用评价器保留的 best 或 final 模型；不重新拟合任何参数。"""
        return self.evaluator.predict(data, state)


@torch.no_grad()
def replay_evolution(artifact):
    """用保存的起点、速度网络和积分设置重放源图搬运，返回表示和边权。

    已训练速度网络不再需要稠密 Π；重放成功只能验证演化保存/恢复一致，
    不能替代训练正确性或泛化效果的检验。
    """
    # 检查点核验：同样的源状态/标签/支持/步数应重建已保存的 evolved_z/边权。
    z, graph, velocity = artifact['source_z'], artifact['graph'], artifact['velocity']
    b = graph.initial_logits.clone()
    out = z.clone()
    if velocity is not None:
        velocity.eval()
        if artifact['flow_state'] == 'node':
            valid = graph.valid
            out[valid] = integrate_heun(velocity, z[valid], artifact['source_labels'][valid],
                                        artifact['ode_steps'])
        else:
            out, b, _ = integrate_joint(velocity, z, artifact['source_labels'], graph,
                                        artifact['ode_steps'])
    return mix_output(z, out, graph, b, artifact['transport_strength'])
