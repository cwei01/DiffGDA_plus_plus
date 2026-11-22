"""PP的表示准备和分类损失：把原图变成可供联合流使用的固定表示。

阅读顺序：prepare -> train_encoder（期间会调用feedback）-> sct.refine；
联合演化结束后主流程再调用head_loss。本模块没有基础/增强两条路线。
约定Ns/Nt为源/目标节点数、F为原特征维数、D为hidden、C为类别数。
"""

from experiment.efficiency import measured as _eff_measured
# 主流程始终调用prepare/head_loss；不再区分基础版或增强版。
import torch
import copy
import torch.nn.functional as F
from experiment.runtime import isolated_rng
from model.pp_components import SharedA2GNN, encode_domain
from model.sct import refine, _reliability

def rbf_mmd(source, target, max_samples=256):
    """比较两域表示整体分布，返回一个可反向求导的标量损失。

    source:[Ns,D]、target:[Nt,D]；每域最多抽max_samples个节点，控制距离表开销。
    它不知道目标真标签，也不逐类配对；对齐的是表示的整体分布。
    """
    # 表示级边际分布对齐，不按类别分组。域内核均值包含对角线（有偏估计）。
    if max_samples < 1 or not len(source) or not len(target):
        raise ValueError('MMD requires nonempty domains and positive max_samples')
    if len(source) > max_samples:
        source = source[torch.randperm(len(source), device=source.device)[:max_samples]]
    if len(target) > max_samples:
        target = target[torch.randperm(len(target), device=target.device)[:max_samples]]
    # Preserve finite squared distances for large ODE outputs.
    source = source.to(torch.float64)
    target = target.to(torch.float64)
    xy = torch.cdist(source, target).square()
    # xy是跨域距离表，xx/yy是各域内部距离表；最后比较核相似度的均值。
    xx = torch.cdist(source, source).square()
    yy = torch.cdist(target, target).square()
    bandwidth = (xy.detach().median() / 2).clamp(min=1e-6)
    # 带宽由当前跨域样本距离估计，但自身不反传；核值仍对表示可微。
    values = []
    for scale in (0.25, 0.5, 1.0, 2.0):
        # 多种距离尺度取平均，避免只用一个核带宽描述分布差异。
        gamma = 1 / (2 * bandwidth * scale)
        values.append(torch.exp(-gamma * xx).mean()
                      + torch.exp(-gamma * yy).mean()
                      - 2 * torch.exp(-gamma * xy).mean())
    return torch.stack(values).mean()


@_eff_measured('encoder', evaluation=False)
def train_encoder(source, target, args, num_classes, history=None, feedback_handler=None):
    """
    输入：
        source: 源域数据
        target: 目标域数据
        args: 训练参数
        num_classes: 类别数
        history: 历史记录
        feedback_handler: 反馈处理函数

    返回值：
        model: 训练好的SharedA2GNN及辅助分类层
        z_source: 源域所有节点的 latent 特征（固定不变）
        z_target: 目标域所有节点的 latent 特征（固定不变）
        target_pseudo: 辅助头argmax诊断返回值；当前PP配对使用软Q和Π而非此值
        source_logits: 源域分类 logits（用来评估 encoder 的准确率）
    """
    # 创建编码器和优化器
    model = SharedA2GNN(source.x.size(1), args.hidden, num_classes,
                        args.dropout, args.t_prop, source_steps=args.s_prop).to(source.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.encoder_lr,
                                 weight_decay=args.weight_decay)
    from model.pp_stability import source_class_weights,update_ema
    class_weights=source_class_weights(source.y,num_classes,getattr(args,'pp_source_balance_power',0.))
    ema_decay=getattr(args,'pp_ema_decay',0.)
    averaged=copy.deepcopy(model).eval().requires_grad_(False) if ema_decay>0 else None
    model.train()

    # 计算反馈开始和刷新轮次，总轮次为150时，前100轮预训练，101轮第一次计算SCT反馈，使用反馈继续训练。第126轮开始前再次刷新反馈。
    feedback = None
    pretrain_epochs = 2 * args.encoder_epochs // 3 if feedback_handler is not None else None
    second_refresh = (pretrain_epochs + (args.encoder_epochs - pretrain_epochs) // 2
                      if feedback_handler is not None else None)

    # 刷新反馈
    for epoch in range(args.encoder_epochs):
        if feedback_handler is not None and epoch in (pretrain_epochs, second_refresh):
            # 生成反馈时关闭dropout；回到训练后只把反馈当固定监督，不反传SCT。
            model.eval()
            feedback = feedback_handler(model, epoch)
            model.train()

        # 计算源域分类损失
        optimizer.zero_grad()
        source_features = encode_domain(model, source.x, source.edge_index, 'source')
        ce = F.cross_entropy(model.classifier(source_features), source.y,weight=class_weights)

        # 编码器阶段只做源监督和SCT反馈；MMD在演化后的分类阶段计算。
        loss = ce

        if feedback is not None:
            with isolated_rng(args.seed + 200000 + epoch):
                target_features = encode_domain(model, target.x, target.edge_index, 'target')

        feedback_loss = None
        #如果有反馈，加入软标签损失
        if feedback is not None:
            soft_q, reliability = feedback
            soft_q, reliability = soft_q.detach(), reliability.detach()
            soft_ce = -(soft_q * F.log_softmax(model.classifier(target_features), dim=1)).sum(dim=1)
            feedback_loss = (reliability * soft_ce).sum() / reliability.sum().clamp_min(1e-12)

            ramp = min(1.0, (epoch - pretrain_epochs + 1) /
                       min(30, args.encoder_epochs - pretrain_epochs))

            loss = loss + args.joint_encoder_soft_weight * ramp * feedback_loss

        # 反向传播并更新参数
        loss.backward()
        optimizer.step()
        if averaged is not None:update_ema(averaged,model,ema_decay)
        if history is not None:
            history.append({'epoch': epoch + 1, 'source_ce': float(ce.detach()),
                            'total_loss': float(loss.detach()),
                            'mmd_enabled': False,
                            'feedback_loss': (float(feedback_loss.detach())
                                              if feedback_loss is not None else None)})

    # 训练结束后生成固定表示
    if averaged is not None:model.load_state_dict(averaged.state_dict())
    model.eval()
    with torch.no_grad():
        z_source = encode_domain(model, source.x, source.edge_index, 'source').detach()
        z_target = encode_domain(model, target.x, target.edge_index, 'target').detach()

        target_pseudo = model.classifier(z_target).argmax(dim=1).detach()
        source_logits = model.classifier(z_source)
    return model, z_source, z_target, target_pseudo, source_logits


def prepare(source, target, args, classes, timings):
    """
    函数功能：为 PP 训练准备必要的数据和环境。

    1.训练共享编码器：用源域标签和编码器反馈训练；MMD留到演化后的分类阶段。
    2.执行 SCT：结合源、目标表示和源标签，估计目标节点的类别软概率、节点可靠性，以及源节点和目标节点之间的匹配质量。
    3.整理结果并返回：把编码器、表示、SCT 结果、训练记录等打包成八项。

    输入：
    例如这次任务有源图 100 个节点，目标图 80 个节点；每个节点有 32 个输入特征；共有 5 个类别。
    那么传入的对象大致是：
        source.x  源节点特征，形状 [100, 32]
        source.y  源节点标签，形状 [100]
        target.x  目标节点特征，形状 [80, 32]
        target.y  None
        args      本次训练设置，例如隐藏层大小、学习率、训练轮数
        classes   类别数，这里是 5
        timings   记录各训练阶段耗时的计时器
    
    输出：
    一个包含编码器、源和目标节点的潜在表示、目标节点的伪标签、源节点的logits以及训练记录的元组。
        encoder: 训练好的共享编码器模型
        zs: 源节点的潜在表示，形状 [100, D]
        zt: 目标节点的潜在表示，形状 [80, D]
        q0: 目标节点的伪标签，形状 [80, 5]
        result: SCT 结果，包含目标节点的类别软概率和源节点与目标节点之间的匹配质量
        reliability: 目标节点的可靠性，形状 [80]
        encoder_history: 编码器训练过程中的记录，例如每轮的损失和准确率
        training_diag: 训练诊断信息，例如编码器反馈的刷新记录和可靠性统计
    """

    # 检查目标域标签是否为None，如果不是则抛出异常
    assert target.y is None, 'Target labels must be isolated before PP training'
    # 初始化状态字典，用于存储上次细化结果和固定先验。这两个变量用来记录编码器反馈的历史信息，以便在训练过程中进行迭代更新。
    state = {'q': None, 'prior': None}
    refreshes = []

    
    def feedback(encoder, epoch):
        # 不直接优化模型，只生成下一段编码器训练使用的软标签与权重。
        with torch.no_grad():
            # 使用当时编码器状态解SCT；不同刷新点表示可变，但不对求解器反传。
            zs = encode_domain(encoder, source.x, source.edge_index, 'source')
            zt = encode_domain(encoder, target.x, target.edge_index, 'target')
            # 用当前编码器状态预测目标域的类别软概率
            current = encoder.classifier(zt).softmax(1)
            # 固定先验混合历史预测
            prior = current if state['prior'] is None else 0.5 * state['prior'] + 0.5 * current
            # 根据当前表示和先验运行SCT，得到目标节点的类别软概率、节点可靠性，以及源节点和目标节点之间的匹配质量。
            result = refine(zs, zt, source.y, prior, args, classes, initial_q=state['q'])
            #保存SCT结果
            q, pi, history, reliability = (result.probabilities, result.coupling,
                                           result.history, result.reliability)
            state.update(q=q.detach(), prior=prior.detach())
            # 记录反馈信息并返回软标签和可靠性
            refreshes.append({'epoch': epoch, 'rounds': len(history),
                              'coupling_mass': float(pi.sum()),
                              'reliability_mean': float(reliability.mean())})
            return q.detach(), reliability.detach()

    # 训练共享编码器
    encoder_history = []
    with timings.stage('encoder'):
        #调用 train_encoder() 开始训练共享编码器。
        encoder, zs, zt, _, _ = train_encoder(source, target, args, classes,
                                             history=encoder_history,
                                             feedback_handler=feedback)
        # 冻结编码器并重新计算目标域的类别软概率
        encoder.requires_grad_(False)
        with torch.no_grad():
            initial = encoder.classifier(zt).softmax(1)

    # 运行最终的SCT，得到目标节点的类别软概率、节点可靠性，以及源节点和目标节点之间的匹配质量。        
    with timings.stage('coupling'), torch.no_grad():
        # 编码器训练完成后再做最终SCT，得到后续流训练全程固定的Q和Π。
        current = initial
        prior = current if state['prior'] is None else 0.5 * state['prior'] + 0.5 * current
        result = refine(zs, zt, source.y, prior, args, classes, initial_q=state['q'])
        reliability = result.reliability

    # 记录训练诊断信息
    diagnostic = {'pp_training_profile': 'unified-v3',
                  'encoder_feedback_refreshes': refreshes,
                  'reliability_mean': float(reliability.mean()),
                  'reliability_min': float(reliability.min()),
                  'head_weighting_actual': args.head_weighting,
                  'source_anchor_weight': args.head_source_anchor_weight}
    return encoder, zs, zt, initial, result, reliability, encoder_history, diagnostic


def head_loss(head, moved_logits, source_labels, weights, valid,
                       zs, source_pairs, source_values, args):
    """最终分类头的一轮损失：返回总损失标量和可记录的分项数值。

    moved_logits:[Ns,C]是当前head在演化源图上的输出；标签仍为原源标签。
    zs:[Ns,D]及source_pairs/values是原源表示图，供可选锚定分支使用。
    anchor=0只用搬运分支，anchor=1只用原源图，通常值混合两个分支。
    两分支使用同一个head，只有head参数会由主流程优化。
    """
    # 最终分类器阶段：L=(1-a)*L_moved + a*L_source，均使用源真标签。
    # moved_logits已在演化源图上计算；source分支在原源图上重新前向同一head。
    # 这里无目标软监督参数，也不更新编码器/速度场。
    from model.pp_stability import source_class_weights
    class_weights=source_class_weights(source_labels,moved_logits.size(1),getattr(args,'pp_source_balance_power',0.))
    moved_ce = F.cross_entropy(moved_logits, source_labels, weight=class_weights,reduction='none')
    transported = valid & (weights > 0) if args.transport_strength > 0 else torch.zeros_like(valid)
    # valid只描述运输支持，不是源标签训练集mask。uniform分支仍平均所有源节点。
    anchor_weight = args.head_source_anchor_weight
    if not bool(transported.any()) or anchor_weight == 1:
        # rho=0或无有效搬运节点时回退原源图监督，即使请求anchor=0也不空训。
        anchor_weight = 1.0
        moved_loss = moved_ce.new_zeros(())
    else:
        if args.head_weighting == 'mass':
            # 分类质量加权是可选项；不同于流匹配阶段始终保留的质量权重。
            r = weights[transported] / weights[transported].sum()
            moved_loss = (1-anchor_weight) * (r * moved_ce[transported]).sum()
        else:
            moved_loss = (1-anchor_weight) * moved_ce.mean()
    source_loss = moved_ce.new_zeros(())
    if anchor_weight > 0:
        # 源锚定始终普通平均CE；GCN使用原源边权1，MLP只使用原源表示。
        source_logits = head(zs, source_pairs, source_values) if args.head_kind == 'gcn' else head(zs)
        source_loss = anchor_weight * F.cross_entropy(source_logits, source_labels,weight=class_weights)
    loss = moved_loss + source_loss
    # detail里的float用于日志，不参与反传；真正反传的仍是上面的Tensor loss。
    return loss, {'transported_loss': float(moved_loss.detach()),
                  'source_loss': float(source_loss.detach())}
