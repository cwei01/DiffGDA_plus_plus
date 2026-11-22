"""PP唯一SCT：连续可靠性、语义细化与归一化运输质量。"""

from experiment.efficiency import measured as _eff_measured
# refine只更新概率和运输计划；后续联合流通过coupling_weights统一接收质量。
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.transport import log_unbalanced_sinkhorn, robust_scale

MASS_FLOOR = 1e-12
PROBABILITY_FLOOR = 1e-8


def finite(value, name):
    """发现NaN/无穷大就带名称报错，避免坏数值悄悄进入后续训练。"""
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Non-finite ' + name)
    return value


def coupling_weights(coupling, power=1.0):
    """把运输质量表拆成“选谁作终点”和“这个源节点占多大权重”。

    输入coupling[i,j]不是概率，它表示源i向目标j分配的质量。
    输出三项：probability[Ns,Nt]、weights[Ns]、valid[Ns]。
    例如两行[.08,.02]与[.0008,.0002]都有概率[.8,.2]，但总支持差100倍；
    weights负责保留这种相对强弱，避免只行归一化后把二者当成同等可靠。
    """
    # Π:[Ns,Nt] -> P(j|i):[Ns,Nt]、w:[Ns]、valid:[Ns]。
    # m_i=∑jΠij；P(j|i)=Πij/m_i；w_i=m_i^γ/∑有效行m^γ。
    # P控制“参考谁”，w控制“贡献多大”；gamma=0为均匀有效源权重消融。
    finite(coupling, 'coupling')
    if bool((coupling < 0).any()):
        raise ValueError('Coupling must be nonnegative')
    mass = coupling.sum(1)
    valid = mass > MASS_FLOOR
    # valid[:,None]把[Ns]变为[Ns,1]，便于对一整行目标候选同时应用掩码。
    # 低于阈值的行不参加端点采样和质量监督，避免空行除零。
    probability = torch.where(valid[:, None], coupling / mass[:, None].clamp_min(MASS_FLOOR), 0)
    tempered = torch.where(valid, mass.clamp_min(MASS_FLOOR).pow(power), 0)
    weights = tempered / tempered.sum().clamp_min(MASS_FLOOR)
    # 全局归一化，不是每个小批次各算一遍。全无效时返回全0权重。
    if bool(valid.any()) and (not bool(torch.isfinite(weights).all()) or
                             float(tempered.sum()) < MASS_FLOOR):
        # Preserve normalization for very large powers/tiny masses without
        # changing the established normal-scale numerical path.
        weights = torch.zeros_like(mass)
        weights[valid] = torch.softmax(power * mass[valid].log().to(torch.float64), 0).to(mass.dtype)
    return finite(probability, 'endpoint probabilities'), finite(weights, 'mass weights'), valid


@dataclass
class SCTResult:
    """SCT结果容器；dataclass自动生成按字段接收参数的构造函数。

    probabilities描述“目标j属于类别c”，coupling描述“源i参考目标j多少”。
    reliability用于目标参考质量和编码器反馈；不要把它误当作目标真标签。
    reliability默认None仅兼容旧结果/手工测试容器，当前refine会实际填写它。
    """
    # probabilities是目标类别概率[Nt,C]；coupling才是跨域质量[Ns,Nt]。
    # history记录细化轮次；allowed_targets/pair_support描述匹配支持，不是图边。
    probabilities: torch.Tensor
    coupling: torch.Tensor
    history: list
    allowed_targets: torch.Tensor
    pair_support: torch.Tensor
    reliability: torch.Tensor | None = None


def _reliability(q, previous=None, *, floor=0.05, scale_floor=0.05,
                 scale_multiplier=1.0, ema=0.5):
    """根据相对置信度给目标节点连续降权，返回[Nt]，不丢弃节点。

    q的每行是C类概率，最大值表示该节点预测的集中程度。
    低于全体平均置信度的节点被柔和降权；fresh的下界为floor。
    previous提供时做滑动平均，避免一次Q变化使权重突然跳动。
    高fresh只代表相对自信，不证明预测正确；EMA后也不一定立即等于1。
    """
    # q:[Nt,C] -> r:[Nt]。相对平均置信度的连续权重，不是正确率/校准概率。
    # 高于平均置信度的fresh=1；低置信度逐渐降权，下界为floor，无硬删除。
    # “固定参数”指CLI固定四个超参数，不是r不随Q变化。
    confidence = q.max(dim=1).values
    center = confidence.mean()
    scale = confidence.std(unbiased=False).clamp_min(scale_floor) * scale_multiplier
    deficit = (confidence - center).clamp_max(0)
    fresh = floor + (1 - floor) * torch.exp(-deficit.square() / (2 * scale.square()))
    return fresh if previous is None else ema * previous + (1 - ema) * fresh


@_eff_measured('sct_refinement', evaluation=False)
@torch.no_grad()
def refine(z_source, z_target, source_labels, initial_prob, args, classes,
                  initial_q=None):
    """SCT：交替完善跨域匹配Π和目标软语义Q，返回SCTResult。

    输入：源表示[Ns,D]、目标表示[Nt,D]、源类别编号[Ns]、目标先验[Nt,C]。
    输出：目标软概率[Nt,C]、运输矩阵[Ns,Nt]、目标可靠性[Nt]和日志。
    initial_prob是本次求解固定参考，initial_q只是迭代从哪里开始。
    no_grad表示这里仅解数值对应关系，不通过迭代反向训练任何网络。
    """
    # 当前PP唯一SCT；no_grad保证仅更新Q/Π，不训练编码器或分类器。
    # initial_prob是本次固定语义先验，initial_q可用上次Q作迭代起点，两者不同。
    # 返回SCTResult，包含q:[Nt,C]、Π:[Ns,Nt]、历史、支持以及r:[Nt]。
    # joint_balanced=True用普通平衡OT替换非平衡OT，仅消融时使用。
    balanced = bool(getattr(args, 'joint_balanced', False))
    feature_cost = robust_scale(torch.cdist(
        F.normalize(z_source, dim=1), F.normalize(z_target, dim=1)).square())
    # legacy代价：表示先单位化，再按正距离中位数缩放，区别于原稿平方距离/D。
    q0 = initial_prob.clamp_min(1e-8)
    # 先把极小概率抬到正数，再归一化，保证后续-log(Q)有定义。
    q0 = q0 / q0.sum(dim=1, keepdim=True)
    q = q0.clone() if initial_q is None else initial_q.clamp_min(1e-8)
    q = q / q.sum(dim=1, keepdim=True)
    # 连续可靠性是当前PP固定组成，超参数由CLI统一填写。
    reliability_options = dict(floor=args.joint_reliability_floor,
                               scale_floor=args.joint_reliability_scale_floor,
                               scale_multiplier=args.joint_reliability_scale,
                               ema=args.joint_reliability_ema)
    reliability = _reliability(q, **reliability_options)
    # 可靠性EMA只在单次细化内传递；新调用从当前Q重新初始化r。
    # 当前PP保留全部跨域节点对支持，不提供硬筛选或相似度截断分支。
    allowed = torch.ones(z_target.size(0), dtype=torch.bool, device=z_target.device)
    support = allowed[None, :].expand_as(feature_cost)
    source_class = F.one_hot(source_labels, classes).to(feature_cost.dtype)
    # 源one-hot:[Ns,C]；后续Πᵀ@onehot才是目标收到的类别质量。
    history = []
    coupling = None
    for round_index in range(args.joint_refine_rounds):
        # 外层一次“根据Q匹配 -> 传递源类别证据 -> 更新Q/r”。
        # 内部每个Sinkhorn解又有自己的迭代预算，两层循环含义不同。
        prior = reliability / reliability.sum()
        # prior:[Nt]是UOT目标参考质量b，也是Q0先验融合的质量，不是类别概率。
        label_cost = robust_scale(-q[:, source_labels].T.clamp_min(1e-8).log())
        # 例：源i类别为2，就看目标j的q[j,2]；概率越低匹配越贵。
        # 这里没有argmax硬分类，非最大概率的类别仍可能获得匹配支持。
        # 第(i,j)项=-log Q[j,y_i]；语义中位数随Q每轮重算。
        cost = args.joint_feature_weight * feature_cost + args.joint_label_weight * label_cost
        cost = cost.masked_fill(~support, 50.0)
        coupling = log_unbalanced_sinkhorn(
            cost, args.joint_epsilon, args.joint_tau, args.joint_sinkhorn_iters,
            target_marginal=prior, balanced=balanced
        )
        received = coupling.T @ source_class
        # Πᵀ:[Nt,Ns]乘源one-hot:[Ns,C]，得到每个目标收到的各类“票数”。
        # 票数是运输质量的加权和，而非整数计数。
        # received:[Nt,C]，每行和=target_mass。固定Q0约束语义漂移。
        target_mass = coupling.sum(dim=0)
        updated = (args.joint_update_weight * received
                   + args.joint_prior_weight * prior[:, None] * q0)
        # 把运输票数与固定初始语义加权融合，随后除总质量变回类别概率。
        # lambda_u/lambda_p控制两种证据强度，不是梯度优化器的学习率。
        updated = updated / (
            args.joint_update_weight * target_mass[:, None]
            + args.joint_prior_weight * prior[:, None]
        ).clamp_min(1e-12)
        updated = updated.clamp_min(1e-8)
        updated = updated / updated.sum(dim=1, keepdim=True)
        # 下限保护后恢复概率行和1；零接收质量时应退回固定先验。
        delta = float((updated - q).abs().max())
        next_reliability = _reliability(updated, reliability, **reliability_options)
        reliability_delta = float((next_reliability - reliability).abs().max())
        history.append({
            'round': round_index + 1,
            'max_probability_change': delta,
            'coupling_mass': float(coupling.sum()),
            'target_mass_mean': float(target_mass.mean()),
            'max_reliability_change': reliability_delta,
        })
        q = updated
        reliability = next_reliability
        if max(delta, reliability_delta) <= args.joint_tolerance:
            # Q和r都基本不变才结束；tolerance不是准确率阈值。
            break
    # Ensure the final coupling is consistent with the final refined Q.
    label_cost = robust_scale(-q[:, source_labels].T.clamp_min(1e-8).log())
    cost = args.joint_feature_weight * feature_cost + args.joint_label_weight * label_cost
    cost = cost.masked_fill(~support, 50.0)
    coupling = log_unbalanced_sinkhorn(cost, args.joint_epsilon, args.joint_tau,
                                       args.joint_sinkhorn_iters,
                                       target_marginal=reliability / reliability.sum(),
                                       balanced=balanced)
    return SCTResult(q, coupling, history, allowed, support, reliability)




def diagnostics(result, initial_prob, power):
    """把Q/Π的统计量转成日志，不改变模型或选择训练样本。

    initial_prob来自最终编码器直接预测；当前prepare可能另混合历史先验，
    因此这里的initial统计是诊断参照，不是所有内部warm-start状态的完整记录。
    ESS=1/sum(w²)：越小表示训练权重集中到少量源节点，不直接等于正确率。
    """
    _, weights, valid = coupling_weights(result.coupling, power)
    q = result.probabilities
    return {
        'joint_refinement_history': result.history,
        'joint_refinement_rounds_actual': len(result.history),
        'joint_initial_target_class_mass': initial_prob.sum(0).cpu().tolist(),
        'joint_refined_target_class_mass': q.sum(0).cpu().tolist(),
        'joint_initial_mean_entropy': float(-(initial_prob * initial_prob.clamp_min(PROBABILITY_FLOOR).log()).sum(1).mean()),
        'joint_refined_mean_entropy': float(-(q * q.clamp_min(PROBABILITY_FLOOR).log()).sum(1).mean()),
        'joint_changed_argmax_fraction': float((initial_prob.argmax(1) != q.argmax(1)).float().mean()),
        'joint_coupling_total_mass': float(result.coupling.sum()),
        'joint_source_row_mass_ratio_mean': float(weights.mean()),
        'joint_source_row_mass_ratio_min': float(weights.min()),
        'joint_source_row_mass_ratio_max': float(weights.max()),
        'joint_source_effective_sample_size': float(weights.sum().square() / weights.square().sum().clamp_min(MASS_FLOOR)),
        'joint_valid_source_fraction': float(valid.float().mean()),
        'pairing_retained_target_fraction': float(result.allowed_targets.float().mean()),
        'pairing_mean_candidates_per_source': float(result.pair_support.sum(1).float().mean()),
        'target_labels_used_for_training': False,
    }
