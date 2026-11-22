"""Log-domain unbalanced transport primitives (no model or dataset imports)."""
# SCT 的通用数值求解器：不读取标签，不更新网络参数，只根据给定代价求 Π。
import math
import torch


def robust_scale(cost):
    """让代价以典型正值为单位计量，返回形状不变的代价表。

    例如正代价中位数为10，代价20变成2；这会影响与epsilon/其他代价项的比例。
    它不是把数值强行限制到[0,1]，也不会把大代价截断。
    """
    # 除以正代价中位数，避免不同代价项尺度悬殊；不做截尾/分位数裁剪。
    # 若调用方每轮重传不同代价，中位数也会变化，不等价于固定一次缩放。
    median = cost[cost > 0].median() if (cost > 0).any() else cost.new_tensor(1.0)
    return cost / torch.clamp(median, min=1e-12)


def log_unbalanced_sinkhorn(cost, epsilon, tau, iterations, source_marginal=None,
                            target_marginal=None, balanced=False):
    """输入“配对有多贵”的表，输出“每对分配多少支持”的运输矩阵。

    cost:[Ns,Nt]，返回Π形状相同，非负但不要求和为1。
    epsilon控制熵正则，tau控制偏离参考质量的惩罚；两者都不是训练学习率。
    a/b参考质量默认均匀，当前PP给b传入归一化的连续可靠性。
    在log域更新可减少极小指数带来的数值问题；求解器本身不读取类别标签。
    """
    # cost:[Ns,Nt]，求解 <cost,Π> + ε∑Π(logΠ-1)
    #                  + τ KL(Π1 || a) + τ KL(Πᵀ1 || b)，KL含 -u+v 项。
    # 未指定边缘时a/b均匀；当前PP传入可靠性加权的target_marginal。
    # 这里的b是目标参考质量[Nt]，不是SSJF中候选边的logit状态[M]。
    ns, nt = cost.shape
    if source_marginal is None:
        log_a = cost.new_full((ns,), -math.log(ns))
    else:
        if source_marginal.shape != (ns,) or not torch.isfinite(source_marginal).all() or not bool((source_marginal > 0).all()):
            raise ValueError('source marginal must be finite, positive and match source nodes')
        log_a = (source_marginal / source_marginal.sum()).log()
        # 外部提供的参考量先归一化；它描述总质量分配意向，而非硬约束。
    if target_marginal is None:
        log_b = cost.new_full((nt,), -math.log(nt))
    else:
        if target_marginal.shape != (nt,) or not torch.isfinite(target_marginal).all() or not bool((target_marginal > 0).all()):
            raise ValueError('target marginal must be finite, positive and match target nodes')
        log_b = (target_marginal / target_marginal.sum()).log()
    log_kernel = -cost / epsilon
    # 广义Sinkhorn指数θ=τ/(τ+ε)。θ小于1允许行列质量偏离参考值，
    # 因此输出Π不要求行和/列和严格等于a/b，也不是逐行概率矩阵。
    # balanced=True即τ→∞的极限θ=1，退化为普通（平衡）Sinkhorn，
    # 行/列边缘被硬性约束到a/b；此时tau不参与计算。
    exponent = 1.0 if balanced else tau / (tau + epsilon)
    log_u = cost.new_zeros(ns)
    log_v = cost.new_zeros(nt)
    # log_u/log_v是一维行列缩放因子，广播到二维表后调节每对的质量。
    for _ in range(iterations):
        # 行更新沿目标维聚合，列更新沿源维聚合；logsumexp避免直接指数下溢。
        log_u = exponent * (
            log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        )
        log_v = exponent * (
            log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
        )
    log_pi = log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)
    # 固定迭代次数的数值近似；此处没有残差提前停止。
    # 截断是数值保护，极端代价下不应宣称仍是无限精度的精确最优解。
    return torch.exp(log_pi.clamp(min=-80.0, max=30.0))
