"""Sparse undirected graph states, differentiable weighted GCNs and joint Heun."""

from experiment.efficiency import measured as _eff_measured
# SSJF定义状态/端点/耦合架构；AFE通过外部联合损失学习该架构，再调用Heun。
# 约定N=源节点数、D=表示维数、M=候选无序边数；仅保存每条无向边一次。
# pairs:[2,M]，节点状态Z:[N,D]，边logit状态B:[M]，解码软边权A:[M]。
import math

import torch
from torch import nn
import torch.nn.functional as F

from model.sct import finite, MASS_FLOOR


def canonical_pairs(edge_index, nodes):
    """把已无向的输入连接整理成每条边一列的[2,M]表。

    比如0->1和1->0只保存一列[0,1]。若反向连接缺失就报错，不悄悄补边。
    这里只整理连接表示；后面卷积时会再展开双向消息。
    """
    # 输入必须已经无向；检查反向边存在，而非静默把有向图转换成无向图。
    # 去自环/去重复后仅保留i<k。GCN内部的计算自环不属于这里保存的结构。
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError('edge_index must have shape [2,E]')
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= nodes):
        raise ValueError('Edge index out of range')
    i, k = edge_index[:, edge_index[0] != edge_index[1]]
    keys = torch.unique(i * nodes + k, sorted=True)
    reverse = torch.unique(k * nodes + i, sorted=True)
    if not torch.equal(keys, reverse):
        raise ValueError('DiffGDA++ requires undirected input; asymmetric edges are unsupported')
    keys = keys[keys // nodes < keys % nodes]
    return torch.stack((keys // nodes, keys % nodes))


def pair_keys(pairs, nodes):
    # 把无序对编码成唯一整数min(i,k)*N+max(i,k)，便于排序、合并和查询。
    return torch.minimum(pairs[0], pairs[1]) * nodes + torch.maximum(pairs[0], pairs[1])


def query_edges(pairs, sorted_keys, nodes, dtype):
    """批量问“这些节点对在观察图里有没有边”，返回[M]个0或1。

    pairs中的编号必须属于被查询图；sorted_keys是该图排序后的连接整数键。
    用二分查找避免先构造完整稠密邻接；自配对一律作为无自环处理。
    """
    # sorted_keys必须已排序。只检查二值连接存在性，不读edge_attr/原始边权。
    # 当两源节点映射到同一目标时pairs两端相同，按无自环设为0。
    if not sorted_keys.numel():
        return torch.zeros(pairs.size(1), device=pairs.device, dtype=dtype)
    keys = pair_keys(pairs, nodes)
    at = torch.searchsorted(sorted_keys, keys).clamp_max(sorted_keys.numel() - 1)
    # searchsorted给插入位置，不代表命中；仍须比较键相等，避免误报连接。
    return ((sorted_keys[at] == keys) & (pairs[0] != pairs[1])).to(dtype)


@_eff_measured('structure_support', evaluation=False)
@torch.no_grad()
def candidate_pairs(z, source_pairs, k, chunk_size=512):
    """先画定允许改动的连接范围：原边加上表示空间的近邻候选。

    z:[Ns,D]，source_pairs:[2,Es]，返回去重的无序候选pairs:[2,M]。
    近邻用普通表示欧氏距离；SCT跨域代价的归一化规则不用于这里。
    k=0时只保留原源边；选好支持后训练和积分都不再改变它。
    """
    # 固定候选支持E_c = 原源边 ∪ 对称KNN(zs)，允许原非边在流中变为软连接。
    # 邻居只根据冻结表示选取，不读取标签；候选集合外始终没有结构状态。
    finite(z, 'candidate representations')
    nodes = z.size(0)
    k = min(k, max(0, nodes - 1))
    keys = [pair_keys(source_pairs, nodes)]
    if k:
        for start in range(0, nodes, chunk_size):
            # 分块仅节约KNN距离内存，不是小批量UOT，也不改变后续整图流损失。
            stop = min(start + chunk_size, nodes)
            cost = torch.cdist(z[start:stop], z, compute_mode='donot_use_mm_for_euclid_dist').square()
            rows = torch.arange(stop - start, device=z.device)
            cost[rows, rows + start] = float('inf')
            nearest = cost.argsort(dim=1, stable=True)[:, :k]
            # 稳定排序让同距离候选按已有索引顺序处理，避免无关随机择邻居。
            origin = torch.arange(start, stop, device=z.device)[:, None].expand_as(nearest)
            keys.append(pair_keys(torch.stack((origin.reshape(-1), nearest.reshape(-1))), nodes))
    merged = torch.unique(torch.cat(keys), sorted=True)
    return torch.stack((merged // nodes, merged % nodes))


class GraphState(nn.Module):
    """保存图的固定说明，帮助外部的连续节点/边状态完成演化。

    pairs是允许调整哪些边；source_values是它们原来的0/1连接；
    valid/active说明哪些节点/边有运输支持。以上都不是可训练网络权重。
    真正随时间变化的z:[Ns,D]和b:[M]由速度网络及积分器传入。
    """
    # buffers随设备迁移/保存，但不是优化器参数。固定的是支持/掩码/原图，
    # ODE演化的是外部传入的z、b；不要把initial_logits当可训练Parameter。
    def __init__(self, pairs, source_values, valid, delta, joint=True):
        super().__init__()
        self.nodes = valid.numel()
        self.joint = joint
        self.register_buffer('pairs', pairs)
        self.register_buffer('source_values', source_values)
        self.register_buffer('valid', valid)
        self.register_buffer('active', valid[pairs[0]] & valid[pairs[1]] if joint else
                             torch.zeros(pairs.size(1), dtype=torch.bool, device=pairs.device))
        # 只有两端均具有运输支持的边才演化；fixed_graph/node的active全False。
        smooth = delta + (1 - 2 * delta) * source_values
        # 原0/1分别平滑为δ和1-δ，再取logit，避免无穷初值；δ=.01时约±4.595。
        self.register_buffer('initial_logits', torch.logit(smooth))
        self.delta = delta

    def decode(self, logits):
        """把不受[0,1]限制的边状态B，转成图卷积可使用的软连接强度A。"""
        # 活动边用sigmoid得到[0,1]软边；非活动边严格保持原0/1连接。
        # 不做阈值化，梯度可以经软边权和度归一化传到节点消息。
        finite(logits, 'edge logits')
        return torch.where(self.active, torch.sigmoid(logits), self.source_values)

    def endpoints(self, assignment, target_keys, target_nodes, target_z):
        """用同一张对应表构造节点与结构的训练终点，返回end_z、end_b。

        如果源i参考目标2、源k参考目标5：节点分别参考zt[2]/zt[5]，
        它们之间的结构终点必须查询目标图2与5是否相连，不能另采一次对应。
        注意这里构造的是流匹配监督，积分结束不保证精确达到某次采样终点。
        """
        # assignment:[N]，允许重复目标选择，不要求Ns=Nt或节点一一对应。
        # 同一J定义z1_i=zt[J_i]和A1_ik=At[J_i,J_k]，是端点一致性的关键。
        z = target_z[assignment]
        mapped = assignment[self.pairs]
        # mapped:[2,M]。查询目标二值图后，用与B0相同的δ映射成B1。
        values = query_edges(mapped, target_keys, target_nodes, target_z.dtype)
        end_b = torch.logit(self.delta + (1 - 2 * self.delta) * values)
        return z, torch.where(self.active, end_b, self.initial_logits)

    def edge_weights(self, weights):
        """将节点支持w扩展到边；只有两个端点都有支持的边才贡献结构损失。"""
        # 式(21)：ω_ik=(w_i*w_k)/∑活动候选边(w_l*w_r)。
        # 节点质量w已经全局归一化，边需要在有效边支持上再次统一归一化。
        raw = weights[self.pairs[0]] * weights[self.pairs[1]] * self.active
        total = raw.sum()
        return raw / total if bool(total > 0) else torch.zeros_like(raw)


class WeightedGraphConv(nn.Module):
    """按当前软连接收集邻居消息的一层GCN，支持对边权反传梯度。

    输入z:[N,输入维]、pairs:[2,M]、values:[M]，输出[N,输出维]。
    每条边的消息会按两端节点度归一化，防止高连接度节点简单累加过大。
    """
    # 计算H = D^(-1/2)(A+I)D^(-1/2)ZW + bias。
    # A是当前连续边权；不用缓存，因为训练路径/ODE每次调用的A都可能不同。
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, z, pairs, values):
        nodes = z.size(0)
        i, k = pairs
        loops = torch.arange(nodes, device=z.device)
        src = torch.cat((i, k, loops))
        dst = torch.cat((k, i, loops))
        # 无序边镜像为双向消息，并添加单位计算自环；不改变GraphState的零对角。
        edge = torch.cat((values, values, values.new_ones(nodes)))
        degrees = z.new_zeros(nodes).index_add(0, dst, edge)
        inv = degrees.clamp_min(MASS_FLOOR).rsqrt()
        normalized = edge * inv[src] * inv[dst]
        # 度与归一化都保留对values的梯度，否则会削弱结构对节点更新的作用。
        features = self.linear(z)
        return features.new_zeros((nodes, features.size(1))).index_add(
            0, dst, features[src] * normalized[:, None]) + self.bias


def mlp(input_dim, width, depth, output_dim):
    layers = []
    for index in range(depth):
        layers.extend((nn.Linear(input_dim if index == 0 else width, width), nn.SiLU()))
    return nn.Sequential(*layers, nn.Linear(width, output_dim))


class JointVelocity(nn.Module):
    """SSJF耦合架构：同时预测“节点表示往哪走”和“连接logit怎样变”。

    输入当前(z,b)、标量时间t、固定源标签；输出node:[Ns,D]、edge:[M]。
    消息层把当前结构送给节点分支；边分支又读取当前节点状态，形成耦合。
    AFE的train_joint_velocity负责用联合监督训练本模型。
    """
    # 一个耦合模型，参数包含共享图消息层、节点/边MLP以及时间/类别条件。
    # structure=False仅去边输出，节点仍使用固定图消息；区别于纯node的无图MLP。
    def __init__(self, hidden, width, depth, classes, structure=True):
        super().__init__()
        self.message1 = WeightedGraphConv(hidden, width)
        self.message2 = WeightedGraphConv(width, width)
        self.register_buffer('frequencies', math.pi * 2.0 ** torch.arange(8))
        # 8个sin/cos频率组成16维时间编码；类别embedding也是16维。
        self.label_embedding = nn.Embedding(classes, 16)
        self.node_net = mlp(hidden + width + 32, width, depth, hidden)
        # 边输入=2D的端点表示特征+W消息+1维B+16维时间+16维类别，故+33。
        self.edge_net = mlp(2 * hidden + width + 33, width, depth, 1) if structure else None

    def forward(self, z, b, t, labels, graph):
        # SSJF式(17)：先从当前B恢复A，再从当前(Z,A)计算消息H。
        values = graph.decode(b)
        h = F.silu(self.message1(z, graph.pairs, values))
        h = F.silu(self.message2(h, graph.pairs, values))
        phase = t * self.frequencies
        # 把一个时间数值展开为sin/cos特征，让MLP更容易表达不同时间的速度。
        time = torch.cat((phase.sin(), phase.cos()))
        y = self.label_embedding(labels)
        # 节点输入[z_i,h_i,e_t,e_y] -> D维速度；固定源标签是条件而非演化状态。
        node = self.node_net(torch.cat((z, h, time.expand(z.size(0), -1), y), 1))
        node = node * graph.valid[:, None]
        edge = torch.zeros_like(b)
        if self.edge_net is not None and b.numel():
            i, k = graph.pairs
            features = torch.cat((z[i] + z[k], (z[i] - z[k]).abs(), h[i] + h[k],
                                  b[:, None], time.expand(b.numel(), -1), y[i] + y[k]), 1)
            # 端点和、绝对差、消息和、类别和均对i/k交换不变，保证无向对称性。
            # 边读取当前节点状态，节点读取当前边状态，形成双向耦合。
            edge = self.edge_net(features).squeeze(1) * graph.active
        return finite(node, 'node velocity'), finite(edge, 'edge velocity')


class UncoupledVelocity(nn.Module):
    """消融：节点/边两个分支不交换“当前”状态，但各自仍保留图结构信息。

    与 JointVelocity 的唯一差别是耦合方向：
      - 节点分支用当前 z，但消息层固定读取源图 A_S（不读当前 b）；
      - 边分支用当前 b，但端点特征固定读取初始 z_S（不读当前 z）。
    时间/类别条件、候选支持、损失和积分器与 JointVelocity 相同，
    因此该对照只移除双向耦合，不混淆“去掉图消息层”或“去掉边演化”。
    """
    def __init__(self, hidden, width, depth, classes, initial_z):
        super().__init__()
        self.message1 = WeightedGraphConv(hidden, width)
        self.message2 = WeightedGraphConv(width, width)
        self.register_buffer('frequencies', math.pi * 2.0 ** torch.arange(8))
        self.label_embedding = nn.Embedding(classes, 16)
        self.node_net = mlp(hidden + width + 32, width, depth, hidden)
        # 边分支不读当前消息H，故输入比JointVelocity少width维。
        self.edge_net = mlp(2 * hidden + 33, width, depth, 1)
        # initial_z只作为固定结构条件，不参与优化；persistent=False避免
        # 把[N,D]张量写进checkpoint的state_dict（replay会重新构造本模块）。
        self.register_buffer('initial_z', initial_z.detach().clone(), persistent=False)

    def forward(self, z, b, t, labels, graph):
        # 节点：当前语义 + 固定源图消息；不读取当前边状态b。
        values = graph.source_values
        h = F.silu(self.message1(z, graph.pairs, values))
        h = F.silu(self.message2(h, graph.pairs, values))
        phase = t * self.frequencies
        time = torch.cat((phase.sin(), phase.cos()))
        y = self.label_embedding(labels)
        node = self.node_net(torch.cat((z, h, time.expand(z.size(0), -1), y), 1))
        node = node * graph.valid[:, None]
        edge = torch.zeros_like(b)
        if b.numel():
            # 边：当前边状态 + 固定初始语义；不读取当前z或当前消息H。
            i, k = graph.pairs
            zi, zk = self.initial_z[i], self.initial_z[k]
            features = torch.cat((zi + zk, (zi - zk).abs(), b[:, None],
                                  time.expand(b.numel(), -1), y[i] + y[k]), 1)
            edge = self.edge_net(features).squeeze(1) * graph.active
        return finite(node, 'node velocity'), finite(edge, 'edge velocity')


class GraphHead(nn.Module):
    """最终任务分类器：读表示和图连接，输出每个节点对C类的分数。

    它使用演化源图训练、目标原图预测；不预测速度，也不求解运输。
    """
    # AFE结束后单独初始化的任务分类器，与速度模型/编码器辅助头不共享参数。
    def __init__(self, hidden, classes, dropout):
        super().__init__()
        self.first = WeightedGraphConv(hidden, hidden)
        self.second = WeightedGraphConv(hidden, classes)
        self.dropout = dropout

    def features(self, z, pairs, values):
        """MMD对齐的分类器隐藏表示；使用输入图的实际边权，不含dropout。"""
        return F.relu(self.first(z, pairs, values))

    def forward(self, z, pairs, values, return_features=False):
        features = self.features(z, pairs, values)
        h = F.dropout(features, self.dropout, self.training)
        logits = self.second(h, pairs, values)
        return (logits, features) if return_features else logits


@_eff_measured('transport', evaluation=False)
@torch.no_grad()
def integrate_joint(velocity, source_z, labels, graph, steps):
    """沿训练好的速度场实际行走，返回节点终态、边logit终态及评估次数。

    每步先预测再校正：第一次速度估计试走位置，第二次用试走位置再估速度，
    最后平均两次速度。两个分量同步更新，结构变化会反馈到下一次节点消息。
    此处no_grad表示只使用已学好的模型，不再根据终点更新参数。
    """
    # AFE式(26)-(27)：从(Zs,B0)联合积分到t=1，固定源标签与候选支持。
    # 不重新采样目标端点；每步两次速度评估，NFE=2*steps。
    if steps < 1:
        raise ValueError('Heun steps must be positive')
    z, b = source_z.clone(), graph.initial_logits.clone()
    dt = 1.0 / steps
    for step in range(steps):
        t = z.new_tensor(step * dt)
        vz, vb = velocity(z, b, t, labels, graph)
        trial_z, trial_b = z + dt * vz, b + dt * vb
        # Heun预测状态：节点和边必须同步向前，不能只更新其中一个分量。
        finite(trial_z, 'predicted node state')
        finite(trial_b, 'predicted edge state')
        next_z, next_b = velocity(trial_z, trial_b, t + dt, labels, graph)
        # 第二次评估用trial_b重算软邻接和消息，再对两个斜率取平均校正。
        z = finite(z + 0.5 * dt * (vz + next_z), 'integrated nodes')
        b = finite(b + 0.5 * dt * (vb + next_b), 'integrated edges')
    return z, b, 2 * steps


def mix_output(source_z, flow_z, graph, flow_b, strength):
    """决定采用多少演化位移：0保留原图，1采用完整积分结果，中间值线性混合。

    返回表示[Ns,D]与软边权[M]；没有把边阈值化，也不改变源节点标签。
    注意混合的是解码后的连接强度，不是在原logit与终logit之间混合。
    """
    # ρ只控制最终位移，不改变训练贡献w_i，也不改变积分的终止时间t=1。
    # 输出仍是源节点集合上的表示和候选软边，不生成新的节点身份/标签。
    # Explicit identity avoids roundoff and restores the unsmoothed source graph.
    if strength == 0:
        return source_z.clone(), graph.source_values.clone()
    z = source_z + strength * (flow_z - source_z)
    a = graph.source_values + strength * (graph.decode(flow_b) - graph.source_values)
    return finite(z, 'mixed representations'), finite(a, 'mixed adjacency')
