"""PP 的基础积木：共享编码器、点式分类头，以及 node 对照的速度/积分。

    当前 PP 的编码器训练在 pp_training.py；默认 joint 速度网络在 joint_graph.py。
    本文件中的 train_source_encoder 只供独立诊断使用，不是另一版 PP。
"""
# 本文件提供共享表示、source-only诊断与node消融；joint主方法的速度模型和
# 联合积分在joint_graph.py，当前PP训练统一在pp_training.py。
import torch
from torch import nn
import torch.nn.functional as F

from experiment.runtime import isolated_rng
from model.sct import coupling_weights, finite

class SharedA2GNN(nn.Module):
    """两域共用一套特征变换，但允许邻域传播次数不同。

    x:[N,输入维度] → encode:[N,hidden] → classifier:[N,类别数]。
    encode 是后续 SCT/流使用的表示；classifier 在编码器训练时提供辅助类别分数，
    最终预测所用的新 GraphHead/LatentHead 在主流程中另行初始化。
    """
    # 两域共享变换W及辅助分类层，但分别用s_prop/t_prop次归一化邻域传播。
    # 编码输出[N,D]，辅助分类层输出[N,C]；最终GraphHead另行初始化。

    def __init__(self, in_dim, hidden, num_classes, dropout, target_steps, *, source_steps=0):
        super().__init__()
        from pygda.nn.prop_gcn_conv import PropGCNConv
        self.propagation = PropGCNConv(in_dim, hidden, cached=False)
        # 不缓存某一个域的归一化连接，防止同一模型切域时误用另一域图结构。
        self.classifier = nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self.target_steps = target_steps
        self.source_steps = source_steps

    def encode(self, x, edge_index, *, domain):
        # 传播层做线性变换、多次传播及偏置，之后统一ReLU/Dropout。
        # 这不是每次传播配一套独立权重，也不是两层GCN堆叠。
        if domain == 'source':
            if getattr(self, 'source_steps', 0) == 0:
                # 源传播0次：仅共享线性变换；后续联合流/分类仍可使用源图。
                z = self.propagation.lin(x)
                if self.propagation.bias is not None:
                    z = z + self.propagation.bias
            else:
                z = self.propagation(x, edge_index, self.source_steps)
        elif domain == 'target':
            z = self.propagation(x, edge_index, self.target_steps)
        else:
            raise ValueError('domain must be source or target')
        return F.dropout(F.relu(z), p=self.dropout, training=self.training)

    def forward(self, x, edge_index, *, domain):
        return self.classifier(self.encode(x, edge_index, domain=domain))


def encode_domain(model, x, edge_index, domain):
    """只取指定域的节点表示，不调用编码器自带的辅助分类层。"""
    return model.encode(x, edge_index, domain=domain)


class ConditionalVelocityMLP(nn.Module):
    """
    这是flow_state=node的表示流对照，既没有边ODE状态，也没有图消息层。
    不要把它与joint/fixed_graph共用的JointVelocity架构混淆。
    条件速度网络：v(z, t, c)
    — 输入：latent 特征 z + 时间 t + 条件类别 c（one-hot）
    — 输出：预测的速度向量 v（与 z 同维度）
    — 网络：多层 MLP + SiLU 激活

    在流匹配中，这个网络学习的是"从源域特征到目标域特征"的速度场。
    条件 c 告诉它："这个节点的类别是 X，请往目标域中类别 X 的方向搬。"
    """

    def __init__(self, hidden, num_classes, width, depth):
        super().__init__()
        layers = []
        # 输入维度 = z(hidden) + t(1维) + c(num_classes维)
        input_dim = hidden + 1 + num_classes
        for index in range(depth):
            layers.extend([
                nn.Linear(input_dim if index == 0 else width, width),
                nn.SiLU()  # SiLU = Swish，比 ReLU 更平滑的激活函数
            ])
        layers.append(nn.Linear(width, hidden))  # 输出层：映射回 latent 维度
        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, z, t, labels):
        """前向传播：输入 z, t, 标签，输出预测速度。"""
        if t.ndim == 0:
            t = t.expand(z.size(0))  # 如果 t 是标量，扩展成 batch 大小
        # 标签转 one-hot → 作为条件输入
        condition = F.one_hot(labels, self.num_classes).to(dtype=z.dtype)
        # 拼接：z | t | condition → 送入 MLP
        return self.net(torch.cat((z, t.reshape(-1, 1), condition), dim=1))


class LatentHead(nn.Module):
    """
    head_kind=mlp时的点式分类对照；默认主方法使用GraphHead利用演化边权。
    轻量分类头。
    在 latent 特征 z 上做分类：Linear → ReLU → Dropout → Linear(输出)。
    每次构造都是独立参数。它只看每个节点的表示，不读取图边。
    """

    def __init__(self, hidden, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def features(self, z):
        """与GCN头一致，返回分类层之前、不含dropout的隐藏表示。"""
        return self.net[1](self.net[0](z))

    def forward(self, z, return_features=False):
        """输入 latent z，输出分类 logits。"""
        features = self.features(z)
        logits = self.net[3](self.net[2](features))
        return (logits, features) if return_features else logits


def train_source_encoder(source, target, args, num_classes, history=None):
    """独立 source-only 诊断：仅用源 CE 训练共享编码器，不读取目标真标签。

    返回 model、源/目标表示、兼容占位 None、源 logits；表示已 detach，
    模型参数尚未 requires_grad_(False)。当前 PP 不经过这个诊断函数。
    """
    # 独立source-only诊断基线，不是PP的另一种运行模式。
    # PP始终使用pp_training.train_encoder。
    model = SharedA2GNN(source.x.size(1), args.hidden, num_classes,
                        args.dropout, args.t_prop, source_steps=args.s_prop).to(source.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.encoder_lr,
                                 weight_decay=args.weight_decay)
    model.train()
    for epoch in range(args.encoder_epochs):
        optimizer.zero_grad()
        source_features = encode_domain(model, source.x, source.edge_index, 'source')
        loss = F.cross_entropy(model.classifier(source_features), source.y)
        finite(loss, 'source pretraining loss').backward()
        optimizer.step()
        if history is not None:
            history.append({'epoch': epoch + 1, 'source_ce': float(loss.detach()),
                            'total_loss': float(loss.detach())})
    model.eval()
    with torch.no_grad():
        # 关闭dropout后冻结表示值；调用方再requires_grad_(False)冻结模型参数。
        zs = encode_domain(model, source.x, source.edge_index, 'source').detach()
        zt = encode_domain(model, target.x, target.edge_index, 'target').detach()
        source_logits = model.classifier(zs)
    return model, zs, zt, None, source_logits


def _integrate_heun_impl(velocity, z_source, labels, steps, *, capture_steps=(), snapshots=None):
    """
    使用 Heun 二阶方法（改进的 Euler）ODE 积分。
    从 z_source 出发，沿速度场走 steps 步，到达搬运后的 latent 特征。

    Heun 方法 = 预测-校正两步法：
      k1 = v(z_t, t)          ← 预测斜率
      k2 = v(z_t + dt·k1, t+dt)  ← 校正斜率
      z_{t+dt} = z_t + dt·(k1 + k2)/2  ← 二阶精度

    参数：
      velocity — 训练好的速度网络
      z_source — 源域 latent 特征（起点）
      labels   — 源域标签（作为条件输入）
      steps    — ODE 步数（默认 12，值越大越精确但越慢）

    返回：
      z — 搬运后的 latent 特征（终点）
    """
    if capture_steps and snapshots is None:
        raise ValueError('snapshots list is required when capturing Heun steps')
    z = z_source.clone()
    dt = 1.0 / steps          # 每步的时间步长
    for step in range(steps):
        t0 = z.new_full((z.size(0),), step * dt)        # 当前时间
        t1 = z.new_full((z.size(0),), (step + 1) * dt)  # 下一时间
        k1 = velocity(z, t0, labels)                     # 预测步
        k2 = velocity(z + dt * k1, t1, labels)           # 校正步
        z = z + 0.5 * dt * (k1 + k2)                     # 二阶更新
        if step + 1 in capture_steps:
            snapshots.append(z.detach().clone())
    return z


@torch.no_grad()
def integrate_heun(velocity, z_source, labels, steps, *, capture_steps=(), snapshots=None):
    """node 对照的无梯度积分封装；每步调用两次速度网络，共 2*steps 次。

    积分期间固定模型参数，输出是新表示；它不会训练速度网络或保证到达某个采样点。
    """
    return _integrate_heun_impl(velocity, z_source, labels, steps,
                                capture_steps=capture_steps, snapshots=snapshots)


def train_node_velocity(z_source, z_target, labels, coupling, args, classes):
    """node 对照的速度训练，返回网络、有效节点掩码、损失历史和节点权重。

    起点/终点都在冻结的表示空间；源标签提供条件。只有本函数的新 MLP 被优化。
    相比 joint，此对照还改变了网络架构和时间采样方式，分析消融时需同时说明。
    """
    # 纯表示流对照：只采有效节点，用同一Π生成条件端点概率和节点质量。
    # 与joint不同：没有边状态，且每个节点独立采t，而非整图共享标量t。
    row_prob, row_ratio, valid = coupling_weights(coupling, args.joint_mass_power)
    if not bool(valid.any()):
        return None, torch.zeros_like(valid), None, row_ratio
    velocity = ConditionalVelocityMLP(args.hidden, classes, args.velocity_width,
                                      args.velocity_depth).to(z_source.device)
    optimizer = torch.optim.Adam(velocity.parameters(), lr=args.flow_lr,
                                 weight_decay=args.flow_weight_decay)
    history = []
    velocity.train()
    for epoch in range(args.flow_epochs):
        optimizer.zero_grad()
        with isolated_rng(args.seed + 300000 + epoch):
            endpoint_index = torch.multinomial(row_prob[valid], 1).squeeze(1)
            starts = z_source[valid]
            endpoints = z_target[endpoint_index]
            source_labels = labels[valid]
            t = torch.rand(starts.size(0), device=starts.device, dtype=starts.dtype)
        path = (1 - t[:, None]) * starts + t[:, None] * endpoints
        # 线性条件路径的解析导数就是endpoints-starts，无需训练期ODE模拟。
        target_velocity = endpoints - starts
        prediction = velocity(path, t, source_labels)
        per_node_loss = (prediction - target_velocity).square().mean(dim=1)
        weights = row_ratio[valid]
        flow_loss = (weights * per_node_loss).sum() / weights.sum().clamp_min(1e-12)
        loss = flow_loss
        if not torch.isfinite(loss):
            raise ValueError('Non-finite coupled flow-matching loss')
        loss.backward()
        optimizer.step()
        history.append({'epoch': epoch + 1,
                        'flow_mse': float(flow_loss.detach()),
                        'weighted_total_loss': float(loss.detach())})
    return velocity.eval(), valid, history, row_ratio
