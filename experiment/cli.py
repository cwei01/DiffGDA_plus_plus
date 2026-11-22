"""把命令行文字变成训练器能读取的实验设置args。

三种对象不要混淆：argv是用户输入的字符串列表，parser是读取规则，
args是读取并检查后的结果。例如['--hidden','128']最终变成args.hidden=128。
add_*函数只是登记规则，不会加载数据、构造模型或开始训练。
当前PP只有一套方案，所有PP参数直接登记，没有增强开启/模块选择开关。
"""
import argparse
import math
import sys
from experiment.protocol import ROOT
PAPER_METHODS = ('diffgda_pp',)
BASELINE_METHODS = ()
PYGDA_DEFAULTS = {'dane': dict(dropout=0.0, lr=0.001, weight_decay=1e-05, num_layers=2), 'udagcn': dict(dropout=0.0, lr=0.004, weight_decay=0.003, num_layers=2), 'adagcn': dict(dropout=0.0, lr=0.004, weight_decay=0.0, num_layers=3), 'strurw': dict(dropout=0.0, lr=0.05, weight_decay=0.0001, num_layers=2), 'grade': dict(dropout=0.0, lr=0.001, weight_decay=0.01, num_layers=2), 'pairalign': dict(dropout=0.0, lr=0.001, weight_decay=0.0001, num_layers=2), 'a2gnn': dict(dropout=0.0, lr=0.004, weight_decay=0.0, num_layers=3), 'tdss': dict(dropout=0.0, lr=0.01, weight_decay=0.005, num_layers=2), 'dgsda': dict(dropout=0.0, lr=0.004, weight_decay=0.0, num_layers=2)}

def add_sct(parser, cost_mode):
    """登记SCT的数值设置：代价、运输求解、软语义细化、质量权重。

    joint_sinkhorn_iters：一次运输计划内部迭代多少步。
    joint_refine_rounds：外部“更新语义再重新匹配”最多做多少轮。
    二者是两层不同的循环，不是同一个训练epoch数。
    """
    parser.add_argument('--joint-feature-weight', type=float, default=1.0)
    parser.add_argument('--joint-label-weight', type=float, default=1.0)
    parser.add_argument('--joint-update-weight', type=float, default=1.0)
    parser.add_argument('--joint-prior-weight', type=float, default=1.0)
    parser.add_argument('--joint-epsilon', type=float, default=0.2)
    parser.add_argument('--joint-tau', type=float, default=1.0)
    parser.add_argument('--joint-sinkhorn-iters', type=int, default=100)
    parser.add_argument('--joint-refine-rounds', type=int, default=5)
    parser.add_argument('--joint-tolerance', type=float, default=0.0001)
    parser.add_argument('--joint-mass-mode', choices=('normalized',), default='normalized')
    parser.add_argument('--joint-mass-power', type=float, default=1.0)
    parser.add_argument('--joint-cost-mode', choices=('legacy_normalized',), default=cost_mode)

def add_diffgda_pp(parser):
    """登记唯一PP的设置表；只负责规定允许填写什么及其默认值。

    type=int/float负责把文字变成整数/小数；choices限制允许的选项。
    例如hidden不写时为64，写--hidden 128后为128；这还不等于建好了网络。
    """
    parser.add_argument('--encoder', choices=('a2gnn',), default='a2gnn')
    parser.add_argument('--s-prop', type=int, default=0)
    parser.add_argument('--t-prop', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    for phase in ('encoder', 'flow', 'head'):
        parser.add_argument('--' + phase + '-lr', type=float, default=0.001)
        parser.add_argument('--' + phase + '-epochs', type=int, default=150)
    parser.add_argument('--flow-weight-decay', type=float, default=0.0)
    parser.add_argument('--velocity-width', type=int, default=128)
    parser.add_argument('--velocity-depth', type=int, default=2)
    parser.add_argument('--ode-steps', type=int, default=12)
    parser.add_argument('--transport-strength', type=float, default=1.0)
    add_sct(parser, 'legacy_normalized')
    parser.add_argument('--structure-knn', type=int, default=10)
    parser.add_argument('--edge-logit-delta', type=float, default=0.01)
    parser.add_argument('--structure-loss-weight', type=float, default=1.0)
    parser.add_argument('--joint-balanced', action='store_true', help='ablation: replace unbalanced OT with balanced OT (tau -> infinity)')
    parser.add_argument('--flow-state', choices=('joint', 'fixed_graph', 'node', 'uncoupled'), default='joint')
    parser.add_argument('--head-kind', choices=('gcn', 'mlp'), default='gcn')
    parser.add_argument('--head-weighting', choices=('uniform', 'mass'), default='uniform')
    parser.set_defaults(method_revision='sct-ssjf-afe-generated-mmd-v4')
    parser.add_argument('--head-mmd-weight', '--encoder-mmd-weight', dest='encoder_mmd_weight', type=float, default=0.1, help='generated-vs-target classifier hidden MMD weight; encoder name is a legacy alias')
    parser.add_argument('--head-mmd-max-samples', '--encoder-mmd-max-samples', dest='encoder_mmd_max_samples', type=int, default=256, help='per-domain sample cap for classification-stage MMD')
    parser.add_argument('--head-source-anchor-weight', type=float, default=0.5)
    parser.add_argument('--joint-encoder-soft-weight', type=float, default=0.1)

def parse_args(argv=None, forced_method=None):
    """执行四步：先识别方法 -> 登记规则 -> 正式读取 -> 检查/补充设置。

    返回argparse.Namespace对象，后续代码用args.hidden这样的点号读字段。
    用户没写的设置由default补齐；--encoder-lr自动对应属性encoder_lr。
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--method', choices=PAPER_METHODS, default=forced_method or 'diffgda_pp')
    (selected, _) = selector.parse_known_args(argv)
    if forced_method and selected.method != forced_method:
        selector.error('compatibility entrypoint cannot select another method')
    method = selected.method
    parser = argparse.ArgumentParser(description='Unified GDA training; target-label-assisted best selection')
    parser.add_argument('--method', choices=PAPER_METHODS, default=method)
    parser.add_argument('--source', default='ACMv9')
    parser.add_argument('--target', default='DBLPv7' if method == 'diffgda' else 'Citationv1')
    parser.add_argument('--data-root', default=str(ROOT / 'data'))
    parser.add_argument('--paper-protocol', action='store_true', help='use the independent DiffGDA paper comparison protocol')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=521070 if method == 'diffgda' else 1)
    parser.add_argument('--out', default='')
    parser.add_argument('--artifact-policy', choices=('metrics', 'full'), default='metrics', help='metrics: JSON/history only (default); full: save checkpoints')
    parser.add_argument('--metrics-only', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--deterministic', choices=('off', 'strict'), default='off', help='determinism policy; off by default, strict enables PyTorch deterministic checks')
    (add_baseline if method in BASELINE_METHODS else globals()['add_' + method])(parser)
    if method == 'graphalign':
        parser.set_defaults(hidden=256, dropout=0.0, lr=0.01, weight_decay=0.0)
    elif method == 'ggda':
        parser.set_defaults(hidden=128, dropout=0.5, lr=0.001, weight_decay=0.0005)
    elif method == 'gaa':
        parser.set_defaults(hidden=128, dropout=0.5, lr=0.0005, weight_decay=0.0005)
    elif method in PYGDA_DEFAULTS:
        parser.set_defaults(**PYGDA_DEFAULTS[method])
    args = parser.parse_args(argv)
    if args.paper_protocol and method in ('diffgda', 'diffgda_pp') and ('--artifact-policy' not in argv) and (not args.metrics_only):
        args.artifact_policy = 'full'
    if args.metrics_only:
        args.artifact_policy = 'metrics'
    if method == 'diffgda':
        args.config = args.config or args.source
    positive = ('nhid', 'hidden', 'epoch', 'encoder_epochs', 'head_epochs', 'flow_epochs', 'ode_steps', 'velocity_width', 'velocity_depth', 'num_layers', 'diffusion_epochs', 'diffusion_steps', 'encoder_lr', 'flow_lr', 'head_lr', 'lr', 'joint_epsilon', 'joint_tau', 'joint_sinkhorn_iters', 'joint_refine_rounds')
    for name in positive:
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error('--' + name.replace('_', '-') + ' must be finite and positive')
    if not math.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        parser.error('dropout must be in [0,1)')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('weight decay must be finite and nonnegative')
    if method == 'diffgda':
        if not 0 < args.alpha <= 1:
            parser.error('--alpha must be in (0,1]')
        return args
    if method in BASELINE_METHODS:
        return args
    if args.s_prop < 0 or args.t_prop < 0:
        parser.error('A2GNN propagation steps must be nonnegative')
    if not math.isfinite(args.transport_strength) or not 0 <= args.transport_strength <= 1:
        parser.error('--transport-strength must be finite and in [0,1]')
    for name in ('joint_feature_weight', 'joint_prior_weight'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error('--' + name.replace('_', '-') + ' must be finite and positive')
    for name in ('joint_label_weight', 'joint_update_weight', 'joint_tolerance', 'joint_mass_power', 'flow_weight_decay', 'structure_loss_weight'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error('--' + name.replace('_', '-') + ' must be finite and nonnegative')
    if args.structure_knn < 0:
        parser.error('--structure-knn must be nonnegative')
    if not math.isfinite(args.edge_logit_delta) or not 0 < args.edge_logit_delta < 0.5:
        parser.error('--edge-logit-delta must be in (0,0.5)')
    if args.flow_state in ('joint', 'uncoupled') and args.structure_loss_weight <= 0:
        parser.error('joint/uncoupled flow requires positive structure loss; use --flow-state fixed_graph/node')
    if args.joint_cost_mode != 'legacy_normalized':
        parser.error('Current PP requires legacy_normalized cost')
    args.joint_reliability_floor = 0.05
    args.joint_reliability_scale_floor = 0.05
    args.joint_reliability_scale = 1.5
    args.joint_reliability_ema = 0.5
    for name in ('encoder_mmd_weight', 'joint_encoder_soft_weight'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(name + ' must be finite and nonnegative; zero disables its loss')
    if not math.isfinite(args.head_source_anchor_weight) or not 0 <= args.head_source_anchor_weight <= 1:
        parser.error('head_source_anchor_weight must be in [0,1]')
    if args.encoder_mmd_max_samples < 1 or args.encoder_epochs < 3:
        parser.error('PP requires positive MMD samples and >=3 encoder epochs')
    return args
