"""运行时的工具箱：数据加载、随机数控制、模型评估、计时器和结果保存。

这个文件像是一个"工具箱"，里面装满了训练过程中需要用到的各种工具：
  • 控制随机数，让实验结果可以复现
  • 加载图数据，预处理特征和标签
  • 计算 F1 分数（模型的考试成绩）
  • 精确计时，统计每个阶段花了多久
  • 保存训练好的模型到硬盘
"""

from experiment.efficiency import measured as _eff_measured

import copy
import json
import os
import random
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch_geometric.utils import degree

from experiment.protocol import FAMILIES


# ==============================================================================
#  第一部分：可复现性控制（减少同环境重复运行的随机差异）
# ==============================================================================


def configure_determinism(mode='off'):
    """配置确定性算法和线程数，配合 seed 提高同环境下的可复现性。

    strict 禁用部分加速并要求确定性算法；不支持的操作可能直接报错。
    off 不主动开启这些限制，也不会撤销同进程中先前开启的 strict 设置。
    两种模式都会把 PyTorch 线程数设为 1。跨硬件/库版本仍可能有数值差异；
    固定 seed 也不能消除模型本身对不同 seed 的敏感性。
    """
    if mode not in ('off', 'strict'):
        raise ValueError('deterministic mode must be off or strict')

    if mode == 'strict':
        # 设置 CUDA 的底层配置，确保卷积等操作是确定性的
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        os.environ.setdefault('PYTHONHASHSEED', '1')  # 对后续子进程生效，不能重置当前进程的哈希种子

        # 关闭 cuDNN 的自动优化器（它会根据输入数据动态选择最快算法）
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # 关闭 TF32：它用较低尾数精度加速部分矩阵运算，这里统一采用更高精度。
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')  # 用最高精度的 FP32

        # 强制所有 PyTorch 操作使用确定性算法
        torch.use_deterministic_algorithms(True)

    # 固定线程配置，减少运行环境差异；并非所有多线程操作都不确定。
    torch.set_num_threads(1)
    # PyTorch forbids setting this pool twice, even to the same value.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def seed_everything(seed):
    """给所有随机数生成器设同一个种子，保证所有随机操作都可复现。

    深度学习中有很多地方用到随机数：
      - 模型参数的初始化（权重一开始是随机的小数）
      - 训练数据的打乱顺序
      - Dropout 丢弃哪些神经元
      - 等等

    这个函数给 Python 内置的 random、numpy 和 PyTorch 都设了同一个种子，
    让随机序列可控；还需确定性算法、相同环境和相同执行顺序才能比较重复结果。

    参数：
        seed: 一个整数，比如 42、2024、521070 等等
    """
    random.seed(seed)                     # Python 内置随机
    np.random.seed(seed)                  # NumPy 随机
    torch.manual_seed(seed)               # PyTorch CPU 随机
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU 随机（所有显卡）


@contextmanager
def isolated_rng(seed=None):
    """一个"沙盒"环境：在这里面修改随机状态，出来后自动恢复原样。

    这个工具用到了 Python 的 contextmanager（上下文管理器），配合 with 语句使用。

    为什么要这个？
    假设你在训练过程中想临时做个评估——评估时不应该打乱主训练流程的随机状态。
    用 isolated_rng 包起来，里面的随机操作不会影响外面的随机序列。

    用法：
        with isolated_rng(seed=42):
            # 在这里做的随机操作，不会影响外面的随机状态
            torch.randn(3)

    参数：
        seed: 可选，如果传了就在沙盒里先设置这个种子
    """
    # 进来前：保存当前的随机状态（Python、NumPy、PyTorch、CUDA）
    py, np_state = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        if seed is not None:
            seed_everything(seed)  # 如果指定了种子，在沙盒里用这个种子
        yield                      # 执行 with 块里面的代码
    finally:
        # 出来后：恢复之前保存的随机状态，就像什么都没发生过一样
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@contextmanager
def evaluating(module):
    """一个"评估模式"沙盒：临时切换到评估模式，结束后恢复训练模式。

    深度学习模型有两种模式：
      - 训练模式（train）：Dropout 会随机丢弃神经元，BatchNorm 会更新统计量
      - 评估模式（eval）：Dropout 关掉，BatchNorm 用固定的统计量

    这个工具自动完成切换，并且在评估时也隔离了随机状态（调用 isolated_rng）。

    用法：
        with evaluating(my_model):
            output = my_model(data)  # 这时模型在 eval 模式，不会影响训练

    参数：
        module: 一个 PyTorch 模型
    """
    # 记录所有子模块当前是训练还是评估模式
    modes = [(m, m.training) for m in module.modules()]
    # 在隔离的随机环境中执行，且不计算梯度（节省内存和计算）
    with isolated_rng(), torch.no_grad():
        module.eval()  # 切换到评估模式
        try:
            yield       # 执行 with 块里面的代码
        finally:
            # 恢复每个模块原来的模式（是训练就回到训练，是评估就还是评估）
            for m, mode in modes:
                m.training = mode


# ==============================================================================
#  第二部分：数据加载和预处理
# ==============================================================================


def _pyg_dataset_safe_globals():
    """Return only the PyG data classes stored in processed graph caches."""
    from torch_geometric.data import Data
    from torch_geometric.data.storage import GlobalStorage
    safe = [Data, GlobalStorage]
    try:
        from torch_geometric.data import data as pyg_data
        for name in ('DataEdgeAttr', 'DataTensorAttr'):
            value = getattr(pyg_data, name, None)
            if value is not None:
                safe.append(value)
    except ImportError:
        pass
    return safe


def _load_pygda_graph(dataset_cls, root, name):
    safe_globals = getattr(torch.serialization, 'safe_globals', None)
    scope = safe_globals(_pyg_dataset_safe_globals()) if safe_globals else nullcontext()
    with scope:
        return dataset_cls(str(Path(root) / name), name)[0].clone()


def load_graph(root, name):
    """从磁盘加载一个图数据集。

    根据数据集的名字，自动判断它属于哪个"家族"（Airport / Blog / Citation），
    然后调用对应的 PyGDA 数据集类来加载。

    参数：
        root: 数据集的根目录路径
        name: 数据集名字，如 'ACMv9', 'BRAZIL', 'Blog1'

    返回值：
        一个 PyTorch Geometric 的 Data 对象，包含：
          - data.x: 节点特征矩阵
          - data.edge_index: 边的连接信息
          - data.y: 节点标签
    """
    from pygda.datasets import AirportDataset, BlogDataset, CitationDataset

    # 数据集名字 → 对应的类
    classes = {
        'Airport': AirportDataset,
        'Blog': BlogDataset,
        'Citation': CitationDataset,
    }

    # 在 FAMILIES 中查找这个名字属于哪个家族
    # FAMILIES 定义在 protocol.py 里：
    #   'Citation': ('ACMv9', 'Citationv1', 'DBLPv7'),
    #   'Airport': ('USA', 'BRAZIL', 'EUROPE'),
    #   'Blog': ('Blog1', 'Blog2'),
    family = next((f for f, names in FAMILIES.items() if name in names), None)
    if family is None:
        raise ValueError(f'Unknown dataset: {name}')

    # PyTorch 2.6+ defaults to weights-only loading, but PyG caches contain Data.
    return _load_pygda_graph(classes[family], Path(root) / family, name)


def prepare_features(data, degree_bins=241):
    """给图数据准备节点特征。

    如果图已经有特征（data.x 不为空），就直接返回。
    如果没有特征（比如机场网络只有连接关系，没有节点属性），
    就用节点的"度"（degree，即每个结点连接了多少条边）来构造特征。

    方法：将度转成 one-hot 编码。
    比如一个节点连接了 5 条边，度 = 5 → 变成一个 241 维的向量，
    第 5 位是 1，其他位是 0。

    参数：
        data: 图数据对象
        degree_bins: one-hot 编码的维度，默认 241（允许度数 0 到 240，超出则报错）

    返回值：
        处理后的图数据对象
    """
    if data.x is None:
        # 计算每个节点的度（连接的边的数量）
        degrees = degree(data.edge_index[0], num_nodes=data.num_nodes).long()
        # 安全检查：如果某个节点的度超过了 241，那 one-hot 编码就不够用了
        if degrees.numel() and degrees.max().item() >= degree_bins:
            raise ValueError('Degree exceeds shared 241-bin one-hot range; no silent clipping')
        # 构造 one-hot 特征：每个节点变成一个 241 维的向量
        data.x = torch.nn.functional.one_hot(degrees, degree_bins).float()
    return data


def prepare_pair(source, target):
    """预处理源域和目标域的一对数据，让它们可以一起训练。

    这个函数做了 4 件事：
    1. 给没有特征的图补充特征（调用 prepare_features）
    2. 检查源域和目标域的特征维度是否一致（必须一致才能共用编码器）
    3. 把源域已有编号压缩成连续编号；目标域按相同原始编号映射。
       前提是数据已经使用共同语义编号；这里不能推断两域类别名称的对应关系。
    4. 把目标域的真实标签"藏起来"——从 data.y 移到 labels 变量，
       这样模型输入的 target.y 不再含真标签；评价器仍持有它们用于打分和选 best。

    参数：
        source: 源域图数据（有标签）
        target: 目标域图数据（有标签，但会被移除）

    返回值：
        source:   处理后的源域
        target:   处理后的目标域（data.y 被设为 None，标签已被移除）
        labels:   目标域的真实标签（给评估器用，不参与训练）
        classes:  所有类别的列表，比如 [0, 1, 2, 3, 4, 5]
    """
    source, target = prepare_features(source.clone()), prepare_features(target.clone())

    # 检查特征维度是否一致（不一致的话编码器没法共用）
    if source.x.size(1) != target.x.size(1):
        raise ValueError('Source and target feature dimensions differ')

    # 找出源域有哪些类别（去重、排序）
    classes = torch.unique(source.y, sorted=True)
    if not len(classes):
        raise ValueError('Source labels cannot be empty')

    # 把源域标签从"原始编号"映射到"统一编号"
    # 比如源域原始标签是 [3, 7, 3, 7]，classes 是 [3, 7]
    # → searchsorted 会找到 3 在 classes 的索引 0，7 的索引 1
    # → 结果就是 [0, 1, 0, 1]
    source.y = torch.searchsorted(classes, source.y).long()

    # 处理目标域标签
    target_labels = target.y.clone()  # 先复制一份
    # 检查目标域的类别是否都在源域中出现过
    if not torch.isin(target_labels, classes).all():
        raise ValueError('Target evaluation contains a class absent from the source label space')
    # 同样把目标域标签映射到统一编号
    labels = torch.searchsorted(classes, target_labels).long()

    # ★ 关键操作：从目标域数据中移除标签！
    # 这是数据流隔离，不是访问权限隔离：训练损失应当只用源标签/生成的软标签。
    target.y = None

    return source, target, labels, classes.cpu().tolist()


# ==============================================================================
#  第三部分：评价指标计算
# ==============================================================================


def metrics(labels, logits, num_classes=None):
    """计算模型预测的 Micro-F1 和 Macro-F1 分数。

    本项目按节点计算单标签分类指标（不是每张图只预测一个类别），简单说：
      - Micro-F1：把所有类别放在一起算总账（每个样本权重相同）
      - Macro-F1：先算每个类别的 F1，再取平均（每个类别权重相同，不管样本多少）

    参数：
        labels: 真实的标签（正确答案）
        logits: 模型输出的原始分数（还没经过 softmax）

    返回值：
        一个字典，如 {'micro_f1': 0.85, 'macro_f1': 0.82}
    """
    if logits.ndim != 2 or logits.size(0) != labels.numel():
        raise ValueError('Prediction shape must be [target_nodes, classes]')
    if num_classes is not None and logits.size(1) != num_classes:
        raise ValueError('Prediction class dimension differs from source label space')
    # 检查 logits 里有没有 NaN 或无穷大（说明训练出问题了）
    if not torch.isfinite(logits).all():
        raise ValueError('Non-finite prediction logits')

    # 取 logits 中最大值的索引作为预测类别
    pred = logits.argmax(1).detach().cpu().numpy()
    truth = labels.detach().cpu().numpy()

    # Macro 的类别集合是 sklearn 默认的真值/预测并集；此处未固定全部源类别。
    # 因此不能把本函数直接当成固定八类 IoU 或缺失类计分协议。
    return {f'{average}_f1': float(f1_score(truth, pred, average=average,
                                          labels=list(range(num_classes)) if num_classes is not None else None,
                                          zero_division=0))
            for average in ('micro', 'macro')}


# ==============================================================================
#  第四部分：计时器
# ==============================================================================


class Timings:
    """精准计时器，记录每个训练阶段花了多少时间。

    用法：
        timers = Timings('cuda:0')
        with timers.stage('data_loading'):
            load_data()            # 这里记"数据加载"的耗时
        with timers.stage('training'):
            train_model()          # 这里记"训练"的耗时
        print(timers.values)       # {'data_loading': 1.2, 'training': 30.5}
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.values = {}  # 存储每个阶段的累计耗时

    def sync(self):
        """同步 GPU 操作。

        GPU 是"异步"的——你调用一个 CUDA 操作，它不会等 GPU 做完就返回。
        如果在计时前不同步，计时器可能少算了 GPU 真正执行的时间。

        比如：
            start = time.time()
            torch.matmul(A, B)  # GPU 可能还没算完就返回了
            end = time.time()   # 这里几乎没花时间！因为 GPU 还在后台算
            正确做法是先 torch.cuda.synchronize() 再计时
        """
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name):
        """记录一个阶段的耗时。

        用法：with timers.stage('my_stage'): ...
        进入时同步+开始计时，退出时同步+停止计时，累加到 self.values[name]
        """
        self.sync()
        start = time.perf_counter()
        try:
            from experiment.efficiency import stage as efficiency_stage
            with efficiency_stage(name):
                yield
        finally:
            self.sync()
            # 累加计时（同一个阶段可能被多次调用，所以是累加而非赋值）
            self.values[name] = self.values.get(name, 0.0) + time.perf_counter() - start


# ==============================================================================
#  第五部分：模型预测器（把编码器和分类头打包成一个完整的模型）
# ==============================================================================


class DiffPredictor(nn.Module):
    """DiffGDA 方法的推理模型封装。"""

    def __init__(self, classifier, prop_steps):
        super().__init__()
        self.classifier = classifier  # 分类器
        self.prop_steps = prop_steps  # 传播步数

    def forward(self, data):
        return self.classifier(data, self.prop_steps)


# ==============================================================================
#  第六部分：评估器（Evaluator）—— 训练场上的"裁判"
# ==============================================================================


class Evaluator:
    """评估器——训练过程中的"裁判"，负责打分和记录。

    Evaluator 持有目标真标签，用于计算 F1 和按 Micro-F1 选 best。
    PP 的训练损失不读取这些真标签；但 best 选择使用了目标监督信息，必须披露。
    这是程序约定的数据流隔离，并非禁止训练对象访问 evaluator.labels 的安全机制。

    工作流程：
      1. 模型每训练完一个 epoch，调用 evaluator(...) 来"汇报"一次
      2. Evaluator 用真实的 labels 算 F1，记录下来
      3. 自动追踪"最好的一次"（best）和"最后一次"（final）的结果
      4. 训练结束后，保存最佳和最终的模型权重

    重要设计：
      - __call__ 方法（第 1 个参数是 predictor，第 2 个是 prediction 函数）
        之所以传函数而非直接传 logits，是为了确保评估时模型处于正确的模式
    """

    def __init__(self, labels, timings, num_classes=None):
        """
        参数：
            labels:  目标域真实标签，用于指标及 best 选择
            timings: 计时器，用来记录评估花了多久
        """
        self.labels = labels          # 评价标签；不应进入方法的训练损失
        self.num_classes = num_classes
        self.timings = timings        # 计时器

        # 评估历史记录
        self.history = []             # 每次评估的结果列表
        self.auxiliary = {}           # 辅助评估结果（如 Source-only 基线）

        # 最佳和最终的结果
        self.best = None              # 最好的一次评估结果（字典）
        self.final = None             # 最后一次评估结果

        # 最佳和最终的模型
        self.best_model = None        # 取得最佳结果时的模型权重
        self.final_model = None       # 训练结束时的模型权重

        # 最佳和最终的预测 logits
        self.best_logits = None
        self.final_logits = None

        self._last_model = None       # 最近一次评估的模型（内部使用）

    @_eff_measured('evaluation', evaluation=True)
    def __call__(self, predictor, prediction, *, round, epoch):
        """★ 核心方法：在目标域上评估模型一次。

        这个方法由模型的训练代码在每个 epoch 回调调用。

        参数：
            predictor:  模型（用于确保评估时模式正确）
            prediction: 一个"能产生预测结果"的函数，调用它得到 logits
            round:      第几轮（某些方法有多轮迭代）
            epoch:      第几个 epoch

        为什么不直接传 logits 而要传函数？
            因为评估时需要确保模型在 eval 模式、不计算梯度、随机状态被隔离。
            传函数可以让 evaluator 自己控制这些环境。
        """
        with self.timings.stage('evaluation'), evaluating(predictor):
            from experiment.efficiency import stage as efficiency_stage
            with efficiency_stage('evaluation_prediction'):
                logits = prediction().detach()  # 得到预测结果

            # 对于 Flow 方法，可以同时记录"原始"和"图优化后"两种结果
            variants = {}
            if getattr(predictor, 'method', None) == 'flow':
                refine = predictor.refine
                try:
                    predictor.refine = False
                    variants['raw'] = prediction().detach().cpu().clone()
                    predictor.refine = True
                    variants['refined'] = prediction().detach().cpu().clone()
                finally:
                    predictor.refine = refine

            # 计算 F1 分数
            record = {
                **metrics(self.labels, logits, self.num_classes),   # micro_f1, macro_f1
                'round': round,
                'epoch': epoch,
                'evaluation_index': len(self.history) + 1,  # 第几次评估
                # 仅由预测得到，用于识别全部节点落入单一类别的塌缩；
                # 不传回训练方法，也不改变模型选择规则。
                'prediction_counts': torch.bincount(
                    logits.argmax(1), minlength=(self.num_classes if self.num_classes is not None
                                                else logits.size(1))).cpu().tolist(),
            }
            if variants:
                record['variants'] = {
                    key: metrics(self.labels, value)
                    for key, value in variants.items()
                }

            # 记录到历史列表
            self.history.append(record)

            # 更新"最终"结果（总是覆盖为最新一次）
            self.final = record
            self.final_logits = logits.cpu().clone()
            self.final_variants = variants
            self._last_model = predictor

            # 只按 Micro-F1 严格提高更新 best；同分保留较早一次。
            # best 的 Macro-F1 来自同一次评价，不另选 Macro 最优的检查点。
            if self.best is None or record['micro_f1'] > self.best['micro_f1']:
                self.best = dict(record)
                self.best_logits = logits.cpu().clone()
                self.best_variants = variants
                # 深拷贝模型（保留取得最佳成绩时的模型状态）
                self.best_model = copy.deepcopy(predictor).cpu().eval()

    def auxiliary_prediction(self, name, predictor, prediction):
        """记录一个辅助评估结果（不参与 best/final 的选择）。

        比如训练 SimpleFlow 时，在正式训练前先评估一个 Source-only 基线，
        就可以用这个方法来记录，方便之后对比。

        参数：
            name:       这个评估的名字，如 'source_only'
            predictor:  模型
            prediction: 产生预测的函数
        """
        with self.timings.stage('auxiliary_evaluation'), evaluating(predictor):
            self.auxiliary[name] = metrics(self.labels, prediction())

    def finish(self):
        """训练结束时调用，保存最终模型。

        把最后一次评估的模型深拷贝一份作为 final_model。
        调用方必须保证最后一次评价后不再更新参数，否则模型可能与 final_logits 不符。
        runner 会重新预测并核对，发现不一致就报错；PP 每轮先更新再评价，满足该约定。
        """
        if self._last_model is None:
            raise RuntimeError('No main classification evaluation was recorded')
        with isolated_rng():
            self.final_model = copy.deepcopy(self._last_model).cpu().eval()

    def predict(self, data, state='best'):
        """用保存的模型对数据进行预测（用于训练后的验证）。

        参数：
            data:  图数据
            state: 'best' 或 'final'，用哪个状态的模型做预测

        返回值：
            预测 logits
        """
        if state not in ('best', 'final'):
            raise ValueError('state must be best or final')
        # 取出保存的模型，搬到数据所在的设备上
        module = copy.deepcopy(getattr(self, state + '_model')).to(data.x.device)
        with evaluating(module):
            return module(data)

    def save(self, directory, metadata):
        """把评估结果和模型保存到磁盘。

        保存的结构：
          directory/
            best.pt          # 最佳模型权重 + 预测结果
            final.pt         # 最终模型权重 + 预测结果
            history.json     # 完整的评估历史

        参数：
            directory: 保存的文件夹路径
            metadata:  附加的元数据（方法名、配置、协议版本等）
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for state in ('best', 'final'):
            module = getattr(self, state + '_model')
            torch.save({
                'predictor': module,                    # 完整的模型对象
                'state_dict': module.state_dict(),       # 模型参数
                'metadata': metadata,                    # 元数据
                'evaluation': getattr(self, state),      # 当时的评估结果
                'logits': getattr(self, state + '_logits'),           # 预测 logits
                'variant_logits': getattr(self, state + '_variants'), # 变体预测（如有）
                'predictions': getattr(self, state + '_logits').argmax(1),  # 预测类别
            }, directory / (state + '.pt'))

        # 保存完整的评估历史（每次评估的 micro_f1 / macro_f1）
        (directory / 'history.json').write_text(
            json.dumps(self.history, indent=2, allow_nan=False), encoding='utf-8'
        )


# ==============================================================================
#  第七部分：工具函数——从检查点恢复预测
# ==============================================================================


def predict_checkpoint(path, data, device='cpu'):
    """加载一个之前保存的模型检查点，对数据进行预测。

    这个函数用于训练完成后，用保存的最佳/最终模型对新数据做预测。

    参数：
        path:    检查点文件路径（.pt 文件）
        data:    要预测的图数据
        device:  使用的设备（默认 CPU）

    返回值：
        模型预测的 logits
    """
    # 加载检查点
    # weights_only=False 是因为检查点里保存的是完整的 Python 对象（不仅仅是权重）
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # 取出模型
    module = checkpoint['predictor'].to(device)
    # 在评估模式下做预测
    with evaluating(module):
        return module(data.clone().to(device))
