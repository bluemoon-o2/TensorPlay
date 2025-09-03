import os
import subprocess
from .core import Tensor, Operator
from .func import mse

# =============================================================================
# Training Utils
# =============================================================================
def train_on_batch(model, batch, optimizer, loss_fn=mse):
    """
    训练一个批次并返回损失和准确率
    :param model: 模型实例
    :param batch: 批次数据 (inputs, labels)
    :param optimizer: 优化器
    :param loss_fn: 损失函数 (output, label)
    :return: (loss, accuracy)
    """
    inputs, labels = batch
    outputs = [model(i) for i in inputs]

    sample_losses = [loss_fn(out, label) for out, label in zip(outputs, labels)]
    loss = Tensor.mean(sample_losses)

    correct = 0
    total = len(outputs)
    for out, label in zip(outputs, labels):
        # 二分类阈值判断
        pred = 1 if out.data.data[0] > 0.5 else 0
        if pred == label.data.data[0]:
            correct += 1
    accuracy = correct / total

    loss.backward()
    optimizer.step()

    return loss.data.data[0], accuracy


def valid_on_batch(model, batch, loss_fn=mse):
    """
    验证一个批次并返回损失和准确率
    :param model: 模型实例
    :param batch: 批次数据 (inputs, labels)
    :param loss_fn: 损失函数 (output, label)
    :return: (loss, accuracy)
    """
    inputs, labels = batch
    outputs = [model(i) for i in inputs]
    sample_losses = [loss_fn(out, label) for out, label in zip(outputs, labels)]
    loss = Tensor.mean(sample_losses)

    correct = 0
    total = len(outputs)
    for out, label in zip(outputs, labels):
        # 二分类阈值判断
        pred = 1 if out.data.data[0] > 0.5 else 0
        if pred == label.data.data[0]:
            correct += 1
    accuracy = correct / total

    return loss.data.data[0], accuracy

# =============================================================================
# Graph Utils
# =============================================================================
def _dot_ten(v: Tensor, verbose=False):
    """绘制张量节点"""
    name = '' if v.name is None else v.name
    if verbose:
        if v.name is not None:
            name += ': '
        if v.ndim > 1:
            name += f"{v.shape} "
        name += f"{v.dtype}"
    text = f'{id(v)} [label="{name}", color=orange, style=filled]\n'
    if v.op is not None:
        text += f'{id(v.op)} -> {id(v)}\n'
    return text


def _dot_op(op: Operator):
    """绘制算符节点"""
    text = f'{id(op)} [label="{op.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'
    dot_edge = '{} -> {}\n'
    inp = op.inp if isinstance(op.inp, list) else [op.inp]
    for x in inp:
        text += dot_edge.format(id(x), id(op))
    return text


def _trace_dot_graph(out: Tensor, verbose=False):
    """获取计算图"""
    text = ''
    op_set = set()
    queue = []
    # 从输出张量开始，收集所有相关运算符
    if out.op is not None:
        queue.append(out.op)
        op_set.add(out.op)
    text += _dot_ten(out, verbose)
    # 广度优先搜索收集所有相关运算符
    while queue:
        current_op = queue.pop(0)  # 取出队首元素
        text += _dot_op(current_op)
        if current_op.inp is None:
            continue
        inputs = current_op.inp if isinstance(current_op.inp, list) else [current_op.inp]
        for inp_tensor in inputs:
            if inp_tensor.op is not None and inp_tensor.op not in op_set:
                queue.append(inp_tensor.op)
                op_set.add(inp_tensor.op)
            text += _dot_ten(inp_tensor, verbose)

    return 'digraph g {\n' + text + '}'


def _instill_dot_graph(book: list, verbose=False):
    """提取计算图"""
    text = ''
    for op in book:
        text += _dot_op(op)
        current = op.inp if isinstance(op.inp, list) else [op.inp]
        for inp_tensor in current:
            text += _dot_ten(inp_tensor, verbose)
    return 'digraph g {\n' + text + '}'


def plot_dot_graph(source, verbose=False, path='graph.png'):
    """绘制计算图"""
    if isinstance(source, Tensor):
        graph = _trace_dot_graph(source, verbose)
    elif isinstance(source, list):
        graph = _instill_dot_graph(source, verbose)
    else:
        raise TypeError("Graph source must be a Tensor or a list of Operators")
    tmp_dir = os.path.join(os.path.expanduser('~'), '.TensorPlay')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')
    # 保存计算图
    with open(graph_path, 'w') as f:
        f.write(graph)
    # 调用dot命令生成图片
    extension = os.path.splitext(path)[1][1:]
    cmd = f'dot {graph_path} -T{extension} -o {path}'
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Graphviz Error: {e}")
