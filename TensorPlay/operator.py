from typing import List, Optional, Union, Tuple
from .core import  Operator, Tensor
import numpy as np

# =============================================================================
# 四则运算算子
# =============================================================================
class Add(Operator):
    """加法算子"""

    def _forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        return Tensor(a + b)

    def _backward(self) -> List[Tensor]:
        ga, gb = self.out().grad, self.out().grad
        if self.inp[0].shape != self.inp[1].shape:
            ga = ga.rebroadcast(self.inp[0].shape)
            gb = gb.rebroadcast(self.inp[1].shape)
        return [ga, gb]

def add(a: Tensor, b: Tensor) -> Tensor:
    return Add()(a, b)

class Sub(Operator):
    """减法算子"""

    def _forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        return Tensor(a - b)

    def _backward(self) -> List[Tensor]:
        ga, gb = self.out().grad, -self.out().grad
        if self.inp[0].shape != self.inp[1].shape:
            ga = ga.rebroadcast(self.inp[0].shape)
            gb = gb.rebroadcast(self.inp[1].shape)
        return [ga, gb]

def sub(a: Tensor, b: Tensor) -> Tensor:
    return Sub()(a, b)

def rsub(a: Tensor, b: Tensor) -> Tensor:
    return Sub()(b, a)

class Neg(Operator):
    """取负算子"""

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(-a)

    def _backward(self) -> List[Tensor]:
        return [-self.out().grad]


def neg(a: Tensor) -> Tensor:
    return Neg()(a)


class Mul(Operator):
    """乘法算子"""

    def _forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        return Tensor(a * b)

    def _backward(self) -> List[Tensor]:
        ga = self.out().grad * self.inp[1]
        gb = self.out().grad * self.inp[0]
        if self.inp[0].shape != self.inp[1].shape:
            ga = ga.rebroadcast(self.inp[0].shape)
            gb = gb.rebroadcast(self.inp[1].shape)
        return [ga, gb]


def mul(a: Tensor, b: Tensor) -> Tensor:
    return Mul()(a, b)


def rmul(a: Tensor, b: Tensor) -> Tensor:
    return Mul()(b, a)


class Div(Operator):
    """除法算子"""

    def _forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        return Tensor(a / b)

    def _backward(self) -> List[Tensor]:
        ga, gb = (self.inp[1] ** -1) * self.out().grad, (-self.inp[0] * self.inp[1] ** -2) * self.out().grad
        if self.inp[0].shape != self.inp[1].shape:
            ga = ga.rebroadcast(self.inp[0].shape)
            gb = gb.rebroadcast(self.inp[1].shape)
        return [ga, gb]


def div(a: Tensor, b: Tensor) -> Tensor:
    return Div()(a, b)


def rdiv(a: Tensor, b: Tensor) -> Tensor:
    return Div()(b, a)


class Pow(Operator):
    """幂算子"""

    def __init__(self, power: float):
        super().__init__()
        self.power = power

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(a ** self.power)

    def _backward(self) -> List[Tensor]:
        # 输入梯度 += n * x ^ (n - 1) * 输出梯度
        return [self.power * self.inp[0] ** (self.power - 1) * self.out().grad]


def ten_pow(a: Tensor, b: float) -> Tensor:
    return Pow(b)(a)


class Exp(Operator):
    """自然指数算子"""

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(np.exp(a))

    def _backward(self) -> List[Tensor]:
        # 输入梯度 += e ^ x * 输出梯度 = 输出值 * 输出梯度
        return [self.out().data * self.out().grad]

def exp(x: Tensor) -> Tensor:
    return Exp()(x)


class Log(Operator):
    """自然对数算子"""

    def _forward(self, a: np.ndarray) -> Tensor:
        # 防止对数输入为非正数
        data = a.copy()
        data[data <= 0] = 1e-10
        return Tensor(np.log(data))

    def _backward(self) -> List[Tensor]:
        # 输入梯度 += (1 / x) * 输出梯度 = 输出梯度 * (输入值的倒数)
        return [self.out().grad * self.inp[0] ** -1]


def log(x: Tensor) -> Tensor:
    return Log()(x)


class Broadcast(Operator):
    """广播算子"""

    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def _forward(self, x: np.ndarray) -> Tensor:
        return Tensor(np.broadcast_to(x, self.shape))

    def _backward(self) -> List[Tensor]:
        return [rebroadcast(self.out().grad, *self.inp[0].shape)]


def broadcast(x: Tensor, *shape: int) -> Tensor:
    if x.shape == shape:
        return x
    return Broadcast(*shape)(x)


class Rebroadcast(Operator):
    """逆广播算子"""

    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def _forward(self, x: np.ndarray) -> Tensor:
        ndim = len(self.shape)
        lead = x.ndim - ndim
        lead_axis = tuple(range(lead))
        axis = tuple([i + lead for i, sx in enumerate(self.shape) if sx == 1])
        y = x.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
            y = y.squeeze(lead_axis)
        return Tensor(y)

    def _backward(self) -> List[Tensor]:
        return [broadcast(self.out().grad, *self.inp[0].shape)]


def rebroadcast(x: Tensor, *shape: int) -> Tensor:
    if x.shape == shape:
        return x
    return Rebroadcast(*shape)(x)


class Sum(Operator):
    """元素级求和运算符"""

    def __init__(self, axis: Optional[int] = None, dims: bool = False):
        super().__init__()
        self.axis = axis
        self.dims = dims

    def _forward(self, a: Tensor) -> Tensor:
        return Tensor(a.data.data.sum(axis=self.axis, keepdims=self.dims), op=self)

    def _backward(self) -> List[Tensor]:
        # 扩展梯度以匹配原始形状
        g = self.out().grad
        if self.axis is not None and not self.dims:
            old = list(self.inp.shape)
            old[self.axis] = 1
            g = self.out().grad.reshape(*tuple(old))
        return [g.repeat(self.inp.shape[self.axis], axis=self.axis)]


class Reshape(Operator):
    """重塑形状运算符"""

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.re = shape

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(a.data.data.reshape(*self.re), op=self)
        return out

    def _backward(self) -> List[Tensor]:
        return [self.out().grad.reshape(self.inp.shape)]


class Transpose(Operator):
    """转置算子"""

    def __init__(self, *axes: int):
        super().__init__()
        self.axes = axes

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(a.transpose(*self.axes))

    def _backward(self) -> List[Tensor]:
        reverse_axes = tuple(np.argsort(self.axes))  # 计算逆变换
        return [self.out().grad.transpose(*reverse_axes)]


def transpose(x: Tensor, *axes: int) -> Tensor:
    return Transpose(*axes)(x)


@property
def T(self: Tensor) -> Tensor:
    return Transpose(*range(self.ndim - 1, -1, -1))(self)


class Slice(Operator):
    """切片算子（不支持重复索引）"""

    def __init__(self, slices: Union[slice, Tuple[slice, ...]]):
        super().__init__()
        self.slices = slices if isinstance(slices, tuple) else (slices,)

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(a[self.slices])

    def _backward(self) -> List[Tensor]:
        return [reslice(self.out().grad, self.slices, self.inp[0].shape)]


def ten_slice(x: Tensor, slices) -> Tensor:
    return Slice(slices)(x)


class Reslice(Operator):
    """逆切片算子（不支持重复索引）"""

    def __init__(self, slices: Union[slice, Tuple[slice, ...]], in_shape: Tuple[int, ...]):
        super().__init__()
        self.slices = slices if isinstance(slices, tuple) else (slices,)
        self.in_shape = in_shape

    def _forward(self, a: np.ndarray) -> Tensor:
        a_grad = np.zeros(self.in_shape)
        a_grad[self.slices] = a
        return Tensor(a_grad)

    def _backward(self) -> List[Tensor]:
        return [ten_slice(self.out().grad, self.slices)]


def reslice(x: Tensor, slices, shape) -> Tensor:
    return Reslice(slices, shape)(x)


class Mean(Operator):
    """平均值运算符"""

    def __init__(self, axis: Optional[int] = None, dims: bool = False):
        super().__init__()
        self.axis = axis
        self.dims = dims

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(a.mean(axis=self.axis, keepdims=self.dims))

    def _backward(self) -> Tensor:
        if self.axis is None:
            normalizer = self.inp[0].size
        else:
            normalizer = self.inp[0].shape[self.axis]
        g = self.out().grad
        if self.axis is not None and not self.dims:
            old = list(self.inp.data.data.shape)
            old[self.axis] = 1
            g = self.out().grad.reshape(*tuple(old))
        return g.repeat(self.inp.shape[self.axis], axis=self.axis) / normalizer


class Concatenate(Operator):
    """拼接运算符"""

    def __init__(self, axis: int = 0):
        super().__init__()
        self.axis = axis

    def _forward(self, tensors: List[Tensor]) -> Tensor:
        if not tensors:
            raise ValueError("Input tensor list is empty!")
        # 检查拼接维度外的其他维度是否匹配
        shapes = [t.data.data.shape for t in tensors]
        for i in range(len(shapes[0])):
            if i != self.axis:
                dims = {s[i] for s in shapes}
                if len(dims) > 1:
                    raise ValueError(f"All input tensors must have the same size in dimension {i}")
        out = Tensor(np.concatenate([t.data.data for t in tensors], axis=self.axis), op=self)
        return out

    def _backward(self) -> List[Tensor]:
        # 计算每个张量对应的梯度切片
        current = 0
        g = []
        for tensor in self.inp:
            size = tensor.shape[self.axis]
            slices = [slice(None)] * self.out().ndim
            slices[self.axis] = slice(current, current + size)
            g.append(self.out().grad.slice(tuple(slices)))
            current += size
        return g


class Repeat(Operator):
    """重复运算符"""

    def __init__(self, repeats: Union[int, Tuple[int, ...]], axis: Optional[int] = None):
        super().__init__()
        self.repeats = repeats
        self.axis = axis

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(np.repeat(a.data.data, repeats=self.repeats, axis=self.axis), op=self)
        return out

    def _backward(self) -> Tensor:
        if self.axis is None:
            n = self.inp.data.data.size
            if isinstance(self.repeats, tuple):
                repeats = self.repeats
            else:
                repeats = (self.repeats,) * n
            indices = []
            current = 0
            for rep in repeats:
                indices.append((current, current + rep))
                current += rep
        else:
            if isinstance(self.repeats, tuple):
                repeats = self.repeats
            else:
                repeats = (self.repeats,) * self.inp.data.data.shape[self.axis]
            indices = []
            current = 0
            for rep in repeats:
                indices.append((current, current + rep))
                current += rep
        return self.out().grad.compress(indices=indices, repeats=self.repeats, axis=self.axis)


class Compress(Operator):
    """压缩运算符"""

    def __init__(self, indices: List[slice], repeats: Union[int, Tuple[int, ...]], axis: Optional[int] = None):
        super().__init__()
        self.indices = indices
        self.repeats = repeats
        self.axis = axis

    def _forward(self, a: Tensor) -> Tensor:
        out = np.zeros(len(self.indices))
        if self.axis is None:
            for i, (start, end) in enumerate(self.indices):
                out.flat[i] += a.data.data.flat[slice(start, end)]
        else:
            for i, (start, end) in enumerate(self.indices):
                slices = [slice(None)] * out.ndim
                slices[self.axis] = slice(start, end)
                target_slices = [slice(None)] * out.ndim
                target_slices[self.axis] = slice(i, i + 1)
                out[tuple(target_slices)] += np.sum(a.data.data[tuple(slices)], axis=self.axis)
        out = Tensor(out, op=self)
        return out

    def _backward(self) -> Tensor:
        return self.out().grad.repeat(repeats=self.repeats, axis=self.axis)


class Relu(Operator):
    """ReLU修正线性单元"""

    def _forward(self, a: np.ndarray) -> Tensor:
        return Tensor(np.maximum(a, 0))

    def _backward(self) -> Tensor:
        mask = Tensor(self.inp[0].data >= 0)
        return self.out().grad * mask


def relu(a: Tensor) -> Tensor:
    return Relu()(a)


class Sigmoid(Operator):
    """Sigmoid激活函数"""

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(1 / (1 + np.exp(-a.data.data)), op=self)
        return out

    def _backward(self) -> Tensor:
        # 输入梯度 += σ(x) * (1 - σ(x)) * 输出梯度
        return self.out() * (1 - self.out()) * self.out().grad


class Tanh(Operator):
    """Tanh激活函数"""

    def _forward(self, a: Tensor) -> Tensor:
        out = Tensor(np.tanh(a.data.data), op=self)
        return out

    def _backward(self):
        # 输入梯度 += (1 - tanh²(x)) * 输出梯度
        return (1 - self.out() ** 2) * self.out().grad


class MeanSquaredError(Operator):
    """均方误差运算符"""

    def _forward(self, a: Tensor, b: Tensor) -> Tensor:
        out = Tensor(np.square((a.data.data - b.data.data).sum(axis=-1, keepdims=True) / a.data.data.shape[-1]), op=self)
        return out

    def _backward(self) -> List[Tensor]:
        g = self.out().grad.repeat(self.inp[0].shape[-1], axis=-1) * 2 * (self.inp[0] - self.inp[1]) / self.inp[0].shape[-1]
        return [g, -g]


class MatMul(Operator):
    """矩阵乘法运算符"""

    def _forward(self, a: np.ndarray, b: np.ndarray) -> Tensor:
        return Tensor(a @ b)

    def _backward(self) -> List[Tensor]:
        return [self.out().grad @ self.inp[1].transpose(-2, -1),
               self.inp[0].transpose(-2, -1) @ self.out().grad]


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatMul()(a, b)


class DenseOp(Operator):
    """线性层运算符"""

    def __init__(self, matrix: 'Dense'):
        super().__init__()
        self.w = matrix.w
        self.b = matrix.b
        self.bias = matrix.bias

    def _forward(self, x: Tensor) -> Tensor:
        a = x.data.data
        if a.ndim == 1:
            a = a.reshape(1, -1)  # 单个样本转为 (1 × in_features)

        # 矩阵乘法：W @ x.T = (out_features × batch_size)
        out = self.w.data.data @ a.T
        if self.bias is not None:
            out += self.b.data.data.repeat(a.shape[0], axis=1)

        # 转置回 (batch_size × out_features)
        if x.ndim == 1:
            out = out.reshape(-1)
        out = Tensor(out.T, op=self)
        return out

    def _backward(self):
        # 1. 处理输出梯度形状 (batch_size × out_features → out_features × batch_size)
        grad = self.out().grad
        a = self.inp
        if self.inp.ndim == 1:
            grad = grad.reshape(1, -1)
            a = a.reshape(1, -1)
        grad = grad.transpose(1, 0)
        # 2. 计算偏置梯度（对所有样本梯度求和）
        if self.bias is not None:
            if self.b.grad is None:
                self.b.grad = grad.sum(axis=1, dims=True)
            else:
                self.b.grad = self.b.grad + grad.sum(axis=1, dims=True)
        # 3. 矩阵乘法的反向传播（计算权重梯度）
        if self.w.grad is None:
            self.w.grad = grad @ a
        else:
            self.w.grad = self.w.grad + grad @ a
        b = (self.w.transpose(1, 0) @ grad).transpose(1, 0)
        if self.inp.ndim == 1:
            b = b.reshape(*self.inp.shape)
        return b