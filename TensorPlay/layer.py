from typing import List, Union, Tuple
import json
import numpy as np
from .core import Tensor, Layer, to_data
from .operator import DenseOp, BatchNormOp
from .initializer import he_init, my_init


class ConstantTensor(Tensor, Layer):
    """单张量参数层"""

    def __init__(self, data: Union[list, np.ndarray]):
        Tensor.__init__(self, data)
        Layer.__init__(self)

    def save(self) -> str:
        return json.dumps(self.data.tolist())

    def load(self, text: str) -> None:
        self.data = to_data(json.loads(text))

    def param(self):
        return [self]


class LayerNorm(Layer):
    """标准化处理层，带可学习参数（Layer Normalization）"""

    def __init__(self, shape: Union[int, Tuple[int, ...]], eps: float = 1e-4):
        self.w = Tensor(np.random.normal(0, 0.04, shape))
        self.b = Tensor(np.random.normal(0, 0.04, shape))
        self.eps = eps
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """前向传播：计算标准化并应用可学习参数"""
        # 计算方差
        y = x - x.mean(axis=0, dims=True)
        var = (y ** 2).mean(axis=0, dims=True)
        # 计算标准化结果
        x_normalized = y / ((var + self.eps) ** 0.5)
        return self.w * x_normalized + self.b

    def save(self) -> str:
        text = {'w': self.w.data.data.tolist(), 'b': self.b.data.data.tolist()}
        return json.dumps(text)

    def load(self, text: str):
        parts = json.loads(text)
        self.w = Tensor(parts['w'])
        self.b = Tensor(parts['b'])

    def param(self) -> List[Tensor]:
        return [self.w, self.b]


class Dense(Layer):
    """全连接层"""

    def __init__(self, inp_size: int, out_size: int, bias=True):
        self.w = he_init(inp_size, out_size)
        self.bias = bias
        if bias:
            self.b = Tensor.zeros((out_size, 1))
        super().__init__()

    def forward(self, a: Tensor) -> Tensor:
        return DenseOp(self)(a)

    def save(self) -> str:
        text = {'w': self.w.data.tolist()}
        if self.bias:
            text['b'] = self.b.data.tolist()
        return json.dumps(text)

    def load(self, text: str) -> None:
        parts = json.loads(text)
        self.w = Tensor(parts['w'])
        if self.bias:
            self.b = Tensor(parts['b'])

    def param(self) -> List[Tensor]:
        if self.bias:
            return [self.w, self.b]
        return [self.w]


class BatchNorm(Layer):
    """批量归一化层"""
    def __init__(self, channels: int, eps: float = 1e-4, decay: float = 0.9):
        self.eps = eps
        self.decay = decay
        self.gamma = Tensor(np.ones(channels))
        self.beta = Tensor(np.zeros(channels))
        self.running_mean = np.zeros(channels)
        self.running_var = np.ones(channels)
        super().__init__()

    def forward(self, a: Tensor) -> Tensor:
        return BatchNormOp(self)(a)

    def save(self) -> str:
        text = {'gamma': self.gamma.data.tolist(), 'beta': self.beta.data.tolist(),
                'running_mean': self.running_mean.tolist(), 'running_var': self.running_var.tolist()}
        return json.dumps(text)

    def load(self, text: str) -> None:
        parts = json.loads(text)
        self.gamma = Tensor(parts['w'])
        self.beta = Tensor(parts['b'])
        self.running_mean = np.array(parts['running_mean'])
        self.running_var = np.array(parts['running_var'])

    def param(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class Conv2D(Layer):
    def __init__(self, width: int, height: int, stride_w=1, stride_h=1, pad=True, bias=True):
        """
        2d卷积层
        :param width: int 卷积核的宽度（如填充，请设为奇数）
        :param height: int 卷积核的高度（如填充，请设为奇数）
        :param stride_w: int 横向的步长
        :param stride_h: int 纵向的步长
        :param pad: bool 是否进行填充（使运算的输入和输出的大小一样）
        :param bias: bool 是否加上偏置
        """
        self.width = width
        self.height = height
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad = pad
        self.kernel = my_init(width * height)
        self.bias = bias
        if bias:
            self.b = my_init(1)
        super().__init__()

    def padding(self, x):
        """
        填充
        :param x: list[Tensor(),Tensor()...]  2d的Ten，或者说列表包着的一列Ten
        :return: list[Tensor(),Tensor()...]
        """
        pad_x = (self.stride_w * (len(x[0]) - 1) - len(x[0]) + self.width) // 2
        pad_y = (self.stride_h * (len(x) - 1) - len(x) + self.height) // 2
        x2 = []
        for i in range(pad_y):
            x2.append(Tensor.zeros(len(x[0]) + pad_x * 2))
        for i in range(len(x)):
            x2.append(Tensor.connect([Tensor.zeros(pad_x), x[i], Tensor.zeros(pad_x)]))
        for i in range(pad_y):
            x2.append(Tensor.zeros(len(x[0]) + pad_x * 2))
        return x2

    def forward(self, x):
        """
        进行运算
        :param x: list[Tensor(),Tensor()...]  2d的Ten，或者说list包着的一列Ten
        :return: list[Tensor(),Tensor()...]
        """
        if self.pad:
            x = self.padding(x)
        x2 = []
        for y_pos in range(0, len(x) - self.height + 1, self.stride_h):
            x2line = []
            for x_pos in range(0, len(x[0]) - self.width + 1, self.stride_w):
                window = Tensor.connect([x[y_pos + i].cut(x_pos, x_pos + self.width) for i in range(self.height)])
                v = (window * self.kernel).sum()
                if self.bias:
                    v += self.b
                x2line.append(v)
            x2.append(Tensor.connect(x2line))
        return x2

    def save(self):
        t = f"{self.width}/{self.height}/{self.kernel.data}/{self.stride_w}/{self.stride_h}/{self.pad}"
        if self.bias:
            t += f"/{self.b.data}"
        return t

    def load(self, t):
        t = t.split("/")
        self.width = int(t[0])
        self.height = int(t[1])
        self.kernel = Tensor(eval(t[2]))
        self.stride_w = int(t[3])
        self.stride_h = int(t[4])
        self.pad = eval(t[5])
        if len(t) == 7:
            self.bias = True
            self.b = Tensor(eval(t[6]))
        else:
            self.bias = False

    def grad_descent_zero(self, lr):
        self.kernel.data -= self.kernel.grad * lr
        self.kernel.zero_grad()

    def param(self):
        return [self.kernel, self.b]


# 一些复杂结构的简单实现，用于实验，未更新到Module管理
class MiniDense:
    """低秩全连接层"""

    def __init__(self, inp_size, out_size, midsize=None, bias=True):
        if midsize is None:
            midsize = round(((inp_size + out_size) / 2) ** 0.5)
        self.f1 = Dense(inp_size, midsize, bias)
        self.f2 = Dense(midsize, out_size, bias)

    def __call__(self, x):
        x = self.f1(x)
        x = self.f2(x)
        return x

    def grad_descent_zero(self, lr):
        self.f1.grad_descent_zero(lr)
        self.f2.grad_descent_zero(lr)


class Attention:
    """单头自注意力模块"""

    def __init__(self, emb_size, qk_size=None, v_size=None):
        """
        :param emb_size: int 输入词向量维度
        :param qk_size: int q、k维度
        :param v_size: int 输出词向量维度，默认与输入相同
        """
        if qk_size is None:
            qk_size = emb_size // 2
        if v_size is None:
            v_size = emb_size
        self.q = Dense(emb_size, qk_size)
        self.k = Dense(emb_size, qk_size)
        self.v = Dense(emb_size, v_size)
        self.emb_size = emb_size
        self.qk_size = qk_size
        self.outsize = v_size

    def __call__(self, x, mask_list=None, tri_mask=False):
        """
        :param x: list[Tensor,Tensor...]  装着词向量的列表
        :param mask_list: list[int,int...] 用于在softmax前盖住填充，输入中表中为1的位置会被替换为-inf
        :param tri_mask: bool 是否使用三角掩码（在计算注意力权重时只关注当前和之前的词）
        :return: list[Tensor,Tensor...]
        """
        q_list = []
        k_list = []
        v_list = []
        for w in x:
            q_list.append(self.q(w))
            k_list.append(self.k(w))
            v_list.append(self.v(w))
        att_list = []
        for i in range(len(q_list)):
            line = []
            for j in range(len(k_list)):
                if (mask_list is not None and (mask_list[i] == 1 or mask_list[j] == 1)) or (tri_mask and j > i):
                    line.append(Tensor([float("-inf")]))
                else:
                    line.append((q_list[i] * k_list[j]).sum() / Tensor([self.qk_size ** 0.5]))
            att_list.append(Tensor.connect(line).softmax())
        new_v_list = []
        for i in range(len(q_list)):
            line = Tensor.zeros(self.outsize)
            for j in range(len(q_list)):
                line += v_list[j] * (att_list[i].cut(j, j + 1).repeat(self.outsize))
            new_v_list.append(line)
        return new_v_list

    def grad_descent_zero(self, k):
        self.q.grad_descent_zero(k)
        self.k.grad_descent_zero(k)
        self.v.grad_descent_zero(k)


class LSTM:
    """长短期记忆网络"""

    def __init__(self, emb_size, out_size):
        self.for_gate = Dense(emb_size + out_size, out_size)
        self.inp_gate1 = Dense(emb_size + out_size, out_size)
        self.inp_gate2 = Dense(emb_size + out_size, out_size)
        self.out_gate = Dense(emb_size + out_size, out_size)
        self.h = Tensor.zeros(out_size)
        self.s = Tensor.zeros(out_size)

    def __call__(self, x):
        out = []
        for i in x:
            i = Tensor.connect([i, self.h])
            self.s *= self.for_gate(i).sigmoid()
            self.s += self.inp_gate1(i).sigmoid() * self.inp_gate2(i).tanh()
            self.h = self.out_gate(i).sigmoid() * self.s.tanh()
            out.append(self.h)
        return out

    def grad_descent_zero(self, lr):
        self.for_gate.grad_descent_zero(lr)
        self.inp_gate1.grad_descent_zero(lr)
        self.inp_gate2.grad_descent_zero(lr)
        self.out_gate.grad_descent_zero(lr)


class RNN:
    """线性循环神经网络"""

    def __init__(self, emb_size, out_size):
        """
        :param emb_size: int 输入的向量大小
        :param out_size: int 输出的向量大小
        """
        self.out_size = out_size
        self.f1 = Dense(emb_size + out_size, out_size)

    def __call__(self, x):
        """
        :param x: list[Tensor,Tensor...]
        :return: list[Tensor,Tensor...]
        """
        hidden = Tensor.zeros(self.out_size)
        out = []
        for i in x:
            hidden = self.f1(Tensor.connect([hidden, i]))
            out.append(hidden)
        return out

    def grad_descent_zero(self, lr):
        self.f1.grad_descent_zero(lr)