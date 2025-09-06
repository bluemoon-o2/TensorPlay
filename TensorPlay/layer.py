import json
from .core import Layer, to_data
from .operator import *
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


class LayerNorm(Layer):
    """层归一化层"""

    def __init__(self, eps: float = 1e-4):
        self.eps = eps
        self.gamma = Tensor(1)
        self.beta = Tensor(0)
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return LayerNormOp(self)(x)

    def save(self) -> str:
        text = {'gamma': self.gamma.data.tolist(), 'beta': self.beta.data.tolist()}
        return json.dumps(text)

    def load(self, text: str):
        parts = json.loads(text)
        self.gamma = Tensor(parts['gamma'])
        self.beta = Tensor(parts['beta'])

    def param(self) -> List[Tensor]:
        return [self.gamma, self.beta]


class Conv2D(Layer):
    """二维卷积层"""

    def __init__(self, width: int, height: int, stride_w=1, stride_h=1, pad=True, bias=True):
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

    def forward(self, x):
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
        pass

    def load(self, t):
        t = t.split("/")


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