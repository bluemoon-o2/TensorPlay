import ast
import random
import math


class Vector(list):
    """向量，可以进行按位加减乘等操作"""

    def __init__(self, data):
        # 检查每个元素是否为数字（int或float）
        for item in data:
            if not isinstance(item, (int, float)):
                raise TypeError(f"Vector elements must be numbers (not {type(item)})")
        super().__init__(data)

    def __repr__(self):
        return f"Vector({list(self)})"

    def __str__(self):
        return str(list(self))

    def __neg__(self):
        return Vector([-i for i in self])

    def __add__(self, other):
        if isinstance(other, (int, float)):
            # 向量 + 标量：每个元素加标量
            return Vector([x + other for x in self])
        elif isinstance(other, Vector):
            # 向量 + 向量：对应元素相加
            if len(self) != len(other):
                raise ValueError(f"Cannot add vectors of different lengths ({len(self)} vs {len(other)})")
            return Vector([i + j for i, j in zip(self, other)])
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Vector' and '{type(other)}'")

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, (int, float)):
            # 向量 + 标量：每个元素加标量
            for i in range(len(self)):
                self[i] += other
        elif isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError(f"Cannot add vectors of different lengths ({len(self)} vs {len(other)})")
            for i in range(len(self)):
                self[i] += other[i]
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'Vector' and '{type(other)}'")
        return self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        # 向量×标量：每个元素乘以标量
        if isinstance(other, (int, float)):
            return Vector([x * other for x in self])
        # 向量×向量：对应元素相乘
        elif isinstance(other, Vector):
            if len(self) != len(other):
                raise ValueError(f"Cannot multiply vectors of different lengths ({len(self)} vs {len(other)})")
            return Vector([i * j for i, j in zip(self, other)])
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector' and '{type(other)}'")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (int, float)):
            # 向量 * 标量：每个元素乘以标量
            for i in range(len(self)):
                self[i] *= other
        elif isinstance(other, Vector):
            # 向量 * 向量：对应元素相乘
            if len(self) != len(other):
                raise ValueError(f"Cannot multiply vectors of different lengths ({len(self)} vs {len(other)})")
            for i in range(len(self)):
                self[i] *= other[i]
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector' and '{type(other)}'")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            # 向量 / 标量：每个元素除以标量
            return Vector([x / other for x in self])
        elif isinstance(other, Vector):
            # 向量 / 向量：对应元素相除
            if len(self) != len(other):
                raise ValueError(f"Cannot divide vectors of different lengths ({len(self)} vs {len(other)})")
            return Vector([i / j for i, j in zip(self, other)])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector' and '{type(other)}'")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            # 标量 / 向量：每个元素除以标量
            return Vector([other / x for x in self])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other)}' and 'Vector'")

    def __pow__(self, power: (int, float)):
        if isinstance(power, (int, float)):
            return Vector([i ** power for i in self])
        else:
            raise TypeError(f"Cannot power Vector with {type(power)}")

    def __rpow__(self, base: (int, float)):
        if isinstance(base, (int, float)):
            return Vector([base ** x for x in self])
        else:
            raise TypeError(f"Cannot power {type(base)} with Vector")


class Tensor:
    """张量，带梯度的向量，梯度是另一个同维向量"""

    def __init__(self, data, op=None):
        self.data = Vector(data)
        self.grad = Vector([0] * len(self.data))
        self.op = op

    def __repr__(self):
        return f"Tensor({list(self.data)}, grad={list(self.grad)})"

    def __len__(self):
        return len(self.data)

    # 重载运算方法，通过运算符类实现（关联计算图）
    def __add__(self, other):
        op = Add()
        return op.compute(self, other)

    def __sub__(self, other):
        op = Sub()
        return op.compute(self, other)

    def __neg__(self):
        op = Sub()
        return op.compute(Tensor([0] * len(self)), self)

    def __mul__(self, other):
        op = Mul()
        return op.compute(self, other)

    def __truediv__(self, other):
        op = Div()
        return op.compute(self, other)

    def __pow__(self, power: (int, float)):
        op = Pow()
        return op.compute(self, power)

    def sum(self):
        op = Sum()
        return op.compute(self)

    def relu(self):
        op = Relu()
        return op.compute(self)

    def exp(self):
        op = Exp()
        return op.compute(self)

    def log(self):
        op = Log()
        return op.compute(self)

    def sigmoid(self):
        op = Sigmoid()
        return op.compute(self)

    def tanh(self):
        op = Tanh()
        return op.compute(self)

    def cut(self, start: int, end: int):
        op = Cut()
        return op.compute(self, start, end)

    @classmethod
    def connect(cls, x: list['Tensor']):
        op = Connect()
        return op.compute(x)

    def clone(self):
        """创建当前张量的副本，确保梯度独立"""
        cloned_tensor = Tensor(self.data.copy(), op=self.op)
        cloned_tensor.grad = self.grad.copy()
        return cloned_tensor

    def repeat(self, times: int):
        """成倍扩展张量维度"""
        out = Tensor.connect([self.clone() for _ in range(times)])
        out.zero_grad()
        return out

    # 复用运算符实现繁琐的激活函数，自动构建计算图
    def softmax(self):
        """softmax激活函数：softmax(x_i) = e^x_i / sum(e^x_j)"""
        exp_tensor = self.exp()
        sum_exp = exp_tensor.sum()  # 返回单标量张量
        inv_sum_exp = sum_exp ** -1 if sum_exp.data[0] != 0 else Tensor([0])
        return exp_tensor * inv_sum_exp.repeat(len(self))

    def gelu(self):
        """GELU激活函数：GELU(x) ≈ x * Sigmoid(1.702x)"""
        return self * (self * 1.702).sigmoid()

    @classmethod
    def mse(cls, a: 'Tensor', b: 'Tensor'):
        """均方误差（Mean Squared Error）：MSE = (1/n) * sum((a - b)²)"""
        if len(a) != len(b):
            raise ValueError("MSE can only be calculated between tensors of the same length")
        return ((a - b) ** 2).sum() / Tensor([len(a)])

    @classmethod
    def sse(cls, a: 'Tensor', b: 'Tensor'):
        """平方误差（Sum of Squared Error）：SSE = sum((a - b)²)"""
        if len(a) != len(b):
            raise ValueError("SSE can only be calculated between tensors of the same length")
        return ((a - b) ** 2).sum()

    @classmethod
    def nll(cls, out: 'Tensor', target: 'Tensor'):
        """交叉熵误差（Negative Log Likelihood）：NLL = -sum(target * log(output))"""
        return -(target * out.log()).sum()

    def zero_grad(self):
        """清空梯度"""
        self.grad = Vector([0] * len(self))

    def one_grad(self):
        """将梯度设为1"""
        self.grad = Vector([1] * len(self))

    @classmethod
    def zeros(cls, size: int):
        """创建一个长度为size的全0张量"""
        return Tensor([0] * size)

    def step(self, lr: float):
        """
        梯度下降，反向传播后进行
        :param lr: 学习率
        """
        self.data -= self.grad * lr

    def backward(self, clean=True):
        """
        计算子图的反向传播
        :param clean: bool 是否清零计算图
        """
        self.one_grad()
        op_list = [self.op]

        for i in op_list:
            if i is None:
                continue
            if type(i.inp) is list:
                for ea in i.inp:
                    if ea.op not in op_list and ea.op is not None:
                        op_list.append(ea.op)
            elif i.inp.op not in op_list and i.inp.op is not None:
                op_list.append(i.inp.op)

        # 按计算顺序排序搜索到的子图
        op_list.sort(key=lambda x: Operator.compute_list.index(x), reverse=True)
        for i in op_list:
            i.propagate_grad()
        if clean:
            Operator.clean()


class Operator:
    """运算符类，所有对张量操作的运算符都需要继承此类，并重写compute和propagate_grad方法"""
    compute_list = []  # 记录计算顺序，用于反向传播

    def __init__(self):
        Operator.compute_list.append(self)
        self.inp = None  # 输入张量
        self.out = None  # 输出张量

    def compute(self, *args):
        """
        子类需实现具体运算
        :param args: Tensor
        :return: Tensor
        """
        raise NotImplementedError

    def propagate_grad(self):
        """梯度反向传播，输入张量积累梯度"""
        raise NotImplementedError

    @classmethod
    def backward(cls, last1grad=True):
        """
        全局反向传播，计算所有相关张量的梯度
        :param last1grad: 是否将计算图中最后一个张量的梯度设为1（反向传播起点）
        :return: None
        """
        if not Operator.compute_list:  # 无运算记录时直接返回
            return

        reversed_ops = list(reversed(Operator.compute_list))
        if last1grad:
            reversed_ops[0].out.one_grad()

        # 逆序遍历运算执行梯度反向传播
        for op in reversed_ops:
            op.propagate_grad()
            op.inp = None
            op.out = None

        Operator.clean()

    @classmethod
    def clean(cls):
        """清理所有运算符，请务必在运行后调用，避免内存占用过大"""
        for o in Operator.compute_list:
            o.inp = None
            o.out = None
        Operator.compute_list = []


class Add(Operator):
    """加法运算符"""

    def __init__(self):
        super().__init__()

    def compute(self, a: Tensor, b: Tensor):
        self.inp = [a, b]
        out = Tensor(a.data + b.data, op=self)
        self.out = out
        return out

    def propagate_grad(self):
        # 加法梯度反向传播：输入梯度+=输出梯度
        self.inp[0].grad += self.out.grad
        self.inp[1].grad += self.out.grad


class Sub(Operator):
    """减法运算符"""

    def __init__(self):
        super().__init__()

    def compute(self, a: Tensor, b: Tensor):
        self.inp = [a, b]
        out = Tensor(a.data - b.data, op=self)
        self.out = out
        return out

    def propagate_grad(self):
        # 减法梯度反向传播：输入梯度+=输出梯度的正负数
        self.inp[0].grad += self.out.grad
        self.inp[1].grad -= self.out.grad


class Mul(Operator):
    """乘法运算符"""

    def __init__(self):
        super().__init__()

    def compute(self, a: Tensor, b: Tensor):
        self.inp = [a, b]
        out = Tensor(a.data * b.data, op=self)
        self.out = out
        return out

    def propagate_grad(self):
        # 乘法梯度反向传播：输入梯度+=输出*输入的函数
        self.inp[0].grad += self.inp[1].data * self.out.grad
        self.inp[1].grad += self.inp[0].data * self.out.grad


class Div(Operator):
    """除法运算符"""

    def __init__(self):
        super().__init__()

    def compute(self, a: Tensor, b: Tensor):
        self.inp = [a, b]
        out = Tensor(a.data / b.data, op=self)
        self.out = out
        return out

    def propagate_grad(self):
        # 除法梯度反向传播：输入梯度+=输出*输入的函数
        self.inp[0].grad += (self.inp[1].data ** -1) * self.out.grad
        self.inp[1].grad += (-self.inp[0].data * self.inp[1].data ** -2) * self.out.grad


class Pow(Operator):
    """幂函数"""

    def __init__(self):
        super().__init__()
        self.power = None

    def compute(self, a: Tensor, power: float):
        self.inp = a
        self.power = power
        out = Tensor(a.data ** power, op=self)
        self.out = out
        return out

    def propagate_grad(self):
        # 输入梯度 += n * x ^ (n - 1) * 输出梯度
        self.inp.grad += self.power * self.inp.data ** (self.power - 1) * self.out.grad


class Exp(Operator):
    """自然指数函数"""

    def __init__(self):
        super().__init__()

    def compute(self, a: Tensor):
        self.inp = a
        out = Tensor(math.e ** a.data, op=self)
        self.out = out
        return out

    def propagate_grad(self):
        # 输入梯度 += e ^ x * 输出梯度 = 输出值 * 输出梯度
        self.inp.grad += self.out.data * self.out.grad


class Log(Operator):
    """自然对数函数"""

    def __init__(self):
        super().__init__()

    def compute(self, a: Tensor):
        self.inp = a
        # 非正数无效标注为无穷大
        out = Tensor([math.log(i) if i > 0 else float("inf") for i in a.data], op=self)
        self.out = out
        return out

    def propagate_grad(self):
        # 输入梯度 += (1 / x) * 输出梯度 = 输出梯度 * (输入值的倒数)
        self.inp.grad += self.out.grad * self.inp.data ** -1


class Sum(Operator):
    """求和运算符"""

    def __init__(self):
        super().__init__()

    def compute(self, a: Tensor):
        self.inp = a
        sum_value = sum(a.data)  # 返回标量
        out = Tensor([sum_value], op=self)
        self.out = out
        return out

    def propagate_grad(self):
        # 输入张量的每个分量梯度 += 输出张量的梯度（单元素）
        self.inp.grad += self.out.grad[0]


class Relu(Operator):
    """ReLU修正线性单元"""

    def __init__(self):
        super().__init__()

    def compute(self, a: Tensor):
        self.inp = a
        out = Tensor([max(i, 0) for i in a.data], op=self)
        self.out = out
        return out

    def propagate_grad(self):
        # 输入梯度+=输出梯度*输入函数（大于等于0）
        for i in range(len(self.inp)):
            if self.inp.data[i] >= 0:
                self.inp.grad[i] += self.out.grad[i]


class Sigmoid(Operator):
    """Sigmoid激活函数"""

    def __init__(self):
        super().__init__()

    def compute(self, a: Tensor):
        self.inp = a
        out = Tensor(1 / (1 + math.e ** -a.data), op=self)
        self.out = out
        return out

    def propagate_grad(self):
        # 输入梯度 += σ(x) * (1 - σ(x)) * 输出梯度
        self.inp.grad += self.out.data * (1 - self.out.data) * self.out.grad


class Tanh(Operator):
    """Tanh激活函数"""

    def __init__(self):
        super().__init__()

    def compute(self, a: Tensor):
        self.inp = a
        exp_p = math.e ** a.data
        exp_n = math.e ** -a.data
        out = Tensor((exp_p - exp_n) / (exp_p + exp_n), op=self)
        self.out = out
        return out

    def propagate_grad(self):
        # 输入梯度 += (1 - tanh²(x)) * 输出梯度
        self.inp.grad += (1 - self.out.data ** 2) * self.out.grad


class Connect(Operator):
    """拼接张量"""

    def __init__(self):
        super().__init__()

    def compute(self, x: list[Tensor]):
        self.inp = x
        out = Tensor([], op=self)
        for tensor in x:
            out.data.extend(tensor.data)
            out.grad.extend(tensor.grad)
        self.out = out
        return out

    def propagate_grad(self):
        # 将输出梯度按原拼接顺序分配给每个输入张量
        seg = self.out.grad
        for tensor in self.inp:
            tensor.grad += Vector(seg[:len(tensor)])
            seg = seg[len(tensor):]


class Cut(Operator):
    """切分张量"""

    def __init__(self):
        super().__init__()
        self.start = None
        self.end = None

    def compute(self, a: Tensor, start: int, end: int):
        self.inp = a
        self.start = start
        self.end = end
        out = Tensor(a.data[start:end], op=self)
        self.out = out
        return out

    def propagate_grad(self):
        # 将输出张量的梯度分配到输入张量的对应索引位置
        for i in range(len(self.out)):
            self.inp.grad[self.start + i] += self.out.grad[i]


class DenseOp(Operator):
    """线性层运算符"""

    def __init__(self):
        super().__init__()

    def compute(self, matrix, x: Tensor):
        self.inp = [matrix, x]
        if matrix.bias:
            out = [sum(matrix.w[i].data * x.data) + matrix.b[i].data[0] for i in range(len(matrix.w))]
        else:
            out = [sum(matrix.w[i].data * x.data) for i in range(len(matrix.w))]
        out = Tensor(out)
        self.out = out
        return out

    def propagate_grad(self):
        """
        线性层梯度反向传播：
        假设前向计算：out[i] = sum(w[i][j] * x[j]) + b[i]（j为输入维度）
        - w[i][j]的梯度：dLoss/dw[i][j] = dLoss/dout[i] * x[j]
        - b[i]的梯度：dLoss/db[i] = dLoss/dout[i]（b[i]仅影响out[i]）
        - x[j]的梯度：dLoss/dx[j] = sum(dLoss/dout[i] * w[i][j])（i为输出维度）
        """
        matrix, x = self.inp
        for i in range(len(matrix.w)):
            matrix.w[i].grad += self.out.grad[i] * x.data
            x.grad += self.out.grad[i] * matrix.w[i].data
            if matrix.bias:
                matrix.b[i].grad += self.out.grad[i]


class Layer:
    """参数层，所有要保存参数的类都需要继承此类"""
    layer_list = []  # 所有要保存参数的实例的列表
    is_load = False  # 是否处于读取状态
    is_save = True  # 是否处于保存状态
    pointer = 0  # 读取参数用的指针
    layer_num = 0  # 读取的参数中子类的数量

    def __init__(self, *args):
        """
        当处于读取状态(is_load==True)时，继承了Layer的实例会按顺序读取layer_list中的内容。
        当处于存储状态(is_save==True)时，继承了Layer的实例会在创建时被加入layer_list，它在save_all调用时会被保存为文件。
        一般情况下，继承了Layer的类初始化需最后调用super().__init__()，防止读取的数据被覆盖。
        或在if not Layer.is_load:中进行初始化。
        """
        if not Layer.is_save:
            return
        if Layer.is_load:
            self.load(Layer.layer_list[Layer.pointer])
            Layer.layer_list[Layer.pointer] = self
            Layer.pointer += 1
            if Layer.layer_num == Layer.pointer:
                Layer.is_load = False
        else:
            Layer.layer_list.append(self)

    def save(self):
        """
        将自身转为字符串的形式，所有继承了Layer的类需重写此方法
        :return: 字符串内容中不能包含换行符
        """
        raise NotImplementedError

    def load(self, *args):
        """
        从字符串中读取数据,所有继承了Layer的类需重写此方法
        :param args: str 字符串内容
        """
        raise NotImplementedError

    def param(self):
        """
        返回自身需要优化的参数，所有继承了Layer的类需重写此方法
        :return: list[Tensor]
        """
        raise NotImplementedError

    @classmethod
    def save_all(cls, name):
        """
        存储所有layer_list中的实例
        :param name: str 保存的文件名
        """
        with open(name, "w") as f:
            for i in Layer.layer_list:
                content = i.save()
                if content is None:
                    continue
                f.write(content + "\n")

    @classmethod
    def load_all(cls, name):
        """
        从文件中读取保存的内容，保存到layer_list
        :param name: str 保存的文件名
        """
        Layer.is_load = True
        with open(name, "r") as f:
            Layer.layer_list = f.readlines()
            Layer.layer_num = len(Layer.layer_list)

    @classmethod
    def get_params(cls):
        """
        返回所有需要优化的参数
        :return: list[Tensor]
        """
        params = []
        for i in Layer.layer_list:
            # Param返回列表
            params += i.param()
        return params


class ConstantTensor(Tensor, Layer):
    """用于保存的张量"""

    def __init__(self, data):
        Tensor.__init__(self, data)
        Layer.__init__(self)

    def save(self):
        text = str(self.data)
        return text

    def load(self, text: str):
        self.data = Vector(ast.literal_eval(text))

    def param(self):
        return [self]


class Dense(Layer):
    """全连接层，保存格式：用 ‘/’ 隔开 w、b"""

    def __init__(self, inp_size: int, out_size: int, bias=True):
        """
        参数初始化
        :param bias: bool 是否加上偏置
        :param w: list[Tensor] 权重矩阵
        :param bias: Tensor 偏置向量
        """
        if not Layer.is_load:
            self.w = [my_init(inp_size) for _ in range(out_size)]
            self.bias = bias
            if bias:
                self.b = [Tensor([0] * inp_size) for _ in range(out_size)]
        super().__init__()

    def __call__(self, a: Tensor, with_op=True):
        """
        前向传播运算
        :param with_op: bool 是否使用单独的dense运算符
        """
        if with_op:
            op = DenseOp()
            out = op.compute(self, a)
            return out
        else:
            # 复用更基础的算符
            if self.bias:
                out = [(self.w[i] * a).sum() + self.b[i] for i in range(len(self.w))]
            else:
                out = [(self.w[i] * a).sum() for i in range(len(self.w))]
            out = Tensor.connect(out)
            return out

    def grad_descent_zero(self, lr: float):
        """
        进行梯度下降，并清空梯度
        :param lr: float 学习率
        """
        for i in range(len(self.w)):
            self.w[i].step(lr)
            self.b[i].step(lr)
            self.w[i].zero_grad()
            self.b[i].zero_grad()

    def copy(self, save=True):
        """
        深拷贝参数层
        :param save: bool 是否保存拷贝出的对象
        注：如果不希望拷贝出的对象被保存，请在调用前把Layer.is_save设定为False
        """
        if not save:
            original = Layer.is_save
            Layer.is_save = False

        new = Dense(len(self.w[0]), len(self.w))
        for i in range(len(self.w)):
            new.w[i] = Tensor(self.w[i].data)
            new.b[i] = Tensor(self.b[i].data)

        if not save:
            Layer.is_save = original
        return new

    def save(self):
        x = [str(i.data) for i in self.w]
        text = '[' + ', '.join(x) + ']'
        if self.bias:
            x = [str(i.data) for i in self.b]
            text += '/[' + ', '.join(x) + ']'
        return text

    def load(self, text: str):
        text = text.split('/')
        # w、b是可迭代对象
        w = ast.literal_eval(text[0])
        self.w = [Tensor(i) for i in w]
        if len(text) == 2:
            self.bias = True
            b = ast.literal_eval(text[1])
            self.b = [Tensor(i) for i in b]
        else:
            self.bias = False

    def param(self):
        return self.w + self.b


class Optimizer(Layer):
    """
    优化器类。需要储存参数需要重写save()和load()函数，详见Layer
    当参数未指定时，需要在所有参数创建后再实例化此类！
    """

    def __init__(self, params=None):
        """
        :param params:list[Tensor,Tensor...] 需要优化的参数的列表，为None时优化所有Layer中的参数
        """
        super().__init__()
        if params is None:
            self.params = Layer.get_params()
        else:
            self.params = params

    def step(self):
        """具体优化方法，必须重写"""
        raise NotImplementedError

    def zero_grad(self):
        """使参数的梯度归零"""
        for i in self.params:
            # 单个张量的梯度归零
            i.zero_grad()


class SGD(Optimizer):
    """梯度下降"""

    def __init__(self, params=None, lr=0.001):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for i in self.params:
            i.step(self.lr)
            i.zero_grad()


class Momentum(Optimizer):
    """带动量的梯度下降"""

    def __init__(self, params=None, lr=0.001, gamma=0.8):
        super().__init__(params)
        if not Layer.is_load:
            if params is None:
                self.m = [0 for _ in range(sum([len(p) for p in Layer.get_params()]))]
            else:
                self.m = [0 for _ in range(sum([len(p) for p in params]))]

        self.lr = lr
        self.gamma = gamma

    def step(self):
        m_index = 0
        for t in self.params:
            for ten_index in range(len(t)):
                t.data[ten_index] -= self.k * self.m[m_index]
                self.m[m_index] = self.gamma * self.m[m_index] + (1 - self.gamma) * t.grad[ten_index]
                t.grad[ten_index] = 0
                m_index += 1

    def save(self):
        return str(self.m)

    def load(self, text):
        self.m = ast.literal_eval(text)


class Adam(Optimizer):
    """Adaptive Moment Estimation（自适应矩估计）"""

    def __init__(self, params=None, k=0.001, b1=0.9, b2=0.999, eps=1e-8):
        if not Layer.is_load:
            if params is None:
                self.m = [0 for i in range(sum([len(p) for p in Layer.get_params()]))]
                self.s = [0 for i in range(sum([len(p) for p in Layer.get_params()]))]
            else:
                self.m = [0 for i in range(sum([len(p) for p in params]))]
                self.s = [0 for i in range(sum([len(p) for p in params]))]
            self.times = 1
        super().__init__(params)
        self.k = k
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def step(self):
        index = 0
        for t in self.params:
            for ten_index in range(len(t)):
                self.m[index] = self.b1 * self.m[index] + (1 - self.b1) * t.grad[ten_index]
                self.s[index] = self.b2 * self.s[index] + (1 - self.b2) * t.grad[ten_index] ** 2
                cm = self.m[index] / (1 - self.b1 ** self.times)
                cs = self.s[index] / (1 - self.b2 ** self.times)
                t.data[ten_index] -= self.k * cm / (cs ** 0.5 + self.eps)
                t.grad[ten_index] = 0
                index += 1
        self.times += 1

    def save(self):
        return str(self.m) + "/" + str(self.s) + "/" + str(self.times)

    def load(self, t):
        t = t.split("/")
        self.m = eval(t[0])
        self.s = eval(t[1])
        self.times = float(t[2])


# 各种初始化方法
def my_init(size):
    """对单个张量初始化权重"""
    sigma = (2 / size) ** 0.5 / 5
    return Tensor([random.gauss(0, sigma) for _ in range(size)])


def xavier_init(inp_size: int, out_size: int):
    """Xavier初始化 - 适用于tanh/sigmoid等激活函数"""
    sigma = (2 / (inp_size + out_size)) ** 0.5
    return [Tensor([random.gauss(0, sigma) for _ in range(inp_size)]) for _ in range(out_size)]


def he_init(inp_size: int, out_size: int):
    """He初始化 - 适用于ReLU及其变体激活函数"""
    sigma = (2 / inp_size) ** 0.5
    return [Tensor([random.gauss(0, sigma) for _ in range(inp_size)]) for _ in range(out_size)]


def uniform_init(size, a=-0.05, b=0.05):
    """均匀分布初始化 - 适用于线性层"""
    return Tensor([random.uniform(a, b) for _ in range(size)])


def sum_chan2d(x):
    """
    对多通道的二维张量求和，变成单通道二维张量
    :param x: 多通道二维张量，结构为 [批次][行][通道张量]
              例如: [[tensor1, tensor2], [tensor3, tensor4]]
    :return: 单通道二维张量，结构为 [行][求和后的张量]
    """
    rows = len(x[0])
    batches = len(x)

    out = [tensor.clone() for tensor in x[0]]
    for batch_idx in range(1, batches):
        if len(x[batch_idx]) != rows:
            raise ValueError(f"The row of batches must be {rows}.")
        for row_idx in range(rows):
            out[row_idx] += x[batch_idx][row_idx]
    return out


def resize2d(x, cols, rows):
    """把一维张量转为二维张量"""
    x2d = []
    for i in range(rows):
        x2d.append(x.cut(i * cols, (i + 1) * cols))
    return x2d