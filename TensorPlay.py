import ast
import random
import math


class Vector(list):
    """向量，可以进行按位加减乘等操作"""

    def __init__(self, data):
        # 检查每个元素是否为数字（int或float）
        for item in data:
            if not isinstance(item, (int, float)):
                raise TypeError(f"Vector elements must be numbers (not {type(item).__name__})")
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
            raise TypeError(f"Unsupported operand type(s) for +: 'Vector' and '{type(other).__name__}'")

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
            raise TypeError(f"Unsupported operand type(s) for +: 'Vector' and '{type(other).__name__}'")
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
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'")

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
            raise TypeError(f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'")

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
            raise TypeError(f"Unsupported operand type(s) for /: 'Vector' and '{type(other).__name__}'")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            # 标量 / 向量：每个元素除以标量
            return Vector([other / x for x in self])
        else:
            raise TypeError(f"Unsupported operand type(s) for /: '{type(other).__name__}' and 'Vector'")

    def __pow__(self, power: (int, float)):
        if isinstance(power, (int, float)):
            return Vector([i ** power for i in self])
        else:
            raise TypeError(f"Cannot power Vector with {type(power).__name__}")

    def __rpow__(self, base: (int, float)):
        if isinstance(base, (int, float)):
            return Vector([base ** x for x in self])
        else:
            raise TypeError(f"Cannot power {type(base).__name__} with Vector")

    @classmethod
    def zeros(cls, size):
        return Vector([0] * size)


class Tensor:
    """张量，带梯度的向量，梯度是另一个同维向量"""

    def __init__(self, data, op=None):
        self.data = Vector(data)
        self.grad = Vector.zeros(len(data))
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
        return op.compute(Tensor.zeros(len(self)), self)

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
        """返回当前张量的副本，确保梯度独立"""
        cloned_tensor = Tensor(self.data, op=self.op)
        cloned_tensor.grad = Vector(self.grad)
        return cloned_tensor

    def detach(self):
        """返回一个不追踪梯度的张量副本"""
        detached = Tensor(self.data)
        return detached

    def repeat(self, times: int):
        """成倍扩展张量维度"""
        out = Tensor.connect([self.clone() for _ in range(times)])
        out.zero_grad()
        return out

    # 复用运算符实现繁琐的激活函数，自动构建计算图
    def softmax(self):
        """softmax激活函数：softmax(x) = e^x / sum(e^x_j)"""
        exp_tensor = self.exp()
        sum_exp = exp_tensor.sum()  # 返回单标量张量
        inv_sum_exp = sum_exp ** -1 if abs(sum_exp.data[0]) > 1e-5 else (sum_exp + Tensor([0.1])) ** -1
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
        self.grad = Vector.zeros(len(self))

    def one_grad(self):
        """将梯度设为1"""
        self.grad = Vector([1] * len(self))

    @classmethod
    def zeros(cls, size: int):
        """创建一个长度为size的全0张量"""
        return Tensor([0] * size)

    def backward(self, clean=True):
        """
        计算子图的反向传播
        :param clean: bool 是否清零计算图
        """
        self.one_grad()
        op_set = set()
        queue = []
        if self.op is not None:
            queue.append(self.op)
            op_set.add(self.op)

        while queue:
            current_op = queue.pop(0)  # 取出队首元素
            if current_op.inp is None:
                continue
            # 处理输入为列表或单个张量的情况
            inputs = current_op.inp if isinstance(current_op.inp, list) else [current_op.inp]
            for inp_tensor in inputs:
                if inp_tensor.op is not None and inp_tensor.op not in op_set:
                    op_set.add(inp_tensor.op)
                    queue.append(inp_tensor.op)

        # 按计算顺序排序搜索到的子图
        op_list = sorted(op_set, key=lambda x: Operator.compute_list.index(x), reverse=True)
        for op in op_list:
            if op.inp is None:
                continue
            op.propagate_grad()
        if clean:
            Operator.clean(specific_ops=op_set)


class Operator:
    """运算符类，所有对张量操作的运算符都需要继承此类，并重写compute和propagate_grad方法"""
    compute_list = []  # 记录计算顺序，用于反向传播
    enable_grad = True  # 控制是否追踪梯度

    def __init__(self):
        if self.enable_grad:
            self.compute_list.append(self)
        self.inp = None  # 输入张量
        self.out = None  # 输出张量

    def __repr__(self):
        return f"TensorOperator.{self.__class__.__name__}"

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
    def set_grad_enabled(cls, mode):
        """控制是否开启梯度追踪"""
        cls.enable_grad = mode

    @classmethod
    def backward(cls, last1grad=True):
        """
        全局反向传播，计算所有相关张量的梯度
        :param last1grad: 是否将计算图中最后一个张量的梯度设为1（反向传播起点）
        :return: None
        """
        if not cls.compute_list:  # 无运算记录时直接返回
            return

        reversed_ops = list(reversed(cls.compute_list))
        if last1grad:
            reversed_ops[0].out.one_grad()

        # 逆序遍历运算执行梯度反向传播
        for op in reversed_ops:
            if op.inp is None:  # 跳过已清理的节点
                continue
            op.propagate_grad()

        cls.clean()

    @classmethod
    def clean(cls, specific_ops=None):
        """
        清理计算图数据
        :param specific_ops: 可选，指定需要清理的运算符集合，默认清理全部
        """
        if specific_ops is not None:
            for op in specific_ops:
                if op in cls.compute_list:
                    op.inp = None
                    op.out = None
                    cls.compute_list.remove(op)
        else:
            for op in cls.compute_list:
                op.inp = None
                op.out = None
            cls.compute_list.clear()


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
        self.matrix = None

    def compute(self, matrix, x: Tensor):
        self.inp = x
        self.matrix = matrix
        if matrix.bias:
            out = [sum(matrix.w[i].data * x.data) + matrix.b[i].data[0] for i in range(len(matrix.w))]
        else:
            out = [sum(matrix.w[i].data * x.data) for i in range(len(matrix.w))]
        out = Tensor(out, op=self)
        self.out = out
        return out

    def propagate_grad(self):
        """
        线性层梯度反向传播：
        假设前向计算：out[i] = sum(w[i][j] * x[j]) + b[i]（j为输入维度）
        - w[i][j]的梯度：dLoss/dw[i][j] = dLoss/d_out[i] * x[j]
        - b[i]的梯度：dLoss/db[i] = dLoss/d_out[i]（b[i]仅影响out[i]）
        - x[j]的梯度：dLoss/dx[j] = sum(dLoss/d_out[i] * w[i][j])（i为输出维度）
        """
        for i in range(len(self.matrix.w)):
            self.matrix.w[i].grad += self.out.grad[i] * self.inp.data
            self.inp.grad += self.out.grad[i] * self.matrix.w[i].data
            if self.matrix.bias:
                self.matrix.b[i].grad += self.out.grad[i]


class Layer:
    """参数层，所有要保存参数的类都需要继承此类"""
    layer_list = []  # 所有要保存参数的层的列表
    optimizer = None  # 保存的优化器实例
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
            Layer.pointer = 0
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
            if i.param() is not None:
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
        :param bias: list[Tensor] 偏置向量
        """
        if not Layer.is_load:
            self.w = [my_init(inp_size) for _ in range(out_size)]
            self.bias = bias
            if bias:
                self.b = [Tensor.zeros(1) for _ in range(out_size)]
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
            self.w[i].data -= lr * self.w[i].grad
            self.b[i].data -= lr * self.b[i].grad
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
        if not Layer.is_load:
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

    def __call__(self, x):
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


class MultiConv:
    """多通道卷积层"""

    def __init__(self, in_channel: int, out_channel: int, width: int, height: int,
                 stride_w: int = 1, stride_h: int = 1, pad: bool = True, bias: bool = True):
        """
        :param in_channel: int 输入通道数
        :param out_channel: int 输出通道数
        :param width: int 卷积核的宽度（如填充，请设为奇数）
        :param height: int 卷积核的高度（如填充，请设为奇数）
        :param stride_w: int 横向的步长
        :param stride_h: int 纵向的步长
        :param pad: bool 是否进行填充（使运算的输入和输出的大小一样）
        :param bias: bool 是否加上偏置
        """
        self.cores = [[Conv2D(width, height, stride_w, stride_h, pad=False, bias=bias) for _ in range(in_channel)]
                      for _ in range(out_channel)]
        self.pad = pad

    def __call__(self, x):
        """
        :param x: list[list[Tensor,Tensor...],list[Tensor,Tensor...]...] 多通道的二维张量，由列表包裹
        :return: list[list[Tensor,Tensor...],list[Tensor,Tensor...]...]
        """
        if self.pad:
            x = [self.cores[0][0].padding(i) for i in x]
        x2 = []
        for chan in self.cores:
            x_chan = sum_chan2d([chan[i](x[i]) for i in range(len(chan))])
            x2.append(x_chan)
        return x2

    def grad_descent_zero(self, lr):
        for chan in self.cores:
            for i in chan:
                i.grad_descent_zero(lr)


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


class Norm:
    """标准化处理层，带可学习参数"""

    def __init__(self):
        self.w = ConstantTensor([random.gauss(0, 0.04)])
        self.b = ConstantTensor([random.gauss(0, 0.04)])

    def __call__(self, x: Tensor, eps=0.0001):
        n = Tensor([len(x)])
        mean = (x.sum() / n).repeat(len(x))
        sigma = (((x - mean) ** 2).sum() / n) ** 0.5
        std = (x - mean) / (Tensor([eps]) + sigma).repeat(len(x))
        out = self.w.repeat(len(x)) * std + self.b.repeat(len(x))
        return out

    def grad_descent_zero(self, lr):
        self.w.data -= lr * self.w.grad
        self.b.data -= lr * self.b.grad
        self.w.zero_grad()
        self.b.zero_grad()

    def param(self):
        return [self.w, self.b]


class MultiAtt:
    def __init__(self, head_num, emb_size, qk_size=None, v_size=None):
        """
        多头注意力模块
        :param head_num: int 注意力头数量
        :param emb_size: int 输入词向量维度
        :param qk_size: int q、k维度
        :param v_size: int 输出向量维度
        """
        self.heads = [Attention(emb_size, qk_size, v_size) for _ in range(head_num)]
        self.emb_size = emb_size

    def __call__(self, x, mask_list=None, tri_mask=False):
        """
        :param x: list[Tensor,Tensor...]  装着(多个词的词向量)的列表
        :return: list[Tensor,Tensor...]
        """
        out = [h(x, mask_list, tri_mask) for h in self.heads]
        out = sum_chan2d(out)
        return out

    def grad_descent_zero(self, k):
        for i in self.heads:
            i.grad_descent_zero(k)


class Transformer:
    """自注意力模块"""

    def __init__(self, head_num, emb_size):
        self.att = MultiAtt(head_num, emb_size)
        self.f1 = Dense(emb_size, emb_size * 4)
        self.f2 = Dense(emb_size * 4, emb_size)
        self.n1 = Norm()
        self.n2 = Norm()

    def __call__(self, x, mask_list=None, tri_mask=False):
        x2 = x
        x = self.att(x, mask_list, tri_mask)
        x = [self.n1(i) for i in x]
        x = [x[i] + x2[i] for i in range(len(x))]

        x2 = x
        x = [self.f2(self.f1(i).relu()) for i in x]
        x = [self.n2(i) for i in x]
        x = [x[i] + x2[i] for i in range(len(x))]
        return x

    def grad_descent_zero(self, lr):
        self.att.grad_descent_zero(lr)
        self.f1.grad_descent_zero(lr)
        self.f2.grad_descent_zero(lr)
        self.n1.grad_descent_zero(lr)
        self.n2.grad_descent_zero(lr)


class MiniTransformer:
    """简单的Transformer模块"""

    def __init__(self, head_num, emb_size, window_size, lowrank=False):
        self.a = MultiAtt(head_num, emb_size)
        if lowrank:
            self.f1 = MiniDense(emb_size * window_size, emb_size * window_size)
            self.f2 = MiniDense(emb_size * window_size, emb_size * window_size)
        else:
            self.f1 = Dense(emb_size * window_size, emb_size * window_size)
            self.f2 = Dense(emb_size * window_size, emb_size * window_size)
        self.n1 = Norm()
        self.n2 = Norm()
        self.emb_size = emb_size
        self.window_size = window_size

    def __call__(self, x, mask_list=None, tri_mask=False):
        x2 = x  # 二维张量
        x = Tensor.connect(self.a(x, mask_list, tri_mask))
        x = self.n1(x)
        x += Tensor.connect(x2)
        x2 = x  # 一维张量
        x = self.f2(self.f1(x).gelu())
        x = self.n2(x)
        x += x2
        x = resize2d(x, self.emb_size, self.window_size)
        return x

    def grad_descent_zero(self, k):
        self.a.grad_descent_zero(k)
        self.f1.grad_descent_zero(k)
        self.f2.grad_descent_zero(k)
        self.n1.grad_descent_zero(k)
        self.n2.grad_descent_zero(k)


class Optimizer:
    """优化器类，在所有层后实例化，储存参数需要重写save()和load()函数"""
    is_load = False

    def __init__(self, params=None):
        """
        :param params:list[Tensor,Tensor...] 需要优化的参数的列表，为None时优化所有Layer中的参数
        """
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
            i.zero_grad()


class SGD(Optimizer):
    """随机梯度下降优化器"""

    def __init__(self, params=None, lr=0.001):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for ten in self.params:
            ten.data -= self.lr * ten.grad
            ten.zero_grad()


class Momentum(Optimizer):
    """动量优化器"""

    def __init__(self, params=None, lr=0.001, gamma=0.8):
        super().__init__(params)
        if not Optimizer.is_load:
            self.momentum = [Vector.zeros(len(ten)) for ten in self.params]
        self.lr = lr
        self.gamma = gamma

    def step(self):
        for i, tensor in enumerate(self.params):
            self.momentum[i] = self.gamma * self.momentum[i] + (1 - self.gamma) * tensor.grad
            tensor.data -= self.momentum[i] * self.lr
            tensor.zero_grad()

    def save(self):
        return str(self.momentum)

    def load(self, text):
        self.momentum = Vector(ast.literal_eval(text))


class Adam(Optimizer):
    """Adaptive Moment Estimation（自适应矩估计）"""

    def __init__(self, params=None, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params)
        if not Optimizer.is_load:
            self.m = [Vector.zeros(len(ten)) for ten in self.params]
            self.s = [Vector.zeros(len(ten)) for ten in self.params]
            self.times = 1
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def step(self):
        for i, tensor in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * tensor.grad
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * tensor.grad ** 2
            cm = self.m[i] / (1 - self.b1 ** self.times)
            cs = self.s[i] / (1 - self.b2 ** self.times)
            tensor.data -= (self.lr * cm) / (cs ** 0.5 + self.eps)
            tensor.zero_grad()
        self.times += 1

    def save(self):
        return str(self.m) + '/' + str(self.s) + '/' + str(self.times)

    def load(self, text):
        text = text.split('/')
        self.m = Vector(ast.literal_eval(text[0]))
        self.s = Vector(ast.literal_eval(text[1]))
        self.times = float(text[2])


class AdamW(Optimizer):
    """AdamW优化器，在Adam基础上改进了权重衰减"""

    def __init__(self, params=None, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-4):
        super().__init__(params)
        if not Optimizer.is_load:
            self.m = [Vector.zeros(len(ten)) for ten in self.params]
            self.s = [Vector.zeros(len(ten)) for ten in self.params]
            self.times = 1
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay  # 权重衰减系数

    def step(self):
        for i, tensor in enumerate(self.params):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * tensor.grad
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * tensor.grad ** 2
            cm = self.m[i] / (1 - self.b1 ** self.times)
            cs = self.s[i] / (1 - self.b2 ** self.times)
            tensor.data -= self.lr * (cm / (cs ** 0.5 + self.eps) + self.weight_decay * tensor.data)
            tensor.zero_grad()

        self.times += 1

    def save(self):
        return f"{str(self.m)}/{str(self.s)}/{str(self.times)}/{str(self.weight_decay)}"

    def load(self, text):
        text = text.split('/')
        self.m = Vector(ast.literal_eval(text[0]))
        self.s = Vector(ast.literal_eval(text[1]))
        self.times = float(text[2])
        self.weight_decay = float(text[3])


class Lookahead(Optimizer):
    """Lookahead优化器，使用主优化器和慢更新策略"""

    def __init__(self, params=None, base_optimizer=Adam, k=5, alpha=0.5, **kwargs):
        super().__init__(params)
        if not Optimizer.is_load:
            # 初始化基础优化器（Adam）
            self.base_optimizer = base_optimizer(params, **kwargs)
            # 慢权重（初始化为参数的副本）
            self.slow_weights = [Vector(ten.data) for ten in self.params]
            self.k = k  # 慢更新间隔
            self.alpha = alpha  # 插值系数
            self.step_counter = 0  # 步数计数器

    def step(self):
        # 调用基础优化器的step方法（快更新）
        self.base_optimizer.step()
        self.step_counter += 1

        # 每k步执行慢更新
        if self.step_counter % self.k == 0:
            for i in range(len(self.params)):
                # 慢权重更新：slow = slow + alpha*(fast - slow)
                self.slow_weights[i] += self.alpha * (self.params[i].data - self.slow_weights[i])
                # 将慢权重复制回参数
                self.params[i].data = Vector(self.slow_weights[i])

    def save(self):
        base_state = self.base_optimizer.save()
        return f"{base_state}/{str(self.slow_weights)}/{str(self.k)}/{str(self.alpha)}/{str(self.step_counter)}"

    def load(self, text):
        text = text.split('/')
        # 前半部分是基础优化器的状态
        base_state = '/'.join(text[:-4])
        self.base_optimizer.load(base_state)
        self.slow_weights = [Vector(ast.literal_eval(part)) for part in text[-4:-3]]
        self.k = int(text[-3])
        self.alpha = float(text[-2])
        self.step_counter = int(text[-1])


class RMSprop(Optimizer):
    """Root Mean Square Propagation（RMSprop），基于梯度平方的移动平均"""

    def __init__(self, params=None, lr=0.001, alpha=0.99, eps=1e-8):
        super().__init__(params)
        if not Optimizer.is_load:
            self.s = [Vector.zeros(len(ten)) for ten in self.params]  # 二阶动量
            self.alpha = alpha  # 衰减系数
        self.lr = lr
        self.eps = eps

    def step(self):
        for i, tensor in enumerate(self.params):
            # 更新二阶动量：s = alpha*s + (1-alpha)*grad^2
            self.s[i] = self.alpha * self.s[i] + (1 - self.alpha) * (tensor.grad ** 2)

            # 参数更新：theta = theta - lr * grad / (sqrt(s) + eps)
            update = self.lr * tensor.grad / (self.s[i] ** 0.5 + self.eps)
            tensor.step(update)

            # 清空梯度
            tensor.zero_grad()

    def save(self):
        return f"{str(self.s)}/{str(self.alpha)}/{str(self.lr)}/{str(self.eps)}"

    def load(self, text):
        text = text.split('/')
        self.s = Vector(ast.literal_eval(text[0]))
        self.alpha = float(text[1])
        self.lr = float(text[2])
        self.eps = float(text[3])


class Nadam(Optimizer):
    """Nadam优化器，结合Nesterov动量和Adam"""

    def __init__(self, params=None, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params)
        if not Optimizer.is_load:
            self.m = [Vector.zeros(len(ten)) for ten in self.params]  # 一阶动量
            self.s = [Vector.zeros(len(ten)) for ten in self.params]  # 二阶动量
            self.times = 1  # 时间步
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def step(self):
        for i, tensor in enumerate(self.params):
            # 更新一阶动量
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * tensor.grad
            # 更新二阶动量
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * (tensor.grad ** 2)

            # 偏差修正
            cm = self.m[i] / (1 - self.b1 ** self.times)
            cs = self.s[i] / (1 - self.b2 ** self.times)

            # Nadam更新：融入Nesterov动量
            update = self.lr * (self.b1 * cm + (1 - self.b1) * tensor.grad / (1 - self.b1 ** self.times)) / (
                    cs ** 0.5 + self.eps)
            tensor.step(update)

            # 清空梯度
            tensor.zero_grad()

        self.times += 1

    def save(self):
        return f"{str(self.m)}/{str(self.s)}/{str(self.times)}"

    def load(self, text):
        text = text.split('/')
        self.m = Vector(ast.literal_eval(text[0]))
        self.s = Vector(ast.literal_eval(text[1]))
        self.times = float(text[2])


# 各种初始化方法
def my_init(size):
    """对单个张量初始化权重"""
    sigma = (2 / size) ** 0.5 / 10
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


# ----------------测试部分---------------
def func2d(x, func):
    """对二维张量进行张量级函数操作"""
    return [func(i) for i in x]


def grad_test(func, x: Tensor):
    """对函数进行梯度测试（关联计算图）"""
    x = func(x)
    x2 = Tensor(x.data)
    x2.data[0] += 0.001
    x2 = func(x2)
    return (x2.data[0] - x.data[0]) / 0.001


def test():
    x = Tensor([1])
    y = Tensor([1])
    z = Tensor([1])

    for i in range(1000):
        s1 = ((x * y + z) - Tensor([40])) ** 2
        s2 = ((x * z + y) - Tensor([51])) ** 2
        s3 = ((x + y + z) - Tensor([19])) ** 2
        if s1.data[0] < 0.01 and s2.data[0] < 0.01 and s3.data[0] < 0.01:
            break
        s1.backward(clean=False)
        s2.backward(clean=False)
        s3.backward()
        # s1.grad=Vector([1])
        # s2.grad = Vector([1])
        # s3.grad = Vector([1])
        # Operator.back()
        x.data -= x.grad * Vector([0.002])
        y.data -= y.grad * Vector([0.002])
        z.data -= z.grad * Vector([0.002])
        print(f"x{x},y{y},z{z}")
        x.zero_grad()
        y.zero_grad()
        z.zero_grad()