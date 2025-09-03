from .core import Tensor
import numpy as np

def my_init(size) -> Tensor:
    """对单个张量初始化权重"""
    std = np.sqrt(2.0 / size)
    return Tensor(np.random.normal(0, std, size))


def xavier_init(inp_size: int, out_size: int) -> Tensor:
    """Xavier初始化 - 适用于tanh/sigmoid等激活函数"""
    std = np.sqrt(2.0 / (inp_size + out_size))
    return Tensor(np.random.normal(0, std, (out_size, inp_size)))


def he_init(inp_size: int, out_size: int) -> Tensor:
    """He初始化 - 适用于ReLU及其变体激活函数"""
    std = np.sqrt(2.0 / inp_size)
    return Tensor(np.random.normal(0, std, (out_size, inp_size)))


def uniform_init(size, a=-0.05, b=0.05) -> Tensor:
    """均匀分布初始化 - 适用于线性层"""
    return Tensor(np.random.uniform(a, b, size))