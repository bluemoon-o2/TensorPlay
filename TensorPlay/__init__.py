"""
TensorPlay - 一个用于深度学习验证的工具包

版本: 0.1.1
作者: Welog
日期: 2025年9月3日

功能特点:
- 提供多阶自动微分处理能力
- 提供计算图可视化功能
- 支持多维度的模型组件管理
- 支持JSON格式保存和加载
- 支持模型结构打印
- 支持钩子调试
"""
__version__ = "0.1.1"
__author__ = "Welog"
__email__ = "2095774200@shu.edu.cn"
__description__ = "一个用于深度学习验证的工具包"
__url__ = "https://github.com/bluemoon-o2/TensorPlay"
__license__ = "MIT"


# =============================================================================
# 全局接口
# =============================================================================
from .core import (config, no_grad, to_data, Tensor, Layer, Operator, Optimizer)
from .layer import (Dense)
from .module import (Module, Sequential)
from .optimizer import (SGD, Adam, Momentum, AdamW, Nadam, Lookahead, RMSprop)
from .operator import (concatenate)
from .func import (mse, sse, nll)
from .initializer import (he_init, xavier_init, uniform_init, my_init)
from .utils import (plot_dot_graph, train_on_batch, valid_on_batch)


# =============================================================================
# 加载算子
# =============================================================================
import TensorPlay.operator as ops
Tensor.__add__ = ops.add
Tensor.__radd__ = ops.add
Tensor.__neg__ = ops.neg
Tensor.__sub__ = ops.sub
Tensor.__rsub__ = ops.rsub
Tensor.__mul__ = ops.mul
Tensor.__rmul__ = ops.rmul
Tensor.__matmul__ = ops.matmul
Tensor.__truediv__ = ops.div
Tensor.__rtruediv__ = ops.rdiv
Tensor.__pow__ = ops.ten_pow
Tensor.__getitem__ = ops.ten_slice
Tensor.reslice = ops.reslice
Tensor.sum = ops.ten_sum
Tensor.exp = ops.exp
Tensor.log = ops.log
Tensor.mean = ops.mean
Tensor.relu = ops.relu
Tensor.gelu = ops.gelu
Tensor.tanh = ops.tanh
Tensor.expand = ops.expand
Tensor.reshape = ops.reshape
Tensor.flatten = ops.flatten
Tensor.sigmoid = ops.sigmoid
Tensor.softmax = ops.softmax
Tensor.broadcast = ops.broadcast
Tensor.rebroadcast = ops.rebroadcast
Tensor.transpose = ops.transpose
Tensor.T = ops.T