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

from typing_extensions import overload

import TensorPlay.operator
from .core import *
from .layer import *
from .module import *
from .optimizer import *
from .utils import (plot_dot_graph, train_on_batch, valid_on_batch)
from .func import *


# load the operator
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
Tensor.exp = ops.exp
Tensor.log = ops.log
Tensor.relu = ops.relu
Tensor.broadcast = ops.broadcast
Tensor.rebroadcast = ops.rebroadcast
Tensor.transpose = ops.transpose
Tensor.T = ops.T



