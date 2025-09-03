"""
TensorPlay - 一个用于深度学习验证的工具包

版本: 0.0.1
作者: Welog
日期: 2025年8月31日

功能特点:
- 提供多阶自动微分处理能力
- 提供计算图可视化功能
- 支持多维度的模型组件管理
- 支持JSON格式保存和加载
- 支持模型结构打印
- 支持钩子调试
"""
__version__ = "0.0.1"
__author__ = "Welog"
__email__ = "2095774200@shu.edu.cn"
__description__ = "一个用于深度学习验证的工具包"
__url__ = "https://github.com/bluemoon-o2/TensorPlay"
__license__ = "MIT"


from .core import *
from .utils import (plot_dot_graph)
from .func import *
