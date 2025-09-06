from .core import Tensor
from .operator import concatenate
import numpy as np

# =============================================================================
# Datasets
# =============================================================================
def load_iris():
    """
    加载鸢尾花数据集
    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============
    """
    from sklearn.datasets import load_iris
    return load_iris()

# =============================================================================
# DataLoader
# =============================================================================
class DataLoader:
    """数据加载器，支持批处理、打乱和自定义转换"""

    def __init__(self, data, batch_size: int = 64, shuffle: bool = True, transform=None):
        """
        :param data: 数据集，格式为[(输入特征, 标签), ...]
        :param batch_size: 批次大小
        :param shuffle: 是否打乱数据集
        :param transform: 数据转换函数，格式为func(input, label) -> (transformed_input, transformed_label)
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        self.indices = list(range(len(data)))
        self.cursor = 0  # 当前批次指针
        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        """迭代器初始化"""
        self.cursor = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        """获取下一个批次"""
        if self.cursor >= len(self.data):
            raise StopIteration
        # 计算当前批次索引范围
        end = min(self.cursor + self.batch_size, len(self.data))
        batch_indices = self.indices[self.cursor:end]
        self.cursor = end

        batch_inputs = []
        batch_labels = []
        for idx in batch_indices:
            x, y = self.data[idx]
            if self.transform:
                x, y = self.transform(x, y)
            batch_inputs.append(Tensor([x]))
            batch_labels.append(Tensor([y]))
        return concatenate(*batch_inputs, axis=0), concatenate(*batch_labels, axis=0)

    def __len__(self):
        """返回批次数量"""
        return (len(self.data) + self.batch_size - 1) // self.batch_size