# Copyright (c) 2025 zlx. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The tensorplay package offers a simple deep-learning framework
designed for educational purposes and small-scale experiments.
It defines a data structure for multidimensional arrays called Tensor,
on which it encapsulates mathematical operations.
"""

__version__ = "1.0.0rc0"

import os

# Add the directory containing this file to DLL search path on Windows
# to allow loading tensorplay_core.dll which _C.pyd depends on.
if os.name == 'nt':
    try:
        os.add_dll_directory(os.path.dirname(__file__))
    except OSError:
        pass

from .tensor import Tensor
from ._C import tensor

from .scalar import Scalar
from .device import Device, DeviceType
# Alias for torch compatibility
device = Device

from . import cuda

from .dtype import DType
from ._C import ops, Size

# Expose factory functions from _C
from ._C import zeros, ones, eye, empty, rand, full, arange, empty_like, zeros_like, ones_like, full_like
from ._C import rand_like, randint, randint_like, randn, randn_like, randperm, bernoulli, normal, poisson
from ._C import mm, matmul, conv2d
from ._C import cat, stack, transpose, permute, squeeze, unsqueeze, t, split, chunk, unbind
from ._C import linspace, logspace
from ._C import Generator, default_generator, manual_seed, seed, initial_seed

# Config
from . import __config__

# DType aliases
float32 = DType.float32
float64 = DType.float64
int32 = DType.int32
int64 = DType.int64
bool = DType.bool

__all__ = [
    'Tensor', 'tensor', 'Scalar', 'Device', 'device', 'DType', 'Size', 'ops', 'Generator', '__config__',
    'zeros', 'ones', 'eye', 'empty', 'rand', 'full', 'arange', 'linspace', 'logspace',
    'rand_like', 'randint', 'randint_like', 'randn', 'randn_like', 'randperm', 'bernoulli', 'normal', 'poisson',
    'empty_like', 'zeros_like', 'ones_like', 'full_like',
    'mm', 'matmul', 'conv2d',
    'cat', 'stack', 'transpose', 'permute', 'squeeze', 'unsqueeze', 't', 'split', 'chunk', 'unbind',
    'float32', 'float64', 'int32', 'int64', 'bool',
    'default_generator', 'manual_seed', 'seed', 'initial_seed'
]
