# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_if_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/vp/cvpx5ygqxz2d7buxlhrix3iumimuetxv3pzn3fgo4tklpfxy3o6f.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 75264, 'x': 37632}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask)
    tl.store(out_ptr0 + (y0 + 3*x2 + 147*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/gv/cgvnage3p672srjovx7tgbc3mpyqclbxbwhoxcuhz3dpcnbf5qni.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 36864, 'x': 24576}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask & ymask)
    tl.store(out_ptr0 + (y0 + 3*x2 + 3072*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/jj/cjjk6f5bqdlnecml5rqo2uhnc3d3olo36baf6clk6n2jrmlsncxo.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 294912, 'x': 147456}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask)
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/qe/cqer2mbtzzao2clkgnb2xwlyamku3vdfzffv4i2izxbwrt24leet.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 589824, 'x': 294912}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask)
    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/ir/cirrkzaxotmpv4tkmcwpbu7sq7nrt4qzmgzhuiikw5k2cdwxl7jl.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 1179648, 'x': 589824}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask)
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/ro/crob34yrzjglj2jp7q2h3xvhqs2ltjypcwiksynmp2aipsr4prtt.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 2359296, 'x': 1179648}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask)
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/dm/cdmjip7n63wbcxo6giyxed6szdemwdl6wgmzotsfacn72e4b6c4j.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 4718592, 'x': 2359296}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask)
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/hh/chhdjatrc75htqme2ikhi2f2zjh6vt5qmuvmzr3wbx7zkziw5p4c.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_7 = async_compile.triton('triton_poi_fused_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 9437184, 'x': 4718592}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask)
    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/hn/chnbf2w47gr2tlzbootrwnz4j2zkkbr5ct5d72dolr74ricf44g2.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_8 = async_compile.triton('triton_poi_fused_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 18874368, 'x': 9437184}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask)
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/ej/cejfauudo6jprwmtjdjcomddgtmqtq4rygjj27aezvkxjczmyqyn.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x_2 => relu
# Graph fragment:
#   %convolution : Tensor "f32[2, 64, 16, 16][16384, 1, 1024, 64]cuda:0" = PlaceHolder[target=convolution]
#   %primals_3 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_3]
#   %primals_4 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_5 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_5]
#   %primals_6 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_6]
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_3, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %sub : Tensor "f32[2, 64, 16, 16][16384, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : Tensor "f32[2, 64, 16, 16][16384, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_5, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[2, 64, 16, 16][16384, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_6, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[2, 64, 16, 16][16384, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[2, 64, 16, 16][16384, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   return %relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 394240}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 0, tl.int32)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/mx/cmxv4sdtibdapic2hkxmup6wjmesbzqbkjzso5kvur7gm36hezpn.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_1 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x_2 => relu
#   x_3 => _low_memory_max_pool_with_offsets, getitem, getitem_1
# Graph fragment:
#   %relu : Tensor "f32[2, 64, 16, 16][16384, 1, 1024, 64]cuda:0" = PlaceHolder[target=relu]
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_3, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %sub : Tensor "f32[2, 64, 16, 16][16384, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : Tensor "f32[2, 64, 16, 16][16384, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_5, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[2, 64, 16, 16][16384, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_6, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[2, 64, 16, 16][16384, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[2, 64, 16, 16][16384, 256, 16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem : Tensor "f32[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets, 0), kwargs = {})
#   %getitem_1 : Tensor "i8[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool_with_offsets, 1), kwargs = {})
#   return %getitem,%getitem_1
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 376832}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_10(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = ((xindex // 512) % 8)
    x1 = ((xindex // 64) % 8)
    x0 = (xindex % 64)
    x5 = xindex // 512
    x6 = xindex
    tmp0 = ((-1) + 2*x2).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = ((-1) + 2*x1).to(tl.int32)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-1088) + x0 + 128*x1 + 2048*x5), tmp10, other=float("-inf"))
    tmp12 = (2*x1).to(tl.int32)
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-1024) + x0 + 128*x1 + 2048*x5), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = (1 + 2*x1).to(tl.int32)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-960) + x0 + 128*x1 + 2048*x5), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = (2*x2).to(tl.int32)
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + 128*x1 + 2048*x5), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 128*x1 + 2048*x5), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 2048*x5), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = (1 + 2*x2).to(tl.int32)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (960 + x0 + 128*x1 + 2048*x5), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (1024 + x0 + 128*x1 + 2048*x5), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (1088 + x0 + 128*x1 + 2048*x5), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tmp52 = tmp11 > tmp17
    tmp53 = tmp11 == tmp17
    tmp54 = tmp11 != tmp11
    tmp55 = tmp17 != tmp17
    tmp56 = tmp54 > tmp55
    tmp57 = tmp52 | tmp56
    tmp58 = tmp54 & tmp55
    tmp59 = tmp53 | tmp58
    tmp60 = tl.full([1], 1, tl.int64)
    tmp61 = tmp1 < tmp60
    tmp62 = tmp59 & tmp61
    tmp63 = tmp57 | tmp62
    tmp64 = tl.where(tmp63, tmp11, tmp17)
    tmp65 = tl.where(tmp63, tmp1, tmp60)
    tmp66 = tmp64 > tmp24
    tmp67 = tmp64 == tmp24
    tmp68 = tmp64 != tmp64
    tmp69 = tmp24 != tmp24
    tmp70 = tmp68 > tmp69
    tmp71 = tmp66 | tmp70
    tmp72 = tmp68 & tmp69
    tmp73 = tmp67 | tmp72
    tmp74 = tl.full([1], 2, tl.int64)
    tmp75 = tmp65 < tmp74
    tmp76 = tmp73 & tmp75
    tmp77 = tmp71 | tmp76
    tmp78 = tl.where(tmp77, tmp64, tmp24)
    tmp79 = tl.where(tmp77, tmp65, tmp74)
    tmp80 = tmp78 > tmp31
    tmp81 = tmp78 == tmp31
    tmp82 = tmp78 != tmp78
    tmp83 = tmp31 != tmp31
    tmp84 = tmp82 > tmp83
    tmp85 = tmp80 | tmp84
    tmp86 = tmp82 & tmp83
    tmp87 = tmp81 | tmp86
    tmp88 = tl.full([1], 3, tl.int64)
    tmp89 = tmp79 < tmp88
    tmp90 = tmp87 & tmp89
    tmp91 = tmp85 | tmp90
    tmp92 = tl.where(tmp91, tmp78, tmp31)
    tmp93 = tl.where(tmp91, tmp79, tmp88)
    tmp94 = tmp92 > tmp34
    tmp95 = tmp92 == tmp34
    tmp96 = tmp92 != tmp92
    tmp97 = tmp34 != tmp34
    tmp98 = tmp96 > tmp97
    tmp99 = tmp94 | tmp98
    tmp100 = tmp96 & tmp97
    tmp101 = tmp95 | tmp100
    tmp102 = tl.full([1], 4, tl.int64)
    tmp103 = tmp93 < tmp102
    tmp104 = tmp101 & tmp103
    tmp105 = tmp99 | tmp104
    tmp106 = tl.where(tmp105, tmp92, tmp34)
    tmp107 = tl.where(tmp105, tmp93, tmp102)
    tmp108 = tmp106 > tmp37
    tmp109 = tmp106 == tmp37
    tmp110 = tmp106 != tmp106
    tmp111 = tmp37 != tmp37
    tmp112 = tmp110 > tmp111
    tmp113 = tmp108 | tmp112
    tmp114 = tmp110 & tmp111
    tmp115 = tmp109 | tmp114
    tmp116 = tl.full([1], 5, tl.int64)
    tmp117 = tmp107 < tmp116
    tmp118 = tmp115 & tmp117
    tmp119 = tmp113 | tmp118
    tmp120 = tl.where(tmp119, tmp106, tmp37)
    tmp121 = tl.where(tmp119, tmp107, tmp116)
    tmp122 = tmp120 > tmp44
    tmp123 = tmp120 == tmp44
    tmp124 = tmp120 != tmp120
    tmp125 = tmp44 != tmp44
    tmp126 = tmp124 > tmp125
    tmp127 = tmp122 | tmp126
    tmp128 = tmp124 & tmp125
    tmp129 = tmp123 | tmp128
    tmp130 = tl.full([1], 6, tl.int64)
    tmp131 = tmp121 < tmp130
    tmp132 = tmp129 & tmp131
    tmp133 = tmp127 | tmp132
    tmp134 = tl.where(tmp133, tmp120, tmp44)
    tmp135 = tl.where(tmp133, tmp121, tmp130)
    tmp136 = tmp134 > tmp47
    tmp137 = tmp134 == tmp47
    tmp138 = tmp134 != tmp134
    tmp139 = tmp47 != tmp47
    tmp140 = tmp138 > tmp139
    tmp141 = tmp136 | tmp140
    tmp142 = tmp138 & tmp139
    tmp143 = tmp137 | tmp142
    tmp144 = tl.full([1], 7, tl.int64)
    tmp145 = tmp135 < tmp144
    tmp146 = tmp143 & tmp145
    tmp147 = tmp141 | tmp146
    tmp148 = tl.where(tmp147, tmp134, tmp47)
    tmp149 = tl.where(tmp147, tmp135, tmp144)
    tmp150 = tmp148 > tmp50
    tmp151 = tmp148 == tmp50
    tmp152 = tmp148 != tmp148
    tmp153 = tmp50 != tmp50
    tmp154 = tmp152 > tmp153
    tmp155 = tmp150 | tmp154
    tmp156 = tmp152 & tmp153
    tmp157 = tmp151 | tmp156
    tmp158 = tl.full([1], 8, tl.int64)
    tmp159 = tmp149 < tmp158
    tmp160 = tmp157 & tmp159
    tmp161 = tmp155 | tmp160
    tmp162 = tl.where(tmp161, tmp148, tmp50)
    tmp163 = tl.where(tmp161, tmp149, tmp158)
    tmp164 = tmp163.to(tl.int8)
    tl.store(out_ptr0 + (x6), tmp51, None)
    tl.store(out_ptr1 + (x6), tmp164, None)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/f2/cf25i2sxm6bksget3selm7ols4sb2jz6iozyolblwduuwffmk5f5.py
# Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_1 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   out_2 => relu_1
# Graph fragment:
#   %convolution_1 : Tensor "f32[2, 64, 8, 8][4096, 1, 512, 64]cuda:0" = PlaceHolder[target=convolution_1]
#   %primals_8 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_8]
#   %primals_9 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_9]
#   %primals_10 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_10]
#   %primals_11 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_11]
#   %add_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_9, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_8 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_8, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %sub_1 : Tensor "f32[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : Tensor "f32[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_10, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_11, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : Tensor "f32[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   return %relu_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 99328}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 0, tl.int32)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/wq/cwq2uykuxcozacz4qnsnj3wst6mhboq7vsgitsig6i2yebxzux72.py
# Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_4 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   out_5 => add_6
#   out_6 => relu_2
# Graph fragment:
#   %convolution_2 : Tensor "f32[2, 64, 8, 8][4096, 1, 512, 64]cuda:0" = PlaceHolder[target=convolution_2]
#   %primals_13 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_13]
#   %primals_14 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_14]
#   %primals_15 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_15]
#   %primals_16 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=primals_16]
#   %getitem : Tensor "f32[2, 64, 8, 8][4096, 1, 512, 64]cuda:0" = PlaceHolder[target=getitem]
#   %add_4 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_14, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_16 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_13, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %sub_2 : Tensor "f32[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : Tensor "f32[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_15, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_16, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %add_6 : Tensor "f32[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %getitem), kwargs = {})
#   %relu_2 : Tensor "f32[2, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_6,), kwargs = {})
#   return %relu_2
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 132096}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tl.store(out_ptr0 + (x2), tmp18, None)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/qw/cqwkuppdp3unto27kym2eh75r2jjzswgjo3a3xwmkcfylql43lql.py
# Topologically Sorted Source Nodes: [out_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_15 => add_12, add_13, mul_15, mul_16, mul_17, reciprocal_5, sqrt_5, sub_5, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, unsqueeze_44, unsqueeze_45, unsqueeze_46, unsqueeze_47
#   out_16 => relu_5
# Graph fragment:
#   %convolution_5 : Tensor "f32[2, 128, 4, 4][2048, 1, 512, 128]cuda:0" = PlaceHolder[target=convolution_5]
#   %primals_28 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_28]
#   %primals_29 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_29]
#   %primals_30 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_30]
#   %primals_31 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_31]
#   %add_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_29, 1e-05), kwargs = {})
#   %sqrt_5 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_12,), kwargs = {})
#   %reciprocal_5 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_5,), kwargs = {})
#   %mul_15 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 1), kwargs = {})
#   %unsqueeze_40 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_28, -1), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, -1), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_15, -1), kwargs = {})
#   %unsqueeze_43 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_42, -1), kwargs = {})
#   %sub_5 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_16 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %unsqueeze_44 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_30, -1), kwargs = {})
#   %unsqueeze_45 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_44, -1), kwargs = {})
#   %mul_17 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %unsqueeze_46 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_31, -1), kwargs = {})
#   %unsqueeze_47 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_46, -1), kwargs = {})
#   %add_13 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_5 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
#   return %relu_5
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 51200}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 0, tl.int32)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/si/csivaj3ne75k4lj4zka5mkkndomu4soxpzvpqttaijo5bpnt2iax.py
# Topologically Sorted Source Nodes: [out_18, input_2, out_19, out_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_16, add_17, mul_21, mul_22, mul_23, reciprocal_7, sqrt_7, sub_7, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
#   out_18 => add_14, add_15, mul_18, mul_19, mul_20, reciprocal_6, sqrt_6, sub_6, unsqueeze_48, unsqueeze_49, unsqueeze_50, unsqueeze_51, unsqueeze_52, unsqueeze_53, unsqueeze_54, unsqueeze_55
#   out_19 => add_18
#   out_20 => relu_6
# Graph fragment:
#   %convolution_6 : Tensor "f32[2, 128, 4, 4][2048, 1, 512, 128]cuda:0" = PlaceHolder[target=convolution_6]
#   %primals_33 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_33]
#   %primals_34 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_34]
#   %primals_35 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_35]
#   %primals_36 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_36]
#   %convolution_7 : Tensor "f32[2, 128, 4, 4][2048, 1, 512, 128]cuda:0" = PlaceHolder[target=convolution_7]
#   %primals_38 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_38]
#   %primals_39 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_39]
#   %primals_40 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_40]
#   %primals_41 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_41]
#   %add_18 : Tensor "f32[2, 128, 4, 4][2048, 1, 512, 128]cuda:0" = PlaceHolder[target=add_18]
#   %add_14 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_34, 1e-05), kwargs = {})
#   %sqrt_6 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_14,), kwargs = {})
#   %reciprocal_6 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_6,), kwargs = {})
#   %mul_18 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, 1), kwargs = {})
#   %unsqueeze_48 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_33, -1), kwargs = {})
#   %unsqueeze_49 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_48, -1), kwargs = {})
#   %unsqueeze_50 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %unsqueeze_51 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_50, -1), kwargs = {})
#   %sub_6 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %unsqueeze_52 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_35, -1), kwargs = {})
#   %unsqueeze_53 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_52, -1), kwargs = {})
#   %mul_20 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %unsqueeze_54 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_36, -1), kwargs = {})
#   %unsqueeze_55 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_54, -1), kwargs = {})
#   %add_15 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %add_16 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_39, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_16,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_21 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_56 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_38, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_21, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %sub_7 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_22 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_40, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_23 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_41, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_17 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %add_18 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %add_17), kwargs = {})
#   %relu_6 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_18,), kwargs = {})
#   return %add_18,%relu_6
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 69632}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), None)
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt_rn(tmp19)
    tmp21 = (tmp7 / tmp20)
    tmp22 = tmp21 * tmp7
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = tl.full([1], 0, tl.int32)
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tl.store(in_out_ptr0 + (x2), tmp30, None)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/6v/c6vo3xuoph6vctrnqv5gztypsf2ql2kujwxu3cfprp5yshpumns3.py
# Topologically Sorted Source Nodes: [out_25, out_26, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_25 => add_21, add_22, mul_27, mul_28, mul_29, reciprocal_9, sqrt_9, sub_9, unsqueeze_72, unsqueeze_73, unsqueeze_74, unsqueeze_75, unsqueeze_76, unsqueeze_77, unsqueeze_78, unsqueeze_79
#   out_26 => add_23
#   out_27 => relu_8
# Graph fragment:
#   %convolution_9 : Tensor "f32[2, 128, 4, 4][2048, 1, 512, 128]cuda:0" = PlaceHolder[target=convolution_9]
#   %primals_48 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_48]
#   %primals_49 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_49]
#   %primals_50 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_50]
#   %primals_51 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=primals_51]
#   %relu_6 : Tensor "f32[2, 128, 4, 4][2048, 1, 512, 128]cuda:0" = PlaceHolder[target=relu_6]
#   %add_21 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_49, 1e-05), kwargs = {})
#   %sqrt_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_21,), kwargs = {})
#   %reciprocal_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_9,), kwargs = {})
#   %mul_27 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_9, 1), kwargs = {})
#   %unsqueeze_72 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_48, -1), kwargs = {})
#   %unsqueeze_73 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_72, -1), kwargs = {})
#   %unsqueeze_74 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_27, -1), kwargs = {})
#   %unsqueeze_75 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_74, -1), kwargs = {})
#   %sub_9 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_28 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %unsqueeze_76 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_50, -1), kwargs = {})
#   %unsqueeze_77 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_76, -1), kwargs = {})
#   %mul_29 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %unsqueeze_78 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_51, -1), kwargs = {})
#   %unsqueeze_79 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_78, -1), kwargs = {})
#   %add_22 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %add_23 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %relu_6), kwargs = {})
#   %relu_8 : Tensor "f32[2, 128, 4, 4][2048, 16, 4, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_23,), kwargs = {})
#   return %relu_8
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 67584}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), None)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tl.store(out_ptr0 + (x2), tmp18, None)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/v6/cv636wrt5jvldy7ewbusegtgh23zgk4opio6sekusu4vksh3au4b.py
# Topologically Sorted Source Nodes: [out_29, out_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_29 => add_24, add_25, mul_30, mul_31, mul_32, reciprocal_10, sqrt_10, sub_10, unsqueeze_80, unsqueeze_81, unsqueeze_82, unsqueeze_83, unsqueeze_84, unsqueeze_85, unsqueeze_86, unsqueeze_87
#   out_30 => relu_9
# Graph fragment:
#   %convolution_10 : Tensor "f32[2, 256, 2, 2][1024, 1, 512, 256]cuda:0" = PlaceHolder[target=convolution_10]
#   %primals_53 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_53]
#   %primals_54 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_54]
#   %primals_55 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_55]
#   %primals_56 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_56]
#   %add_24 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_54, 1e-05), kwargs = {})
#   %sqrt_10 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_24,), kwargs = {})
#   %reciprocal_10 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_10,), kwargs = {})
#   %mul_30 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_10, 1), kwargs = {})
#   %unsqueeze_80 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_53, -1), kwargs = {})
#   %unsqueeze_81 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_80, -1), kwargs = {})
#   %unsqueeze_82 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_30, -1), kwargs = {})
#   %unsqueeze_83 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_82, -1), kwargs = {})
#   %sub_10 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %mul_31 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %unsqueeze_84 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_55, -1), kwargs = {})
#   %unsqueeze_85 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_84, -1), kwargs = {})
#   %mul_32 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %unsqueeze_86 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_56, -1), kwargs = {})
#   %unsqueeze_87 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_86, -1), kwargs = {})
#   %add_25 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %relu_9 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_25,), kwargs = {})
#   return %relu_9
triton_poi_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 28672}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 0, tl.int32)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/6q/c6qjjh3nabt45ktds6jcquugcrl5rbj55tnrryttjif6qo5az3ji.py
# Topologically Sorted Source Nodes: [out_32, input_4, out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_4 => add_28, add_29, mul_36, mul_37, mul_38, reciprocal_12, sqrt_12, sub_12, unsqueeze_100, unsqueeze_101, unsqueeze_102, unsqueeze_103, unsqueeze_96, unsqueeze_97, unsqueeze_98, unsqueeze_99
#   out_32 => add_26, add_27, mul_33, mul_34, mul_35, reciprocal_11, sqrt_11, sub_11, unsqueeze_88, unsqueeze_89, unsqueeze_90, unsqueeze_91, unsqueeze_92, unsqueeze_93, unsqueeze_94, unsqueeze_95
#   out_33 => add_30
#   out_34 => relu_10
# Graph fragment:
#   %convolution_11 : Tensor "f32[2, 256, 2, 2][1024, 1, 512, 256]cuda:0" = PlaceHolder[target=convolution_11]
#   %primals_58 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_58]
#   %primals_59 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_59]
#   %primals_60 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_60]
#   %primals_61 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_61]
#   %convolution_12 : Tensor "f32[2, 256, 2, 2][1024, 1, 512, 256]cuda:0" = PlaceHolder[target=convolution_12]
#   %primals_63 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_63]
#   %primals_64 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_64]
#   %primals_65 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_65]
#   %primals_66 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_66]
#   %add_30 : Tensor "f32[2, 256, 2, 2][1024, 1, 512, 256]cuda:0" = PlaceHolder[target=add_30]
#   %add_26 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_59, 1e-05), kwargs = {})
#   %sqrt_11 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_26,), kwargs = {})
#   %reciprocal_11 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_11,), kwargs = {})
#   %mul_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_11, 1), kwargs = {})
#   %unsqueeze_88 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_58, -1), kwargs = {})
#   %unsqueeze_89 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_88, -1), kwargs = {})
#   %unsqueeze_90 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_33, -1), kwargs = {})
#   %unsqueeze_91 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_90, -1), kwargs = {})
#   %sub_11 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_34 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %unsqueeze_92 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_60, -1), kwargs = {})
#   %unsqueeze_93 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_92, -1), kwargs = {})
#   %mul_35 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %unsqueeze_94 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_61, -1), kwargs = {})
#   %unsqueeze_95 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_94, -1), kwargs = {})
#   %add_27 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %add_28 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_64, 1e-05), kwargs = {})
#   %sqrt_12 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_28,), kwargs = {})
#   %reciprocal_12 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_12,), kwargs = {})
#   %mul_36 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, 1), kwargs = {})
#   %unsqueeze_96 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_63, -1), kwargs = {})
#   %unsqueeze_97 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_96, -1), kwargs = {})
#   %unsqueeze_98 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_36, -1), kwargs = {})
#   %unsqueeze_99 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_98, -1), kwargs = {})
#   %sub_12 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %mul_37 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %unsqueeze_100 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_65, -1), kwargs = {})
#   %unsqueeze_101 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_100, -1), kwargs = {})
#   %mul_38 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %unsqueeze_102 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_66, -1), kwargs = {})
#   %unsqueeze_103 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_102, -1), kwargs = {})
#   %add_29 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %add_30 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %add_29), kwargs = {})
#   %relu_10 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_30,), kwargs = {})
#   return %add_30,%relu_10
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 40960}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
    tmp16 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt_rn(tmp19)
    tmp21 = (tmp7 / tmp20)
    tmp22 = tmp21 * tmp7
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = tl.full([1], 0, tl.int32)
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tl.store(in_out_ptr0 + (x2), tmp30, xmask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/5b/c5bqemiprj6xe66o7ctorhy777acrjtadmbof3aasngfupzjalg6.py
# Topologically Sorted Source Nodes: [out_39, out_40, out_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_39 => add_33, add_34, mul_42, mul_43, mul_44, reciprocal_14, sqrt_14, sub_14, unsqueeze_112, unsqueeze_113, unsqueeze_114, unsqueeze_115, unsqueeze_116, unsqueeze_117, unsqueeze_118, unsqueeze_119
#   out_40 => add_35
#   out_41 => relu_12
# Graph fragment:
#   %convolution_14 : Tensor "f32[2, 256, 2, 2][1024, 1, 512, 256]cuda:0" = PlaceHolder[target=convolution_14]
#   %primals_73 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_73]
#   %primals_74 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_74]
#   %primals_75 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_75]
#   %primals_76 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=primals_76]
#   %relu_10 : Tensor "f32[2, 256, 2, 2][1024, 1, 512, 256]cuda:0" = PlaceHolder[target=relu_10]
#   %add_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_74, 1e-05), kwargs = {})
#   %sqrt_14 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_33,), kwargs = {})
#   %reciprocal_14 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_14,), kwargs = {})
#   %mul_42 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_14, 1), kwargs = {})
#   %unsqueeze_112 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_73, -1), kwargs = {})
#   %unsqueeze_113 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_112, -1), kwargs = {})
#   %unsqueeze_114 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_42, -1), kwargs = {})
#   %unsqueeze_115 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_114, -1), kwargs = {})
#   %sub_14 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %mul_43 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %unsqueeze_116 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_75, -1), kwargs = {})
#   %unsqueeze_117 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_116, -1), kwargs = {})
#   %mul_44 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %unsqueeze_118 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_76, -1), kwargs = {})
#   %unsqueeze_119 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_118, -1), kwargs = {})
#   %add_34 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %add_35 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %relu_10), kwargs = {})
#   %relu_12 : Tensor "f32[2, 256, 2, 2][1024, 4, 2, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_35,), kwargs = {})
#   return %relu_12
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 36864}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tl.store(out_ptr0 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/gz/cgzovapqf2acygsbr2djkopv2i4f2l2ckjiafpddobul2pixx4or.py
# Topologically Sorted Source Nodes: [out_43, out_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_43 => add_36, add_37, mul_45, mul_46, mul_47, reciprocal_15, sqrt_15, sub_15, unsqueeze_120, unsqueeze_121, unsqueeze_122, unsqueeze_123, unsqueeze_124, unsqueeze_125, unsqueeze_126, unsqueeze_127
#   out_44 => relu_13
# Graph fragment:
#   %convolution_15 : Tensor "f32[2, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=convolution_15]
#   %primals_78 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_78]
#   %primals_79 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_79]
#   %primals_80 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_80]
#   %primals_81 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_81]
#   %add_36 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_79, 1e-05), kwargs = {})
#   %sqrt_15 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_36,), kwargs = {})
#   %reciprocal_15 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_15,), kwargs = {})
#   %mul_45 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_15, 1), kwargs = {})
#   %unsqueeze_120 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_78, -1), kwargs = {})
#   %unsqueeze_121 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_120, -1), kwargs = {})
#   %unsqueeze_122 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_45, -1), kwargs = {})
#   %unsqueeze_123 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_122, -1), kwargs = {})
#   %sub_15 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_121), kwargs = {})
#   %mul_46 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %unsqueeze_124 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_80, -1), kwargs = {})
#   %unsqueeze_125 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_124, -1), kwargs = {})
#   %mul_47 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %unsqueeze_126 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_81, -1), kwargs = {})
#   %unsqueeze_127 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_126, -1), kwargs = {})
#   %add_37 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %relu_13 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_37,), kwargs = {})
#   return %relu_13
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 20480}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 0, tl.int32)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/q6/cq6hanwydju3i6ify7stfikdhbmhgxyl52dpgs7g4f76637gic6u.py
# Topologically Sorted Source Nodes: [out_46, input_6, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_6 => add_40, add_41, mul_51, mul_52, mul_53, reciprocal_17, sqrt_17, sub_17, unsqueeze_136, unsqueeze_137, unsqueeze_138, unsqueeze_139, unsqueeze_140, unsqueeze_141, unsqueeze_142, unsqueeze_143
#   out_46 => add_38, add_39, mul_48, mul_49, mul_50, reciprocal_16, sqrt_16, sub_16, unsqueeze_128, unsqueeze_129, unsqueeze_130, unsqueeze_131, unsqueeze_132, unsqueeze_133, unsqueeze_134, unsqueeze_135
#   out_47 => add_42
#   out_48 => relu_14
# Graph fragment:
#   %convolution_16 : Tensor "f32[2, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=convolution_16]
#   %primals_83 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_83]
#   %primals_84 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_84]
#   %primals_85 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_85]
#   %primals_86 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_86]
#   %convolution_17 : Tensor "f32[2, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=convolution_17]
#   %primals_88 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_88]
#   %primals_89 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_89]
#   %primals_90 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_90]
#   %primals_91 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_91]
#   %add_42 : Tensor "f32[2, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=add_42]
#   %add_38 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_84, 1e-05), kwargs = {})
#   %sqrt_16 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_38,), kwargs = {})
#   %reciprocal_16 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_16,), kwargs = {})
#   %mul_48 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_16, 1), kwargs = {})
#   %unsqueeze_128 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_83, -1), kwargs = {})
#   %unsqueeze_129 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_128, -1), kwargs = {})
#   %unsqueeze_130 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_48, -1), kwargs = {})
#   %unsqueeze_131 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_130, -1), kwargs = {})
#   %sub_16 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_129), kwargs = {})
#   %mul_49 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_131), kwargs = {})
#   %unsqueeze_132 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_85, -1), kwargs = {})
#   %unsqueeze_133 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_132, -1), kwargs = {})
#   %mul_50 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %unsqueeze_134 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_86, -1), kwargs = {})
#   %unsqueeze_135 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_134, -1), kwargs = {})
#   %add_39 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %add_40 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_89, 1e-05), kwargs = {})
#   %sqrt_17 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_40,), kwargs = {})
#   %reciprocal_17 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_17,), kwargs = {})
#   %mul_51 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_17, 1), kwargs = {})
#   %unsqueeze_136 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_88, -1), kwargs = {})
#   %unsqueeze_137 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_136, -1), kwargs = {})
#   %unsqueeze_138 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_51, -1), kwargs = {})
#   %unsqueeze_139 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_138, -1), kwargs = {})
#   %sub_17 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_52 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %unsqueeze_140 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_90, -1), kwargs = {})
#   %unsqueeze_141 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_140, -1), kwargs = {})
#   %mul_53 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %unsqueeze_142 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_91, -1), kwargs = {})
#   %unsqueeze_143 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_142, -1), kwargs = {})
#   %add_41 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %add_42 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %add_41), kwargs = {})
#   %relu_14 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_42,), kwargs = {})
#   return %add_42,%relu_14
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 32768}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
    tmp16 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt_rn(tmp19)
    tmp21 = (tmp7 / tmp20)
    tmp22 = tmp21 * tmp7
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = tl.full([1], 0, tl.int32)
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tl.store(in_out_ptr0 + (x2), tmp30, xmask)
''', device_str='cuda')


# kernel path: /tmp/tensorplay-torchinductor-debug.nBJfto/kt/cktlm2t6cn725k2qk5papitvpho5cjc3g4uqs246wwpox22llwtc.py
# Topologically Sorted Source Nodes: [out_53, out_54, out_55, x_4, le], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean, aten.threshold_backward]
# Source node to ATen node mapping:
#   le => le
#   out_53 => add_45, add_46, mul_57, mul_58, mul_59, reciprocal_19, sqrt_19, sub_19, unsqueeze_152, unsqueeze_153, unsqueeze_154, unsqueeze_155, unsqueeze_156, unsqueeze_157, unsqueeze_158, unsqueeze_159
#   out_54 => add_47
#   out_55 => relu_16
#   x_4 => mean
# Graph fragment:
#   %convolution_19 : Tensor "f32[2, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=convolution_19]
#   %primals_98 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_98]
#   %primals_99 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_99]
#   %primals_100 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_100]
#   %primals_101 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=primals_101]
#   %relu_14 : Tensor "f32[2, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=relu_14]
#   %relu_16 : Tensor "f32[2, 512, 1, 1][512, 1, 1024, 1024]cuda:0" = PlaceHolder[target=relu_16]
#   %add_45 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_99, 1e-05), kwargs = {})
#   %sqrt_19 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_45,), kwargs = {})
#   %reciprocal_19 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_19,), kwargs = {})
#   %mul_57 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_19, 1), kwargs = {})
#   %unsqueeze_152 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_98, -1), kwargs = {})
#   %unsqueeze_153 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_152, -1), kwargs = {})
#   %unsqueeze_154 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_57, -1), kwargs = {})
#   %unsqueeze_155 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_154, -1), kwargs = {})
#   %sub_19 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_153), kwargs = {})
#   %mul_58 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_155), kwargs = {})
#   %unsqueeze_156 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_100, -1), kwargs = {})
#   %unsqueeze_157 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_156, -1), kwargs = {})
#   %mul_59 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_157), kwargs = {})
#   %unsqueeze_158 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_101, -1), kwargs = {})
#   %unsqueeze_159 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_158, -1), kwargs = {})
#   %add_46 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_159), kwargs = {})
#   %add_47 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_46, %relu_14), kwargs = {})
#   %relu_16 : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_47,), kwargs = {})
#   %mean : Tensor "f32[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_16, [-1, -2], True), kwargs = {})
#   %le : Tensor "b8[2, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_16, 0), kwargs = {})
#   return %relu_16,%mean,%le
triton_poi_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 26624}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask)
    tmp2 = tmp0 - tmp1
    tmp4 = tl.full([1], 1e-05, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt_rn(tmp5)
    tmp7 = tl.full([1], 1.0, tl.float32)
    tmp8 = (tmp7 / tmp6)
    tmp9 = tmp8 * tmp7
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = (tmp18 / tmp7)
    tmp20 = tl.full([1], 0.0, tl.float32)
    tmp21 = tmp18 <= tmp20
    tl.store(out_ptr1 + (x2), tmp19, xmask)
    tl.store(out_ptr2 + (x2), tmp21, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103 = args
        args.clear()
        assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_0:1
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_0.run(primals_1, buf0, 192, 49, stream=raw_stream0)
            del primals_1
            assert_size_stride(primals_2, (2, 3, 32, 32), (3072, 1024, 32, 1), 'input')
            primals_2 = copy_if_misaligned(primals_2)
            buf1 = empty_strided_cuda((2, 3, 32, 32), (3072, 1, 96, 3), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_1:2
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_1.run(primals_2, buf1, 6, 1024, stream=raw_stream0)
            del primals_2
            assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1), 'input')
            buf2 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_2:3
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_2.run(primals_7, buf2, 4096, 9, stream=raw_stream0)
            del primals_7
            assert_size_stride(primals_12, (64, 64, 3, 3), (576, 9, 3, 1), 'input')
            buf3 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_2:4
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_2.run(primals_12, buf3, 4096, 9, stream=raw_stream0)
            del primals_12
            assert_size_stride(primals_17, (64, 64, 3, 3), (576, 9, 3, 1), 'input')
            buf4 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_2:5
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_2.run(primals_17, buf4, 4096, 9, stream=raw_stream0)
            del primals_17
            assert_size_stride(primals_22, (64, 64, 3, 3), (576, 9, 3, 1), 'input')
            buf5 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_2:6
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_2.run(primals_22, buf5, 4096, 9, stream=raw_stream0)
            del primals_22
            assert_size_stride(primals_27, (128, 64, 3, 3), (576, 9, 3, 1), 'input')
            buf6 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_3:7
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_3.run(primals_27, buf6, 8192, 9, stream=raw_stream0)
            del primals_27
            assert_size_stride(primals_32, (128, 128, 3, 3), (1152, 9, 3, 1), 'input')
            buf7 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_4:8
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_4.run(primals_32, buf7, 16384, 9, stream=raw_stream0)
            del primals_32
            assert_size_stride(primals_42, (128, 128, 3, 3), (1152, 9, 3, 1), 'input')
            buf8 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_4:9
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_4.run(primals_42, buf8, 16384, 9, stream=raw_stream0)
            del primals_42
            assert_size_stride(primals_47, (128, 128, 3, 3), (1152, 9, 3, 1), 'input')
            buf9 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_4:10
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_4.run(primals_47, buf9, 16384, 9, stream=raw_stream0)
            del primals_47
            assert_size_stride(primals_52, (256, 128, 3, 3), (1152, 9, 3, 1), 'input')
            buf10 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_5:11
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_5.run(primals_52, buf10, 32768, 9, stream=raw_stream0)
            del primals_52
            assert_size_stride(primals_57, (256, 256, 3, 3), (2304, 9, 3, 1), 'input')
            buf11 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_6:12
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_6.run(primals_57, buf11, 65536, 9, stream=raw_stream0)
            del primals_57
            assert_size_stride(primals_67, (256, 256, 3, 3), (2304, 9, 3, 1), 'input')
            buf12 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_6:13
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_6.run(primals_67, buf12, 65536, 9, stream=raw_stream0)
            del primals_67
            assert_size_stride(primals_72, (256, 256, 3, 3), (2304, 9, 3, 1), 'input')
            buf13 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_6:14
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_6.run(primals_72, buf13, 65536, 9, stream=raw_stream0)
            del primals_72
            assert_size_stride(primals_77, (512, 256, 3, 3), (2304, 9, 3, 1), 'input')
            buf14 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_7:15
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_7.run(primals_77, buf14, 131072, 9, stream=raw_stream0)
            del primals_77
            assert_size_stride(primals_82, (512, 512, 3, 3), (4608, 9, 3, 1), 'input')
            buf15 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_8:16
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_8.run(primals_82, buf15, 262144, 9, stream=raw_stream0)
            del primals_82
            assert_size_stride(primals_92, (512, 512, 3, 3), (4608, 9, 3, 1), 'input')
            buf16 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_8:17
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_8.run(primals_92, buf16, 262144, 9, stream=raw_stream0)
            del primals_92
            assert_size_stride(primals_97, (512, 512, 3, 3), (4608, 9, 3, 1), 'input')
            buf17 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Unsorted Source Nodes: [], Original ATen: []
            # [Provenance debug handles] triton_poi_fused_8:18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_8.run(primals_97, buf17, 262144, 9, stream=raw_stream0)
            del primals_97
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:19
            buf18 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf18, (2, 64, 16, 16), (16384, 1, 1024, 64), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_3, (64, ), (1, ), 'input')
            assert_size_stride(primals_4, (64, ), (1, ), 'input')
            assert_size_stride(primals_5, (64, ), (1, ), 'input')
            assert_size_stride(primals_6, (64, ), (1, ), 'input')
            buf19 = empty_strided_cuda((2, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_9:20
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf18, primals_3, primals_4, primals_5, primals_6, buf19, 32768, stream=raw_stream0)
            buf20 = empty_strided_cuda((2, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
            buf21 = empty_strided_cuda((2, 64, 8, 8), (4096, 1, 512, 64), torch.int8)
            # Topologically Sorted Source Nodes: [x_1, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_10:21
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_10.run(buf19, buf20, buf21, 8192, stream=raw_stream0)
            del buf19
            # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:22
            buf22 = extern_kernels.convolution(buf20, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf22, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_8, (64, ), (1, ), 'input')
            assert_size_stride(primals_9, (64, ), (1, ), 'input')
            assert_size_stride(primals_10, (64, ), (1, ), 'input')
            assert_size_stride(primals_11, (64, ), (1, ), 'input')
            buf23 = empty_strided_cuda((2, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
            # Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_11:23
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf22, primals_8, primals_9, primals_10, primals_11, buf23, 8192, stream=raw_stream0)
            del primals_11
            # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:24
            buf24 = extern_kernels.convolution(buf23, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf24, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_13, (64, ), (1, ), 'input')
            assert_size_stride(primals_14, (64, ), (1, ), 'input')
            assert_size_stride(primals_15, (64, ), (1, ), 'input')
            assert_size_stride(primals_16, (64, ), (1, ), 'input')
            buf25 = empty_strided_cuda((2, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
            # Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12:25
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf24, primals_13, primals_14, primals_15, primals_16, buf20, buf25, 8192, stream=raw_stream0)
            del primals_16
            # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:26
            buf26 = extern_kernels.convolution(buf25, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf26, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_18, (64, ), (1, ), 'input')
            assert_size_stride(primals_19, (64, ), (1, ), 'input')
            assert_size_stride(primals_20, (64, ), (1, ), 'input')
            assert_size_stride(primals_21, (64, ), (1, ), 'input')
            buf27 = empty_strided_cuda((2, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
            # Topologically Sorted Source Nodes: [out_8, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_11:27
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf26, primals_18, primals_19, primals_20, primals_21, buf27, 8192, stream=raw_stream0)
            del primals_21
            # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:28
            buf28 = extern_kernels.convolution(buf27, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf28, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_23, (64, ), (1, ), 'input')
            assert_size_stride(primals_24, (64, ), (1, ), 'input')
            assert_size_stride(primals_25, (64, ), (1, ), 'input')
            assert_size_stride(primals_26, (64, ), (1, ), 'input')
            buf29 = empty_strided_cuda((2, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
            # Topologically Sorted Source Nodes: [out_11, out_12, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12:29
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf28, primals_23, primals_24, primals_25, primals_26, buf25, buf29, 8192, stream=raw_stream0)
            del primals_26
            # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:30
            buf30 = extern_kernels.convolution(buf29, buf6, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf30, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_28, (128, ), (1, ), 'input')
            assert_size_stride(primals_29, (128, ), (1, ), 'input')
            assert_size_stride(primals_30, (128, ), (1, ), 'input')
            assert_size_stride(primals_31, (128, ), (1, ), 'input')
            buf31 = empty_strided_cuda((2, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
            # Topologically Sorted Source Nodes: [out_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_13:31
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf30, primals_28, primals_29, primals_30, primals_31, buf31, 4096, stream=raw_stream0)
            del primals_31
            # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:32
            buf32 = extern_kernels.convolution(buf31, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf32, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_37, (128, 64, 1, 1), (64, 1, 1, 1), 'input')
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:33
            buf33 = extern_kernels.convolution(buf29, reinterpret_tensor(primals_37, (128, 64, 1, 1), (64, 1, 64, 64), 0), stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf33, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_33, (128, ), (1, ), 'input')
            assert_size_stride(primals_34, (128, ), (1, ), 'input')
            assert_size_stride(primals_35, (128, ), (1, ), 'input')
            assert_size_stride(primals_36, (128, ), (1, ), 'input')
            assert_size_stride(primals_38, (128, ), (1, ), 'input')
            assert_size_stride(primals_39, (128, ), (1, ), 'input')
            assert_size_stride(primals_40, (128, ), (1, ), 'input')
            assert_size_stride(primals_41, (128, ), (1, ), 'input')
            buf34 = empty_strided_cuda((2, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
            buf35 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [out_18, input_2, out_19, out_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14:34
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_14.run(buf35, buf32, primals_33, primals_34, primals_35, primals_36, buf33, primals_38, primals_39, primals_40, primals_41, 4096, stream=raw_stream0)
            del primals_36
            del primals_41
            # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:35
            buf36 = extern_kernels.convolution(buf35, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf36, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_43, (128, ), (1, ), 'input')
            assert_size_stride(primals_44, (128, ), (1, ), 'input')
            assert_size_stride(primals_45, (128, ), (1, ), 'input')
            assert_size_stride(primals_46, (128, ), (1, ), 'input')
            buf37 = empty_strided_cuda((2, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
            # Topologically Sorted Source Nodes: [out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_13:36
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf36, primals_43, primals_44, primals_45, primals_46, buf37, 4096, stream=raw_stream0)
            del primals_46
            # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:37
            buf38 = extern_kernels.convolution(buf37, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf38, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_48, (128, ), (1, ), 'input')
            assert_size_stride(primals_49, (128, ), (1, ), 'input')
            assert_size_stride(primals_50, (128, ), (1, ), 'input')
            assert_size_stride(primals_51, (128, ), (1, ), 'input')
            buf39 = empty_strided_cuda((2, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
            # Topologically Sorted Source Nodes: [out_25, out_26, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15:38
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf38, primals_48, primals_49, primals_50, primals_51, buf35, buf39, 4096, stream=raw_stream0)
            del primals_51
            # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:39
            buf40 = extern_kernels.convolution(buf39, buf10, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf40, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_53, (256, ), (1, ), 'input')
            assert_size_stride(primals_54, (256, ), (1, ), 'input')
            assert_size_stride(primals_55, (256, ), (1, ), 'input')
            assert_size_stride(primals_56, (256, ), (1, ), 'input')
            buf41 = empty_strided_cuda((2, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
            # Topologically Sorted Source Nodes: [out_29, out_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_16:40
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf40, primals_53, primals_54, primals_55, primals_56, buf41, 2048, stream=raw_stream0)
            del primals_56
            # Topologically Sorted Source Nodes: [out_31], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:41
            buf42 = extern_kernels.convolution(buf41, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf42, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_62, (256, 128, 1, 1), (128, 1, 1, 1), 'input')
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:42
            buf43 = extern_kernels.convolution(buf39, reinterpret_tensor(primals_62, (256, 128, 1, 1), (128, 1, 128, 128), 0), stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf43, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_58, (256, ), (1, ), 'input')
            assert_size_stride(primals_59, (256, ), (1, ), 'input')
            assert_size_stride(primals_60, (256, ), (1, ), 'input')
            assert_size_stride(primals_61, (256, ), (1, ), 'input')
            assert_size_stride(primals_63, (256, ), (1, ), 'input')
            assert_size_stride(primals_64, (256, ), (1, ), 'input')
            assert_size_stride(primals_65, (256, ), (1, ), 'input')
            assert_size_stride(primals_66, (256, ), (1, ), 'input')
            buf44 = empty_strided_cuda((2, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
            buf45 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [out_32, input_4, out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17:43
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf45, buf42, primals_58, primals_59, primals_60, primals_61, buf43, primals_63, primals_64, primals_65, primals_66, 2048, stream=raw_stream0)
            del primals_61
            del primals_66
            # Topologically Sorted Source Nodes: [out_35], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:44
            buf46 = extern_kernels.convolution(buf45, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf46, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_68, (256, ), (1, ), 'input')
            assert_size_stride(primals_69, (256, ), (1, ), 'input')
            assert_size_stride(primals_70, (256, ), (1, ), 'input')
            assert_size_stride(primals_71, (256, ), (1, ), 'input')
            buf47 = empty_strided_cuda((2, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
            # Topologically Sorted Source Nodes: [out_36, out_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_16:45
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf46, primals_68, primals_69, primals_70, primals_71, buf47, 2048, stream=raw_stream0)
            del primals_71
            # Topologically Sorted Source Nodes: [out_38], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:46
            buf48 = extern_kernels.convolution(buf47, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf48, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_73, (256, ), (1, ), 'input')
            assert_size_stride(primals_74, (256, ), (1, ), 'input')
            assert_size_stride(primals_75, (256, ), (1, ), 'input')
            assert_size_stride(primals_76, (256, ), (1, ), 'input')
            buf49 = empty_strided_cuda((2, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
            # Topologically Sorted Source Nodes: [out_39, out_40, out_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18:47
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_18.run(buf48, primals_73, primals_74, primals_75, primals_76, buf45, buf49, 2048, stream=raw_stream0)
            del primals_76
            # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:48
            buf50 = extern_kernels.convolution(buf49, buf14, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf50, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_78, (512, ), (1, ), 'input')
            assert_size_stride(primals_79, (512, ), (1, ), 'input')
            assert_size_stride(primals_80, (512, ), (1, ), 'input')
            assert_size_stride(primals_81, (512, ), (1, ), 'input')
            buf51 = empty_strided_cuda((2, 512, 1, 1), (512, 1, 512, 512), torch.float32)
            # Topologically Sorted Source Nodes: [out_43, out_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_19:49
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf50, primals_78, primals_79, primals_80, primals_81, buf51, 1024, stream=raw_stream0)
            del primals_81
            # Topologically Sorted Source Nodes: [out_45], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:50
            buf52 = extern_kernels.convolution(buf51, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf52, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_87, (512, 256, 1, 1), (256, 1, 1, 1), 'input')
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:51
            buf53 = extern_kernels.convolution(buf49, reinterpret_tensor(primals_87, (512, 256, 1, 1), (256, 1, 256, 256), 0), stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf53, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_83, (512, ), (1, ), 'input')
            assert_size_stride(primals_84, (512, ), (1, ), 'input')
            assert_size_stride(primals_85, (512, ), (1, ), 'input')
            assert_size_stride(primals_86, (512, ), (1, ), 'input')
            assert_size_stride(primals_88, (512, ), (1, ), 'input')
            assert_size_stride(primals_89, (512, ), (1, ), 'input')
            assert_size_stride(primals_90, (512, ), (1, ), 'input')
            assert_size_stride(primals_91, (512, ), (1, ), 'input')
            buf54 = empty_strided_cuda((2, 512, 1, 1), (512, 1, 512, 512), torch.float32)
            buf55 = buf54; del buf54  # reuse
            # Topologically Sorted Source Nodes: [out_46, input_6, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20:52
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf55, buf52, primals_83, primals_84, primals_85, primals_86, buf53, primals_88, primals_89, primals_90, primals_91, 1024, stream=raw_stream0)
            del primals_86
            del primals_91
            # Topologically Sorted Source Nodes: [out_49], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:53
            buf56 = extern_kernels.convolution(buf55, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf56, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_93, (512, ), (1, ), 'input')
            assert_size_stride(primals_94, (512, ), (1, ), 'input')
            assert_size_stride(primals_95, (512, ), (1, ), 'input')
            assert_size_stride(primals_96, (512, ), (1, ), 'input')
            buf57 = empty_strided_cuda((2, 512, 1, 1), (512, 1, 512, 512), torch.float32)
            # Topologically Sorted Source Nodes: [out_50, out_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_19:54
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf56, primals_93, primals_94, primals_95, primals_96, buf57, 1024, stream=raw_stream0)
            del primals_96
            # Topologically Sorted Source Nodes: [out_52], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:55
            buf58 = extern_kernels.convolution(buf57, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf58, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
            assert_size_stride(primals_98, (512, ), (1, ), 'input')
            assert_size_stride(primals_99, (512, ), (1, ), 'input')
            assert_size_stride(primals_100, (512, ), (1, ), 'input')
            assert_size_stride(primals_101, (512, ), (1, ), 'input')
            buf60 = empty_strided_cuda((2, 512, 1, 1), (512, 1, 1024, 1024), torch.float32)
            buf62 = empty_strided_cuda((2, 512, 1, 1), (512, 1, 1, 1), torch.bool)
            # Topologically Sorted Source Nodes: [out_53, out_54, out_55, x_4, le], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean, aten.threshold_backward]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_21:56
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_21.run(buf58, primals_98, primals_99, primals_100, primals_101, buf55, buf60, buf62, 1024, stream=raw_stream0)
            del primals_101
            assert_size_stride(primals_103, (3, ), (1, ), 'input')
            assert_size_stride(primals_102, (3, 512), (512, 1), 'input')
            buf61 = empty_strided_cuda((2, 3), (3, 1), torch.float32)
            # Topologically Sorted Source Nodes: [x_4, x_5, x_6], Original ATen: [aten.mean, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:57
            extern_kernels.addmm(primals_103, reinterpret_tensor(buf60, (2, 512), (512, 1), 0), reinterpret_tensor(primals_102, (512, 3), (1, 512), 0), alpha=1, beta=1, out=buf61)
            del primals_103
        return (buf61, buf0, buf1, primals_3, primals_4, primals_5, primals_6, buf2, primals_8, primals_9, primals_10, buf3, primals_13, primals_14, primals_15, buf4, primals_18, primals_19, primals_20, buf5, primals_23, primals_24, primals_25, buf6, primals_28, primals_29, primals_30, buf7, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, buf8, primals_43, primals_44, primals_45, buf9, primals_48, primals_49, primals_50, buf10, primals_53, primals_54, primals_55, buf11, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, buf12, primals_68, primals_69, primals_70, buf13, primals_73, primals_74, primals_75, buf14, primals_78, primals_79, primals_80, buf15, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, buf16, primals_93, primals_94, primals_95, buf17, primals_98, primals_99, primals_100, primals_102, buf18, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf55, buf56, buf57, buf58, reinterpret_tensor(buf60, (2, 512), (512, 1), 0), buf62, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((2, 3, 32, 32), (3072, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((3, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
