# AOT ID: ['0_inference']
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


# kernel path: /tmp/torchinductor_bluemoon/r6/cr6dxftycx6cmwmligydzepgqoldwnqwkiqg2dqazrovoxs5qexf.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %arg1_1 : Tensor "f32[64, 3, 16, 16][768, 256, 16, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %convolution : Tensor "f32[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf0
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 393216, 'x': 196608}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask)
    tl.store(out_ptr0 + (y0 + 3*x2 + 768*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_bluemoon/2b/c2bvklmv2rwiwrh4mvfrl2ypheodool262qyyj32fmkji7g4xjyh.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %arg0_1 : Tensor "f32[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %convolution : Tensor "f32[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%arg1_1, %arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf1
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 75264, 'x': 37632}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_bluemoon/ao/caohsit6euqe4xxsz6v7hjynwl2wp6sjmjk24kazghc44melzdai.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x_2 => relu
# Graph fragment:
#   %convolution : Tensor "f32[64, 64, 8, 8][4096, 1, 512, 64]cuda:0" = PlaceHolder[target=convolution]
#   %arg2_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %arg4_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %arg5_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   return %relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 3146752}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_bluemoon/uz/cuzo6rklonybjp4sseu4mmfcll43lcgagmwxguz5qrxqybom53d3.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_1 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub, unsqueeze, unsqueeze_1, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6, unsqueeze_7
#   x_2 => relu
#   x_3 => _low_memory_max_pool_with_offsets
# Graph fragment:
#   %relu : Tensor "f32[64, 64, 8, 8][4096, 1, 512, 64]cuda:0" = PlaceHolder[target=relu]
#   %unsqueeze : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg2_1, -1), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, -1), kwargs = {})
#   %sub : Tensor "f32[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %add : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg3_1, 1e-05), kwargs = {})
#   %sqrt : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %mul_1 : Tensor "f32[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %mul_2 : Tensor "f32[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %unsqueeze_6 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg5_1, -1), kwargs = {})
#   %unsqueeze_7 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_6, -1), kwargs = {})
#   %add_1 : Tensor "f32[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : Tensor "f32[64, 64, 8, 8][4096, 64, 8, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
#   %_low_memory_max_pool_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool_with_offsets.default](args = (%relu, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   return %getitem
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 2883584}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = ((xindex // 256) % 4)
    x1 = ((xindex // 64) % 4)
    x0 = (xindex % 64)
    x5 = xindex // 256
    x6 = xindex
    tmp0 = ((-1) + 2*x2).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = ((-1) + 2*x1).to(tl.int32)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-576) + x0 + 128*x1 + 1024*x5), tmp10, other=float("-inf"))
    tmp12 = (2*x1).to(tl.int32)
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-512) + x0 + 128*x1 + 1024*x5), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp11, tmp17)
    tmp19 = (1 + 2*x1).to(tl.int32)
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-448) + x0 + 128*x1 + 1024*x5), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp18, tmp24)
    tmp26 = (2*x2).to(tl.int32)
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-64) + x0 + 128*x1 + 1024*x5), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 128*x1 + 1024*x5), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (64 + x0 + 128*x1 + 1024*x5), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp35, tmp37)
    tmp39 = (1 + 2*x2).to(tl.int32)
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (448 + x0 + 128*x1 + 1024*x5), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (512 + x0 + 128*x1 + 1024*x5), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (576 + x0 + 128*x1 + 1024*x5), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x6), tmp51, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_bluemoon/io/ciofgqqbcm74igpp433275ircqcmm3zedr26oaknr57rdmngw4na.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out => convolution_1
# Graph fragment:
#   %arg6_1 : Tensor "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %convolution_1 : Tensor "f32[64, 64, 4, 4][1024, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%getitem, %arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf5
triton_poi_fused_convolution_4 = async_compile.triton('triton_poi_fused_convolution_4', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 294912, 'x': 147456}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_bluemoon/fp/cfpa3ygsoyy3nkyzhszgmlgxrzlrof7skpb7blungelludppcu44.py
# Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_1 => add_2, add_3, mul_3, mul_4, mul_5, reciprocal_1, sqrt_1, sub_1, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_14, unsqueeze_15, unsqueeze_8, unsqueeze_9
#   out_2 => relu_1
# Graph fragment:
#   %convolution_1 : Tensor "f32[64, 64, 4, 4][1024, 1, 256, 64]cuda:0" = PlaceHolder[target=convolution_1]
#   %arg7_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %arg8_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg8_1]
#   %arg9_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %arg10_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg10_1]
#   %unsqueeze_8 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg7_1, -1), kwargs = {})
#   %unsqueeze_9 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, -1), kwargs = {})
#   %sub_1 : Tensor "f32[64, 64, 4, 4][1024, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %add_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg8_1, 1e-05), kwargs = {})
#   %sqrt_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2,), kwargs = {})
#   %reciprocal_1 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_1,), kwargs = {})
#   %mul_3 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_1, 1), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_3, -1), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, -1), kwargs = {})
#   %mul_4 : Tensor "f32[64, 64, 4, 4][1024, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %unsqueeze_12 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg9_1, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %mul_5 : Tensor "f32[64, 64, 4, 4][1024, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %unsqueeze_14 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg10_1, -1), kwargs = {})
#   %unsqueeze_15 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_14, -1), kwargs = {})
#   %add_3 : Tensor "f32[64, 64, 4, 4][1024, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : Tensor "f32[64, 64, 4, 4][1024, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
#   return %relu_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 787456}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_bluemoon/5c/c5chfcjsh4rwla7hsylxby6tsdwwmqhuyzwc7lveuoxhw6ggjwxb.py
# Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_4 => add_4, add_5, mul_6, mul_7, mul_8, reciprocal_2, sqrt_2, sub_2, unsqueeze_16, unsqueeze_17, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23
#   out_5 => add_6
#   out_6 => relu_2
# Graph fragment:
#   %convolution_2 : Tensor "f32[64, 64, 4, 4][1024, 1, 256, 64]cuda:0" = PlaceHolder[target=convolution_2]
#   %arg12_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg12_1]
#   %arg13_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %arg14_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg14_1]
#   %arg15_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg15_1]
#   %getitem : Tensor "f32[64, 64, 4, 4][1024, 1, 256, 64]cuda:0" = PlaceHolder[target=getitem]
#   %unsqueeze_16 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg12_1, -1), kwargs = {})
#   %unsqueeze_17 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_16, -1), kwargs = {})
#   %sub_2 : Tensor "f32[64, 64, 4, 4][1024, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %add_4 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg13_1, 1e-05), kwargs = {})
#   %sqrt_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_4,), kwargs = {})
#   %reciprocal_2 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_2,), kwargs = {})
#   %mul_6 : Tensor "f32[64][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_2, 1), kwargs = {})
#   %unsqueeze_18 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_6, -1), kwargs = {})
#   %unsqueeze_19 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_18, -1), kwargs = {})
#   %mul_7 : Tensor "f32[64, 64, 4, 4][1024, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %unsqueeze_20 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg14_1, -1), kwargs = {})
#   %unsqueeze_21 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_20, -1), kwargs = {})
#   %mul_8 : Tensor "f32[64, 64, 4, 4][1024, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %unsqueeze_22 : Tensor "f32[64, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg15_1, -1), kwargs = {})
#   %unsqueeze_23 : Tensor "f32[64, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_22, -1), kwargs = {})
#   %add_5 : Tensor "f32[64, 64, 4, 4][1024, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %add_6 : Tensor "f32[64, 64, 4, 4][1024, 16, 4, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %getitem), kwargs = {})
#   %relu_2 : Tensor "f32[64, 64, 4, 4][1024, 16, 4, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_6,), kwargs = {})
#   return %relu_2
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 1049600}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_bluemoon/m5/cm5lpl5uxa5n5oej5y4n6mpj4jnzrshfg4djxygn7rapkfgajzyl.py
# Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_14 => convolution_5
# Graph fragment:
#   %arg26_1 : Tensor "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = PlaceHolder[target=arg26_1]
#   %convolution_5 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_4, %arg26_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf17
triton_poi_fused_convolution_7 = async_compile.triton('triton_poi_fused_convolution_7', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 589824, 'x': 294912}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_bluemoon/nh/cnhqnprtpfuhclxvo2a63rmza4xh2dohewao6iaprkebl45x77xb.py
# Topologically Sorted Source Nodes: [out_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_15 => add_12, add_13, mul_15, mul_16, mul_17, reciprocal_5, sqrt_5, sub_5, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, unsqueeze_44, unsqueeze_45, unsqueeze_46, unsqueeze_47
#   out_16 => relu_5
# Graph fragment:
#   %convolution_5 : Tensor "f32[64, 128, 2, 2][512, 1, 256, 128]cuda:0" = PlaceHolder[target=convolution_5]
#   %arg27_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg27_1]
#   %arg28_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg28_1]
#   %arg29_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg29_1]
#   %arg30_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %unsqueeze_40 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg27_1, -1), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, -1), kwargs = {})
#   %sub_5 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %add_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg28_1, 1e-05), kwargs = {})
#   %sqrt_5 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_12,), kwargs = {})
#   %reciprocal_5 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_5,), kwargs = {})
#   %mul_15 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 1), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_15, -1), kwargs = {})
#   %unsqueeze_43 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_42, -1), kwargs = {})
#   %mul_16 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %unsqueeze_44 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg29_1, -1), kwargs = {})
#   %unsqueeze_45 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_44, -1), kwargs = {})
#   %mul_17 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %unsqueeze_46 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg30_1, -1), kwargs = {})
#   %unsqueeze_47 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_46, -1), kwargs = {})
#   %add_13 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_5 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
#   return %relu_5
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 395264}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_bluemoon/kn/cknkuscbyilobfd4omwhasa35f7e6zdxxoo6cfnauui646ges2s6.py
# Topologically Sorted Source Nodes: [out_15, out_16, out_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_15 => add_12, add_13, mul_15, mul_16, mul_17, reciprocal_5, sqrt_5, sub_5, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, unsqueeze_44, unsqueeze_45, unsqueeze_46, unsqueeze_47
#   out_16 => relu_5
#   out_17 => convolution_6
# Graph fragment:
#   %arg31_1 : Tensor "f32[128, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg31_1]
#   %unsqueeze_40 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg27_1, -1), kwargs = {})
#   %unsqueeze_41 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_40, -1), kwargs = {})
#   %sub_5 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %add_12 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg28_1, 1e-05), kwargs = {})
#   %sqrt_5 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_12,), kwargs = {})
#   %reciprocal_5 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_5,), kwargs = {})
#   %mul_15 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_5, 1), kwargs = {})
#   %unsqueeze_42 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_15, -1), kwargs = {})
#   %unsqueeze_43 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_42, -1), kwargs = {})
#   %mul_16 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %unsqueeze_44 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg29_1, -1), kwargs = {})
#   %unsqueeze_45 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_44, -1), kwargs = {})
#   %mul_17 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %unsqueeze_46 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg30_1, -1), kwargs = {})
#   %unsqueeze_47 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_46, -1), kwargs = {})
#   %add_13 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_5 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
#   %convolution_6 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %arg31_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf20
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 1179648, 'x': 589824}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_bluemoon/7j/c7jf5ov2dsbgkddibrjqaaqpz65xkhkpyuxlovznqxibe5g26i37.py
# Topologically Sorted Source Nodes: [out_18, input_2, out_19, out_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_16, add_17, mul_21, mul_22, mul_23, reciprocal_7, sqrt_7, sub_7, unsqueeze_56, unsqueeze_57, unsqueeze_58, unsqueeze_59, unsqueeze_60, unsqueeze_61, unsqueeze_62, unsqueeze_63
#   out_18 => add_14, add_15, mul_18, mul_19, mul_20, reciprocal_6, sqrt_6, sub_6, unsqueeze_48, unsqueeze_49, unsqueeze_50, unsqueeze_51, unsqueeze_52, unsqueeze_53, unsqueeze_54, unsqueeze_55
#   out_19 => add_18
#   out_20 => relu_6
# Graph fragment:
#   %convolution_6 : Tensor "f32[64, 128, 2, 2][512, 1, 256, 128]cuda:0" = PlaceHolder[target=convolution_6]
#   %arg32_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg32_1]
#   %arg33_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg33_1]
#   %arg34_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg34_1]
#   %arg35_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg35_1]
#   %convolution_7 : Tensor "f32[64, 128, 2, 2][512, 1, 256, 128]cuda:0" = PlaceHolder[target=convolution_7]
#   %arg37_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg37_1]
#   %arg38_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg38_1]
#   %arg39_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %arg40_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg40_1]
#   %add_18 : Tensor "f32[64, 128, 2, 2][512, 1, 256, 128]cuda:0" = PlaceHolder[target=add_18]
#   %unsqueeze_48 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg32_1, -1), kwargs = {})
#   %unsqueeze_49 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_48, -1), kwargs = {})
#   %sub_6 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %add_14 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg33_1, 1e-05), kwargs = {})
#   %sqrt_6 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_14,), kwargs = {})
#   %reciprocal_6 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_6,), kwargs = {})
#   %mul_18 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_6, 1), kwargs = {})
#   %unsqueeze_50 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_18, -1), kwargs = {})
#   %unsqueeze_51 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_50, -1), kwargs = {})
#   %mul_19 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %unsqueeze_52 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg34_1, -1), kwargs = {})
#   %unsqueeze_53 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_52, -1), kwargs = {})
#   %mul_20 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %unsqueeze_54 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg35_1, -1), kwargs = {})
#   %unsqueeze_55 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_54, -1), kwargs = {})
#   %add_15 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %unsqueeze_56 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg37_1, -1), kwargs = {})
#   %unsqueeze_57 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_56, -1), kwargs = {})
#   %sub_7 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %add_16 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg38_1, 1e-05), kwargs = {})
#   %sqrt_7 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_16,), kwargs = {})
#   %reciprocal_7 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_7,), kwargs = {})
#   %mul_21 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_7, 1), kwargs = {})
#   %unsqueeze_58 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_21, -1), kwargs = {})
#   %unsqueeze_59 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_58, -1), kwargs = {})
#   %mul_22 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %unsqueeze_60 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg39_1, -1), kwargs = {})
#   %unsqueeze_61 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_60, -1), kwargs = {})
#   %mul_23 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %unsqueeze_62 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg40_1, -1), kwargs = {})
#   %unsqueeze_63 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_62, -1), kwargs = {})
#   %add_17 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %add_18 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %add_17), kwargs = {})
#   %relu_6 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_18,), kwargs = {})
#   return %add_18,%relu_6
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 528384}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2), None)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_bluemoon/qq/cqqokdm7vgvkgcilydhshri66aobmveecw55w5z4ewlsg3ango5f.py
# Topologically Sorted Source Nodes: [out_25, out_26, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_25 => add_21, add_22, mul_27, mul_28, mul_29, reciprocal_9, sqrt_9, sub_9, unsqueeze_72, unsqueeze_73, unsqueeze_74, unsqueeze_75, unsqueeze_76, unsqueeze_77, unsqueeze_78, unsqueeze_79
#   out_26 => add_23
#   out_27 => relu_8
# Graph fragment:
#   %convolution_9 : Tensor "f32[64, 128, 2, 2][512, 1, 256, 128]cuda:0" = PlaceHolder[target=convolution_9]
#   %arg47_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg47_1]
#   %arg48_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg48_1]
#   %arg49_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg49_1]
#   %arg50_1 : Tensor "f32[128][1]cuda:0" = PlaceHolder[target=arg50_1]
#   %relu_6 : Tensor "f32[64, 128, 2, 2][512, 1, 256, 128]cuda:0" = PlaceHolder[target=relu_6]
#   %unsqueeze_72 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg47_1, -1), kwargs = {})
#   %unsqueeze_73 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_72, -1), kwargs = {})
#   %sub_9 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %add_21 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg48_1, 1e-05), kwargs = {})
#   %sqrt_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_21,), kwargs = {})
#   %reciprocal_9 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_9,), kwargs = {})
#   %mul_27 : Tensor "f32[128][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_9, 1), kwargs = {})
#   %unsqueeze_74 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_27, -1), kwargs = {})
#   %unsqueeze_75 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_74, -1), kwargs = {})
#   %mul_28 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %unsqueeze_76 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg49_1, -1), kwargs = {})
#   %unsqueeze_77 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_76, -1), kwargs = {})
#   %mul_29 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %unsqueeze_78 : Tensor "f32[128, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg50_1, -1), kwargs = {})
#   %unsqueeze_79 : Tensor "f32[128, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_78, -1), kwargs = {})
#   %add_22 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %add_23 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %relu_6), kwargs = {})
#   %relu_8 : Tensor "f32[64, 128, 2, 2][512, 4, 2, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_23,), kwargs = {})
#   return %relu_8
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 526336}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_bluemoon/ih/cihtlqtv5hnu23atnptgvnc3mp5tvwagsmslkzbv3l5oodj3dw5u.py
# Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_28 => convolution_10
# Graph fragment:
#   %arg51_1 : Tensor "f32[256, 128, 3, 3][1152, 9, 3, 1]cuda:0" = PlaceHolder[target=arg51_1]
#   %convolution_10 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %arg51_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf31
triton_poi_fused_convolution_12 = async_compile.triton('triton_poi_fused_convolution_12', '''
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
    inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 2359296, 'x': 1179648}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_bluemoon/r3/cr3qpso2swqisrbrxfehqdzruxhdvtusddmxqlknajwic7bzh2au.py
# Topologically Sorted Source Nodes: [out_29, out_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_29 => add_24, add_25, mul_30, mul_31, mul_32, reciprocal_10, sqrt_10, sub_10, unsqueeze_80, unsqueeze_81, unsqueeze_82, unsqueeze_83, unsqueeze_84, unsqueeze_85, unsqueeze_86, unsqueeze_87
#   out_30 => relu_9
# Graph fragment:
#   %convolution_10 : Tensor "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=convolution_10]
#   %arg52_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg52_1]
#   %arg53_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg53_1]
#   %arg54_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg54_1]
#   %arg55_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg55_1]
#   %unsqueeze_80 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg52_1, -1), kwargs = {})
#   %unsqueeze_81 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_80, -1), kwargs = {})
#   %sub_10 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %add_24 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg53_1, 1e-05), kwargs = {})
#   %sqrt_10 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_24,), kwargs = {})
#   %reciprocal_10 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_10,), kwargs = {})
#   %mul_30 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_10, 1), kwargs = {})
#   %unsqueeze_82 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_30, -1), kwargs = {})
#   %unsqueeze_83 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_82, -1), kwargs = {})
#   %mul_31 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %unsqueeze_84 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg54_1, -1), kwargs = {})
#   %unsqueeze_85 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_84, -1), kwargs = {})
#   %mul_32 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %unsqueeze_86 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg55_1, -1), kwargs = {})
#   %unsqueeze_87 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_86, -1), kwargs = {})
#   %add_25 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %relu_9 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_25,), kwargs = {})
#   return %relu_9
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 200704}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_bluemoon/cz/cczal5axssv66oo3ylbc5xdwgsropxlipt33nebeacg4ypeqxqid.py
# Topologically Sorted Source Nodes: [out_29, out_30, out_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_29 => add_24, add_25, mul_30, mul_31, mul_32, reciprocal_10, sqrt_10, sub_10, unsqueeze_80, unsqueeze_81, unsqueeze_82, unsqueeze_83, unsqueeze_84, unsqueeze_85, unsqueeze_86, unsqueeze_87
#   out_30 => relu_9
#   out_31 => convolution_11
# Graph fragment:
#   %arg56_1 : Tensor "f32[256, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg56_1]
#   %unsqueeze_80 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg52_1, -1), kwargs = {})
#   %unsqueeze_81 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_80, -1), kwargs = {})
#   %sub_10 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %add_24 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg53_1, 1e-05), kwargs = {})
#   %sqrt_10 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_24,), kwargs = {})
#   %reciprocal_10 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_10,), kwargs = {})
#   %mul_30 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_10, 1), kwargs = {})
#   %unsqueeze_82 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_30, -1), kwargs = {})
#   %unsqueeze_83 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_82, -1), kwargs = {})
#   %mul_31 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %unsqueeze_84 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg54_1, -1), kwargs = {})
#   %unsqueeze_85 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_84, -1), kwargs = {})
#   %mul_32 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %unsqueeze_86 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg55_1, -1), kwargs = {})
#   %unsqueeze_87 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_86, -1), kwargs = {})
#   %add_25 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %relu_9 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_25,), kwargs = {})
#   %convolution_11 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %arg56_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf34
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 4718592, 'x': 2359296}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_bluemoon/gd/cgdc3j3ufjovuoqslccmmcmnb5vzq2s4derf3aipbvfebn3wzjb7.py
# Topologically Sorted Source Nodes: [out_32, input_4, out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_4 => add_28, add_29, mul_36, mul_37, mul_38, reciprocal_12, sqrt_12, sub_12, unsqueeze_100, unsqueeze_101, unsqueeze_102, unsqueeze_103, unsqueeze_96, unsqueeze_97, unsqueeze_98, unsqueeze_99
#   out_32 => add_26, add_27, mul_33, mul_34, mul_35, reciprocal_11, sqrt_11, sub_11, unsqueeze_88, unsqueeze_89, unsqueeze_90, unsqueeze_91, unsqueeze_92, unsqueeze_93, unsqueeze_94, unsqueeze_95
#   out_33 => add_30
#   out_34 => relu_10
# Graph fragment:
#   %convolution_11 : Tensor "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=convolution_11]
#   %arg57_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg57_1]
#   %arg58_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg58_1]
#   %arg59_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg59_1]
#   %arg60_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg60_1]
#   %convolution_12 : Tensor "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=convolution_12]
#   %arg62_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg62_1]
#   %arg63_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg63_1]
#   %arg64_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg64_1]
#   %arg65_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg65_1]
#   %add_30 : Tensor "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=add_30]
#   %unsqueeze_88 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg57_1, -1), kwargs = {})
#   %unsqueeze_89 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_88, -1), kwargs = {})
#   %sub_11 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %add_26 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg58_1, 1e-05), kwargs = {})
#   %sqrt_11 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_26,), kwargs = {})
#   %reciprocal_11 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_11,), kwargs = {})
#   %mul_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_11, 1), kwargs = {})
#   %unsqueeze_90 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_33, -1), kwargs = {})
#   %unsqueeze_91 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_90, -1), kwargs = {})
#   %mul_34 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %unsqueeze_92 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg59_1, -1), kwargs = {})
#   %unsqueeze_93 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_92, -1), kwargs = {})
#   %mul_35 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %unsqueeze_94 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg60_1, -1), kwargs = {})
#   %unsqueeze_95 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_94, -1), kwargs = {})
#   %add_27 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %unsqueeze_96 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg62_1, -1), kwargs = {})
#   %unsqueeze_97 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_96, -1), kwargs = {})
#   %sub_12 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %add_28 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg63_1, 1e-05), kwargs = {})
#   %sqrt_12 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_28,), kwargs = {})
#   %reciprocal_12 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_12,), kwargs = {})
#   %mul_36 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_12, 1), kwargs = {})
#   %unsqueeze_98 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_36, -1), kwargs = {})
#   %unsqueeze_99 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_98, -1), kwargs = {})
#   %mul_37 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %unsqueeze_100 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg64_1, -1), kwargs = {})
#   %unsqueeze_101 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_100, -1), kwargs = {})
#   %mul_38 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %unsqueeze_102 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg65_1, -1), kwargs = {})
#   %unsqueeze_103 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_102, -1), kwargs = {})
#   %add_29 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %add_30 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_27, %add_29), kwargs = {})
#   %relu_10 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_30,), kwargs = {})
#   return %add_30,%relu_10
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 270336}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2), None)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_bluemoon/cv/ccvgptxhsjr7b4g3q4kbpnpeigohu3gpdlcf2yrkbovqsrctuyhn.py
# Topologically Sorted Source Nodes: [out_39, out_40, out_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   out_39 => add_33, add_34, mul_42, mul_43, mul_44, reciprocal_14, sqrt_14, sub_14, unsqueeze_112, unsqueeze_113, unsqueeze_114, unsqueeze_115, unsqueeze_116, unsqueeze_117, unsqueeze_118, unsqueeze_119
#   out_40 => add_35
#   out_41 => relu_12
# Graph fragment:
#   %convolution_14 : Tensor "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = PlaceHolder[target=convolution_14]
#   %arg72_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg72_1]
#   %arg73_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg73_1]
#   %arg74_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg74_1]
#   %arg75_1 : Tensor "f32[256][1]cuda:0" = PlaceHolder[target=arg75_1]
#   %relu_10 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0" = PlaceHolder[target=relu_10]
#   %unsqueeze_112 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg72_1, -1), kwargs = {})
#   %unsqueeze_113 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_112, -1), kwargs = {})
#   %sub_14 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %add_33 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg73_1, 1e-05), kwargs = {})
#   %sqrt_14 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_33,), kwargs = {})
#   %reciprocal_14 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_14,), kwargs = {})
#   %mul_42 : Tensor "f32[256][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_14, 1), kwargs = {})
#   %unsqueeze_114 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_42, -1), kwargs = {})
#   %unsqueeze_115 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_114, -1), kwargs = {})
#   %mul_43 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %unsqueeze_116 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg74_1, -1), kwargs = {})
#   %unsqueeze_117 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_116, -1), kwargs = {})
#   %mul_44 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %unsqueeze_118 : Tensor "f32[256, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg75_1, -1), kwargs = {})
#   %unsqueeze_119 : Tensor "f32[256, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_118, -1), kwargs = {})
#   %add_34 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %add_35 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %relu_10), kwargs = {})
#   %relu_12 : Tensor "f32[64, 256, 1, 1][256, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_35,), kwargs = {})
#   return %relu_12
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 266240}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp18, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_bluemoon/ah/cahnmzd4p2xhbt3rhesxtc5jjpk354z2ptj4hdi4s3bzgyopnqcz.py
# Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_42 => convolution_15
# Graph fragment:
#   %arg76_1 : Tensor "f32[512, 256, 3, 3][2304, 9, 3, 1]cuda:0" = PlaceHolder[target=arg76_1]
#   %convolution_15 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %arg76_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf45
triton_poi_fused_convolution_17 = async_compile.triton('triton_poi_fused_convolution_17', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 9437184, 'x': 4718592}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_bluemoon/kv/ckvyqldept67wyyozwg6ij6rtfq5wqom3x632k2cgfrv23tksesi.py
# Topologically Sorted Source Nodes: [out_43, out_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out_43 => add_36, add_37, mul_45, mul_46, mul_47, reciprocal_15, sqrt_15, sub_15, unsqueeze_120, unsqueeze_121, unsqueeze_122, unsqueeze_123, unsqueeze_124, unsqueeze_125, unsqueeze_126, unsqueeze_127
#   out_44 => relu_13
# Graph fragment:
#   %convolution_15 : Tensor "f32[64, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=convolution_15]
#   %arg77_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg77_1]
#   %arg78_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg78_1]
#   %arg79_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg79_1]
#   %arg80_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg80_1]
#   %unsqueeze_120 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg77_1, -1), kwargs = {})
#   %unsqueeze_121 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_120, -1), kwargs = {})
#   %sub_15 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_121), kwargs = {})
#   %add_36 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg78_1, 1e-05), kwargs = {})
#   %sqrt_15 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_36,), kwargs = {})
#   %reciprocal_15 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_15,), kwargs = {})
#   %mul_45 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_15, 1), kwargs = {})
#   %unsqueeze_122 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_45, -1), kwargs = {})
#   %unsqueeze_123 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_122, -1), kwargs = {})
#   %mul_46 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %unsqueeze_124 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg79_1, -1), kwargs = {})
#   %unsqueeze_125 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_124, -1), kwargs = {})
#   %mul_47 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %unsqueeze_126 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg80_1, -1), kwargs = {})
#   %unsqueeze_127 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_126, -1), kwargs = {})
#   %add_37 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %relu_13 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_37,), kwargs = {})
#   return %relu_13
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 401408}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_bluemoon/pt/cpteeraz2vng2xboopxdi2reheaw2f4xsksxb73shpauwvuwpcha.py
# Topologically Sorted Source Nodes: [out_43, out_44, out_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
# Source node to ATen node mapping:
#   out_43 => add_36, add_37, mul_45, mul_46, mul_47, reciprocal_15, sqrt_15, sub_15, unsqueeze_120, unsqueeze_121, unsqueeze_122, unsqueeze_123, unsqueeze_124, unsqueeze_125, unsqueeze_126, unsqueeze_127
#   out_44 => relu_13
#   out_45 => convolution_16
# Graph fragment:
#   %arg81_1 : Tensor "f32[512, 512, 3, 3][4608, 9, 3, 1]cuda:0" = PlaceHolder[target=arg81_1]
#   %unsqueeze_120 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg77_1, -1), kwargs = {})
#   %unsqueeze_121 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_120, -1), kwargs = {})
#   %sub_15 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_121), kwargs = {})
#   %add_36 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg78_1, 1e-05), kwargs = {})
#   %sqrt_15 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_36,), kwargs = {})
#   %reciprocal_15 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_15,), kwargs = {})
#   %mul_45 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_15, 1), kwargs = {})
#   %unsqueeze_122 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_45, -1), kwargs = {})
#   %unsqueeze_123 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_122, -1), kwargs = {})
#   %mul_46 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %unsqueeze_124 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg79_1, -1), kwargs = {})
#   %unsqueeze_125 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_124, -1), kwargs = {})
#   %mul_47 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %unsqueeze_126 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg80_1, -1), kwargs = {})
#   %unsqueeze_127 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_126, -1), kwargs = {})
#   %add_37 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %relu_13 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_37,), kwargs = {})
#   %convolution_16 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_13, %arg81_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   return %buf48
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19', '''
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
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'y': 18874368, 'x': 9437184}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_bluemoon/fu/cfuhgg3xjsvcbzakslprjj4k76xfpsncuoxo2qtyh6dzfp6uppgp.py
# Topologically Sorted Source Nodes: [out_46, input_6, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_6 => add_40, add_41, mul_51, mul_52, mul_53, reciprocal_17, sqrt_17, sub_17, unsqueeze_136, unsqueeze_137, unsqueeze_138, unsqueeze_139, unsqueeze_140, unsqueeze_141, unsqueeze_142, unsqueeze_143
#   out_46 => add_38, add_39, mul_48, mul_49, mul_50, reciprocal_16, sqrt_16, sub_16, unsqueeze_128, unsqueeze_129, unsqueeze_130, unsqueeze_131, unsqueeze_132, unsqueeze_133, unsqueeze_134, unsqueeze_135
#   out_47 => add_42
#   out_48 => relu_14
# Graph fragment:
#   %convolution_16 : Tensor "f32[64, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=convolution_16]
#   %arg82_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg82_1]
#   %arg83_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg83_1]
#   %arg84_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg84_1]
#   %arg85_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg85_1]
#   %convolution_17 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=convolution_17]
#   %arg87_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg87_1]
#   %arg88_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg88_1]
#   %arg89_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg89_1]
#   %arg90_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg90_1]
#   %add_42 : Tensor "f32[64, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=add_42]
#   %unsqueeze_128 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg82_1, -1), kwargs = {})
#   %unsqueeze_129 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_128, -1), kwargs = {})
#   %sub_16 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_129), kwargs = {})
#   %add_38 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg83_1, 1e-05), kwargs = {})
#   %sqrt_16 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_38,), kwargs = {})
#   %reciprocal_16 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_16,), kwargs = {})
#   %mul_48 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_16, 1), kwargs = {})
#   %unsqueeze_130 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_48, -1), kwargs = {})
#   %unsqueeze_131 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_130, -1), kwargs = {})
#   %mul_49 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_131), kwargs = {})
#   %unsqueeze_132 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg84_1, -1), kwargs = {})
#   %unsqueeze_133 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_132, -1), kwargs = {})
#   %mul_50 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %unsqueeze_134 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg85_1, -1), kwargs = {})
#   %unsqueeze_135 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_134, -1), kwargs = {})
#   %add_39 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %unsqueeze_136 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg87_1, -1), kwargs = {})
#   %unsqueeze_137 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_136, -1), kwargs = {})
#   %sub_17 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %add_40 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg88_1, 1e-05), kwargs = {})
#   %sqrt_17 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_40,), kwargs = {})
#   %reciprocal_17 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_17,), kwargs = {})
#   %mul_51 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_17, 1), kwargs = {})
#   %unsqueeze_138 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_51, -1), kwargs = {})
#   %unsqueeze_139 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_138, -1), kwargs = {})
#   %mul_52 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %unsqueeze_140 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg89_1, -1), kwargs = {})
#   %unsqueeze_141 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_140, -1), kwargs = {})
#   %mul_53 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %unsqueeze_142 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg90_1, -1), kwargs = {})
#   %unsqueeze_143 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_142, -1), kwargs = {})
#   %add_41 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %add_42 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %add_41), kwargs = {})
#   %relu_14 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_42,), kwargs = {})
#   return %add_42,%relu_14
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 540672}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2), None)
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_bluemoon/zk/czk2hn6siguue2kuw5gb7hw6fgzckgunkgrsvp3l225un5ahgvvv.py
# Topologically Sorted Source Nodes: [out_53, out_54, out_55, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   out_53 => add_45, add_46, mul_57, mul_58, mul_59, reciprocal_19, sqrt_19, sub_19, unsqueeze_152, unsqueeze_153, unsqueeze_154, unsqueeze_155, unsqueeze_156, unsqueeze_157, unsqueeze_158, unsqueeze_159
#   out_54 => add_47
#   out_55 => relu_16
#   x_4 => mean
# Graph fragment:
#   %convolution_19 : Tensor "f32[64, 512, 1, 1][512, 1, 512, 512]cuda:0" = PlaceHolder[target=convolution_19]
#   %arg97_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg97_1]
#   %arg98_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg98_1]
#   %arg99_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg99_1]
#   %arg100_1 : Tensor "f32[512][1]cuda:0" = PlaceHolder[target=arg100_1]
#   %relu_14 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0" = PlaceHolder[target=relu_14]
#   %unsqueeze_152 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg97_1, -1), kwargs = {})
#   %unsqueeze_153 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_152, -1), kwargs = {})
#   %sub_19 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_153), kwargs = {})
#   %add_45 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg98_1, 1e-05), kwargs = {})
#   %sqrt_19 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_45,), kwargs = {})
#   %reciprocal_19 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt_19,), kwargs = {})
#   %mul_57 : Tensor "f32[512][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal_19, 1), kwargs = {})
#   %unsqueeze_154 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_57, -1), kwargs = {})
#   %unsqueeze_155 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_154, -1), kwargs = {})
#   %mul_58 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_155), kwargs = {})
#   %unsqueeze_156 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg99_1, -1), kwargs = {})
#   %unsqueeze_157 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_156, -1), kwargs = {})
#   %mul_59 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_157), kwargs = {})
#   %unsqueeze_158 : Tensor "f32[512, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg100_1, -1), kwargs = {})
#   %unsqueeze_159 : Tensor "f32[512, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_158, -1), kwargs = {})
#   %add_46 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_159), kwargs = {})
#   %add_47 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_46, %relu_14), kwargs = {})
#   %relu_16 : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_47,), kwargs = {})
#   %mean : Tensor "f32[64, 512, 1, 1][512, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_16, [-1, -2], True), kwargs = {})
#   return %mean
triton_poi_fused__native_batch_norm_legit_no_training_add_mean_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mean_relu_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=82, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mean_relu_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'tiling_scores': {'x': 532480}, 'backend_hash': 'ACA358AF06E1552031ECD536BFAFA6061E8E904924A766FE2FB7C46D9E43572A', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mean_relu_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x2), None)
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
    tl.store(in_out_ptr0 + (x2), tmp19, None)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1 = args
        args.clear()
        assert_size_stride(arg1_1, (64, 3, 16, 16), (768, 256, 16, 1), 'input')
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg1_1 = copy_if_misaligned(arg1_1)
            buf0 = empty_strided_cuda((64, 3, 16, 16), (768, 1, 48, 3), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_0:1
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_0.run(arg1_1, buf0, 192, 256, stream=raw_stream0)
            del arg1_1
            assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1), 'input')
            buf1 = empty_strided_cuda((64, 3, 7, 7), (147, 1, 21, 3), torch.float32)
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_1:2
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 49, stream=raw_stream0)
            del arg0_1
            # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:3
            buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf2, (64, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution.default')
            del buf0
            del buf1
            assert_size_stride(arg2_1, (64, ), (1, ), 'input')
            assert_size_stride(arg3_1, (64, ), (1, ), 'input')
            assert_size_stride(arg4_1, (64, ), (1, ), 'input')
            assert_size_stride(arg5_1, (64, ), (1, ), 'input')
            buf3 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_2:4
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg2_1, arg3_1, arg4_1, arg5_1, 262144, stream=raw_stream0)
            del arg2_1
            del arg3_1
            del arg4_1
            del arg5_1
            buf4 = empty_strided_cuda((64, 64, 4, 4), (1024, 1, 256, 64), torch.float32)
            # Topologically Sorted Source Nodes: [x_1, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.max_pool2d_with_indices]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3:5
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.run(buf3, buf4, 65536, stream=raw_stream0)
            del buf3
            assert_size_stride(arg6_1, (64, 64, 3, 3), (576, 9, 3, 1), 'input')
            buf5 = empty_strided_cuda((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_4:6
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_4.run(arg6_1, buf5, 4096, 9, stream=raw_stream0)
            del arg6_1
            # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:7
            buf6 = extern_kernels.convolution(buf4, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf6, (64, 64, 4, 4), (1024, 1, 256, 64), 'torch.ops.aten.convolution.default')
            assert_size_stride(arg7_1, (64, ), (1, ), 'input')
            assert_size_stride(arg8_1, (64, ), (1, ), 'input')
            assert_size_stride(arg9_1, (64, ), (1, ), 'input')
            assert_size_stride(arg10_1, (64, ), (1, ), 'input')
            buf7 = buf6; del buf6  # reuse
            # Topologically Sorted Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_5:8
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf7, arg7_1, arg8_1, arg9_1, arg10_1, 65536, stream=raw_stream0)
            del arg10_1
            del arg7_1
            del arg8_1
            del arg9_1
            assert_size_stride(arg11_1, (64, 64, 3, 3), (576, 9, 3, 1), 'input')
            buf8 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [out_1, out_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_4:9
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_4.run(arg11_1, buf8, 4096, 9, stream=raw_stream0)
            del arg11_1
            # Topologically Sorted Source Nodes: [out_1, out_2, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:10
            buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf9, (64, 64, 4, 4), (1024, 1, 256, 64), 'torch.ops.aten.convolution.default')
            del buf7
            assert_size_stride(arg12_1, (64, ), (1, ), 'input')
            assert_size_stride(arg13_1, (64, ), (1, ), 'input')
            assert_size_stride(arg14_1, (64, ), (1, ), 'input')
            assert_size_stride(arg15_1, (64, ), (1, ), 'input')
            buf10 = buf9; del buf9  # reuse
            # Topologically Sorted Source Nodes: [out_4, out_5, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6:11
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf10, arg12_1, arg13_1, arg14_1, arg15_1, buf4, 65536, stream=raw_stream0)
            del arg12_1
            del arg13_1
            del arg14_1
            del arg15_1
            del buf4
            assert_size_stride(arg16_1, (64, 64, 3, 3), (576, 9, 3, 1), 'input')
            buf11 = buf8; del buf8  # reuse
            # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_4:12
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_4.run(arg16_1, buf11, 4096, 9, stream=raw_stream0)
            del arg16_1
            # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:13
            buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf12, (64, 64, 4, 4), (1024, 1, 256, 64), 'torch.ops.aten.convolution.default')
            assert_size_stride(arg17_1, (64, ), (1, ), 'input')
            assert_size_stride(arg18_1, (64, ), (1, ), 'input')
            assert_size_stride(arg19_1, (64, ), (1, ), 'input')
            assert_size_stride(arg20_1, (64, ), (1, ), 'input')
            buf13 = buf12; del buf12  # reuse
            # Topologically Sorted Source Nodes: [out_8, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_5:14
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf13, arg17_1, arg18_1, arg19_1, arg20_1, 65536, stream=raw_stream0)
            del arg17_1
            del arg18_1
            del arg19_1
            del arg20_1
            assert_size_stride(arg21_1, (64, 64, 3, 3), (576, 9, 3, 1), 'input')
            buf14 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [out_8, out_9, out_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_4:15
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_4.run(arg21_1, buf14, 4096, 9, stream=raw_stream0)
            del arg21_1
            # Topologically Sorted Source Nodes: [out_8, out_9, out_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:16
            buf15 = extern_kernels.convolution(buf13, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf15, (64, 64, 4, 4), (1024, 1, 256, 64), 'torch.ops.aten.convolution.default')
            del buf13
            del buf14
            assert_size_stride(arg22_1, (64, ), (1, ), 'input')
            assert_size_stride(arg23_1, (64, ), (1, ), 'input')
            assert_size_stride(arg24_1, (64, ), (1, ), 'input')
            assert_size_stride(arg25_1, (64, ), (1, ), 'input')
            buf16 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [out_11, out_12, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6:17
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_6.run(buf16, arg22_1, arg23_1, arg24_1, arg25_1, buf10, 65536, stream=raw_stream0)
            del arg22_1
            del arg23_1
            del arg24_1
            del arg25_1
            del buf10
            assert_size_stride(arg26_1, (128, 64, 3, 3), (576, 9, 3, 1), 'input')
            buf17 = empty_strided_cuda((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
            # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_7:18
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_7.run(arg26_1, buf17, 8192, 9, stream=raw_stream0)
            del arg26_1
            # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:19
            buf18 = extern_kernels.convolution(buf16, buf17, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf18, (64, 128, 2, 2), (512, 1, 256, 128), 'torch.ops.aten.convolution.default')
            del buf17
            assert_size_stride(arg27_1, (128, ), (1, ), 'input')
            assert_size_stride(arg28_1, (128, ), (1, ), 'input')
            assert_size_stride(arg29_1, (128, ), (1, ), 'input')
            assert_size_stride(arg30_1, (128, ), (1, ), 'input')
            buf19 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [out_15, out_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_8:20
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf19, arg27_1, arg28_1, arg29_1, arg30_1, 32768, stream=raw_stream0)
            del arg27_1
            del arg28_1
            del arg29_1
            del arg30_1
            assert_size_stride(arg31_1, (128, 128, 3, 3), (1152, 9, 3, 1), 'input')
            buf20 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [out_15, out_16, out_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9:21
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg31_1, buf20, 16384, 9, stream=raw_stream0)
            del arg31_1
            # Topologically Sorted Source Nodes: [out_15, out_16, out_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:22
            buf21 = extern_kernels.convolution(buf19, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf21, (64, 128, 2, 2), (512, 1, 256, 128), 'torch.ops.aten.convolution.default')
            del buf19
            assert_size_stride(arg36_1, (128, 64, 1, 1), (64, 1, 1, 1), 'input')
            # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:23
            buf22 = extern_kernels.convolution(buf16, arg36_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf22, (64, 128, 2, 2), (512, 1, 256, 128), 'torch.ops.aten.convolution.default')
            del arg36_1
            del buf16
            assert_size_stride(arg32_1, (128, ), (1, ), 'input')
            assert_size_stride(arg33_1, (128, ), (1, ), 'input')
            assert_size_stride(arg34_1, (128, ), (1, ), 'input')
            assert_size_stride(arg35_1, (128, ), (1, ), 'input')
            assert_size_stride(arg37_1, (128, ), (1, ), 'input')
            assert_size_stride(arg38_1, (128, ), (1, ), 'input')
            assert_size_stride(arg39_1, (128, ), (1, ), 'input')
            assert_size_stride(arg40_1, (128, ), (1, ), 'input')
            buf23 = buf21; del buf21  # reuse
            buf24 = buf23; del buf23  # reuse
            # Topologically Sorted Source Nodes: [out_18, input_2, out_19, out_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10:24
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf24, arg32_1, arg33_1, arg34_1, arg35_1, buf22, arg37_1, arg38_1, arg39_1, arg40_1, 32768, stream=raw_stream0)
            del arg32_1
            del arg33_1
            del arg34_1
            del arg35_1
            del arg37_1
            del arg38_1
            del arg39_1
            del arg40_1
            del buf22
            assert_size_stride(arg41_1, (128, 128, 3, 3), (1152, 9, 3, 1), 'input')
            buf25 = buf20; del buf20  # reuse
            # Topologically Sorted Source Nodes: [out_20, out_21], Original ATen: [aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9:25
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg41_1, buf25, 16384, 9, stream=raw_stream0)
            del arg41_1
            # Topologically Sorted Source Nodes: [out_20, out_21], Original ATen: [aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:26
            buf26 = extern_kernels.convolution(buf24, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf26, (64, 128, 2, 2), (512, 1, 256, 128), 'torch.ops.aten.convolution.default')
            assert_size_stride(arg42_1, (128, ), (1, ), 'input')
            assert_size_stride(arg43_1, (128, ), (1, ), 'input')
            assert_size_stride(arg44_1, (128, ), (1, ), 'input')
            assert_size_stride(arg45_1, (128, ), (1, ), 'input')
            buf27 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [out_22, out_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_8:27
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf27, arg42_1, arg43_1, arg44_1, arg45_1, 32768, stream=raw_stream0)
            del arg42_1
            del arg43_1
            del arg44_1
            del arg45_1
            assert_size_stride(arg46_1, (128, 128, 3, 3), (1152, 9, 3, 1), 'input')
            buf28 = buf25; del buf25  # reuse
            # Topologically Sorted Source Nodes: [out_22, out_23, out_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9:28
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg46_1, buf28, 16384, 9, stream=raw_stream0)
            del arg46_1
            # Topologically Sorted Source Nodes: [out_22, out_23, out_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:29
            buf29 = extern_kernels.convolution(buf27, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf29, (64, 128, 2, 2), (512, 1, 256, 128), 'torch.ops.aten.convolution.default')
            del buf27
            del buf28
            assert_size_stride(arg47_1, (128, ), (1, ), 'input')
            assert_size_stride(arg48_1, (128, ), (1, ), 'input')
            assert_size_stride(arg49_1, (128, ), (1, ), 'input')
            assert_size_stride(arg50_1, (128, ), (1, ), 'input')
            buf30 = buf29; del buf29  # reuse
            # Topologically Sorted Source Nodes: [out_25, out_26, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11:30
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf30, arg47_1, arg48_1, arg49_1, arg50_1, buf24, 32768, stream=raw_stream0)
            del arg47_1
            del arg48_1
            del arg49_1
            del arg50_1
            del buf24
            assert_size_stride(arg51_1, (256, 128, 3, 3), (1152, 9, 3, 1), 'input')
            buf31 = empty_strided_cuda((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
            # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_12:31
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_12.run(arg51_1, buf31, 32768, 9, stream=raw_stream0)
            del arg51_1
            # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:32
            buf32 = extern_kernels.convolution(buf30, buf31, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf32, (64, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution.default')
            del buf31
            assert_size_stride(arg52_1, (256, ), (1, ), 'input')
            assert_size_stride(arg53_1, (256, ), (1, ), 'input')
            assert_size_stride(arg54_1, (256, ), (1, ), 'input')
            assert_size_stride(arg55_1, (256, ), (1, ), 'input')
            buf33 = reinterpret_tensor(buf32, (64, 256, 1, 1), (256, 1, 1, 1), 0); del buf32  # reuse
            # Topologically Sorted Source Nodes: [out_29, out_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_13:33
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf33, arg52_1, arg53_1, arg54_1, arg55_1, 16384, stream=raw_stream0)
            del arg52_1
            del arg53_1
            del arg54_1
            del arg55_1
            assert_size_stride(arg56_1, (256, 256, 3, 3), (2304, 9, 3, 1), 'input')
            buf34 = empty_strided_cuda((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [out_29, out_30, out_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14:34
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg56_1, buf34, 65536, 9, stream=raw_stream0)
            del arg56_1
            # Topologically Sorted Source Nodes: [out_29, out_30, out_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:35
            buf35 = extern_kernels.convolution(buf33, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf35, (64, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution.default')
            del buf33
            assert_size_stride(arg61_1, (256, 128, 1, 1), (128, 1, 1, 1), 'input')
            # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:36
            buf36 = extern_kernels.convolution(buf30, arg61_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf36, (64, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution.default')
            del arg61_1
            del buf30
            assert_size_stride(arg57_1, (256, ), (1, ), 'input')
            assert_size_stride(arg58_1, (256, ), (1, ), 'input')
            assert_size_stride(arg59_1, (256, ), (1, ), 'input')
            assert_size_stride(arg60_1, (256, ), (1, ), 'input')
            assert_size_stride(arg62_1, (256, ), (1, ), 'input')
            assert_size_stride(arg63_1, (256, ), (1, ), 'input')
            assert_size_stride(arg64_1, (256, ), (1, ), 'input')
            assert_size_stride(arg65_1, (256, ), (1, ), 'input')
            buf37 = buf35; del buf35  # reuse
            buf38 = reinterpret_tensor(buf37, (64, 256, 1, 1), (256, 1, 1, 1), 0); del buf37  # reuse
            # Topologically Sorted Source Nodes: [out_32, input_4, out_33, out_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15:37
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf38, arg57_1, arg58_1, arg59_1, arg60_1, buf36, arg62_1, arg63_1, arg64_1, arg65_1, 16384, stream=raw_stream0)
            del arg57_1
            del arg58_1
            del arg59_1
            del arg60_1
            del arg62_1
            del arg63_1
            del arg64_1
            del arg65_1
            del buf36
            assert_size_stride(arg66_1, (256, 256, 3, 3), (2304, 9, 3, 1), 'input')
            buf39 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [out_34, out_35], Original ATen: [aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14:38
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg66_1, buf39, 65536, 9, stream=raw_stream0)
            del arg66_1
            # Topologically Sorted Source Nodes: [out_34, out_35], Original ATen: [aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:39
            buf40 = extern_kernels.convolution(buf38, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf40, (64, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution.default')
            assert_size_stride(arg67_1, (256, ), (1, ), 'input')
            assert_size_stride(arg68_1, (256, ), (1, ), 'input')
            assert_size_stride(arg69_1, (256, ), (1, ), 'input')
            assert_size_stride(arg70_1, (256, ), (1, ), 'input')
            buf41 = reinterpret_tensor(buf40, (64, 256, 1, 1), (256, 1, 1, 1), 0); del buf40  # reuse
            # Topologically Sorted Source Nodes: [out_36, out_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_13:40
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf41, arg67_1, arg68_1, arg69_1, arg70_1, 16384, stream=raw_stream0)
            del arg67_1
            del arg68_1
            del arg69_1
            del arg70_1
            assert_size_stride(arg71_1, (256, 256, 3, 3), (2304, 9, 3, 1), 'input')
            buf42 = buf39; del buf39  # reuse
            # Topologically Sorted Source Nodes: [out_36, out_37, out_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14:41
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(arg71_1, buf42, 65536, 9, stream=raw_stream0)
            del arg71_1
            # Topologically Sorted Source Nodes: [out_36, out_37, out_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:42
            buf43 = extern_kernels.convolution(buf41, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf43, (64, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution.default')
            del buf41
            del buf42
            assert_size_stride(arg72_1, (256, ), (1, ), 'input')
            assert_size_stride(arg73_1, (256, ), (1, ), 'input')
            assert_size_stride(arg74_1, (256, ), (1, ), 'input')
            assert_size_stride(arg75_1, (256, ), (1, ), 'input')
            buf44 = reinterpret_tensor(buf43, (64, 256, 1, 1), (256, 1, 1, 1), 0); del buf43  # reuse
            # Topologically Sorted Source Nodes: [out_39, out_40, out_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16:43
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf44, arg72_1, arg73_1, arg74_1, arg75_1, buf38, 16384, stream=raw_stream0)
            del arg72_1
            del arg73_1
            del arg74_1
            del arg75_1
            del buf38
            assert_size_stride(arg76_1, (512, 256, 3, 3), (2304, 9, 3, 1), 'input')
            buf45 = empty_strided_cuda((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
            # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.convolution]
            # [Provenance debug handles] triton_poi_fused_convolution_17:44
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused_convolution_17.run(arg76_1, buf45, 131072, 9, stream=raw_stream0)
            del arg76_1
            # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:45
            buf46 = extern_kernels.convolution(buf44, buf45, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf46, (64, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
            del buf45
            assert_size_stride(arg77_1, (512, ), (1, ), 'input')
            assert_size_stride(arg78_1, (512, ), (1, ), 'input')
            assert_size_stride(arg79_1, (512, ), (1, ), 'input')
            assert_size_stride(arg80_1, (512, ), (1, ), 'input')
            buf47 = reinterpret_tensor(buf46, (64, 512, 1, 1), (512, 1, 1, 1), 0); del buf46  # reuse
            # Topologically Sorted Source Nodes: [out_43, out_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_18:46
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf47, arg77_1, arg78_1, arg79_1, arg80_1, 32768, stream=raw_stream0)
            del arg77_1
            del arg78_1
            del arg79_1
            del arg80_1
            assert_size_stride(arg81_1, (512, 512, 3, 3), (4608, 9, 3, 1), 'input')
            buf48 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [out_43, out_44, out_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19:47
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(arg81_1, buf48, 262144, 9, stream=raw_stream0)
            del arg81_1
            # Topologically Sorted Source Nodes: [out_43, out_44, out_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:48
            buf49 = extern_kernels.convolution(buf47, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf49, (64, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
            del buf47
            del buf48
            assert_size_stride(arg86_1, (512, 256, 1, 1), (256, 1, 1, 1), 'input')
            # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:49
            buf50 = extern_kernels.convolution(buf44, arg86_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf50, (64, 512, 1, 1), (512, 1, 1, 1), 'torch.ops.aten.convolution.default')
            del arg86_1
            del buf44
            assert_size_stride(arg82_1, (512, ), (1, ), 'input')
            assert_size_stride(arg83_1, (512, ), (1, ), 'input')
            assert_size_stride(arg84_1, (512, ), (1, ), 'input')
            assert_size_stride(arg85_1, (512, ), (1, ), 'input')
            assert_size_stride(arg87_1, (512, ), (1, ), 'input')
            assert_size_stride(arg88_1, (512, ), (1, ), 'input')
            assert_size_stride(arg89_1, (512, ), (1, ), 'input')
            assert_size_stride(arg90_1, (512, ), (1, ), 'input')
            buf51 = buf49; del buf49  # reuse
            buf52 = reinterpret_tensor(buf51, (64, 512, 1, 1), (512, 1, 1, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [out_46, input_6, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20:50
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_relu_20.run(buf52, arg82_1, arg83_1, arg84_1, arg85_1, buf50, arg87_1, arg88_1, arg89_1, arg90_1, 32768, stream=raw_stream0)
            del arg82_1
            del arg83_1
            del arg84_1
            del arg85_1
            del arg87_1
            del arg88_1
            del arg89_1
            del arg90_1
            del buf50
            assert_size_stride(arg91_1, (512, 512, 3, 3), (4608, 9, 3, 1), 'input')
            buf53 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
            # Topologically Sorted Source Nodes: [out_48, out_49], Original ATen: [aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19:51
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(arg91_1, buf53, 262144, 9, stream=raw_stream0)
            del arg91_1
            # Topologically Sorted Source Nodes: [out_48, out_49], Original ATen: [aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:52
            buf54 = extern_kernels.convolution(buf52, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf54, (64, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
            assert_size_stride(arg92_1, (512, ), (1, ), 'input')
            assert_size_stride(arg93_1, (512, ), (1, ), 'input')
            assert_size_stride(arg94_1, (512, ), (1, ), 'input')
            assert_size_stride(arg95_1, (512, ), (1, ), 'input')
            buf55 = reinterpret_tensor(buf54, (64, 512, 1, 1), (512, 1, 1, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [out_50, out_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_relu_18:53
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf55, arg92_1, arg93_1, arg94_1, arg95_1, 32768, stream=raw_stream0)
            del arg92_1
            del arg93_1
            del arg94_1
            del arg95_1
            assert_size_stride(arg96_1, (512, 512, 3, 3), (4608, 9, 3, 1), 'input')
            buf56 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [out_50, out_51, out_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19:54
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19.run(arg96_1, buf56, 262144, 9, stream=raw_stream0)
            del arg96_1
            # Topologically Sorted Source Nodes: [out_50, out_51, out_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.convolution]
            # [Provenance debug handles] extern_kernels.convolution:55
            buf57 = extern_kernels.convolution(buf55, buf56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
            assert_size_stride(buf57, (64, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
            del buf55
            del buf56
            assert_size_stride(arg97_1, (512, ), (1, ), 'input')
            assert_size_stride(arg98_1, (512, ), (1, ), 'input')
            assert_size_stride(arg99_1, (512, ), (1, ), 'input')
            assert_size_stride(arg100_1, (512, ), (1, ), 'input')
            buf58 = reinterpret_tensor(buf57, (64, 512, 1, 1), (512, 1, 32768, 32768), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [out_53, out_54, out_55, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean]
            # [Provenance debug handles] triton_poi_fused__native_batch_norm_legit_no_training_add_mean_relu_21:56
            raw_stream0 = get_raw_stream(0)
            triton_poi_fused__native_batch_norm_legit_no_training_add_mean_relu_21.run(buf58, arg97_1, arg98_1, arg99_1, arg100_1, buf52, 32768, stream=raw_stream0)
            del arg100_1
            del arg97_1
            del arg98_1
            del arg99_1
            del buf52
            assert_size_stride(arg102_1, (3, ), (1, ), 'input')
            assert_size_stride(arg101_1, (3, 512), (512, 1), 'input')
            buf59 = empty_strided_cuda((64, 3), (3, 1), torch.float32)
            # Topologically Sorted Source Nodes: [out_53, out_54, out_55, x_4, x_5, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu, aten.mean, aten.view, aten.t, aten.addmm]
            # [Provenance debug handles] extern_kernels.addmm:57
            extern_kernels.addmm(arg102_1, reinterpret_tensor(buf58, (64, 512), (512, 1), 0), reinterpret_tensor(arg101_1, (512, 3), (1, 512), 0), alpha=1, beta=1, out=buf59)
            del arg101_1
            del arg102_1
            del buf58
        return (buf59, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((3, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
