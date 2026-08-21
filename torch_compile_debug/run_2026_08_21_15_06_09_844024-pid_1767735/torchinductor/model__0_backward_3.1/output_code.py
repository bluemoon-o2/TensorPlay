# AOT ID: ['0_backward']
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


cpp_fused__native_batch_norm_legit_functional_convolution_backward_div_native_batch_norm_backward_threshold_backward_view_0 = async_compile.cpp_pybinding(['const bool*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const bool* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                {
                    auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp8 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(8));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(8));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(8));
                    auto tmp2 = static_cast<float>(1.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0.template cast<float,1>());
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = decltype(tmp6)::blendv(tmp10, tmp6, tmp8.template cast<float,1>());
                    auto tmp12 = tmp7 + tmp11;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = static_cast<float>(2.0);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 / tmp17;
                    auto tmp19 = tmp13 - tmp18;
                    auto tmp20 = tmp7 * tmp19;
                    auto tmp21 = tmp14 - tmp18;
                    auto tmp22 = tmp11 * tmp21;
                    auto tmp23 = tmp20 + tmp22;
                    auto tmp24 = tmp19 * tmp19;
                    auto tmp25 = tmp21 * tmp21;
                    auto tmp26 = tmp24 + tmp25;
                    auto tmp27 = tmp26 / tmp17;
                    auto tmp28 = static_cast<float>(1e-05);
                    auto tmp29 = at::vec::Vectorized<float>(tmp28);
                    auto tmp30 = tmp27 + tmp29;
                    auto tmp31 = tmp30.rsqrt();
                    auto tmp32 = tmp23 * tmp31;
                    tmp12.store(out_ptr0 + static_cast<int64_t>(x0));
                    tmp18.store(out_ptr1 + static_cast<int64_t>(x0));
                    tmp23.store(out_ptr2 + static_cast<int64_t>(x0));
                    tmp32.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(512L + x1), static_cast<int64_t>(8));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp36 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0.template cast<float,1>());
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = static_cast<float>(0.5);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp16 = tmp15 - tmp9;
                        auto tmp17 = tmp16 * tmp16;
                        auto tmp19 = tmp18 - tmp9;
                        auto tmp20 = tmp19 * tmp19;
                        auto tmp21 = tmp17 + tmp20;
                        auto tmp22 = static_cast<float>(2.0);
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp21 / tmp23;
                        auto tmp25 = static_cast<float>(1e-05);
                        auto tmp26 = at::vec::Vectorized<float>(tmp25);
                        auto tmp27 = tmp24 + tmp26;
                        auto tmp28 = tmp27.rsqrt();
                        auto tmp29 = tmp28 * tmp28;
                        auto tmp30 = tmp14 * tmp29;
                        auto tmp31 = tmp10 * tmp30;
                        auto tmp32 = tmp7 - tmp31;
                        auto tmp34 = tmp33 * tmp13;
                        auto tmp35 = tmp32 - tmp34;
                        auto tmp37 = tmp28 * tmp36;
                        auto tmp38 = tmp35 * tmp37;
                        tmp38.store(out_ptr4 + static_cast<int64_t>(x1 + 512L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_1 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto in_ptr1 = in_out_ptr0;
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(8));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(8));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(8));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(2.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 / tmp4;
                    auto tmp7 = static_cast<float>(0.0);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = at::vec::VecMask<float,1>(tmp6 <= tmp8);
                    auto tmp11 = decltype(tmp8)::blendv(tmp10, tmp8, tmp9.template cast<float,1>());
                    auto tmp13 = at::vec::VecMask<float,1>(tmp12 <= tmp8);
                    auto tmp15 = decltype(tmp8)::blendv(tmp14, tmp8, tmp13.template cast<float,1>());
                    auto tmp16 = tmp11 + tmp15;
                    auto tmp17 = tmp0 - tmp5;
                    auto tmp18 = tmp11 * tmp17;
                    auto tmp19 = tmp1 - tmp5;
                    auto tmp20 = tmp15 * tmp19;
                    auto tmp21 = tmp18 + tmp20;
                    auto tmp22 = tmp17 * tmp17;
                    auto tmp23 = tmp19 * tmp19;
                    auto tmp24 = tmp22 + tmp23;
                    auto tmp25 = tmp24 / tmp4;
                    auto tmp26 = static_cast<float>(1e-05);
                    auto tmp27 = at::vec::Vectorized<float>(tmp26);
                    auto tmp28 = tmp25 + tmp27;
                    auto tmp29 = tmp28.rsqrt();
                    auto tmp30 = tmp21 * tmp29;
                    tmp5.store(out_ptr0 + static_cast<int64_t>(x0));
                    tmp16.store(out_ptr1 + static_cast<int64_t>(x0));
                    tmp21.store(out_ptr2 + static_cast<int64_t>(x0));
                    tmp30.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(512L + x1), static_cast<int64_t>(8));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3.template cast<float,1>());
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.5);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = tmp13 - tmp7;
                        auto tmp15 = tmp14 * tmp14;
                        auto tmp17 = tmp16 - tmp7;
                        auto tmp18 = tmp17 * tmp17;
                        auto tmp19 = tmp15 + tmp18;
                        auto tmp20 = static_cast<float>(2.0);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 / tmp21;
                        auto tmp23 = static_cast<float>(1e-05);
                        auto tmp24 = at::vec::Vectorized<float>(tmp23);
                        auto tmp25 = tmp22 + tmp24;
                        auto tmp26 = tmp25.rsqrt();
                        auto tmp27 = tmp26 * tmp26;
                        auto tmp28 = tmp12 * tmp27;
                        auto tmp29 = tmp8 * tmp28;
                        auto tmp30 = tmp5 - tmp29;
                        auto tmp32 = tmp31 * tmp11;
                        auto tmp33 = tmp30 - tmp32;
                        auto tmp35 = tmp26 * tmp34;
                        auto tmp36 = tmp33 * tmp35;
                        tmp36.store(in_out_ptr0 + static_cast<int64_t>(x1 + 512L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_functional_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_view_2 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const bool*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const bool* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(512L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(8));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(8));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp14 = at::vec::VecMask<float,1>::from(in_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(8));
                    auto tmp25 = at::vec::VecMask<float,1>::from(in_ptr3 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(8));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(8));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(512L + x0), static_cast<int64_t>(8));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp3 = static_cast<float>(2.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 / tmp4;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = tmp8 / tmp4;
                    auto tmp11 = static_cast<float>(0.0);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = at::vec::VecMask<float,1>(tmp10 <= tmp12);
                    auto tmp16 = static_cast<float>(1.0);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp15 / tmp17;
                    auto tmp19 = decltype(tmp12)::blendv(tmp18, tmp12, tmp14.template cast<float,1>());
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp22 = decltype(tmp12)::blendv(tmp21, tmp12, tmp13.template cast<float,1>());
                    auto tmp24 = at::vec::VecMask<float,1>(tmp23 <= tmp12);
                    auto tmp27 = tmp26 / tmp17;
                    auto tmp28 = decltype(tmp12)::blendv(tmp27, tmp12, tmp25.template cast<float,1>());
                    auto tmp30 = tmp28 + tmp29;
                    auto tmp31 = decltype(tmp12)::blendv(tmp30, tmp12, tmp24.template cast<float,1>());
                    auto tmp32 = tmp22 + tmp31;
                    auto tmp33 = tmp6 - tmp9;
                    auto tmp34 = tmp22 * tmp33;
                    auto tmp35 = tmp7 - tmp9;
                    auto tmp36 = tmp31 * tmp35;
                    auto tmp37 = tmp34 + tmp36;
                    auto tmp38 = tmp0 - tmp5;
                    auto tmp39 = tmp22 * tmp38;
                    auto tmp40 = tmp1 - tmp5;
                    auto tmp41 = tmp31 * tmp40;
                    auto tmp42 = tmp39 + tmp41;
                    auto tmp43 = tmp33 * tmp33;
                    auto tmp44 = tmp35 * tmp35;
                    auto tmp45 = tmp43 + tmp44;
                    auto tmp46 = tmp45 / tmp4;
                    auto tmp47 = static_cast<float>(1e-05);
                    auto tmp48 = at::vec::Vectorized<float>(tmp47);
                    auto tmp49 = tmp46 + tmp48;
                    auto tmp50 = tmp49.rsqrt();
                    auto tmp51 = tmp37 * tmp50;
                    auto tmp52 = tmp38 * tmp38;
                    auto tmp53 = tmp40 * tmp40;
                    auto tmp54 = tmp52 + tmp53;
                    auto tmp55 = tmp54 / tmp4;
                    auto tmp56 = tmp55 + tmp48;
                    auto tmp57 = tmp56.rsqrt();
                    auto tmp58 = tmp42 * tmp57;
                    tmp5.store(out_ptr0 + static_cast<int64_t>(x0));
                    tmp9.store(out_ptr1 + static_cast<int64_t>(x0));
                    tmp32.store(out_ptr2 + static_cast<int64_t>(x0));
                    tmp37.store(out_ptr3 + static_cast<int64_t>(x0));
                    tmp32.store(out_ptr4 + static_cast<int64_t>(x0));
                    tmp42.store(out_ptr5 + static_cast<int64_t>(x0));
                    tmp51.store(out_ptr6 + static_cast<int64_t>(x0));
                    tmp58.store(out_ptr7 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp4 = at::vec::VecMask<float,1>::from(in_ptr3 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(512L + x1), static_cast<int64_t>(8));
                        auto tmp38 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp41 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp44 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp45 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp47 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp49 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp52 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(512L + x1), static_cast<int64_t>(8));
                        auto tmp63 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp66 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                        auto tmp6 = static_cast<float>(1.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 / tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp4.template cast<float,1>());
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3.template cast<float,1>());
                        auto tmp15 = tmp13 - tmp14;
                        auto tmp17 = static_cast<float>(0.5);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 * tmp18;
                        auto tmp21 = tmp20 - tmp14;
                        auto tmp22 = tmp21 * tmp21;
                        auto tmp24 = tmp23 - tmp14;
                        auto tmp25 = tmp24 * tmp24;
                        auto tmp26 = tmp22 + tmp25;
                        auto tmp27 = static_cast<float>(2.0);
                        auto tmp28 = at::vec::Vectorized<float>(tmp27);
                        auto tmp29 = tmp26 / tmp28;
                        auto tmp30 = static_cast<float>(1e-05);
                        auto tmp31 = at::vec::Vectorized<float>(tmp30);
                        auto tmp32 = tmp29 + tmp31;
                        auto tmp33 = tmp32.rsqrt();
                        auto tmp34 = tmp33 * tmp33;
                        auto tmp35 = tmp19 * tmp34;
                        auto tmp36 = tmp15 * tmp35;
                        auto tmp37 = tmp12 - tmp36;
                        auto tmp39 = tmp38 * tmp18;
                        auto tmp40 = tmp37 - tmp39;
                        auto tmp42 = tmp33 * tmp41;
                        auto tmp43 = tmp40 * tmp42;
                        auto tmp46 = tmp44 - tmp45;
                        auto tmp48 = tmp47 * tmp18;
                        auto tmp50 = tmp49 - tmp45;
                        auto tmp51 = tmp50 * tmp50;
                        auto tmp53 = tmp52 - tmp45;
                        auto tmp54 = tmp53 * tmp53;
                        auto tmp55 = tmp51 + tmp54;
                        auto tmp56 = tmp55 / tmp28;
                        auto tmp57 = tmp56 + tmp31;
                        auto tmp58 = tmp57.rsqrt();
                        auto tmp59 = tmp58 * tmp58;
                        auto tmp60 = tmp48 * tmp59;
                        auto tmp61 = tmp46 * tmp60;
                        auto tmp62 = tmp12 - tmp61;
                        auto tmp64 = tmp63 * tmp18;
                        auto tmp65 = tmp62 - tmp64;
                        auto tmp67 = tmp58 * tmp66;
                        auto tmp68 = tmp65 * tmp67;
                        tmp43.store(out_ptr8 + static_cast<int64_t>(x1 + 512L*x0));
                        tmp68.store(out_ptr9 + static_cast<int64_t>(x1 + 512L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_3 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr0 = in_out_ptr1;
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(8L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                    }
                }
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x0));
                }
            }
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<int64_t>(x0));
                }
            }
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                float tmp_acc2 = 0;
                at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            auto tmp5 = static_cast<float>(0.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = at::vec::VecMask<float,1>(tmp4 <= tmp6);
                            auto tmp10 = tmp8 + tmp9;
                            auto tmp11 = decltype(tmp6)::blendv(tmp10, tmp6, tmp7.template cast<float,1>());
                            auto tmp12 = tmp11 * tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp11;
                            tmp_acc2_vec = tmp_acc2_vec + tmp12;
                        }
                    }
                }
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    tmp_acc0_vec.store(out_ptr1 + static_cast<int64_t>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<int64_t>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp2 = static_cast<float>(8.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp0 * tmp8;
                    tmp9.store(out_ptr4 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(256L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3.template cast<float,1>());
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = static_cast<float>(0.125);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp16 = static_cast<float>(8.0);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 / tmp17;
                        auto tmp19 = static_cast<float>(1e-05);
                        auto tmp20 = at::vec::Vectorized<float>(tmp19);
                        auto tmp21 = tmp18 + tmp20;
                        auto tmp22 = tmp21.rsqrt();
                        auto tmp23 = tmp22 * tmp22;
                        auto tmp24 = tmp14 * tmp23;
                        auto tmp25 = tmp10 * tmp24;
                        auto tmp26 = tmp7 - tmp25;
                        auto tmp28 = tmp27 * tmp13;
                        auto tmp29 = tmp26 - tmp28;
                        auto tmp31 = tmp22 * tmp30;
                        auto tmp32 = tmp29 * tmp31;
                        tmp32.store(in_out_ptr1 + static_cast<int64_t>(x1 + 256L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto out_ptr0 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(8L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                    }
                }
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x0));
                }
            }
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<int64_t>(x0));
                }
            }
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                float tmp_acc2 = 0;
                at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp3 = tmp2 * tmp2;
                            auto tmp5 = static_cast<float>(0.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = at::vec::VecMask<float,1>(tmp4 <= tmp6);
                            auto tmp9 = decltype(tmp6)::blendv(tmp8, tmp6, tmp7.template cast<float,1>());
                            auto tmp10 = tmp9 * tmp2;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp9;
                            tmp_acc2_vec = tmp_acc2_vec + tmp10;
                        }
                    }
                }
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    tmp_acc0_vec.store(out_ptr1 + static_cast<int64_t>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<int64_t>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp2 = static_cast<float>(8.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp0 * tmp8;
                    tmp9.store(out_ptr4 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(256L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3.template cast<float,1>());
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp10 = static_cast<float>(0.125);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp14 = static_cast<float>(8.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 / tmp15;
                        auto tmp17 = static_cast<float>(1e-05);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp16 + tmp18;
                        auto tmp20 = tmp19.rsqrt();
                        auto tmp21 = tmp20 * tmp20;
                        auto tmp22 = tmp12 * tmp21;
                        auto tmp23 = tmp8 * tmp22;
                        auto tmp24 = tmp5 - tmp23;
                        auto tmp26 = tmp25 * tmp11;
                        auto tmp27 = tmp24 - tmp26;
                        auto tmp29 = tmp20 * tmp28;
                        auto tmp30 = tmp27 * tmp29;
                        tmp30.store(in_out_ptr1 + static_cast<int64_t>(x1 + 256L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_5 = async_compile.cpp_pybinding(['float*', 'float*', 'float*', 'float*', 'float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
                       float* in_out_ptr4,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto out_ptr0 = in_out_ptr1;
    auto out_ptr1 = in_out_ptr2;
    auto in_ptr4 = in_out_ptr3;
    auto in_ptr5 = in_out_ptr4;
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2048L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(2048L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                    auto tmp5 = at::vec::VecMask<float,1>(tmp4 <= tmp2);
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5.template cast<float,1>());
                    auto tmp11 = tmp9 + tmp10;
                    auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3.template cast<float,1>());
                    tmp12.store(in_out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(8L))
        {
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                    }
                }
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x0));
                }
            }
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<int64_t>(x0));
                }
            }
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                    }
                }
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    tmp_acc0_vec.store(out_ptr1 + static_cast<int64_t>(x0));
                }
            }
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(8.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr2 + static_cast<int64_t>(x0));
                }
            }
            {
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                float tmp_acc2 = 0;
                at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                float tmp_acc3 = 0;
                at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                float tmp_acc4 = 0;
                at::vec::Vectorized<float> tmp_acc4_vec = at::vec::Vectorized<float>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
                {
                    {
                        if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x0 + 256L*x1), static_cast<int64_t>(8));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp4 = tmp3 * tmp3;
                            auto tmp5 = tmp0 * tmp3;
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp9 = tmp8 * tmp8;
                            auto tmp10 = tmp0 * tmp8;
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            tmp_acc1_vec = tmp_acc1_vec + tmp4;
                            tmp_acc2_vec = tmp_acc2_vec + tmp5;
                            tmp_acc3_vec = tmp_acc3_vec + tmp9;
                            tmp_acc4_vec = tmp_acc4_vec + tmp10;
                        }
                    }
                }
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    tmp_acc0_vec.store(out_ptr2 + static_cast<int64_t>(x0));
                    tmp_acc1_vec.store(out_ptr3 + static_cast<int64_t>(x0));
                    tmp_acc2_vec.store(out_ptr4 + static_cast<int64_t>(x0));
                    tmp_acc0_vec.store(out_ptr5 + static_cast<int64_t>(x0));
                    tmp_acc3_vec.store(out_ptr6 + static_cast<int64_t>(x0));
                    tmp_acc4_vec.store(out_ptr7 + static_cast<int64_t>(x0));
                }
            }
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp2 = static_cast<float>(8.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp0 * tmp8;
                    tmp9.store(out_ptr8 + static_cast<int64_t>(x0));
                }
            }
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(256L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp2 = static_cast<float>(8.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 / tmp3;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.rsqrt();
                    auto tmp9 = tmp0 * tmp8;
                    tmp9.store(out_ptr9 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(256L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp39 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(0.125);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(8.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = static_cast<float>(1e-05);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 + tmp13;
                        auto tmp15 = tmp14.rsqrt();
                        auto tmp16 = tmp15 * tmp15;
                        auto tmp17 = tmp7 * tmp16;
                        auto tmp18 = tmp3 * tmp17;
                        auto tmp19 = tmp0 - tmp18;
                        auto tmp21 = tmp20 * tmp6;
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = tmp15 * tmp23;
                        auto tmp25 = tmp22 * tmp24;
                        auto tmp28 = tmp26 - tmp27;
                        auto tmp30 = tmp29 * tmp6;
                        auto tmp32 = tmp31 / tmp10;
                        auto tmp33 = tmp32 + tmp13;
                        auto tmp34 = tmp33.rsqrt();
                        auto tmp35 = tmp34 * tmp34;
                        auto tmp36 = tmp30 * tmp35;
                        auto tmp37 = tmp28 * tmp36;
                        auto tmp38 = tmp0 - tmp37;
                        auto tmp40 = tmp39 * tmp6;
                        auto tmp41 = tmp38 - tmp40;
                        auto tmp43 = tmp34 * tmp42;
                        auto tmp44 = tmp41 * tmp43;
                        tmp25.store(in_out_ptr3 + static_cast<int64_t>(x1 + 256L*x0));
                        tmp44.store(in_out_ptr4 + static_cast<int64_t>(x1 + 256L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto in_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(8L))
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(1L))
                    {
                        {
                            if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(128L)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0 + 128L*x1), static_cast<int64_t>(8));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0 + 128L*x1), static_cast<int64_t>(8));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0 + 128L*x1), static_cast<int64_t>(8));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x0 + 128L*x1), static_cast<int64_t>(8));
                                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                                auto tmp6 = tmp4 + tmp5;
                                auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3.template cast<float,1>());
                                auto tmp10 = tmp8 - tmp9;
                                auto tmp11 = tmp7 * tmp10;
                                tmp_acc0_vec = tmp_acc0_vec + tmp7;
                                tmp_acc1_vec = tmp_acc1_vec + tmp11;
                            }
                        }
                    }
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(128L)))
                    {
                        tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x0));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<int64_t>(x0));
                    }
                }
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(128L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr2 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(32L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3.template cast<float,1>());
                            auto tmp10 = tmp8 - tmp9;
                            auto tmp12 = static_cast<float>(0.03125);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 * tmp13;
                            auto tmp16 = tmp15 * tmp15;
                            auto tmp17 = tmp14 * tmp16;
                            auto tmp18 = tmp10 * tmp17;
                            auto tmp19 = tmp7 - tmp18;
                            auto tmp21 = tmp20 * tmp13;
                            auto tmp22 = tmp19 - tmp21;
                            auto tmp24 = tmp15 * tmp23;
                            auto tmp25 = tmp22 * tmp24;
                            tmp25.store(in_out_ptr0 + static_cast<int64_t>(x1 + 128L*x0));
                        }
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(8L))
            {
                {
                    float tmp_acc0_arr[8];
                    for (int i = 0; i < 8; i++)
                    {
                        tmp_acc0_arr[i] = 0;
                    }
                    float tmp_acc1_arr[8];
                    for (int i = 0; i < 8; i++)
                    {
                        tmp_acc1_arr[i] = 0;
                    }
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(1L))
                    {
                        {
                            if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(128L)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0 + 128L*x1), static_cast<int64_t>(8));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0 + 128L*x1), static_cast<int64_t>(8));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0 + 128L*x1), static_cast<int64_t>(8));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3.template cast<float,1>());
                                auto tmp8 = tmp6 - tmp7;
                                auto tmp9 = tmp5 * tmp8;
                                tmp_acc0_vec = tmp_acc0_vec + tmp5;
                                tmp_acc1_vec = tmp_acc1_vec + tmp9;
                            }
                        }
                    }
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(128L)))
                    {
                        tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x0));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<int64_t>(x0));
                    }
                }
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(128L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr2 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(32L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3.template cast<float,1>());
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(0.03125);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp14 = tmp13 * tmp13;
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp8 * tmp15;
                            auto tmp17 = tmp5 - tmp16;
                            auto tmp19 = tmp18 * tmp11;
                            auto tmp20 = tmp17 - tmp19;
                            auto tmp22 = tmp13 * tmp21;
                            auto tmp23 = tmp20 * tmp22;
                            tmp23.store(in_out_ptr0 + static_cast<int64_t>(x1 + 128L*x0));
                        }
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.cpp_pybinding(['float*', 'float*', 'float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr5,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto in_ptr4 = in_out_ptr1;
    auto in_ptr6 = in_out_ptr2;
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4096L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4096L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                        auto tmp5 = at::vec::VecMask<float,1>(tmp4 <= tmp2);
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5.template cast<float,1>());
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3.template cast<float,1>());
                        tmp12.store(in_out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(8L))
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(32L); x1+=static_cast<int64_t>(1L))
                    {
                        {
                            if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(128L)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0 + 128L*x1), static_cast<int64_t>(8));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x0 + 128L*x1), static_cast<int64_t>(8));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x0 + 128L*x1), static_cast<int64_t>(8));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                                auto tmp3 = tmp1 - tmp2;
                                auto tmp4 = tmp0 * tmp3;
                                auto tmp7 = tmp5 - tmp6;
                                auto tmp8 = tmp0 * tmp7;
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                                tmp_acc1_vec = tmp_acc1_vec + tmp4;
                                tmp_acc2_vec = tmp_acc2_vec + tmp8;
                            }
                        }
                    }
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(128L)))
                    {
                        tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x0));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<int64_t>(x0));
                        tmp_acc0_vec.store(out_ptr2 + static_cast<int64_t>(x0));
                        tmp_acc2_vec.store(out_ptr3 + static_cast<int64_t>(x0));
                    }
                }
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(128L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr4 + static_cast<int64_t>(x0));
                    }
                }
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(128L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr5 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(32L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(0.03125);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = tmp8 * tmp8;
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp12 = tmp0 - tmp11;
                            auto tmp14 = tmp13 * tmp6;
                            auto tmp15 = tmp12 - tmp14;
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp18 = tmp15 * tmp17;
                            auto tmp21 = tmp19 - tmp20;
                            auto tmp23 = tmp22 * tmp6;
                            auto tmp25 = tmp24 * tmp24;
                            auto tmp26 = tmp23 * tmp25;
                            auto tmp27 = tmp21 * tmp26;
                            auto tmp28 = tmp0 - tmp27;
                            auto tmp30 = tmp29 * tmp6;
                            auto tmp31 = tmp28 - tmp30;
                            auto tmp33 = tmp24 * tmp32;
                            auto tmp34 = tmp31 * tmp33;
                            tmp18.store(in_out_ptr1 + static_cast<int64_t>(x1 + 128L*x0));
                            tmp34.store(in_out_ptr2 + static_cast<int64_t>(x1 + 128L*x0));
                        }
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto in_ptr3 = in_out_ptr0;
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(8L))
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(1L))
                    {
                        {
                            if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0 + 64L*x1), static_cast<int64_t>(8));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0 + 64L*x1), static_cast<int64_t>(8));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0 + 64L*x1), static_cast<int64_t>(8));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x0 + 64L*x1), static_cast<int64_t>(8));
                                auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                                auto tmp6 = tmp4 + tmp5;
                                auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3.template cast<float,1>());
                                auto tmp10 = tmp8 - tmp9;
                                auto tmp11 = tmp7 * tmp10;
                                tmp_acc0_vec = tmp_acc0_vec + tmp7;
                                tmp_acc1_vec = tmp_acc1_vec + tmp11;
                            }
                        }
                    }
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L)))
                    {
                        tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x0));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<int64_t>(x0));
                    }
                }
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr2 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3.template cast<float,1>());
                            auto tmp10 = tmp8 - tmp9;
                            auto tmp12 = static_cast<float>(0.0078125);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 * tmp13;
                            auto tmp16 = tmp15 * tmp15;
                            auto tmp17 = tmp14 * tmp16;
                            auto tmp18 = tmp10 * tmp17;
                            auto tmp19 = tmp7 - tmp18;
                            auto tmp21 = tmp20 * tmp13;
                            auto tmp22 = tmp19 - tmp21;
                            auto tmp24 = tmp15 * tmp23;
                            auto tmp25 = tmp22 * tmp24;
                            tmp25.store(in_out_ptr0 + static_cast<int64_t>(x1 + 64L*x0));
                        }
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(8L))
            {
                {
                    float tmp_acc0_arr[8];
                    for (int i = 0; i < 8; i++)
                    {
                        tmp_acc0_arr[i] = 0;
                    }
                    float tmp_acc1_arr[8];
                    for (int i = 0; i < 8; i++)
                    {
                        tmp_acc1_arr[i] = 0;
                    }
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(1L))
                    {
                        {
                            if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0 + 64L*x1), static_cast<int64_t>(8));
                                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0 + 64L*x1), static_cast<int64_t>(8));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0 + 64L*x1), static_cast<int64_t>(8));
                                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                                auto tmp1 = static_cast<float>(0.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                                auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3.template cast<float,1>());
                                auto tmp8 = tmp6 - tmp7;
                                auto tmp9 = tmp5 * tmp8;
                                tmp_acc0_vec = tmp_acc0_vec + tmp5;
                                tmp_acc1_vec = tmp_acc1_vec + tmp9;
                            }
                        }
                    }
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L)))
                    {
                        tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x0));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<int64_t>(x0));
                    }
                }
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr2 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3.template cast<float,1>());
                            auto tmp8 = tmp6 - tmp7;
                            auto tmp10 = static_cast<float>(0.0078125);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 * tmp11;
                            auto tmp14 = tmp13 * tmp13;
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp16 = tmp8 * tmp15;
                            auto tmp17 = tmp5 - tmp16;
                            auto tmp19 = tmp18 * tmp11;
                            auto tmp20 = tmp17 - tmp19;
                            auto tmp22 = tmp13 * tmp21;
                            auto tmp23 = tmp20 * tmp22;
                            tmp23.store(in_out_ptr0 + static_cast<int64_t>(x1 + 64L*x0));
                        }
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.cpp_pybinding(['float*', 'float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto in_ptr4 = in_out_ptr1;
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8192L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8192L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = at::vec::VecMask<float,1>(tmp0 <= tmp2);
                        auto tmp5 = at::vec::VecMask<float,1>(tmp4 <= tmp2);
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5.template cast<float,1>());
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3.template cast<float,1>());
                        tmp12.store(in_out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(8L))
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(1L))
                    {
                        {
                            if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L)))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0 + 64L*x1), static_cast<int64_t>(8));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x0 + 64L*x1), static_cast<int64_t>(8));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                                auto tmp3 = tmp1 - tmp2;
                                auto tmp4 = tmp0 * tmp3;
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                                tmp_acc1_vec = tmp_acc1_vec + tmp4;
                            }
                        }
                    }
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L)))
                    {
                        tmp_acc0_vec.store(out_ptr0 + static_cast<int64_t>(x0));
                        tmp_acc1_vec.store(out_ptr1 + static_cast<int64_t>(x0));
                    }
                }
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr2 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp3 = tmp1 - tmp2;
                            auto tmp5 = static_cast<float>(0.0078125);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = tmp8 * tmp8;
                            auto tmp10 = tmp7 * tmp9;
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp12 = tmp0 - tmp11;
                            auto tmp14 = tmp13 * tmp6;
                            auto tmp15 = tmp12 - tmp14;
                            auto tmp17 = tmp8 * tmp16;
                            auto tmp18 = tmp15 * tmp17;
                            tmp18.store(in_out_ptr1 + static_cast<int64_t>(x1 + 64L*x0));
                        }
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_12 = async_compile.cpp_pybinding(['float*', 'const float*', 'const int8_t*', 'float*', 'float*', 'int64_t*', 'int64_t*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const int8_t* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       int64_t* out_ptr2,
                       int64_t* out_ptr3)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8192L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8192L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        tmp2.store(in_out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(64L*x0 + 4096L*(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(64L))) + ((static_cast<int64_t>(x1) % static_cast<int64_t>(64L)))), static_cast<int64_t>(8));
                            tmp0.store(out_ptr0 + static_cast<int64_t>(x1 + 128L*x0));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(32768L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(32768L)))
                    {
                        auto tmp0 = static_cast<float>(0.0);
                        auto tmp1 = at::vec::Vectorized<float>(tmp0);
                        tmp1.store(out_ptr1 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for collapse(2)
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(1L))
                {
                    #pragma GCC ivdep
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(8L); x2+=static_cast<int64_t>(1L))
                    {
                        for(int64_t x3=static_cast<int64_t>(0L); x3<static_cast<int64_t>(64L); x3+=static_cast<int64_t>(8L))
                        {
                            {
                                if(C10_LIKELY(x3 >= static_cast<int64_t>(0) && x3 < static_cast<int64_t>(64L)))
                                {
                                    auto tmp0 = at::vec::Vectorized<int8_t>::loadu(in_ptr1 + static_cast<int64_t>(x3 + 64L*x2 + 512L*x1 + 4096L*x0), static_cast<int64_t>(8));
                                    auto tmp1 = 9L;
                                    auto tmp2 = c10::convert<int64_t>(tmp1);
                                    auto tmp3 = at::vec::convert<int64_t,2,int8_t,1>(tmp0);
                                    auto tmp4 = at::vec::VectorizedN<int64_t,2>(tmp2);
                                    auto tmp5 = tmp3 + tmp4;
                                    auto tmp6 = static_cast<int64_t>(0);
                                    auto tmp7 = at::vec::VectorizedN<int64_t,2>(tmp6);
                                    auto tmp8 = at::vec::VecMask<int64_t,2>(tmp3 < tmp7);
                                    auto tmp9 = decltype(tmp5)::blendv(tmp3, tmp5, tmp8.template cast<int64_t,2>());
                                    auto tmp10 =
                                    [&]
                                    {
                                        __at_align__ std::array<int64_t, 8> tmpbuf;
                                        tmp9.store(tmpbuf.data(), static_cast<int64_t>(8));
                                        return tmpbuf;
                                    }
                                    ()
                                    ;
                                    auto tmp11 =
                                    [&]
                                    {
                                        __at_align__ std::array<int64_t, 8> tmpbuf;
                                        #pragma GCC unroll 8
                                        for (long x3_inner = 0; x3_inner < static_cast<int64_t>(8); x3_inner++)
                                        {
                                            tmpbuf[x3_inner] = static_cast<int64_t>(tmp10[x3_inner]);
                                        }
                                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                    }
                                    ()
                                    ;
                                    TORCH_CHECK((at::vec::VecMask<int64_t,2>((at::vec::VectorizedN<int64_t,2>(0) <= tmp11) & (tmp11 < at::vec::VectorizedN<int64_t,2>(9L)))).all_masked(), "index out of bounds: 0 <= tmp11 < 9L");
                                    auto tmp13 =
                                    [&]
                                    {
                                        __at_align__ std::array<int64_t, 8> tmpbuf;
                                        #pragma GCC unroll 8
                                        for (long x3_inner = 0; x3_inner < static_cast<int64_t>(8); x3_inner++)
                                        {
                                            tmpbuf[x3_inner] = static_cast<int64_t>((-17L) + tmp10[x3_inner] + 2L*x2 + 13L*(c10::div_floor_integer(static_cast<int64_t>(tmp10[x3_inner]), static_cast<int64_t>(3L))) + 32L*x1);
                                        }
                                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), static_cast<int64_t>(8));
                                    }
                                    ()
                                    ;
                                    tmp13.store(out_ptr2 + static_cast<int64_t>(x3 + 64L*x2 + 512L*x1 + 4096L*x0), static_cast<int64_t>(8));
                                }
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L)))
                        {
                            auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(out_ptr2 + static_cast<int64_t>(64L*x0 + 4096L*(c10::div_floor_integer(static_cast<int64_t>(x1), static_cast<int64_t>(64L))) + ((static_cast<int64_t>(x1) % static_cast<int64_t>(64L)))), static_cast<int64_t>(8));
                            tmp0.store(out_ptr3 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                        }
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_functional_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_relu_threshold_backward_13 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for collapse(2)
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(256L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(256L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp0[8*8];
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(in_ptr0 + static_cast<int64_t>(x1 + 64L*x2 + 16384L*x0), static_cast<int64_t>(64L), tmp0, static_cast<int64_t>(8));
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(8L*x1_inner), static_cast<int64_t>(8));
                                    auto tmp2 = in_ptr1[static_cast<int64_t>(x1 + x1_inner)];
                                    auto tmp5 = in_ptr2[static_cast<int64_t>(x1 + x1_inner)];
                                    auto tmp8 = in_ptr3[static_cast<int64_t>(x1 + x1_inner)];
                                    auto tmp11 = in_ptr4[static_cast<int64_t>(x1 + x1_inner)];
                                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x2 + 256L*x1 + 256L*x1_inner + 16384L*x0), static_cast<int64_t>(8));
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 - tmp3;
                                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                    auto tmp7 = tmp4 * tmp6;
                                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                    auto tmp10 = tmp7 * tmp9;
                                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                    auto tmp13 = tmp10 + tmp12;
                                    auto tmp14 = at::vec::clamp_min(tmp13, decltype(tmp13)(0));
                                    auto tmp15 = static_cast<float>(0.0);
                                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                                    auto tmp17 = at::vec::VecMask<float,1>(tmp14 <= tmp16);
                                    auto tmp19 = decltype(tmp16)::blendv(tmp18, tmp16, tmp17.template cast<float,1>());
                                    tmp19.store(out_ptr0 + static_cast<int64_t>(x2 + 256L*x1 + 256L*x1_inner + 16384L*x0));
                                }
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(8L))
            {
                {
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(2L); x1+=static_cast<int64_t>(1L))
                    {
                        for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(256L); x2+=static_cast<int64_t>(8L))
                        {
                            {
                                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(256L)))
                                {
                                    alignas(std::max(std::size_t(8), alignof(float))) float tmp0[8*8];
                                    transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(out_ptr0 + static_cast<int64_t>(x2 + 256L*x0 + 16384L*x1), static_cast<int64_t>(256L), tmp0, static_cast<int64_t>(8));
                                    transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(out_ptr0 + static_cast<int64_t>(x2 + 256L*x0 + 16384L*x1), static_cast<int64_t>(256L), tmp0, static_cast<int64_t>(8));
                                    for (long x2_inner = 0; x2_inner < static_cast<int64_t>(8); x2_inner++)
                                    {
                                        auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(8L*x2_inner), static_cast<int64_t>(8));
                                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0 + 64L*x2 + 64L*x2_inner + 16384L*x1), static_cast<int64_t>(8));
                                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                                        auto tmp4 = tmp2 - tmp3;
                                        auto tmp5 = tmp1 * tmp4;
                                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                                    }
                                }
                            }
                        }
                    }
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L)))
                    {
                        tmp_acc0_vec.store(out_ptr1 + static_cast<int64_t>(x0));
                        tmp_acc1_vec.store(out_ptr2 + static_cast<int64_t>(x0));
                    }
                }
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(64L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr3 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
        {
            #pragma omp for collapse(2)
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(256L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(64L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(64L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp0[8*8];
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(out_ptr0 + static_cast<int64_t>(x1 + 256L*x2 + 16384L*x0), static_cast<int64_t>(256L), tmp0, static_cast<int64_t>(8));
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<int64_t>(8L*x1_inner), static_cast<int64_t>(8));
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x2 + 64L*x1 + 64L*x1_inner + 16384L*x0), static_cast<int64_t>(8));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2), static_cast<int64_t>(8));
                                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<int64_t>(x2), static_cast<int64_t>(8));
                                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x2), static_cast<int64_t>(8));
                                    auto tmp14 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x2), static_cast<int64_t>(8));
                                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x2), static_cast<int64_t>(8));
                                    auto tmp4 = tmp2 - tmp3;
                                    auto tmp6 = static_cast<float>(0.001953125);
                                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                                    auto tmp8 = tmp5 * tmp7;
                                    auto tmp10 = tmp9 * tmp9;
                                    auto tmp11 = tmp8 * tmp10;
                                    auto tmp12 = tmp4 * tmp11;
                                    auto tmp13 = tmp1 - tmp12;
                                    auto tmp15 = tmp14 * tmp7;
                                    auto tmp16 = tmp13 - tmp15;
                                    auto tmp18 = tmp9 * tmp17;
                                    auto tmp19 = tmp16 * tmp18;
                                    tmp19.store(in_out_ptr0 + static_cast<int64_t>(x2 + 64L*x1 + 64L*x1_inner + 16384L*x0));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused_sum_14 = async_compile.cpp_pybinding(['const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0L) && x0 < static_cast<int64_t>(3L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(3L));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(3L + x0), static_cast<int64_t>(3L));
                    auto tmp2 = tmp0 + tmp1;
                    tmp2.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(3L));
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


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
        primals_1, primals_2, primals_6, primals_7, primals_8, primals_12, primals_14, primals_18, primals_20, primals_24, primals_26, primals_30, primals_32, primals_36, primals_38, primals_42, primals_44, primals_48, primals_50, primals_54, primals_56, primals_60, primals_62, primals_66, primals_68, primals_72, primals_74, primals_78, primals_80, primals_84, primals_86, primals_90, primals_92, primals_96, primals_98, primals_102, primals_104, primals_108, primals_110, primals_114, primals_116, primals_120, primals_122, convolution, getitem_1, rsqrt, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, convolution_5, squeeze_16, relu_5, convolution_6, squeeze_19, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, relu_9, convolution_11, convolution_12, relu_10, convolution_13, relu_11, convolution_14, relu_12, convolution_15, relu_13, convolution_16, convolution_17, relu_14, convolution_18, relu_15, convolution_19, view, le, unsqueeze_202, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, tangents_1 = args
        args.clear()
        assert_size_stride(tangents_1, (2, 3), (3, 1), 'input')
        assert_size_stride(primals_122, (3, 512), (512, 1), 'input')
        buf0 = empty_strided_cpu((2, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6, permute_1, mm], Original ATen: [aten.t, aten.mm]
        # [Provenance debug handles] extern_kernels.mm:1
        extern_kernels.mm(tangents_1, primals_122, out=buf0)
        del primals_122
        assert_size_stride(le, (2, 512, 1, 1), (512, 1, 1, 1), 'input')
        assert_size_stride(convolution_19, (2, 512, 1, 1), (512, 1, 512, 512), 'input')
        buf4 = empty_strided_cpu((512, ), (1, ), torch.float32)
        buf3 = empty_strided_cpu((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf5 = empty_strided_cpu((512, ), (1, ), torch.float32)
        buf6 = empty_strided_cpu((512, ), (1, ), torch.float32)
        assert_size_stride(primals_120, (512, ), (1, ), 'input')
        buf7 = empty_strided_cpu((2, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        assert_size_stride(relu_15, (2, 512, 1, 1), (512, 1, 512, 512), 'input')
        assert_size_stride(primals_116, (512, 512, 3, 3), (4608, 1, 1536, 512), 'input')
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_functional_convolution_backward_div_native_batch_norm_backward_threshold_backward_view_0:2
        cpp_fused__native_batch_norm_legit_functional_convolution_backward_div_native_batch_norm_backward_threshold_backward_view_0(le, buf0, convolution_19, primals_120, buf4, buf3, buf5, buf6, buf7)
        del convolution_19
        del primals_120
        # Topologically Sorted Source Nodes: [view_2, div, scalar_tensor, where, out_53, unsqueeze_80, unsqueeze_81, unsqueeze_82, sub_20, mul_141, unsqueeze_83, unsqueeze_84, unsqueeze_85, mul_142, mul_143, mul_144, unsqueeze_86, unsqueeze_87, unsqueeze_88, mul_145, unsqueeze_89, unsqueeze_90, unsqueeze_91, mul_146, sub_22, sub_23, mul_147, convolution_backward], Original ATen: [aten.view, aten.div, aten.threshold_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:3
        buf8 = torch.ops.aten.convolution_backward.default(buf7, relu_15, primals_116, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del primals_116
        buf9 = buf8[0]
        assert_size_stride(buf9, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
        # buffer buf9 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf10 = buf8[1]
        assert_size_stride(buf10, (512, 512, 3, 3), (4608, 1, 1536, 512), 'torch.ops.aten.convolution_backward.default')
        # buffer buf10 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf8
        assert_size_stride(convolution_18, (2, 512, 1, 1), (512, 1, 512, 512), 'input')
        buf11 = reinterpret_tensor(buf5, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf5  # reuse
        buf12 = reinterpret_tensor(buf3, (512, ), (1, ), 0); del buf3  # reuse
        buf13 = empty_strided_cpu((512, ), (1, ), torch.float32)
        buf14 = empty_strided_cpu((512, ), (1, ), torch.float32)
        assert_size_stride(primals_114, (512, ), (1, ), 'input')
        buf15 = relu_15; del relu_15  # reuse
        assert_size_stride(relu_14, (2, 512, 1, 1), (512, 1, 512, 512), 'input')
        assert_size_stride(primals_110, (512, 512, 3, 3), (4608, 1, 1536, 512), 'input')
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_1:4
        cpp_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_1(buf15, convolution_18, buf9, primals_114, buf11, buf12, buf13, buf14)
        del buf9
        del convolution_18
        del primals_114
        # Topologically Sorted Source Nodes: [scalar_tensor, le_1, where_1, out_50, unsqueeze_92, unsqueeze_93, unsqueeze_94, sub_24, mul_150, unsqueeze_95, unsqueeze_96, unsqueeze_97, mul_151, mul_152, mul_153, unsqueeze_98, unsqueeze_99, unsqueeze_100, mul_154, unsqueeze_101, unsqueeze_102, unsqueeze_103, mul_155, sub_26, sub_27, mul_156, convolution_backward_1], Original ATen: [aten.threshold_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:5
        buf16 = torch.ops.aten.convolution_backward.default(buf15, relu_14, primals_110, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf15
        del primals_110
        buf17 = buf16[0]
        assert_size_stride(buf17, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
        # buffer buf17 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf18 = buf16[1]
        assert_size_stride(buf18, (512, 512, 3, 3), (4608, 1, 1536, 512), 'torch.ops.aten.convolution_backward.default')
        # buffer buf18 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf16
        assert_size_stride(convolution_16, (2, 512, 1, 1), (512, 1, 512, 512), 'input')
        assert_size_stride(convolution_17, (2, 512, 1, 1), (512, 1, 512, 512), 'input')
        buf27 = reinterpret_tensor(buf13, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf13  # reuse
        buf19 = buf11; del buf11  # reuse
        buf20 = empty_strided_cpu((512, ), (1, ), torch.float32)
        buf21 = empty_strided_cpu((512, ), (1, ), torch.float32)
        buf28 = empty_strided_cpu((512, ), (1, ), torch.float32)
        buf29 = empty_strided_cpu((512, ), (1, ), torch.float32)
        buf22 = empty_strided_cpu((512, ), (1, ), torch.float32)
        buf30 = empty_strided_cpu((512, ), (1, ), torch.float32)
        assert_size_stride(primals_108, (512, ), (1, ), 'input')
        assert_size_stride(primals_102, (512, ), (1, ), 'input')
        buf23 = buf7; del buf7  # reuse
        buf31 = empty_strided_cpu((2, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        assert_size_stride(relu_12, (2, 256, 2, 2), (1024, 1, 512, 256), 'input')
        assert_size_stride(primals_104, (512, 256, 1, 1), (256, 1, 1, 1), 'input')
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_functional_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_view_2:6
        cpp_fused__native_batch_norm_legit_functional_add_convolution_backward_div_native_batch_norm_backward_threshold_backward_view_2(convolution_16, convolution_17, relu_14, le, buf0, buf17, primals_108, primals_102, buf27, buf19, buf20, buf21, buf28, buf29, buf22, buf30, buf23, buf31)
        del buf0
        del buf17
        del convolution_16
        del convolution_17
        del le
        del primals_102
        del primals_108
        del relu_14
        # Topologically Sorted Source Nodes: [view_2, div, scalar_tensor, where, add_108, le_2, where_2, input_6, unsqueeze_104, unsqueeze_105, unsqueeze_106, sub_28, mul_159, unsqueeze_107, unsqueeze_108, unsqueeze_109, mul_160, mul_161, mul_162, unsqueeze_110, unsqueeze_111, unsqueeze_112, mul_163, unsqueeze_113, unsqueeze_114, unsqueeze_115, mul_164, sub_30, sub_31, mul_165, convolution_backward_2], Original ATen: [aten.view, aten.div, aten.threshold_backward, aten.add, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:7
        buf24 = torch.ops.aten.convolution_backward.default(buf23, relu_12, primals_104, [0], (2, 2), (0, 0), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf23
        del primals_104
        buf25 = buf24[0]
        assert_size_stride(buf25, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution_backward.default')
        # buffer buf25 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf26 = buf24[1]
        assert_size_stride(buf26, (512, 256, 1, 1), (256, 1, 256, 256), 'torch.ops.aten.convolution_backward.default')
        # buffer buf26 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf24
        assert_size_stride(relu_13, (2, 512, 1, 1), (512, 1, 512, 512), 'input')
        assert_size_stride(primals_98, (512, 512, 3, 3), (4608, 1, 1536, 512), 'input')
        # Topologically Sorted Source Nodes: [view_2, div, scalar_tensor, where, add_108, le_2, where_2, out_46, unsqueeze_116, unsqueeze_117, unsqueeze_118, sub_32, mul_168, unsqueeze_119, unsqueeze_120, unsqueeze_121, mul_169, mul_170, mul_171, unsqueeze_122, unsqueeze_123, unsqueeze_124, mul_172, unsqueeze_125, unsqueeze_126, unsqueeze_127, mul_173, sub_34, sub_35, mul_174, convolution_backward_3], Original ATen: [aten.view, aten.div, aten.threshold_backward, aten.add, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:8
        buf32 = torch.ops.aten.convolution_backward.default(buf31, relu_13, primals_98, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf31
        del primals_98
        buf33 = buf32[0]
        assert_size_stride(buf33, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution_backward.default')
        # buffer buf33 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf34 = buf32[1]
        assert_size_stride(buf34, (512, 512, 3, 3), (4608, 1, 1536, 512), 'torch.ops.aten.convolution_backward.default')
        # buffer buf34 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf32
        assert_size_stride(convolution_15, (2, 512, 1, 1), (512, 1, 512, 512), 'input')
        buf35 = reinterpret_tensor(buf29, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf29  # reuse
        buf36 = reinterpret_tensor(buf27, (512, ), (1, ), 0); del buf27  # reuse
        buf37 = buf21; del buf21  # reuse
        buf38 = reinterpret_tensor(buf19, (512, ), (1, ), 0); del buf19  # reuse
        assert_size_stride(primals_96, (512, ), (1, ), 'input')
        buf39 = relu_13; del relu_13  # reuse
        assert_size_stride(primals_92, (512, 256, 3, 3), (2304, 1, 768, 256), 'input')
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_1:9
        cpp_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_1(buf39, convolution_15, buf33, primals_96, buf35, buf36, buf37, buf38)
        del buf33
        del buf35
        del buf37
        del convolution_15
        del primals_96
        # Topologically Sorted Source Nodes: [scalar_tensor, le_3, where_3, out_43, unsqueeze_128, unsqueeze_129, unsqueeze_130, sub_36, mul_177, unsqueeze_131, unsqueeze_132, unsqueeze_133, mul_178, mul_179, mul_180, unsqueeze_134, unsqueeze_135, unsqueeze_136, mul_181, unsqueeze_137, unsqueeze_138, unsqueeze_139, mul_182, sub_38, sub_39, mul_183, convolution_backward_4], Original ATen: [aten.threshold_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:10
        buf40 = torch.ops.aten.convolution_backward.default(buf39, relu_12, primals_92, [0], (2, 2), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf39
        del primals_92
        buf41 = buf40[0]
        assert_size_stride(buf41, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution_backward.default')
        # buffer buf41 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf42 = buf40[1]
        assert_size_stride(buf42, (512, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
        # buffer buf42 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf40
        assert_size_stride(convolution_14, (2, 256, 2, 2), (1024, 1, 512, 256), 'input')
        buf43 = empty_strided_cpu((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf44 = buf43; del buf43  # reuse
        buf45 = empty_strided_cpu((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf46 = empty_strided_cpu((256, ), (1, ), torch.float32)
        buf47 = empty_strided_cpu((256, ), (1, ), torch.float32)
        buf48 = empty_strided_cpu((256, ), (1, ), torch.float32)
        assert_size_stride(primals_90, (256, ), (1, ), 'input')
        buf49 = convolution_14; del convolution_14  # reuse
        assert_size_stride(relu_11, (2, 256, 2, 2), (1024, 1, 512, 256), 'input')
        assert_size_stride(primals_86, (256, 256, 3, 3), (2304, 1, 768, 256), 'input')
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_3:11
        cpp_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_3(buf44, buf49, relu_12, buf25, buf41, primals_90, buf45, buf46, buf47, buf48)
        del primals_90
        # Topologically Sorted Source Nodes: [scalar_tensor, add_109, le_4, where_4, out_39, unsqueeze_140, unsqueeze_141, unsqueeze_142, sub_40, mul_186, unsqueeze_143, unsqueeze_144, unsqueeze_145, mul_187, mul_188, mul_189, unsqueeze_146, unsqueeze_147, unsqueeze_148, mul_190, unsqueeze_149, unsqueeze_150, unsqueeze_151, mul_191, sub_42, sub_43, mul_192, convolution_backward_5], Original ATen: [aten.threshold_backward, aten.add, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:12
        buf50 = torch.ops.aten.convolution_backward.default(buf49, relu_11, primals_86, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf49
        del primals_86
        buf51 = buf50[0]
        assert_size_stride(buf51, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution_backward.default')
        # buffer buf51 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf52 = buf50[1]
        assert_size_stride(buf52, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
        # buffer buf52 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf50
        assert_size_stride(convolution_13, (2, 256, 2, 2), (1024, 1, 512, 256), 'input')
        buf53 = reinterpret_tensor(buf47, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf47  # reuse
        buf54 = buf53; del buf53  # reuse
        buf55 = buf45; del buf45  # reuse
        buf56 = reinterpret_tensor(buf44, (256, ), (1, ), 0); del buf44  # reuse
        buf57 = empty_strided_cpu((256, ), (1, ), torch.float32)
        buf58 = empty_strided_cpu((256, ), (1, ), torch.float32)
        assert_size_stride(primals_84, (256, ), (1, ), 'input')
        buf59 = relu_11; del relu_11  # reuse
        assert_size_stride(relu_10, (2, 256, 2, 2), (1024, 1, 512, 256), 'input')
        assert_size_stride(primals_80, (256, 256, 3, 3), (2304, 1, 768, 256), 'input')
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_4:13
        cpp_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_4(buf54, buf59, convolution_13, buf51, primals_84, buf55, buf56, buf57, buf58)
        del buf51
        del convolution_13
        del primals_84
        # Topologically Sorted Source Nodes: [scalar_tensor, le_5, where_5, out_36, unsqueeze_152, unsqueeze_153, unsqueeze_154, sub_44, mul_195, unsqueeze_155, unsqueeze_156, unsqueeze_157, mul_196, mul_197, mul_198, unsqueeze_158, unsqueeze_159, unsqueeze_160, mul_199, unsqueeze_161, unsqueeze_162, unsqueeze_163, mul_200, sub_46, sub_47, mul_201, convolution_backward_6], Original ATen: [aten.threshold_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:14
        buf60 = torch.ops.aten.convolution_backward.default(buf59, relu_10, primals_80, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf59
        del primals_80
        buf61 = buf60[0]
        assert_size_stride(buf61, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution_backward.default')
        # buffer buf61 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf62 = buf60[1]
        assert_size_stride(buf62, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
        # buffer buf62 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf60
        buf63 = relu_10; del relu_10  # reuse
        assert_size_stride(convolution_12, (2, 256, 2, 2), (1024, 1, 512, 256), 'input')
        assert_size_stride(convolution_11, (2, 256, 2, 2), (1024, 1, 512, 256), 'input')
        buf64 = reinterpret_tensor(buf57, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf57  # reuse
        buf65 = buf64; del buf64  # reuse
        buf74 = buf55; del buf55  # reuse
        buf75 = buf74; del buf74  # reuse
        buf67 = reinterpret_tensor(buf54, (256, ), (1, ), 0); del buf54  # reuse
        buf66 = empty_strided_cpu((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf68 = empty_strided_cpu((256, ), (1, ), torch.float32)
        buf77 = empty_strided_cpu((256, ), (1, ), torch.float32)
        buf76 = empty_strided_cpu((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf78 = empty_strided_cpu((256, ), (1, ), torch.float32)
        buf69 = empty_strided_cpu((256, ), (1, ), torch.float32)
        buf79 = empty_strided_cpu((256, ), (1, ), torch.float32)
        assert_size_stride(primals_78, (256, ), (1, ), 'input')
        assert_size_stride(primals_72, (256, ), (1, ), 'input')
        buf70 = convolution_12; del convolution_12  # reuse
        buf80 = convolution_11; del convolution_11  # reuse
        assert_size_stride(relu_8, (2, 128, 4, 4), (2048, 1, 512, 128), 'input')
        assert_size_stride(primals_74, (256, 128, 1, 1), (128, 1, 1, 1), 'input')
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_5:15
        cpp_fused__native_batch_norm_legit_functional_add_convolution_backward_native_batch_norm_backward_threshold_backward_5(buf63, buf65, buf75, buf70, buf80, relu_12, buf25, buf41, buf61, primals_78, primals_72, buf67, buf66, buf68, buf77, buf76, buf78, buf69, buf79)
        del buf25
        del buf41
        del buf61
        del buf63
        del buf65
        del primals_72
        del primals_78
        del relu_12
        # Topologically Sorted Source Nodes: [input_4, unsqueeze_164, unsqueeze_165, unsqueeze_166, sub_48, mul_204, unsqueeze_167, unsqueeze_168, unsqueeze_169, mul_205, mul_206, mul_207, unsqueeze_170, unsqueeze_171, unsqueeze_172, mul_208, unsqueeze_173, unsqueeze_174, unsqueeze_175, mul_209, sub_50, sub_51, mul_210, convolution_backward_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:16
        buf71 = torch.ops.aten.convolution_backward.default(buf70, relu_8, primals_74, [0], (2, 2), (0, 0), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf70
        del primals_74
        buf72 = buf71[0]
        assert_size_stride(buf72, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution_backward.default')
        # buffer buf72 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf73 = buf71[1]
        assert_size_stride(buf73, (256, 128, 1, 1), (128, 1, 128, 128), 'torch.ops.aten.convolution_backward.default')
        # buffer buf73 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf71
        assert_size_stride(relu_9, (2, 256, 2, 2), (1024, 1, 512, 256), 'input')
        assert_size_stride(primals_68, (256, 256, 3, 3), (2304, 1, 768, 256), 'input')
        # Topologically Sorted Source Nodes: [out_32, unsqueeze_176, unsqueeze_177, unsqueeze_178, sub_52, mul_213, unsqueeze_179, unsqueeze_180, unsqueeze_181, mul_214, mul_215, mul_216, unsqueeze_182, unsqueeze_183, unsqueeze_184, mul_217, unsqueeze_185, unsqueeze_186, unsqueeze_187, mul_218, sub_54, sub_55, mul_219, convolution_backward_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:17
        buf81 = torch.ops.aten.convolution_backward.default(buf80, relu_9, primals_68, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf80
        del primals_68
        buf82 = buf81[0]
        assert_size_stride(buf82, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution_backward.default')
        # buffer buf82 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf83 = buf81[1]
        assert_size_stride(buf83, (256, 256, 3, 3), (2304, 1, 768, 256), 'torch.ops.aten.convolution_backward.default')
        # buffer buf83 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf81
        assert_size_stride(convolution_10, (2, 256, 2, 2), (1024, 1, 512, 256), 'input')
        buf84 = reinterpret_tensor(buf78, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf78  # reuse
        buf85 = buf84; del buf84  # reuse
        buf86 = buf76; del buf76  # reuse
        buf87 = reinterpret_tensor(buf75, (256, ), (1, ), 0); del buf75  # reuse
        buf88 = buf68; del buf68  # reuse
        buf89 = reinterpret_tensor(buf66, (256, ), (1, ), 0); del buf66  # reuse
        assert_size_stride(primals_66, (256, ), (1, ), 'input')
        buf90 = relu_9; del relu_9  # reuse
        assert_size_stride(primals_62, (256, 128, 3, 3), (1152, 1, 384, 128), 'input')
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_4:18
        cpp_fused__native_batch_norm_legit_functional_convolution_backward_native_batch_norm_backward_threshold_backward_4(buf85, buf90, convolution_10, buf82, primals_66, buf86, buf87, buf88, buf89)
        del buf82
        del buf85
        del buf86
        del buf88
        del convolution_10
        del primals_66
        # Topologically Sorted Source Nodes: [scalar_tensor, le_7, where_7, out_29, unsqueeze_188, unsqueeze_189, unsqueeze_190, sub_56, mul_222, unsqueeze_191, unsqueeze_192, unsqueeze_193, mul_223, mul_224, mul_225, unsqueeze_194, unsqueeze_195, unsqueeze_196, mul_226, unsqueeze_197, unsqueeze_198, unsqueeze_199, mul_227, sub_58, sub_59, mul_228, convolution_backward_9], Original ATen: [aten.threshold_backward, aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:19
        buf91 = torch.ops.aten.convolution_backward.default(buf90, relu_8, primals_62, [0], (2, 2), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf90
        del primals_62
        buf92 = buf91[0]
        assert_size_stride(buf92, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution_backward.default')
        # buffer buf92 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf93 = buf91[1]
        assert_size_stride(buf93, (256, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
        # buffer buf93 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf91
        assert_size_stride(convolution_9, (2, 128, 4, 4), (2048, 1, 512, 128), 'input')
        assert_size_stride(unsqueeze_202, (1, 128, 1, 1), (128, 1, 1, 1), 'input')
        assert_size_stride(squeeze_28, (128, ), (1, ), 'input')
        buf94 = empty_strided_cpu((128, ), (1, ), torch.float32)
        buf95 = empty_strided_cpu((128, ), (1, ), torch.float32)
        buf96 = empty_strided_cpu((128, ), (1, ), torch.float32)
        assert_size_stride(primals_60, (128, ), (1, ), 'input')
        buf97 = convolution_9; del convolution_9  # reuse
        assert_size_stride(relu_7, (2, 128, 4, 4), (2048, 1, 512, 128), 'input')
        assert_size_stride(primals_56, (128, 128, 3, 3), (1152, 1, 384, 128), 'input')
        # [Provenance debug handles] cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_6:20
        cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_6(buf97, relu_8, buf72, buf92, unsqueeze_202, squeeze_28, primals_60, buf94, buf95, buf96)
        del primals_60
        del squeeze_28
        del unsqueeze_202
        # Topologically Sorted Source Nodes: [scalar_tensor, add_111, le_8, where_8, sub_60, mul_231, unsqueeze_203, unsqueeze_204, unsqueeze_205, mul_232, mul_233, mul_234, unsqueeze_206, unsqueeze_207, unsqueeze_208, mul_235, unsqueeze_209, unsqueeze_210, unsqueeze_211, mul_236, sub_62, sub_63, mul_237, convolution_backward_10], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:21
        buf98 = torch.ops.aten.convolution_backward.default(buf97, relu_7, primals_56, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf97
        del primals_56
        buf99 = buf98[0]
        assert_size_stride(buf99, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution_backward.default')
        # buffer buf99 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf100 = buf98[1]
        assert_size_stride(buf100, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
        # buffer buf100 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf98
        assert_size_stride(convolution_8, (2, 128, 4, 4), (2048, 1, 512, 128), 'input')
        assert_size_stride(unsqueeze_214, (1, 128, 1, 1), (128, 1, 1, 1), 'input')
        assert_size_stride(squeeze_25, (128, ), (1, ), 'input')
        buf101 = buf95; del buf95  # reuse
        buf102 = empty_strided_cpu((128, ), (1, ), torch.float32)
        buf103 = empty_strided_cpu((128, ), (1, ), torch.float32)
        assert_size_stride(primals_54, (128, ), (1, ), 'input')
        buf104 = relu_7; del relu_7  # reuse
        assert_size_stride(relu_6, (2, 128, 4, 4), (2048, 1, 512, 128), 'input')
        assert_size_stride(primals_50, (128, 128, 3, 3), (1152, 1, 384, 128), 'input')
        # [Provenance debug handles] cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7:22
        cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7(buf104, buf99, convolution_8, unsqueeze_214, squeeze_25, primals_54, buf101, buf102, buf103)
        del buf99
        del convolution_8
        del primals_54
        del squeeze_25
        del unsqueeze_214
        # Topologically Sorted Source Nodes: [scalar_tensor, le_9, where_9, sub_64, mul_240, unsqueeze_215, unsqueeze_216, unsqueeze_217, mul_241, mul_242, mul_243, unsqueeze_218, unsqueeze_219, unsqueeze_220, mul_244, unsqueeze_221, unsqueeze_222, unsqueeze_223, mul_245, sub_66, sub_67, mul_246, convolution_backward_11], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:23
        buf105 = torch.ops.aten.convolution_backward.default(buf104, relu_6, primals_50, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf104
        del primals_50
        buf106 = buf105[0]
        assert_size_stride(buf106, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution_backward.default')
        # buffer buf106 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf107 = buf105[1]
        assert_size_stride(buf107, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
        # buffer buf107 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf105
        buf108 = relu_6; del relu_6  # reuse
        assert_size_stride(convolution_7, (2, 128, 4, 4), (2048, 1, 512, 128), 'input')
        assert_size_stride(unsqueeze_226, (1, 128, 1, 1), (128, 1, 1, 1), 'input')
        assert_size_stride(convolution_6, (2, 128, 4, 4), (2048, 1, 512, 128), 'input')
        assert_size_stride(unsqueeze_238, (1, 128, 1, 1), (128, 1, 1, 1), 'input')
        assert_size_stride(squeeze_22, (128, ), (1, ), 'input')
        assert_size_stride(squeeze_19, (128, ), (1, ), 'input')
        buf109 = buf102; del buf102  # reuse
        buf110 = empty_strided_cpu((128, ), (1, ), torch.float32)
        buf116 = empty_strided_cpu((128, ), (1, ), torch.float32)
        buf117 = empty_strided_cpu((128, ), (1, ), torch.float32)
        buf111 = empty_strided_cpu((128, ), (1, ), torch.float32)
        buf118 = empty_strided_cpu((128, ), (1, ), torch.float32)
        assert_size_stride(primals_48, (128, ), (1, ), 'input')
        assert_size_stride(primals_42, (128, ), (1, ), 'input')
        buf112 = convolution_7; del convolution_7  # reuse
        buf119 = convolution_6; del convolution_6  # reuse
        assert_size_stride(relu_4, (2, 64, 8, 8), (4096, 1, 512, 64), 'input')
        assert_size_stride(primals_44, (128, 64, 1, 1), (64, 1, 1, 1), 'input')
        # [Provenance debug handles] cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_8:24
        cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_8(buf108, buf112, buf119, relu_8, buf72, buf92, buf106, unsqueeze_226, unsqueeze_238, squeeze_22, squeeze_19, primals_48, primals_42, buf109, buf110, buf116, buf117, buf111, buf118)
        del buf106
        del buf108
        del buf72
        del buf92
        del primals_42
        del primals_48
        del relu_8
        del squeeze_19
        del squeeze_22
        del unsqueeze_226
        del unsqueeze_238
        # Topologically Sorted Source Nodes: [sub_68, mul_249, unsqueeze_227, unsqueeze_228, unsqueeze_229, mul_250, mul_251, mul_252, unsqueeze_230, unsqueeze_231, unsqueeze_232, mul_253, unsqueeze_233, unsqueeze_234, unsqueeze_235, mul_254, sub_70, sub_71, mul_255, convolution_backward_12], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:25
        buf113 = torch.ops.aten.convolution_backward.default(buf112, relu_4, primals_44, [0], (2, 2), (0, 0), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf112
        del primals_44
        buf114 = buf113[0]
        assert_size_stride(buf114, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution_backward.default')
        # buffer buf114 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf115 = buf113[1]
        assert_size_stride(buf115, (128, 64, 1, 1), (64, 1, 64, 64), 'torch.ops.aten.convolution_backward.default')
        # buffer buf115 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf113
        assert_size_stride(relu_5, (2, 128, 4, 4), (2048, 1, 512, 128), 'input')
        assert_size_stride(primals_38, (128, 128, 3, 3), (1152, 1, 384, 128), 'input')
        # Topologically Sorted Source Nodes: [sub_72, mul_258, unsqueeze_239, unsqueeze_240, unsqueeze_241, mul_259, mul_260, mul_261, unsqueeze_242, unsqueeze_243, unsqueeze_244, mul_262, unsqueeze_245, unsqueeze_246, unsqueeze_247, mul_263, sub_74, sub_75, mul_264, convolution_backward_13], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:26
        buf120 = torch.ops.aten.convolution_backward.default(buf119, relu_5, primals_38, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf119
        del primals_38
        buf121 = buf120[0]
        assert_size_stride(buf121, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution_backward.default')
        # buffer buf121 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf122 = buf120[1]
        assert_size_stride(buf122, (128, 128, 3, 3), (1152, 1, 384, 128), 'torch.ops.aten.convolution_backward.default')
        # buffer buf122 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf120
        assert_size_stride(convolution_5, (2, 128, 4, 4), (2048, 1, 512, 128), 'input')
        assert_size_stride(unsqueeze_250, (1, 128, 1, 1), (128, 1, 1, 1), 'input')
        assert_size_stride(squeeze_16, (128, ), (1, ), 'input')
        buf123 = buf117; del buf117  # reuse
        buf124 = buf110; del buf110  # reuse
        buf125 = empty_strided_cpu((128, ), (1, ), torch.float32)
        assert_size_stride(primals_36, (128, ), (1, ), 'input')
        buf126 = relu_5; del relu_5  # reuse
        assert_size_stride(primals_32, (128, 64, 3, 3), (576, 1, 192, 64), 'input')
        # [Provenance debug handles] cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7:27
        cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_7(buf126, buf121, convolution_5, unsqueeze_250, squeeze_16, primals_36, buf123, buf124, buf125)
        del buf121
        del buf124
        del convolution_5
        del primals_36
        del squeeze_16
        del unsqueeze_250
        # Topologically Sorted Source Nodes: [scalar_tensor, le_11, where_11, sub_76, mul_267, unsqueeze_251, unsqueeze_252, unsqueeze_253, mul_268, mul_269, mul_270, unsqueeze_254, unsqueeze_255, unsqueeze_256, mul_271, unsqueeze_257, unsqueeze_258, unsqueeze_259, mul_272, sub_78, sub_79, mul_273, convolution_backward_14], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:28
        buf127 = torch.ops.aten.convolution_backward.default(buf126, relu_4, primals_32, [0], (2, 2), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf126
        del primals_32
        buf128 = buf127[0]
        assert_size_stride(buf128, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution_backward.default')
        # buffer buf128 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf129 = buf127[1]
        assert_size_stride(buf129, (128, 64, 3, 3), (576, 1, 192, 64), 'torch.ops.aten.convolution_backward.default')
        # buffer buf129 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf127
        assert_size_stride(convolution_4, (2, 64, 8, 8), (4096, 1, 512, 64), 'input')
        assert_size_stride(unsqueeze_262, (1, 64, 1, 1), (64, 1, 1, 1), 'input')
        assert_size_stride(squeeze_13, (64, ), (1, ), 'input')
        buf130 = empty_strided_cpu((64, ), (1, ), torch.float32)
        buf131 = empty_strided_cpu((64, ), (1, ), torch.float32)
        buf132 = empty_strided_cpu((64, ), (1, ), torch.float32)
        assert_size_stride(primals_30, (64, ), (1, ), 'input')
        buf133 = convolution_4; del convolution_4  # reuse
        assert_size_stride(relu_3, (2, 64, 8, 8), (4096, 1, 512, 64), 'input')
        assert_size_stride(primals_26, (64, 64, 3, 3), (576, 1, 192, 64), 'input')
        # [Provenance debug handles] cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_9:29
        cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_9(buf133, relu_4, buf114, buf128, unsqueeze_262, squeeze_13, primals_30, buf130, buf131, buf132)
        del primals_30
        del squeeze_13
        del unsqueeze_262
        # Topologically Sorted Source Nodes: [scalar_tensor, add_113, le_12, where_12, sub_80, mul_276, unsqueeze_263, unsqueeze_264, unsqueeze_265, mul_277, mul_278, mul_279, unsqueeze_266, unsqueeze_267, unsqueeze_268, mul_280, unsqueeze_269, unsqueeze_270, unsqueeze_271, mul_281, sub_82, sub_83, mul_282, convolution_backward_15], Original ATen: [aten.threshold_backward, aten.add, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:30
        buf134 = torch.ops.aten.convolution_backward.default(buf133, relu_3, primals_26, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf133
        del primals_26
        buf135 = buf134[0]
        assert_size_stride(buf135, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution_backward.default')
        # buffer buf135 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf136 = buf134[1]
        assert_size_stride(buf136, (64, 64, 3, 3), (576, 1, 192, 64), 'torch.ops.aten.convolution_backward.default')
        # buffer buf136 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf134
        assert_size_stride(convolution_3, (2, 64, 8, 8), (4096, 1, 512, 64), 'input')
        assert_size_stride(unsqueeze_274, (1, 64, 1, 1), (64, 1, 1, 1), 'input')
        assert_size_stride(squeeze_10, (64, ), (1, ), 'input')
        buf137 = buf131; del buf131  # reuse
        buf138 = empty_strided_cpu((64, ), (1, ), torch.float32)
        buf139 = empty_strided_cpu((64, ), (1, ), torch.float32)
        assert_size_stride(primals_24, (64, ), (1, ), 'input')
        buf140 = relu_3; del relu_3  # reuse
        assert_size_stride(relu_2, (2, 64, 8, 8), (4096, 1, 512, 64), 'input')
        assert_size_stride(primals_20, (64, 64, 3, 3), (576, 1, 192, 64), 'input')
        # [Provenance debug handles] cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10:31
        cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10(buf140, buf135, convolution_3, unsqueeze_274, squeeze_10, primals_24, buf137, buf138, buf139)
        del buf135
        del convolution_3
        del primals_24
        del squeeze_10
        del unsqueeze_274
        # Topologically Sorted Source Nodes: [scalar_tensor, le_13, where_13, sub_84, mul_285, unsqueeze_275, unsqueeze_276, unsqueeze_277, mul_286, mul_287, mul_288, unsqueeze_278, unsqueeze_279, unsqueeze_280, mul_289, unsqueeze_281, unsqueeze_282, unsqueeze_283, mul_290, sub_86, sub_87, mul_291, convolution_backward_16], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:32
        buf141 = torch.ops.aten.convolution_backward.default(buf140, relu_2, primals_20, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf140
        del primals_20
        buf142 = buf141[0]
        assert_size_stride(buf142, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution_backward.default')
        # buffer buf142 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf143 = buf141[1]
        assert_size_stride(buf143, (64, 64, 3, 3), (576, 1, 192, 64), 'torch.ops.aten.convolution_backward.default')
        # buffer buf143 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf141
        buf144 = relu_2; del relu_2  # reuse
        assert_size_stride(convolution_2, (2, 64, 8, 8), (4096, 1, 512, 64), 'input')
        assert_size_stride(unsqueeze_286, (1, 64, 1, 1), (64, 1, 1, 1), 'input')
        assert_size_stride(squeeze_7, (64, ), (1, ), 'input')
        buf145 = buf138; del buf138  # reuse
        buf146 = empty_strided_cpu((64, ), (1, ), torch.float32)
        buf147 = empty_strided_cpu((64, ), (1, ), torch.float32)
        assert_size_stride(primals_18, (64, ), (1, ), 'input')
        buf148 = convolution_2; del convolution_2  # reuse
        assert_size_stride(relu_1, (2, 64, 8, 8), (4096, 1, 512, 64), 'input')
        assert_size_stride(primals_14, (64, 64, 3, 3), (576, 1, 192, 64), 'input')
        # [Provenance debug handles] cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_11:33
        cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_11(buf144, buf148, relu_4, buf114, buf128, buf142, unsqueeze_286, squeeze_7, primals_18, buf145, buf146, buf147)
        del buf114
        del buf128
        del buf142
        del primals_18
        del relu_4
        del squeeze_7
        del unsqueeze_286
        # Topologically Sorted Source Nodes: [sub_88, mul_294, unsqueeze_287, unsqueeze_288, unsqueeze_289, mul_295, mul_296, mul_297, unsqueeze_290, unsqueeze_291, unsqueeze_292, mul_298, unsqueeze_293, unsqueeze_294, unsqueeze_295, mul_299, sub_90, sub_91, mul_300, convolution_backward_17], Original ATen: [aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:34
        buf149 = torch.ops.aten.convolution_backward.default(buf148, relu_1, primals_14, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf148
        del primals_14
        buf150 = buf149[0]
        assert_size_stride(buf150, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution_backward.default')
        # buffer buf150 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf151 = buf149[1]
        assert_size_stride(buf151, (64, 64, 3, 3), (576, 1, 192, 64), 'torch.ops.aten.convolution_backward.default')
        # buffer buf151 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf149
        assert_size_stride(convolution_1, (2, 64, 8, 8), (4096, 1, 512, 64), 'input')
        assert_size_stride(unsqueeze_298, (1, 64, 1, 1), (64, 1, 1, 1), 'input')
        assert_size_stride(squeeze_4, (64, ), (1, ), 'input')
        buf152 = buf146; del buf146  # reuse
        buf153 = empty_strided_cpu((64, ), (1, ), torch.float32)
        buf154 = empty_strided_cpu((64, ), (1, ), torch.float32)
        assert_size_stride(primals_12, (64, ), (1, ), 'input')
        buf155 = relu_1; del relu_1  # reuse
        assert_size_stride(getitem_2, (2, 64, 8, 8), (4096, 1, 512, 64), 'input')
        assert_size_stride(primals_8, (64, 64, 3, 3), (576, 1, 192, 64), 'input')
        # [Provenance debug handles] cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10:35
        cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10(buf155, buf150, convolution_1, unsqueeze_298, squeeze_4, primals_12, buf152, buf153, buf154)
        del buf150
        del buf153
        del convolution_1
        del primals_12
        del squeeze_4
        del unsqueeze_298
        # Topologically Sorted Source Nodes: [scalar_tensor, le_15, where_15, sub_92, mul_303, unsqueeze_299, unsqueeze_300, unsqueeze_301, mul_304, mul_305, mul_306, unsqueeze_302, unsqueeze_303, unsqueeze_304, mul_307, unsqueeze_305, unsqueeze_306, unsqueeze_307, mul_308, sub_94, sub_95, mul_309, convolution_backward_18], Original ATen: [aten.threshold_backward, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:36
        buf156 = torch.ops.aten.convolution_backward.default(buf155, getitem_2, primals_8, [0], (1, 1), (1, 1), (1, 1), False, (0, 0), 1, [True, True, False])
        del buf155
        del getitem_2
        del primals_8
        buf157 = buf156[0]
        assert_size_stride(buf157, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution_backward.default')
        # buffer buf157 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        buf158 = buf156[1]
        assert_size_stride(buf158, (64, 64, 3, 3), (576, 1, 192, 64), 'torch.ops.aten.convolution_backward.default')
        # buffer buf158 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf156
        buf162 = buf144; del buf144  # reuse
        buf163 = empty_strided_cpu((128, 64), (1, 128), torch.float32)
        buf159 = empty_strided_cpu((128, 256), (256, 1), torch.float32)
        assert_size_stride(getitem_3, (2, 64, 8, 8), (4096, 1, 512, 64), 'input')
        buf160 = empty_strided_cpu((2, 64, 8, 8), (4096, 1, 512, 64), torch.int64)
        buf161 = empty_strided_cpu((128, 64), (1, 128), torch.int64)
        # [Provenance debug handles] cpp_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_12:37
        cpp_fused_add_max_pool2d_with_indices_max_pool2d_with_indices_backward_12(buf162, buf157, getitem_3, buf163, buf159, buf160, buf161)
        del buf157
        del buf160
        del buf162
        del getitem_3
        aten.scatter_reduce_.two(buf159,1,buf161,buf163, reduce='sum', include_self=True)
        del buf161
        del buf163
        assert_size_stride(convolution, (2, 64, 16, 16), (16384, 1, 1024, 64), 'input')
        assert_size_stride(getitem_1, (1, 64, 1, 1), (64, 1, 64, 64), 'input')
        assert_size_stride(rsqrt, (1, 64, 1, 1), (64, 1, 64, 64), 'input')
        assert_size_stride(primals_6, (64, ), (1, ), 'input')
        assert_size_stride(primals_7, (64, ), (1, ), 'input')
        buf165 = empty_strided_cpu((2, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf166 = empty_strided_cpu((64, ), (1, ), torch.float32)
        buf167 = empty_strided_cpu((64, ), (1, ), torch.float32)
        buf168 = empty_strided_cpu((64, ), (1, ), torch.float32)
        buf169 = convolution; del convolution  # reuse
        assert_size_stride(primals_2, (2, 3, 32, 32), (3072, 1, 96, 3), 'input')
        assert_size_stride(primals_1, (64, 3, 7, 7), (147, 1, 21, 3), 'input')
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_functional_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_relu_threshold_backward_13:38
        cpp_fused__native_batch_norm_legit_functional_convolution_backward_max_pool2d_with_indices_backward_native_batch_norm_backward_relu_threshold_backward_13(buf169, getitem_1, rsqrt, primals_6, primals_7, buf159, buf165, buf166, buf167, buf168)
        del buf159
        del buf165
        del buf167
        del getitem_1
        del primals_6
        del primals_7
        del rsqrt
        # Topologically Sorted Source Nodes: [x_1, unsqueeze_308, unsqueeze_309, unsqueeze_310, sub_96, mul_312, unsqueeze_311, unsqueeze_312, unsqueeze_313, mul_313, mul_314, mul_315, unsqueeze_314, unsqueeze_315, unsqueeze_316, mul_316, unsqueeze_317, unsqueeze_318, unsqueeze_319, mul_317, sub_98, sub_99, mul_318, convolution_backward_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.native_batch_norm_backward, aten.convolution_backward]
        # [Provenance debug handles] torch.ops.aten.convolution_backward.default:39
        buf170 = torch.ops.aten.convolution_backward.default(buf169, primals_2, primals_1, [0], (2, 2), (3, 3), (1, 1), False, (0, 0), 1, [False, True, False])
        del buf169
        del primals_1
        del primals_2
        buf171 = buf170[1]
        assert_size_stride(buf171, (64, 3, 7, 7), (147, 1, 21, 3), 'torch.ops.aten.convolution_backward.default')
        # buffer buf171 (op: torch.ops.aten.convolution_backward.default) is assumed to be not aligned
        del buf170
        buf2 = empty_strided_cpu((1, 3), (3, 1), torch.float32)
        assert_size_stride(view, (2, 512), (512, 1), 'input')
        # [Provenance debug handles] cpp_fused_sum_14:40
        cpp_fused_sum_14(tangents_1, buf2)
        buf1 = empty_strided_cpu((3, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [permute_2, permute_4], Original ATen: [aten.t, aten.mm]
        # [Provenance debug handles] extern_kernels.mm:41
        extern_kernels.mm(reinterpret_tensor(tangents_1, (3, 2), (1, 3), 0), view, out=buf1)
        del tangents_1
        del view
        return (buf171, None, None, None, None, buf168, buf166, buf158, None, None, None, buf154, buf152, buf151, None, None, None, buf147, buf145, buf143, None, None, None, buf139, buf137, buf136, None, None, None, buf132, buf130, buf129, None, None, None, buf125, buf123, buf122, None, None, None, buf118, buf116, buf115, None, None, None, buf111, buf109, buf107, None, None, None, buf103, buf101, buf100, None, None, None, buf96, buf94, buf93, None, None, None, buf89, buf87, buf83, None, None, None, buf79, buf77, buf73, None, None, None, buf69, buf67, buf62, None, None, None, buf58, buf56, buf52, None, None, None, buf48, buf46, buf42, None, None, None, buf38, buf36, buf34, None, None, None, buf30, buf28, buf26, None, None, None, buf22, buf20, buf18, None, None, None, buf14, buf12, buf10, None, None, None, buf6, buf4, buf1, reinterpret_tensor(buf2, (3, ), (1, ), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((2, 3, 32, 32), (3072, 1, 96, 3), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((3, 512), (512, 1), device='cpu', dtype=torch.float32)
    convolution = rand_strided((2, 64, 16, 16), (16384, 1, 1024, 64), device='cpu', dtype=torch.float32)
    getitem_1 = rand_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    rsqrt = rand_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    getitem_2 = rand_strided((2, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    getitem_3 = rand_strided((2, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.int8)
    convolution_1 = rand_strided((2, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((2, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((2, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((2, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((2, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((2, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((2, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((2, 64, 8, 8), (4096, 1, 512, 64), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((2, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((2, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((2, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((2, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((2, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((2, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((2, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((2, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((2, 128, 4, 4), (2048, 1, 512, 128), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((2, 256, 2, 2), (1024, 1, 512, 256), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((2, 256, 2, 2), (1024, 1, 512, 256), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((2, 256, 2, 2), (1024, 1, 512, 256), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((2, 256, 2, 2), (1024, 1, 512, 256), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((2, 256, 2, 2), (1024, 1, 512, 256), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((2, 256, 2, 2), (1024, 1, 512, 256), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((2, 256, 2, 2), (1024, 1, 512, 256), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((2, 256, 2, 2), (1024, 1, 512, 256), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((2, 256, 2, 2), (1024, 1, 512, 256), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((2, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((2, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((2, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((2, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((2, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((2, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((2, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((2, 512, 1, 1), (512, 1, 512, 512), device='cpu', dtype=torch.float32)
    view = rand_strided((2, 512), (512, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((2, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.bool)
    unsqueeze_202 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_214 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_226 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_238 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_250 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((2, 3), (3, 1), device='cpu', dtype=torch.float32)
    return [primals_1, primals_2, primals_6, primals_7, primals_8, primals_12, primals_14, primals_18, primals_20, primals_24, primals_26, primals_30, primals_32, primals_36, primals_38, primals_42, primals_44, primals_48, primals_50, primals_54, primals_56, primals_60, primals_62, primals_66, primals_68, primals_72, primals_74, primals_78, primals_80, primals_84, primals_86, primals_90, primals_92, primals_96, primals_98, primals_102, primals_104, primals_108, primals_110, primals_114, primals_116, primals_120, primals_122, convolution, getitem_1, rsqrt, getitem_2, getitem_3, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, convolution_5, squeeze_16, relu_5, convolution_6, squeeze_19, convolution_7, squeeze_22, relu_6, convolution_8, squeeze_25, relu_7, convolution_9, squeeze_28, relu_8, convolution_10, relu_9, convolution_11, convolution_12, relu_10, convolution_13, relu_11, convolution_14, relu_12, convolution_15, relu_13, convolution_16, convolution_17, relu_14, convolution_18, relu_15, convolution_19, view, le, unsqueeze_202, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, tangents_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cpu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
