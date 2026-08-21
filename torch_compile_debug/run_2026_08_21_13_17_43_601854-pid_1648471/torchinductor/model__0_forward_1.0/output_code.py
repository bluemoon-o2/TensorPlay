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


cpp_fused_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14,
                       float* out_ptr15,
                       float* out_ptr16,
                       float* out_ptr17)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(3L); x1+=static_cast<int64_t>(8L))
            {
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(49L); x2+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0L) && x1 < static_cast<int64_t>(3L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(48L)))
                        {
                            alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                            for (long x1_inner = 0; x1_inner < static_cast<int64_t>(3L); x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + 49L*x1 + 49L*x1_inner + 147L*x0), static_cast<int64_t>(8));
                                tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                            }
                            transpose_mxn<float,static_cast<int64_t>(3L),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr0 + static_cast<int64_t>(x1 + 3L*x2 + 147L*x0), static_cast<int64_t>(3L));
                        }
                        if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0L) && x1 < static_cast<int64_t>(3L) && x2 >= static_cast<int64_t>(48L) && x2 < static_cast<int64_t>(49L)))
                        {
                            alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                            for (long x1_inner = 0; x1_inner < static_cast<int64_t>(3L); x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x2 + 49L*x1 + 49L*x1_inner + 147L*x0), static_cast<int64_t>(1L));
                                tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                            }
                            transpose_mxn<float,static_cast<int64_t>(3L),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr0 + static_cast<int64_t>(x1 + 3L*x2 + 147L*x0), static_cast<int64_t>(3L));
                        }
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(3L); x1+=static_cast<int64_t>(8L))
            {
                for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(1024L); x2+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0L) && x1 < static_cast<int64_t>(3L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(1024L)))
                        {
                            alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                            for (long x1_inner = 0; x1_inner < static_cast<int64_t>(3L); x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x2 + 1024L*x1 + 1024L*x1_inner + 3072L*x0), static_cast<int64_t>(8));
                                tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                            }
                            transpose_mxn<float,static_cast<int64_t>(3L),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr1 + static_cast<int64_t>(x1 + 3L*x2 + 3072L*x0), static_cast<int64_t>(3L));
                        }
                    }
                }
            }
        }
    }
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(64L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 576L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr2 + static_cast<int64_t>(x1 + 64L*x2 + 576L*x0), static_cast<int64_t>(64L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 576L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr2 + static_cast<int64_t>(x1 + 64L*x2 + 576L*x0), static_cast<int64_t>(64L));
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
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 576L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr3 + static_cast<int64_t>(x1 + 64L*x2 + 576L*x0), static_cast<int64_t>(64L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 576L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr3 + static_cast<int64_t>(x1 + 64L*x2 + 576L*x0), static_cast<int64_t>(64L));
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
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 576L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr4 + static_cast<int64_t>(x1 + 64L*x2 + 576L*x0), static_cast<int64_t>(64L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 576L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr4 + static_cast<int64_t>(x1 + 64L*x2 + 576L*x0), static_cast<int64_t>(64L));
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
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 576L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr5 + static_cast<int64_t>(x1 + 64L*x2 + 576L*x0), static_cast<int64_t>(64L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 576L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr5 + static_cast<int64_t>(x1 + 64L*x2 + 576L*x0), static_cast<int64_t>(64L));
                            }
                        }
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
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 576L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr6 + static_cast<int64_t>(x1 + 64L*x2 + 576L*x0), static_cast<int64_t>(64L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 576L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr6 + static_cast<int64_t>(x1 + 64L*x2 + 576L*x0), static_cast<int64_t>(64L));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 1152L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr7 + static_cast<int64_t>(x1 + 128L*x2 + 1152L*x0), static_cast<int64_t>(128L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 1152L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr7 + static_cast<int64_t>(x1 + 128L*x2 + 1152L*x0), static_cast<int64_t>(128L));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 1152L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr8 + static_cast<int64_t>(x1 + 128L*x2 + 1152L*x0), static_cast<int64_t>(128L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 1152L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr8 + static_cast<int64_t>(x1 + 128L*x2 + 1152L*x0), static_cast<int64_t>(128L));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 1152L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr9 + static_cast<int64_t>(x1 + 128L*x2 + 1152L*x0), static_cast<int64_t>(128L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 1152L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr9 + static_cast<int64_t>(x1 + 128L*x2 + 1152L*x0), static_cast<int64_t>(128L));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 1152L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr10 + static_cast<int64_t>(x1 + 128L*x2 + 1152L*x0), static_cast<int64_t>(128L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 1152L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr10 + static_cast<int64_t>(x1 + 128L*x2 + 1152L*x0), static_cast<int64_t>(128L));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(256L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 2304L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr11 + static_cast<int64_t>(x1 + 256L*x2 + 2304L*x0), static_cast<int64_t>(256L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 2304L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr11 + static_cast<int64_t>(x1 + 256L*x2 + 2304L*x0), static_cast<int64_t>(256L));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(256L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 2304L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr12 + static_cast<int64_t>(x1 + 256L*x2 + 2304L*x0), static_cast<int64_t>(256L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 2304L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr12 + static_cast<int64_t>(x1 + 256L*x2 + 2304L*x0), static_cast<int64_t>(256L));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(256L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(256L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 2304L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr13 + static_cast<int64_t>(x1 + 256L*x2 + 2304L*x0), static_cast<int64_t>(256L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 2304L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr13 + static_cast<int64_t>(x1 + 256L*x2 + 2304L*x0), static_cast<int64_t>(256L));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(256L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 2304L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr14 + static_cast<int64_t>(x1 + 256L*x2 + 2304L*x0), static_cast<int64_t>(256L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 2304L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr14 + static_cast<int64_t>(x1 + 256L*x2 + 2304L*x0), static_cast<int64_t>(256L));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 4608L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr15 + static_cast<int64_t>(x1 + 512L*x2 + 4608L*x0), static_cast<int64_t>(512L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 4608L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr15 + static_cast<int64_t>(x1 + 512L*x2 + 4608L*x0), static_cast<int64_t>(512L));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 4608L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr16 + static_cast<int64_t>(x1 + 512L*x2 + 4608L*x0), static_cast<int64_t>(512L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr16 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 4608L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr16 + static_cast<int64_t>(x1 + 512L*x2 + 4608L*x0), static_cast<int64_t>(512L));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(8L))
                {
                    for(int64_t x2=static_cast<int64_t>(0L); x2<static_cast<int64_t>(9L); x2+=static_cast<int64_t>(8L))
                    {
                        {
                            if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512L) && x2 >= static_cast<int64_t>(0) && x2 < static_cast<int64_t>(8L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 4608L*x0), static_cast<int64_t>(8));
                                    tmp0.store(tmp1 + static_cast<int64_t>(8L*x1_inner));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(8),false>(tmp1, static_cast<int64_t>(8), out_ptr17 + static_cast<int64_t>(x1 + 512L*x2 + 4608L*x0), static_cast<int64_t>(512L));
                            }
                            if(C10_UNLIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512L) && x2 >= static_cast<int64_t>(8L) && x2 < static_cast<int64_t>(9L)))
                            {
                                alignas(std::max(std::size_t(8), alignof(float))) float tmp1[8*8];
                                for (long x1_inner = 0; x1_inner < static_cast<int64_t>(8); x1_inner++)
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<int64_t>(x2 + 9L*x1 + 9L*x1_inner + 4608L*x0), static_cast<int64_t>(1L));
                                    tmp0.store(tmp1 + static_cast<int64_t>(x1_inner), static_cast<int64_t>(1L));
                                }
                                transpose_mxn<float,static_cast<int64_t>(8),static_cast<int64_t>(1L),false>(tmp1, static_cast<int64_t>(1L), out_ptr17 + static_cast<int64_t>(x1 + 512L*x2 + 4608L*x0), static_cast<int64_t>(512L));
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


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'int8_t*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       int8_t* out_ptr2)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(512L); x0+=static_cast<int64_t>(1L))
            {
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.sqrt();
                            auto tmp8 = tmp7.reciprocal();
                            auto tmp9 = static_cast<float>(1.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 * tmp10;
                            auto tmp12 = tmp2 * tmp11;
                            auto tmp14 = tmp12 * tmp13;
                            auto tmp16 = tmp14 + tmp15;
                            auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                            tmp17.store(out_ptr0 + static_cast<int64_t>(x1 + 64L*x0));
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
                {
                    #pragma GCC ivdep
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
                                        auto tmp0 = (-1L) + 2L*x1;
                                        auto tmp1 = c10::convert<int64_t>(tmp0);
                                        auto tmp2 = static_cast<int64_t>(0);
                                        auto tmp3 = tmp1 >= tmp2;
                                        auto tmp4 = static_cast<int64_t>(16);
                                        auto tmp5 = tmp1 < tmp4;
                                        auto tmp6 = tmp3 & tmp5;
                                        auto tmp7 = (-1L) + 2L*x2;
                                        auto tmp8 = c10::convert<int64_t>(tmp7);
                                        auto tmp9 = tmp8 >= tmp2;
                                        auto tmp10 = tmp8 < tmp4;
                                        auto tmp11 = tmp9 & tmp10;
                                        auto tmp12 = tmp6 & tmp11;
                                        auto tmp13 = [&]
                                        {
                                            auto tmp14 = at::vec::VecMask<float,1>::from(tmp12).template loadu<float,1>(out_ptr0 + static_cast<int64_t>((-1088L) + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp14;
                                        }
                                        ;
                                        auto tmp15 = tmp12 ? tmp13() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp16 = 2L*x2;
                                        auto tmp17 = c10::convert<int64_t>(tmp16);
                                        auto tmp18 = tmp17 >= tmp2;
                                        auto tmp19 = tmp17 < tmp4;
                                        auto tmp20 = tmp18 & tmp19;
                                        auto tmp21 = tmp6 & tmp20;
                                        auto tmp22 = [&]
                                        {
                                            auto tmp23 = at::vec::VecMask<float,1>::from(tmp21).template loadu<float,1>(out_ptr0 + static_cast<int64_t>((-1024L) + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp23;
                                        }
                                        ;
                                        auto tmp24 = tmp21 ? tmp22() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp25 = at::vec::maximum(tmp15, tmp24);
                                        auto tmp26 = 1L + 2L*x2;
                                        auto tmp27 = c10::convert<int64_t>(tmp26);
                                        auto tmp28 = tmp27 >= tmp2;
                                        auto tmp29 = tmp27 < tmp4;
                                        auto tmp30 = tmp28 & tmp29;
                                        auto tmp31 = tmp6 & tmp30;
                                        auto tmp32 = [&]
                                        {
                                            auto tmp33 = at::vec::VecMask<float,1>::from(tmp31).template loadu<float,1>(out_ptr0 + static_cast<int64_t>((-960L) + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp33;
                                        }
                                        ;
                                        auto tmp34 = tmp31 ? tmp32() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp35 = at::vec::maximum(tmp25, tmp34);
                                        auto tmp36 = 2L*x1;
                                        auto tmp37 = c10::convert<int64_t>(tmp36);
                                        auto tmp38 = tmp37 >= tmp2;
                                        auto tmp39 = tmp37 < tmp4;
                                        auto tmp40 = tmp38 & tmp39;
                                        auto tmp41 = tmp40 & tmp11;
                                        auto tmp42 = [&]
                                        {
                                            auto tmp43 = at::vec::VecMask<float,1>::from(tmp41).template loadu<float,1>(out_ptr0 + static_cast<int64_t>((-64L) + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp43;
                                        }
                                        ;
                                        auto tmp44 = tmp41 ? tmp42() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp45 = at::vec::maximum(tmp35, tmp44);
                                        auto tmp46 = tmp40 & tmp20;
                                        auto tmp47 = [&]
                                        {
                                            auto tmp48 = at::vec::VecMask<float,1>::from(tmp46).template loadu<float,1>(out_ptr0 + static_cast<int64_t>(x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp48;
                                        }
                                        ;
                                        auto tmp49 = tmp46 ? tmp47() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp50 = at::vec::maximum(tmp45, tmp49);
                                        auto tmp51 = tmp40 & tmp30;
                                        auto tmp52 = [&]
                                        {
                                            auto tmp53 = at::vec::VecMask<float,1>::from(tmp51).template loadu<float,1>(out_ptr0 + static_cast<int64_t>(64L + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp53;
                                        }
                                        ;
                                        auto tmp54 = tmp51 ? tmp52() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp55 = at::vec::maximum(tmp50, tmp54);
                                        auto tmp56 = 1L + 2L*x1;
                                        auto tmp57 = c10::convert<int64_t>(tmp56);
                                        auto tmp58 = tmp57 >= tmp2;
                                        auto tmp59 = tmp57 < tmp4;
                                        auto tmp60 = tmp58 & tmp59;
                                        auto tmp61 = tmp60 & tmp11;
                                        auto tmp62 = [&]
                                        {
                                            auto tmp63 = at::vec::VecMask<float,1>::from(tmp61).template loadu<float,1>(out_ptr0 + static_cast<int64_t>(960L + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp63;
                                        }
                                        ;
                                        auto tmp64 = tmp61 ? tmp62() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp65 = at::vec::maximum(tmp55, tmp64);
                                        auto tmp66 = tmp60 & tmp20;
                                        auto tmp67 = [&]
                                        {
                                            auto tmp68 = at::vec::VecMask<float,1>::from(tmp66).template loadu<float,1>(out_ptr0 + static_cast<int64_t>(1024L + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp68;
                                        }
                                        ;
                                        auto tmp69 = tmp66 ? tmp67() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp70 = at::vec::maximum(tmp65, tmp69);
                                        auto tmp71 = tmp60 & tmp30;
                                        auto tmp72 = [&]
                                        {
                                            auto tmp73 = at::vec::VecMask<float,1>::from(tmp71).template loadu<float,1>(out_ptr0 + static_cast<int64_t>(1088L + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp73;
                                        }
                                        ;
                                        auto tmp74 = tmp71 ? tmp72() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp75 = at::vec::maximum(tmp70, tmp74);
                                        auto tmp76 = [&]
                                        {
                                            auto tmp77 = at::vec::VecMask<float,1>::from(tmp12).template loadu<float,1>(out_ptr0 + static_cast<int64_t>((-1088L) + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp77;
                                        }
                                        ;
                                        auto tmp78 = tmp12 ? tmp76() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp79 = [&]
                                        {
                                            auto tmp80 = at::vec::VecMask<float,1>::from(tmp21).template loadu<float,1>(out_ptr0 + static_cast<int64_t>((-1024L) + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp80;
                                        }
                                        ;
                                        auto tmp81 = tmp21 ? tmp79() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp82 = at::vec::VecMask<float,1>(tmp78 > tmp81);
                                        auto tmp83 = at::vec::VecMask<float,1>(tmp78 == tmp81);
                                        auto tmp84 = at::vec::VecMask<float,1>(tmp78 != tmp78);
                                        auto tmp85 = at::vec::VecMask<float,1>(tmp81 != tmp81);
                                        auto tmp86 = (tmp84 > tmp85);
                                        auto tmp87 = tmp82.template cast<int32_t,1>();
                                        auto tmp88 = tmp86.template cast<int32_t,1>();
                                        auto tmp89 = tmp87 | tmp88;
                                        auto tmp90 = tmp84.template cast<int32_t,1>();
                                        auto tmp91 = tmp85.template cast<int32_t,1>();
                                        auto tmp92 = tmp90 & tmp91;
                                        auto tmp93 = tmp83.template cast<int32_t,1>();
                                        auto tmp94 = tmp92.template cast<int32_t,1>();
                                        auto tmp95 = tmp93 | tmp94;
                                        auto tmp96 = static_cast<int64_t>(1);
                                        auto tmp97 = tmp2 < tmp96;
                                        auto tmp98 = at::vec::VecMask<float,1>::from(tmp97);
                                        auto tmp99 = tmp95.template cast<int32_t,1>();
                                        auto tmp100 = tmp98.template cast<int32_t,1>();
                                        auto tmp101 = tmp99 & tmp100;
                                        auto tmp102 = tmp89.template cast<int32_t,1>();
                                        auto tmp103 = tmp101.template cast<int32_t,1>();
                                        auto tmp104 = tmp102 | tmp103;
                                        auto tmp105 = decltype(tmp78)::blendv(tmp81, tmp78, tmp104.template cast<float,1>());
                                        auto tmp106 = at::vec::VectorizedN<int64_t,2>(tmp2);
                                        auto tmp107 = at::vec::VectorizedN<int64_t,2>(tmp96);
                                        auto tmp108 = decltype(tmp106)::blendv(tmp107, tmp106, tmp104.template cast<int64_t,2>());
                                        auto tmp109 = [&]
                                        {
                                            auto tmp110 = at::vec::VecMask<float,1>::from(tmp31).template loadu<float,1>(out_ptr0 + static_cast<int64_t>((-960L) + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp110;
                                        }
                                        ;
                                        auto tmp111 = tmp31 ? tmp109() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp112 = at::vec::VecMask<float,1>(tmp105 > tmp111);
                                        auto tmp113 = at::vec::VecMask<float,1>(tmp105 == tmp111);
                                        auto tmp114 = at::vec::VecMask<float,1>(tmp105 != tmp105);
                                        auto tmp115 = at::vec::VecMask<float,1>(tmp111 != tmp111);
                                        auto tmp116 = (tmp114 > tmp115);
                                        auto tmp117 = tmp112.template cast<int32_t,1>();
                                        auto tmp118 = tmp116.template cast<int32_t,1>();
                                        auto tmp119 = tmp117 | tmp118;
                                        auto tmp120 = tmp114.template cast<int32_t,1>();
                                        auto tmp121 = tmp115.template cast<int32_t,1>();
                                        auto tmp122 = tmp120 & tmp121;
                                        auto tmp123 = tmp113.template cast<int32_t,1>();
                                        auto tmp124 = tmp122.template cast<int32_t,1>();
                                        auto tmp125 = tmp123 | tmp124;
                                        auto tmp126 = static_cast<int64_t>(2);
                                        auto tmp127 = at::vec::VectorizedN<int64_t,2>(tmp126);
                                        auto tmp128 = at::vec::VecMask<int64_t,2>(tmp108 < tmp127);
                                        auto tmp129 = tmp125.template cast<int32_t,1>();
                                        auto tmp130 = tmp128.template cast<int32_t,1>();
                                        auto tmp131 = tmp129 & tmp130;
                                        auto tmp132 = tmp119.template cast<int32_t,1>();
                                        auto tmp133 = tmp131.template cast<int32_t,1>();
                                        auto tmp134 = tmp132 | tmp133;
                                        auto tmp135 = decltype(tmp105)::blendv(tmp111, tmp105, tmp134.template cast<float,1>());
                                        auto tmp136 = decltype(tmp108)::blendv(tmp127, tmp108, tmp134.template cast<int64_t,2>());
                                        auto tmp137 = [&]
                                        {
                                            auto tmp138 = at::vec::VecMask<float,1>::from(tmp41).template loadu<float,1>(out_ptr0 + static_cast<int64_t>((-64L) + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp138;
                                        }
                                        ;
                                        auto tmp139 = tmp41 ? tmp137() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp140 = at::vec::VecMask<float,1>(tmp135 > tmp139);
                                        auto tmp141 = at::vec::VecMask<float,1>(tmp135 == tmp139);
                                        auto tmp142 = at::vec::VecMask<float,1>(tmp135 != tmp135);
                                        auto tmp143 = at::vec::VecMask<float,1>(tmp139 != tmp139);
                                        auto tmp144 = (tmp142 > tmp143);
                                        auto tmp145 = tmp140.template cast<int32_t,1>();
                                        auto tmp146 = tmp144.template cast<int32_t,1>();
                                        auto tmp147 = tmp145 | tmp146;
                                        auto tmp148 = tmp142.template cast<int32_t,1>();
                                        auto tmp149 = tmp143.template cast<int32_t,1>();
                                        auto tmp150 = tmp148 & tmp149;
                                        auto tmp151 = tmp141.template cast<int32_t,1>();
                                        auto tmp152 = tmp150.template cast<int32_t,1>();
                                        auto tmp153 = tmp151 | tmp152;
                                        auto tmp154 = static_cast<int64_t>(3);
                                        auto tmp155 = at::vec::VectorizedN<int64_t,2>(tmp154);
                                        auto tmp156 = at::vec::VecMask<int64_t,2>(tmp136 < tmp155);
                                        auto tmp157 = tmp153.template cast<int32_t,1>();
                                        auto tmp158 = tmp156.template cast<int32_t,1>();
                                        auto tmp159 = tmp157 & tmp158;
                                        auto tmp160 = tmp147.template cast<int32_t,1>();
                                        auto tmp161 = tmp159.template cast<int32_t,1>();
                                        auto tmp162 = tmp160 | tmp161;
                                        auto tmp163 = decltype(tmp135)::blendv(tmp139, tmp135, tmp162.template cast<float,1>());
                                        auto tmp164 = decltype(tmp136)::blendv(tmp155, tmp136, tmp162.template cast<int64_t,2>());
                                        auto tmp165 = [&]
                                        {
                                            auto tmp166 = at::vec::VecMask<float,1>::from(tmp46).template loadu<float,1>(out_ptr0 + static_cast<int64_t>(x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp166;
                                        }
                                        ;
                                        auto tmp167 = tmp46 ? tmp165() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp168 = at::vec::VecMask<float,1>(tmp163 > tmp167);
                                        auto tmp169 = at::vec::VecMask<float,1>(tmp163 == tmp167);
                                        auto tmp170 = at::vec::VecMask<float,1>(tmp163 != tmp163);
                                        auto tmp171 = at::vec::VecMask<float,1>(tmp167 != tmp167);
                                        auto tmp172 = (tmp170 > tmp171);
                                        auto tmp173 = tmp168.template cast<int32_t,1>();
                                        auto tmp174 = tmp172.template cast<int32_t,1>();
                                        auto tmp175 = tmp173 | tmp174;
                                        auto tmp176 = tmp170.template cast<int32_t,1>();
                                        auto tmp177 = tmp171.template cast<int32_t,1>();
                                        auto tmp178 = tmp176 & tmp177;
                                        auto tmp179 = tmp169.template cast<int32_t,1>();
                                        auto tmp180 = tmp178.template cast<int32_t,1>();
                                        auto tmp181 = tmp179 | tmp180;
                                        auto tmp182 = static_cast<int64_t>(4);
                                        auto tmp183 = at::vec::VectorizedN<int64_t,2>(tmp182);
                                        auto tmp184 = at::vec::VecMask<int64_t,2>(tmp164 < tmp183);
                                        auto tmp185 = tmp181.template cast<int32_t,1>();
                                        auto tmp186 = tmp184.template cast<int32_t,1>();
                                        auto tmp187 = tmp185 & tmp186;
                                        auto tmp188 = tmp175.template cast<int32_t,1>();
                                        auto tmp189 = tmp187.template cast<int32_t,1>();
                                        auto tmp190 = tmp188 | tmp189;
                                        auto tmp191 = decltype(tmp163)::blendv(tmp167, tmp163, tmp190.template cast<float,1>());
                                        auto tmp192 = decltype(tmp164)::blendv(tmp183, tmp164, tmp190.template cast<int64_t,2>());
                                        auto tmp193 = [&]
                                        {
                                            auto tmp194 = at::vec::VecMask<float,1>::from(tmp51).template loadu<float,1>(out_ptr0 + static_cast<int64_t>(64L + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp194;
                                        }
                                        ;
                                        auto tmp195 = tmp51 ? tmp193() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp196 = at::vec::VecMask<float,1>(tmp191 > tmp195);
                                        auto tmp197 = at::vec::VecMask<float,1>(tmp191 == tmp195);
                                        auto tmp198 = at::vec::VecMask<float,1>(tmp191 != tmp191);
                                        auto tmp199 = at::vec::VecMask<float,1>(tmp195 != tmp195);
                                        auto tmp200 = (tmp198 > tmp199);
                                        auto tmp201 = tmp196.template cast<int32_t,1>();
                                        auto tmp202 = tmp200.template cast<int32_t,1>();
                                        auto tmp203 = tmp201 | tmp202;
                                        auto tmp204 = tmp198.template cast<int32_t,1>();
                                        auto tmp205 = tmp199.template cast<int32_t,1>();
                                        auto tmp206 = tmp204 & tmp205;
                                        auto tmp207 = tmp197.template cast<int32_t,1>();
                                        auto tmp208 = tmp206.template cast<int32_t,1>();
                                        auto tmp209 = tmp207 | tmp208;
                                        auto tmp210 = static_cast<int64_t>(5);
                                        auto tmp211 = at::vec::VectorizedN<int64_t,2>(tmp210);
                                        auto tmp212 = at::vec::VecMask<int64_t,2>(tmp192 < tmp211);
                                        auto tmp213 = tmp209.template cast<int32_t,1>();
                                        auto tmp214 = tmp212.template cast<int32_t,1>();
                                        auto tmp215 = tmp213 & tmp214;
                                        auto tmp216 = tmp203.template cast<int32_t,1>();
                                        auto tmp217 = tmp215.template cast<int32_t,1>();
                                        auto tmp218 = tmp216 | tmp217;
                                        auto tmp219 = decltype(tmp191)::blendv(tmp195, tmp191, tmp218.template cast<float,1>());
                                        auto tmp220 = decltype(tmp192)::blendv(tmp211, tmp192, tmp218.template cast<int64_t,2>());
                                        auto tmp221 = [&]
                                        {
                                            auto tmp222 = at::vec::VecMask<float,1>::from(tmp61).template loadu<float,1>(out_ptr0 + static_cast<int64_t>(960L + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp222;
                                        }
                                        ;
                                        auto tmp223 = tmp61 ? tmp221() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp224 = at::vec::VecMask<float,1>(tmp219 > tmp223);
                                        auto tmp225 = at::vec::VecMask<float,1>(tmp219 == tmp223);
                                        auto tmp226 = at::vec::VecMask<float,1>(tmp219 != tmp219);
                                        auto tmp227 = at::vec::VecMask<float,1>(tmp223 != tmp223);
                                        auto tmp228 = (tmp226 > tmp227);
                                        auto tmp229 = tmp224.template cast<int32_t,1>();
                                        auto tmp230 = tmp228.template cast<int32_t,1>();
                                        auto tmp231 = tmp229 | tmp230;
                                        auto tmp232 = tmp226.template cast<int32_t,1>();
                                        auto tmp233 = tmp227.template cast<int32_t,1>();
                                        auto tmp234 = tmp232 & tmp233;
                                        auto tmp235 = tmp225.template cast<int32_t,1>();
                                        auto tmp236 = tmp234.template cast<int32_t,1>();
                                        auto tmp237 = tmp235 | tmp236;
                                        auto tmp238 = static_cast<int64_t>(6);
                                        auto tmp239 = at::vec::VectorizedN<int64_t,2>(tmp238);
                                        auto tmp240 = at::vec::VecMask<int64_t,2>(tmp220 < tmp239);
                                        auto tmp241 = tmp237.template cast<int32_t,1>();
                                        auto tmp242 = tmp240.template cast<int32_t,1>();
                                        auto tmp243 = tmp241 & tmp242;
                                        auto tmp244 = tmp231.template cast<int32_t,1>();
                                        auto tmp245 = tmp243.template cast<int32_t,1>();
                                        auto tmp246 = tmp244 | tmp245;
                                        auto tmp247 = decltype(tmp219)::blendv(tmp223, tmp219, tmp246.template cast<float,1>());
                                        auto tmp248 = decltype(tmp220)::blendv(tmp239, tmp220, tmp246.template cast<int64_t,2>());
                                        auto tmp249 = [&]
                                        {
                                            auto tmp250 = at::vec::VecMask<float,1>::from(tmp66).template loadu<float,1>(out_ptr0 + static_cast<int64_t>(1024L + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp250;
                                        }
                                        ;
                                        auto tmp251 = tmp66 ? tmp249() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp252 = at::vec::VecMask<float,1>(tmp247 > tmp251);
                                        auto tmp253 = at::vec::VecMask<float,1>(tmp247 == tmp251);
                                        auto tmp254 = at::vec::VecMask<float,1>(tmp247 != tmp247);
                                        auto tmp255 = at::vec::VecMask<float,1>(tmp251 != tmp251);
                                        auto tmp256 = (tmp254 > tmp255);
                                        auto tmp257 = tmp252.template cast<int32_t,1>();
                                        auto tmp258 = tmp256.template cast<int32_t,1>();
                                        auto tmp259 = tmp257 | tmp258;
                                        auto tmp260 = tmp254.template cast<int32_t,1>();
                                        auto tmp261 = tmp255.template cast<int32_t,1>();
                                        auto tmp262 = tmp260 & tmp261;
                                        auto tmp263 = tmp253.template cast<int32_t,1>();
                                        auto tmp264 = tmp262.template cast<int32_t,1>();
                                        auto tmp265 = tmp263 | tmp264;
                                        auto tmp266 = static_cast<int64_t>(7);
                                        auto tmp267 = at::vec::VectorizedN<int64_t,2>(tmp266);
                                        auto tmp268 = at::vec::VecMask<int64_t,2>(tmp248 < tmp267);
                                        auto tmp269 = tmp265.template cast<int32_t,1>();
                                        auto tmp270 = tmp268.template cast<int32_t,1>();
                                        auto tmp271 = tmp269 & tmp270;
                                        auto tmp272 = tmp259.template cast<int32_t,1>();
                                        auto tmp273 = tmp271.template cast<int32_t,1>();
                                        auto tmp274 = tmp272 | tmp273;
                                        auto tmp275 = decltype(tmp247)::blendv(tmp251, tmp247, tmp274.template cast<float,1>());
                                        auto tmp276 = decltype(tmp248)::blendv(tmp267, tmp248, tmp274.template cast<int64_t,2>());
                                        auto tmp277 = [&]
                                        {
                                            auto tmp278 = at::vec::VecMask<float,1>::from(tmp71).template loadu<float,1>(out_ptr0 + static_cast<int64_t>(1088L + x3 + 128L*x2 + 2048L*x1 + 16384L*x0));
                                            return tmp278;
                                        }
                                        ;
                                        auto tmp279 = tmp71 ? tmp277() : at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                                        auto tmp280 = at::vec::VecMask<float,1>(tmp275 > tmp279);
                                        auto tmp281 = at::vec::VecMask<float,1>(tmp275 == tmp279);
                                        auto tmp282 = at::vec::VecMask<float,1>(tmp275 != tmp275);
                                        auto tmp283 = at::vec::VecMask<float,1>(tmp279 != tmp279);
                                        auto tmp284 = (tmp282 > tmp283);
                                        auto tmp285 = tmp280.template cast<int32_t,1>();
                                        auto tmp286 = tmp284.template cast<int32_t,1>();
                                        auto tmp287 = tmp285 | tmp286;
                                        auto tmp288 = tmp282.template cast<int32_t,1>();
                                        auto tmp289 = tmp283.template cast<int32_t,1>();
                                        auto tmp290 = tmp288 & tmp289;
                                        auto tmp291 = tmp281.template cast<int32_t,1>();
                                        auto tmp292 = tmp290.template cast<int32_t,1>();
                                        auto tmp293 = tmp291 | tmp292;
                                        auto tmp294 = static_cast<int64_t>(8);
                                        auto tmp295 = at::vec::VectorizedN<int64_t,2>(tmp294);
                                        auto tmp296 = at::vec::VecMask<int64_t,2>(tmp276 < tmp295);
                                        auto tmp297 = tmp293.template cast<int32_t,1>();
                                        auto tmp298 = tmp296.template cast<int32_t,1>();
                                        auto tmp299 = tmp297 & tmp298;
                                        auto tmp300 = tmp287.template cast<int32_t,1>();
                                        auto tmp301 = tmp299.template cast<int32_t,1>();
                                        auto tmp302 = tmp300 | tmp301;
                                        auto tmp303 = decltype(tmp275)::blendv(tmp279, tmp275, tmp302.template cast<float,1>());
                                        auto tmp304 = decltype(tmp276)::blendv(tmp295, tmp276, tmp302.template cast<int64_t,2>());
                                        auto tmp305 = at::vec::convert<int8_t,1,int64_t,2>(tmp304);
                                        tmp75.store(out_ptr1 + static_cast<int64_t>(x3 + 64L*x2 + 512L*x1 + 4096L*x0));
                                        tmp305.store(out_ptr2 + static_cast<int64_t>(x3 + 64L*x2 + 512L*x1 + 4096L*x0), static_cast<int64_t>(8));
                                    }
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


cpp_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        tmp17.store(out_ptr0 + static_cast<int64_t>(x1 + 64L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_3 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(128L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(64L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(64L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + 64L*x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp18 = tmp16 + tmp17;
                        auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                        tmp19.store(out_ptr0 + static_cast<int64_t>(x1 + 64L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(32L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        tmp17.store(out_ptr0 + static_cast<int64_t>(x1 + 128L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_5 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(32L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp21 = tmp20 + tmp5;
                        auto tmp22 = tmp21.sqrt();
                        auto tmp23 = tmp22.reciprocal();
                        auto tmp24 = tmp23 * tmp10;
                        auto tmp25 = tmp19 * tmp24;
                        auto tmp27 = tmp25 * tmp26;
                        auto tmp29 = tmp27 + tmp28;
                        auto tmp30 = tmp16 + tmp29;
                        auto tmp31 = at::vec::clamp_min(tmp30, decltype(tmp30)(0));
                        tmp31.store(out_ptr0 + static_cast<int64_t>(x1 + 128L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(32L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(128L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(128L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + 128L*x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp18 = tmp16 + tmp17;
                        auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                        tmp19.store(out_ptr0 + static_cast<int64_t>(x1 + 128L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(256L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        tmp17.store(out_ptr0 + static_cast<int64_t>(x1 + 256L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_8 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(256L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp21 = tmp20 + tmp5;
                        auto tmp22 = tmp21.sqrt();
                        auto tmp23 = tmp22.reciprocal();
                        auto tmp24 = tmp23 * tmp10;
                        auto tmp25 = tmp19 * tmp24;
                        auto tmp27 = tmp25 * tmp26;
                        auto tmp29 = tmp27 + tmp28;
                        auto tmp30 = tmp16 + tmp29;
                        auto tmp31 = at::vec::clamp_min(tmp30, decltype(tmp30)(0));
                        tmp31.store(out_ptr0 + static_cast<int64_t>(x1 + 256L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_9 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(256L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(256L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + 256L*x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp18 = tmp16 + tmp17;
                        auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                        tmp19.store(out_ptr0 + static_cast<int64_t>(x1 + 256L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                        tmp17.store(out_ptr0 + static_cast<int64_t>(x1 + 512L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp21 = tmp20 + tmp5;
                        auto tmp22 = tmp21.sqrt();
                        auto tmp23 = tmp22.reciprocal();
                        auto tmp24 = tmp23 * tmp10;
                        auto tmp25 = tmp19 * tmp24;
                        auto tmp27 = tmp25 * tmp26;
                        auto tmp29 = tmp27 + tmp28;
                        auto tmp30 = tmp16 + tmp29;
                        auto tmp31 = at::vec::clamp_min(tmp30, decltype(tmp30)(0));
                        tmp31.store(out_ptr0 + static_cast<int64_t>(x1 + 512L*x0));
                    }
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_12 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'const float*', 'float*', 'bool*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr1,
                       bool* out_ptr2)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(512L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(512L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(1e-05);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.sqrt();
                        auto tmp8 = tmp7.reciprocal();
                        auto tmp9 = static_cast<float>(1.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 * tmp10;
                        auto tmp12 = tmp2 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp18 = tmp16 + tmp17;
                        auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                        auto tmp20 = tmp19 / tmp10;
                        auto tmp21 = static_cast<float>(0.0);
                        auto tmp22 = at::vec::Vectorized<float>(tmp21);
                        auto tmp23 = at::vec::VecMask<float,1>(tmp19 <= tmp22);
                        tmp20.store(out_ptr1 + static_cast<int64_t>(x1 + 512L*x0));
                        tmp23.store(out_ptr2 + static_cast<int64_t>(x1 + 512L*x0), static_cast<int64_t>(8));
                    }
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103 = args
        args.clear()
        assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1), 'input')
        buf0 = empty_strided_cpu((64, 3, 7, 7), (147, 1, 21, 3), torch.float32)
        assert_size_stride(primals_2, (2, 3, 32, 32), (3072, 1024, 32, 1), 'input')
        buf1 = empty_strided_cpu((2, 3, 32, 32), (3072, 1, 96, 3), torch.float32)
        assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1), 'input')
        buf2 = empty_strided_cpu((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        assert_size_stride(primals_12, (64, 64, 3, 3), (576, 9, 3, 1), 'input')
        buf3 = empty_strided_cpu((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        assert_size_stride(primals_17, (64, 64, 3, 3), (576, 9, 3, 1), 'input')
        buf4 = empty_strided_cpu((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        assert_size_stride(primals_22, (64, 64, 3, 3), (576, 9, 3, 1), 'input')
        buf5 = empty_strided_cpu((64, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        assert_size_stride(primals_27, (128, 64, 3, 3), (576, 9, 3, 1), 'input')
        buf6 = empty_strided_cpu((128, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        assert_size_stride(primals_32, (128, 128, 3, 3), (1152, 9, 3, 1), 'input')
        buf7 = empty_strided_cpu((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        assert_size_stride(primals_42, (128, 128, 3, 3), (1152, 9, 3, 1), 'input')
        buf8 = empty_strided_cpu((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        assert_size_stride(primals_47, (128, 128, 3, 3), (1152, 9, 3, 1), 'input')
        buf9 = empty_strided_cpu((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        assert_size_stride(primals_52, (256, 128, 3, 3), (1152, 9, 3, 1), 'input')
        buf10 = empty_strided_cpu((256, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        assert_size_stride(primals_57, (256, 256, 3, 3), (2304, 9, 3, 1), 'input')
        buf11 = empty_strided_cpu((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        assert_size_stride(primals_67, (256, 256, 3, 3), (2304, 9, 3, 1), 'input')
        buf12 = empty_strided_cpu((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        assert_size_stride(primals_72, (256, 256, 3, 3), (2304, 9, 3, 1), 'input')
        buf13 = empty_strided_cpu((256, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        assert_size_stride(primals_77, (512, 256, 3, 3), (2304, 9, 3, 1), 'input')
        buf14 = empty_strided_cpu((512, 256, 3, 3), (2304, 1, 768, 256), torch.float32)
        assert_size_stride(primals_82, (512, 512, 3, 3), (4608, 9, 3, 1), 'input')
        buf15 = empty_strided_cpu((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        assert_size_stride(primals_92, (512, 512, 3, 3), (4608, 9, 3, 1), 'input')
        buf16 = empty_strided_cpu((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        assert_size_stride(primals_97, (512, 512, 3, 3), (4608, 9, 3, 1), 'input')
        buf17 = empty_strided_cpu((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # [Provenance debug handles] cpp_fused_0:1
        cpp_fused_0(primals_1, primals_2, primals_7, primals_12, primals_17, primals_22, primals_27, primals_32, primals_42, primals_47, primals_52, primals_57, primals_67, primals_72, primals_77, primals_82, primals_92, primals_97, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17)
        del primals_1
        del primals_12
        del primals_17
        del primals_2
        del primals_22
        del primals_27
        del primals_32
        del primals_42
        del primals_47
        del primals_52
        del primals_57
        del primals_67
        del primals_7
        del primals_72
        del primals_77
        del primals_82
        del primals_92
        del primals_97
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:2
        buf18 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (2, 64, 16, 16), (16384, 1, 1024, 64), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_3, (64, ), (1, ), 'input')
        assert_size_stride(primals_4, (64, ), (1, ), 'input')
        assert_size_stride(primals_5, (64, ), (1, ), 'input')
        assert_size_stride(primals_6, (64, ), (1, ), 'input')
        buf19 = empty_strided_cpu((2, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf20 = empty_strided_cpu((2, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        buf21 = empty_strided_cpu((2, 64, 8, 8), (4096, 1, 512, 64), torch.int8)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1:3
        cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1(buf18, primals_3, primals_4, primals_5, primals_6, buf19, buf20, buf21)
        del buf19
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:4
        buf22 = extern_kernels.convolution(buf20, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_8, (64, ), (1, ), 'input')
        assert_size_stride(primals_9, (64, ), (1, ), 'input')
        assert_size_stride(primals_10, (64, ), (1, ), 'input')
        assert_size_stride(primals_11, (64, ), (1, ), 'input')
        buf23 = empty_strided_cpu((2, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_relu_2:5
        cpp_fused__native_batch_norm_legit_no_training_relu_2(buf22, primals_8, primals_9, primals_10, primals_11, buf23)
        del primals_11
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:6
        buf24 = extern_kernels.convolution(buf23, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_13, (64, ), (1, ), 'input')
        assert_size_stride(primals_14, (64, ), (1, ), 'input')
        assert_size_stride(primals_15, (64, ), (1, ), 'input')
        assert_size_stride(primals_16, (64, ), (1, ), 'input')
        buf25 = empty_strided_cpu((2, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_add_relu_3:7
        cpp_fused__native_batch_norm_legit_no_training_add_relu_3(buf24, primals_13, primals_14, primals_15, primals_16, buf20, buf25)
        del primals_16
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:8
        buf26 = extern_kernels.convolution(buf25, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_18, (64, ), (1, ), 'input')
        assert_size_stride(primals_19, (64, ), (1, ), 'input')
        assert_size_stride(primals_20, (64, ), (1, ), 'input')
        assert_size_stride(primals_21, (64, ), (1, ), 'input')
        buf27 = empty_strided_cpu((2, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_relu_2:9
        cpp_fused__native_batch_norm_legit_no_training_relu_2(buf26, primals_18, primals_19, primals_20, primals_21, buf27)
        del primals_21
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:10
        buf28 = extern_kernels.convolution(buf27, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (2, 64, 8, 8), (4096, 1, 512, 64), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_23, (64, ), (1, ), 'input')
        assert_size_stride(primals_24, (64, ), (1, ), 'input')
        assert_size_stride(primals_25, (64, ), (1, ), 'input')
        assert_size_stride(primals_26, (64, ), (1, ), 'input')
        buf29 = empty_strided_cpu((2, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_add_relu_3:11
        cpp_fused__native_batch_norm_legit_no_training_add_relu_3(buf28, primals_23, primals_24, primals_25, primals_26, buf25, buf29)
        del primals_26
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:12
        buf30 = extern_kernels.convolution(buf29, buf6, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_28, (128, ), (1, ), 'input')
        assert_size_stride(primals_29, (128, ), (1, ), 'input')
        assert_size_stride(primals_30, (128, ), (1, ), 'input')
        assert_size_stride(primals_31, (128, ), (1, ), 'input')
        buf31 = empty_strided_cpu((2, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_relu_4:13
        cpp_fused__native_batch_norm_legit_no_training_relu_4(buf30, primals_28, primals_29, primals_30, primals_31, buf31)
        del primals_31
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:14
        buf32 = extern_kernels.convolution(buf31, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_37, (128, 64, 1, 1), (64, 1, 1, 1), 'input')
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:15
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
        buf34 = empty_strided_cpu((2, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_add_relu_5:16
        cpp_fused__native_batch_norm_legit_no_training_add_relu_5(buf32, primals_33, primals_34, primals_35, primals_36, buf33, primals_38, primals_39, primals_40, primals_41, buf34)
        del primals_36
        del primals_41
        # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:17
        buf35 = extern_kernels.convolution(buf34, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_43, (128, ), (1, ), 'input')
        assert_size_stride(primals_44, (128, ), (1, ), 'input')
        assert_size_stride(primals_45, (128, ), (1, ), 'input')
        assert_size_stride(primals_46, (128, ), (1, ), 'input')
        buf36 = empty_strided_cpu((2, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_relu_4:18
        cpp_fused__native_batch_norm_legit_no_training_relu_4(buf35, primals_43, primals_44, primals_45, primals_46, buf36)
        del primals_46
        # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:19
        buf37 = extern_kernels.convolution(buf36, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (2, 128, 4, 4), (2048, 1, 512, 128), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_48, (128, ), (1, ), 'input')
        assert_size_stride(primals_49, (128, ), (1, ), 'input')
        assert_size_stride(primals_50, (128, ), (1, ), 'input')
        assert_size_stride(primals_51, (128, ), (1, ), 'input')
        buf38 = empty_strided_cpu((2, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_add_relu_6:20
        cpp_fused__native_batch_norm_legit_no_training_add_relu_6(buf37, primals_48, primals_49, primals_50, primals_51, buf34, buf38)
        del primals_51
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:21
        buf39 = extern_kernels.convolution(buf38, buf10, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_53, (256, ), (1, ), 'input')
        assert_size_stride(primals_54, (256, ), (1, ), 'input')
        assert_size_stride(primals_55, (256, ), (1, ), 'input')
        assert_size_stride(primals_56, (256, ), (1, ), 'input')
        buf40 = empty_strided_cpu((2, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_relu_7:22
        cpp_fused__native_batch_norm_legit_no_training_relu_7(buf39, primals_53, primals_54, primals_55, primals_56, buf40)
        del primals_56
        # Topologically Sorted Source Nodes: [out_31], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:23
        buf41 = extern_kernels.convolution(buf40, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_62, (256, 128, 1, 1), (128, 1, 1, 1), 'input')
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:24
        buf42 = extern_kernels.convolution(buf38, reinterpret_tensor(primals_62, (256, 128, 1, 1), (128, 1, 128, 128), 0), stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_58, (256, ), (1, ), 'input')
        assert_size_stride(primals_59, (256, ), (1, ), 'input')
        assert_size_stride(primals_60, (256, ), (1, ), 'input')
        assert_size_stride(primals_61, (256, ), (1, ), 'input')
        assert_size_stride(primals_63, (256, ), (1, ), 'input')
        assert_size_stride(primals_64, (256, ), (1, ), 'input')
        assert_size_stride(primals_65, (256, ), (1, ), 'input')
        assert_size_stride(primals_66, (256, ), (1, ), 'input')
        buf43 = empty_strided_cpu((2, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_add_relu_8:25
        cpp_fused__native_batch_norm_legit_no_training_add_relu_8(buf41, primals_58, primals_59, primals_60, primals_61, buf42, primals_63, primals_64, primals_65, primals_66, buf43)
        del primals_61
        del primals_66
        # Topologically Sorted Source Nodes: [out_35], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:26
        buf44 = extern_kernels.convolution(buf43, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_68, (256, ), (1, ), 'input')
        assert_size_stride(primals_69, (256, ), (1, ), 'input')
        assert_size_stride(primals_70, (256, ), (1, ), 'input')
        assert_size_stride(primals_71, (256, ), (1, ), 'input')
        buf45 = empty_strided_cpu((2, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_relu_7:27
        cpp_fused__native_batch_norm_legit_no_training_relu_7(buf44, primals_68, primals_69, primals_70, primals_71, buf45)
        del primals_71
        # Topologically Sorted Source Nodes: [out_38], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:28
        buf46 = extern_kernels.convolution(buf45, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (2, 256, 2, 2), (1024, 1, 512, 256), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_73, (256, ), (1, ), 'input')
        assert_size_stride(primals_74, (256, ), (1, ), 'input')
        assert_size_stride(primals_75, (256, ), (1, ), 'input')
        assert_size_stride(primals_76, (256, ), (1, ), 'input')
        buf47 = empty_strided_cpu((2, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_add_relu_9:29
        cpp_fused__native_batch_norm_legit_no_training_add_relu_9(buf46, primals_73, primals_74, primals_75, primals_76, buf43, buf47)
        del primals_76
        # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:30
        buf48 = extern_kernels.convolution(buf47, buf14, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_78, (512, ), (1, ), 'input')
        assert_size_stride(primals_79, (512, ), (1, ), 'input')
        assert_size_stride(primals_80, (512, ), (1, ), 'input')
        assert_size_stride(primals_81, (512, ), (1, ), 'input')
        buf49 = empty_strided_cpu((2, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_relu_10:31
        cpp_fused__native_batch_norm_legit_no_training_relu_10(buf48, primals_78, primals_79, primals_80, primals_81, buf49)
        del primals_81
        # Topologically Sorted Source Nodes: [out_45], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:32
        buf50 = extern_kernels.convolution(buf49, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_87, (512, 256, 1, 1), (256, 1, 1, 1), 'input')
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:33
        buf51 = extern_kernels.convolution(buf47, reinterpret_tensor(primals_87, (512, 256, 1, 1), (256, 1, 256, 256), 0), stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_83, (512, ), (1, ), 'input')
        assert_size_stride(primals_84, (512, ), (1, ), 'input')
        assert_size_stride(primals_85, (512, ), (1, ), 'input')
        assert_size_stride(primals_86, (512, ), (1, ), 'input')
        assert_size_stride(primals_88, (512, ), (1, ), 'input')
        assert_size_stride(primals_89, (512, ), (1, ), 'input')
        assert_size_stride(primals_90, (512, ), (1, ), 'input')
        assert_size_stride(primals_91, (512, ), (1, ), 'input')
        buf52 = empty_strided_cpu((2, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_add_relu_11:34
        cpp_fused__native_batch_norm_legit_no_training_add_relu_11(buf50, primals_83, primals_84, primals_85, primals_86, buf51, primals_88, primals_89, primals_90, primals_91, buf52)
        del primals_86
        del primals_91
        # Topologically Sorted Source Nodes: [out_49], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:35
        buf53 = extern_kernels.convolution(buf52, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_93, (512, ), (1, ), 'input')
        assert_size_stride(primals_94, (512, ), (1, ), 'input')
        assert_size_stride(primals_95, (512, ), (1, ), 'input')
        assert_size_stride(primals_96, (512, ), (1, ), 'input')
        buf54 = empty_strided_cpu((2, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_relu_10:36
        cpp_fused__native_batch_norm_legit_no_training_relu_10(buf53, primals_93, primals_94, primals_95, primals_96, buf54)
        del primals_96
        # Topologically Sorted Source Nodes: [out_52], Original ATen: [aten.convolution]
        # [Provenance debug handles] extern_kernels.convolution:37
        buf55 = extern_kernels.convolution(buf54, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (2, 512, 1, 1), (512, 1, 512, 512), 'torch.ops.aten.convolution.default')
        assert_size_stride(primals_98, (512, ), (1, ), 'input')
        assert_size_stride(primals_99, (512, ), (1, ), 'input')
        assert_size_stride(primals_100, (512, ), (1, ), 'input')
        assert_size_stride(primals_101, (512, ), (1, ), 'input')
        buf57 = empty_strided_cpu((2, 512, 1, 1), (512, 1, 1024, 1024), torch.float32)
        buf59 = empty_strided_cpu((2, 512, 1, 1), (512, 1, 1, 1), torch.bool)
        assert_size_stride(primals_103, (3, ), (1, ), 'input')
        assert_size_stride(primals_102, (3, 512), (512, 1), 'input')
        # [Provenance debug handles] cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_12:38
        cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_12(buf55, primals_98, primals_99, primals_100, primals_101, buf52, buf57, buf59)
        del primals_101
        buf58 = empty_strided_cpu((2, 3), (3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5, x_6], Original ATen: [aten.mean, aten.view, aten.t, aten.addmm]
        # [Provenance debug handles] extern_kernels.addmm:39
        extern_kernels.addmm(primals_103, reinterpret_tensor(buf57, (2, 512), (512, 1), 0), reinterpret_tensor(primals_102, (512, 3), (1, 512), 0), alpha=1, beta=1, out=buf58)
        del primals_103
        return (buf58, buf0, buf1, primals_3, primals_4, primals_5, primals_6, buf2, primals_8, primals_9, primals_10, buf3, primals_13, primals_14, primals_15, buf4, primals_18, primals_19, primals_20, buf5, primals_23, primals_24, primals_25, buf6, primals_28, primals_29, primals_30, buf7, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, buf8, primals_43, primals_44, primals_45, buf9, primals_48, primals_49, primals_50, buf10, primals_53, primals_54, primals_55, buf11, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, buf12, primals_68, primals_69, primals_70, buf13, primals_73, primals_74, primals_75, buf14, primals_78, primals_79, primals_80, buf15, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, buf16, primals_93, primals_94, primals_95, buf17, primals_98, primals_99, primals_100, primals_102, buf18, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, reinterpret_tensor(buf57, (2, 512), (512, 1), 0), buf59, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((2, 3, 32, 32), (3072, 1024, 32, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((3, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((3, ), (1, ), device='cpu', dtype=torch.float32)
    return [primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cpu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
