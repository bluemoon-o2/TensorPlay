//
// (2.15.0a0) via p10/include/SpecialMath.h; launch plumbing follows the house
// float_math_cuda / binary_float_cuda pattern from TierOpsKernels.cu.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"

#include <SpecialMath.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace {

using tensorplay::special_math::airy_ai_forward;
using tensorplay::special_math::bessel_j0_forward;
using tensorplay::special_math::bessel_j1_forward;
using tensorplay::special_math::bessel_y0_forward;
using tensorplay::special_math::bessel_y1_forward;
using tensorplay::special_math::calc_i0e;
using tensorplay::special_math::calc_i1e;
using tensorplay::special_math::calc_igamma;
using tensorplay::special_math::calc_igammac;
using tensorplay::special_math::calc_polygamma;
using tensorplay::special_math::chebyshev_polynomial_t_forward;
using tensorplay::special_math::chebyshev_polynomial_u_forward;
using tensorplay::special_math::chebyshev_polynomial_v_forward;
using tensorplay::special_math::chebyshev_polynomial_w_forward;
using tensorplay::special_math::hermite_polynomial_h_forward;
using tensorplay::special_math::hermite_polynomial_he_forward;
using tensorplay::special_math::laguerre_polynomial_l_forward;
using tensorplay::special_math::legendre_polynomial_p_forward;
using tensorplay::special_math::modified_bessel_i1_forward;
using tensorplay::special_math::modified_bessel_k0_forward;
using tensorplay::special_math::modified_bessel_k1_forward;
using tensorplay::special_math::scaled_modified_bessel_k0_forward;
using tensorplay::special_math::scaled_modified_bessel_k1_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_t_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_u_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_v_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_w_forward;
using tensorplay::special_math::spherical_bessel_j0_forward;
using tensorplay::special_math::zeta;

constexpr int kThreads = 256;

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

template <typename F>  // double(double)
__global__ void fm_unary_f64_kernel(int64_t n, const double* a, double* out, F f) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = f(a[i]);
}

template <typename F>  // double(double)
__global__ void fm_unary_f32_kernel(int64_t n, const float* a, float* out, F f) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = static_cast<float>(f(static_cast<double>(a[i])));
}

template <typename F>  // double(double,double)
__global__ void fm_binary_f64_kernel(int64_t n, const double* a, const double* b, double* out, F f) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = f(a[i], b[i]);
}

template <typename F>  // double(double,double)
__global__ void fm_binary_f32_kernel(int64_t n, const float* a, const float* b, float* out, F f) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = static_cast<float>(f(static_cast<double>(a[i]),
                                                             static_cast<double>(b[i])));
}

void launch_ew(dim3& grid, dim3& block, int64_t n) {
    block = dim3(kThreads);
    grid = dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads));
}

// Host-stage fallback for reference-math chains that cannot be annotated
// __device__ (same strategy as Tier5OpsKernels.cu's linear-algebra family):
// pull to CPU, run the double-precision scalar loops, ship back. Dtype
// semantics mirror float_math_cuda / binary_float_cuda.
template <typename F>
Tensor host_staged_unary(const Tensor& self, F fn) {
    DType in = self.dtype();
    DType out_dt = isFloatingType(in) ? in : DType::Float32;
    DType compute_dt = (in == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor w = self.contiguous().to(compute_dt).to(Device(DeviceType::CPU));
    Tensor t = Tensor::empty(shape_of(w), compute_dt, Device(DeviceType::CPU));
    int64_t n = w.numel();
    if (compute_dt == DType::Float64) {
        const double* sp = w.data_ptr<double>();
        double* dp = t.data_ptr<double>();
        for (int64_t i = 0; i < n; ++i) dp[i] = static_cast<double>(fn(sp[i]));
    } else {
        const float* sp = w.data_ptr<float>();
        float* dp = t.data_ptr<float>();
        for (int64_t i = 0; i < n; ++i)
            dp[i] = static_cast<float>(fn(static_cast<double>(sp[i])));
    }
    Tensor r = t.to(self.device());
    return (out_dt == compute_dt) ? r : r.to(out_dt);
}

template <typename F>
Tensor host_staged_binary(const Tensor& a_in, const Tensor& b_in, F fn) {
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (!isFloatingType(dt)) dt = DType::Float32;
    Tensor ac = a_in.to(dt).expand(broadcast_shapes(shape_of(a_in), shape_of(b_in)))
                    .contiguous()
                    .to(Device(DeviceType::CPU));
    Tensor bc = b_in.to(dt).expand(shape_of(ac)).contiguous().to(Device(DeviceType::CPU));
    Tensor t = Tensor::empty(shape_of(ac), dt, Device(DeviceType::CPU));
    int64_t n = t.numel();
    if (dt == DType::Float64) {
        const double* ap = ac.data_ptr<double>();
        const double* bp = bc.data_ptr<double>();
        double* dp = t.data_ptr<double>();
        for (int64_t i = 0; i < n; ++i) dp[i] = static_cast<double>(fn(ap[i], bp[i]));
    } else {
        const float* ap = ac.data_ptr<float>();
        const float* bp = bc.data_ptr<float>();
        float* dp = t.data_ptr<float>();
        for (int64_t i = 0; i < n; ++i)
            dp[i] = static_cast<float>(fn(static_cast<double>(ap[i]),
                                          static_cast<double>(bp[i])));
    }
    return t.to(a_in.device());
}

template <typename F>
Tensor float_math_cuda(const Tensor& self, F f, const char* name) {
    DType in = self.dtype();
    DType out_dt = isFloatingType(in) ? in : DType::Float32;
    DType compute_dt = (in == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor w = (in == compute_dt) ? self.contiguous() : self.to(compute_dt).contiguous();
    Tensor t = Tensor::empty(shape_of(w), compute_dt, w.device());
    int64_t n = w.numel();
    if (n > 0) {
        dim3 grid, block;
        launch_ew(grid, block, n);
        auto stream = getCurrentCUDAStream().stream();
        if (compute_dt == DType::Float64) {
            fm_unary_f64_kernel<<<grid, block, 0, stream>>>(n, w.data_ptr<double>(),
                                                            t.data_ptr<double>(), f);
        } else {
            fm_unary_f32_kernel<<<grid, block, 0, stream>>>(n, w.data_ptr<float>(),
                                                            t.data_ptr<float>(), f);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    return (out_dt == compute_dt) ? t : t.to(out_dt);
}

template <typename F>
Tensor binary_float_cuda(const Tensor& a_in, const Tensor& b_in, F f, const char* name) {
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (!isFloatingType(dt)) dt = DType::Float32;
    Tensor ac = a_in.to(dt).expand(broadcast_shapes(shape_of(a_in), shape_of(b_in))).contiguous();
    Tensor bc = b_in.to(dt).expand(shape_of(ac)).contiguous();
    Tensor out = Tensor::empty(shape_of(ac), dt, a_in.device());
    int64_t n = out.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
    if (dt == DType::Float64) {
        fm_binary_f64_kernel<<<grid, block, 0, stream>>>(
            n, ac.data_ptr<double>(), bc.data_ptr<double>(), out.data_ptr<double>(), f);
    } else {
        fm_binary_f32_kernel<<<grid, block, 0, stream>>>(
            n, ac.data_ptr<float>(), bc.data_ptr<float>(), out.data_ptr<float>(), f);
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// ---------------------------------------------------------------------------
// Unary specials
// ---------------------------------------------------------------------------

Tensor airy_ai_cuda(const Tensor& self) {
    return float_math_cuda(self,
                           [] __device__ (double x) { return airy_ai_forward(x); },
                           "airy_ai");
}
Tensor bessel_j0_cuda(const Tensor& self) {
    return float_math_cuda(self,
                           [] __device__ (double x) { return bessel_j0_forward(x); },
                           "bessel_j0");
}
Tensor bessel_j1_cuda(const Tensor& self) {
    return float_math_cuda(self,
                           [] __device__ (double x) { return bessel_j1_forward(x); },
                           "bessel_j1");
}
Tensor bessel_y0_cuda(const Tensor& self) {
    return float_math_cuda(self,
                           [] __device__ (double x) { return bessel_y0_forward(x); },
                           "bessel_y0");
}
Tensor bessel_y1_cuda(const Tensor& self) {
    return float_math_cuda(self,
                           [] __device__ (double x) { return bessel_y1_forward(x); },
                           "bessel_y1");
}
Tensor spherical_bessel_j0_cuda(const Tensor& self) {
    return float_math_cuda(self,
                           [] __device__ (double x) { return spherical_bessel_j0_forward(x); },
                           "spherical_bessel_j0");
}
Tensor modified_bessel_i1_cuda(const Tensor& self) {
    return float_math_cuda(self,
                           [] __device__ (double x) { return modified_bessel_i1_forward(x); },
                           "modified_bessel_i1");
}
Tensor modified_bessel_k0_cuda(const Tensor& self) {
    return float_math_cuda(self,
                           [] __device__ (double x) { return modified_bessel_k0_forward(x); },
                           "modified_bessel_k0");
}
Tensor modified_bessel_k1_cuda(const Tensor& self) {
    return float_math_cuda(self,
                           [] __device__ (double x) { return modified_bessel_k1_forward(x); },
                           "modified_bessel_k1");
}
Tensor scaled_modified_bessel_k0_cuda(const Tensor& self) {
    return float_math_cuda(self,
                           [] __device__ (double x) { return scaled_modified_bessel_k0_forward(x); },
                           "scaled_modified_bessel_k0");
}
Tensor scaled_modified_bessel_k1_cuda(const Tensor& self) {
    return float_math_cuda(self,
                           [] __device__ (double x) { return scaled_modified_bessel_k1_forward(x); },
                           "scaled_modified_bessel_k1");
}
Tensor i0e_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double x) { return calc_i0e(x); }, "i0e");
}
Tensor i1_cuda(const Tensor& self) {
    return float_math_cuda(self,
                           [] __device__ (double x) { return modified_bessel_i1_forward(x); },
                           "i1");
}
Tensor i1e_cuda(const Tensor& self) {
    // calc_i1e 的 Cephes 参考链无法整体标注 __device__（nvcc 拒绝
    // host→device 调用），与 Tier5OpsKernels.cu 的线性代数族一样 host-stage。
    return host_staged_unary(self, [](double x) { return calc_i1e(x); });
}

// ---------------------------------------------------------------------------
// Polynomial family P(x, n): n arrives as an integer-valued Tensor.
// ---------------------------------------------------------------------------

Tensor chebyshev_polynomial_t_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return chebyshev_polynomial_t_forward(a, b);
    }, "chebyshev_polynomial_t");
}
Tensor chebyshev_polynomial_u_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return chebyshev_polynomial_u_forward(a, b);
    }, "chebyshev_polynomial_u");
}
Tensor chebyshev_polynomial_v_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return chebyshev_polynomial_v_forward(a, b);
    }, "chebyshev_polynomial_v");
}
Tensor chebyshev_polynomial_w_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return chebyshev_polynomial_w_forward(a, b);
    }, "chebyshev_polynomial_w");
}
Tensor shifted_chebyshev_polynomial_t_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return shifted_chebyshev_polynomial_t_forward(a, b);
    }, "shifted_chebyshev_polynomial_t");
}
Tensor shifted_chebyshev_polynomial_u_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return shifted_chebyshev_polynomial_u_forward(a, b);
    }, "shifted_chebyshev_polynomial_u");
}
Tensor shifted_chebyshev_polynomial_v_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return shifted_chebyshev_polynomial_v_forward(a, b);
    }, "shifted_chebyshev_polynomial_v");
}
Tensor shifted_chebyshev_polynomial_w_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return shifted_chebyshev_polynomial_w_forward(a, b);
    }, "shifted_chebyshev_polynomial_w");
}
Tensor hermite_polynomial_h_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return hermite_polynomial_h_forward(a, b);
    }, "hermite_polynomial_h");
}
Tensor hermite_polynomial_he_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return hermite_polynomial_he_forward(a, b);
    }, "hermite_polynomial_he");
}
Tensor laguerre_polynomial_l_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return laguerre_polynomial_l_forward(a, b);
    }, "laguerre_polynomial_l");
}
Tensor legendre_polynomial_p_cuda(const Tensor& x, const Tensor& n) {
    return binary_float_cuda(x, n, [] __device__ (double a, double b) {
        return legendre_polynomial_p_forward(a, b);
    }, "legendre_polynomial_p");
}

// ---------------------------------------------------------------------------
// Two-tensor specials
// ---------------------------------------------------------------------------

Tensor zeta_cuda(const Tensor& s, const Tensor& q) {
    return binary_float_cuda(s, q, [] __device__ (double a, double b) {
        return zeta(a, b);
    }, "zeta");
}
Tensor gammainc_cuda(const Tensor& a, const Tensor& x) {
    return host_staged_binary(a, x, [](double p, double v) { return calc_igamma(p, v); });
}
Tensor gammaincc_cuda(const Tensor& a, const Tensor& x) {
    return host_staged_binary(a, x, [](double p, double v) { return calc_igammac(p, v); });
}
Tensor polygamma_cuda(int64_t n, const Tensor& x) {
    return float_math_cuda(x, [n] __device__ (double v) {
        return calc_polygamma(v, static_cast<int>(n));
    }, "polygamma");
}

}  // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, SpecialKernels) {
    // Unary specials
    m.impl("bessel_j0", bessel_j0_cuda);
    m.impl("bessel_j1", bessel_j1_cuda);
    m.impl("bessel_y0", bessel_y0_cuda);
    m.impl("bessel_y1", bessel_y1_cuda);
    m.impl("airy_ai", airy_ai_cuda);
    m.impl("spherical_bessel_j0", spherical_bessel_j0_cuda);
    m.impl("modified_bessel_i1", modified_bessel_i1_cuda);
    m.impl("modified_bessel_k0", modified_bessel_k0_cuda);
    m.impl("modified_bessel_k1", modified_bessel_k1_cuda);
    m.impl("scaled_modified_bessel_k0", scaled_modified_bessel_k0_cuda);
    m.impl("scaled_modified_bessel_k1", scaled_modified_bessel_k1_cuda);
    m.impl("i0e", i0e_cuda);
    m.impl("i1", i1_cuda);
    m.impl("i1e", i1e_cuda);
    // Polynomial family
    m.impl("chebyshev_polynomial_t", chebyshev_polynomial_t_cuda);
    m.impl("chebyshev_polynomial_u", chebyshev_polynomial_u_cuda);
    m.impl("chebyshev_polynomial_v", chebyshev_polynomial_v_cuda);
    m.impl("chebyshev_polynomial_w", chebyshev_polynomial_w_cuda);
    m.impl("shifted_chebyshev_polynomial_t", shifted_chebyshev_polynomial_t_cuda);
    m.impl("shifted_chebyshev_polynomial_u", shifted_chebyshev_polynomial_u_cuda);
    m.impl("shifted_chebyshev_polynomial_v", shifted_chebyshev_polynomial_v_cuda);
    m.impl("shifted_chebyshev_polynomial_w", shifted_chebyshev_polynomial_w_cuda);
    m.impl("hermite_polynomial_h", hermite_polynomial_h_cuda);
    m.impl("hermite_polynomial_he", hermite_polynomial_he_cuda);
    m.impl("laguerre_polynomial_l", laguerre_polynomial_l_cuda);
    m.impl("legendre_polynomial_p", legendre_polynomial_p_cuda);
    // Two-tensor / parametric
    m.impl("zeta", zeta_cuda);
    m.impl("gammainc", gammainc_cuda);
    m.impl("gammaincc", gammaincc_cuda);
    m.impl("polygamma", polygamma_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
