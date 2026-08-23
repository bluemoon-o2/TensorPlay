// Tier 2-4 operators - CUDA kernels (mirror of cpu/TierOpsKernels.cpp).
//
// Same op set and ATen anchors as the CPU file. Elementwise work runs as
// grid-stride loops; single-dim reductions assign one thread per output
// slice and walk the reduced dimension sequentially. Rare complex ops
// (mode/kthvalue/nanmedian/dist-special-p) are host-staged reference paths.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "SpecialMath.h"

#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <utility>
#include <type_traits>

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

constexpr int kThreads = 256;

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(RuntimeError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim - ndim, ")");
    }
    return dim;
}

inline void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& inner) {
    outer = 1; inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
}

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

// ---------------------------------------------------------------------------
// Generic elementwise device kernels
// ---------------------------------------------------------------------------

template <typename T, typename Op>
__global__ void ew_binary_kernel(int64_t n, const T* a, const T* b, T* out, Op op) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = op(a[i], b[i]);
}

template <typename T, typename Pred>
__global__ void ew_bool_binary_kernel(int64_t n, const T* a, const T* b, bool* out, Pred pred) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = pred(a[i], b[i]);
}

template <typename T, typename Pred>
__global__ void ew_bool_unary_kernel(int64_t n, const T* a, bool* out, Pred pred) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = pred(a[i]);
}

template <typename T, typename F>
__global__ void ew_unary_kernel(int64_t n, const T* a, T* out, F f) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = f(a[i]);
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

void launch_ew(dim3& grid, dim3& block, int64_t n) {
    block = dim3(kThreads);
    grid = dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads));
}

// Binary on common promoted dtype.
template <typename Op>
Tensor binary_same_cuda(const Tensor& a_in, const Tensor& b_in, Op op, const char* name) {
    std::vector<int64_t> out_shape = broadcast_shapes(shape_of(a_in), shape_of(b_in));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    Tensor ac = a_in.to(dt).expand(out_shape).contiguous();
    Tensor bc = b_in.to(dt).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    int64_t n = out.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_BIN(ctype, name_) \
    case DType::name_: \
        ew_binary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, ac.data_ptr<ctype>(), bc.data_ptr<ctype>(), out.data_ptr<ctype>(), op); \
        break;
    switch (dt) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BIN)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BIN
    CUDA_CHECK(cudaGetLastError());
    return out;
}

template <typename Pred>
Tensor binary_bool_cuda(const Tensor& a_in, const Tensor& b_in, Pred pred, const char* name) {
    std::vector<int64_t> out_shape = broadcast_shapes(shape_of(a_in), shape_of(b_in));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    Tensor ac = a_in.to(dt).expand(out_shape).contiguous();
    Tensor bc = b_in.to(dt).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, DType::Bool, a_in.device());
    int64_t n = out.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_BBIN(ctype, name_) \
    case DType::name_: \
        ew_bool_binary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, ac.data_ptr<ctype>(), bc.data_ptr<ctype>(), out.data_ptr<bool>(), pred); \
        break;
    switch (dt) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BBIN)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BBIN
    CUDA_CHECK(cudaGetLastError());
    return out;
}

template <typename Pred>
Tensor bool_unary_cuda(const Tensor& self, Pred pred, const char* name) {
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(self), DType::Bool, self.device());
    int64_t n = self.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_BU(ctype, name_) \
    case DType::name_: \
        ew_bool_unary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, sc.data_ptr<ctype>(), out.data_ptr<bool>(), pred); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BU)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BU
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// Dtype-preserving unary.
template <typename F>
Tensor dtype_unary_cuda(const Tensor& self, F f, const char* name) {
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return out;
    dim3 grid, block;
    launch_ew(grid, block, n);
    auto stream = getCurrentCUDAStream().stream();
#define TP_DU(ctype, name_) \
    case DType::name_: \
        ew_unary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, sc.data_ptr<ctype>(), out.data_ptr<ctype>(), f); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_DU)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_DU
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// Math unary: f: double->double. Integral->Float32; Half/BF16 compute in
// float and keep dtype; Float32/Float64 preserved.
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

// Math binary with floating promotion.
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
// Single-dim slice reduction: one thread per output slice, sequential along
// the reduced dimension. Accumulates in double.
// ---------------------------------------------------------------------------

template <typename T, class Step>
__global__ void slice_reduce_f64_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                        const T* in, double* out, double init, Step step) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        double acc = init;
        for (int64_t j = 0; j < d_size; ++j) acc = step(acc, static_cast<double>(sp[j * inner]));
        out[si] = acc;
    }
}

} // anonymous namespace

// ===========================================================================
// Arithmetic
// ===========================================================================

Tensor rsub_scalar_cuda(const Tensor& self, Scalar other, Scalar alpha) {
    DType dt = isFloatingType(other.dtype())
                   ? (isFloatingType(self.dtype()) ? self.dtype() : DType::Float32)
                   : self.dtype();
    Tensor sc = self.to(dt).contiguous();
    Tensor full = Tensor::full({}, other, dt, self.device())
                      .expand(shape_of(sc)).contiguous();
    double al = alpha.toDouble();
    return binary_same_cuda(sc, full,
                            [al] __device__ (auto s, auto o) {
                                using T = decltype(o);
                                return static_cast<T>(o * al - s);
                            },
                            "rsub");
}

Tensor rsub_tensor_cuda(const Tensor& self, const Tensor& other, Scalar alpha) {
    double al = alpha.toDouble();
    return binary_same_cuda(self, other,
                            [al] __device__ (auto s, auto o) {
                                using T = decltype(o);
                                return static_cast<T>(o * al - s);
                            },
                            "rsub");
}

Tensor true_divide_tensor_cuda(const Tensor& self, const Tensor& other) {
    return binary_float_cuda(self, other,
                             [] __device__ (double x, double y) { return x / y; }, "true_divide");
}
Tensor true_divide_scalar_cuda(const Tensor& self, Scalar other) {
    return binary_float_cuda(self, Tensor::full({}, other, DType::Float32, self.device()),
                             [] __device__ (double x, double y) { return x / y; }, "true_divide");
}
Tensor divide_tensor_cuda(const Tensor& self, const Tensor& other) {
    return true_divide_tensor_cuda(self, other);
}
Tensor divide_scalar_cuda(const Tensor& self, Scalar other) {
    return true_divide_scalar_cuda(self, other);
}

Tensor remainder_tensor_cuda(const Tensor& self, const Tensor& other) {
    return binary_same_cuda(self, other,
                            [] __device__ (auto x, auto y) -> decltype(x) {
                                using T = decltype(x);
                                T r;
                                if constexpr (std::is_integral_v<T>)
                                    r = static_cast<T>(x % y);
                                else  // Half/BFloat16/float/double via fmod
                                    r = static_cast<T>(::fmod(static_cast<double>(x), static_cast<double>(y)));
                                if (r != T(0) && ((r < 0) != (y < 0))) r = static_cast<T>(r + y);
                                return r;
                            },
                            "remainder");
}
Tensor remainder_scalar_cuda(const Tensor& self, Scalar other) {
    return remainder_tensor_cuda(
        self, Tensor::full({}, other, DType::Undefined, self.device()).to(self.dtype()));
}
Tensor fmod_tensor_cuda(const Tensor& self, const Tensor& other) {
    return binary_same_cuda(self, other,
                            [] __device__ (auto x, auto y) -> decltype(x) {
                                if constexpr (std::is_integral_v<decltype(x)>)
                                    return static_cast<decltype(x)>(x % y);
                                else
                                    return static_cast<decltype(x)>(::fmod(static_cast<double>(x), static_cast<double>(y)));
                            },
                            "fmod");
}
Tensor fmod_scalar_cuda(const Tensor& self, Scalar other) {
    return fmod_tensor_cuda(
        self, Tensor::full({}, other, DType::Undefined, self.device()).to(self.dtype()));
}
Tensor subtract_tensor_cuda(const Tensor& self, const Tensor& other) {
    return binary_same_cuda(self, other,
                            [] __device__ (auto x, auto y) { return x - y; }, "subtract");
}
Tensor subtract_scalar_cuda(const Tensor& self, Scalar other) {
    return subtract_tensor_cuda(self,
                                Tensor::full({}, other, self.dtype(), self.device()));
}
Tensor multiply_tensor_cuda(const Tensor& self, const Tensor& other) {
    return binary_same_cuda(self, other,
                            [] __device__ (auto x, auto y) { return x * y; }, "multiply");
}
Tensor multiply_scalar_cuda(const Tensor& self, Scalar other) {
    double ov = other.toDouble();
    return dtype_unary_cuda(self,
                            [ov] __device__ (auto x) {
                                using T = decltype(x);
                                return static_cast<T>(static_cast<double>(x) * ov);
                            },
                            "multiply");
}
Tensor negative_cuda(const Tensor& self) {
    return dtype_unary_cuda(self,
                            [] __device__ (auto x) { return static_cast<decltype(x)>(-x); },
                            "negative");
}
Tensor positive_cuda(const Tensor& self) { return self.clone(); }

// ===========================================================================
// Comparisons / logic
// ===========================================================================

Tensor greater_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) { return x > y; }, "greater");
}
Tensor greater_equal_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) { return x >= y; }, "greater_equal");
}
Tensor less_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) { return x < y; }, "less");
}
Tensor less_equal_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) { return x <= y; }, "less_equal");
}
Tensor not_equal_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) { return x != y; }, "not_equal");
}
Tensor signbit_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        return static_cast<double>(x) < 0.0 ||
               (static_cast<double>(x) == 0.0 && 1.0 / static_cast<double>(x) < 0.0);
    }, "signbit");
}
Tensor logical_not_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        return !static_cast<bool>(x);
    }, "logical_not");
}
Tensor logical_and_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) -> bool {
        return static_cast<bool>(x) && static_cast<bool>(y);
    }, "logical_and");
}
Tensor logical_or_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) -> bool {
        return static_cast<bool>(x) || static_cast<bool>(y);
    }, "logical_or");
}
Tensor logical_xor_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, [] __device__ (auto x, auto y) -> bool {
        return static_cast<bool>(x) != static_cast<bool>(y);
    }, "logical_xor");
}
Tensor isfinite_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        double d = static_cast<double>(x);
        return d == d && d != std::numeric_limits<double>::infinity() &&
               d != -std::numeric_limits<double>::infinity();
    }, "isfinite");
}
Tensor isinf_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        double d = static_cast<double>(x);
        return d == std::numeric_limits<double>::infinity() ||
               d == -std::numeric_limits<double>::infinity();
    }, "isinf");
}
Tensor isnan_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        double d = static_cast<double>(x);
        return d != d;
    }, "isnan");
}
Tensor isneginf_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        return static_cast<double>(x) == -std::numeric_limits<double>::infinity();
    }, "isneginf");
}
Tensor isposinf_cuda(const Tensor& self) {
    return bool_unary_cuda(self, [] __device__ (auto x) -> bool {
        return static_cast<double>(x) == std::numeric_limits<double>::infinity();
    }, "isposinf");
}

// ===========================================================================
// Math functions
// ===========================================================================

Tensor reciprocal_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double x) { return 1.0 / x; }, "reciprocal");
}
Tensor sgn_cuda(const Tensor& self) {
    return dtype_unary_cuda(self,
                            [] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                double d = static_cast<double>(x);
                                if (d != d) return static_cast<T>(x);
                                if (d > 0) return static_cast<T>(1);
                                if (d < 0) return static_cast<T>(-1);
                                return static_cast<T>(0);
                            },
                            "sgn");
}
Tensor exp2_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double x) { return ::exp2(x); }, "exp2");
}
Tensor sinc_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double x) {
        double px = M_PI * x;
        return ::fabs(px) < 1e-30 ? 1.0 : ::sin(px) / px;
    }, "sinc");
}
Tensor deg2rad_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double x) { return x * (M_PI / 180.0); }, "deg2rad");
}
Tensor rad2deg_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double x) { return x * (180.0 / M_PI); }, "rad2deg");
}
Tensor fix_cuda(const Tensor& self) {
    return dtype_unary_cuda(self,
                            [] __device__ (auto x) -> decltype(x) {
                                if constexpr (std::is_floating_point_v<decltype(x)>)
                                    return static_cast<decltype(x)>(::trunc(static_cast<double>(x)));
                                else
                                    return x;
                            },
                            "fix");
}
Tensor erfinv_cuda(const Tensor& self) {
    // CUDA has no native erfinv; use the Cephes calc_erfinv from SpecialMath.h
    // (the same implementation ATen's erfinv_kernel uses), not glibc's
    // host-only ::erfinv — linking that from device code leaves an undefined
    // symbol in libp10.so.
    return float_math_cuda(self, [] __device__ (double x) { return tensorplay::special_math::calc_erfinv(x); }, "erfinv");
}
Tensor logit_cuda(const Tensor& self, std::optional<Scalar> eps) {
    double e = eps.has_value() ? eps->toDouble() : 0.0;
    return float_math_cuda(self,
                           [e] __device__ (double p) {
                               if (e > 0) p = ::fmin(::fmax(p, e), 1.0 - e);
                               return ::log(p / (1.0 - p));
                           },
                           "logit");
}
Tensor digamma_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double v) {
        if (v <= 0 && v == ::floor(v)) return ::nan("");
        double r = 0;
        while (v < 6.0) { r -= 1.0 / v; v += 1.0; }
        double inv = 1.0 / v, inv2 = inv * inv;
        r += ::log(v) - 0.5 * inv
             - inv2 * (1.0/12.0 - inv2 * (1.0/120.0 - inv2 * (1.0/252.0 - inv2 * (1.0/240.0 - inv2 / 132.0))));
        return r;
    }, "digamma");
}
Tensor i0_cuda(const Tensor& self) {
    return float_math_cuda(self, [] __device__ (double v) {
        double half = 0.5 * ::fabs(v);
        double term = 1.0, sum = 1.0;
        for (int k = 1; k < 60; ++k) {
            term *= half / static_cast<double>(k);
            double term2 = term * term;
            sum += term2;
            if (term2 < 1e-18 * sum) break;
        }
        return sum;
    }, "i0");
}
Tensor nan_to_num_cuda(const Tensor& self, Scalar nan,
                       std::optional<Scalar> posinf, std::optional<Scalar> neginf) {
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return out;
    double nan_v = nan.toDouble();
    bool has_pos = posinf.has_value(), has_neg = neginf.has_value();
    double pos_v = has_pos ? posinf->toDouble() : std::numeric_limits<double>::infinity();
    double neg_v = has_neg ? neginf->toDouble() : -std::numeric_limits<double>::infinity();
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid, block;
    launch_ew(grid, block, n);
    // Per-dtype kernel with the replacement rules baked in.
#define TP_NTN(ctype, name_) \
    case DType::name_: { \
        ctype pv = has_pos ? static_cast<ctype>(pos_v) \
                           : (std::numeric_limits<ctype>::has_infinity \
                                  ? std::numeric_limits<ctype>::infinity() \
                                  : std::numeric_limits<ctype>::max()); \
        ctype nv = has_neg ? static_cast<ctype>(neg_v) \
                           : (std::numeric_limits<ctype>::has_infinity \
                                  ? -std::numeric_limits<ctype>::infinity() \
                                  : std::numeric_limits<ctype>::lowest()); \
        ctype na = static_cast<ctype>(nan_v); \
        ew_unary_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, sc.data_ptr<ctype>(), out.data_ptr<ctype>(), \
            [na, pv, nv] __device__ (ctype v) { \
                if constexpr (std::is_floating_point_v<ctype>) { \
                    if (v != v) return na; \
                    if (v == std::numeric_limits<ctype>::infinity()) return pv; \
                    if (v == -std::numeric_limits<ctype>::infinity()) return nv; \
                } \
                return v; \
            }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NTN)
        default: TP_THROW(TypeError, "nan_to_num: unsupported dtype");
    }
#undef TP_NTN
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor xlogy_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, [] __device__ (double x, double y) {
        if (x == 0.0) return 0.0;
        return x * ::log(y);
    }, "xlogy");
}
Tensor logaddexp_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, [] __device__ (double x, double y) {
        double m = ::fmax(x, y);
        if (m == -std::numeric_limits<double>::infinity() || m != m) return m;
        return m + ::log1p(::exp(-::fabs(x - y)));
    }, "logaddexp");
}
Tensor logaddexp2_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, [] __device__ (double x, double y) {
        double m = ::fmax(x, y);
        if (m == -std::numeric_limits<double>::infinity() || m != m) return m;
        return m + ::log1p(::exp2(-::fabs(x - y))) / M_LN2;
    }, "logaddexp2");
}
Tensor copysign_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, [] __device__ (double x, double y) {
        return ::copysign(x, y);
    }, "copysign");
}
Tensor hypot_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, [] __device__ (double x, double y) {
        return ::hypot(x, y);
    }, "hypot");
}
Tensor nextafter_cuda(const Tensor& a, const Tensor& b) {
    return binary_float_cuda(a, b, [] __device__ (double x, double y) {
        return ::nextafter(x, y);
    }, "nextafter");
}
Tensor gcd_cuda(const Tensor& a, const Tensor& b) {
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (isFloatingType(dt)) TP_THROW(TypeError, "gcd only supports integral tensors");
    return binary_same_cuda(a, b,
                            [] __device__ (auto x, auto y) -> decltype(x) {
                                using T = decltype(x);
                                long long ux = static_cast<long long>(x < 0 ? -x : x);
                                long long uy = static_cast<long long>(y < 0 ? -y : y);
                                while (uy) { long long t = ux % uy; ux = uy; uy = t; }
                                return static_cast<T>(ux);
                            },
                            "gcd");
}
Tensor lcm_cuda(const Tensor& a, const Tensor& b) {
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (isFloatingType(dt)) TP_THROW(TypeError, "lcm only supports integral tensors");
    return binary_same_cuda(a, b,
                            [] __device__ (auto x, auto y) -> decltype(x) {
                                using T = decltype(x);
                                long long ux = static_cast<long long>(x < 0 ? -x : x);
                                long long uy = static_cast<long long>(y < 0 ? -y : y);
                                long long g = ux, t2 = uy;
                                while (t2) { long long t3 = g % t2; g = t2; t2 = t3; }
                                if (g == 0) return static_cast<T>(0);
                                return static_cast<T>(ux / g * uy);
                            },
                            "lcm");
}
Tensor heaviside_cuda(const Tensor& a, const Tensor& values) {
    return binary_same_cuda(a, values,
                            [] __device__ (auto x, auto v) -> decltype(x) {
                                using T = decltype(x);
                                double xd = static_cast<double>(x);
                                if (xd < 0.0) return static_cast<T>(0);
                                if (xd == 0.0) return static_cast<T>(v);
                                return static_cast<T>(1);
                            },
                            "heaviside");
}

// ===========================================================================
// Clamp family
// ===========================================================================

Tensor clamp_min_scalar_cuda(const Tensor& self, Scalar min) {
    double lo = min.toDouble();
    return dtype_unary_cuda(self,
                            [lo] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                return static_cast<double>(x) < lo ? static_cast<T>(lo)
                                                                   : static_cast<T>(x);
                            },
                            "clamp_min");
}
Tensor clamp_max_scalar_cuda(const Tensor& self, Scalar max) {
    double hi = max.toDouble();
    return dtype_unary_cuda(self,
                            [hi] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                return static_cast<double>(x) > hi ? static_cast<T>(hi)
                                                                   : static_cast<T>(x);
                            },
                            "clamp_max");
}
Tensor clamp_min_tensor_cuda(const Tensor& self, const Tensor& min) {
    return binary_same_cuda(self, min,
                            [] __device__ (auto x, auto m) -> decltype(x) {
                                using T = decltype(x);
                                return static_cast<double>(m) > static_cast<double>(x)
                                           ? static_cast<T>(m) : static_cast<T>(x);
                            },
                            "clamp_min");
}
Tensor clamp_max_tensor_cuda(const Tensor& self, const Tensor& max) {
    return binary_same_cuda(self, max,
                            [] __device__ (auto x, auto m) -> decltype(x) {
                                using T = decltype(x);
                                return static_cast<double>(m) < static_cast<double>(x)
                                           ? static_cast<T>(m) : static_cast<T>(x);
                            },
                            "clamp_max");
}
Tensor clip_cuda(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    if (min.has_value() && max.has_value()) {
        Tensor r = clamp_min_scalar_cuda(self, *min);
        return clamp_max_scalar_cuda(r, *max);
    }
    if (min.has_value()) return clamp_min_scalar_cuda(self, *min);
    if (max.has_value()) return clamp_max_scalar_cuda(self, *max);
    return self.clone();
}
Tensor& clamp__cuda(Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor r = clip_cuda(self, std::move(min), std::move(max));
    self.copy_(r);
    return self;
}

// ===========================================================================
// Activations
// ===========================================================================

Tensor selu_cuda(const Tensor& self) {
    constexpr double kAlpha = 1.6732632423543772848170429916717;
    constexpr double kScale = 1.0507009873554804934193349852946;
    return dtype_unary_cuda(self,
                            [=] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                double v = static_cast<double>(x);
                                return static_cast<T>(v > 0 ? kScale * v
                                                            : kScale * kAlpha * (::exp(v) - 1.0));
                            },
                            "selu");
}
Tensor celu_cuda(const Tensor& self, Scalar alpha) {
    double a = alpha.toDouble();
    return dtype_unary_cuda(self,
                            [a] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                double v = static_cast<double>(x);
                                return static_cast<T>(v > 0 ? v : a * (::exp(v / a) - 1.0));
                            },
                            "celu");
}
Tensor hardshrink_cuda(const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    return dtype_unary_cuda(self,
                            [l] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                double v = static_cast<double>(x);
                                return (v > l || v < -l) ? static_cast<T>(x) : static_cast<T>(0);
                            },
                            "hardshrink");
}
Tensor softshrink_cuda(const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    return dtype_unary_cuda(self,
                            [l] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                double v = static_cast<double>(x);
                                if (v > l) return static_cast<T>(v - l);
                                if (v < -l) return static_cast<T>(v + l);
                                return static_cast<T>(0);
                            },
                            "softshrink");
}
Tensor threshold_cuda(const Tensor& self, Scalar threshold, Scalar value) {
    double t = threshold.toDouble(), val = value.toDouble();
    return dtype_unary_cuda(self,
                            [t, val] __device__ (auto x) -> decltype(x) {
                                using T = decltype(x);
                                return static_cast<double>(x) <= t ? static_cast<T>(val)
                                                                   : static_cast<T>(x);
                            },
                            "threshold");
}
namespace {
__global__ void prelu_channel_f32_kernel(int64_t n, int64_t C, int64_t outer, int64_t inner,
                                         const float* sp, const float* wp, float* dp) {
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < n; li += stride) {
        int64_t tail = li % inner;
        int64_t rest = li / inner;
        int64_t c = rest % C;
        float v = sp[li];
        float w = wp[c];
        dp[li] = v > 0 ? v : w * v;
    }
}
} // anonymous namespace

Tensor prelu_cuda(const Tensor& self, const Tensor& weight) {
    Tensor wc = weight.contiguous();
    if (wc.numel() == 1) {
        double w0 = wc.item().toDouble();
        return dtype_unary_cuda(self,
                                [w0] __device__ (auto x) -> decltype(x) {
                                    using T = decltype(x);
                                    double v = static_cast<double>(x);
                                    return static_cast<T>(v > 0 ? v : w0 * v);
                                },
                                "prelu");
    }
    int64_t cdim = self.dim() >= 2 ? 1 : 0;
    int64_t C = self.size(cdim);
    int64_t outer = 1;
    for (int64_t i = 0; i < cdim; ++i) outer *= self.size(i);
    int64_t inner = 1;
    for (int64_t i = cdim + 1; i < self.dim(); ++i) inner *= self.size(i);
    Tensor sc = self.contiguous().to(DType::Float32);
    Tensor wf = wc.to(DType::Float32).contiguous();
    Tensor out32 = Tensor::empty(shape_of(sc), DType::Float32, self.device());
    int64_t n = sc.numel();
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid, block;
    launch_ew(grid, block, n);
    prelu_channel_f32_kernel<<<grid, block, 0, stream>>>(
        n, C, outer, inner, sc.data_ptr<float>(), wf.data_ptr<float>(),
        out32.data_ptr<float>());
    CUDA_CHECK(cudaGetLastError());
    return out32.to(self.dtype());
}

TENSORPLAY_LIBRARY_IMPL(CUDA, TierOpsKernels) {
    m.impl("rsub.Scalar", rsub_scalar_cuda);
    m.impl("rsub.Tensor", rsub_tensor_cuda);
    m.impl("true_divide.Tensor", true_divide_tensor_cuda);
    m.impl("true_divide.Scalar", true_divide_scalar_cuda);
    m.impl("divide.Tensor", divide_tensor_cuda);
    m.impl("divide.Scalar", divide_scalar_cuda);
    m.impl("remainder.Tensor", remainder_tensor_cuda);
    m.impl("remainder.Scalar", remainder_scalar_cuda);
    m.impl("fmod.Tensor", fmod_tensor_cuda);
    m.impl("fmod.Scalar", fmod_scalar_cuda);
    m.impl("subtract.Tensor", subtract_tensor_cuda);
    m.impl("subtract.Scalar", subtract_scalar_cuda);
    m.impl("multiply.Tensor", multiply_tensor_cuda);
    m.impl("multiply.Scalar", multiply_scalar_cuda);
    m.impl("negative", negative_cuda);
    m.impl("positive", positive_cuda);
    m.impl("greater", greater_cuda);
    m.impl("greater_equal", greater_equal_cuda);
    m.impl("less", less_cuda);
    m.impl("less_equal", less_equal_cuda);
    m.impl("not_equal", not_equal_cuda);
    m.impl("signbit", signbit_cuda);
    m.impl("logical_not", logical_not_cuda);
    m.impl("logical_and", logical_and_cuda);
    m.impl("logical_or", logical_or_cuda);
    m.impl("logical_xor", logical_xor_cuda);
    m.impl("isfinite", isfinite_cuda);
    m.impl("isinf", isinf_cuda);
    m.impl("isnan", isnan_cuda);
    m.impl("isneginf", isneginf_cuda);
    m.impl("isposinf", isposinf_cuda);
    m.impl("reciprocal", reciprocal_cuda);
    m.impl("sgn", sgn_cuda);
    m.impl("exp2", exp2_cuda);
    m.impl("sinc", sinc_cuda);
    m.impl("deg2rad", deg2rad_cuda);
    m.impl("rad2deg", rad2deg_cuda);
    m.impl("fix", fix_cuda);
    m.impl("erfinv", erfinv_cuda);
    m.impl("logit", logit_cuda);
    m.impl("digamma", digamma_cuda);
    m.impl("i0", i0_cuda);
    m.impl("nan_to_num", nan_to_num_cuda);
    m.impl("xlogy.Tensor", xlogy_cuda);
    m.impl("logaddexp", logaddexp_cuda);
    m.impl("logaddexp2", logaddexp2_cuda);
    m.impl("copysign.Tensor", copysign_cuda);
    m.impl("hypot", hypot_cuda);
    m.impl("nextafter", nextafter_cuda);
    m.impl("gcd", gcd_cuda);
    m.impl("lcm", lcm_cuda);
    m.impl("heaviside", heaviside_cuda);
    m.impl("clamp_", clamp__cuda);
    m.impl("clamp_min.Scalar", clamp_min_scalar_cuda);
    m.impl("clamp_max.Scalar", clamp_max_scalar_cuda);
    m.impl("clamp_min.Tensor", clamp_min_tensor_cuda);
    m.impl("clamp_max.Tensor", clamp_max_tensor_cuda);
    m.impl("clip", clip_cuda);
    m.impl("selu", selu_cuda);
    m.impl("celu", celu_cuda);
    m.impl("hardshrink", hardshrink_cuda);
    m.impl("softshrink", softshrink_cuda);
    m.impl("threshold", threshold_cuda);
    m.impl("prelu", prelu_cuda);
}

} // namespace cuda
} // namespace tensorplay
