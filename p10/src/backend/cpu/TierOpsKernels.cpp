// Tier 2-4 operators (arithmetic aliases, comparisons/logic, clamp family,
// activations, math functions, reductions, shape ops) - CPU kernels.
//
//   BinaryOps.cpp:954 true_divide, :1169/:1203 rsub, :1184 remainder,
//                  :1498 logical_and, :1540 fmod
//   TensorCompare.cpp:435 isnan, :458 isinf, :474 isfinite
//   ReduceOps.cpp:1357 trace_cpu, :1578 logsumexp, :1801 amax_out, :1310 nansum
//   Activation.cpp:525 selu, :541 celu, :202 hardshrink, :697 prelu
//   TensorShape.cpp:1272 diag_embed, :4426 unfold
//   TensorTransformations.cpp:36 flip, :110 roll, :145 rot90
//   PixelShuffle.cpp:23 pixel_shuffle_cpu
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Utils.h"
#include "TensorIteratorOps.h"
#include "Exception.h"
#include "Parallel.h"
#include "TypePromotion.h"
#include "SpecialMath.h"
#include "cpu/ComplexUnary.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <cstring>
#include <utility>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <type_traits>
#include <optional>
#include <string>

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace {

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    // Dimension wrapping reports the original (unwrapped) value on error.
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 min, ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

// Scalar wrapping: rank-0 accepts dims [-1, 0] (both wrap to 0).  Used by
// flip's dim-list conversion.
inline int64_t wrap_dim_scalar(int64_t dim, int64_t ndim) {
    return wrap_dim(dim, ndim == 0 ? 1 : ndim);
}

inline void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& inner) {
    outer = 1; inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
}

inline DType scalar_promote(DType t, const Scalar& s) {
    // Weak scalar participation: scalars only promote the tensor dtype when
    // they carry a floating type of their own.
    if (!isFloatingType(s.dtype())) return t;
    if (isFloatingType(t)) return t;
    return DType::Float32;
}

// ---------------------------------------------------------------------------
// Elementwise helpers
// ---------------------------------------------------------------------------

// Broadcast both inputs to a common promoted dtype; op returns that dtype.
// kArith selects the complex-capable TensorIterator applier for pure
// arithmetic functors (rsub/subtract/multiply); ordering/fmod-style callers
// must keep the default: those ops are not defined over complex, following
template <bool kArith = false, typename Op>
Tensor binary_same_kernel(const Tensor& a_in, const Tensor& b_in, Op op, const char* name) {
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    // Cast to the promoted dtype but do NOT materialize broadcasts:
    // TensorIterator handles expansion + strided access natively.
    Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt));
    Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt));
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    if constexpr (kArith) {
        ti_apply_arith(out, ac, bc, op);
    } else {
        ti_apply_binary(out, ac, bc, op);
    }
    (void)name;
    return out;
}

// Binary op whose inputs promote to a FLOATING dtype first (ints -> Float32).
template <typename F>  // F: (double,double) -> double
Tensor binary_float_kernel(const Tensor& a_in, const Tensor& b_in, F f, const char* name) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (!isFloatingType(dt)) dt = DType::Float32;
    // Reduced-width inputs are evaluated in Float32 and narrowed once at the
    // end; the loops below only ever address float or double buffers.
    DType compute_dt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor ac = a_in.to(compute_dt).expand(out_shape).contiguous();
    Tensor bc = b_in.to(compute_dt).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, compute_dt, a_in.device());
    int64_t n = out.numel();
    if (compute_dt == DType::Float64) {
        const double* ap = ac.data_ptr<double>();
        const double* bp = bc.data_ptr<double>();
        double* dp = out.data_ptr<double>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = f(ap[i], bp[i]);
        });
    } else {
        const float* ap = ac.data_ptr<float>();
        const float* bp = bc.data_ptr<float>();
        float* dp = out.data_ptr<float>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = static_cast<float>(f(ap[i], bp[i]));
        });
    }
    return (dt == compute_dt) ? out : out.to(dt);
}

template <typename Pred>
Tensor binary_bool_kernel(const Tensor& a_in, const Tensor& b_in, Pred pred, const char* name) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt)).expand(out_shape).contiguous();
    Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt)).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, DType::Bool, a_in.device());
    int64_t n = out.numel();
#define TP_BBIN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* ap = ac.data_ptr<ctype>(); \
        const ctype* bp = bc.data_ptr<ctype>(); \
        bool* dp = out.data_ptr<bool>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) dp[i] = pred(ap[i], bp[i]); \
        }); \
        break; \
    }
    switch (dt) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BBIN_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BBIN_CASE
    return out;
}

template <typename Pred>
Tensor bool_unary_kernel(const Tensor& self, Pred pred, const char* name) {
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), DType::Bool, self.device());
    Tensor sc = self.contiguous();
    int64_t n = self.numel();
#define TP_BU_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        bool* dp = out.data_ptr<bool>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) dp[i] = pred(sp[i]); \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BU_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BU_CASE
    return out;
}

// Dtype-preserving unary (used by sgn, fix, negative, ...).
template <typename F>
Tensor dtype_unary_kernel(const Tensor& self, F f, const char* name) {
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    Tensor sc = self.contiguous();
    int64_t n = self.numel();
#define TP_DU_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) dp[i] = static_cast<ctype>(f(sp[i])); \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_DU_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_DU_CASE
    return out;
}

// integral inputs yield Float32; Half/BFloat16 compute in float and keep
// their dtype; Float32/Float64 preserved.
template <typename F>
Tensor float_math_kernel(const Tensor& self, F f, const char* name) {
    DType in = self.dtype();
    DType out_dt = isFloatingType(in) ? in : DType::Float32;
    DType compute_dt = (in == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor w = (self.dtype() == compute_dt) ? self.contiguous()
                                            : self.to(compute_dt).contiguous();
    Tensor t = Tensor::empty(static_cast<std::vector<int64_t>>(w.shape()), compute_dt, w.device());
    int64_t n = w.numel();
    if (compute_dt == DType::Float64) {
        const double* sp = w.data_ptr<double>();
        double* dp = t.data_ptr<double>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) dp[i] = f(sp[i]);
        });
    } else {
        const float* sp = w.data_ptr<float>();
        float* dp = t.data_ptr<float>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) dp[i] = static_cast<float>(f(sp[i]));
        });
    }
    return (out_dt == compute_dt) ? t : t.to(out_dt);
}

// ---------------------------------------------------------------------------
// Reduction driver over an arbitrary set of dims. done(acc) returns the
// stored value as double; storage conversion handled per out dtype.
// ---------------------------------------------------------------------------

template <class AccT, class Step, class Done>
Tensor reduce_dims_impl(const Tensor& self, std::vector<int64_t> dims_in,
                        bool keepdim, DType out_dtype, AccT init, Step step, Done done) {
    int64_t nd = self.dim();
    std::vector<bool> reduced(nd, false);
    for (auto& d : dims_in) { d = wrap_dim(d, nd); reduced[d] = true; }
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i) {
        if (!reduced[i]) out_shape.push_back(self.size(i));
        else if (keepdim) out_shape.push_back(1);
    }
    std::vector<int64_t> strides(nd, 0);
    {
        int64_t s = 1;
        for (int64_t i = nd - 1; i >= 0; --i) { strides[i] = s; s *= self.size(i); }
    }
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(out_shape, out_dtype, self.device());
    int64_t out_numel = out.numel();
    if (out_numel == 0) return out;
    std::vector<int64_t> red_dims, red_strides;
    for (int64_t i = 0; i < nd; ++i) {
        if (reduced[i]) { red_dims.push_back(i); red_strides.push_back(strides[i]); }
    }
    int64_t total_red = 1;
    for (int64_t d : red_dims) total_red *= self.size(d);

    parallel_for(0, out_numel, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        std::vector<int64_t> coords(red_dims.size(), 0);
        for (int64_t oi = begin; oi < end; ++oi) {
            // Decode output coordinates against the non-reduced shape (the
            // keepdim size-1 slots are not part of the linear index), then
            // compute the input base offset over the non-reduced dims.
            int64_t base = 0;
            {
                int64_t r2 = oi;
                std::vector<int64_t> oc_shape;
                for (int64_t i = 0; i < nd; ++i)
                    if (!reduced[i]) oc_shape.push_back(self.size(i));
                std::vector<int64_t> oc(std::max<int64_t>(oc_shape.size(), 1), 0);
                for (int64_t i = static_cast<int64_t>(oc_shape.size()) - 1; i >= 0; --i) {
                    oc[i] = r2 % oc_shape[i];
                    r2 /= oc_shape[i];
                }
                int64_t ok = 0;
                for (int64_t i = 0; i < nd; ++i) {
                    if (reduced[i]) continue;
                    base += oc[ok] * strides[i];
                    ++ok;
                }
            }
            AccT acc = init;
            std::fill(coords.begin(), coords.end(), 0);
            for (int64_t c = 0; c < total_red; ++c) {
                int64_t off = base;
                for (size_t r = 0; r < red_dims.size(); ++r) off += coords[r] * red_strides[r];
                switch (sc.dtype()) {
#define TP_RD_STEP(ctype, name_) \
    case DType::name_: acc = step(acc, static_cast<double>(sc.data_ptr<ctype>()[off])); break;
                    TENSORPLAY_FORALL_SCALAR_TYPES(TP_RD_STEP)
#undef TP_RD_STEP
                    default: TP_THROW(TypeError, "reduce: unsupported dtype");
                }
                for (int64_t r = static_cast<int64_t>(red_dims.size()) - 1; r >= 0; --r) {
                    if (++coords[r] < self.size(red_dims[r])) break;
                    coords[r] = 0;
                }
            }
            double v = done(acc);
            switch (out_dtype) {
#define TP_RD_DONE(ctype, name_) \
    case DType::name_: out.data_ptr<ctype>()[oi] = static_cast<ctype>(v); break;
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_RD_DONE)
#undef TP_RD_DONE
                default: TP_THROW(TypeError, "reduce: unsupported out dtype");
            }
        }
    });
    return out;
}

std::pair<Tensor, Tensor> mean_var_over_dims(const Tensor& self, std::vector<int64_t> dims_in,
                                             bool unbiased, bool keepdim) {
    int64_t nd = self.dim();
    std::vector<int64_t> dims = dims_in;
    if (dims.empty()) {
        // No dim named: reduce over every axis.
        for (int64_t i = 0; i < nd; ++i) dims.push_back(i);
    }
    std::vector<bool> reduced(nd, false);
    for (auto& d : dims) { d = wrap_dim(d, nd); reduced[d] = true; }
    bool all_reduced = true;
    for (bool b : reduced) if (!b) all_reduced = false;
    if (all_reduced) { for (int64_t i = 0; i < nd; ++i) reduced[i] = true; }
    std::vector<int64_t> ksizes, out_sizes;
    for (int64_t i = 0; i < nd; ++i) {
        if (reduced[i]) { ksizes.push_back(1); if (keepdim) out_sizes.push_back(1); }
        else { ksizes.push_back(self.size(i)); out_sizes.push_back(self.size(i)); }
    }
    DType dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Float32;
    Tensor sc = self.to(dt).contiguous();
    Tensor mean = Tensor::empty(out_sizes, dt, self.device());
    Tensor var = Tensor::empty(out_sizes, dt, self.device());
    int64_t out_numel = mean.numel();
    std::vector<int64_t> strides(nd, 0);
    { int64_t s = 1; for (int64_t i = nd - 1; i >= 0; --i) { strides[i] = s; s *= self.size(i); } }
    std::vector<int64_t> red_dims, red_strides;
    for (int64_t i = 0; i < nd; ++i) if (reduced[i]) { red_dims.push_back(i); red_strides.push_back(strides[i]); }
    int64_t n_red = 1;
    for (int64_t d : red_dims) n_red *= self.size(d);
    double ddof = (unbiased && n_red > 1) ? 1.0 : 0.0;

    if (dt == DType::Float64) {
        const double* sp = sc.data_ptr<double>();
        double* mp = mean.data_ptr<double>();
        double* vp = var.data_ptr<double>();
        parallel_for(0, out_numel, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            std::vector<int64_t> coords(red_dims.size(), 0);
            std::vector<int64_t> oc(out_sizes.size(), 0);
            for (int64_t oi = b; oi < e; ++oi) {
                int64_t r2 = oi;
                for (int64_t i = static_cast<int64_t>(out_sizes.size()) - 1; i >= 0; --i) {
                    oc[i] = r2 % out_sizes[i]; r2 /= out_sizes[i];
                }
                int64_t base = 0; int64_t ok = 0;
                for (int64_t i = 0; i < nd; ++i) {
                    if (reduced[i]) continue;
                    base += oc[ok++] * strides[i];
                }
                double s = 0, sq = 0;
                std::fill(coords.begin(), coords.end(), 0);
                for (int64_t c = 0; c < n_red; ++c) {
                    int64_t off = base;
                    for (size_t rr = 0; rr < red_dims.size(); ++rr) off += coords[rr] * red_strides[rr];
                    double v = sp[off];
                    s += v; sq += v * v;
                    for (int64_t rr = static_cast<int64_t>(red_dims.size()) - 1; rr >= 0; --rr) {
                        if (++coords[rr] < self.size(red_dims[rr])) break;
                        coords[rr] = 0;
                    }
                }
                double m = s / n_red;
                mp[oi] = m;
                double vv = (sq - m * m * n_red) / (n_red - ddof);
                vp[oi] = vv > 0 ? vv : 0.0;
            }
        });
    } else {
        const float* sp = sc.data_ptr<float>();
        float* mp = mean.data_ptr<float>();
        float* vp = var.data_ptr<float>();
        parallel_for(0, out_numel, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            std::vector<int64_t> coords(red_dims.size(), 0);
            std::vector<int64_t> oc(out_sizes.size(), 0);
            for (int64_t oi = b; oi < e; ++oi) {
                int64_t r2 = oi;
                for (int64_t i = static_cast<int64_t>(out_sizes.size()) - 1; i >= 0; --i) {
                    oc[i] = r2 % out_sizes[i]; r2 /= out_sizes[i];
                }
                int64_t base = 0; int64_t ok = 0;
                for (int64_t i = 0; i < nd; ++i) {
                    if (reduced[i]) continue;
                    base += oc[ok++] * strides[i];
                }
                double s = 0, sq = 0;
                std::fill(coords.begin(), coords.end(), 0);
                for (int64_t c = 0; c < n_red; ++c) {
                    int64_t off = base;
                    for (size_t rr = 0; rr < red_dims.size(); ++rr) off += coords[rr] * red_strides[rr];
                    double v = sp[off];
                    s += v; sq += v * v;
                    for (int64_t rr = static_cast<int64_t>(red_dims.size()) - 1; rr >= 0; --rr) {
                        if (++coords[rr] < self.size(red_dims[rr])) break;
                        coords[rr] = 0;
                    }
                }
                double m = s / n_red;
                mp[oi] = static_cast<float>(m);
                double vv = (sq - m * m * n_red) / (n_red - ddof);
                vp[oi] = static_cast<float>(vv > 0 ? vv : 0.0);
            }
        });
    }
    return {var, mean};  // (variance/std caller picks)
}

struct LseState { double m; double s; bool nan_flag; };

} // anonymous namespace

// ===========================================================================
// Arithmetic (BinaryOps.cpp anchors)
// ===========================================================================

Tensor rsub_scalar_cpu(const Tensor& self, Scalar other, Scalar alpha) {
    // Reversed subtraction: other - alpha * self, under weak scalar promotion.
    // alpha scales the subtrahend, which is self here, not other.
    DType dt = scalar_promote(self.dtype(), other);
    Tensor sc = self.to(dt).contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()), dt, self.device());
    double o = other.toDouble(), al = alpha.toDouble();
    int64_t n = sc.numel();
#define TP_RS_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        ctype ov = static_cast<ctype>(o), av = static_cast<ctype>(al); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t i = b; i < e; ++i) dp[i] = static_cast<ctype>(ov - av * sp[i]); \
        }); \
        break; \
    }
    switch (dt) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_RS_CASE)
        default: TP_THROW(TypeError, "rsub: unsupported dtype");
    }
#undef TP_RS_CASE
    return out;
}

Tensor rsub_tensor_cpu(const Tensor& self, const Tensor& other, Scalar alpha) {
    // other - alpha * self: the same arithmetic as sub with the operands
    // exchanged, so alpha still scales the subtrahend.
    return binary_same_kernel<true>(self, other,
        [alpha](auto s, auto o) {
            using T = decltype(o);
            if constexpr (is_complex_type_v<T> || std::is_floating_point_v<T>) {
                return o - s * alpha.to<T>();
            } else {
                return static_cast<T>(o - s * alpha.to<double>());
            }
        }, "rsub");
}

static Tensor true_divide_core(const Tensor& a, const Tensor& b) {
    // BinaryOps.cpp:954: integral inputs promote to the default float type.
    // A complex operand keeps its own width instead -- the float loop only
    // addresses real buffers, so it would drop the imaginary halves.
    if (isComplexType(a.dtype()) || isComplexType(b.dtype())) {
        return binary_same_kernel<true>(a, b,
            [](auto x, auto y) { return x / y; }, "true_divide");
    }
    return binary_float_kernel(a, b, [](double x, double y) { return x / y; }, "true_divide");
}
Tensor true_divide_tensor_cpu(const Tensor& self, const Tensor& other) { return true_divide_core(self, other); }
Tensor true_divide_scalar_cpu(const Tensor& self, Scalar other) {
    // A Float32 stand-in would widen Half/BFloat16 inputs; the weak-scalar
    // rule keeps the tensor dtype unless the scalar itself is floating.
    return true_divide_core(
        self, Tensor::full({}, other, scalar_promote(self.dtype(), other), self.device()));
}
Tensor divide_tensor_cpu(const Tensor& self, const Tensor& other) { return true_divide_core(self, other); }
Tensor divide_scalar_cpu(const Tensor& self, Scalar other) { return true_divide_scalar_cpu(self, other); }

Tensor remainder_tensor_cpu(const Tensor& self, const Tensor& other) {
    // Python modulo: sign follows divisor (BinaryOps.cpp:1184)
    return binary_same_kernel(self, other, [](auto x, auto y) {
        using T = decltype(x);
        if constexpr (std::is_integral_v<T>) {
            auto r = x % y;
            if (r != T(0) && ((r < 0) != (y < 0))) r = static_cast<T>(r + y);
            return static_cast<T>(r);
        } else {
            // Float/Half/BFloat16 route through fmod on double opmath.
            double xf = static_cast<double>(x), yf = static_cast<double>(y);
            double r = std::fmod(xf, yf);
            if (r != 0.0 && ((r < 0.0) != (yf < 0.0))) r += yf;
            return static_cast<T>(r);
        }
    }, "remainder");
}
Tensor remainder_scalar_cpu(const Tensor& self, Scalar other) {
    // Forcing the scalar into self's dtype would truncate a float divisor
    // against an integral tensor; the pair promotes first.
    const DType dt = scalar_promote(self.dtype(), other);
    return remainder_tensor_cpu(self.to(dt), Tensor::full({}, other, dt, self.device()));
}

Tensor remainder_scalar_tensor_cpu(Scalar self, const Tensor& other) {
    const DType dt = scalar_promote(other.dtype(), self);
    return remainder_tensor_cpu(Tensor::full({}, self, dt, other.device()), other.to(dt));
}

Tensor fmod_tensor_cpu(const Tensor& self, const Tensor& other) {
    // C fmod: sign follows dividend (BinaryOps.cpp:1540)
    return binary_same_kernel(self, other, [](auto x, auto y) -> decltype(x) {
        if constexpr (std::is_integral_v<decltype(x)>)
            return static_cast<decltype(x)>(x % y);
        else
            // Covers float/double/Half/BFloat16 via double opmath.
            return static_cast<decltype(x)>(std::fmod(static_cast<double>(x),
                                                      static_cast<double>(y)));
    }, "fmod");
}
Tensor fmod_scalar_cpu(const Tensor& self, Scalar other) {
    const DType dt = scalar_promote(self.dtype(), other);
    return fmod_tensor_cpu(self.to(dt), Tensor::full({}, other, dt, self.device()));
}

Tensor subtract_tensor_cpu(const Tensor& self, const Tensor& other, Scalar alpha) {
    // alpha == 1 is by far the common call, and the scaled form costs a
    // Scalar conversion per element, so it keeps its own loop.
    if (!alpha.isComplex() && alpha.toDouble() == 1.0) {
        return binary_same_kernel<true>(self, other,
            [](auto x, auto y) { return x - y; }, "subtract");
    }
    return binary_same_kernel<true>(self, other,
        [alpha](auto x, auto y) {
            using T = decltype(x);
            if constexpr (is_complex_type_v<T> || std::is_floating_point_v<T>) {
                return x - y * alpha.to<T>();
            } else {
                return static_cast<T>(x - y * alpha.to<double>());
            }
        }, "subtract");
}
Tensor subtract_scalar_cpu(const Tensor& self, Scalar other, Scalar alpha) {
    DType dt = scalar_promote(self.dtype(), other);
    return subtract_tensor_cpu(self.to(dt),
                               Tensor::full({}, other, dt, self.device()), alpha);
}
Tensor multiply_tensor_cpu(const Tensor& self, const Tensor& other) {
    return binary_same_kernel<true>(self, other, [](auto x, auto y) { return x * y; }, "multiply");
}
Tensor multiply_scalar_cpu(const Tensor& self, Scalar other) {
    if (isComplexType(self.dtype()) || other.isComplex()) {
        DType dt;
        if (isComplexType(self.dtype())) {
            dt = other.isComplex() ? promoteTypes(self.dtype(), other.dtype())
                                   : self.dtype();
        } else {
            // weak-scalar rule: int tensors go to complex64; float tensors
            // widen through their own complex width
            dt = promoteTypes(
                isFloatingType(self.dtype()) ? toComplexType(self.dtype())
                                             : DType::ComplexFloat,
                other.dtype());
        }
        return complex_unary_op_kernel(self.to(dt), [other](auto x) {
            return x * other.to<std::decay_t<decltype(x)>>();
        });
    }
    return dtype_unary_kernel(self, [other](auto x) {
        return static_cast<decltype(x)>(x * other.to<double>());
    }, "multiply");
}
// ---------------------------------------------------------------------------
// Division with an explicit rounding mode
// ---------------------------------------------------------------------------
namespace {

enum class DivRounding { kTrue, kTrunc, kFloor };

DivRounding parse_div_rounding(const std::optional<std::string>& mode) {
    if (!mode.has_value()) return DivRounding::kTrue;
    if (*mode == "trunc") return DivRounding::kTrunc;
    if (*mode == "floor") return DivRounding::kFloor;
    TP_THROW(RuntimeError,
             std::string("div expected rounding_mode to be one of None, 'trunc' "
                         "or 'floor' but found '") + *mode + "'");
}

// The hardware quotient truncates toward zero, so a remainder whose sign
// disagrees with the divisor sits one step above the floor.
template <typename T>
inline T int_floor_div(T x, T y) {
    T q = static_cast<T>(x / y);
    T r = static_cast<T>(x - q * y);
    if (r != T(0) && ((r < T(0)) != (y < T(0)))) q = static_cast<T>(q - T(1));
    return q;
}

Tensor div_rounded_core(const Tensor& a, const Tensor& b, DivRounding rounding) {
    if (rounding == DivRounding::kTrue) return true_divide_core(a, b);
    // Rounded division stays in the input dtype: an integral pair must come
    // back integral, which the float promotion of true division loses.
    const bool floor_mode = (rounding == DivRounding::kFloor);
    return binary_same_kernel(a, b, [floor_mode](auto x, auto y) -> decltype(x) {
        using T = decltype(x);
        if constexpr (std::is_integral_v<T>) {
            if (y == T(0)) TP_THROW(RuntimeError, "ZeroDivisionError");
            return floor_mode ? int_floor_div<T>(x, y) : static_cast<T>(x / y);
        } else {
            // Half/BFloat16 round through Float32, the width their arithmetic
            // is defined at; float and double keep their own.
            using C = std::conditional_t<std::is_same_v<T, double>, double, float>;
            const C q = static_cast<C>(x) / static_cast<C>(y);
            return static_cast<T>(floor_mode ? std::floor(q) : std::trunc(q));
        }
    }, "div");
}

Tensor div_rounded_scalar(const Tensor& self, Scalar other, DivRounding rounding) {
    if (rounding == DivRounding::kTrue) return true_divide_scalar_cpu(self, other);
    const DType dt = scalar_promote(self.dtype(), other);
    return div_rounded_core(self.to(dt), Tensor::full({}, other, dt, self.device()),
                            rounding);
}

}  // namespace

Tensor div_mode_tensor_cpu(const Tensor& self, const Tensor& other,
                           std::optional<std::string> rounding_mode) {
    return div_rounded_core(self, other, parse_div_rounding(rounding_mode));
}
Tensor div_mode_scalar_cpu(const Tensor& self, Scalar other,
                           std::optional<std::string> rounding_mode) {
    return div_rounded_scalar(self, other, parse_div_rounding(rounding_mode));
}
Tensor floor_divide_cpu(const Tensor& self, const Tensor& other) {
    return div_rounded_core(self, other, DivRounding::kFloor);
}
Tensor floor_divide_scalar_cpu(const Tensor& self, Scalar other) {
    return div_rounded_scalar(self, other, DivRounding::kFloor);
}

Tensor negative_cpu(const Tensor& self) {
    if (isComplexType(self.dtype())) {
        return complex_unary_op_kernel(self, [](auto x) { return -x; });
    }
    return dtype_unary_kernel(self, [](auto x) { return static_cast<decltype(x)>(-x); }, "negative");
}
Tensor positive_cpu(const Tensor& self) { return self.clone(); }

// ===========================================================================
// Comparisons / logic (TensorCompare.cpp / BinaryOps.cpp:1498)
// ===========================================================================

Tensor greater_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return x > y; }, "greater");
}
Tensor greater_equal_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return x >= y; }, "greater_equal");
}
Tensor less_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return x < y; }, "less");
}
Tensor less_equal_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return x <= y; }, "less_equal");
}
Tensor not_equal_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return x != y; }, "not_equal");
}
Tensor signbit_cpu(const Tensor& self) {
    return bool_unary_kernel(self, [](auto x) {
        return static_cast<double>(x) < 0.0;
    }, "signbit");
}
Tensor logical_not_cpu(const Tensor& self) {
    return bool_unary_kernel(self, [](auto x) { return !static_cast<bool>(x); }, "logical_not");
}
Tensor logical_and_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return static_cast<bool>(x) && static_cast<bool>(y); },
                              "logical_and");
}
Tensor logical_or_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return static_cast<bool>(x) || static_cast<bool>(y); },
                              "logical_or");
}
Tensor logical_xor_cpu(const Tensor& a, const Tensor& b) {
    return binary_bool_kernel(a, b, [](auto x, auto y) { return static_cast<bool>(x) != static_cast<bool>(y); },
                              "logical_xor");
}
Tensor isfinite_cpu(const Tensor& self) {
    // TensorCompare.cpp:474
    return bool_unary_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_floating_point_v<T>)
            return std::isfinite(static_cast<double>(x));
        else return true;
    }, "isfinite");
}
Tensor isinf_cpu(const Tensor& self) {
    // TensorCompare.cpp:458: integral tensors never infinite
    return bool_unary_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_floating_point_v<T>)
            return std::isinf(static_cast<double>(x));
        else return false;
    }, "isinf");
}
Tensor isnan_cpu(const Tensor& self) {
    // TensorCompare.cpp:435: self != self
    return bool_unary_kernel(self, [](auto x) {
        return static_cast<double>(x) != static_cast<double>(x);
    }, "isnan");
}
Tensor isneginf_cpu(const Tensor& self) {
    return bool_unary_kernel(self, [](auto x) {
        return static_cast<double>(x) == -std::numeric_limits<double>::infinity();
    }, "isneginf");
}
Tensor isposinf_cpu(const Tensor& self) {
    return bool_unary_kernel(self, [](auto x) {
        return static_cast<double>(x) == std::numeric_limits<double>::infinity();
    }, "isposinf");
}

// ===========================================================================
// Math functions
// ===========================================================================

Tensor reciprocal_cpu(const Tensor& self) {
    // the old float_math_kernel path silently dropped the imaginary part.
    if (isComplexType(self.dtype())) {
        return complex_unary_op_kernel(self, [](auto z) {
            using T = decltype(z);
            return static_cast<T>(1) / z;
        });
    }
    return float_math_kernel(self, [](double x) { return 1.0 / x; }, "reciprocal");
}
Tensor sgn_cpu(const Tensor& self) {
    return dtype_unary_kernel(self, [](auto x) -> decltype(x) {
        using T = decltype(x);
        double d = static_cast<double>(x);
        if (d != d) return static_cast<T>(x);           // NaN passthrough
        if (d > 0) return static_cast<T>(1);
        if (d < 0) return static_cast<T>(-1);
        return static_cast<T>(0);
    }, "sgn");
}
Tensor exp2_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return std::exp2(x); }, "exp2");
}
Tensor sinc_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) {
        double px = M_PI * x;
        return std::fabs(px) < 1e-30 ? 1.0 : std::sin(px) / px;
    }, "sinc");
}
Tensor deg2rad_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return x * (M_PI / 180.0); }, "deg2rad");
}
Tensor rad2deg_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) { return x * (180.0 / M_PI); }, "rad2deg");
}
Tensor fix_cpu(const Tensor& self) {
    return dtype_unary_kernel(self, [](auto x) -> decltype(x) {
        if constexpr (std::is_floating_point_v<decltype(x)>) return std::trunc(x);
        else return x;
    }, "fix");
}
Tensor erfinv_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double x) {
        return special_math::calc_erfinv(x);
    }, "erfinv");
}
Tensor logit_cpu(const Tensor& self, std::optional<Scalar> eps) {
    double e = eps.has_value() ? eps->toDouble() : -1.0;
    return float_math_kernel(self, [e](double p) {
        if (e >= 0) p = std::min(std::max(p, e), 1.0 - e);
        return std::log(p / (1.0 - p));
    }, "logit");
}
Tensor digamma_cpu(const Tensor& self) {
    return float_math_kernel(self, [](double v) {
        if (v <= 0 && v == std::floor(v)) return std::numeric_limits<double>::quiet_NaN();
        double r = 0;
        while (v < 6.0) { r -= 1.0 / v; v += 1.0; }
        double inv = 1.0 / v, inv2 = inv * inv;
        r += std::log(v) - 0.5 * inv
             - inv2 * (1.0/12.0 - inv2 * (1.0/120.0 - inv2 * (1.0/252.0 - inv2 * (1.0/240.0 - inv2 / 132.0))));
        return r;
    }, "digamma");
}
Tensor i0_cpu(const Tensor& self) {
    // Modified Bessel I0.  The Chebyshev expansion holds across the whole
    // range; the ((|x|/2)^k / k!)^2 series it replaces needs more terms than
    // any fixed cap allows once |x| passes ~50.
    return float_math_kernel(self, [](double v) {
        return tensorplay::special_math::modified_bessel_i0_forward(v);
    }, "i0");
}
Tensor nan_to_num_cpu(const Tensor& self, Scalar nan,
                      std::optional<Scalar> posinf, std::optional<Scalar> neginf) {
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    double nan_v = nan.toDouble();
    bool has_pos = posinf.has_value(), has_neg = neginf.has_value();
    double pos_v = has_pos ? posinf->toDouble() : std::numeric_limits<double>::infinity();
    double neg_v = has_neg ? neginf->toDouble() : -std::numeric_limits<double>::infinity();
#define TP_NTN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        ctype pv, nv; \
        if (has_pos) pv = static_cast<ctype>(pos_v); \
        else pv = std::numeric_limits<ctype>::max(); \
        if (has_neg) nv = static_cast<ctype>(neg_v); \
        else nv = std::numeric_limits<ctype>::lowest(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t i = b; i < e; ++i) { \
                ctype v = sp[i]; \
                if constexpr (std::is_floating_point_v<ctype>) { \
                    double dv = static_cast<double>(v); \
                    if (dv != dv) v = static_cast<ctype>(nan_v); \
                    else if (v == std::numeric_limits<ctype>::infinity()) v = pv; \
                    else if (v == -std::numeric_limits<ctype>::infinity()) v = nv; \
                } \
                dp[i] = v; \
            } \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NTN_CASE)
        default: TP_THROW(TypeError, "nan_to_num: unsupported dtype");
    }
#undef TP_NTN_CASE
    return out;
}

Tensor xlogy_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        return tensorplay::special_math::calc_xlogy(x, y);
    }, "xlogy");
}
Tensor logaddexp_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        double m = std::max(x, y);
        if (m == -std::numeric_limits<double>::infinity()) return m;
        if (m != m) return m;  // NaN propagates
        return m + std::log1p(std::exp(-std::fabs(x - y)));
    }, "logaddexp");
}
Tensor logaddexp2_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        double m = std::max(x, y);
        if (m == -std::numeric_limits<double>::infinity()) return m;
        if (m != m) return m;
        return m + std::log1p(std::exp2(-std::fabs(x - y))) / M_LN2;
    }, "logaddexp2");
}
Tensor copysign_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        return std::copysign(x, y);
    }, "copysign");
}
Tensor copysign_scalar_cpu(const Tensor& self, Scalar other) {
    // The sign comes from the scalar alone, so the divisor width never
    // participates in promotion -- Float32 carries every sign bit exactly.
    return copysign_cpu(self, Tensor::full({}, other, DType::Float32, self.device()));
}
Tensor hypot_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        return std::hypot(x, y);
    }, "hypot");
}

Tensor atan2_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        return std::atan2(x, y);
    }, "atan2");
}
Tensor nextafter_cpu(const Tensor& a, const Tensor& b) {
    // The step must happen in the element dtype: a double-precision step from
    // a Float32 value rounds back to the original number when narrowed.
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a.shape()),
        static_cast<std::vector<int64_t>>(b.shape()));
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (!isFloatingType(dt)) dt = DType::Float32;
    Tensor ac = a.to(dt).expand(out_shape).contiguous();
    Tensor bc = b.to(dt).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, dt, a.device());
    int64_t n = out.numel();
    if (dt == DType::Float64) {
        const double* ap = ac.data_ptr<double>();
        const double* bp = bc.data_ptr<double>();
        double* dp = out.data_ptr<double>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = std::nextafter(ap[i], bp[i]);
        });
    } else {
        const float* ap = ac.data_ptr<float>();
        const float* bp = bc.data_ptr<float>();
        float* dp = out.data_ptr<float>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = std::nextafter(ap[i], bp[i]);
        });
    }
    return out;
}
Tensor gcd_cpu(const Tensor& a, const Tensor& b) {
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (isFloatingType(dt)) TP_THROW(TypeError, "gcd only supports integral tensors");
    return binary_same_kernel(a, b, [](auto x, auto y) -> decltype(x) {
        using T = decltype(x);
        long long ux = static_cast<long long>(x < T(0) ? -x : x);
        long long uy = static_cast<long long>(y < T(0) ? -y : y);
        while (uy) { long long t = ux % uy; ux = uy; uy = t; }
        return static_cast<T>(ux);
    }, "gcd");
}
Tensor lcm_cpu(const Tensor& a, const Tensor& b) {
    DType dt = promoteTypes(a.dtype(), b.dtype());
    if (isFloatingType(dt)) TP_THROW(TypeError, "lcm only supports integral tensors");
    return binary_same_kernel(a, b, [](auto x, auto y) -> decltype(x) {
        using T = decltype(x);
        long long ux = static_cast<long long>(static_cast<float>(x) < 0.0f ? -x : x);
        long long uy = static_cast<long long>(static_cast<float>(y) < 0.0f ? -y : y);
        long long g = ux, t2 = uy;
        while (t2) { long long t3 = g % t2; g = t2; t2 = t3; }
        if (g == 0) return static_cast<T>(0);
        return static_cast<T>(ux / g * uy);
    }, "lcm");
}
Tensor heaviside_cpu(const Tensor& a, const Tensor& values) {
    return binary_same_kernel(a, values, [](auto x, auto v) -> decltype(x) {
        using T = decltype(x);
        double xd = static_cast<double>(x);
        if (xd < 0.0) return static_cast<T>(0);
        if (xd == 0.0) return static_cast<T>(v);
        return static_cast<T>(1);
    }, "heaviside");
}

// ===========================================================================
// Clamp family
// ===========================================================================

namespace clamp_row {
#if defined(__x86_64__)
inline bool avx512_ok() {
    static const bool ok = __builtin_cpu_supports("avx512f") != 0 &&
                           __builtin_cpu_supports("avx512vl") != 0 &&
                           __builtin_cpu_supports("avx512dq") != 0;
    return ok;
}

// NaN propagation: the finite bound is the first source of max/min so a NaN
// lane (second source) flows through untouched, matching the scalar ternary.
__attribute__((target("avx512f")))
inline void f32_512(const float* in, float* out, int64_t n, float lo, float hi) {
    const __m512 vlo = _mm512_set1_ps(lo), vhi = _mm512_set1_ps(hi);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(in + i);
        v = _mm512_min_ps(vhi, _mm512_max_ps(vlo, v));
        _mm512_storeu_ps(out + i, v);
    }
    for (; i < n; ++i) {
        float v = in[i];
        v = v < lo ? lo : v;
        v = v > hi ? hi : v;
        out[i] = v;
    }
}

__attribute__((target("avx512f")))
inline void f64_512(const double* in, double* out, int64_t n, double lo, double hi) {
    const __m512d vlo = _mm512_set1_pd(lo), vhi = _mm512_set1_pd(hi);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(in + i);
        v = _mm512_min_pd(vhi, _mm512_max_pd(vlo, v));
        _mm512_storeu_pd(out + i, v);
    }
    for (; i < n; ++i) {
        double v = in[i];
        v = v < lo ? lo : v;
        v = v > hi ? hi : v;
        out[i] = v;
    }
}
#endif
}  // namespace clamp_row

namespace {
// Shared contiguous implementation: one streaming pass, no intermediate
// tensor, bounds applied together.  NaN input stays NaN (comparisons are
// false), matching the per-bound ternary kernels below.
template <typename T>
Tensor clamp_range_contig(const Tensor& self, T lo, T hi, bool has_lo, bool has_hi) {
    Tensor self_c = self.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    const T* in = self_c.data_ptr<T>();
    T* out = result.data_ptr<T>();
    const int64_t n = self_c.numel();
    tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            T v = in[i];
            if (has_lo && v < lo) v = lo;
            if (has_hi && v > hi) v = hi;
            out[i] = v;
        }
    });
    return result;
}
}  // namespace

Tensor clamp_min_scalar_cpu(const Tensor& self, Scalar min) {
    if (self.dtype() == DType::Float32 && self.is_contiguous()) {
        Tensor self_c = self.contiguous();
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
        float lo = static_cast<float>(min.toDouble());
        const float* in = self_c.data_ptr<float>();
        float* out = result.data_ptr<float>();
        const int64_t n = self_c.numel();
#if defined(__x86_64__)
        if (clamp_row::avx512_ok()) {
            tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                clamp_row::f32_512(in + b, out + b, e - b, lo,
                                   std::numeric_limits<float>::infinity());
            });
            return result;
        }
#endif
        tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) out[i] = in[i] < lo ? lo : in[i];
        });
        return result;
    }
    double lo = min.toDouble();
    return dtype_unary_kernel(self, [lo](auto x) -> decltype(x) {
        using T = decltype(x);
        return static_cast<double>(x) < lo ? static_cast<T>(lo) : static_cast<T>(x);
    }, "clamp_min");
}
Tensor clamp_max_scalar_cpu(const Tensor& self, Scalar max) {
    if (self.dtype() == DType::Float32 && self.is_contiguous()) {
        Tensor self_c = self.contiguous();
        Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
        float hi = static_cast<float>(max.toDouble());
        const float* in = self_c.data_ptr<float>();
        float* out = result.data_ptr<float>();
        const int64_t n = self_c.numel();
#if defined(__x86_64__)
        if (clamp_row::avx512_ok()) {
            tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                clamp_row::f32_512(in + b, out + b, e - b,
                                   -std::numeric_limits<float>::infinity(), hi);
            });
            return result;
        }
#endif
        tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) out[i] = in[i] > hi ? hi : in[i];
        });
        return result;
    }
    double hi = max.toDouble();
    return dtype_unary_kernel(self, [hi](auto x) -> decltype(x) {
        using T = decltype(x);
        return static_cast<double>(x) > hi ? static_cast<T>(hi) : static_cast<T>(x);
    }, "clamp_max");
}
Tensor clamp_min_tensor_cpu(const Tensor& self, const Tensor& min) {
    return binary_same_kernel(self, min, [](auto x, auto m) -> decltype(x) {
        using T = decltype(x);
        return static_cast<double>(m) > static_cast<double>(x) ? static_cast<T>(m) : static_cast<T>(x);
    }, "clamp_min");
}
Tensor clamp_max_tensor_cpu(const Tensor& self, const Tensor& max) {
    return binary_same_kernel(self, max, [](auto x, auto m) -> decltype(x) {
        using T = decltype(x);
        return static_cast<double>(m) < static_cast<double>(x) ? static_cast<T>(m) : static_cast<T>(x);
    }, "clamp_max");
}
Tensor clip_cpu(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    if (min.has_value() && max.has_value()) {
        const double lo = min->toDouble();
        const double hi = max->toDouble();
        if (self.dtype() == DType::Float32 && self.is_contiguous()) {
            Tensor self_c = self.contiguous();
            Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
            const float lo32 = static_cast<float>(lo), hi32 = static_cast<float>(hi);
            const float* in = self_c.data_ptr<float>();
            float* out = result.data_ptr<float>();
            const int64_t n = self_c.numel();
#if defined(__x86_64__)
            if (clamp_row::avx512_ok()) {
                tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                    clamp_row::f32_512(in + b, out + b, e - b, lo32, hi32);
                });
                return result;
            }
#endif
            tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) {
                    float v = in[i];
                    v = v < lo32 ? lo32 : v;
                    v = v > hi32 ? hi32 : v;
                    out[i] = v;
                }
            });
            return result;
        }
        if (self.dtype() == DType::Float64 && self.is_contiguous()) {
            Tensor self_c = self.contiguous();
            Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
            const double* in = self_c.data_ptr<double>();
            double* out = result.data_ptr<double>();
            const int64_t n = self_c.numel();
#if defined(__x86_64__)
            if (clamp_row::avx512_ok()) {
                tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                    clamp_row::f64_512(in + b, out + b, e - b, lo, hi);
                });
                return result;
            }
#endif
            tensorplay::parallel::parallel_for(0, n, 8192, [&](int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) {
                    double v = in[i];
                    v = v < lo ? lo : v;
                    v = v > hi ? hi : v;
                    out[i] = v;
                }
            });
            return result;
        }
        Tensor r = clamp_min_scalar_cpu(self, *min);
        return clamp_max_scalar_cpu(r, *max);
    }
    if (min.has_value()) return clamp_min_scalar_cpu(self, *min);
    if (max.has_value()) return clamp_max_scalar_cpu(self, *max);
    return self.clone();
}
Tensor& clamp__cpu(Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor r = clip_cpu(self, std::move(min), std::move(max));
    self.copy_(r);
    return self;
}

// ===========================================================================
// Activations (Activation.cpp:525/:541/:202/:697)
// ===========================================================================

Tensor selu_cpu(const Tensor& self) {
    // Activation.cpp:525: scale * max(0,x) + scale * alpha * (exp(min(0,x)) - 1)
    constexpr double kAlpha = 1.6732632423543772848170429916717;
    constexpr double kScale = 1.0507009873554804934193349852946;
    return dtype_unary_kernel(self, [](auto x) -> decltype(x) {
        using T = decltype(x);
        double v = static_cast<double>(x);
        return static_cast<T>(v > 0 ? kScale * v : kScale * kAlpha * (std::exp(v) - 1.0));
    }, "selu");
}
Tensor celu_cpu(const Tensor& self, Scalar alpha) {
    // Activation.cpp:541: max(0,x) + alpha * (exp(x/alpha) - 1) on the negative side
    double a = alpha.toDouble();
    return dtype_unary_kernel(self, [a](auto x) -> decltype(x) {
        using T = decltype(x);
        double v = static_cast<double>(x);
        return static_cast<T>(v > 0 ? v : a * (std::exp(v / a) - 1.0));
    }, "celu");
}
Tensor hardshrink_cpu(const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    // lambd.to<scalar_t>(), so float32 boundary values compare exactly.
    return dtype_unary_kernel(self, [l](auto x) -> decltype(x) {
        using T = decltype(x);
        const double lt = static_cast<double>(static_cast<T>(l));
        double v = static_cast<double>(x);
        return (v >= -lt && v <= lt) ? static_cast<T>(0) : x;
    }, "hardshrink");
}
Tensor softshrink_cpu(const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    return dtype_unary_kernel(self, [l](auto x) -> decltype(x) {
        using T = decltype(x);
        const double lt = static_cast<double>(static_cast<T>(l));
        double v = static_cast<double>(x);
        if (v > lt) return static_cast<T>(v - lt);
        if (v < -lt) return static_cast<T>(v + lt);
        return static_cast<T>(v * 0.0);
    }, "softshrink");
}
// passes through where self is outside the inclusive [-lambd, lambd] band.
Tensor hardshrink_backward_cpu(const Tensor& grad_out, const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    return binary_same_kernel(grad_out, self, [l](auto g, auto s) -> decltype(g) {
        using T = decltype(g);
        const double lt = static_cast<double>(static_cast<T>(l));
        double v = static_cast<double>(s);
        return (v >= -lt && v <= lt) ? static_cast<T>(0) : g;
    }, "hardshrink_backward");
}
Tensor softshrink_backward_cpu(const Tensor& grad_output, const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    return binary_same_kernel(grad_output, self, [l](auto g, auto s) -> decltype(g) {
        using T = decltype(g);
        const double lt = static_cast<double>(static_cast<T>(l));
        double v = static_cast<double>(s);
        return (v >= -lt && v <= lt) ? static_cast<T>(0) : g;
    }, "softshrink_backward");
}
// where `output` is the saved forward result of sigmoid.
Tensor sigmoid_backward_cpu(const Tensor& grad_output, const Tensor& output) {
    return binary_same_kernel(grad_output, output, [](auto g, auto o) -> decltype(g) {
        using T = decltype(o);
        return g * o * (static_cast<T>(1) - o);
    }, "sigmoid_backward");
}
// where `output` is the saved forward result of tanh.
Tensor tanh_backward_cpu(const Tensor& grad_output, const Tensor& output) {
    return binary_same_kernel(grad_output, output, [](auto g, auto o) -> decltype(g) {
        using T = decltype(o);
        return g * (static_cast<T>(1) - o * o);
    }, "tanh_backward");
}
// gradient is dy/(x(1-x)) inside [0,1], NaN outside, and dy*inf at exact
// 0/1; with eps>=0 values outside [eps, 1-eps] (compared in scalar_t) are
// masked to zero.
Tensor logit_backward_cpu(const Tensor& grad_output, const Tensor& self, std::optional<Scalar> eps) {
    double e = eps.has_value() ? eps->toDouble() : -1.0;
    return binary_same_kernel(grad_output, self, [e](auto g, auto s) -> decltype(g) {
        using T = decltype(s);
        const T zero = static_cast<T>(0);
        const T one = static_cast<T>(1);
        if (e < 0) {
            if (s < zero || s > one) return std::numeric_limits<T>::quiet_NaN();
            if (s == zero || s == one) return g * std::numeric_limits<T>::infinity();
            return g / (s * (one - s));
        }
        // (float32 1 - 0.2f == 0.8f), so the band check must too.
        const T lo = static_cast<T>(e);
        const T hi = one - lo;
        if (s < lo || s > hi) return zero;
        if (s == zero || s == one) return g * std::numeric_limits<T>::infinity();
        return g / (s * (one - s));
    }, "logit_backward");
}
Tensor threshold_cpu(const Tensor& self, Scalar threshold, Scalar value) {
    double t = threshold.toDouble(), val = value.toDouble();
    return dtype_unary_kernel(self, [t, val](auto x) -> decltype(x) {
        using T = decltype(x);
        return static_cast<double>(x) <= t ? static_cast<T>(val) : static_cast<T>(x);
    }, "threshold");
}
Tensor prelu_cpu(const Tensor& self, const Tensor& weight) {
    // Activation.cpp:697: weight shared when numel==1, otherwise per-channel
    // (channel = dim 0 for 1-D input, dim 1 for >=2-D input).
    Tensor wc = weight.contiguous();
    if (wc.numel() == 1) {
        double w = wc.data_ptr<double>() ? wc.item().toDouble() : 0.0;
        return dtype_unary_kernel(self, [w](auto x) -> decltype(x) {
            using T = decltype(x);
            double v = static_cast<double>(x);
            return static_cast<T>(v > 0 ? v : w * v);
        }, "prelu");
    }
    int64_t channels = self.dim() >= 1 ? self.size(0) : 1;
    int64_t per_ch = self.numel() / std::max<int64_t>(channels, 1);
    if (self.dim() >= 2) {
        // channel dim is dim 1; outer = dim 0
        int64_t N = self.size(0), C = self.size(1);
        per_ch = self.numel() / std::max<int64_t>(N * C, 1);
        channels = C;
    }
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
#define TP_PRELU_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        const ctype* wp = wc.to(self.dtype()).contiguous().data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t i = b; i < e; ++i) { \
                int64_t ch = 0; \
                if (self.dim() >= 2) ch = (i / per_ch) % channels; \
                else ch = (per_ch > 0) ? (i / per_ch) % channels : 0; \
                double v = static_cast<double>(sp[i]); \
                double w = static_cast<double>(wp[ch]); \
                dp[i] = static_cast<ctype>(v > 0 ? v : w * v); \
            } \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_PRELU_CASE)
        default: TP_THROW(TypeError, "prelu: unsupported dtype");
    }
#undef TP_PRELU_CASE
    return out;
}

// ===========================================================================
// Reductions (ReduceOps.cpp / Sorting.cpp anchors)
// ===========================================================================

// zero_numel_check_dims): reducing an empty tensor is only valid along an
// explicitly given non-empty dim; a full reduction has no identity.
static void zero_numel_check_dims(const Tensor& self, const std::vector<int64_t>& dims,
                                  const char* fn_name) {
    if (dims.empty()) {
        TP_THROW(RuntimeError, fn_name,
                 ": Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    const int64_t nd = self.dim();
    for (int64_t d : dims) {
        if (d < 0) d += nd;
        TP_CHECK_INDEX(self.size(d) != 0, fn_name,
                       ": Expected reduction dim ", d, " to have non-zero size.");
    }
}

Tensor amax_cpu(const Tensor& self, const std::vector<int64_t>& dim_in, bool keepdim) {
    // ReduceOps.cpp:1801 amax_out
    if (self.numel() == 0) zero_numel_check_dims(self, dim_in, "amax()");
    std::vector<int64_t> resolved = dim_in;
    if (resolved.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) resolved.push_back(i);
    }
    const std::vector<int64_t>& dim = resolved;
    return reduce_dims_impl<double>(
        self, dim, keepdim, isFloatingType(self.dtype()) ? self.dtype() : self.dtype(),
        -std::numeric_limits<double>::infinity(),
        // (strictly propagates), matching slice_max_kernel on CUDA.
        [](double acc, double v) { return (v != v || v > acc) ? v : acc; },
        [](double acc) { return acc; });
}

Tensor amin_cpu(const Tensor& self, const std::vector<int64_t>& dim_in, bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim_in, "amin()");
    std::vector<int64_t> resolved = dim_in;
    if (resolved.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) resolved.push_back(i);
    }
    const std::vector<int64_t>& dim = resolved;
    return reduce_dims_impl<double>(
        self, dim, keepdim, self.dtype(),
        std::numeric_limits<double>::infinity(),
        // (strictly propagates), matching slice_min_kernel on CUDA.
        [](double acc, double v) { return (v != v || v < acc) ? v : acc; },
        [](double acc) { return acc; });
}

std::tuple<Tensor, Tensor> aminmax_cpu(const Tensor& self, const std::vector<int64_t>& dim_in, bool keepdim) {
    if (self.numel() == 0) {
        // ReduceOps.cpp aminmax meta: full reduction has no identity; dim
        // reductions must name a non-empty dim.
        if (dim_in.empty()) {
            TP_THROW(RuntimeError, "aminmax(): cannot compute aminmax over an empty dimension as "
                     "the operation has no identity.");
        }
        zero_numel_check_dims(self, dim_in, "aminmax");
    }
    std::vector<int64_t> resolved = dim_in;
    if (resolved.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) resolved.push_back(i);
    }
    const std::vector<int64_t>& dim = resolved;
    Tensor mn = amin_cpu(self, dim, keepdim);
    Tensor mx = amax_cpu(self, dim, keepdim);
    return {mn, mx};
}

Tensor logsumexp_cpu(const Tensor& self, int64_t dim, bool keepdim) {
    // ReduceOps.cpp:1578; requires floating input
    if (!isFloatingType(self.dtype()))
        TP_THROW(RuntimeError, "logsumexp(): Expected floating point type");
    LseState init{ -std::numeric_limits<double>::infinity(), 0.0, false };
    return reduce_dims_impl<LseState>(
        self, {dim}, keepdim, self.dtype(), init,
        [](LseState acc, double v) -> LseState {
            if (v != v) { acc.nan_flag = true; return acc; }
            if (acc.m == -std::numeric_limits<double>::infinity()) { acc.m = v; acc.s = 1.0; return acc; }
            if (v > acc.m) { acc.s = acc.s * std::exp(acc.m - v) + 1.0; acc.m = v; }
            else acc.s += std::exp(v - acc.m);
            return acc;
        },
        [](LseState acc) {
            if (acc.nan_flag) return std::numeric_limits<double>::quiet_NaN();
            if (acc.m == -std::numeric_limits<double>::infinity()) return acc.m;
            return acc.m + std::log(acc.s);
        });
}

Tensor nansum_cpu(const Tensor& self, const std::vector<int64_t>& dim_in, bool keepdim) {
    // ReduceOps.cpp:1310 nansum_out: NaN treated as 0
    DType out_dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    std::vector<int64_t> dim = dim_in;
    if (dim.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) dim.push_back(i);
    }
    return reduce_dims_impl<double>(
        self, dim, keepdim, out_dt, 0.0,
        [](double acc, double v) { return (v != v) ? acc : acc + v; },
        [](double acc) { return acc; });
}

std::tuple<Tensor, Tensor> cummax_cpu(const Tensor& self, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor vals = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()), sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()), DType::Int64, sc.device());
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
#define TP_CM_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = vals.data_ptr<ctype>(); \
        int64_t* ip = idxs.data_ptr<int64_t>(); \
        parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t si = b; si < e; ++si) { \
                int64_t o = si / inner, in2 = si % inner; \
                const ctype* s = sp + o * d_size * inner + in2; \
                ctype* v = vp + o * d_size * inner + in2; \
                int64_t* ix = ip + o * d_size * inner + in2; \
                ctype best = s[0]; int64_t bi = 0; \
                v[0] = best; ix[0] = 0; \
                for (int64_t j = 1; j < d_size; ++j) { \
                    if (s[j * inner] > best) { best = s[j * inner]; bi = j; } \
                    v[j * inner] = best; ix[j * inner] = bi; \
                } \
            } \
        }); \
        break; \
    }
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_CM_CASE)
        default: TP_THROW(TypeError, "cummax: unsupported dtype");
    }
#undef TP_CM_CASE
    return {vals, idxs};
}

std::tuple<Tensor, Tensor> cummin_cpu(const Tensor& self, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor vals = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()), sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()), DType::Int64, sc.device());
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
#define TP_CMIN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = vals.data_ptr<ctype>(); \
        int64_t* ip = idxs.data_ptr<int64_t>(); \
        parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t si = b; si < e; ++si) { \
                int64_t o = si / inner, in2 = si % inner; \
                const ctype* s = sp + o * d_size * inner + in2; \
                ctype* v = vp + o * d_size * inner + in2; \
                int64_t* ix = ip + o * d_size * inner + in2; \
                ctype best = s[0]; int64_t bi = 0; \
                v[0] = best; ix[0] = 0; \
                for (int64_t j = 1; j < d_size; ++j) { \
                    if (s[j * inner] < best) { best = s[j * inner]; bi = j; } \
                    v[j * inner] = best; ix[j * inner] = bi; \
                } \
            } \
        }); \
        break; \
    }
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_CMIN_CASE)
        default: TP_THROW(TypeError, "cummin: unsupported dtype");
    }
#undef TP_CMIN_CASE
    return {vals, idxs};
}

std::tuple<Tensor, Tensor> std_mean_cpu(const Tensor& self, std::vector<int64_t> dim,
                                        bool unbiased, bool keepdim) {
    auto vr_mean = mean_var_over_dims(self, dim, unbiased, keepdim);
    Tensor std_t = vr_mean.first.sqrt();
    return {std_t, vr_mean.second};
}

std::tuple<Tensor, Tensor> var_mean_cpu(const Tensor& self, std::vector<int64_t> dim,
                                        bool unbiased, bool keepdim) {
    auto vr_mean = mean_var_over_dims(self, dim, unbiased, keepdim);
    return {vr_mean.first, vr_mean.second};
}

Tensor nanmedian_cpu(const Tensor& self) {
    // Flattened median ignoring NaNs; lower-middle convention like median().
    Tensor flat = self.to(isFloatingType(self.dtype()) ? DType::Float64 : DType::Int64)
                      .reshape({self.numel()});
    std::vector<double> vals;
    vals.reserve(flat.numel());
    const double* p = flat.data_ptr<double>();
    for (int64_t i = 0; i < flat.numel(); ++i) {
        if (!(p[i] != p[i])) vals.push_back(p[i]);
    }
    DType out_dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    if (vals.empty()) {
        return Tensor::zeros({}, out_dt, self.device());
    }
    std::sort(vals.begin(), vals.end());
    double med = vals[(vals.size() - 1) / 2];
    return Tensor::zeros({}, out_dt, self.device()).fill_(Scalar(med));
}

std::tuple<Tensor, Tensor> mode_cpu(const Tensor& self, int64_t dim, bool keepdim) {
    // Most frequent value per slice; ties -> smallest value, earliest index.
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i) out_shape.push_back(i == dim ? 1 : sc.size(i));
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor vals = Tensor::empty(out_shape, sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(out_shape, DType::Int64, sc.device());
#define TP_MODE_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = vals.data_ptr<ctype>(); \
        int64_t* ip = idxs.data_ptr<int64_t>(); \
        for (int64_t si = 0; si < outer * inner; ++si) { \
            int64_t o = si / inner, in2 = si % inner; \
            const ctype* s = sp + o * d_size * inner + in2; \
            std::vector<std::pair<ctype, int64_t>> buf(d_size); \
            for (int64_t j = 0; j < d_size; ++j) buf[j] = {s[j * inner], j}; \
            std::sort(buf.begin(), buf.end(), [](const std::pair<ctype,int64_t>& a2, const std::pair<ctype,int64_t>& b2) { \
                if (!(a2.first < b2.first) && !(b2.first < a2.first)) return a2.second < b2.second; \
                return a2.first < b2.first; \
            }); \
            ctype best_v = buf[0].first; int64_t best_c = 0, best_i = buf[0].second; \
            int64_t run = 0; \
            for (int64_t j = 0; j < d_size; ++j) { \
                bool same_as_prev = j > 0 && !(buf[j].first < buf[j-1].first) && !(buf[j-1].first < buf[j].first); \
                run = same_as_prev ? run + 1 : 1; \
                if (run > best_c) { best_c = run; best_v = buf[j].first; best_i = buf[j].second; } \
            } \
            vp[si] = best_v; \
            ip[si] = best_i; \
        } \
        break; \
    }
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MODE_CASE)
        default: TP_THROW(TypeError, "mode: unsupported dtype");
    }
#undef TP_MODE_CASE
    return {vals, idxs};
}


std::tuple<Tensor, Tensor> kthvalue_cpu(const Tensor& self, int64_t k, int64_t dim, bool keepdim) {
    // Sorting.cpp kthvalue_out_cpu: k-th smallest with original index.
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    int64_t d_size = sc.size(dim);
    if (k < 1 || k > d_size) {
        TP_THROW(RuntimeError, "kthvalue(): selected number k out of range for dim ", dim);
    }
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i) out_shape.push_back(i == dim ? 1 : sc.size(i));
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    Tensor vals = Tensor::empty(out_shape, sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(out_shape, DType::Int64, sc.device());
#define TP_KTH_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* vp = vals.data_ptr<ctype>(); \
        int64_t* ip = idxs.data_ptr<int64_t>(); \
        parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            std::vector<std::pair<ctype, int64_t>> buf(d_size); \
            for (int64_t si = b; si < e; ++si) { \
                int64_t o = si / inner, in2 = si % inner; \
                const ctype* s = sp + o * d_size * inner + in2; \
                for (int64_t j = 0; j < d_size; ++j) buf[j] = {s[j * inner], j}; \
                std::stable_sort(buf.begin(), buf.end(), [](auto& a2, auto& b2) { return a2.first < b2.first; }); \
                int64_t oi = keepdim ? si : (o * inner + in2); \
                vp[oi] = buf[k - 1].first; \
                ip[oi] = buf[k - 1].second; \
            } \
        }); \
        break; \
    }
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_KTH_CASE)
        default: TP_THROW(TypeError, "kthvalue: unsupported dtype");
    }
#undef TP_KTH_CASE
    return {vals, idxs};
}

Tensor count_nonzero_cpu(const Tensor& self, const std::vector<int64_t>& dim) {
    DType dt = DType::Int64;
    if (dim.empty()) {
        // global count -> 0-dim tensor
        int64_t cnt = 0;
        Tensor sc = self.contiguous();
        for (int64_t i = 0; i < sc.numel(); ++i) {
            bool nz = false;
            switch (sc.dtype()) {
#define TP_CNZ_PEEK(ctype, name_) case DType::name_: nz = static_cast<bool>(sc.data_ptr<ctype>()[i]); break;
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_CNZ_PEEK)
#undef TP_CNZ_PEEK
                default: break;
            }
            if (nz) ++cnt;
        }
        return Tensor::zeros({}, dt, self.device()).fill_(Scalar(cnt));
    }
    return reduce_dims_impl<double>(
        self, dim, false, dt, 0.0,
        [](double acc, double v) { return v != 0 ? acc + 1 : acc; },
        [](double acc) { return acc; });
}

Tensor dist_cpu(const Tensor& self, const Tensor& other, Scalar p) {
    double pd = p.toDouble();
    Tensor a = self.to(DType::Float64).contiguous();
    Tensor b = other.to(DType::Float64).expand(
        broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()),
                         static_cast<std::vector<int64_t>>(other.shape()))).to(DType::Float64).contiguous();
    int64_t n = a.numel();
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    double result = 0;
    if (pd == std::numeric_limits<double>::infinity()) {
        result = 0;
        for (int64_t i = 0; i < n; ++i) result = std::max(result, std::fabs(ap[i] - bp[i]));
    } else if (pd == -std::numeric_limits<double>::infinity()) {
        result = std::numeric_limits<double>::infinity();
        for (int64_t i = 0; i < n; ++i) result = std::min(result, std::fabs(ap[i] - bp[i]));
    } else if (pd == 0.0) {
        for (int64_t i = 0; i < n; ++i) if (ap[i] != bp[i]) result += 1;
    } else {
        double s = 0;
        for (int64_t i = 0; i < n; ++i) s += std::pow(std::fabs(ap[i] - bp[i]), pd);
        result = std::pow(s, 1.0 / pd);
    }
    DType out_dt = promoteTypes(self.dtype(), other.dtype());
    if (!isFloatingType(out_dt)) out_dt = DType::Float32;
    return Tensor::zeros({}, out_dt, self.device()).fill_(Scalar(static_cast<double>(result)));
}

Tensor renorm_cpu(const Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
    // `dim` (i.e. fixing the dim coordinate, reducing over all other dims)
    // is scaled so its p-norm does not exceed maxnorm.
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    double pd = p.toDouble(), mn = maxnorm.toDouble();
    Tensor sc = self.to(DType::Float64).contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()), DType::Float64, sc.device());
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(sc.shape()), dim, outer, inner);
    const double* sp = sc.data_ptr<double>();
    double* dp = out.data_ptr<double>();
    const int64_t slice_numel = outer * inner;
    parallel_for(0, d_size, GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t j = b; j < e; ++j) {
            double norm = 0;
            if (pd == std::numeric_limits<double>::infinity()) {
                for (int64_t si = 0; si < slice_numel; ++si) {
                    int64_t o = si / inner, in2 = si % inner;
                    norm = std::max(norm, std::fabs(sp[(o * d_size + j) * inner + in2]));
                }
            } else {
                double s = 0;
                for (int64_t si = 0; si < slice_numel; ++si) {
                    int64_t o = si / inner, in2 = si % inner;
                    s += std::pow(std::fabs(sp[(o * d_size + j) * inner + in2]), pd);
                }
                norm = std::pow(s, 1.0 / pd);
            }
            const double factor = norm > mn ? mn / norm : 1.0;
            for (int64_t si = 0; si < slice_numel; ++si) {
                int64_t o = si / inner, in2 = si % inner;
                dp[(o * d_size + j) * inner + in2] = sp[(o * d_size + j) * inner + in2] * factor;
            }
        }
    });
    return out.to(self.dtype());
}

// ===========================================================================
// Shape ops (TensorShape.cpp / TensorTransformations.cpp / ReduceOps.cpp)
// ===========================================================================

Tensor trace_cpu(const Tensor& self) {
    // ReduceOps.cpp:1357 trace_cpu: sum of the diagonal of the last 2 dims.
    if (self.dim() < 2) TP_THROW(RuntimeError, "trace: input must have at least 2 dimensions");
    int64_t rows = self.size(-2), cols = self.size(-1);
    int64_t d = std::min(rows, cols);
    int64_t batch = self.numel() / (rows * cols);
    Tensor sc64 = self.to(DType::Float64).contiguous();
    const Size sc64_shape = sc64.shape();
    std::vector<int64_t> out_shape(sc64_shape.begin(), sc64_shape.end() - 2);
    Tensor out = Tensor::zeros(out_shape, DType::Float64, self.device());
    double* dp = out.data_ptr<double>();
    const double* sp = sc64.data_ptr<double>();
    parallel_for(0, batch, GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t bi = b; bi < e; ++bi) {
            double s = 0;
            for (int64_t i = 0; i < d; ++i) s += sp[bi * rows * cols + i * cols + i];
            dp[bi] = s;
        }
    });
    DType out_dt = self.dtype();
    return out.to(out_dt).reshape(out_shape);
}

Tensor diag_cpu(const Tensor& self, int64_t diagonal) {
    int64_t nd = self.dim();
    Tensor sc = self.contiguous();
    if (nd == 1) {
        int64_t n = sc.size(0);
        int64_t size = n + std::abs(diagonal);
        Tensor outc = Tensor::zeros({size, size}, sc.dtype(), sc.device());
        switch (sc.dtype()) {
#define TP_DIAG_FILL(ctype, name_) \
    case DType::name_: { \
        const ctype* s = sc.data_ptr<ctype>(); \
        ctype* d = outc.data_ptr<ctype>(); \
        for (int64_t i = 0; i < n; ++i) { \
            int64_t r = diagonal >= 0 ? i : i - diagonal; \
            int64_t c = diagonal >= 0 ? i + diagonal : i; \
            d[r * size + c] = s[i]; \
        } \
        break; \
    }
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_DIAG_FILL)
#undef TP_DIAG_FILL
            default: TP_THROW(TypeError, "diag: unsupported dtype");
        }
        return outc;
    }
    if (nd == 2) {
        int64_t rows = sc.size(0), cols = sc.size(1);
        std::vector<int64_t> idx;
        if (diagonal >= 0) {
            for (int64_t i = 0; i + diagonal < cols && i < rows; ++i) idx.push_back(i * cols + i + diagonal);
        } else {
            for (int64_t i = 0; i - diagonal < rows && i < cols; ++i) idx.push_back((i - diagonal) * cols + i);
        }
        Tensor out = Tensor::zeros({static_cast<int64_t>(idx.size())}, sc.dtype(), sc.device());
        switch (sc.dtype()) {
#define TP_DIAG_EX(ctype, name_) \
    case DType::name_: { \
        const ctype* s = sc.data_ptr<ctype>(); \
        ctype* d = out.data_ptr<ctype>(); \
        for (size_t k = 0; k < idx.size(); ++k) d[k] = s[idx[k]]; \
        break; \
    }
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_DIAG_EX)
#undef TP_DIAG_EX
            default: TP_THROW(TypeError, "diag: unsupported dtype");
        }
        return out;
    }
    TP_THROW(RuntimeError, "diag: input must be 1-D or 2-D");
}

Tensor diag_embed_cpu(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_) {
    // TensorShape.cpp:1272 diag_embed (exact structure port).
    int64_t nDims = self.dim() + 1;
    int64_t dim1 = wrap_dim(dim1_, nDims);
    int64_t dim2 = wrap_dim(dim2_, nDims);
    if (dim1 == dim2) TP_THROW(RuntimeError, "diagonal dimensions cannot be identical");
    int64_t new_dim_len = std::abs(offset) + self.size(-1);
    const Size self_shape = self.shape();
    std::vector<int64_t> sizes(self_shape.begin(), self_shape.end());
    sizes.pop_back();
    sizes.insert(sizes.begin() + std::min(dim1, dim2), new_dim_len);
    sizes.insert(sizes.begin() + std::max(dim1, dim2), new_dim_len);
    Tensor result = Tensor::zeros(sizes, self.dtype(), self.device());
    int64_t n = self.numel();
    int64_t last = self.size(-1);
    Tensor sc = self.contiguous();
    int64_t rows = new_dim_len, cols = new_dim_len;
    int64_t mid = std::max(dim1, dim2);
    int64_t lowdim = std::min(dim1, dim2);
    // iterate self elements: coords (..., t); place at (i=t or t-offset, j=t+offset or t)
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
        std::vector<int64_t> cs(sizes.size(), 0);
        for (int64_t li = b; li < e; ++li) {
            // decode self linear index against self sizes (= sizes minus inserted dims)
            int64_t rem = li;
            std::vector<int64_t> self_coords(self.dim(), 0);
            for (int64_t d2 = static_cast<int64_t>(self.dim()) - 1; d2 >= 0; --d2) {
                self_coords[d2] = rem % self.size(d2);
                rem /= self.size(d2);
            }
            int64_t t = self_coords.back();
            int64_t i = offset >= 0 ? t : t - offset;
            int64_t j = offset >= 0 ? t + offset : t;
            // build result coords: insert i at lowdim, j at highdim among remaining
            std::vector<int64_t> rc(sizes.size(), 0);
            int64_t sk = 0;
            for (int64_t d2 = 0; d2 < static_cast<int64_t>(sizes.size()); ++d2) {
                if (d2 == lowdim) { rc[d2] = i; }
                else if (d2 == mid) { rc[d2] = j; }
                else { rc[d2] = (sk < static_cast<int64_t>(self_coords.size()) - 1) ? self_coords[sk] : 0; ++sk; }
            }
            int64_t lin = 0;
            for (int64_t d2 = 0; d2 < static_cast<int64_t>(sizes.size()); ++d2)
                lin = lin * sizes[d2] + rc[d2];
            switch (self.dtype()) {
#define TP_DE_WRITE(ctype, name_) \
    case DType::name_: reinterpret_cast<ctype*>(result.data_ptr())[lin] = reinterpret_cast<const ctype*>(sc.data_ptr())[li]; break;
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_DE_WRITE)
#undef TP_DE_WRITE
                default: break;
            }
            (void)rows; (void)cols;
        }
    });
    return result;
}

Tensor narrow_cpu(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "narrow() cannot be applied to a 0-dim tensor.");
    }
    if (length < 0) {
        TP_THROW(RuntimeError, "narrow(): length must be non-negative.");
    }
    dim = wrap_dim(dim, self.dim());
    const int64_t cur_size = self.size(dim);
    if (start < -cur_size || start > cur_size) {
        TP_THROW(IndexError, "start out of range (expected to be in range of [",
                 -cur_size, ", ", cur_size, "], but got ", start, ")");
    }
    if (start < 0) start += cur_size;
    if (start > cur_size - length) {
        TP_THROW(RuntimeError, "start (", start, ") + length (", length,
                 ") exceeds dimension size (", cur_size, ").");
    }
    return self.slice(dim, start, start + length, 1);
}

std::vector<Tensor> split_with_sizes_cpu(const Tensor& self, std::vector<int64_t> split_sizes, int64_t dim) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "split expects at least a 1-dimensional tensor");
    }
    const int64_t nd = self.dim();
    if (dim < -nd || dim >= nd) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 -nd, ", ", nd - 1, "], but got ", dim, ")");
    }
    if (dim < 0) dim += nd;
    const int64_t dim_size = self.size(dim);
    std::vector<Tensor> outs;
    outs.reserve(split_sizes.size());
    int64_t start = 0;
    for (const int64_t len : split_sizes) {
        if (len < 0) {
            TP_THROW(RuntimeError, "split_with_sizes expects split_sizes have only non-negative "
                     "entries, but got split_sizes=[", [&] {
                         std::string s;
                         for (size_t i = 0; i < split_sizes.size(); ++i) {
                             if (i) s += ", ";
                             s += std::to_string(split_sizes[i]);
                         }
                         return s;
                     }(), "]");
        }
        outs.push_back(self.slice(dim, start, start + len));
        start += len;
    }
    if (start != dim_size) {
        TP_THROW(RuntimeError, "split_with_sizes expects split_sizes to sum exactly to ",
                 dim_size, " (input tensor's size at dimension ", dim, "), but got split_sizes=[",
                 [&] {
                     std::string s;
                     for (size_t i = 0; i < split_sizes.size(); ++i) {
                         if (i) s += ", ";
                         s += std::to_string(split_sizes[i]);
                     }
                     return s;
                 }(), "]");
    }
    return outs;
}

std::vector<Tensor> tensor_split_cpu(const Tensor& self, int64_t sections, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (sections <= 0) TP_THROW(RuntimeError, "tensor_split: number of sections must be larger than 0");
    int64_t size = self.size(dim);
    int64_t chunk_base = size / sections, chunk_rem = size % sections;
    std::vector<Tensor> outs;
    int64_t start = 0;
    for (int64_t i = 0; i < sections; ++i) {
        int64_t len = chunk_base + (i < chunk_rem ? 1 : 0);
        if (len > 0) outs.push_back(narrow_cpu(self, dim, start, len));
        else outs.emplace_back();
        start += len;
    }
    return outs;
}

Tensor flip_cpu(const Tensor& self, const std::vector<int64_t>& dims) {
    // TensorTransformations.cpp:36 flip: dim_list_to_bitset (WrapDimUtilsMulti.h)
    // wraps with wrap_scalar=true and rejects duplicate dims, then reverses
    // each listed dim.
    int64_t nd = self.dim();
    std::vector<bool> seen(nd > 0 ? nd : 1, false);
    std::vector<bool> flip_mask(nd, false);
    for (auto d : dims) {
        int64_t w = wrap_dim_scalar(d, nd);
        if (nd > 0) {
            if (seen[w]) {
                TP_THROW(RuntimeError, "dim ", w,
                         " appears multiple times in the list of dims");
            }
            seen[w] = true;
            flip_mask[w] = true;
        }
    }
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()), sc.dtype(), sc.device());
    int64_t n = sc.numel();
    auto worker = [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            // Decode output linear index and map flipped coordinates back to
            // the source offset.
            int64_t r2 = li, srco = 0, mult = 1;
            for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
                int64_t c = r2 % sc.size(d2);
                r2 /= sc.size(d2);
                int64_t sc3 = flip_mask[d2] ? (sc.size(d2) - 1 - c) : c;
                srco += sc3 * mult;
                mult *= sc.size(d2);
            }
            switch (sc.dtype()) {
#define TP_FLIP_W(ctype, name_) \
    case DType::name_: reinterpret_cast<ctype*>(out.data_ptr())[li] = reinterpret_cast<const ctype*>(sc.data_ptr())[srco]; break;
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_FLIP_W)
#undef TP_FLIP_W
                default: break;
            }
        }
    };
    parallel_for(0, n, GRAIN_SIZE, worker);
    return out;
}

Tensor roll_cpu(const Tensor& self, const std::vector<int64_t>& shifts, const std::vector<int64_t>& dims) {
    // TensorTransformations.cpp:110 roll + TensorTransformations.h roll_common.
    if (dims.size() != 1 || shifts.size() != 1) {
        if (shifts.empty()) TP_THROW(RuntimeError, "`shifts` required");
        if (dims.empty() && shifts.size() == 1) {
            // Flatten-roll: roll the flattened tensor and view back.
            Tensor flat = self.contiguous().reshape({self.numel()});
            Tensor rolled = roll_cpu(flat, {shifts[0]}, {0});
            return rolled.reshape(static_cast<std::vector<int64_t>>(self.shape()));
        }
        if (shifts.size() != dims.size()) {
            TP_THROW(RuntimeError, "shifts and dimensions must align. shifts: ",
                     shifts.size(), ", dims:", dims.size());
        }
        Tensor cur = self;
        for (size_t i = 0; i < dims.size(); ++i) {
            cur = roll_cpu(cur, {shifts[i]}, {dims[i]});
        }
        return cur;
    }
    // Avoid a div zero error below; empty input rolls to
    // itself.
    if (self.numel() == 0) return self.clone();
    const int64_t nd = self.dim();
    if (nd == 0) {
        // wrap_scalar=false rejects any dim.
        TP_THROW(IndexError, "Dimension specified as ", dims[0],
                 " but tensor has no dimensions");
    }
    const int64_t dim = wrap_dim(dims[0], nd);
    const int64_t size = self.size(dim);
    int64_t start = (size - shifts[0]) % size;
    // Behavior of % is different in C++ vs Python for negative numbers.
    if (start < 0) start += size;
    // Equivalent to cat({narrow(dim, start, size-start), narrow(dim, 0, start)}):
    // destination coord c along dim reads source coord (c + start) % size.
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()), sc.dtype(), sc.device());
    int64_t n = sc.numel();
    auto worker = [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            int64_t r2 = li, src = 0, mult = 1;
            for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
                int64_t c = r2 % sc.size(d2);
                r2 /= sc.size(d2);
                int64_t sc3 = d2 == dim ? (c + start) % size : c;
                src += sc3 * mult;
                mult *= sc.size(d2);
            }
            switch (sc.dtype()) {
#define TP_ROLL_W(ctype, name_) case DType::name_: reinterpret_cast<ctype*>(out.data_ptr())[li] = reinterpret_cast<const ctype*>(sc.data_ptr())[src]; break;
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_ROLL_W)
#undef TP_ROLL_W
                default: break;
            }
        }
    };
    parallel_for(0, n, GRAIN_SIZE, worker);
    return out;
}

Tensor rot90_cpu(const Tensor& self, int64_t k, const std::vector<int64_t>& dims) {
    // TensorTransformations.cpp:127 rot90.
    const int64_t total_dims = self.dim();
    const int64_t total_rot_dims = static_cast<int64_t>(dims.size());
    if (total_rot_dims != 2) {
        TP_THROW(RuntimeError, "expected total rotation dims == 2, but got dims = ",
                 total_rot_dims);
    }
    if (total_dims < 2) {
        TP_THROW(RuntimeError, "expected total dims >= 2, but got total dims = ",
                 total_dims);
    }
    // Validate range first so out-of-range dims raise IndexError, then
    // normalize before checking for duplicates (e.g. [1, -1] on a 2D tensor).
    const int64_t dim0 = wrap_dim(dims[0], total_dims);
    const int64_t dim1 = wrap_dim(dims[1], total_dims);
    if (dim0 == dim1) {
        TP_THROW(RuntimeError, "expected rotation dims to be different, but got dim0 = ",
                 dims[0], " and dim1 = ", dims[1]);
    }
    // handle modulo with negative k
    k = (4 + (k % 4)) % 4;
    // transpose_ on the fresh flip result: a view with swapped sizes/strides.
    auto transpose_view = [](const Tensor& x, int64_t a, int64_t b) {
        std::vector<int64_t> sizes(x.dim()), strides(x.dim());
        for (int64_t i = 0; i < x.dim(); ++i) {
            sizes[i] = x.size(i);
            strides[i] = x.stride(i);
        }
        std::swap(sizes[a], sizes[b]);
        std::swap(strides[a], strides[b]);
        return x.as_strided(sizes, strides);
    };
    switch (k) {
        case 1: return transpose_view(flip_cpu(self, {dim1}), dim0, dim1);
        case 2: return flip_cpu(self, {dim0, dim1});
        case 3: return transpose_view(flip_cpu(self, {dim0}), dim0, dim1);
        default: return detail::contiguous_clone(self);
    }
}

Tensor repeat_interleave_cpu(const Tensor& self, int64_t repeats, int64_t dim) {
    int64_t nd = self.dim();
    if (nd == 0) TP_THROW(RuntimeError, "repeat_interleave: dimension required for scalar");
    dim = wrap_dim(dim, nd);
    if (repeats < 0) TP_THROW(RuntimeError, "repeat_interleave: repeats can not be negative");
    std::vector<int64_t> out_shape(static_cast<std::vector<int64_t>>(self.shape()));
    out_shape[dim] *= repeats;
    Tensor out = Tensor::empty(out_shape, self.dtype(), self.device());
    int64_t d_size = self.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    Tensor sc = self.contiguous();
    int64_t out_d = out_shape[dim];
#define TP_RI_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, outer * out_d, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t t = b; t < e; ++t) { \
                int64_t o = t / out_d, j = t % out_d; \
                int64_t src_j = j / repeats; \
                const ctype* s = sp + (o * d_size + src_j) * inner; \
                ctype* d = dp + (o * out_d + j) * inner; \
                std::memcpy(d, s, inner * sizeof(ctype)); \
            } \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_RI_CASE)
        default: TP_THROW(TypeError, "repeat_interleave: unsupported dtype");
    }
#undef TP_RI_CASE
    return out;
}

std::vector<Tensor> meshgrid_cpu(const std::vector<Tensor>& tensors, const std::string& indexing) {
    // along dim j; common promoted dtype; "xy" swaps the first two axes.
    size_t k = tensors.size();
    if (k == 0) return {};
    if (indexing != "ij" && indexing != "xy") {
        TP_THROW(RuntimeError, "meshgrid: indexing must be 'ij' or 'xy', got " + indexing);
    }
    std::vector<Tensor> flat;
    flat.reserve(k);
    for (auto& t : tensors) {
        if (t.dim() > 1) {
            TP_THROW(RuntimeError, "meshgrid: expected 0-D or 1-D tensors");
        }
        flat.push_back(t.contiguous());
    }
    DType common = flat[0].dtype();
    for (size_t j = 1; j < k; ++j) {
        common = promoteTypes(common, flat[j].dtype());
    }
    for (auto& t : flat) {
        if (t.dtype() != common) t = t.to(common);
    }
    // working order: "xy" builds with the first two inputs swapped
    std::vector<Tensor> order = flat;
    if (indexing == "xy" && k >= 2) {
        std::swap(order[0], order[1]);
    }
    std::vector<int64_t> sizes;
    sizes.reserve(k);
    for (auto& t : order) sizes.push_back(t.dim() == 1 ? t.size(0) : 1);
    const size_t esz = elementSize(static_cast<ScalarType>(common));
    std::vector<Tensor> outs;
    outs.reserve(k);
    for (size_t j = 0; j < k; ++j) {
        int64_t nj = sizes[j];
        int64_t outer = 1;
        for (size_t d = 0; d < j; ++d) outer *= sizes[d];
        int64_t inner = 1;
        for (size_t d = j + 1; d < k; ++d) inner *= sizes[d];
        Tensor g = Tensor::empty(sizes, common, tensors[0].device());
        const char* src = reinterpret_cast<const char*>(order[j].data_ptr());
        char* dst = reinterpret_cast<char*>(g.data_ptr());
        for (int64_t o = 0; o < outer; ++o) {
            for (int64_t r = 0; r < nj; ++r) {
                char* drow = dst + ((o * nj + r) * inner) * esz;
                for (int64_t v = 0; v < inner; ++v) {
                    std::memcpy(drow + v * esz, src + r * esz, esz);
                }
            }
        }
        outs.push_back(g);
    }
    return outs;
}

std::vector<Tensor> broadcast_tensors_cpu(const std::vector<Tensor>& tensors) {
    // common broadcast shape.  Returns stride-0 views; gradients flow through
    // the dispatcher expand op (sum-to-size backward).
    std::vector<int64_t> shape{};
    for (auto& t : tensors) {
        const Size t_shape = t.shape();
        std::vector<int64_t> ts(t_shape.begin(), t_shape.end());
        shape = broadcast_shapes(shape, ts);
    }
    std::vector<Tensor> outs;
    outs.reserve(tensors.size());
    for (auto& t : tensors) {
        const Size t_shape = t.shape();
        std::vector<int64_t> ts(t_shape.begin(), t_shape.end());
        if (ts == shape) { outs.push_back(t); continue; }
        outs.push_back(t.expand(shape));
    }
    return outs;
}


Tensor block_diag_cpu(const std::vector<Tensor>& tensors) {
    // 2-D rectangular blocks; result dtype = promoted inputs; empty call
    // yields a (1, 0) tensor.
    if (tensors.empty()) {
        return Tensor::empty(
            std::vector<int64_t>{1, 0},
            std::optional<DType>(DType::Float32),
            std::nullopt,
            false);
    }
    const Device& device = tensors[0].device();
    DType out_dtype = tensors[0].dtype();
    int64_t rows = 0, cols = 0;
    std::vector<Tensor> blocks2d;
    blocks2d.reserve(tensors.size());
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        const Tensor& t = tensors[idx];
        if (!(t.device() == device)) {
            TP_THROW(RuntimeError,
                     "block_diag: input tensors must all be on the same device.");
        }
        out_dtype = promoteTypes(out_dtype, t.dtype());
        const int64_t nd = t.dim();
        if (nd > 2) {
            TP_THROW(RuntimeError,
                     "block_diag: Input tensors must have 2 or fewer dimensions. Input ",
                     static_cast<int64_t>(idx), " has ", nd, " dimensions");
        }
        Tensor b2 = t;
        if (nd == 1) b2 = t.expand({1, t.size(0)});
        else if (nd == 0) b2 = t.expand({1, 1});
        blocks2d.push_back(b2);
        rows += b2.size(0);
        cols += b2.size(1);
    }
    Tensor out = Tensor::zeros({rows, cols}, out_dtype, device);
    int64_t off0 = 0, off1 = 0;
    for (const auto& b : blocks2d) {
        out.slice(0, off0, off0 + b.size(0))
           .slice(1, off1, off1 + b.size(1))
           .copy_(b);
        off0 += b.size(0);
        off1 += b.size(1);
    }
    return out;
}

Tensor pixel_shuffle_cpu(const Tensor& self, int64_t upscale_factor) {
    // PixelShuffle.cpp:23: (N, C*r^2, H, W) -> (N, C, H*r, W*r)
    if (self.dim() != 4) TP_THROW(RuntimeError, "pixel_shuffle expects 4D input");
    int64_t N = self.size(0);
    int64_t C = self.size(1) / (upscale_factor * upscale_factor);
    int64_t H = self.size(2), W = self.size(3);
    if (C * upscale_factor * upscale_factor != self.size(1))
        TP_THROW(RuntimeError, "pixel_shuffle: channel dim must be divisible by r^2");
    Tensor out = Tensor::empty({N, C, H * upscale_factor, W * upscale_factor}, self.dtype(), self.device());
    Tensor sc = self.contiguous();
    int64_t r = upscale_factor;
    int64_t n = self.numel();
    auto wk = [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            int64_t rem = li;
            int64_t w = rem % W; rem /= W;
            int64_t h = rem % H; rem /= H;
            int64_t c = rem % C; rem /= C;
            int64_t bn = rem;
            int64_t ih = h % r, iw = w % r;
            int64_t src = (((bn * (C * r * r) + c * r * r + ih * r + iw) * H + (h / r)) * W + (w / r));
            switch (self.dtype()) {
#define TP_PS_W(ctype, name_) case DType::name_: reinterpret_cast<ctype*>(out.data_ptr())[li] = reinterpret_cast<const ctype*>(sc.data_ptr())[src]; break;
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_PS_W)
#undef TP_PS_W
                default: break;
            }
        }
    };
    parallel_for(0, n, GRAIN_SIZE, wk);
    return out;
}

Tensor pixel_unshuffle_cpu(const Tensor& self, int64_t downscale_factor) {
    if (self.dim() != 4) TP_THROW(RuntimeError, "pixel_unshuffle expects 4D input");
    int64_t r = downscale_factor;
    int64_t N = self.size(0);
    int64_t C = self.size(1);
    int64_t H = self.size(2) / r, W = self.size(3) / r;
    if (H * r != self.size(2) || W * r != self.size(3))
        TP_THROW(RuntimeError, "pixel_unshuffle: spatial dims must be divisible by r");
    Tensor out = Tensor::empty({N, C * r * r, H, W}, self.dtype(), self.device());
    Tensor sc = self.contiguous();
    int64_t n = out.numel();
    auto wk = [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            int64_t rem = li;
            int64_t w = rem % W; rem /= W;
            int64_t h = rem % H; rem /= H;
            int64_t cc = rem % (C * r * r); rem /= (C * r * r);
            int64_t bn = rem;
            int64_t c = cc / (r * r);
            int64_t ij = cc % (r * r);
            int64_t ih = ij / r, iw = ij % r;
            int64_t src = ((((bn * C + c) * (H * r) + h * r + ih) * (W * r)) + w * r + iw);
            switch (self.dtype()) {
#define TP_PU_W(ctype, name_) case DType::name_: reinterpret_cast<ctype*>(out.data_ptr())[li] = reinterpret_cast<const ctype*>(sc.data_ptr())[src]; break;
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_PU_W)
#undef TP_PU_W
                default: break;
            }
        }
    };
    parallel_for(0, n, GRAIN_SIZE, wk);
    return out;
}

Tensor channel_shuffle_cpu(const Tensor& self, int64_t groups) {
    // ChannelShuffle: view(N, g, C/g, ...) -> transpose(1,2). Channel dim is
    // dim 1 for >=2D input, dim 0 for 1-D input.
    if (self.dim() < 1) TP_THROW(RuntimeError, "channel_shuffle expects >= 1D input");
    int64_t cdim = self.dim() >= 2 ? 1 : 0;
    int64_t C = self.size(cdim);
    int64_t outer = 1;   // product of dims before cdim
    for (int64_t i = 0; i < cdim; ++i) outer *= self.size(i);
    int64_t inner = 1;   // product of dims after cdim
    for (int64_t i = cdim + 1; i < self.dim(); ++i) inner *= self.size(i);
    if (C % groups) TP_THROW(RuntimeError, "channel_shuffle: channel dim not divisible by groups");
    int64_t cg = C / groups;
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    auto wk = [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            // li layout: (outer * C + c) * inner + tail
            int64_t tail = li % inner;
            int64_t rest = li / inner;
            int64_t c = rest % C;
            int64_t o = rest / C;
            int64_t j = c / cg, gi = c % cg;
            int64_t src_c = gi * cg + j;
            int64_t src = ((o * C) + src_c) * inner + tail;
            switch (self.dtype()) {
#define TP_CS_W(ctype, name_) case DType::name_: reinterpret_cast<ctype*>(out.data_ptr())[li] = reinterpret_cast<const ctype*>(sc.data_ptr())[src]; break;
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_CS_W)
#undef TP_CS_W
                default: break;
            }
        }
    };
    parallel_for(0, n, GRAIN_SIZE, wk);
    return out;
}

Tensor unfold_cpu(const Tensor& self, int64_t dimension, int64_t size, int64_t step) {
    // TensorShape.cpp unfold: an as_strided view.  wrap_scalar=true allows
    // dimension == 0 on 0-d tensors (max_size becomes 1).
    const int64_t nd = self.dim();
    dimension = wrap_dim_scalar(dimension, nd);

    std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<int64_t> strides = self.strides();
    const int64_t max_size = nd == 0 ? 1 : sizes[dimension];
    if (size < 0) TP_THROW(RuntimeError, "size is ", size, " but must be >= 0");
    if (size > max_size) {
        TP_THROW(RuntimeError, "maximum size for tensor at dimension ", dimension,
                 " is ", max_size, " but size is ", size);
    }
    if (step <= 0) TP_THROW(RuntimeError, "step is ", step, " but must be > 0");
    sizes.push_back(size);
    strides.push_back(nd == 0 ? 1 : strides[dimension]);
    // The if handles the self.dim() == 0 case
    if (dimension < nd) {
        sizes[dimension] = (sizes[dimension] - size) / step + 1;
        strides[dimension] *= step;
    }
    return self.as_strided(sizes, strides);
}

Tensor unfold_backward_cpu(const Tensor& grad, const std::vector<int64_t>& input_sizes,
                           int64_t dim, int64_t size, int64_t step) {
    // window's gradient back onto `dim`, accumulating where windows overlap
    // (step < size).  We gather over grad_input elements (race-free), which
    // degenerates to a plain copy when step >= size.
    if (step <= 0) TP_THROW(RuntimeError, "step is ", step, " but must be > 0");
    Tensor grad_input = Tensor::zeros(input_sizes, grad.dtype(), grad.device());
    const int64_t nd = static_cast<int64_t>(input_sizes.size());
    if (nd == 0) {
        // 0-d input: unfold appended a single axis; the lone element is hit once.
        if (size > 0) grad_input.copy_(grad.select(0, 0));
        return grad_input;
    }
    dim = wrap_dim(dim, nd);
    const int64_t input_dim_size = input_sizes[dim];
    const int64_t count = grad.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(input_sizes, dim, outer, inner);
    Tensor gc = grad.contiguous();
    const int64_t total = outer * input_dim_size * inner;
    if (total == 0) return grad_input;
#define TP_UFB(ctype, name_) \
    case DType::name_: { \
        const ctype* gp = gc.data_ptr<ctype>(); \
        ctype* gip = grad_input.data_ptr<ctype>(); \
        parallel_for(0, total, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t t = b; t < e; ++t) { \
                int64_t inner_idx = t % inner; \
                int64_t rest = t / inner; \
                int64_t idx_dim = rest % input_dim_size; \
                int64_t outer_idx = rest / input_dim_size; \
                int64_t left = (idx_dim > size) ? (idx_dim - size) / step : 0; \
                if (!(left * step <= idx_dim && idx_dim < left * step + size)) ++left; \
                int64_t right = idx_dim / step; \
                if (right >= count) right = count - 1; \
                ctype acc{}; \
                for (int64_t fold = left; fold <= right; ++fold) { \
                    int64_t j = idx_dim - fold * step; \
                    acc += gp[((outer_idx * count + fold) * inner + inner_idx) * size + j]; \
                } \
                gip[t] = acc; \
            } \
        }); \
        break; \
    }
    switch (grad.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_UFB)
        default: TP_THROW(TypeError, "unfold_backward: unsupported dtype");
    }
#undef TP_UFB
    return grad_input;
}

// ===========================================================================
//
//   TensorShape.cpp select_scatter_out / slice_scatter_out
//   (composite: clone the base, mutate a view, return);
//   TensorAdvancedIndexing.cpp take_along_dim_out (broadcast indices against
//   self on all non-dim axes, then gather);
//   Sorting.cpp msort_cpu -> sort(dim=0).values;
//   ReduceOps.cpp nanmean -> nansum / valid-count with all-NaN -> NaN guard
//   (ReduceOpsKernel nanmean_kernel);
//   TensorCompare.cpp isclose (|a-b| <= atol + rtol*|b| with inf/nan rules);
//   isreal: complex tensors test imag==0, everything else is true.
// ===========================================================================

Tensor select_scatter_cpu(const Tensor& self, const Tensor& src, int64_t dim, int64_t index) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index < 0) index += self.size(dim);
    if (index < 0 || index >= self.size(dim)) {
        TP_THROW(IndexError, "select_scatter: index ", index, " is out of bounds for dimension ",
                 dim, " with size ", self.size(dim));
    }
    Tensor result = self.clone();
    result.select(dim, index).copy_(src);
    return result;
}

Tensor slice_scatter_cpu(const Tensor& self, const Tensor& src, int64_t dim,
                         std::optional<int64_t> start, std::optional<int64_t> end, int64_t step) {
    if (step <= 0) TP_THROW(RuntimeError, "slice_scatter: step must be positive");
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    int64_t length = self.size(dim);
    int64_t s = start.has_value() ? *start : 0;
    int64_t e = end.has_value() ? *end : length;
    if (s < 0) s += length;
    if (e < 0) e += length;
    s = std::max<int64_t>(0, std::min<int64_t>(s, length));
    e = std::max<int64_t>(0, std::min<int64_t>(e, length));
    if (e < s) e = s;
    Tensor result = self.clone();
    result.slice(dim, s, e, step).copy_(src);
    return result;
}

Tensor diagonal_scatter_cpu(const Tensor& self, const Tensor& src, int64_t offset,
                            int64_t dim1, int64_t dim2) {
    Tensor result = self.clone();
    Tensor diag = result.diagonal(offset, dim1, dim2);
    std::vector<int64_t> diag_shape(static_cast<std::vector<int64_t>>(diag.shape()));
    result.diagonal(offset, dim1, dim2).copy_(src.reshape(diag_shape));
    return result;
}

Tensor take_along_dim_cpu(const Tensor& self, const Tensor& indices, std::optional<int64_t> dim) {
    // TensorAdvancedIndexing.cpp take_along_dim_out
    if (!dim.has_value()) {
        Tensor flat = self.reshape({-1});
        Tensor idx = indices.to(DType::Int64).reshape({-1});
        return flat.gather(0, idx);
    }
    int64_t nd = self.dim();
    int64_t d = wrap_dim(*dim, nd);
    if (indices.dim() != nd) {
        TP_THROW(RuntimeError, "take_along_dim: indices must have the same number of dimensions as input");
    }
    // Broadcast both operands over every axis except d; along d only the
    // index extent matters (that is the gather length).
    std::vector<int64_t> target(nd);
    for (int64_t i = 0; i < nd; ++i) {
        if (i == d) { target[i] = indices.size(i); continue; }
        int64_t a = self.size(i), b = indices.size(i);
        if (a != b && a != 1 && b != 1) {
            TP_THROW(RuntimeError, "take_along_dim: input and indices must match on non-selected dimensions");
        }
        target[i] = std::max(a, b);
    }
    std::vector<int64_t> idx_target = target;
    std::vector<int64_t> self_target = target;
    self_target[d] = self.size(d);
    Tensor idx_b = indices.expand(idx_target).contiguous().to(DType::Int64);
    Tensor self_b = self.expand(self_target).contiguous();
    return self_b.gather(d, idx_b);
}

Tensor msort_cpu(const Tensor& self) {
    // Sorting.cpp msort_cpu: values of sort along dim 0.
    Tensor values = std::get<0>(self.sort(0, false));
    return values;
}

Tensor nanmean_cpu(const Tensor& self, std::optional<int64_t> dim_opt, bool keepdim,
                   std::optional<DType> dtype) {
    // ReduceOps.cpp nanmean: sum of non-NaN / count of non-NaN; an empty
    // count yields NaN (unlike mean's error).
    DType acc_dt = dtype.value_or(DType::Undefined);
    Tensor x = self;
    if (!isFloatingType(x.dtype()) && !isComplexType(x.dtype())) {
        x = x.to(acc_dt != DType::Undefined ? acc_dt : DType::Float32);
    } else if (isReducedFloatingType(x.dtype()) && acc_dt == DType::Undefined) {
        x = x.to(DType::Float32);
    }
    std::vector<int64_t> dims;
    if (dim_opt.has_value()) dims.push_back(*dim_opt);
    else if (!dims.empty()) { /* unreachable */ }
    else {
        // global reduction over every dimension
        for (int64_t i = 0; i < x.dim(); ++i) dims.push_back(i);
    }
    Tensor total = nansum_cpu(x, dims, keepdim);
    Tensor valid = isnan_cpu(x).logical_not();
    Tensor count = reduce_dims_impl<double>(
        valid, dims, keepdim, DType::Float32, 0.0,
        [](double acc, double v) { return acc + v; },
        [](double acc) { return acc; });
    Tensor zero = count.eq(Scalar(0.0));
    Tensor quot = total.to(DType::Float32).div(count);
    return quot.masked_fill(zero, Scalar(std::numeric_limits<double>::quiet_NaN()))
               .to(acc_dt != DType::Undefined ? acc_dt
                                              : (isComplexType(self.dtype()) ? self.dtype()
                                                                             : total.dtype()));
}

Tensor isclose_cpu(const Tensor& self, const Tensor& other, double rtol, double atol, bool equal_nan) {
    // |a-b| <= atol + rtol*|b|; values also count as close when they are
    // equal, infinities match each other, and NaNs match under equal_nan.
    // Complex inputs keep complex arithmetic: both the equality check and
    // the error use the full two-component value.
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, toString(self.dtype()), " did not match ",
                 toString(other.dtype()));
    }
    TP_THROW_IF(rtol < 0, RuntimeError,
                "rtol must be greater than or equal to zero, but got ", rtol);
    TP_THROW_IF(atol < 0, RuntimeError,
                "atol must be greater than or equal to zero, but got ", atol);

    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(other.shape()));
    Tensor out = Tensor::empty(out_shape, DType::Bool, self.device());
    int64_t n = out.numel();
    bool* dp = out.data_ptr<bool>();

    if (isComplexType(self.dtype())) {
        Tensor a = self.to(DType::ComplexDouble).expand(out_shape).contiguous();
        Tensor b = other.to(DType::ComplexDouble).expand(out_shape).contiguous();
        const std::complex<double>* ap =
            static_cast<const std::complex<double>*>(a.data_ptr());
        const std::complex<double>* bp =
            static_cast<const std::complex<double>*>(b.data_ptr());
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                std::complex<double> x = ap[i], y = bp[i];
                bool close = x == y;
                if (equal_nan) {
                    auto is_nan = [](const std::complex<double>& v) {
                        return std::isnan(v.real()) || std::isnan(v.imag());
                    };
                    close = close || (is_nan(x) && is_nan(y));
                }
                if (!close) {
                    double actual = std::abs(x - y);
                    double allowed = atol + rtol * std::abs(y);
                    close = std::isfinite(actual) && actual <= allowed;
                }
                dp[i] = close;
            }
        });
        return out;
    }

    Tensor a = self.to(DType::Float64).expand(out_shape).contiguous();
    Tensor b = other.to(DType::Float64).expand(out_shape).contiguous();
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            double x = ap[i], y = bp[i];
            bool close = x == y;
            if (equal_nan && x != x && y != y) {
                close = true;
            }
            if (!close) {
                double actual = std::fabs(x - y);
                double allowed = atol + rtol * std::fabs(y);
                close = std::isfinite(actual) && actual <= allowed;
            }
            dp[i] = close;
        }
    });
    return out;
}

Tensor isreal_cpu(const Tensor& self) {
    // ComplexHelper: real dtypes are trivially real; complex tests imag==0.
    if (!isComplexType(self.dtype())) {
        return Tensor::ones(static_cast<std::vector<int64_t>>(self.shape()),
                            DType::Bool, self.device());
    }
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                               DType::Bool, self.device());
    int64_t n = out.numel();
    bool* dp = out.data_ptr<bool>();
    if (self.dtype() == DType::ComplexFloat) {
        const auto* sp = static_cast<const std::complex<float>*>(sc.data_ptr());
        for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].imag() == 0.0f;
    } else {
        const auto* sp = static_cast<const std::complex<double>*>(sc.data_ptr());
        for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].imag() == 0.0;
    }
    return out;
}

// --- Bitwise family --------------------------------------------------------
// BinaryOps.cpp bitwise_*: integer/bool only; bool computes the logical op.

#define TENSORPLAY_FORALL_INT_TYPES(_) \
    _(uint8_t, UInt8)                  \
    _(int8_t, Int8)                    \
    _(int16_t, Int16)                  \
    _(int32_t, Int32)                  \
    _(int64_t, Int64)                  \
    _(uint16_t, UInt16)                \
    _(uint32_t, UInt32)                \
    _(uint64_t, UInt64)

inline void bitwise_check_cpu(const Tensor& t, const char* name) {
    DType d = t.dtype();
    if (d == DType::Bool || isIntegralType(d)) return;
    TP_THROW(TypeError, name, ": only integral and boolean types are supported");
}

template <typename Pred>
Tensor bitwise_binary_cpu(const Tensor& a_in, const Tensor& b_in, Pred pred, const char* name) {
    bitwise_check_cpu(a_in, name);
    bitwise_check_cpu(b_in, name);
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (a_in.dtype() == DType::Bool && b_in.dtype() == DType::Bool) dt = DType::Bool;
    if (dt != DType::Bool && !isIntegralType(dt)) {
        TP_THROW(TypeError, name, ": only integral and boolean types are supported");
    }
    Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt)).expand(out_shape).contiguous();
    Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt)).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    int64_t n = out.numel();
    if (dt == DType::Bool) {
        const bool* ap = ac.data_ptr<bool>();
        const bool* bp = bc.data_ptr<bool>();
        bool* dp = out.data_ptr<bool>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i)
                dp[i] = pred(static_cast<uint8_t>(ap[i]), static_cast<uint8_t>(bp[i]));
        });
        return out;
    }
#define TP_BIT_BIN_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* ap = ac.data_ptr<ctype>(); \
        const ctype* bp = bc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) dp[i] = pred(ap[i], bp[i]); \
        }); \
        break; \
    }
    switch (dt) {
        TENSORPLAY_FORALL_INT_TYPES(TP_BIT_BIN_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BIT_BIN_CASE
    return out;
}

template <typename Pred>
Tensor bitwise_scalar_cpu(const Tensor& self_in, Scalar other, Pred pred, const char* name) {
    bitwise_check_cpu(self_in, name);
    Tensor sc = self_in.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self_in.shape()),
                               self_in.dtype(), self_in.device());
    int64_t n = out.numel();
    if (self_in.dtype() == DType::Bool) {
        const bool* sp = sc.data_ptr<bool>();
        uint8_t o = other.to<bool>() ? 1 : 0;
        bool* dp = out.data_ptr<bool>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i)
                dp[i] = pred(static_cast<uint8_t>(sp[i]), o);
        });
        return out;
    }
#define TP_BIT_SCALAR_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype ov = static_cast<ctype>(other.to<int64_t>()); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) dp[i] = pred(sp[i], ov); \
        }); \
        break; \
    }
    switch (self_in.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_BIT_SCALAR_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BIT_SCALAR_CASE
    return out;
}

template <typename Pred>
Tensor bitwise_shift_scalar_cpu(const Tensor& self_in, Scalar other, Pred pred, const char* name) {
    bitwise_check_cpu(self_in, name);
    int64_t bits = self_in.itemsize() * 8;
    int64_t shift = other.to<int64_t>();
    if (shift < 0 || shift >= bits) {
        TP_THROW(RuntimeError, name, ": shift amount ", shift,
                 " must be in [0, ", bits, ")");
    }
    Tensor sc = self_in.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self_in.shape()),
                               self_in.dtype(), self_in.device());
    int64_t n = out.numel();
#define TP_SHIFT_SCALAR_CASE(ctype, name_) \
    case DType::name_: { \
        using U = typename std::make_unsigned<ctype>::type; \
        const ctype* sp = sc.data_ptr<ctype>(); \
        U sh = static_cast<U>(shift % bits); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) \
                dp[i] = pred(static_cast<U>(sp[i]), sh); \
        }); \
        break; \
    }
    switch (self_in.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_SHIFT_SCALAR_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_SHIFT_SCALAR_CASE
    return out;
}

Tensor bitwise_not_cpu(const Tensor& self) {
    bitwise_check_cpu(self, "bitwise_not");
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                               self.dtype(), self.device());
    int64_t n = out.numel();
    if (self.dtype() == DType::Bool) {
        const bool* sp = sc.data_ptr<bool>();
        bool* dp = out.data_ptr<bool>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = !sp[i];
        });
        return out;
    }
#define TP_BNOT_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) dp[i] = static_cast<ctype>(~sp[i]); \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES(TP_BNOT_CASE)
        default: TP_THROW(TypeError, "bitwise_not: unsupported dtype");
    }
#undef TP_BNOT_CASE
    return out;
}

template <bool kLeft>
Tensor bitwise_shift_tensor_cpu(const Tensor& a_in, const Tensor& b_in, const char* name) {
    // bit width; shifting through the unsigned domain keeps << defined.
    bitwise_check_cpu(a_in, name);
    bitwise_check_cpu(b_in, name);
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (a_in.dtype() == DType::Bool && b_in.dtype() == DType::Bool) dt = DType::Bool;
    if (dt != DType::Bool && !isIntegralType(dt)) {
        TP_THROW(TypeError, name, ": only integral and boolean types are supported");
    }
    Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt)).expand(out_shape).contiguous();
    Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt)).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    int64_t n = out.numel();
#define TP_SHIFT_BIN_CASE(ctype, name_) \
    case DType::name_: { \
        using U = typename std::make_unsigned<ctype>::type; \
        constexpr int64_t kBits = static_cast<int64_t>(sizeof(ctype) * 8); \
        const ctype* ap = ac.data_ptr<ctype>(); \
        const ctype* bp = bc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) { \
                U x = static_cast<U>(ap[i]); \
                U sh = static_cast<U>(static_cast<uint64_t>(bp[i]) % \
                                      static_cast<uint64_t>(kBits)); \
                U r = kLeft ? static_cast<U>(x << sh) : static_cast<U>(x >> sh); \
                dp[i] = static_cast<ctype>(r); \
            } \
        }); \
        break; \
    }
    switch (dt) {
        TENSORPLAY_FORALL_INT_TYPES(TP_SHIFT_BIN_CASE)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_SHIFT_BIN_CASE
    return out;
}

// Named entry points registered with the dispatcher.
Tensor bitwise_and_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_binary_cpu(a, b,
        [](auto x, auto y) { return static_cast<decltype(x)>(x & y); }, "bitwise_and");
}
Tensor bitwise_or_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_binary_cpu(a, b,
        [](auto x, auto y) { return static_cast<decltype(x)>(x | y); }, "bitwise_or");
}
Tensor bitwise_xor_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_binary_cpu(a, b,
        [](auto x, auto y) { return static_cast<decltype(x)>(x ^ y); }, "bitwise_xor");
}
Tensor bitwise_and_scalar_cpu(const Tensor& a, Scalar b) {
    return bitwise_scalar_cpu(a, b,
        [](auto x, auto y) { return static_cast<decltype(x)>(x & y); }, "bitwise_and");
}
Tensor bitwise_or_scalar_cpu(const Tensor& a, Scalar b) {
    return bitwise_scalar_cpu(a, b,
        [](auto x, auto y) { return static_cast<decltype(x)>(x | y); }, "bitwise_or");
}
Tensor bitwise_xor_scalar_cpu(const Tensor& a, Scalar b) {
    return bitwise_scalar_cpu(a, b,
        [](auto x, auto y) { return static_cast<decltype(x)>(x ^ y); }, "bitwise_xor");
}
Tensor bitwise_lshift_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_shift_tensor_cpu<true>(a, b, "bitwise_left_shift");
}
Tensor bitwise_rshift_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_shift_tensor_cpu<false>(a, b, "bitwise_right_shift");
}
Tensor bitwise_lshift_scalar_cpu(const Tensor& a, Scalar b) {
    return bitwise_shift_scalar_cpu(a, b,
        [](auto x, auto sh) { return static_cast<decltype(x)>(x << sh); },
        "bitwise_left_shift");
}
Tensor bitwise_rshift_scalar_cpu(const Tensor& a, Scalar b) {
    return bitwise_shift_scalar_cpu(a, b,
        [](auto x, auto sh) { return static_cast<decltype(x)>(x >> sh); },
        "bitwise_right_shift");
}

TENSORPLAY_LIBRARY_IMPL(CPU, TierOpsKernels) {
    // Arithmetic
    m.impl("rsub.Scalar", rsub_scalar_cpu);
    m.impl("rsub.Tensor", rsub_tensor_cpu);
    m.impl("true_divide.Tensor", true_divide_tensor_cpu);
    m.impl("true_divide.Scalar", true_divide_scalar_cpu);
    m.impl("divide.Tensor", divide_tensor_cpu);
    m.impl("divide.Scalar", divide_scalar_cpu);
    m.impl("remainder.Tensor", remainder_tensor_cpu);
    m.impl("remainder.Scalar", remainder_scalar_cpu);
    m.impl("fmod.Tensor", fmod_tensor_cpu);
    m.impl("fmod.Scalar", fmod_scalar_cpu);
    m.impl("subtract.Tensor", subtract_tensor_cpu);
    m.impl("subtract.Scalar", subtract_scalar_cpu);
    m.impl("multiply.Tensor", multiply_tensor_cpu);
    m.impl("multiply.Scalar", multiply_scalar_cpu);
    m.impl("remainder.Scalar_Tensor", remainder_scalar_tensor_cpu);
    m.impl("div.Tensor_mode", div_mode_tensor_cpu);
    m.impl("div.Scalar_mode", div_mode_scalar_cpu);
    m.impl("divide.Tensor_mode", div_mode_tensor_cpu);
    m.impl("divide.Scalar_mode", div_mode_scalar_cpu);
    m.impl("floor_divide", floor_divide_cpu);
    m.impl("floor_divide.Scalar", floor_divide_scalar_cpu);
    m.impl("negative", negative_cpu);
    m.impl("positive", positive_cpu);
    // Comparisons / logic
    m.impl("greater", greater_cpu);
    m.impl("greater_equal", greater_equal_cpu);
    m.impl("less", less_cpu);
    m.impl("less_equal", less_equal_cpu);
    m.impl("not_equal", not_equal_cpu);
    m.impl("signbit", signbit_cpu);
    m.impl("logical_not", logical_not_cpu);
    m.impl("logical_and", logical_and_cpu);
    m.impl("logical_or", logical_or_cpu);
    m.impl("logical_xor", logical_xor_cpu);
    m.impl("isfinite", isfinite_cpu);
    m.impl("isinf", isinf_cpu);
    m.impl("isnan", isnan_cpu);
    m.impl("isneginf", isneginf_cpu);
    m.impl("isposinf", isposinf_cpu);
    // Math
    m.impl("reciprocal", reciprocal_cpu);
    m.impl("sgn", sgn_cpu);
    m.impl("exp2", exp2_cpu);
    m.impl("sinc", sinc_cpu);
    m.impl("deg2rad", deg2rad_cpu);
    m.impl("rad2deg", rad2deg_cpu);
    m.impl("fix", fix_cpu);
    m.impl("erfinv", erfinv_cpu);
    m.impl("logit", logit_cpu);
    m.impl("digamma", digamma_cpu);
    m.impl("i0", i0_cpu);
    m.impl("nan_to_num", nan_to_num_cpu);
    m.impl("xlogy", xlogy_cpu);
    m.impl("logaddexp", logaddexp_cpu);
    m.impl("logaddexp2", logaddexp2_cpu);
    m.impl("copysign.Tensor", copysign_cpu);
    m.impl("copysign.Scalar", copysign_scalar_cpu);
    m.impl("hypot", hypot_cpu);
    m.impl("atan2", atan2_cpu);
    m.impl("nextafter", nextafter_cpu);
    m.impl("gcd", gcd_cpu);
    m.impl("lcm", lcm_cpu);
    m.impl("heaviside", heaviside_cpu);
    // Clamp family
    m.impl("clamp_", clamp__cpu);
    m.impl("clamp_min.Scalar", clamp_min_scalar_cpu);
    m.impl("clamp_max.Scalar", clamp_max_scalar_cpu);
    m.impl("clamp_min.Tensor", clamp_min_tensor_cpu);
    m.impl("clamp_max.Tensor", clamp_max_tensor_cpu);
    m.impl("clip", clip_cpu);
    // Activations
    m.impl("selu", selu_cpu);
    m.impl("celu", celu_cpu);
    m.impl("hardshrink", hardshrink_cpu);
    m.impl("hardshrink_backward", hardshrink_backward_cpu);
    m.impl("softshrink", softshrink_cpu);
    m.impl("softshrink_backward", softshrink_backward_cpu);
    m.impl("sigmoid_backward", sigmoid_backward_cpu);
    m.impl("tanh_backward", tanh_backward_cpu);
    m.impl("logit_backward", logit_backward_cpu);
    m.impl("threshold", threshold_cpu);
    m.impl("prelu", prelu_cpu);
    // Reductions
    m.impl("amax", amax_cpu);
    m.impl("amin", amin_cpu);
    m.impl("aminmax", aminmax_cpu);
    m.impl("logsumexp", logsumexp_cpu);
    m.impl("nansum", nansum_cpu);
    m.impl("nanmedian", nanmedian_cpu);
    m.impl("cummax", cummax_cpu);
    m.impl("cummin", cummin_cpu);
    m.impl("std_mean", std_mean_cpu);
    m.impl("var_mean", var_mean_cpu);
    m.impl("mode", mode_cpu);
    m.impl("kthvalue", kthvalue_cpu);
    m.impl("count_nonzero", count_nonzero_cpu);
    m.impl("dist", dist_cpu);
    m.impl("renorm", renorm_cpu);
    // Shape ops
    m.impl("trace", trace_cpu);
    m.impl("diag", diag_cpu);
    m.impl("diag_embed", diag_embed_cpu);
    m.impl("narrow", narrow_cpu);
    m.impl("split_with_sizes", split_with_sizes_cpu);
    // view semantics + indices/tensor overloads); duplicate removed.
    m.impl("roll", roll_cpu);
    m.impl("flip", flip_cpu);
    m.impl("rot90", rot90_cpu);
    m.impl("repeat_interleave.self_int", repeat_interleave_cpu);
    m.impl("meshgrid", meshgrid_cpu);
    m.impl("broadcast_tensors", broadcast_tensors_cpu);
    m.impl("block_diag", block_diag_cpu);
    m.impl("pixel_shuffle", pixel_shuffle_cpu);
    m.impl("pixel_unshuffle", pixel_unshuffle_cpu);
    m.impl("channel_shuffle", channel_shuffle_cpu);
    m.impl("unfold", unfold_cpu);
    m.impl("unfold_backward", unfold_backward_cpu);
    // Index/scatter complements
    m.impl("select_scatter", select_scatter_cpu);
    m.impl("slice_scatter", slice_scatter_cpu);
    m.impl("diagonal_scatter", diagonal_scatter_cpu);
    m.impl("take_along_dim", take_along_dim_cpu);
    m.impl("msort", msort_cpu);
    m.impl("nanmean", nanmean_cpu);
    m.impl("isclose", isclose_cpu);
    m.impl("isreal", isreal_cpu);
    // Bitwise family
    m.impl("bitwise_not", bitwise_not_cpu);
    m.impl("bitwise_and.Tensor", bitwise_and_tensor_cpu);
    m.impl("bitwise_or.Tensor", bitwise_or_tensor_cpu);
    m.impl("bitwise_xor.Tensor", bitwise_xor_tensor_cpu);
    m.impl("bitwise_and.Scalar", bitwise_and_scalar_cpu);
    m.impl("bitwise_or.Scalar", bitwise_or_scalar_cpu);
    m.impl("bitwise_xor.Scalar", bitwise_xor_scalar_cpu);
    m.impl("bitwise_left_shift.Tensor", bitwise_lshift_tensor_cpu);
    m.impl("bitwise_right_shift.Tensor", bitwise_rshift_tensor_cpu);
    m.impl("bitwise_left_shift.Tensor_Scalar", bitwise_lshift_scalar_cpu);
    m.impl("bitwise_right_shift.Tensor_Scalar", bitwise_rshift_scalar_cpu);
}

} // namespace cpu
} // namespace tensorplay
