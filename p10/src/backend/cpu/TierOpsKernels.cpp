// Tier 2-4 operators (arithmetic aliases, comparisons/logic, clamp family,
// activations, math functions, reductions, shape ops) - CPU kernels.
//
// Algorithms ported from the vendored PyTorch tree at third_party/pytorch
// (2.15.0a0). Verified ATen anchors cited per section:
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

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <cstring>
#include <utility>
#include <type_traits>

#if defined(__GLIBC__)
extern "C" double erfinv(double) __THROW __attribute_const__;
extern "C" float erfinvf(float) __THROW __attribute_const__;
#endif

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace {

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

inline DType scalar_promote(DType t, const Scalar& s) {
    // Weak scalar participation (mirrors ComparisonKernels.cpp result rule).
    if (!isFloatingType(s.dtype())) return t;
    if (isFloatingType(t)) return t;
    return DType::Float32;
}

// ---------------------------------------------------------------------------
// Elementwise helpers
// ---------------------------------------------------------------------------

// Broadcast both inputs to a common promoted dtype; op returns that dtype.
template <typename Op>
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
    ti_apply_binary(out, ac, bc, op);
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
    Tensor ac = a_in.to(dt).expand(out_shape).contiguous();
    Tensor bc = b_in.to(dt).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    int64_t n = out.numel();
    if (dt == DType::Float64) {
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
    return out;
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

// Math unary with torch result-type semantics: f maps double -> double;
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
            // Decode output coordinates against out_shape, then compute the
            // input base offset over the non-reduced dims.
            int64_t base = 0;
            {
                int64_t r2 = oi;
                std::vector<int64_t> oc(std::max<int64_t>(out_shape.size(), 1), 0);
                for (int64_t i = static_cast<int64_t>(out_shape.size()) - 1; i >= 0; --i) {
                    oc[i] = r2 % out_shape[i];
                    r2 /= out_shape[i];
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
    std::vector<bool> reduced(nd, false);
    for (auto& d : dims_in) { d = wrap_dim(d, nd); reduced[d] = true; }
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
    // BinaryOps.cpp:1203: other * alpha - self (weak scalar promotion)
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
            for (int64_t i = b; i < e; ++i) dp[i] = static_cast<ctype>(ov * av - sp[i]); \
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
    // BinaryOps.cpp:1169
    return binary_same_kernel(self, other,
        [alpha](auto s, auto o) {
            using T = decltype(o);
            return static_cast<T>(o * alpha.to<double>() - s);
        }, "rsub");
}

static Tensor true_divide_core(const Tensor& a, const Tensor& b) {
    // BinaryOps.cpp:954: integral inputs promote to the default float type.
    return binary_float_kernel(a, b, [](double x, double y) { return x / y; }, "true_divide");
}
Tensor true_divide_tensor_cpu(const Tensor& self, const Tensor& other) { return true_divide_core(self, other); }
Tensor true_divide_scalar_cpu(const Tensor& self, Scalar other) {
    return true_divide_core(self, Tensor::full({}, other, DType::Float32, self.device()));
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
    return remainder_tensor_cpu(self, Tensor::full({}, other, DType::Undefined, self.device())
                                            .to(self.dtype()));
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
    return fmod_tensor_cpu(self, Tensor::full({}, other, DType::Undefined, self.device()).to(self.dtype()));
}

Tensor subtract_tensor_cpu(const Tensor& self, const Tensor& other) {
    return binary_same_kernel(self, other, [](auto x, auto y) { return x - y; }, "subtract");
}
Tensor subtract_scalar_cpu(const Tensor& self, Scalar other) {
    DType dt = scalar_promote(self.dtype(), other);
    return subtract_tensor_cpu(self.to(dt),
                               Tensor::full({}, other, dt, self.device()));
}
Tensor multiply_tensor_cpu(const Tensor& self, const Tensor& other) {
    return binary_same_kernel(self, other, [](auto x, auto y) { return x * y; }, "multiply");
}
Tensor multiply_scalar_cpu(const Tensor& self, Scalar other) {
    return dtype_unary_kernel(self, [other](auto x) {
        return static_cast<decltype(x)>(x * other.to<double>());
    }, "multiply");
}
Tensor negative_cpu(const Tensor& self) {
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
#if defined(__GLIBC__)
    return float_math_kernel(self, [](double x) {
        if constexpr (false) { (void)x; }
        return erfinv(x);
    }, "erfinv");
#else
    (void)0;
    TP_THROW(NotImplementedError, "erfinv requires a C library providing erfinv");
#endif
}
Tensor logit_cpu(const Tensor& self, std::optional<Scalar> eps) {
    double e = eps.has_value() ? eps->toDouble() : 0.0;
    return float_math_kernel(self, [e](double p) {
        if (e > 0) p = std::min(std::max(p, e), 1.0 - e);
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
    // Modified Bessel I0 via power series: sum_k ((|x|/2)^k / k!)^2
    return float_math_kernel(self, [](double v) {
        double half = 0.5 * std::fabs(v);
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
        else pv = std::numeric_limits<ctype>::has_infinity \
                      ? std::numeric_limits<ctype>::infinity() \
                      : std::numeric_limits<ctype>::max(); \
        if (has_neg) nv = static_cast<ctype>(neg_v); \
        else nv = std::numeric_limits<ctype>::has_infinity \
                      ? -std::numeric_limits<ctype>::infinity() \
                      : std::numeric_limits<ctype>::lowest(); \
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
        if (x == 0.0) return 0.0;
        return x * std::log(y);
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
Tensor hypot_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        return std::hypot(x, y);
    }, "hypot");
}
Tensor nextafter_cpu(const Tensor& a, const Tensor& b) {
    return binary_float_kernel(a, b, [](double x, double y) {
        return std::nextafter(x, y);
    }, "nextafter");
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

Tensor clamp_min_scalar_cpu(const Tensor& self, Scalar min) {
    double lo = min.toDouble();
    return dtype_unary_kernel(self, [lo](auto x) -> decltype(x) {
        using T = decltype(x);
        return static_cast<double>(x) < lo ? static_cast<T>(lo) : static_cast<T>(x);
    }, "clamp_min");
}
Tensor clamp_max_scalar_cpu(const Tensor& self, Scalar max) {
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
    return dtype_unary_kernel(self, [l](auto x) -> decltype(x) {
        using T = decltype(x);
        double v = static_cast<double>(x);
        return (v > l || v < -l) ? static_cast<T>(x) : static_cast<T>(0);
    }, "hardshrink");
}
Tensor softshrink_cpu(const Tensor& self, Scalar lambd) {
    double l = lambd.toDouble();
    return dtype_unary_kernel(self, [l](auto x) -> decltype(x) {
        using T = decltype(x);
        double v = static_cast<double>(x);
        if (v > l) return static_cast<T>(v - l);
        if (v < -l) return static_cast<T>(v + l);
        return static_cast<T>(0);
    }, "softshrink");
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

Tensor amax_cpu(const Tensor& self, std::vector<int64_t> dim, bool keepdim) {
    // ReduceOps.cpp:1801 amax_out
    if (dim.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) dim.push_back(i);
    }
    return reduce_dims_impl<double>(
        self, dim, keepdim, isFloatingType(self.dtype()) ? self.dtype() : self.dtype(),
        -std::numeric_limits<double>::infinity(),
        [](double acc, double v) { return v > acc ? v : acc; },
        [](double acc) { return acc; });
}

Tensor amin_cpu(const Tensor& self, std::vector<int64_t> dim, bool keepdim) {
    if (dim.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) dim.push_back(i);
    }
    return reduce_dims_impl<double>(
        self, dim, keepdim, self.dtype(),
        std::numeric_limits<double>::infinity(),
        [](double acc, double v) { return v < acc ? v : acc; },
        [](double acc) { return acc; });
}

std::tuple<Tensor, Tensor> aminmax_cpu(const Tensor& self, std::vector<int64_t> dim, bool keepdim) {
    if (dim.empty()) {
        for (int64_t i = 0; i < self.dim(); ++i) dim.push_back(i);
    }
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

Tensor nansum_cpu(const Tensor& self, std::vector<int64_t> dim, bool keepdim) {
    // ReduceOps.cpp:1310 nansum_out: NaN treated as 0
    DType out_dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    return reduce_dims_impl<double>(
        self, dim.empty() ? std::vector<int64_t>{} : dim, keepdim, out_dt, 0.0,
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
    return Tensor::zeros({}, DType::Float64, self.device()).fill_(Scalar(result));
}

Tensor renorm_cpu(const Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
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
    parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) {
        std::vector<double> slice(d_size);
        for (int64_t si = b; si < e; ++si) {
            int64_t o = si / inner, in2 = si % inner;
            for (int64_t j = 0; j < d_size; ++j) slice[j] = sp[(o * d_size + j) * inner + in2];
            double norm = 0;
            if (pd == std::numeric_limits<double>::infinity()) {
                for (double v : slice) norm = std::max(norm, std::fabs(v));
            } else {
                double s = 0;
                for (double v : slice) s += std::pow(std::fabs(v), pd);
                norm = std::pow(s, 1.0 / pd);
            }
            double factor = norm > mn ? mn / norm : 1.0;
            for (int64_t j = 0; j < d_size; ++j)
                dp[(o * d_size + j) * inner + in2] = slice[j] * factor;
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
    std::vector<int64_t> out_shape(sc64.shape().begin(), sc64.shape().end() - 2);
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
    std::vector<int64_t> sizes(self.shape().begin(), self.shape().end());
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
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (start < 0) start += self.size(dim);
    if (start < 0 || length < 0 || start + length > self.size(dim)) {
        TP_THROW(RuntimeError, "narrow: invalid start/length for dim ", dim);
    }
    std::vector<int64_t> out_shape(static_cast<std::vector<int64_t>>(self.shape()));
    out_shape[dim] = length;
    Tensor out = Tensor::empty(out_shape, self.dtype(), self.device());
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dim, outer, inner);
    Tensor sc = self.contiguous();
    int64_t row = self.size(dim);
#define TP_NARROW_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, outer * length, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t t = b; t < e; ++t) { \
                int64_t o = t / length, k = t % length; \
                std::memcpy(dp + static_cast<int64_t>(t) * inner, \
                            sp + (o * row + start + k) * inner, inner * sizeof(ctype)); \
            } \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_NARROW_CASE)
        default: TP_THROW(TypeError, "narrow: unsupported dtype");
    }
#undef TP_NARROW_CASE
    return out;
}

std::vector<Tensor> split_with_sizes_cpu(const Tensor& self, std::vector<int64_t> split_sizes, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    int64_t total = 0;
    for (int64_t s : split_sizes) total += s;
    if (total != self.size(dim)) {
        TP_THROW(RuntimeError, "split_with_sizes: split sizes sum (", total,
                 ") expected to equal size of dim ", dim, " (", self.size(dim), ")");
    }
    std::vector<Tensor> outs;
    int64_t start = 0;
    for (int64_t len : split_sizes) {
        if (len == 0) { outs.emplace_back(); continue; }
        outs.push_back(narrow_cpu(self, dim, start, len));
        start += len;
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
    // TensorTransformations.cpp:36 flip: reverse each listed dim.
    int64_t nd = self.dim();
    std::vector<bool> flip_mask(nd, false);
    for (auto& d : dims) flip_mask[wrap_dim(d, nd)] = true;
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

Tensor roll_cpu(const Tensor& self, std::vector<int64_t> shifts, std::vector<int64_t> dims) {
    // TensorTransformations.cpp:110 roll.
    int64_t nd = self.dim();
    Tensor sc = self.contiguous();
    if (dims.empty()) {
        // flatten-roll semantics
        Tensor flat_in = sc.reshape({sc.numel()});
        if (shifts.empty()) return sc.clone();
        int64_t s = ((shifts[0] % sc.numel()) + sc.numel()) % sc.numel();
        Tensor flat_out = Tensor::empty(flat_in.shape(), flat_in.dtype(), flat_in.device());
        int64_t n = flat_in.numel();
        if (n == 0) return sc;
        // out[(i+s)%n] = in[i]
        auto worker = [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                int64_t dst = (i + s) % n;
                switch (flat_in.dtype()) {
#define TP_ROLL_F(ctype, name_) case DType::name_: reinterpret_cast<ctype*>(flat_out.data_ptr())[dst] = reinterpret_cast<const ctype*>(flat_in.data_ptr())[i]; break;
                    TENSORPLAY_FORALL_SCALAR_TYPES(TP_ROLL_F)
#undef TP_ROLL_F
                    default: break;
                }
            }
        };
        parallel_for(0, n, GRAIN_SIZE, worker);
        return flat_out.reshape(static_cast<std::vector<int64_t>>(sc.shape()));
    }
    if (dims.size() != shifts.size()) {
        TP_THROW(RuntimeError, "roll: shifts and dims must have the same length");
    }
    std::vector<int64_t> sh(nd, 0);
    for (size_t i = 0; i < dims.size(); ++i) {
        int64_t d2 = wrap_dim(dims[i], nd);
        int64_t sz = sc.size(d2);
        int64_t s = ((shifts[i] % sz) + sz) % sz;
        sh[d2] = s;
    }
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(sc.shape()), sc.dtype(), sc.device());
    int64_t n = sc.numel();
    auto worker = [&](int64_t b, int64_t e) {
        for (int64_t li = b; li < e; ++li) {
            int64_t r2 = li, src = 0, mult = 1;
            for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
                int64_t c = r2 % sc.size(d2);
                r2 /= sc.size(d2);
                int64_t sc3 = c - sh[d2];
                if (sc3 < 0) sc3 += sc.size(d2);
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

Tensor rot90_cpu(const Tensor& self, int64_t k, std::vector<int64_t> dims) {
    // TensorTransformations.cpp:145 rot90 switch.
    int64_t total_dims = self.dim();
    if (dims.size() != 2) TP_THROW(RuntimeError, "expected total rotation dims == 2");
    if (total_dims < 2) TP_THROW(RuntimeError, "expected total dims >= 2");
    int64_t dim0 = wrap_dim(dims[0], total_dims);
    int64_t dim1 = wrap_dim(dims[1], total_dims);
    if (dim0 == dim1) TP_THROW(RuntimeError, "expected rotation dims to be different");
    k = ((k % 4) + 4) % 4;
    Tensor t = self.contiguous();
    auto transpose_copy = [](const Tensor& x, int64_t a2, int64_t b2) {
        int64_t nd2 = x.dim();
        std::vector<int64_t> perm(nd2);
        for (int64_t i = 0; i < nd2; ++i) perm[i] = i;
        std::swap(perm[a2], perm[b2]);
        std::vector<int64_t> new_shape;
        for (int64_t i = 0; i < nd2; ++i) new_shape.push_back(x.size(perm[i]));
        Tensor out = Tensor::empty(new_shape, x.dtype(), x.device());
        int64_t n = x.numel();
        std::vector<int64_t> xs(x.shape().begin(), x.shape().end());
        std::vector<int64_t> xs_strides(nd2, 0);
        { int64_t s = 1; for (int64_t i = nd2 - 1; i >= 0; --i) { xs_strides[i] = s; s *= xs[i]; } }
        std::vector<int64_t> out_strides(nd2, 0);
        { int64_t s = 1; for (int64_t i = nd2 - 1; i >= 0; --i) { out_strides[i] = s; s *= new_shape[i]; } }
        auto wk = [&](int64_t b, int64_t e) {
            std::vector<int64_t> oc(nd2, 0);
            for (int64_t li = b; li < e; ++li) {
                // out axis d2 corresponds to input axis perm[d2]
                int64_t rr = li;
                for (int64_t d2 = nd2 - 1; d2 >= 0; --d2) { oc[d2] = rr % new_shape[d2]; rr /= new_shape[d2]; }
                int64_t src_lin = 0;
                for (int64_t d2 = 0; d2 < nd2; ++d2) src_lin += oc[perm[d2]] * xs_strides[d2];
                switch (x.dtype()) {
#define TP_ROT_W(ctype, name_) case DType::name_: reinterpret_cast<ctype*>(out.data_ptr())[li] = reinterpret_cast<const ctype*>(x.data_ptr())[src_lin]; break;
                    TENSORPLAY_FORALL_SCALAR_TYPES(TP_ROT_W)
#undef TP_ROT_W
                    default: break;
                }
            }
        };
        parallel_for(0, n, GRAIN_SIZE, wk);
        return out;
    };
    switch (k) {
        case 1: return transpose_copy(flip_cpu(t, {dim1}), dim0, dim1);
        case 2: return flip_cpu(flip_cpu(t, {dim0}), {dim1});
        case 3: return transpose_copy(flip_cpu(t, {dim0}), dim0, dim1);
        default: return t.clone();
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
    // ij-indexing semantics.
    size_t k = tensors.size();
    if (k == 0) return {};
    std::vector<int64_t> sizes;
    sizes.reserve(k);
    for (auto& t : tensors) sizes.push_back(static_cast<int64_t>(t.numel()));
    std::vector<int64_t> grid_shape(sizes.begin(), sizes.end());
    std::vector<Tensor> outs;
    int64_t total = 1;
    for (int64_t s : sizes) total *= s;
    for (size_t j = 0; j < k; ++j) {
        Tensor g = Tensor::empty(grid_shape, DType::Int64, tensors[0].device());
        int64_t* gp = g.data_ptr<int64_t>();
        // coordinate along axis j equals decoded index at dim j
        parallel_for(0, total, GRAIN_SIZE, [&](int64_t b, int64_t e) {
            for (int64_t li = b; li < e; ++li) {
                int64_t r2 = li;
                std::vector<int64_t> coords(k, 0);
                for (int64_t d2 = static_cast<int64_t>(k) - 1; d2 >= 0; --d2) {
                    coords[d2] = r2 % sizes[d2];
                    r2 /= sizes[d2];
                }
                gp[li] = coords[j];
            }
        });
        outs.push_back(g);
    }
    return outs;
}

std::vector<Tensor> broadcast_tensors_cpu(const std::vector<Tensor>& tensors) {
    std::vector<int64_t> shape{};
    for (auto& t : tensors) {
        std::vector<int64_t> ts(t.shape().begin(), t.shape().end());
        shape = broadcast_shapes(shape, ts);
    }
    std::vector<Tensor> outs;
    for (auto& t : tensors) {
        std::vector<int64_t> ts(t.shape().begin(), t.shape().end());
        if (ts == shape) { outs.push_back(t); continue; }
        // manual broadcast copy (stride-0 on broadcast dims)
        Tensor src = t.contiguous();
        Tensor out = Tensor::empty(shape, t.dtype(), t.device());
        int64_t nd = static_cast<int64_t>(shape.size());
        std::vector<int64_t> padded(nd, 1), src_strides_padded(nd, 0);
        int64_t off = nd - static_cast<int64_t>(ts.size());
        for (int64_t i = 0; i < static_cast<int64_t>(ts.size()); ++i) padded[off + i] = ts[i];
        { int64_t s = 1; for (int64_t i = static_cast<int64_t>(ts.size()) - 1; i >= 0; --i) { src_strides_padded[off + i] = s; s *= ts[i]; } }
        int64_t n = out.numel();
        auto wk = [&](int64_t b, int64_t e) {
            for (int64_t li = b; li < e; ++li) {
                int64_t r2 = li, src_off = 0, mult = 1;
                for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
                    int64_t c = r2 % shape[d2];
                    r2 /= shape[d2];
                    src_off += (padded[d2] == 1 ? 0 : c) * src_strides_padded[d2];
                    (void)mult;
                }
                (void)mult;
                switch (t.dtype()) {
#define TP_BC_W(ctype, name_) case DType::name_: reinterpret_cast<ctype*>(out.data_ptr())[li] = reinterpret_cast<const ctype*>(src.data_ptr())[src_off]; break;
                    TENSORPLAY_FORALL_SCALAR_TYPES(TP_BC_W)
#undef TP_BC_W
                    default: break;
                }
            }
        };
        parallel_for(0, n, GRAIN_SIZE, wk);
        outs.push_back(out);
    }
    return outs;
}

Tensor block_diag_cpu(const std::vector<Tensor>& tensors) {
    int64_t total = 0;
    for (auto& t : tensors) {
        if (t.dim() != 2) TP_THROW(RuntimeError, "block_diag: expecting a list of 2D tensors");
        if (t.size(0) != t.size(1)) TP_THROW(RuntimeError, "block_diag: expecting square matrices");
        total += t.size(0);
    }
    Tensor out = Tensor::zeros({total, total}, tensors.at(0).dtype(), tensors.at(0).device());
    int64_t off = 0;
    for (auto& t : tensors) {
        int64_t m = t.size(0);
        Tensor sc = t.contiguous();
        for (int64_t r = 0; r < m; ++r) {
            for (int64_t c = 0; c < m; ++c) {
                switch (t.dtype()) {
#define TP_BD_W(ctype, name_) \
    case DType::name_: \
        out.data_ptr<ctype>()[(off + r) * total + off + c] = sc.data_ptr<ctype>()[r * m + c]; \
        break;
                    TENSORPLAY_FORALL_SCALAR_TYPES(TP_BD_W)
#undef TP_BD_W
                    default: break;
                }
            }
        }
        off += m;
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
    // TensorShape.cpp:4426 unfold, materialized copy.
    int64_t nd = self.dim();
    dimension = wrap_dim(dimension, nd);
    if (size <= 0) TP_THROW(RuntimeError, "unfold: size must be positive");
    if (step <= 0) TP_THROW(RuntimeError, "unfold: step must be positive");
    int64_t d_size = self.size(dimension);
    if (d_size < size) TP_THROW(RuntimeError, "unfold: maximum size for tensor at dimension ", dimension,
                                 " is ", d_size, " but size is ", size);
    int64_t count = (d_size - size) / step + 1;
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < nd; ++i) out_shape.push_back(i == dimension ? count : self.size(i));
    out_shape.push_back(size);
    Tensor out = Tensor::empty(out_shape, self.dtype(), self.device());
    int64_t outer = 1, inner = 1;
    outer_inner(static_cast<std::vector<int64_t>>(self.shape()), dimension, outer, inner);
    Tensor sc = self.contiguous();
    int64_t total = out.numel();
#define TP_UNF_CASE(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = sc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, outer * count * inner, GRAIN_SIZE, [&](int64_t b, int64_t e) { \
            for (int64_t t = b; t < e; ++t) { \
                int64_t c2 = t % inner; int64_t rest = t / inner; \
                int64_t blk = rest % count; int64_t o = rest / count; \
                for (int64_t kk = 0; kk < size; ++kk) { \
                    dp[((o * count + blk) * size) + kk] = sp[(o * d_size + blk * step + kk) * inner + c2]; \
                } \
            } \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_UNF_CASE)
        default: TP_THROW(TypeError, "unfold: unsupported dtype");
    }
#undef TP_UNF_CASE
    return out;
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
    m.impl("xlogy.Tensor", xlogy_cpu);
    m.impl("logaddexp", logaddexp_cpu);
    m.impl("logaddexp2", logaddexp2_cpu);
    m.impl("copysign.Tensor", copysign_cpu);
    m.impl("hypot", hypot_cpu);
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
    m.impl("softshrink", softshrink_cpu);
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
    m.impl("tensor_split.sections", tensor_split_cpu);
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
    m.impl("unfold.Tensor", unfold_cpu);
}

} // namespace cpu
} // namespace tensorplay
