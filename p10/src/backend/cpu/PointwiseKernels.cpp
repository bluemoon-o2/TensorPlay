#include "Tensor.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "TypePromotion.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <type_traits>

namespace tensorplay {
namespace cpu {

// --- Unary Kernels ---

// Helper for operations that preserve dtype (e.g. abs, neg, square)
template<typename Func>
Tensor unary_op_kernel(const Tensor& self, Func func) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* src = self_contig.data_ptr<ctype>(); \
        ctype* dst = result.data_ptr<ctype>(); \
        for(int64_t i=0; i<n; ++i) dst[i] = func(src[i]); \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

// Helper for operations that promote integer to float (e.g. sin, cos, exp)
template<typename Func>
Tensor unary_float_op_kernel(const Tensor& self, Func func) {
    DType out_dtype = self.dtype();
    if (isIntegralType(out_dtype)) {
        out_dtype = DType::Float32;
    }
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
    int64_t n = self.numel();
    
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    
    // We need to handle the case where input is int, output is float
    // And input is float, output is float
    
    if (isIntegralType(self.dtype())) {
        // Input int, Output float
        #define INT_CASE(ctype, name) \
        case DType::name: { \
            const ctype* src = self_contig.data_ptr<ctype>(); \
            float* dst = result.data_ptr<float>(); \
            for(int64_t i=0; i<n; ++i) dst[i] = static_cast<float>(func(static_cast<float>(src[i]))); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(INT_CASE) // This macro covers floats too, but we filtered with if
            default: TP_THROW(TypeError, "Unsupported dtype");
        }
        #undef INT_CASE
    } else {
        // Input float, Output float
        #define FLOAT_CASE(ctype, name) \
        case DType::name: { \
            const ctype* src = self_contig.data_ptr<ctype>(); \
            ctype* dst = result.data_ptr<ctype>(); \
            for(int64_t i=0; i<n; ++i) dst[i] = func(src[i]); \
            break; \
        }
        switch (self.dtype()) {
            case DType::Float32: {
                 const float* src = self_contig.data_ptr<float>();
                 float* dst = result.data_ptr<float>();
                 for(int64_t i=0; i<n; ++i) dst[i] = func(src[i]);
                 break;
            }
            case DType::Float64: {
                 const double* src = self_contig.data_ptr<double>();
                 double* dst = result.data_ptr<double>();
                 for(int64_t i=0; i<n; ++i) dst[i] = func(src[i]);
                 break;
            }
            default: TP_THROW(TypeError, "Unsupported dtype (expected float)");
        }
        #undef FLOAT_CASE
    }
    
    return result;
}

// Implementations

Tensor abs_kernel(const Tensor& self) {
    return unary_op_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_unsigned_v<T>) {
            return x;
        } else {
            return std::abs(x);
        }
    });
}

Tensor neg_kernel(const Tensor& self) {
    return unary_op_kernel(self, [](auto x) {
        if constexpr (std::is_same_v<decltype(x), bool>) {
             return x; // neg(bool) in same dtype is weird, just return x to avoid warning
        } else {
             return -x;
        }
    });
}

Tensor square_kernel(const Tensor& self) {
    return unary_op_kernel(self, [](auto x) { return x * x; });
}

Tensor sign_kernel(const Tensor& self) {
    return unary_op_kernel(self, [](auto x) {
        if constexpr (std::is_same_v<decltype(x), bool>) {
            return x ? 1 : 0;
        } else {
            if (x > 0) return 1;
            if (x < 0) return -1;
            return 0;
        }
    });
}

Tensor floor_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::floor(x); });
}

Tensor ceil_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::ceil(x); });
}

Tensor round_kernel(const Tensor& self) {
    if (isIntegralType(self.dtype())) return self.clone();
    return unary_op_kernel(self, [](auto x) { return std::round(x); });
}

// Float ops

Tensor acos_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::acos(x); }); }
Tensor acosh_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::acosh(x); }); }
Tensor asin_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::asin(x); }); }
Tensor asinh_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::asinh(x); }); }
Tensor atan_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::atan(x); }); }
Tensor atanh_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::atanh(x); }); }
Tensor cos_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::cos(x); }); }
Tensor cosh_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::cosh(x); }); }
Tensor sin_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::sin(x); }); }
Tensor sinh_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::sinh(x); }); }
Tensor tan_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::tan(x); }); }
Tensor tanh_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::tanh(x); }); }
Tensor exp_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::exp(x); }); }
Tensor log_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::log(x); }); }
Tensor sqrt_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return std::sqrt(x); }); }
Tensor rsqrt_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return 1.0 / std::sqrt(x); }); }
Tensor sigmoid_kernel(const Tensor& self) { return unary_float_op_kernel(self, [](auto x) { return 1.0 / (1.0 + std::exp(-x)); }); }

Tensor relu_kernel(const Tensor& self) {
    return unary_op_kernel(self, [](auto x) {
        using T = decltype(x);
        if constexpr (std::is_unsigned_v<T>) {
            return x;
        } else {
            return std::max(static_cast<T>(0), x);
        }
    });
}

Tensor gelu_kernel(const Tensor& self) {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        constexpr T kAlpha = static_cast<T>(0.70710678118654752440); // 1/sqrt(2)
        return static_cast<T>(0.5) * x * (static_cast<T>(1) + std::erf(x * kAlpha));
    });
}

Tensor silu_kernel(const Tensor& self) {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    return unary_float_op_kernel(self, [](auto x) {
        using T = decltype(x);
        return x / (static_cast<T>(1) + std::exp(-x));
    });
}

Tensor pow_scalar_kernel(const Tensor& self, Scalar exponent) {
    if (exponent.isFloatingPoint()) {
        double exp_val = exponent.toDouble();
        return unary_float_op_kernel(self, [exp_val](auto x) { return std::pow(x, exp_val); });
    } else {
        int64_t exp_val = exponent.to<int64_t>();
        if (exp_val < 0) {
             return unary_float_op_kernel(self, [exp_val](auto x) { return std::pow(x, static_cast<double>(exp_val)); });
        }
        return unary_op_kernel(self, [exp_val](auto x) {
             using T = decltype(x);
             return static_cast<T>(std::pow(x, exp_val));
        });
    }
}

TENSORPLAY_REGISTER_KERNEL(abs, CPU, abs_kernel)
TENSORPLAY_REGISTER_KERNEL(neg, CPU, neg_kernel)
TENSORPLAY_REGISTER_KERNEL(square, CPU, square_kernel)
TENSORPLAY_REGISTER_KERNEL(sign, CPU, sign_kernel)
TENSORPLAY_REGISTER_KERNEL(floor, CPU, floor_kernel)
TENSORPLAY_REGISTER_KERNEL(ceil, CPU, ceil_kernel)
TENSORPLAY_REGISTER_KERNEL(round, CPU, round_kernel)
TENSORPLAY_REGISTER_KERNEL(acos, CPU, acos_kernel)
TENSORPLAY_REGISTER_KERNEL(acosh, CPU, acosh_kernel)
TENSORPLAY_REGISTER_KERNEL(asin, CPU, asin_kernel)
TENSORPLAY_REGISTER_KERNEL(asinh, CPU, asinh_kernel)
TENSORPLAY_REGISTER_KERNEL(atan, CPU, atan_kernel)
TENSORPLAY_REGISTER_KERNEL(atanh, CPU, atanh_kernel)
TENSORPLAY_REGISTER_KERNEL(cos, CPU, cos_kernel)
TENSORPLAY_REGISTER_KERNEL(cosh, CPU, cosh_kernel)
TENSORPLAY_REGISTER_KERNEL(sin, CPU, sin_kernel)
TENSORPLAY_REGISTER_KERNEL(sinh, CPU, sinh_kernel)
TENSORPLAY_REGISTER_KERNEL(tan, CPU, tan_kernel)
TENSORPLAY_REGISTER_KERNEL(tanh, CPU, tanh_kernel)
TENSORPLAY_REGISTER_KERNEL(exp, CPU, exp_kernel)
TENSORPLAY_REGISTER_KERNEL(log, CPU, log_kernel)
TENSORPLAY_REGISTER_KERNEL(sqrt, CPU, sqrt_kernel)
TENSORPLAY_REGISTER_KERNEL(rsqrt, CPU, rsqrt_kernel)
TENSORPLAY_REGISTER_KERNEL(sigmoid, CPU, sigmoid_kernel)
TENSORPLAY_REGISTER_KERNEL(relu, CPU, relu_kernel)
TENSORPLAY_REGISTER_KERNEL(gelu, CPU, gelu_kernel)
TENSORPLAY_REGISTER_KERNEL(silu, CPU, silu_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("pow.Tensor_Scalar", CPU, pow_scalar_kernel)

Tensor angle_kernel(const Tensor& self) {
    // For real numbers, angle is 0 if >=0, pi if <0
    return unary_float_op_kernel(self, [](auto x) { 
        if (x >= 0) return 0.0;
        return 3.14159265358979323846; 
    });
}

// --- Binary/Ternary Kernels ---

// Helper for clamp
Tensor clamp_kernel(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    Tensor self_contig = self.is_contiguous() ? self : self.clone();

    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* src = self_contig.data_ptr<ctype>(); \
        ctype* dst = result.data_ptr<ctype>(); \
        ctype min_val = min.has_value() ? min->to<ctype>() : std::numeric_limits<ctype>::lowest(); \
        ctype max_val = max.has_value() ? max->to<ctype>() : std::numeric_limits<ctype>::max(); \
        for(int64_t i=0; i<n; ++i) { \
            ctype val = src[i]; \
            if (min.has_value() && val < min_val) val = min_val; \
            if (max.has_value() && val > max_val) val = max_val; \
            dst[i] = val; \
        } \
        break; \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "Unsupported dtype");
    }
    #undef OP_CASE
    return result;
}

// Softmax
Tensor softmax_kernel(const Tensor& self, int64_t dim, std::optional<DType> dtype) {
    DType out_dtype = dtype.value_or(self.dtype());
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;
    
    Tensor input = self.to(out_dtype);
    
    Tensor max_val = input.max({dim}, true);
    Tensor shifted = input - max_val;
    Tensor exp_val = shifted.exp();
    Tensor sum_exp = exp_val.sum({dim}, true);
    return exp_val / sum_exp;
}

TENSORPLAY_REGISTER_KERNEL(angle, CPU, angle_kernel)
TENSORPLAY_REGISTER_KERNEL(clamp, CPU, clamp_kernel)
TENSORPLAY_REGISTER_KERNEL(softmax, CPU, softmax_kernel)

// Helper for pow (Tensor, Tensor)
Tensor pow_tensor_tensor_kernel(const Tensor& self, const Tensor& exponent) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(exponent.shape()));
    DType result_dtype = promoteTypes(self.dtype(), exponent.dtype());
    
    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());
    
    Tensor self_expanded = self.expand(out_shape);
    Tensor exp_expanded = exponent.expand(out_shape);
    
    if (self_expanded.dtype() != result_dtype) self_expanded = self_expanded.to(result_dtype);
    if (exp_expanded.dtype() != result_dtype) exp_expanded = exp_expanded.to(result_dtype);
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [](ctype b, ctype e) -> ctype { \
             return static_cast<ctype>(std::pow(static_cast<double>(b), static_cast<double>(e))); \
        }; \
        apply_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                 self_expanded, self_expanded.strides(), \
                                 exp_expanded, exp_expanded.strides(), \
                                 0, 0, 0, 0, out_shape, op); \
        break; \
    }
    
    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "pow: unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

// Lerp implementations using composition
Tensor lerp_tensor_kernel(const Tensor& self, const Tensor& end, const Tensor& weight) {
    DType common_dtype = promoteTypes(self.dtype(), end.dtype());
    common_dtype = promoteTypes(common_dtype, weight.dtype());
    if (isIntegralType(common_dtype)) common_dtype = DType::Float32;
    
    // result = self + weight * (end - self)
    // Ensure all operands are cast to common_dtype
    Tensor s = self.to(common_dtype);
    Tensor e = end.to(common_dtype);
    Tensor w = weight.to(common_dtype);
    
    return s + w * (e - s);
}

Tensor lerp_scalar_kernel(const Tensor& self, const Tensor& end, Scalar weight) {
    DType common_dtype = promoteTypes(self.dtype(), end.dtype());
    if (weight.isFloatingPoint()) common_dtype = promoteTypes(common_dtype, DType::Float32);
    if (isIntegralType(common_dtype)) common_dtype = DType::Float32;
    
    Tensor s = self.to(common_dtype);
    Tensor e = end.to(common_dtype);
    
    return s + weight * (e - s);
}

TENSORPLAY_REGISTER_KERNEL_STR("pow.Tensor_Tensor", CPU, pow_tensor_tensor_kernel)
TENSORPLAY_REGISTER_KERNEL(lerp, CPU, lerp_scalar_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("lerp.Tensor", CPU, lerp_tensor_kernel)

} // namespace cpu
} // namespace tensorplay
