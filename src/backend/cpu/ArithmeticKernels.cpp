#include "tensorplay/core/Tensor.h"
#include "tensorplay/core/Dispatcher.h"
#include "tensorplay/core/Scalar.h"
#include "tensorplay/core/TypePromotion.h"
#include "tensorplay/utils/Utils.h"
#include "tensorplay/core/Exception.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef USE_MKL
#include <mkl.h>
#elif defined(USE_BLAS)
#include <cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorplay {
namespace cpu {

// --- Arithmetic Kernels ---

template<typename Op, typename MklOp>
Tensor binary_op_kernel_impl(const Tensor& self, const Tensor& other, Op op, MklOp mkl_op, bool use_mkl_op = false, bool force_float = false) {
    // Broadcast shapes
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    
    // Type promotion
    DType result_dtype = promoteTypes(self.dtype(), other.dtype());
    if (force_float && isIntegralType(result_dtype, true)) {
        result_dtype = DType::Float32;
    }

    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());
    
    // Check for optimized path (Float32, Contiguous, Same Shape)
    bool optimized = false;
    if (result_dtype == DType::Float32 && 
        self.dtype() == DType::Float32 && 
        other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() && result.is_contiguous() &&
        self.shape() == other.shape()) { // Strict shape check implies no broadcasting needed
        
        #ifdef USE_MKL
        if (use_mkl_op) {
            int64_t n = self.numel();
            mkl_op((int)n, self.data_ptr<float>(), other.data_ptr<float>(), result.data_ptr<float>());
            optimized = true;
        }
        #endif
    }
    
    if (!optimized) {
        // Fallback to recursive
        Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
        Tensor other_casted = (other.dtype() == result_dtype) ? other : other.to(result_dtype);

        Tensor self_expanded = self_casted.expand(out_shape);
        Tensor other_expanded = other_casted.expand(out_shape);
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            apply_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                     self_expanded, self_expanded.strides(), \
                                     other_expanded, other_expanded.strides(), \
                                     0, 0, 0, 0, out_shape, op); \
            break; \
        }

        switch (result_dtype) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(TypeError, "binary_op: unsupported dtype");
        }
        #undef OP_CASE
    }
    
    return result;
}

// Special handling for Add/Sub which can use BLAS AXPY
Tensor add_kernel(const Tensor& self, const Tensor& other, Scalar alpha) {
    // Check for BLAS/MKL optimization
    // result = self + alpha * other
    // Can use cblas_saxpy: y = alpha*x + y.
    // Set y = self (copy), x = other.
    
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    DType result_dtype = promoteTypes(self.dtype(), other.dtype());
    // Promote for float alpha
    if (alpha.isFloatingPoint()) {
        result_dtype = promoteTypes(result_dtype, DType::Float32);
    }
    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());

    bool optimized = false;
    if (result_dtype == DType::Float32 && 
        self.dtype() == DType::Float32 && 
        other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() && result.is_contiguous() &&
        self.shape() == other.shape()) {
        
        #if defined(USE_MKL) || defined(USE_BLAS)
        float alpha_val = alpha.to<float>();
        int64_t n = self.numel();
        // result = self
        std::memcpy(result.data_ptr<float>(), self.data_ptr<float>(), n * sizeof(float));
        // result += alpha * other
        cblas_saxpy((int)n, alpha_val, other.data_ptr<float>(), 1, result.data_ptr<float>(), 1);
        optimized = true;
        #endif
    }
    
    if (!optimized) {
        Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
        Tensor other_casted = (other.dtype() == result_dtype) ? other : other.to(result_dtype);

        Tensor self_expanded = self_casted.expand(out_shape);
        Tensor other_expanded = other_casted.expand(out_shape);
        auto op = [alpha](auto a, auto b) { 
            using T = decltype(a);
            if constexpr (std::is_floating_point_v<T>) {
                return a + alpha.to<T>() * b;
            } else {
                if (alpha.isFloatingPoint()) {
                    return static_cast<T>(a + alpha.toDouble() * b);
                } else {
                    return static_cast<T>(a + alpha.to<int64_t>() * b);
                }
            }
        };
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            apply_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                     self_expanded, self_expanded.strides(), \
                                     other_expanded, other_expanded.strides(), \
                                     0, 0, 0, 0, out_shape, op); \
            break; \
        }
        switch (result_dtype) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(TypeError, "add: unsupported dtype");
        }
        #undef OP_CASE
    }
    return result;
}

Tensor sub_kernel(const Tensor& self, const Tensor& other, Scalar alpha) {
    // sub(self, other, alpha) = self - alpha * other = self + (-alpha) * other
    // reuse add implementation logic or just call add logic?
    // But we need to implement sub_kernel separately if we want to be clean.
    // For optimization, it's saxpy with -alpha.
    
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    DType result_dtype = promoteTypes(self.dtype(), other.dtype());
    // Promote for float alpha
    if (alpha.isFloatingPoint()) {
        result_dtype = promoteTypes(result_dtype, DType::Float32);
    }
    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());

    bool optimized = false;
    if (result_dtype == DType::Float32 && 
        self.dtype() == DType::Float32 && 
        other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() && result.is_contiguous() &&
        self.shape() == other.shape()) {
        
        #if defined(USE_MKL) || defined(USE_BLAS)
        float alpha_val = alpha.to<float>();
        int64_t n = self.numel();
        std::memcpy(result.data_ptr<float>(), self.data_ptr<float>(), n * sizeof(float));
        cblas_saxpy((int)n, -alpha_val, other.data_ptr<float>(), 1, result.data_ptr<float>(), 1);
        optimized = true;
        #endif
    }
    
    if (!optimized) {
        Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
        Tensor other_casted = (other.dtype() == result_dtype) ? other : other.to(result_dtype);

        Tensor self_expanded = self_casted.expand(out_shape);
        Tensor other_expanded = other_casted.expand(out_shape);
        auto op = [alpha](auto a, auto b) { 
            using T = decltype(a);
            if constexpr (std::is_floating_point_v<T>) {
                return a - alpha.to<T>() * b;
            } else {
                if (alpha.isFloatingPoint()) {
                    return static_cast<T>(a - alpha.toDouble() * b);
                } else {
                    return static_cast<T>(a - alpha.to<int64_t>() * b);
                }
            }
        };
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            apply_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                     self_expanded, self_expanded.strides(), \
                                     other_expanded, other_expanded.strides(), \
                                     0, 0, 0, 0, out_shape, op); \
            break; \
        }
        switch (result_dtype) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(TypeError, "sub: unsupported dtype");
        }
        #undef OP_CASE
    }
    return result;
}

Tensor mul_kernel(const Tensor& self, const Tensor& other) {
    auto op = [](auto a, auto b) { return a * b; };
    #ifdef USE_MKL
    return binary_op_kernel_impl(self, other, op, vsMul, true);
    #else
    return binary_op_kernel_impl(self, other, op, [](int, float*, float*, float*){}, false);
    #endif
}

Tensor div_kernel(const Tensor& self, const Tensor& other) {
    auto op = [](auto a, auto b) { 
        if constexpr (std::is_same_v<decltype(a), bool>) {
            return static_cast<float>(a) / static_cast<float>(b);
        } else {
            return a / b;
        }
    };
    #ifdef USE_MKL
    return binary_op_kernel_impl(self, other, op, vsDiv, true, true);
    #else
    return binary_op_kernel_impl(self, other, op, [](int, float*, float*, float*){}, false, true);
    #endif
}

// --- In-place Arithmetic Kernels ---

Tensor& add_inplace_kernel(Tensor& self, const Tensor& other, Scalar alpha) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    
    if (static_cast<std::vector<int64_t>>(self.shape()) != out_shape) {
        TP_THROW(RuntimeError, "output with shape " + self.shape().toString() + " doesn't match the broadcast shape " + Size(out_shape).toString());
    }

    bool optimized = false;
    if (self.dtype() == DType::Float32 && 
        other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() &&
        self.shape() == other.shape()) {
        
        #if defined(USE_MKL) || defined(USE_BLAS)
        float alpha_val = alpha.to<float>();
        int64_t n = self.numel();
        // self += alpha * other -> saxpy
        cblas_saxpy((int)n, alpha_val, other.data_ptr<float>(), 1, self.data_ptr<float>(), 1);
        optimized = true;
        #endif
    }
    
    if (!optimized) {
        Tensor other_expanded = other.expand(out_shape);
        if (other_expanded.dtype() != self.dtype()) {
            other_expanded = other_expanded.to(self.dtype());
        }
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            auto op = [alpha](ctype a, ctype b) -> ctype { \
                if constexpr (std::is_floating_point_v<ctype>) { \
                    return a + alpha.to<ctype>() * b; \
                } else { \
                    if (alpha.isFloatingPoint()) { \
                        return static_cast<ctype>(a + alpha.toDouble() * b); \
                    } else { \
                        return static_cast<ctype>(a + alpha.to<int64_t>() * b); \
                    } \
                } \
            }; \
            apply_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                     self, self.strides(), \
                                     other_expanded, other_expanded.strides(), \
                                     0, 0, 0, 0, out_shape, op); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(TypeError, "add_: unsupported dtype");
        }
        #undef OP_CASE
    }
    return self;
}

Tensor& sub_inplace_kernel(Tensor& self, const Tensor& other, Scalar alpha) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    
    if (static_cast<std::vector<int64_t>>(self.shape()) != out_shape) {
        TP_THROW(RuntimeError, "output with shape " + self.shape().toString() + " doesn't match the broadcast shape " + Size(out_shape).toString());
    }

    bool optimized = false;
    if (self.dtype() == DType::Float32 && 
        other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() &&
        self.shape() == other.shape()) {
        
        #if defined(USE_MKL) || defined(USE_BLAS)
        float alpha_val = alpha.to<float>();
        int64_t n = self.numel();
        cblas_saxpy((int)n, -alpha_val, other.data_ptr<float>(), 1, self.data_ptr<float>(), 1);
        optimized = true;
        #endif
    }
    
    if (!optimized) {
        Tensor other_expanded = other.expand(out_shape);
        if (other_expanded.dtype() != self.dtype()) {
            other_expanded = other_expanded.to(self.dtype());
        }
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            auto op = [alpha](ctype a, ctype b) -> ctype { \
                if constexpr (std::is_floating_point_v<ctype>) { \
                    return a - alpha.to<ctype>() * b; \
                } else { \
                    if (alpha.isFloatingPoint()) { \
                        return static_cast<ctype>(a - alpha.toDouble() * b); \
                    } else { \
                        return static_cast<ctype>(a - alpha.to<int64_t>() * b); \
                    } \
                } \
            }; \
            apply_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                     self, self.strides(), \
                                     other_expanded, other_expanded.strides(), \
                                     0, 0, 0, 0, out_shape, op); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(TypeError, "sub_: unsupported dtype");
        }
        #undef OP_CASE
    }
    return self;
}

Tensor& mul_inplace_kernel(Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    
    if (static_cast<std::vector<int64_t>>(self.shape()) != out_shape) {
        TP_THROW(RuntimeError, "output with shape " + self.shape().toString() + " doesn't match the broadcast shape " + Size(out_shape).toString());
    }
    
    bool optimized = false;
    if (self.dtype() == DType::Float32 && other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() && self.shape() == other.shape()) {
        #ifdef USE_MKL
        int64_t n = self.numel();
        vsMul((int)n, self.data_ptr<float>(), other.data_ptr<float>(), self.data_ptr<float>());
        optimized = true;
        #endif
    }
    
    if (!optimized) {
        Tensor other_expanded = other.expand(out_shape);
        if (other_expanded.dtype() != self.dtype()) {
            other_expanded = other_expanded.to(self.dtype());
        }

        auto op = [](auto a, auto b) { return a * b; };
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            apply_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                     self, self.strides(), \
                                     other_expanded, other_expanded.strides(), \
                                     0, 0, 0, 0, out_shape, op); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(TypeError, "mul_: unsupported dtype");
        }
        #undef OP_CASE
    }
    return self;
}

Tensor& div_inplace_kernel(Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    
    if (static_cast<std::vector<int64_t>>(self.shape()) != out_shape) {
        TP_THROW(RuntimeError, "output with shape " + self.shape().toString() + " doesn't match the broadcast shape " + Size(out_shape).toString());
    }
    
    bool optimized = false;
    if (self.dtype() == DType::Float32 && other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() && self.shape() == other.shape()) {
        #ifdef USE_MKL
        int64_t n = self.numel();
        vsDiv((int)n, self.data_ptr<float>(), other.data_ptr<float>(), self.data_ptr<float>());
        optimized = true;
        #endif
    }
    
    if (!optimized) {
        Tensor other_expanded = other.expand(out_shape);
        if (other_expanded.dtype() != self.dtype()) {
            other_expanded = other_expanded.to(self.dtype());
        }

        auto op = [](auto a, auto b) { 
             if constexpr (std::is_same_v<decltype(a), bool>) {
                return static_cast<float>(a) / static_cast<float>(b);
            } else {
                return a / b;
            }
        };
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            apply_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                     self, self.strides(), \
                                     other_expanded, other_expanded.strides(), \
                                     0, 0, 0, 0, out_shape, op); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(TypeError, "div_: unsupported dtype");
        }
        #undef OP_CASE
    }
    return self;
}

TENSORPLAY_REGISTER_KERNEL_STR("add.Tensor", CPU, add_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("sub.Tensor", CPU, sub_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("mul.Tensor", CPU, mul_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("div.Tensor", CPU, div_kernel)

TENSORPLAY_REGISTER_KERNEL_STR("add_.Tensor", CPU, add_inplace_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("sub_.Tensor", CPU, sub_inplace_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("mul_.Tensor", CPU, mul_inplace_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("div_.Tensor", CPU, div_inplace_kernel)

} // namespace cpu
} // namespace tensorplay
