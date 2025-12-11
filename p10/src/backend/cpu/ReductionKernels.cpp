#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Utils.h"
#include <numeric>
#include <vector>
#include <algorithm>
#include <limits>

namespace tensorplay {
namespace cpu {

template <typename T>
struct Accumulator {
    static void add(T& acc, T val) { acc += val; }
    static void mul(T& acc, T val) { acc *= val; }
};

template <>
struct Accumulator<bool> {
    static void add(bool& acc, bool val) { acc = acc || val; }
    static void mul(bool& acc, bool val) { acc = acc && val; }
};

// Helper to convert any type to Scalar safely
template <typename T>
Scalar to_scalar(T val) {
    if constexpr (std::is_integral_v<T>) {
        return Scalar(static_cast<int64_t>(val));
    } else {
        return Scalar(val);
    }
}

// Helper to compute output shape for reduction
std::vector<int64_t> compute_reduction_shape(const Tensor& self, const std::vector<int64_t>& dims, bool keepdim) {
    std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<bool> is_reduced(shape.size(), false);
    
    for (int64_t d : dims) {
        int64_t dim = d;
        if (dim < 0) dim += shape.size();
        if (dim < 0 || dim >= (int64_t)shape.size()) {
             TP_THROW(RuntimeError, "Dimension out of range");
        }
        is_reduced[dim] = true;
    }
    
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (is_reduced[i]) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape[i]);
        }
    }
    return out_shape;
}

Tensor sum_kernel(const Tensor& self, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
         out_dtype = self.dtype();
         if (isIntegralType(self.dtype(), true)) {
             out_dtype = DType::Int64;
         }
    }
    
    Tensor out = Tensor::zeros({}, out_dtype, self.device()); // Scalar tensor
    
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    if (self_contig.dtype() != out_dtype) {
        self_contig = self_contig.to(out_dtype);
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype sum_val = 0; \
        const ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        for(int64_t i=0; i<n; ++i) Accumulator<ctype>::add(sum_val, data[i]); \
        out.fill_(to_scalar(sum_val)); \
        break; \
    }
    
    switch (out_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "sum not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}

Tensor sum_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
         out_dtype = self.dtype();
         if (isIntegralType(self.dtype(), true)) {
             out_dtype = DType::Int64;
         }
    }
    
    if (dims.empty()) {
        return sum_kernel(self, dtype);
    }
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);
    Tensor out = Tensor::zeros(out_shape, out_dtype, self.device());
    
    Tensor self_in = self;
    if (self.dtype() != out_dtype) {
        self_in = self.to(out_dtype);
    }
    
    std::vector<int64_t> inp_strides = static_cast<std::vector<int64_t>>(self_in.strides());
    std::vector<int64_t> out_strides = static_cast<std::vector<int64_t>>(out.strides());
    std::vector<int64_t> inp_shape = static_cast<std::vector<int64_t>>(self_in.shape());
    
    std::vector<bool> dim_mask(inp_shape.size(), false);
    for (int64_t d : dims) {
        if (d < 0) d += inp_shape.size();
        dim_mask[d] = true;
    }
    
    std::vector<int64_t> inp_dim_to_out_stride(inp_shape.size(), 0);
    int64_t out_dim_idx = 0;
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        if (dim_mask[i]) {
            inp_dim_to_out_stride[i] = 0; 
            if (keepdim) out_dim_idx++;
        } else {
            inp_dim_to_out_stride[i] = out_strides[out_dim_idx];
            out_dim_idx++;
        }
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* inp_data = self_in.data_ptr<ctype>(); \
        ctype* out_data = out.data_ptr<ctype>(); \
        \
        auto recurse = [&](auto&& self_recurse, int64_t dim, int64_t inp_off, int64_t out_off) -> void { \
            if (dim == (int64_t)inp_shape.size()) { \
                Accumulator<ctype>::add(out_data[out_off], inp_data[inp_off]); \
                return; \
            } \
            int64_t size = inp_shape[dim]; \
            int64_t i_stride = inp_strides[dim]; \
            int64_t o_stride = inp_dim_to_out_stride[dim]; \
            for (int64_t i = 0; i < size; ++i) { \
                self_recurse(self_recurse, dim + 1, inp_off + i * i_stride, out_off + i * o_stride); \
            } \
        }; \
        recurse(recurse, 0, 0, 0); \
        break; \
    }
    
    switch (out_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "sum_dim not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}

Tensor mean_kernel(const Tensor& self, DType dtype) {
    DType out_dtype = (dtype == DType::Undefined) ? (isFloatingType(self.dtype()) ? self.dtype() : DType::Float32) : dtype;
    
    Tensor s = sum_kernel(self, out_dtype);
    return s / Scalar((float)self.numel());
}

Tensor mean_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim, DType dtype) {
    DType out_dtype = (dtype == DType::Undefined) ? (isFloatingType(self.dtype()) ? self.dtype() : DType::Float32) : dtype;
    
    Tensor s = sum_dim_kernel(self, dims, keepdim, out_dtype);
    
    int64_t count = 1;
    std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(self.shape());
    for (int64_t d : dims) {
        if (d < 0) d += shape.size();
        count *= shape[d];
    }
    
    return s / Scalar((float)count);
}

TENSORPLAY_REGISTER_KERNEL(sum, CPU, sum_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("sum.dim_IntList", CPU, sum_dim_kernel)
TENSORPLAY_REGISTER_KERNEL(mean, CPU, mean_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("mean.dim", CPU, mean_dim_kernel)

template <typename T>
T get_lowest() {
    if constexpr (std::is_floating_point_v<T>) {
        return -std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::lowest();
    }
}

template <typename T>
T get_highest() {
    if constexpr (std::is_floating_point_v<T>) {
        return std::numeric_limits<T>::infinity();
    } else {
        return std::numeric_limits<T>::max();
    }
}

Tensor max_kernel(const Tensor& self) {
    Tensor out = Tensor::zeros({}, self.dtype(), self.device());
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype max_val = get_lowest<ctype>(); \
        const ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        if (n == 0) TP_THROW(RuntimeError, "max(): Expected reduction dim to be non-empty"); \
        for(int64_t i=0; i<n; ++i) { \
            if (data[i] > max_val) max_val = data[i]; \
        } \
        out.fill_(to_scalar(max_val)); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "max not implemented for this dtype");
    }
    #undef OP_CASE
    return out;
}

Tensor max_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim) {
    if (dims.empty()) return max_kernel(self);
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);
    Tensor out = Tensor::empty(out_shape, self.dtype(), self.device());
    
    std::vector<int64_t> inp_strides = static_cast<std::vector<int64_t>>(self.strides());
    std::vector<int64_t> out_strides = static_cast<std::vector<int64_t>>(out.strides());
    std::vector<int64_t> inp_shape = static_cast<std::vector<int64_t>>(self.shape());
    
    std::vector<bool> dim_mask(inp_shape.size(), false);
    for (int64_t d : dims) {
        if (d < 0) d += inp_shape.size();
        dim_mask[d] = true;
    }
    
    std::vector<int64_t> inp_dim_to_out_stride(inp_shape.size(), 0);
    int64_t out_dim_idx = 0;
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        if (dim_mask[i]) {
            inp_dim_to_out_stride[i] = 0; 
            if (keepdim) out_dim_idx++;
        } else {
            inp_dim_to_out_stride[i] = out_strides[out_dim_idx];
            out_dim_idx++;
        }
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* inp_data = self.data_ptr<ctype>(); \
        ctype* out_data = out.data_ptr<ctype>(); \
        ctype init_val = get_lowest<ctype>(); \
        int64_t out_n = out.numel(); \
        for(int64_t i=0; i<out_n; ++i) out_data[i] = init_val; \
        \
        auto recurse = [&](auto&& self_recurse, int64_t dim, int64_t inp_off, int64_t out_off) -> void { \
            if (dim == (int64_t)inp_shape.size()) { \
                if (inp_data[inp_off] > out_data[out_off]) out_data[out_off] = inp_data[inp_off]; \
                return; \
            } \
            int64_t size = inp_shape[dim]; \
            int64_t i_stride = inp_strides[dim]; \
            int64_t o_stride = inp_dim_to_out_stride[dim]; \
            for (int64_t i = 0; i < size; ++i) { \
                self_recurse(self_recurse, dim + 1, inp_off + i * i_stride, out_off + i * o_stride); \
            } \
        }; \
        recurse(recurse, 0, 0, 0); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "max_dim not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}

Tensor min_kernel(const Tensor& self) {
    Tensor out = Tensor::zeros({}, self.dtype(), self.device());
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype min_val = get_highest<ctype>(); \
        const ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        if (n == 0) TP_THROW(RuntimeError, "min(): Expected reduction dim to be non-empty"); \
        for(int64_t i=0; i<n; ++i) { \
            if (data[i] < min_val) min_val = data[i]; \
        } \
        out.fill_(to_scalar(min_val)); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "min not implemented for this dtype");
    }
    #undef OP_CASE
    return out;
}

Tensor min_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim) {
    if (dims.empty()) return min_kernel(self);
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);
    Tensor out = Tensor::empty(out_shape, self.dtype(), self.device());
    
    std::vector<int64_t> inp_strides = static_cast<std::vector<int64_t>>(self.strides());
    std::vector<int64_t> out_strides = static_cast<std::vector<int64_t>>(out.strides());
    std::vector<int64_t> inp_shape = static_cast<std::vector<int64_t>>(self.shape());
    
    std::vector<bool> dim_mask(inp_shape.size(), false);
    for (int64_t d : dims) {
        if (d < 0) d += inp_shape.size();
        dim_mask[d] = true;
    }
    
    std::vector<int64_t> inp_dim_to_out_stride(inp_shape.size(), 0);
    int64_t out_dim_idx = 0;
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        if (dim_mask[i]) {
            inp_dim_to_out_stride[i] = 0; 
            if (keepdim) out_dim_idx++;
        } else {
            inp_dim_to_out_stride[i] = out_strides[out_dim_idx];
            out_dim_idx++;
        }
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* inp_data = self.data_ptr<ctype>(); \
        ctype* out_data = out.data_ptr<ctype>(); \
        ctype init_val = get_highest<ctype>(); \
        int64_t out_n = out.numel(); \
        for(int64_t i=0; i<out_n; ++i) out_data[i] = init_val; \
        \
        auto recurse = [&](auto&& self_recurse, int64_t dim, int64_t inp_off, int64_t out_off) -> void { \
            if (dim == (int64_t)inp_shape.size()) { \
                if (inp_data[inp_off] < out_data[out_off]) out_data[out_off] = inp_data[inp_off]; \
                return; \
            } \
            int64_t size = inp_shape[dim]; \
            int64_t i_stride = inp_strides[dim]; \
            int64_t o_stride = inp_dim_to_out_stride[dim]; \
            for (int64_t i = 0; i < size; ++i) { \
                self_recurse(self_recurse, dim + 1, inp_off + i * i_stride, out_off + i * o_stride); \
            } \
        }; \
        recurse(recurse, 0, 0, 0); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "min_dim not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}

TENSORPLAY_REGISTER_KERNEL(max, CPU, max_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("max.dim", CPU, max_dim_kernel)
TENSORPLAY_REGISTER_KERNEL(min, CPU, min_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("min.dim", CPU, min_dim_kernel)

// Product
Tensor prod_kernel(const Tensor& self, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
         out_dtype = self.dtype();
         if (isIntegralType(self.dtype(), true)) {
             out_dtype = DType::Int64;
         }
    }
    
    Tensor out = Tensor::zeros({}, out_dtype, self.device());
    
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    if (self_contig.dtype() != out_dtype) {
        self_contig = self_contig.to(out_dtype);
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        ctype prod_val = 1; \
        ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        for(int64_t i=0; i<n; ++i) Accumulator<ctype>::mul(prod_val, data[i]); \
        out.fill_(to_scalar(prod_val)); \
        break; \
    }
    
    switch (out_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "prod not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}

Tensor prod_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim, DType dtype) {
    DType out_dtype = dtype;
    if (out_dtype == DType::Undefined) {
         out_dtype = self.dtype();
         if (isIntegralType(self.dtype(), true)) {
             out_dtype = DType::Int64;
         }
    }
    
    if (dims.empty()) {
        return prod_kernel(self, dtype);
    }
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);
    Tensor out = Tensor::ones(out_shape, out_dtype, self.device());
    
    Tensor self_in = self;
    if (self.dtype() != out_dtype) {
        self_in = self.to(out_dtype);
    }
    
    std::vector<int64_t> inp_strides = static_cast<std::vector<int64_t>>(self_in.strides());
    std::vector<int64_t> out_strides = static_cast<std::vector<int64_t>>(out.strides());
    std::vector<int64_t> inp_shape = static_cast<std::vector<int64_t>>(self_in.shape());
    
    std::vector<bool> dim_mask(inp_shape.size(), false);
    for (int64_t d : dims) {
        if (d < 0) d += inp_shape.size();
        dim_mask[d] = true;
    }
    
    std::vector<int64_t> inp_dim_to_out_stride(inp_shape.size(), 0);
    int64_t out_dim_idx = 0;
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        if (dim_mask[i]) {
            inp_dim_to_out_stride[i] = 0; 
            if (keepdim) out_dim_idx++;
        } else {
            inp_dim_to_out_stride[i] = out_strides[out_dim_idx];
            out_dim_idx++;
        }
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* inp_data = self_in.data_ptr<ctype>(); \
        ctype* out_data = out.data_ptr<ctype>(); \
        \
        auto recurse = [&](auto&& self_recurse, int64_t dim, int64_t inp_off, int64_t out_off) -> void { \
            if (dim == (int64_t)inp_shape.size()) { \
                Accumulator<ctype>::mul(out_data[out_off], inp_data[inp_off]); \
                return; \
            } \
            int64_t size = inp_shape[dim]; \
            int64_t i_stride = inp_strides[dim]; \
            int64_t o_stride = inp_dim_to_out_stride[dim]; \
            for (int64_t i = 0; i < size; ++i) { \
                self_recurse(self_recurse, dim + 1, inp_off + i * i_stride, out_off + i * o_stride); \
            } \
        }; \
        recurse(recurse, 0, 0, 0); \
        break; \
    }
    
    switch (out_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "prod_dim not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}

TENSORPLAY_REGISTER_KERNEL(prod, CPU, prod_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("prod.dim_IntList", CPU, prod_dim_kernel)

// All/Any
Tensor all_kernel(const Tensor& self) {
    Tensor out = Tensor::zeros({}, DType::Bool, self.device());
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        bool val = true; \
        const ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        for(int64_t i=0; i<n; ++i) { \
            if (!static_cast<bool>(data[i])) { val = false; break; } \
        } \
        out.fill_(Scalar(val)); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "all not implemented for this dtype");
    }
    #undef OP_CASE
    return out;
}

Tensor any_kernel(const Tensor& self) {
    Tensor out = Tensor::zeros({}, DType::Bool, self.device());
    Tensor self_contig = self.is_contiguous() ? self : self.clone();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        bool val = false; \
        const ctype* data = self_contig.data_ptr<ctype>(); \
        int64_t n = self_contig.numel(); \
        for(int64_t i=0; i<n; ++i) { \
            if (static_cast<bool>(data[i])) { val = true; break; } \
        } \
        out.fill_(Scalar(val)); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "any not implemented for this dtype");
    }
    #undef OP_CASE
    return out;
}

Tensor all_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim) {
    if (dims.empty()) return all_kernel(self);
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);
    Tensor out = Tensor::ones(out_shape, DType::Bool, self.device()); // Init with True
    
    std::vector<int64_t> inp_strides = static_cast<std::vector<int64_t>>(self.strides());
    std::vector<int64_t> out_strides = static_cast<std::vector<int64_t>>(out.strides());
    std::vector<int64_t> inp_shape = static_cast<std::vector<int64_t>>(self.shape());
    
    std::vector<bool> dim_mask(inp_shape.size(), false);
    for (int64_t d : dims) {
        if (d < 0) d += inp_shape.size();
        dim_mask[d] = true;
    }
    
    std::vector<int64_t> inp_dim_to_out_stride(inp_shape.size(), 0);
    int64_t out_dim_idx = 0;
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        if (dim_mask[i]) {
            inp_dim_to_out_stride[i] = 0; 
            if (keepdim) out_dim_idx++;
        } else {
            inp_dim_to_out_stride[i] = out_strides[out_dim_idx];
            out_dim_idx++;
        }
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* inp_data = self.data_ptr<ctype>(); \
        bool* out_data = out.data_ptr<bool>(); \
        \
        auto recurse = [&](auto&& self_recurse, int64_t dim, int64_t inp_off, int64_t out_off) -> void { \
            if (dim == (int64_t)inp_shape.size()) { \
                if (!static_cast<bool>(inp_data[inp_off])) out_data[out_off] = false; \
                return; \
            } \
            int64_t size = inp_shape[dim]; \
            int64_t i_stride = inp_strides[dim]; \
            int64_t o_stride = inp_dim_to_out_stride[dim]; \
            for (int64_t i = 0; i < size; ++i) { \
                self_recurse(self_recurse, dim + 1, inp_off + i * i_stride, out_off + i * o_stride); \
            } \
        }; \
        recurse(recurse, 0, 0, 0); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "all_dim not implemented for this dtype");
    }
    #undef OP_CASE
    return out;
}

Tensor any_dim_kernel(const Tensor& self, std::vector<int64_t> dims, bool keepdim) {
    if (dims.empty()) return any_kernel(self);
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, dims, keepdim);
    Tensor out = Tensor::zeros(out_shape, DType::Bool, self.device()); // Init with False
    
    std::vector<int64_t> inp_strides = static_cast<std::vector<int64_t>>(self.strides());
    std::vector<int64_t> out_strides = static_cast<std::vector<int64_t>>(out.strides());
    std::vector<int64_t> inp_shape = static_cast<std::vector<int64_t>>(self.shape());
    
    std::vector<bool> dim_mask(inp_shape.size(), false);
    for (int64_t d : dims) {
        if (d < 0) d += inp_shape.size();
        dim_mask[d] = true;
    }
    
    std::vector<int64_t> inp_dim_to_out_stride(inp_shape.size(), 0);
    int64_t out_dim_idx = 0;
    for (size_t i = 0; i < inp_shape.size(); ++i) {
        if (dim_mask[i]) {
            inp_dim_to_out_stride[i] = 0; 
            if (keepdim) out_dim_idx++;
        } else {
            inp_dim_to_out_stride[i] = out_strides[out_dim_idx];
            out_dim_idx++;
        }
    }
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* inp_data = self.data_ptr<ctype>(); \
        bool* out_data = out.data_ptr<bool>(); \
        \
        auto recurse = [&](auto&& self_recurse, int64_t dim, int64_t inp_off, int64_t out_off) -> void { \
            if (dim == (int64_t)inp_shape.size()) { \
                if (static_cast<bool>(inp_data[inp_off])) out_data[out_off] = true; \
                return; \
            } \
            int64_t size = inp_shape[dim]; \
            int64_t i_stride = inp_strides[dim]; \
            int64_t o_stride = inp_dim_to_out_stride[dim]; \
            for (int64_t i = 0; i < size; ++i) { \
                self_recurse(self_recurse, dim + 1, inp_off + i * i_stride, out_off + i * o_stride); \
            } \
        }; \
        recurse(recurse, 0, 0, 0); \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "any_dim not implemented for this dtype");
    }
    #undef OP_CASE
    return out;
}

TENSORPLAY_REGISTER_KERNEL(all, CPU, all_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("all.dim", CPU, all_dim_kernel)
TENSORPLAY_REGISTER_KERNEL(any, CPU, any_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("any.dim", CPU, any_dim_kernel)

// Argmax/Argmin
Tensor argmax_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    if (!dim.has_value()) {
        // Flatten
        Tensor self_contig = self.is_contiguous() ? self : self.clone();
        int64_t max_idx = 0;
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            const ctype* data = self_contig.data_ptr<ctype>(); \
            int64_t n = self_contig.numel(); \
            ctype max_val = get_lowest<ctype>(); \
            for(int64_t i=0; i<n; ++i) { \
                if (data[i] > max_val) { max_val = data[i]; max_idx = i; } \
            } \
            break; \
        }
        
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(NotImplementedError, "argmax not implemented for this dtype");
        }
        #undef OP_CASE
        
        Tensor out = Tensor::zeros({}, DType::Int64, self.device());
        out.fill_(Scalar(max_idx));
        return out;
    }
    
    int64_t d = dim.value();
    if (d < 0) d += self.dim();
    
    // Transpose d to end, reshape to (-1, size), find max idx per row
    Tensor t = self.transpose(d, -1);
    t = t.is_contiguous() ? t : t.clone(); // Force copy/compact
    
    int64_t size = t.size(-1);
    int64_t n_rows = t.numel() / size;
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, {d}, keepdim);
    Tensor out = Tensor::empty(out_shape, DType::Int64, self.device());
    int64_t* out_data = out.data_ptr<int64_t>();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* data = t.data_ptr<ctype>(); \
        for(int64_t i=0; i<n_rows; ++i) { \
            ctype max_val = get_lowest<ctype>(); \
            int64_t max_idx = 0; \
            for(int64_t j=0; j<size; ++j) { \
                ctype val = data[i*size + j]; \
                if (val > max_val) { max_val = val; max_idx = j; } \
            } \
            out_data[i] = max_idx; \
        } \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "argmax not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}

Tensor argmin_kernel(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    if (!dim.has_value()) {
        // Flatten
        Tensor self_contig = self.is_contiguous() ? self : self.clone();
        int64_t min_idx = 0;
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            const ctype* data = self_contig.data_ptr<ctype>(); \
            int64_t n = self_contig.numel(); \
            ctype min_val = get_highest<ctype>(); \
            for(int64_t i=0; i<n; ++i) { \
                if (data[i] < min_val) { min_val = data[i]; min_idx = i; } \
            } \
            break; \
        }
        
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(NotImplementedError, "argmin not implemented for this dtype");
        }
        #undef OP_CASE
        
        Tensor out = Tensor::zeros({}, DType::Int64, self.device());
        out.fill_(Scalar(min_idx));
        return out;
    }
    
    int64_t d = dim.value();
    if (d < 0) d += self.dim();
    
    // Transpose d to end, reshape to (-1, size), find min idx per row
    Tensor t = self.transpose(d, -1);
    t = t.is_contiguous() ? t : t.clone(); 
    
    int64_t size = t.size(-1);
    int64_t n_rows = t.numel() / size;
    
    std::vector<int64_t> out_shape = compute_reduction_shape(self, {d}, keepdim);
    Tensor out = Tensor::empty(out_shape, DType::Int64, self.device());
    int64_t* out_data = out.data_ptr<int64_t>();
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        const ctype* data = t.data_ptr<ctype>(); \
        for(int64_t i=0; i<n_rows; ++i) { \
            ctype min_val = get_highest<ctype>(); \
            int64_t min_idx = 0; \
            for(int64_t j=0; j<size; ++j) { \
                ctype val = data[i*size + j]; \
                if (val < min_val) { min_val = val; min_idx = j; } \
            } \
            out_data[i] = min_idx; \
        } \
        break; \
    }
    
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(NotImplementedError, "argmin not implemented for this dtype");
    }
    #undef OP_CASE
    
    return out;
}

TENSORPLAY_REGISTER_KERNEL(argmax, CPU, argmax_kernel)
TENSORPLAY_REGISTER_KERNEL(argmin, CPU, argmin_kernel)

// Var/Std
Tensor var_kernel(const Tensor& self, int64_t correction) {
    if (self.numel() == 0) return Tensor::empty({}, DType::Float32, self.device()).fill_(Scalar(std::numeric_limits<float>::quiet_NaN()));
    Tensor mean = self.mean();
    Tensor diff = self - mean;
    Tensor sq_diff = diff * diff;
    Tensor sum_sq = sq_diff.sum();
    int64_t n = self.numel();
    double div_val = std::max<double>(0.0, static_cast<double>(n - correction));
    return (sum_sq / Scalar(div_val)).to(self.dtype());
}

Tensor var_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, int64_t correction, bool keepdim) {
    if (self.numel() == 0) return Tensor::empty(compute_reduction_shape(self, dim, keepdim), DType::Float32, self.device()).fill_(Scalar(std::numeric_limits<float>::quiet_NaN()));
    // Make a copy of dim because mean() seems to modify it (bug in mean dispatch?)
    std::vector<int64_t> dims = dim;
    Tensor mean = self.mean(dims, true);
    // Re-copy dim for sum() just in case mean() modified it
    std::vector<int64_t> dims_sum = dim;
    Tensor diff = self - mean;
    Tensor sq_diff = diff * diff;
    Tensor sum_sq = sq_diff.sum(dims_sum, keepdim);
    int64_t n = 1;
    std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(self.shape());
    for (int64_t d : dim) {
        if (d < 0) d += shape.size();
        n *= shape[d];
    }
    double div = std::max<double>(0.0, static_cast<double>(n - correction));
    return (sum_sq / Scalar(div)).to(self.dtype());
}

Tensor std_kernel(const Tensor& self, int64_t correction) {
    return var_kernel(self, correction).sqrt();
}

Tensor std_dim_kernel(const Tensor& self, std::vector<int64_t> dim, int64_t correction, bool keepdim) {
    return var_dim_kernel(self, dim, correction, keepdim).sqrt();
}

// Norm
Tensor norm_kernel(const Tensor& self, double p) {
    if (std::isinf(p)) {
        if (p > 0) return self.abs().max();
        else return self.abs().min();
    }
    return self.abs().pow(Scalar(p)).sum().pow(Scalar(1.0/p));
}

Tensor norm_dim_kernel(const Tensor& self, std::vector<int64_t> dim, double p, bool keepdim) {
    if (std::isinf(p)) {
        if (p > 0) return self.abs().max(dim, keepdim);
        else return self.abs().min(dim, keepdim);
    }
    return self.abs().pow(Scalar(p)).sum(dim, keepdim).pow(Scalar(1.0/p));
}

TENSORPLAY_REGISTER_KERNEL(var, CPU, var_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("var.dim", CPU, var_dim_kernel)
TENSORPLAY_REGISTER_KERNEL(std, CPU, std_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("std.dim", CPU, std_dim_kernel)
TENSORPLAY_REGISTER_KERNEL(norm, CPU, norm_kernel)
TENSORPLAY_REGISTER_KERNEL_STR("norm.dim", CPU, norm_dim_kernel)

} // namespace cpu
} // namespace tensorplay
