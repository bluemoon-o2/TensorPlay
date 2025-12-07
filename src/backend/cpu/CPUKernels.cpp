#include "tensorplay/core/Tensor.h"
#include "tensorplay/core/Dispatcher.h"
#include "tensorplay/core/Generator.h"
#include "tensorplay/core/Scalar.h"
#include "tensorplay/core/TypePromotion.h"
#include <random>
#include <cstring>
#include <cmath>

namespace tensorplay {
namespace cpu {

// --- Arithmetic Kernels ---

template<typename Op>
Tensor binary_op_kernel(const Tensor& self, const Tensor& other, Op op) {
    // TODO: Broadcasting support
    if (self.shape() != other.shape()) {
        throw std::runtime_error("Broadcasting not yet supported");
    }
    
    // Type promotion
    DType result_dtype = promoteTypes(self.dtype(), other.dtype());
    Tensor result(static_cast<std::vector<int64_t>>(self.shape()), result_dtype, self.device());
    
    // Naive implementation for Float32 only for now
    if (result_dtype == DType::Float32) {
        float* r_data = result.data_ptr<float>();
        const float* s_data = self.data_ptr<float>(); // Cast if needed?
        const float* o_data = other.data_ptr<float>(); // Cast if needed?
        
        int64_t n = self.numel();
        for(int64_t i=0; i<n; ++i) {
            r_data[i] = op(s_data[i], o_data[i]);
        }
    } else {
        throw std::runtime_error("Binary ops only support Float32 for now");
    }
    return result;
}

Tensor add_kernel(const Tensor& self, const Tensor& other, Scalar alpha) {
    float alpha_val = alpha.to<float>();
    return binary_op_kernel(self, other, [alpha_val](float a, float b) { return a + alpha_val * b; });
}

Tensor sub_kernel(const Tensor& self, const Tensor& other, Scalar alpha) {
    float alpha_val = alpha.to<float>();
    return binary_op_kernel(self, other, [alpha_val](float a, float b) { return a - alpha_val * b; });
}

Tensor mul_kernel(const Tensor& self, const Tensor& other) {
    return binary_op_kernel(self, other, [](float a, float b) { return a * b; });
}

Tensor div_kernel(const Tensor& self, const Tensor& other) {
    return binary_op_kernel(self, other, [](float a, float b) { return a / b; });
}

Tensor to_kernel(Tensor& self, DType dtype, bool non_blocking, bool copy) {
    if (self.dtype() == dtype) {
        return copy ? self.clone() : self; // clone not impl yet, use copy_
    }
    // Create new tensor
    Tensor result(static_cast<std::vector<int64_t>>(self.shape()), dtype, self.device());
    result.copy_(self);
    return result;
}

// --- Factories ---

Tensor rand_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        int64_t n = t.numel();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        auto& gen = get_generator();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else {
         throw std::runtime_error("rand() only supports Float32 for now");
    }
    return t;
}

Tensor zeros_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    size_t nbytes = t.numel() * t.itemsize();
    if (t.data_ptr()) {
        std::memset(t.data_ptr(), 0, nbytes);
    }
    return t;
}

Tensor ones_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        int64_t n = t.numel();
        for (int64_t i = 0; i < n; ++i) data[i] = 1.0f;
    } else if (dtype == DType::Int64) {
        int64_t* data = t.data_ptr<int64_t>();
        int64_t n = t.numel();
        for (int64_t i = 0; i < n; ++i) data[i] = 1;
    } else {
        // Fallback or throw
        // For now support float32/int64
    }
    return t;
}

Tensor full_kernel(const std::vector<int64_t>& size, double fill_value, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        int64_t n = t.numel();
        float val = static_cast<float>(fill_value);
        for (int64_t i = 0; i < n; ++i) data[i] = val;
    } else {
        // TODO: support other types
    }
    return t;
}

Tensor arange_kernel(double start, double end, double step, DType dtype, Device device) {
    int64_t len = static_cast<int64_t>(std::ceil((end - start) / step));
    if (len < 0) len = 0;
    
    Tensor t({len}, dtype, device);
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        for (int64_t i = 0; i < len; ++i) data[i] = static_cast<float>(start + i * step);
    } else if (dtype == DType::Int64) {
        int64_t* data = t.data_ptr<int64_t>();
        for (int64_t i = 0; i < len; ++i) data[i] = static_cast<int64_t>(start + i * step);
    }
    return t;
}

// Register kernels
TENSORPLAY_REGISTER_KERNEL(rand, CPU, rand_kernel)
TENSORPLAY_REGISTER_KERNEL(zeros, CPU, zeros_kernel)
TENSORPLAY_REGISTER_KERNEL(ones, CPU, ones_kernel)
TENSORPLAY_REGISTER_KERNEL(full, CPU, full_kernel)
TENSORPLAY_REGISTER_KERNEL(arange, CPU, arange_kernel)

TENSORPLAY_REGISTER_KERNEL(add, CPU, add_kernel)
TENSORPLAY_REGISTER_KERNEL(sub, CPU, sub_kernel)
TENSORPLAY_REGISTER_KERNEL(mul, CPU, mul_kernel)
TENSORPLAY_REGISTER_KERNEL(div, CPU, div_kernel)
TENSORPLAY_REGISTER_KERNEL(to, CPU, to_kernel)

// Helper for recursive copy
template <typename T_SELF, typename T_SRC>
void copy_recursive(
    T_SELF* self_data, const std::vector<int64_t>& self_strides,
    const T_SRC* src_data, const std::vector<int64_t>& src_strides,
    const std::vector<int64_t>& sizes,
    int64_t dim,
    int64_t self_offset, int64_t src_offset) {
    
    if (sizes.empty()) { // Scalar case
        self_data[self_offset] = static_cast<T_SELF>(src_data[src_offset]);
        return;
    }

    if (dim == sizes.size() - 1) {
        int64_t n = sizes[dim];
        int64_t self_stride = self_strides[dim];
        int64_t src_stride = src_strides[dim];
        for (int64_t i = 0; i < n; ++i) {
            self_data[self_offset + i * self_stride] = static_cast<T_SELF>(src_data[src_offset + i * src_stride]);
        }
    } else {
        int64_t n = sizes[dim];
        int64_t self_stride = self_strides[dim];
        int64_t src_stride = src_strides[dim];
        for (int64_t i = 0; i < n; ++i) {
            copy_recursive(self_data, self_strides, src_data, src_strides, sizes, dim + 1, 
                           self_offset + i * self_stride, src_offset + i * src_stride);
        }
    }
}

Tensor& copy_kernel(Tensor& self, const Tensor& src) {
    if (!self.device().is_cpu() || !src.device().is_cpu()) {
        throw std::runtime_error("copy_kernel only supports CPU tensors");
    }
    
    // Dispatch based on self dtype
    if (self.dtype() == DType::Float32) {
        if (src.dtype() == DType::Float32) {
            // Optimization for contiguous same-dtype copy
            if (self.is_contiguous() && src.is_contiguous()) {
                size_t nbytes = self.numel() * self.itemsize();
                std::memcpy(self.data_ptr(), src.data_ptr(), nbytes);
                return self;
            }
            // General case
            copy_recursive(self.data_ptr<float>(), self.strides(), 
                           src.data_ptr<float>(), src.strides(), 
                           static_cast<std::vector<int64_t>>(self.shape()), 0, 0, 0);
        } else if (src.dtype() == DType::Int64) {
            copy_recursive(self.data_ptr<float>(), self.strides(), 
                           src.data_ptr<int64_t>(), src.strides(), 
                           static_cast<std::vector<int64_t>>(self.shape()), 0, 0, 0);
        } else {
            throw std::runtime_error("copy_ implementation missing for this src dtype (Float32 dst)");
        }
    } else if (self.dtype() == DType::Int64) {
         if (src.dtype() == DType::Int64) {
             if (self.is_contiguous() && src.is_contiguous()) {
                size_t nbytes = self.numel() * self.itemsize();
                std::memcpy(self.data_ptr(), src.data_ptr(), nbytes);
                return self;
            }
            copy_recursive(self.data_ptr<int64_t>(), self.strides(), 
                           src.data_ptr<int64_t>(), src.strides(), 
                           static_cast<std::vector<int64_t>>(self.shape()), 0, 0, 0);
         } else if (src.dtype() == DType::Float32) {
             copy_recursive(self.data_ptr<int64_t>(), self.strides(), 
                           src.data_ptr<float>(), src.strides(), 
                           static_cast<std::vector<int64_t>>(self.shape()), 0, 0, 0);
         } else {
             throw std::runtime_error("copy_ implementation missing for this src dtype (Int64 dst)");
         }
    } else {
        throw std::runtime_error("copy_ implementation missing for this self dtype");
    }
    
    return self;
}

TENSORPLAY_REGISTER_KERNEL(copy_, CPU, copy_kernel)

} // namespace cpu
} // namespace tensorplay
