#include "tensorplay/core/Tensor.h"
#include "tensorplay/core/Dispatcher.h"
#include "tensorplay/core/Scalar.h"
#include "tensorplay/core/TypePromotion.h"
#include <cstring>
#include <vector>

namespace tensorplay {
namespace cpu {

Tensor to_kernel(Tensor& self, DType dtype, bool non_blocking, bool copy) {
    if (self.dtype() == dtype) {
        return copy ? self.clone() : self; // clone not impl yet, use copy_
    }
    // Create new tensor
    Tensor result(static_cast<std::vector<int64_t>>(self.shape()), dtype, self.device());
    result.copy_(self);
    return result;
}

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

TENSORPLAY_REGISTER_KERNEL(to, CPU, to_kernel)
TENSORPLAY_REGISTER_KERNEL(copy_, CPU, copy_kernel)

} // namespace cpu
} // namespace tensorplay
