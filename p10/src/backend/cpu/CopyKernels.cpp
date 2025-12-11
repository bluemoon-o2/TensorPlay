#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "Utils.h"
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

// Helper for dynamic dispatch
template <typename T>
struct TypeTag { using type = T; };

template <typename F>
void dispatch_dtype(DType dtype, F&& callback) {
    #define DISPATCH_CASE(ctype, name) \
    case DType::name: { \
        callback(TypeTag<ctype>{}); \
        return; \
    }

    switch (dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(DISPATCH_CASE)
        default:
            throw std::runtime_error("Unsupported dtype in dispatch");
    }
    #undef DISPATCH_CASE
}

Tensor& copy_kernel(Tensor& self, const Tensor& src) {
    if (!self.device().is_cpu() || !src.device().is_cpu()) {
        throw std::runtime_error("copy_kernel only supports CPU tensors");
    }
    
    dispatch_dtype(self.dtype(), [&](auto self_tag) {
        using self_t = typename decltype(self_tag)::type;
        
        dispatch_dtype(src.dtype(), [&](auto src_tag) {
            using src_t = typename decltype(src_tag)::type;
            
            // Optimization for contiguous same-dtype copy
            if (std::is_same_v<self_t, src_t> && self.is_contiguous() && src.is_contiguous()) {
                 size_t nbytes = self.numel() * self.itemsize();
                 std::memcpy(self.data_ptr(), src.data_ptr(), nbytes);
                 return;
            }
            
            copy_recursive(self.data_ptr<self_t>(), self.strides(), 
                           src.data_ptr<src_t>(), src.strides(), 
                           static_cast<std::vector<int64_t>>(self.shape()), 0, 0, 0);
        });
    });
    
    return self;
}

TENSORPLAY_REGISTER_KERNEL(to, CPU, to_kernel)

Tensor masked_select_cpu(const Tensor& self, const Tensor& mask) {
    // 1. Broadcast shapes
    std::vector<int64_t> broadcast_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(mask.shape()));
    
    Tensor self_expanded = self.expand(broadcast_shape);
    Tensor mask_expanded = mask.expand(broadcast_shape);
    
    // 2. Make contiguous for simple iteration
    Tensor self_contig = self_expanded.is_contiguous() ? self_expanded : self_expanded.clone();
    Tensor mask_contig = mask_expanded.is_contiguous() ? mask_expanded : mask_expanded.clone();
    
    int64_t numel = self_contig.numel();
    const uint8_t* mask_ptr = nullptr;
    
    // Handle mask dtype
    if (mask_contig.dtype() == DType::Bool || mask_contig.dtype() == DType::UInt8) {
        mask_ptr = mask_contig.data_ptr<uint8_t>();
    } else {
        TP_THROW(TypeError, "masked_select: mask must be Bool or Byte");
    }
    
    // 3. Count true elements
    int64_t true_count = 0;
    for (int64_t i = 0; i < numel; ++i) {
        if (mask_ptr[i]) true_count++;
    }
    
    // 4. Allocate result
    Tensor result = Tensor::empty({true_count}, self.dtype(), self.device());
    
    // 5. Fill result
    dispatch_dtype(self.dtype(), [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T* src = self_contig.data_ptr<T>();
        T* dst = result.data_ptr<T>();
        
        int64_t idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (mask_ptr[i]) {
                dst[idx++] = src[i];
            }
        }
    });
    
    return result;
}

TENSORPLAY_REGISTER_KERNEL(masked_select, CPU, masked_select_cpu)
TENSORPLAY_REGISTER_KERNEL(copy_, CPU, copy_kernel)

} // namespace cpu
} // namespace tensorplay
