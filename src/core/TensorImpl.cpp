#include "tensorplay/core/TensorImpl.h"
#include "tensorplay/core/Tensor.h"
#include "tensorplay/core/Allocator.h"
#include <iostream>

namespace tensorplay {

TensorImpl::TensorImpl() 
    : storage_offset_(0), dtype_(DType::Float32), device_(DeviceType::CPU),
      is_contiguous_(true), is_channels_last_(false) {}

TensorImpl::TensorImpl(const std::vector<int64_t>& sizes, DType dtype, const Device& device)
    : storage_offset_(0), sizes_and_strides_(sizes), dtype_(dtype), device_(device),
      is_contiguous_(true), is_channels_last_(false) {
    
    int64_t num_elements = sizes_and_strides_.numel();
    if (num_elements > 0) {
        size_t total_bytes = static_cast<size_t>(num_elements) * elementSize(dtype);
        Allocator* allocator = getAllocator(device.type());
        storage_ = Storage(total_bytes, allocator);
    }
}

TensorImpl::TensorImpl(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, const Device& device)
    : storage_offset_(0), sizes_and_strides_(sizes, strides), dtype_(dtype), device_(device),
      is_contiguous_(false), is_channels_last_(false) {
    
    is_contiguous_ = sizes_and_strides_.is_contiguous();
    
    int64_t num_elements = sizes_and_strides_.numel();
    if (num_elements > 0) {
        size_t total_bytes = static_cast<size_t>(num_elements) * elementSize(dtype);
        Allocator* allocator = getAllocator(device.type());
        storage_ = Storage(total_bytes, allocator);
    }
}

TensorImpl::TensorImpl(Storage storage, const std::vector<int64_t>& sizes, DType dtype, size_t storage_offset)
    : storage_(storage), storage_offset_(storage_offset), sizes_and_strides_(sizes), dtype_(dtype),
      device_(storage.device()), is_contiguous_(true), is_channels_last_(false) {
    
    is_contiguous_ = sizes_and_strides_.is_contiguous();
}

TensorImpl::TensorImpl(Storage storage, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, size_t storage_offset)
    : storage_(storage), storage_offset_(storage_offset), sizes_and_strides_(sizes, strides), dtype_(dtype),
      device_(storage.device()), is_contiguous_(false), is_channels_last_(false) {
      
    is_contiguous_ = sizes_and_strides_.is_contiguous();
}

TensorImpl::TensorImpl(const TensorImpl& other) = default;
TensorImpl::TensorImpl(TensorImpl&& other) noexcept = default;
TensorImpl& TensorImpl::operator=(const TensorImpl& other) = default;
TensorImpl& TensorImpl::operator=(TensorImpl&& other) noexcept = default;

void TensorImpl::set_requires_grad(bool requires_grad) {
    if (requires_grad) {
        if (!autograd_meta_) {
            autograd_meta_ = std::make_shared<AutogradMeta>(true);
        } else {
            autograd_meta_->set_requires_grad(true);
        }
    } else {
        if (autograd_meta_) {
            autograd_meta_->set_requires_grad(false);
        }
    }
}

} // namespace tensorplay
