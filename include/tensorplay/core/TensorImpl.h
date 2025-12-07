#pragma once

#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <stdexcept>
#include <random>
#include <cstring>
#include "tensorplay/core/DType.h"
#include "tensorplay/core/Device.h"
#include "tensorplay/core/Autograd.h"
#include "tensorplay/core/Storage.h"
#include "tensorplay/core/SizesAndStrides.h"

namespace tensorplay {

class Tensor;

class TensorImpl {
private:
    Storage storage_;
    size_t storage_offset_;
    SizesAndStrides sizes_and_strides_;
    DType dtype_;
    Device device_;
    VariableVersion version_counter_;
    std::shared_ptr<AutogradMetaInterface> autograd_meta_;
    bool is_contiguous_;
    bool is_channels_last_;

public:
    TensorImpl();
    TensorImpl(const std::vector<int64_t>& sizes, DType dtype, const Device& device = Device());
    TensorImpl(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, const Device& device = Device());
    TensorImpl(Storage storage, const std::vector<int64_t>& sizes, DType dtype, size_t storage_offset = 0);
    TensorImpl(Storage storage, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, size_t storage_offset = 0);
    
    // Copy/Move
    TensorImpl(const TensorImpl& other);
    TensorImpl(TensorImpl&& other) noexcept;
    TensorImpl& operator=(const TensorImpl& other);
    TensorImpl& operator=(TensorImpl&& other) noexcept;
    
    ~TensorImpl() = default;

    // Storage
    const Storage& storage() const { return storage_; }
    size_t storage_offset() const { return storage_offset_; }
    void set_storage(Storage storage) { storage_ = storage; }
    void set_storage_offset(size_t offset) { storage_offset_ = offset; }
    bool has_storage() const { return storage_.defined(); }
    
    // Metadata
    const SizesAndStrides& sizes_and_strides() const { return sizes_and_strides_; }
    const std::vector<int64_t>& sizes() const { return sizes_and_strides_.sizes(); }
    const std::vector<int64_t>& strides() const { return sizes_and_strides_.strides(); }
    int64_t size(size_t dim) const { return sizes_and_strides_.size(dim); }
    int64_t stride(size_t dim) const { return sizes_and_strides_.stride(dim); }
    size_t dim() const { return sizes_and_strides_.dim(); }
    int64_t numel() const { return sizes_and_strides_.numel(); }
    DType dtype() const { return dtype_; }
    Device device() const { return device_; }
    
    size_t itemsize() const { return elementSize(dtype_); }
    bool is_contiguous() const { return is_contiguous_; }
    
    // Autograd
    bool requires_grad() const { return autograd_meta_ && autograd_meta_->requires_grad(); }
    void set_requires_grad(bool requires_grad);
    AutogradMetaInterface* autograd_meta() const { return autograd_meta_.get(); }
    
    // Access to raw data pointer (typed)
    template<typename T>
    T* data() const {
        if (!has_storage()) return nullptr;
        // Check type?
        return static_cast<T*>(storage_.data()) + storage_offset_; 
    }
    
    void* data() const {
        if (!has_storage()) return nullptr;
        size_t elem_size = elementSize(dtype_);
        return static_cast<char*>(storage_.data()) + storage_offset_ * elem_size;
    }
};

} // namespace tensorplay
