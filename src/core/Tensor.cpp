#include "tensorplay/core/Tensor.h"
#include "tensorplay/core/Dispatcher.h"
#include "tensorplay/core/Exception.h"
#include "tensorplay/core/Generator.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <functional>
#include <type_traits>
#include <numeric>

namespace tensorplay {

// Size implementation
std::string Size::toString() const {
    std::ostringstream ss;
    ss << "tensorplay.Size([";
    for (size_t i = 0; i < sizes_.size(); ++i) {
        ss << sizes_[i];
        if (i < sizes_.size() - 1) ss << ", ";
    }
    ss << "])";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Size& s) {
    os << s.toString();
    return os;
}

// Constructors
Tensor::Tensor(const std::vector<int64_t>& sizes, DType dtype, const Device& device)
    : impl_(std::make_shared<TensorImpl>(sizes, dtype, device)) {}

Tensor::Tensor(Storage storage, const std::vector<int64_t>& sizes, DType dtype)
    : impl_(std::make_shared<TensorImpl>(std::move(storage), sizes, dtype)) {}

Tensor::Tensor(const std::vector<int64_t>& sizes, Scalar fill_value, const Device& device)
    : impl_(std::make_shared<TensorImpl>(sizes, fill_value.dtype(), device)) {
    fill_(fill_value);
}

// Accessors implementation
int64_t Tensor::dim() const { return impl_ ? impl_->dim() : 0; }
int64_t Tensor::numel() const { return impl_ ? impl_->numel() : 0; }
Size Tensor::shape() const { return impl_ ? Size(impl_->sizes()) : Size({}); }
std::vector<int64_t> Tensor::strides() const { return impl_ ? impl_->strides() : std::vector<int64_t>{}; }

int64_t Tensor::size(int64_t dim) const { 
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    if (dim < 0) dim += impl_->dim();
    return impl_->size(dim); 
}

int64_t Tensor::stride(int64_t dim) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    if (dim < 0) dim += impl_->dim();
    return impl_->stride(dim);
}

DType Tensor::dtype() const { return impl_ ? impl_->dtype() : DType::Undefined; }
Device Tensor::device() const { return impl_ ? impl_->device() : Device(DeviceType::Unknown); }
size_t Tensor::itemsize() const { return impl_ ? impl_->itemsize() : 0; }
bool Tensor::is_contiguous() const { return impl_ ? impl_->is_contiguous() : false; }

bool Tensor::requires_grad() const { return impl_ ? impl_->requires_grad() : false; }
void Tensor::set_requires_grad(bool r) { if (impl_) impl_->set_requires_grad(r); }

Tensor Tensor::grad() const { 
    if (impl_) return impl_->grad();
    return Tensor();
}

void Tensor::set_grad(const Tensor& grad) {
    if (impl_) impl_->set_grad(grad);
}

void Tensor::retain_grad() {
    if (impl_) impl_->retain_grad();
}

bool Tensor::is_leaf() const {
    // In PyTorch, a tensor is a leaf if:
    // 1. It does not require grad (requires_grad = False)
    // 2. It requires grad but was created by the user (does not have a grad_fn)
    // Since we don't have grad_fn yet, we consider all tensors created explicitly as leaf tensors.
    // If we implement autograd ops later, they will produce tensors with grad_fn, which are not leaf.
    // For now, checking if it has a "grad_fn" equivalent would be needed, but since we don't have it,
    // we return true for now, unless we can distinguish.
    // Actually, in our current simple implementation, all tensors are leaf tensors because we don't
    // construct the graph yet.
    return true; 
}

Tensor Tensor::detach() const {
    if (!impl_) return Tensor();
    
    // Create a new Tensor that shares the same storage
    Tensor result;
    // We need to access the storage and properties of the current tensor
    // and create a new TensorImpl with them, but WITHOUT autograd history/meta.
    // However, TensorImpl constructor takes Storage, but we need to share it properly.
    
    // We can manually construct a TensorImpl sharing storage
    std::shared_ptr<TensorImpl> new_impl = std::make_shared<TensorImpl>(
        impl_->storage(), 
        static_cast<std::vector<int64_t>>(shape()), 
        strides(),
        dtype(), 
        impl_->storage_offset()
    );
    
    // detach() means requires_grad=False
    // The default TensorImpl constructor sets autograd_meta_ to null (effectively requires_grad=False)
    
    return Tensor(std::move(new_impl));
}

void* Tensor::data_ptr() const { return impl_ ? impl_->data() : nullptr; }

Scalar Tensor::item() const {
    if (numel() != 1) {
        TP_THROW(ValueError, "item() only supported for 1-element tensors");
    }
    
    #define ITEM_CASE(ctype, name) \
    case DType::name: { \
        if constexpr (std::is_floating_point_v<ctype>) { \
            return Scalar(static_cast<double>(*data_ptr<ctype>())); \
        } else if constexpr (std::is_same_v<ctype, bool>) { \
            return Scalar(static_cast<bool>(*data_ptr<ctype>())); \
        } else if constexpr (std::is_integral_v<ctype>) { \
            return Scalar(static_cast<int64_t>(*data_ptr<ctype>())); \
        } else { \
             TP_THROW(NotImplementedError, "item() not implemented for complex types yet"); \
        } \
    }

    switch (dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(ITEM_CASE)
        default: TP_THROW(NotImplementedError, "item() not implemented for this dtype");
    }
    #undef ITEM_CASE
}

std::string Tensor::toString() const {
    if (!impl_) return "Tensor(Undefined)";
    std::ostringstream ss;
    ss << "tensor(";
    
    // Recursive data printing helper
    std::function<void(const Tensor&, int64_t, int)> print_recursive = 
        [&](const Tensor& t, int64_t current_dim, int indent) {
        if (t.dim() == 0) {
            Scalar s = t.item();
            if (s.isFloatingPoint()) {
                double val = (s.dtype() == DType::Float64) ? s.to<double>() : s.to<float>();
                if (std::abs(val - std::round(val)) < 1e-9 && std::abs(val) < 1e15) {
                     ss << (long long)std::round(val) << ".";
                } else {
                     ss << std::fixed << std::setprecision(4) << val;
                }
            } else if (s.isBoolean()) {
                ss << (s.to<bool>() ? "True" : "False");
            } else {
                ss << s.to<int64_t>();
            }
            return;
        }

        ss << "[";
        int64_t size = t.size(0);
        bool is_leaf = (t.dim() == 1);
        
        // Limit printing for large tensors
        int64_t edge_items = 3;
        if (size > 2 * edge_items) {
             // Print start
             for (int64_t i = 0; i < edge_items; ++i) {
                 if (i > 0) {
                     ss << ",";
                     if (!is_leaf) { ss << "\n"; for(int k=0; k<indent+1; ++k) ss << " "; } else ss << " ";
                 }
                 print_recursive(t.select(0, i), current_dim + 1, indent + 1);
             }
             ss << ", ...";
             if (!is_leaf) { ss << "\n"; for(int k=0; k<indent+1; ++k) ss << " "; } else ss << " ";
             
             // Print end
             for (int64_t i = size - edge_items; i < size; ++i) {
                 print_recursive(t.select(0, i), current_dim + 1, indent + 1);
                 if (i < size - 1) {
                     ss << ",";
                     if (!is_leaf) { ss << "\n"; for(int k=0; k<indent+1; ++k) ss << " "; } else ss << " ";
                 }
             }
        } else {
            for (int64_t i = 0; i < size; ++i) {
                if (i > 0) {
                    ss << ",";
                    if (!is_leaf) {
                        ss << "\n";
                        for(int k=0; k<indent+1; ++k) ss << " ";
                    } else {
                        ss << " ";
                    }
                }
                print_recursive(t.select(0, i), current_dim + 1, indent + 1);
            }
        }
        ss << "]";
    };

    if (dim() == 0) {
        print_recursive(*this, 0, 0);
    } else {
        print_recursive(*this, 0, 7); // 7 for "tensor(" length
    }
    
    // Append metadata
    if (device().type() != DeviceType::CPU) {
        ss << ", device='" << device().toString() << "'";
    }
    
    if (dtype() != DType::Float32) {
        std::string dt = ::tensorplay::toString(dtype());
        std::transform(dt.begin(), dt.end(), dt.begin(), [](unsigned char c){ return std::tolower(c); });
        ss << ", dtype=tensorplay." << dt;
    }
    
    ss << ")";
    return ss.str();
}

// View methods

Tensor Tensor::as_strided(const std::vector<int64_t>& size, const std::vector<int64_t>& stride, std::optional<int64_t> storage_offset) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    size_t new_offset = storage_offset.value_or(impl_->storage_offset());
    return Tensor(std::make_shared<TensorImpl>(impl_->storage(), size, stride, impl_->dtype(), new_offset));
}

Tensor Tensor::view(const std::vector<int64_t>& shape) const {
    if (!is_contiguous()) {
        TP_THROW(RuntimeError, "view() is only supported on contiguous tensors. Use reshape() instead.");
    }
    
    int64_t new_numel = 1;
    int infer_dim = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            if (infer_dim != -1) TP_THROW(RuntimeError, "view(): only one dimension can be inferred");
            infer_dim = i;
        } else {
            if (shape[i] < 0) TP_THROW(RuntimeError, "view(): invalid negative dimension");
            new_numel *= shape[i];
        }
    }
    
    std::vector<int64_t> final_shape = shape;
    if (infer_dim != -1) {
        if (new_numel == 0) TP_THROW(RuntimeError, "view(): cannot infer shape when other dimensions are 0");
        if (numel() % new_numel != 0) TP_THROW(RuntimeError, "view(): shape inference failed");
        final_shape[infer_dim] = numel() / new_numel;
        new_numel *= final_shape[infer_dim];
    }
    
    if (new_numel != numel()) {
        TP_THROW(RuntimeError, "view(): invalid shape, numel mismatch");
    }
    
    std::vector<int64_t> new_strides = SizesAndStrides::compute_contiguous_strides(final_shape);
    return as_strided(final_shape, new_strides);
}


Tensor Tensor::select(int64_t dim, int64_t index) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    int64_t ndim = this->dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) TP_THROW(IndexError, "Dimension out of range");
    
    int64_t size_dim = size(dim);
    if (index < 0) index += size_dim;
    if (index < 0 || index >= size_dim) TP_THROW(IndexError, "Index out of range");
    
    std::vector<int64_t> new_sizes = static_cast<std::vector<int64_t>>(shape());
    std::vector<int64_t> new_strides = strides();
    
    size_t new_offset = impl_->storage_offset() + index * new_strides[dim];
    
    new_sizes.erase(new_sizes.begin() + dim);
    new_strides.erase(new_strides.begin() + dim);
    
    return as_strided(new_sizes, new_strides, new_offset);
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    int64_t ndim = this->dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) TP_THROW(IndexError, "Dimension out of range");
    
    int64_t size_dim = size(dim);
    if (start < 0) start += size_dim;
    if (end < 0) end += size_dim;
    if (start < 0) start = 0;
    if (start > size_dim) start = size_dim;
    if (end < start) end = start;
    if (end > size_dim) end = size_dim;
    if (step <= 0) TP_THROW(ValueError, "Step must be positive");
    
    int64_t new_len = (end - start + step - 1) / step;
    if (new_len < 0) new_len = 0;
    
    std::vector<int64_t> new_sizes = static_cast<std::vector<int64_t>>(shape());
    std::vector<int64_t> new_strides = strides();
    
    new_sizes[dim] = new_len;
    new_strides[dim] *= step;
    
    size_t new_offset = impl_->storage_offset() + start * stride(dim);
    
    return as_strided(new_sizes, new_strides, new_offset);
}

Tensor Tensor::expand(const std::vector<int64_t>& size) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    int64_t ndim = dim();
    int64_t new_ndim = size.size();
    
    if (new_ndim < ndim) {
        TP_THROW(RuntimeError, "expand(): the number of sizes provided must be greater or equal to the number of dimensions in the tensor.");
    }
    
    std::vector<int64_t> new_sizes(size);
    std::vector<int64_t> new_strides(new_ndim);
    
    for (int64_t i = new_ndim - 1; i >= 0; --i) {
        int64_t offset = new_ndim - 1 - i;
        int64_t dim_index = ndim - 1 - offset;
        
        if (dim_index >= 0) {
            int64_t size_dim = this->size(dim_index);
            int64_t stride_dim = this->stride(dim_index);
            
            if (size_dim == 1 && new_sizes[i] > 1) {
                new_strides[i] = 0;
            } else if (size_dim == new_sizes[i]) {
                new_strides[i] = stride_dim;
            } else if (new_sizes[i] == -1) {
                new_sizes[i] = size_dim;
                new_strides[i] = stride_dim;
            } else {
                TP_THROW(RuntimeError, "expand(): inconsistent tensor size.");
            }
        } else {
            // New dimension added at front
            if (new_sizes[i] == -1) TP_THROW(RuntimeError, "expand(): cannot infer size for new dimension");
            new_strides[i] = 0;
        }
    }
    
    return as_strided(new_sizes, new_strides);
}

// Modification
// copy_ and fill_ are generated


Tensor& Tensor::zero_() {
    return fill_(Scalar(0));
}

Tensor Tensor::clone() const {
    Tensor t = Tensor::empty(static_cast<std::vector<int64_t>>(shape()), dtype(), device());
    t.copy_(*this);
    return t;
}

// Factories
// empty, full, zeros, ones, eye, arange are generated

// *_like factories
Tensor Tensor::empty_like(const Tensor& input, std::optional<DType> dtype, std::optional<Device> device) {
    return empty(static_cast<std::vector<int64_t>>(input.shape()), dtype.value_or(input.dtype()), device.value_or(input.device()));
}

Tensor Tensor::full_like(const Tensor& input, Scalar fill_value, std::optional<DType> dtype, std::optional<Device> device) {
    return full(static_cast<std::vector<int64_t>>(input.shape()), fill_value, dtype.value_or(input.dtype()), device.value_or(input.device()));
}

Tensor Tensor::zeros_like(const Tensor& input, std::optional<DType> dtype, std::optional<Device> device) {
    return zeros(static_cast<std::vector<int64_t>>(input.shape()), dtype.value_or(input.dtype()), device.value_or(input.device()));
}

Tensor Tensor::ones_like(const Tensor& input, std::optional<DType> dtype, std::optional<Device> device) {
    return ones(static_cast<std::vector<int64_t>>(input.shape()), dtype.value_or(input.dtype()), device.value_or(input.device()));
}

// Operators
// add, sub, mul, div, mm are generated

Tensor Tensor::operator+(const Tensor& other) const { return add(other); }
Tensor Tensor::operator-(const Tensor& other) const { return sub(other); }
Tensor Tensor::operator*(const Tensor& other) const { return mul(other); }
Tensor Tensor::operator/(const Tensor& other) const { return div(other); }


Tensor Tensor::to(DType dtype, bool non_blocking, bool copy) const {
    return to(device(), dtype, non_blocking, copy);
}

Tensor Tensor::to(Device device, bool non_blocking, bool copy) const {
    return to(device, this->dtype(), non_blocking, copy);
}

Tensor Tensor::to(Device dst_device, DType dst_dtype, bool non_blocking, bool copy) const {
    if (dst_device == this->device() && dst_dtype == this->dtype()) {
        if (copy) {
            return this->clone();
        }
        return *this;
    }

    if (dst_device.type() != DeviceType::CPU || this->device().type() != DeviceType::CPU) {
        TP_THROW(NotImplementedError, "to(): GPU support not implemented yet");
    }

    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(this->shape()), dst_dtype, dst_device);
    
    // Check contiguous. If not, clone first to make contiguous (simplified approach)
    if (!is_contiguous()) {
         Tensor c = this->clone();
         return c.to(dst_device, dst_dtype, non_blocking, copy);
    }
    
    int64_t n = numel();
    
    // Helper macro for destination dispatch
    #define DISPATCH_DST(T_SRC) \
        switch(dst_dtype) { \
            TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(DST_CASE_##T_SRC) \
            default: TP_THROW(TypeError, "to(): unsupported destination dtype"); \
        }

    // Helper for safe casting (including complex handling)
    auto cast_and_store = [&]<typename Dst, typename Src>(Dst* dst, const Src* src, int64_t n) {
        for(int64_t i=0; i<n; ++i) {
            if constexpr ((std::is_same_v<Src, std::complex<float>> || std::is_same_v<Src, std::complex<double>>) &&
                          !(std::is_same_v<Dst, std::complex<float>> || std::is_same_v<Dst, std::complex<double>>)) {
                // Complex to Scalar: take real part
                dst[i] = static_cast<Dst>(src[i].real());
            } else {
                // Scalar to Scalar, Scalar to Complex, Complex to Complex
                dst[i] = static_cast<Dst>(src[i]);
            }
        }
    };

    // Since we cannot nest macros easily with arguments, we use a trick or just templates.
    // Actually, let's use a templated helper lambda.
    auto dispatch_src = [&]<typename T_SRC>() {
        T_SRC* src_ptr = this->data_ptr<T_SRC>();
        
        #define DST_CASE(ctype, name) \
        case DType::name: { \
            ctype* dst_ptr = out.data_ptr<ctype>(); \
            cast_and_store(dst_ptr, src_ptr, n); \
            break; \
        }

        switch(dst_dtype) {
            TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(DST_CASE)
            default: 
                TP_THROW(TypeError, "to(): unsupported destination dtype");
        }
        #undef DST_CASE
    };

    #define SRC_CASE(ctype, name) \
    case DType::name: { dispatch_src.template operator()<ctype>(); break; }

    switch(this->dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(SRC_CASE)
        default: 
            TP_THROW(TypeError, "to(): unsupported source dtype");
    }
    #undef SRC_CASE

    return out;
}

// mm is generated

Tensor Tensor::matmul(const Tensor& other) const {
    if (this->dim() == 2 && other.dim() == 2) {
        return mm(other);
    }
    TP_THROW(RuntimeError, "matmul: Only 2D matrices supported for now (use mm)");
}

Tensor Tensor::operator+(Scalar other) const { 
    // Create a 0-dim tensor from scalar and add
    Tensor t = Tensor({1}, other, device()).view({}); // 0-dim scalar tensor
    return *this + t; 
}
Tensor Tensor::operator-(Scalar other) const { 
    Tensor t = Tensor({1}, other, device()).view({});
    return *this - t; 
}
Tensor Tensor::operator*(Scalar other) const { 
    Tensor t = Tensor({1}, other, device()).view({});
    return *this * t; 
}
Tensor Tensor::operator/(Scalar other) const { 
    Tensor t = Tensor({1}, other, device()).view({});
    return *this / t; 
}

Tensor& Tensor::operator+=(const Tensor& other) { return add_(other); }
Tensor& Tensor::operator-=(const Tensor& other) { return sub_(other); }
Tensor& Tensor::operator*=(const Tensor& other) { return mul_(other); }
Tensor& Tensor::operator/=(const Tensor& other) { return div_(other); }

Tensor& Tensor::operator+=(Scalar other) { 
    Tensor t = Tensor({1}, other, device()).view({});
    return add_(t); 
}
Tensor& Tensor::operator-=(Scalar other) { 
    Tensor t = Tensor({1}, other, device()).view({});
    return sub_(t); 
}
Tensor& Tensor::operator*=(Scalar other) { 
    Tensor t = Tensor({1}, other, device()).view({});
    return mul_(t); 
}
Tensor& Tensor::operator/=(Scalar other) { 
    Tensor t = Tensor({1}, other, device()).view({});
    return div_(t); 
}

Tensor operator-(const Tensor& t) {
    return t * Scalar(-1);
}


std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << t.toString();
    return os;
}

} // namespace tensorplay
