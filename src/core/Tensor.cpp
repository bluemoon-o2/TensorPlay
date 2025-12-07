#include "tensorplay/core/Tensor.h"
#include "tensorplay/core/Generator.h"
#include "tensorplay/core/Dispatcher.h"
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
    if (!impl_) throw std::runtime_error("Tensor not defined");
    if (dim < 0) dim += impl_->dim();
    return impl_->size(dim); 
}

int64_t Tensor::stride(int64_t dim) const {
    if (!impl_) throw std::runtime_error("Tensor not defined");
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
    if (impl_ && impl_->autograd_meta()) return impl_->autograd_meta()->grad();
    return Tensor();
}

void* Tensor::data_ptr() const { return impl_ ? impl_->data() : nullptr; }

Scalar Tensor::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item() only supported for 1-element tensors");
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
             throw std::runtime_error("item() not implemented for complex types yet"); \
        } \
    }

    switch (dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(ITEM_CASE)
        default: throw std::runtime_error("item() not implemented for this dtype");
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
    
    // Append size and dtype if needed
    // PyTorch style: does not print shape usually in __repr__ unless very large?
    // Actually PyTorch prints nothing else usually for default repr.
    // But we might want to add dtype if not float32?
    // For now, simple.
    
    if (dtype() != DType::Float32) {
        ss << ", dtype=" << ::tensorplay::toString(dtype());
    }
    ss << ")";
    return ss.str();
}

// View methods

Tensor Tensor::as_strided(const std::vector<int64_t>& size, const std::vector<int64_t>& stride, std::optional<int64_t> storage_offset) const {
    if (!impl_) throw std::runtime_error("Tensor not defined");
    size_t new_offset = storage_offset.value_or(impl_->storage_offset());
    return Tensor(std::make_shared<TensorImpl>(impl_->storage(), size, stride, impl_->dtype(), new_offset));
}

Tensor Tensor::view(const std::vector<int64_t>& shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("view() is only supported on contiguous tensors. Use reshape() instead.");
    }
    
    int64_t new_numel = 1;
    for (auto s : shape) new_numel *= s;
    if (new_numel != numel()) {
        throw std::runtime_error("view(): invalid shape, numel mismatch");
    }
    
    std::vector<int64_t> new_strides = SizesAndStrides::compute_contiguous_strides(shape);
    return as_strided(shape, new_strides);
}

Tensor Tensor::reshape(const std::vector<int64_t>& shape) const {
    if (is_contiguous()) {
         return view(shape);
    }
    // TODO: if not contiguous, clone then view
    return clone().view(shape);
}

Tensor Tensor::select(int64_t dim, int64_t index) const {
    if (!impl_) throw std::runtime_error("Tensor not defined");
    int64_t ndim = this->dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::out_of_range("Dimension out of range");
    
    int64_t size_dim = size(dim);
    if (index < 0) index += size_dim;
    if (index < 0 || index >= size_dim) throw std::out_of_range("Index out of range");
    
    std::vector<int64_t> new_sizes = static_cast<std::vector<int64_t>>(shape());
    std::vector<int64_t> new_strides = strides();
    
    size_t new_offset = impl_->storage_offset() + index * new_strides[dim];
    
    new_sizes.erase(new_sizes.begin() + dim);
    new_strides.erase(new_strides.begin() + dim);
    
    return as_strided(new_sizes, new_strides, new_offset);
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    if (!impl_) throw std::runtime_error("Tensor not defined");
    int64_t ndim = this->dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) throw std::out_of_range("Dimension out of range");
    
    int64_t size_dim = size(dim);
    if (start < 0) start += size_dim;
    if (end < 0) end += size_dim;
    if (start < 0) start = 0;
    if (start > size_dim) start = size_dim;
    if (end < start) end = start;
    if (end > size_dim) end = size_dim;
    if (step <= 0) throw std::invalid_argument("Step must be positive");
    
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
    if (!impl_) throw std::runtime_error("Tensor not defined");
    int64_t ndim = dim();
    int64_t new_ndim = size.size();
    
    if (new_ndim < ndim) {
        throw std::runtime_error("expand(): the number of sizes provided must be greater or equal to the number of dimensions in the tensor.");
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
                throw std::runtime_error("expand(): inconsistent tensor size.");
            }
        } else {
            // New dimension added at front
            if (new_sizes[i] == -1) throw std::runtime_error("expand(): cannot infer size for new dimension");
            new_strides[i] = 0;
        }
    }
    
    return as_strided(new_sizes, new_strides);
}

// Modification
// Recursive application of copy with type conversion
template <typename T_DST, typename T_SRC>
void apply_copy_recursive(T_DST* dst_ptr, const std::vector<int64_t>& dst_strides,
                          const T_SRC* src_ptr, const std::vector<int64_t>& src_strides,
                          int dim, int64_t dst_offset, int64_t src_offset,
                          const std::vector<int64_t>& shape) {
    int64_t size = shape[dim];
    if (dim == shape.size() - 1) {
        // Inner loop
        int64_t dst_stride = dst_strides[dim];
        int64_t src_stride = src_strides[dim];
        for (int64_t i = 0; i < size; ++i) {
            dst_ptr[dst_offset + i * dst_stride] = static_cast<T_DST>(src_ptr[src_offset + i * src_stride]);
        }
    } else {
        for (int64_t i = 0; i < size; ++i) {
            apply_copy_recursive(dst_ptr, dst_strides, src_ptr, src_strides,
                                 dim + 1,
                                 dst_offset + i * dst_strides[dim],
                                 src_offset + i * src_strides[dim],
                                 shape);
        }
    }
}

Tensor& Tensor::copy_(const Tensor& src) {
    if (numel() != src.numel()) {
        throw std::runtime_error("copy_: shapes mismatch (broadcasting not full implemented in copy_)");
    }
    
    // Fast path: Contiguous and same type
    if (is_contiguous() && src.is_contiguous() && dtype() == src.dtype()) {
        std::memcpy(data_ptr(), src.data_ptr(), numel() * itemsize());
        return *this;
    }

    // Generic path: Recursive copy with casting
    if (device().type() == DeviceType::CPU && src.device().type() == DeviceType::CPU) {
        
        #define DISPATCH_COPY_SRC(T_DST, T_SRC) \
            apply_copy_recursive<T_DST, T_SRC>( \
                this->data_ptr<T_DST>(), this->strides(), \
                src.data_ptr<T_SRC>(), src.strides(), \
                0, 0, 0, static_cast<std::vector<int64_t>>(this->shape()));

        auto dispatch_src = [&]<typename T_DST>() {
            #define SRC_CASE(ctype, name) \
            case DType::name: { DISPATCH_COPY_SRC(T_DST, ctype); break; }
            
            switch (src.dtype()) {
                TENSORPLAY_FORALL_SCALAR_TYPES(SRC_CASE)
                default: throw std::runtime_error("copy_: unsupported source dtype");
            }
            #undef SRC_CASE
        };

        #define DST_CASE(ctype, name) \
        case DType::name: { dispatch_src.template operator()<ctype>(); break; }

        switch (dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(DST_CASE)
            default: throw std::runtime_error("copy_: unsupported destination dtype");
        }
        #undef DST_CASE
        #undef DISPATCH_COPY_SRC
        
    } else {
        throw std::runtime_error("copy_ only supports CPU tensors for now");
    }

    return *this;
}

Tensor& Tensor::fill_(Scalar value) {
    // Dispatch based on internal type
    if (device().is_cpu()) {
        #define FILL_CASE(ctype, name) \
        case DType::name: { \
            ctype val = value.to<ctype>(); \
            ctype* data = data_ptr<ctype>(); \
            int64_t n = numel(); \
            for (int64_t i = 0; i < n; ++i) data[i] = val; \
            break; \
        }

        switch (dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(FILL_CASE)
            default: throw std::runtime_error("Fill not implemented for this dtype");
        }
        #undef FILL_CASE
    }
    return *this;
}

Tensor& Tensor::zero_() {
    return fill_(Scalar(0));
}

Tensor Tensor::clone() const {
    Tensor t = Tensor::empty(static_cast<std::vector<int64_t>>(shape()), dtype(), device());
    t.copy_(*this);
    return t;
}

// Factories
Tensor Tensor::empty(const std::vector<int64_t>& size, DType dtype, Device device) {
    return Tensor(size, dtype, device);
}

Tensor Tensor::full(const std::vector<int64_t>& size, Scalar fill_value, DType dtype, Device device) {
    if (dtype == DType::Undefined) dtype = fill_value.dtype();
    Tensor t = empty(size, dtype, device);
    t.fill_(fill_value);
    return t;
}

Tensor Tensor::zeros(const std::vector<int64_t>& size, DType dtype, Device device) {
    return full(size, Scalar(0), dtype, device);
}

Tensor Tensor::ones(const std::vector<int64_t>& size, DType dtype, Device device) {
    return full(size, Scalar(1), dtype, device);
}

Tensor Tensor::eye(int64_t n, int64_t m, DType dtype, Device device) {
    if (m == -1) m = n;
    Tensor t = zeros({n, m}, dtype, device);
    int64_t min_dim = std::min(n, m);
    
    if (device.is_cpu()) {
        #define EYE_CASE(ctype, name) \
        case DType::name: { \
            ctype* ptr = t.data_ptr<ctype>(); \
            for (int64_t i = 0; i < min_dim; ++i) { \
                int64_t offset = i * m + i; \
                ptr[offset] = static_cast<ctype>(1); \
            } \
            break; \
        }

        switch (dtype) {
            TENSORPLAY_FORALL_SCALAR_TYPES(EYE_CASE)
            default: {
                 // Fallback for non-contiguous or unknown types (though zeros guarantees contiguous here)
                 for (int64_t i = 0; i < min_dim; ++i) {
                     t.select(0, i).select(0, i).fill_(Scalar(1));
                 }
            }
        }
        #undef EYE_CASE
    }
    return t;
}

Tensor Tensor::arange(Scalar start, Scalar end, Scalar step, DType dtype, Device device) {
    double s = start.to<double>();
    double e = end.to<double>();
    double st = step.to<double>();
    int64_t len = static_cast<int64_t>(std::ceil((e - s) / st));
    
    if (len < 0) len = 0; // Safety check

    if (dtype == DType::Undefined) {
        // Infer from scalars (simple rule: if any float, result float)
        if (start.isFloatingPoint() || end.isFloatingPoint() || step.isFloatingPoint()) dtype = DType::Float32;
        else dtype = DType::Int64;
    }
    
    Tensor t = empty({len}, dtype, device);
    
    if (device.is_cpu()) {
        #define ARANGE_CASE(ctype, name) \
        case DType::name: { \
            ctype* ptr = t.data_ptr<ctype>(); \
            for(int64_t i=0; i<len; ++i) ptr[i] = static_cast<ctype>(s + i * st); \
            break; \
        }

        switch (dtype) {
            TENSORPLAY_FORALL_SCALAR_TYPES(ARANGE_CASE)
            default: throw std::runtime_error("arange not implemented for this dtype");
        }
        #undef ARANGE_CASE
    }
    
    return t;
}

Tensor Tensor::arange(Scalar end, DType dtype, Device device) {
    return arange(Scalar(0), end, Scalar(1), dtype, device);
}

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

// Helper to broadcast shapes
static std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2) {
    int64_t ndim1 = shape1.size();
    int64_t ndim2 = shape2.size();
    int64_t ndim = std::max(ndim1, ndim2);
    std::vector<int64_t> result_shape(ndim);
    
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t dim1 = (i < ndim - ndim1) ? 1 : shape1[i - (ndim - ndim1)];
        int64_t dim2 = (i < ndim - ndim2) ? 1 : shape2[i - (ndim - ndim2)];
        
        if (dim1 == 1) result_shape[i] = dim2;
        else if (dim2 == 1) result_shape[i] = dim1;
        else if (dim1 == dim2) result_shape[i] = dim1;
        else throw std::runtime_error("The size of tensor a must match the size of tensor b at non-singleton dimension");
    }
    return result_shape;
}

// Recursive application of binary op
template <typename T, typename Op>
void apply_op_recursive(T* out_ptr, const std::vector<int64_t>& out_strides,
                       const Tensor& a, const std::vector<int64_t>& a_strides,
                       const Tensor& b, const std::vector<int64_t>& b_strides,
                       int dim, int64_t out_offset, int64_t a_offset, int64_t b_offset,
                       const std::vector<int64_t>& shape, Op op) {
    int64_t size = shape[dim];
    if (dim == shape.size() - 1) {
        // Base case: inner loop
        T* a_data = a.data_ptr<T>();
        T* b_data = b.data_ptr<T>();
        
        for (int64_t i = 0; i < size; ++i) {
            out_ptr[out_offset + i * out_strides[dim]] = op(
                a_data[a_offset + i * a_strides[dim]],
                b_data[b_offset + i * b_strides[dim]]
            );
        }
    } else {
        for (int64_t i = 0; i < size; ++i) {
            apply_op_recursive<T>(out_ptr, out_strides, a, a_strides, b, b_strides,
                                 dim + 1,
                                 out_offset + i * out_strides[dim],
                                 a_offset + i * a_strides[dim],
                                 b_offset + i * b_strides[dim],
                                 shape, op);
        }
    }
}

template <typename Op>
Tensor binary_op(const Tensor& a, const Tensor& b, Op op) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(a.shape()), static_cast<std::vector<int64_t>>(b.shape()));
    
    // Determine output dtype (simplified)
    DType out_dtype = a.dtype(); 
    auto isFloating = [](DType dt) {
        return dt == DType::Float32 || dt == DType::Float64;
    };
    if (isFloating(b.dtype()) && !isFloating(out_dtype)) out_dtype = b.dtype();
    
    // Cast inputs if necessary (to ensure type safety in kernel)
    Tensor a_cast = (a.dtype() == out_dtype) ? a : a.to(out_dtype);
    Tensor b_cast = (b.dtype() == out_dtype) ? b : b.to(out_dtype);

    // Expand inputs to output shape (logical expansion via strides)
    Tensor a_expanded = a_cast.expand(out_shape);
    Tensor b_expanded = b_cast.expand(out_shape);
    
    Tensor out = Tensor::empty(out_shape, out_dtype, a.device());
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        apply_op_recursive<ctype>(out.data_ptr<ctype>(), out.strides(), \
                                 a_expanded, a_expanded.strides(), \
                                 b_expanded, b_expanded.strides(), \
                                 0, 0, 0, 0, out_shape, op); \
        break; \
    }

    switch (out_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: throw std::runtime_error("binary_op: unsupported dtype");
    }
    #undef OP_CASE
    
    return out;
}

// Explicit arithmetic methods
Tensor Tensor::add(const Tensor& other, Scalar alpha) const {
    if (alpha.to<double>() == 1.0) {
        return *this + other;
    }
    return *this + (other * alpha);
}

Tensor Tensor::sub(const Tensor& other, Scalar alpha) const {
    if (alpha.to<double>() == 1.0) {
        return *this - other;
    }
    return *this - (other * alpha);
}

Tensor Tensor::mul(const Tensor& other) const {
    return *this * other;
}

Tensor Tensor::div(const Tensor& other) const {
    return *this / other;
}

// Type conversion
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
        throw std::runtime_error("to(): GPU support not implemented yet");
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
            TENSORPLAY_FORALL_SCALAR_TYPES(DST_CASE_##T_SRC) \
            default: throw std::runtime_error("to(): unsupported destination dtype"); \
        }

    // Since we cannot nest macros easily with arguments, we use a trick or just templates.
    // Actually, let's use a templated helper lambda.
    auto dispatch_src = [&]<typename T_SRC>() {
        T_SRC* src_ptr = this->data_ptr<T_SRC>();
        
        #define DST_CASE(ctype, name) \
        case DType::name: { \
            ctype* dst_ptr = out.data_ptr<ctype>(); \
            for(int64_t i=0; i<n; ++i) dst_ptr[i] = static_cast<ctype>(src_ptr[i]); \
            break; \
        }

        switch(dst_dtype) {
            TENSORPLAY_FORALL_SCALAR_TYPES(DST_CASE)
            default: throw std::runtime_error("to(): unsupported destination dtype");
        }
        #undef DST_CASE
    };

    #define SRC_CASE(ctype, name) \
    case DType::name: { dispatch_src.template operator()<ctype>(); break; }

    switch(this->dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(SRC_CASE)
        default: throw std::runtime_error("to(): unsupported source dtype");
    }
    #undef SRC_CASE

    return out;
}

Tensor Tensor::operator+(const Tensor& other) const { return binary_op(*this, other, std::plus<>()); }
Tensor Tensor::operator-(const Tensor& other) const { return binary_op(*this, other, std::minus<>()); }
Tensor Tensor::operator*(const Tensor& other) const { return binary_op(*this, other, std::multiplies<>()); }
Tensor Tensor::operator/(const Tensor& other) const { 
    return binary_op(*this, other, [](auto a, auto b) {
        if constexpr (std::is_same_v<decltype(a), bool>) {
            return static_cast<float>(a) / static_cast<float>(b);
        } else {
            return a / b;
        }
    }); 
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

Tensor& Tensor::operator+=(const Tensor& other) { *this = *this + other; return *this; }
Tensor& Tensor::operator-=(const Tensor& other) { *this = *this - other; return *this; }
Tensor& Tensor::operator*=(const Tensor& other) { *this = *this * other; return *this; }
Tensor& Tensor::operator/=(const Tensor& other) { *this = *this / other; return *this; }

Tensor& Tensor::operator+=(Scalar other) { *this = *this + other; return *this; }
Tensor& Tensor::operator-=(Scalar other) { *this = *this - other; return *this; }
Tensor& Tensor::operator*=(Scalar other) { *this = *this * other; return *this; }
Tensor& Tensor::operator/=(Scalar other) { *this = *this / other; return *this; }

Tensor operator-(const Tensor& t) {
    return t * Scalar(-1);
}

Tensor Tensor::rand(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t = empty(size, dtype, device);
    if (device.is_cpu()) {
        if (dtype == DType::Float32) {
             float* ptr = t.data_ptr<float>();
             int64_t n = t.numel();
             // Simple random generation
             for(int64_t i=0; i<n; ++i) ptr[i] = static_cast<float>(std::rand()) / RAND_MAX;
        } else if (dtype == DType::Float64) {
             double* ptr = t.data_ptr<double>();
             int64_t n = t.numel();
             for(int64_t i=0; i<n; ++i) ptr[i] = static_cast<double>(std::rand()) / RAND_MAX;
        } else {
             throw std::runtime_error("rand() only implemented for float32/float64 on CPU");
        }
    } else {
         throw std::runtime_error("rand() only supports CPU");
    }
    return t;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << t.toString();
    return os;
}

} // namespace tensorplay
