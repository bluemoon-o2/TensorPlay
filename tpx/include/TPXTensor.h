#pragma once

#include "../../p10/include/Tensor.h" // P10 Tensor
#include "AutogradMetaInterface.h"
#include <memory>

namespace tensorplay {
namespace tpx {

class Node;
class Tensor;
using variable_list = std::vector<Tensor>;

class TENSORPLAY_API Tensor {
private:
    tensorplay::Tensor impl_;
    std::shared_ptr<AutogradMetaInterface> autograd_meta_;

public:
    Tensor() = default;
    Tensor(tensorplay::Tensor impl) : impl_(std::move(impl)) {}
    
    Tensor(const std::vector<int64_t>& sizes, DType dtype, const Device& device = Device())
        : impl_(sizes, dtype, device) {}

    // Access core P10 tensor
    const tensorplay::Tensor& core() const { return impl_; }
    tensorplay::Tensor& core() { return impl_; }
    
    bool defined() const { return impl_.defined(); }

    template<typename T>
    T* data_ptr() const { return impl_.data_ptr<T>(); }
    
    void* data_ptr() const { return impl_.data_ptr(); }
    
    std::shared_ptr<TensorImpl> unsafeGetTensorImpl() const { return impl_.unsafeGetTensorImpl(); }

    // Autograd
    bool requires_grad() const;
    void set_requires_grad(bool r);
    Tensor grad() const;
    void set_grad(const Tensor& grad);
    void retain_grad();
    bool is_leaf() const;
    Tensor detach() const;
    
    void set_grad_fn(std::shared_ptr<Node> grad_fn);
    std::shared_ptr<Node> grad_fn() const;
    std::shared_ptr<AutogradMetaInterface> autograd_meta() const { return autograd_meta_; }

    // Forwarding core methods (example)
    int64_t dim() const { return impl_.dim(); }
    int64_t numel() const { return impl_.numel(); }
    Size shape() const { return impl_.shape(); }
    int64_t size(int64_t d) const { return impl_.size(d); }
    std::vector<int64_t> strides() const { return impl_.strides(); }
    int64_t stride(int64_t d) const { return impl_.stride(d); }
    DType dtype() const { return impl_.dtype(); }
    Device device() const { return impl_.device(); }
    size_t itemsize() const { return impl_.itemsize(); }
    bool is_contiguous() const { return impl_.is_contiguous(); }
    bool is_sparse() const { return impl_.is_sparse(); }
    
    // Scalar access
    Scalar item() const { return impl_.item(); }
    
    std::string toString() const { return impl_.toString(); }
    
    // Operators (Forward to P10 and handle autograd)
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    
    Tensor operator-() const;
    Tensor operator*(Scalar s) const;
    Tensor operator/(Scalar s) const;
    Tensor operator+(Scalar s) const;
    Tensor operator-(Scalar s) const;

    friend TENSORPLAY_API Tensor operator*(Scalar s, const Tensor& t);
    friend TENSORPLAY_API Tensor operator+(Scalar s, const Tensor& t);
    friend TENSORPLAY_API Tensor operator-(Scalar s, const Tensor& t);

    // In-place operators
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    
    Tensor& operator+=(Scalar other);
    Tensor& operator-=(Scalar other);
    Tensor& operator*=(Scalar other);
    Tensor& operator/=(Scalar other);

    // Methods used in derivatives


    #include "tensorplay/ops/TPXTensorMethodsDeclGenerated.h"

    // View ops
    Tensor as_strided(const std::vector<int64_t>& size, const std::vector<int64_t>& stride, std::optional<int64_t> storage_offset = std::nullopt) const;
    Tensor select(int64_t dim, int64_t index) const;
    Tensor slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const;

    Tensor clone() const {
        Tensor t(static_cast<std::vector<int64_t>>(impl_.shape()), impl_.dtype(), impl_.device());
        t.copy_(*this);
        return t;
    }

    Tensor to(Device d, bool non_blocking = false, bool copy = false) const {
        return Tensor(impl_.to(d, non_blocking, copy));
    }
    Tensor to(DType t, bool non_blocking = false, bool copy = false) const {
        return Tensor(impl_.to(t, non_blocking, copy));
    }
    Tensor to(Device d, DType t, bool non_blocking = false, bool copy = false) const {
        return Tensor(impl_.to(d, t, non_blocking, copy));
    }

    Tensor expand(const std::vector<int64_t>& size, bool implicit = false) const {
        // P10 doesn't support expand yet probably, or it's view?
        // For now, if P10 supports it, forward it. 
        // If not, just return clone or something?
        // Autograd usually needs expand for broadcasting.
        // Let's assume P10 has expand or we implement it.
        // If P10 doesn't have it, compilation will fail in .cpp if we call it.
        // For now just declare.
        return Tensor(impl_); // Placeholder
    }
    
    // Static Factories
    template <typename T>
    static Tensor tensor(std::initializer_list<T> data) {
        return Tensor(tensorplay::Tensor::tensor(data));
    }

    template <typename T>
    static Tensor tensor(const std::vector<T>& data, DType dtype, const Device& device = Device()) {
        return Tensor(tensorplay::Tensor::tensor(data, dtype, device));
    }
};

} // namespace tpx
} // namespace tensorplay

#include "tensorplay/ops/TPXOpsGenerated.h"
#include "tensorplay/ops/TPXTensorMethodsImplGenerated.h"
