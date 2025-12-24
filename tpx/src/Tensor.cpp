#include "TPXTensor.h"
#include "Autograd.h"
#include "Node.h"
#include "ManualNodes.h" // For SelectBackward/SliceBackward
#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {
namespace tpx {

bool Tensor::requires_grad() const {
    return autograd_meta_ && autograd_meta_->requires_grad();
}

void Tensor::set_requires_grad(bool r) {
    if (r) {
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

Tensor Tensor::grad() const {
    if (autograd_meta_) return autograd_meta_->grad();
    return Tensor();
}

void Tensor::set_grad(const Tensor& grad) {
    if (!autograd_meta_) autograd_meta_ = std::make_shared<AutogradMeta>();
    autograd_meta_->set_grad(grad);
}

void Tensor::retain_grad() {
    if (!autograd_meta_) autograd_meta_ = std::make_shared<AutogradMeta>();
    autograd_meta_->set_retain_grad(true);
}

bool Tensor::is_leaf() const {
    return !autograd_meta_ || !autograd_meta_->grad_fn();
}

Tensor Tensor::detach() const {
    // Create new tpx::Tensor wrapping the same P10 Tensor but without AutogradMeta
    return Tensor(impl_);
}

void Tensor::set_grad_fn(std::shared_ptr<Node> grad_fn) {
    if (!autograd_meta_) autograd_meta_ = std::make_shared<AutogradMeta>();
    autograd_meta_->set_grad_fn(std::move(grad_fn));
}

void Tensor::set_grad_fn(std::shared_ptr<Node> grad_fn, int output_nr) {
    if (!autograd_meta_) autograd_meta_ = std::make_shared<AutogradMeta>();
    autograd_meta_->set_grad_fn(std::move(grad_fn));
    autograd_meta_->set_output_nr(output_nr);
}

std::shared_ptr<Node> Tensor::grad_fn() const {
    if (autograd_meta_) return autograd_meta_->grad_fn();
    return nullptr;
}

// Operators
Tensor Tensor::operator+(const Tensor& other) const {
    return ops::add(*this, other);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return ops::sub(*this, other);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return ops::mul(*this, other);
}

Tensor Tensor::operator/(const Tensor& other) const {
    return ops::div(*this, other);
}

Tensor Tensor::operator-() const {
    return ops::neg(*this);
}

Tensor Tensor::operator*(Scalar s) const {
    return ops::mul(*this, s);
}

Tensor Tensor::operator/(Scalar s) const {
    return ops::div(*this, s);
}

Tensor Tensor::operator+(Scalar s) const {
    return ops::add(*this, s);
}

Tensor Tensor::operator-(Scalar s) const {
    return ops::sub(*this, s);
}

// In-place operators
Tensor& Tensor::operator+=(const Tensor& other) {
    ops::add_(*this, other);
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    ops::sub_(*this, other);
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    ops::mul_(*this, other);
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    ops::div_(*this, other);
    return *this;
}

Tensor& Tensor::operator+=(Scalar s) {
    ops::add_(*this, s);
    return *this;
}

Tensor& Tensor::operator-=(Scalar s) {
    ops::sub_(*this, s);
    return *this;
}

Tensor& Tensor::operator*=(Scalar s) {
    ops::mul_(*this, s);
    return *this;
}

Tensor& Tensor::operator/=(Scalar s) {
    ops::div_(*this, s);
    return *this;
}

TENSORPLAY_API Tensor operator*(Scalar s, const Tensor& t) {
    return t * s;
}

TENSORPLAY_API Tensor operator+(Scalar s, const Tensor& t) {
    return t + s;
}

TENSORPLAY_API Tensor operator-(Scalar s, const Tensor& t) {
    return (-t) + s;
}

TENSORPLAY_API Tensor operator/(Scalar s, const Tensor& t) {
    return t.pow(-1) * s;
}

// Methods used in derivatives
Tensor Tensor::as_strided(const std::vector<int64_t>& size, const std::vector<int64_t>& stride, std::optional<int64_t> storage_offset) const {
    bool requires_grad = this->requires_grad();
    std::shared_ptr<Node> grad_fn;
    if (requires_grad) {
        grad_fn = std::make_shared<AsStridedBackward>(this->shape(), size, stride, storage_offset, this->dtype(), this->device());
        grad_fn->add_next_edge_list(collect_next_edges(*this));
    }
    
    Tensor result(impl_.as_strided(size, stride, storage_offset));
    if (requires_grad && result.core().defined()) {
        result.set_grad_fn(grad_fn);
    }
    return result;
}

Tensor Tensor::select(int64_t dim, int64_t index) const {
    bool requires_grad = this->requires_grad();
    std::shared_ptr<Node> grad_fn;
    if (requires_grad) {
        grad_fn = std::make_shared<SelectBackward>(this->shape(), dim, index, this->dtype(), this->device());
        grad_fn->add_next_edge_list(collect_next_edges(*this));
    }
    
    Tensor result(impl_.select(dim, index));
    if (requires_grad && result.core().defined()) {
        result.set_grad_fn(grad_fn);
    }
    return result;
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    bool requires_grad = this->requires_grad();
    std::shared_ptr<Node> grad_fn;
    if (requires_grad) {
        grad_fn = std::make_shared<SliceBackward>(this->shape(), dim, start, end, step, this->dtype(), this->device());
        grad_fn->add_next_edge_list(collect_next_edges(*this));
    }
    
    Tensor result(impl_.slice(dim, start, end, step));
    if (requires_grad && result.core().defined()) {
        result.set_grad_fn(grad_fn);
    }
    return result;
}

} // namespace tpx
} // namespace tensorplay
