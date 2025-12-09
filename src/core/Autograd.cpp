#include "tensorplay/core/Autograd.h"
#include "tensorplay/core/Tensor.h"

namespace tensorplay {

AutogradMeta::AutogradMeta(bool requires_grad) : requires_grad_(requires_grad), retain_grad_(false) {}

bool AutogradMeta::requires_grad() const {
    return requires_grad_;
}

void AutogradMeta::set_requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;
}

bool AutogradMeta::retain_grad() const {
    return retain_grad_;
}

void AutogradMeta::set_retain_grad(bool retain_grad) {
    retain_grad_ = retain_grad;
}

Tensor AutogradMeta::grad() const {
    if (grad_) {
        return *grad_;
    }
    return Tensor(); 
}

void AutogradMeta::set_grad(const Tensor& grad) {
    if (grad.defined()) {
        grad_ = std::make_shared<Tensor>(grad);
    } else {
        grad_ = nullptr;
    }
}

void AutogradMeta::accum_grad(const Tensor& grad) {
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(grad);
    } else {
        // Accumulate gradient
        *grad_ += grad;
    }
}

} // namespace tensorplay
