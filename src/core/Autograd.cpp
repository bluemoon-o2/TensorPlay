#include "tensorplay/core/Autograd.h"
#include "tensorplay/core/Tensor.h"

namespace tensorplay {

AutogradMeta::AutogradMeta(bool requires_grad) : requires_grad_(requires_grad) {}

bool AutogradMeta::requires_grad() const {
    return requires_grad_;
}

void AutogradMeta::set_requires_grad(bool requires_grad) {
    requires_grad_ = requires_grad;
}

Tensor AutogradMeta::grad() const {
    if (grad_) {
        return *grad_;
    }
    return Tensor(); 
}

void AutogradMeta::set_grad(const Tensor& grad) {
    grad_ = std::make_shared<Tensor>(grad);
}

void AutogradMeta::accum_grad(const Tensor& grad) {
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(grad);
    } else {
        // TODO: Implement actual accumulation (addition)
        // For now, replace
        *grad_ = grad;
    }
}

} // namespace tensorplay
