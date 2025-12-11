#pragma once
#include "Node.h"
#include "TPXTensor.h"
#include "AutogradMetaInterface.h"

namespace tensorplay {
namespace tpx {

struct AccumulateGrad : public Node {
    std::weak_ptr<AutogradMetaInterface> meta_;
    
    explicit AccumulateGrad(const Tensor& t) : meta_(t.autograd_meta()) {}

    variable_list apply(variable_list&& inputs) override {
        if (inputs.empty() || !inputs[0].core().defined()) return {};
        
        if (auto meta = meta_.lock()) {
             Tensor grad = inputs[0];
             meta->accum_grad(grad);
        }
        return {};
    }
};

} // namespace tpx
} // namespace tensorplay
