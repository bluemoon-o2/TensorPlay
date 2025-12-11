#include "Autograd.h"
#include "TPXTensor.h"
#include "TensorImpl.h"
#include "AccumulateGrad.h"
#include "Engine.h"

namespace tensorplay {
namespace tpx {

struct AutogradMetaFactoryImpl : public AutogradMetaFactory {
    std::shared_ptr<AutogradMetaInterface> make() const override {
        return std::make_shared<AutogradMeta>();
    }
};

static AutogradMetaFactoryImpl global_autograd_meta_factory_impl;

// Register factory on startup
struct AutogradMetaFactoryRegister {
    AutogradMetaFactoryRegister() {
        SetAutogradMetaFactory(&global_autograd_meta_factory_impl);
    }
};

static AutogradMetaFactoryRegister global_autograd_meta_factory_register;

AutogradMeta::AutogradMeta(bool requires_grad) 
    : requires_grad_(requires_grad), retain_grad_(false), output_nr_(0) {}

bool AutogradMeta::requires_grad() const {
    return requires_grad_ || grad_fn_ != nullptr;
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
        *grad_ = grad_->add(grad);
    }
}

void AutogradMeta::set_grad_fn(std::shared_ptr<Node> grad_fn) {
    grad_fn_ = std::move(grad_fn);
}

std::shared_ptr<Node> AutogradMeta::grad_fn() const {
    return grad_fn_;
}

void AutogradMeta::set_grad_accumulator(std::shared_ptr<Node> grad_accumulator) {
    grad_accumulator_ = std::move(grad_accumulator);
}

std::shared_ptr<Node> AutogradMeta::grad_accumulator() const {
    return grad_accumulator_;
}

uint32_t AutogradMeta::output_nr() const {
    return output_nr_;
}

void AutogradMeta::set_output_nr(uint32_t output_nr) {
    output_nr_ = output_nr;
}

bool AutogradMeta::retain_grad() const {
    return retain_grad_;
}

void AutogradMeta::set_retain_grad(bool retain_grad) {
    retain_grad_ = retain_grad;
}

std::vector<Edge> collect_next_edges(const Tensor& t) {
    std::vector<Edge> edges;
    if (t.requires_grad()) {
        auto fn = t.grad_fn();
        if (fn) {
             uint32_t output_nr = 0;
             if (t.autograd_meta()) output_nr = t.autograd_meta()->output_nr();
             edges.emplace_back(fn, output_nr);
        } else {
             // Leaf
             auto meta = t.autograd_meta();
             if (meta) {
                 if (!meta->grad_accumulator()) {
                     meta->set_grad_accumulator(std::make_shared<AccumulateGrad>(t));
                 }
                 edges.emplace_back(meta->grad_accumulator(), 0);
             } else {
                 edges.emplace_back();
             }
        }
    } else {
        edges.emplace_back(); 
    }
    return edges;
}

std::vector<Edge> collect_next_edges(const Tensor& t1, const Tensor& t2) {
    std::vector<Edge> edges1 = collect_next_edges(t1);
    std::vector<Edge> edges2 = collect_next_edges(t2);
    edges1.insert(edges1.end(), edges2.begin(), edges2.end());
    return edges1;
}

void backward(const Tensor& tensor, const Tensor& gradient, bool retain_graph, bool create_graph) {
    if (!tensor.requires_grad()) {
        TP_THROW(RuntimeError, "Tensor does not require grad and does not have a grad_fn");
    }

    Tensor grad = gradient;
    if (!grad.defined()) {
        if (tensor.numel() != 1) {
             TP_THROW(RuntimeError, "grad can be implicitly created only for scalar outputs");
        }
        std::vector<float> data = {1.0f};
        grad = Tensor::tensor(data, tensor.dtype(), tensor.device()).reshape({});
    }

    std::vector<Edge> roots;
    
    if (auto fn = tensor.grad_fn()) {
        uint32_t output_nr = 0;
        if (tensor.autograd_meta()) {
            output_nr = tensor.autograd_meta()->output_nr();
        }
        roots.emplace_back(fn, output_nr);
    } else if (tensor.requires_grad()) {
        // Leaf node
        auto meta = tensor.autograd_meta();
        if (meta) {
            if (!meta->grad_accumulator()) {
                meta->set_grad_accumulator(std::make_shared<AccumulateGrad>(tensor));
            }
            roots.emplace_back(meta->grad_accumulator(), 0);
        }
    }

    std::vector<Tensor> inputs = {grad};
    Engine::get_default_engine().test_link();
    Engine::get_default_engine().execute_graph(roots, inputs, retain_graph, create_graph);
}

} // namespace tpx
} // namespace tensorplay
