#pragma once
#include <memory>
#include <cstdint>
#include <vector>
#include "Macros.h"
#include "TPXTensor.h"
#include "AutogradMetaInterface.h"

namespace tensorplay {
namespace tpx {

class Node; // Forward declaration
class Edge;


// Helper to collect next edges for autograd
TENSORPLAY_API std::vector<Edge> collect_next_edges(const Tensor& t);
TENSORPLAY_API std::vector<Edge> collect_next_edges(const Tensor& t1, const Tensor& t2);

TENSORPLAY_API void backward(const Tensor& tensor, const Tensor& gradient = {}, bool retain_graph = false, bool create_graph = false);

class TENSORPLAY_API AutogradMeta : public AutogradMetaInterface {
private:
    bool requires_grad_;
    bool retain_grad_;
    std::shared_ptr<Tensor> grad_;
    std::shared_ptr<Node> grad_fn_;
    std::shared_ptr<Node> grad_accumulator_;
    uint32_t output_nr_;
    
public:
    explicit AutogradMeta(bool requires_grad = false);
    
    bool requires_grad() const override;
    void set_requires_grad(bool requires_grad) override;
    Tensor grad() const override;
    void set_grad(const Tensor& grad) override;
    void accum_grad(const Tensor& grad) override;
    
    bool retain_grad() const override;
    void set_retain_grad(bool retain_grad) override;
    
    void set_grad_fn(std::shared_ptr<Node> grad_fn) override;
    std::shared_ptr<Node> grad_fn() const override;
    
    void set_grad_accumulator(std::shared_ptr<Node> grad_accumulator) override;
    std::shared_ptr<Node> grad_accumulator() const override;
    
    uint32_t output_nr() const override;
    void set_output_nr(uint32_t output_nr) override;
};

} // namespace tpx
} // namespace tensorplay
