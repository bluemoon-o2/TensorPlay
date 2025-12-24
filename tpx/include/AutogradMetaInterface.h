#pragma once

#include <memory>
#include <cstdint>
#include "Macros.h"

namespace tensorplay {
namespace tpx {

class Tensor; // Forward declaration
class Node; // Forward declaration

class TENSORPLAY_API AutogradMetaInterface {
public:
    virtual ~AutogradMetaInterface() = default;
    
    // Check if gradient computation is required
    virtual bool requires_grad() const = 0;
    
    // Set requires_grad flag
    virtual void set_requires_grad(bool requires_grad) = 0;
    
    // Get gradient tensor
    virtual tpx::Tensor grad() const = 0;
    
    // Set gradient tensor
    virtual void set_grad(const tpx::Tensor& grad) = 0;
    
    // Accumulate gradient
    virtual void accum_grad(const tpx::Tensor& grad) = 0;

    // Retain gradient (for non-leaf tensors)
    virtual bool retain_grad() const = 0;
    virtual void set_retain_grad(bool retain_grad) = 0;
    
    // Graph connections
    virtual void set_grad_fn(std::shared_ptr<Node> grad_fn) = 0;
    virtual std::shared_ptr<Node> grad_fn() const = 0;
    
    virtual void set_grad_accumulator(std::shared_ptr<Node> grad_accumulator) = 0;
    virtual std::shared_ptr<Node> grad_accumulator() const = 0;
    
    virtual uint32_t output_nr() const = 0;
    virtual void set_output_nr(uint32_t output_nr) = 0;
};

// Factory interface for creating AutogradMeta
struct TENSORPLAY_API AutogradMetaFactory {
    virtual ~AutogradMetaFactory() = default;
    virtual std::shared_ptr<AutogradMetaInterface> make() const = 0;
};

void SetAutogradMetaFactory(AutogradMetaFactory* factory);
AutogradMetaFactory* GetAutogradMetaFactory();

} // namespace tpx
} // namespace tensorplay
