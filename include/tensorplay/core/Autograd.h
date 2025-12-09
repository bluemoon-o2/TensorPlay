#pragma once
#include <memory>
#include <cstdint>
#include "tensorplay/core/Macros.h"

namespace tensorplay {

class Tensor; // Forward declaration

class TENSORPLAY_API AutogradMetaInterface {
public:
    virtual ~AutogradMetaInterface() = default;
    
    // Check if gradient computation is required
    virtual bool requires_grad() const = 0;
    
    // Set requires_grad flag
    virtual void set_requires_grad(bool requires_grad) = 0;
    
    // Get gradient tensor
    virtual Tensor grad() const = 0;
    
    // Set gradient tensor
    virtual void set_grad(const Tensor& grad) = 0;
    
    // Accumulate gradient
    virtual void accum_grad(const Tensor& grad) = 0;

    // Retain gradient (for non-leaf tensors)
    virtual bool retain_grad() const = 0;
    virtual void set_retain_grad(bool retain_grad) = 0;
};

class TENSORPLAY_API AutogradMeta : public AutogradMetaInterface {
private:
    bool requires_grad_;
    bool retain_grad_;
    std::shared_ptr<Tensor> grad_;
    
public:
    explicit AutogradMeta(bool requires_grad = false);
    
    bool requires_grad() const override;
    void set_requires_grad(bool requires_grad) override;
    Tensor grad() const override;
    void set_grad(const Tensor& grad) override;
    void accum_grad(const Tensor& grad) override;
    
    bool retain_grad() const override;
    void set_retain_grad(bool retain_grad) override;
};

// Version counter for tracking tensor modifications
class TENSORPLAY_API VariableVersion {
private:
    uint32_t version_;
    bool enabled_;
    
public:
    VariableVersion() : version_(0), enabled_(true) {}
    
    explicit VariableVersion(bool enabled) : version_(0), enabled_(enabled) {}
    
    // Get current version
    uint32_t current_version() const { return version_; }
    
    // Check if version tracking is enabled
    bool is_enabled() const { return enabled_; }
    
    // Increment version
    void bump() {
        if (enabled_) {
            ++version_;
        }
    }
    
    // Reset version
    void reset() {
        if (enabled_) {
            version_ = 0;
        }
    }
    
    // Equality operator
    bool operator==(const VariableVersion& other) const {
        return version_ == other.version_ && enabled_ == other.enabled_;
    }
    
    bool operator!=(const VariableVersion& other) const {
        return !(*this == other);
    }
};

} // namespace tensorplay
