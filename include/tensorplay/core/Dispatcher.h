#pragma once

#include <unordered_map>
#include <string>
#include <mutex>
#include <stdexcept>
#include "DispatchKey.h"
#include "tensorplay/core/Device.h"

namespace tensorplay {

// Helper to determine dispatch key
inline DispatchKey computeDispatchKey(const Device& device) {
    if (device.is_cuda()) return DispatchKey::CUDA;
    return DispatchKey::CPU;
}

// Generic kernel function pointer type (type-erased)
using KernelFunction = void*;

class Dispatcher {
public:
    static Dispatcher& singleton();

    // Register a kernel for a specific operator and dispatch key
    void registerKernel(const std::string& op_name, DispatchKey key, KernelFunction kernel);

    // Get the kernel for a specific operator and dispatch key
    KernelFunction getKernel(const std::string& op_name, DispatchKey key);

private:
    Dispatcher() = default;
    
    struct OpDispatchTable {
        std::unordered_map<DispatchKey, KernelFunction> kernels;
    };

    std::unordered_map<std::string, OpDispatchTable> operators_;
    std::mutex mutex_;
};

// Helper for type-safe dispatch
template<typename Return, typename... Args>
class DispatchStub {
public:
    static Return call(const std::string& op_name, DispatchKey key, Args... args) {
        auto kernel_void = Dispatcher::singleton().getKernel(op_name, key);
        if (!kernel_void) {
            throw std::runtime_error("Kernel not found for op: " + op_name + " on backend: " + toString(key));
        }
        using FuncType = Return(*)(Args...);
        auto kernel = reinterpret_cast<FuncType>(kernel_void);
        return kernel(std::forward<Args>(args)...);
    }
};

// Macro for registration
#define TENSORPLAY_REGISTER_KERNEL(OP_NAME, KEY, FUNC) \
    static struct Register##OP_NAME##KEY { \
        Register##OP_NAME##KEY() { \
            ::tensorplay::Dispatcher::singleton().registerKernel(#OP_NAME, ::tensorplay::DispatchKey::KEY, (::tensorplay::KernelFunction)FUNC); \
        } \
    } register_##OP_NAME##KEY;

} // namespace tensorplay
