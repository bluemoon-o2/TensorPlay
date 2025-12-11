#include "tensorplay/core/Tensor.h"
#include "tensorplay/core/Dispatcher.h"
#include "tensorplay/core/Generator.h"
#include "tensorplay/core/Scalar.h"
#include <random>
#include <cstring>
#include <cmath>

namespace tensorplay {
namespace cpu {

// --- Factories ---

Tensor rand_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        int64_t n = t.numel();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        auto& gen = default_generator().engine();
        for (int64_t i = 0; i < n; ++i) {
            data[i] = dist(gen);
        }
    } else {
         throw std::runtime_error("rand() only supports Float32 for now");
    }
    return t;
}

Tensor zeros_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    size_t nbytes = t.numel() * t.itemsize();
    if (t.data_ptr()) {
        std::memset(t.data_ptr(), 0, nbytes);
    }
    return t;
}

Tensor ones_kernel(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        int64_t n = t.numel();
        for (int64_t i = 0; i < n; ++i) data[i] = 1.0f;
    } else if (dtype == DType::Int64) {
        int64_t* data = t.data_ptr<int64_t>();
        int64_t n = t.numel();
        for (int64_t i = 0; i < n; ++i) data[i] = 1;
    } else {
        // Fallback or throw
        // For now support float32/int64
    }
    return t;
}

Tensor full_kernel(const std::vector<int64_t>& size, double fill_value, DType dtype, Device device) {
    Tensor t(size, dtype, device);
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        int64_t n = t.numel();
        float val = static_cast<float>(fill_value);
        for (int64_t i = 0; i < n; ++i) data[i] = val;
    } else {
        // TODO: support other types
    }
    return t;
}

Tensor arange_kernel(double start, double end, double step, DType dtype, Device device) {
    int64_t len = static_cast<int64_t>(std::ceil((end - start) / step));
    if (len < 0) len = 0;
    
    Tensor t({len}, dtype, device);
    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        for (int64_t i = 0; i < len; ++i) data[i] = static_cast<float>(start + i * step);
    } else if (dtype == DType::Int64) {
        int64_t* data = t.data_ptr<int64_t>();
        for (int64_t i = 0; i < len; ++i) data[i] = static_cast<int64_t>(start + i * step);
    }
    return t;
}

TENSORPLAY_REGISTER_KERNEL(rand, CPU, rand_kernel)
TENSORPLAY_REGISTER_KERNEL(zeros, CPU, zeros_kernel)
TENSORPLAY_REGISTER_KERNEL(ones, CPU, ones_kernel)
TENSORPLAY_REGISTER_KERNEL(full, CPU, full_kernel)
TENSORPLAY_REGISTER_KERNEL(arange, CPU, arange_kernel)

} // namespace cpu
} // namespace tensorplay
