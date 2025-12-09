#include "tensorplay/core/Allocator.h"
#include "tensorplay/core/Exception.h"
#include <memory>
#include <mutex>
#include <cstdlib>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace tensorplay {

namespace {

// Portable aligned allocation
void* alloc_aligned(size_t nbytes, size_t alignment = 64) {
    if (nbytes == 0) return nullptr;
#ifdef _WIN32
    return _aligned_malloc(nbytes, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, nbytes) != 0) return nullptr;
    return ptr;
#endif
}

void free_aligned(void* data) {
    if (!data) return;
#ifdef _WIN32
    _aligned_free(data);
#else
    free(data);
#endif
}

// Default CPU Allocator implementation
class CPUAllocator : public Allocator {
public:
    DataPtr allocate(size_t nbytes) const override {
        void* data = alloc_aligned(nbytes);
        // Create DataPtr with free_aligned deleter
        return DataPtr(data, free_aligned, Device(DeviceType::CPU));
    }
};

static CPUAllocator g_cpu_allocator;

} // namespace

Allocator* getCPUAllocator() {
    return &g_cpu_allocator;
}

Allocator* getAllocator(DeviceType t) {
    if (t == DeviceType::CPU) {
        return getCPUAllocator();
    }
    // Future expansion: CUDA allocator
    // if (t == DeviceType::CUDA) return getCUDAAllocator();
    
    TP_THROW(NotImplementedError, "Allocator not implemented for this device type");
}

} // namespace tensorplay
