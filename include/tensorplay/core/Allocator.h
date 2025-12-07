#pragma once

#include "tensorplay/core/DataPtr.h"
#include "tensorplay/core/Device.h"

namespace tensorplay {

class Allocator {
public:
    virtual ~Allocator() = default;
    virtual DataPtr allocate(size_t nbytes) const = 0;
};

// Get the allocator for a specific device
Allocator* getAllocator(DeviceType t);
Allocator* getCPUAllocator();

} // namespace tensorplay
