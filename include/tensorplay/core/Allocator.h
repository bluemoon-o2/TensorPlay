#pragma once

#include "tensorplay/core/DataPtr.h"
#include "tensorplay/core/Device.h"
#include "tensorplay/core/Macros.h"

namespace tensorplay {

class TENSORPLAY_API Allocator {
public:
    virtual ~Allocator() = default;
    virtual DataPtr allocate(size_t nbytes) const = 0;
};

// Get the allocator for a specific device
TENSORPLAY_API Allocator* getAllocator(DeviceType t);
TENSORPLAY_API Allocator* getCPUAllocator();

} // namespace tensorplay
