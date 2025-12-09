#pragma once
#include <random>
#include <cstdint>
#include <memory>
#include "tensorplay/core/Macros.h"

namespace tensorplay {

class TENSORPLAY_API Generator {
public:
    Generator(uint64_t seed_val = 0);
    
    // Sets the seed and returns it
    uint64_t manual_seed(uint64_t seed);
    
    // Returns the current seed
    uint64_t seed();
    
    // Returns the initial seed
    uint64_t initial_seed() const;
    
    std::mt19937& engine();

private:
    std::mt19937 mt_;
    uint64_t initial_seed_;
};

TENSORPLAY_API Generator& default_generator();

TENSORPLAY_API void manual_seed(uint64_t seed);

} // namespace tensorplay
