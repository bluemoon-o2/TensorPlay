#pragma once
#include <random>
#include <cstdint>

namespace tensorplay {

std::mt19937& get_generator();
inline void manual_seed(uint32_t seed) {
    get_generator().seed(seed);
}

} // namespace tensorplay
