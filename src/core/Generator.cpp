#include "tensorplay/core/Generator.h"

namespace tensorplay {

std::mt19937& get_generator() {
    static std::mt19937 generator(42);
    return generator;
}

} // namespace tensorplay
