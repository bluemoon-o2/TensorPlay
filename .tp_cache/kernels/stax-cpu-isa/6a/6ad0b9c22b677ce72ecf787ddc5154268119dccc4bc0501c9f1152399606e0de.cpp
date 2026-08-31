
#include "cpu/vec/vec.h"
using V = tensorplay::vec::Vectorized<float>;

alignas(64) float tp_isa_input[64] = {0};

extern "C" int tp_isa_probe_entry() {
    V a = V::loadu(tp_isa_input, V::size());
    V b = tensorplay::vec::maximum(a, V(0.0f)) + a.exp();
    V m = (a > V(0.0f));
    V c = V::blendv(a, b, m);
    __attribute__((aligned(64))) float out[64];
    c.store(out, V::size());
    return (out[0] == out[0]) ? 0 : 1;
}
