#include "cpu/vec/vec.h"
using V = tensorplay::vec::Vectorized<float>;

typedef void (*tp_parallel_body_c)(void* ctx, long long b, long long e);
extern "C" void tp_parallel_for_c(long long begin, long long end, long long grain, tp_parallel_body_c body, void* ctx);

typedef struct TP_Ctx {
    const float* in0;
    float* out;
} TP_Ctx;

static void tp_body(void* ctxp, long long b, long long e) {
    const TP_Ctx* c = (const TP_Ctx*)ctxp;
    const float* __restrict__ in0 = c->in0;
    float* __restrict__ out = c->out;
    const long W = V::size();
    const V c0 = V(1.0f);
    long i = b;
    #pragma GCC ivdep
    for (; i + 4 * W <= e; i += 4 * W) {
        V x00 = V::loadu(in0 + i, W);
        V x01 = V::loadu(in0 + i + 1 * W, W);
        V x02 = V::loadu(in0 + i + 2 * W, W);
        V x03 = V::loadu(in0 + i + 3 * W, W);
        V t00 = x00.exp();
        V t10 = (t00 - V(1.0f));
        V t01 = x01.exp();
        V t11 = (t01 - V(1.0f));
        V t02 = x02.exp();
        V t12 = (t02 - V(1.0f));
        V t03 = x03.exp();
        V t13 = (t03 - V(1.0f));
        t10.store(out + i, W);
        t11.store(out + i + 1 * W, W);
        t12.store(out + i + 2 * W, W);
        t13.store(out + i + 3 * W, W);
    }
    #pragma GCC ivdep
    for (; i + W <= e; i += W) {
        V x0 = V::loadu(in0 + i, W);
        V t0 = x0.exp();
        V t1 = (t0 - V(1.0f));
        t1.store(out + i, W);
    }
    if (i < e) {
        const long count = e - i;
        V x0 = V::loadu(in0 + i, count);
        V t0 = x0.exp();
        V t1 = (t0 - V(1.0f));
        t1.store(out + i, count);
    }
}

extern "C" void tp_native_3b5f20514557fb08(long n, const float* __restrict__ in0, float* __restrict__ out) {
    TP_Ctx ctx{in0, out};
    tp_parallel_for_c(0, n, 32768, tp_body, &ctx);
}
