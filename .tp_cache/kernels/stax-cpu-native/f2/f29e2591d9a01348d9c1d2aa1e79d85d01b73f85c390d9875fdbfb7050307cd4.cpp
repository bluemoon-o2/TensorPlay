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
    const V c0 = V(2.0f);
    const V c1 = V(3.0f);
    long i = b;
    #pragma GCC ivdep
    for (; i + 4 * W <= e; i += 4 * W) {
        V x00 = V::loadu(in0 + i, W);
        V x01 = V::loadu(in0 + i + 1 * W, W);
        V x02 = V::loadu(in0 + i + 2 * W, W);
        V x03 = V::loadu(in0 + i + 3 * W, W);
        V t00 = (x00 * V(2.0f));
        V t10 = (t00 + V(3.0f));
        V t20 = tensorplay::vec::maximum(t10, V(0.0f));
        V t30 = t20.sin();
        V t01 = (x01 * V(2.0f));
        V t11 = (t01 + V(3.0f));
        V t21 = tensorplay::vec::maximum(t11, V(0.0f));
        V t31 = t21.sin();
        V t02 = (x02 * V(2.0f));
        V t12 = (t02 + V(3.0f));
        V t22 = tensorplay::vec::maximum(t12, V(0.0f));
        V t32 = t22.sin();
        V t03 = (x03 * V(2.0f));
        V t13 = (t03 + V(3.0f));
        V t23 = tensorplay::vec::maximum(t13, V(0.0f));
        V t33 = t23.sin();
        t30.store(out + i, W);
        t31.store(out + i + 1 * W, W);
        t32.store(out + i + 2 * W, W);
        t33.store(out + i + 3 * W, W);
    }
    #pragma GCC ivdep
    for (; i + W <= e; i += W) {
        V x0 = V::loadu(in0 + i, W);
        V t0 = (x0 * V(2.0f));
        V t1 = (t0 + V(3.0f));
        V t2 = tensorplay::vec::maximum(t1, V(0.0f));
        V t3 = t2.sin();
        t3.store(out + i, W);
    }
    if (i < e) {
        const long count = e - i;
        V x0 = V::loadu(in0 + i, count);
        V t0 = (x0 * V(2.0f));
        V t1 = (t0 + V(3.0f));
        V t2 = tensorplay::vec::maximum(t1, V(0.0f));
        V t3 = t2.sin();
        t3.store(out + i, count);
    }
}

extern "C" void tp_native_72f61ab2e7fcd476(long n, const float* __restrict__ in0, float* __restrict__ out) {
    TP_Ctx ctx{in0, out};
    if (n < 4096LL) {
        tp_body(&ctx, 0, n);
    } else {
        tp_parallel_for_c(0, n, 512LL, tp_body, &ctx);
    }
}
