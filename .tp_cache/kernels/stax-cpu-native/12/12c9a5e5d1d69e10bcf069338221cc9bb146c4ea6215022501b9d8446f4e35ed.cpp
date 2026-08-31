#include "cpu/vec/vec.h"
using V = tensorplay::vec::Vectorized<float>;

typedef void (*tp_parallel_body_c)(void* ctx, long long b, long long e);
extern "C" void tp_parallel_for_c(long long begin, long long end, long long grain, tp_parallel_body_c body, void* ctx);

typedef struct TP_Ctx {
    const float* in0;
    const float* in1;
    float* out;
} TP_Ctx;

static void tp_body(void* ctxp, long long b, long long e) {
    const TP_Ctx* c = (const TP_Ctx*)ctxp;
    const float* __restrict__ in0 = c->in0;
    const float* __restrict__ in1 = c->in1;
    float* __restrict__ out = c->out;
    const long W = V::size();
    const V c0 = V(0.0f);
    const V c1 = V(1.0f);
    long i = b;
    #pragma GCC ivdep
    for (; i + 4 * W <= e; i += 4 * W) {
        V x00 = V::loadu(in0 + i, W);
        V x10 = V::loadu(in1 + i, W);
        V x01 = V::loadu(in0 + i + 1 * W, W);
        V x11 = V::loadu(in1 + i + 1 * W, W);
        V x02 = V::loadu(in0 + i + 2 * W, W);
        V x12 = V::loadu(in1 + i + 2 * W, W);
        V x03 = V::loadu(in0 + i + 3 * W, W);
        V x13 = V::loadu(in1 + i + 3 * W, W);
        V t00 = tensorplay::vec::maximum(x00, V(0.0f));
        V t10 = tensorplay::vec::minimum(x10, V(1.0f));
        V t20 = (t00 + t10);
        V t01 = tensorplay::vec::maximum(x01, V(0.0f));
        V t11 = tensorplay::vec::minimum(x11, V(1.0f));
        V t21 = (t01 + t11);
        V t02 = tensorplay::vec::maximum(x02, V(0.0f));
        V t12 = tensorplay::vec::minimum(x12, V(1.0f));
        V t22 = (t02 + t12);
        V t03 = tensorplay::vec::maximum(x03, V(0.0f));
        V t13 = tensorplay::vec::minimum(x13, V(1.0f));
        V t23 = (t03 + t13);
        t20.store(out + i, W);
        t21.store(out + i + 1 * W, W);
        t22.store(out + i + 2 * W, W);
        t23.store(out + i + 3 * W, W);
    }
    #pragma GCC ivdep
    for (; i + W <= e; i += W) {
        V x0 = V::loadu(in0 + i, W);
        V x1 = V::loadu(in1 + i, W);
        V t0 = tensorplay::vec::maximum(x0, V(0.0f));
        V t1 = tensorplay::vec::minimum(x1, V(1.0f));
        V t2 = (t0 + t1);
        t2.store(out + i, W);
    }
    if (i < e) {
        const long count = e - i;
        V x0 = V::loadu(in0 + i, count);
        V x1 = V::loadu(in1 + i, count);
        V t0 = tensorplay::vec::maximum(x0, V(0.0f));
        V t1 = tensorplay::vec::minimum(x1, V(1.0f));
        V t2 = (t0 + t1);
        t2.store(out + i, count);
    }
}

extern "C" void tp_native_8fdb74cf546cb7ac(long n, const float* __restrict__ in0, const float* __restrict__ in1, float* __restrict__ out) {
    TP_Ctx ctx{in0, in1, out};
    if (n < 4096LL) {
        tp_body(&ctx, 0, n);
    } else {
        tp_parallel_for_c(0, n, 512LL, tp_body, &ctx);
    }
}
