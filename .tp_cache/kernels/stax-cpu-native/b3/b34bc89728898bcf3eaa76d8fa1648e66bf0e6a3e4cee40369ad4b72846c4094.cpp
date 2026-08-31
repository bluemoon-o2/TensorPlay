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
    const V c1 = V(1.0f);
    const V c2 = V(0.5f);
    long i = b;
    #pragma GCC ivdep
    for (; i + 4 * W <= e; i += 4 * W) {
        V x00 = V::loadu(in0 + i, W);
        V x01 = V::loadu(in0 + i + 1 * W, W);
        V x02 = V::loadu(in0 + i + 2 * W, W);
        V x03 = V::loadu(in0 + i + 3 * W, W);
        V t00 = (x00 + V(1.0f));
        V t10 = (V(1.0f) - x00);
        V t20 = (t00 / t10);
        V t30 = t20.log();
        V t40 = (t30 * V(0.5f));
        V t01 = (x01 + V(1.0f));
        V t11 = (V(1.0f) - x01);
        V t21 = (t01 / t11);
        V t31 = t21.log();
        V t41 = (t31 * V(0.5f));
        V t02 = (x02 + V(1.0f));
        V t12 = (V(1.0f) - x02);
        V t22 = (t02 / t12);
        V t32 = t22.log();
        V t42 = (t32 * V(0.5f));
        V t03 = (x03 + V(1.0f));
        V t13 = (V(1.0f) - x03);
        V t23 = (t03 / t13);
        V t33 = t23.log();
        V t43 = (t33 * V(0.5f));
        t40.store(out + i, W);
        t41.store(out + i + 1 * W, W);
        t42.store(out + i + 2 * W, W);
        t43.store(out + i + 3 * W, W);
    }
    #pragma GCC ivdep
    for (; i + W <= e; i += W) {
        V x0 = V::loadu(in0 + i, W);
        V t0 = (x0 + V(1.0f));
        V t1 = (V(1.0f) - x0);
        V t2 = (t0 / t1);
        V t3 = t2.log();
        V t4 = (t3 * V(0.5f));
        t4.store(out + i, W);
    }
    if (i < e) {
        const long count = e - i;
        V x0 = V::loadu(in0 + i, count);
        V t0 = (x0 + V(1.0f));
        V t1 = (V(1.0f) - x0);
        V t2 = (t0 / t1);
        V t3 = t2.log();
        V t4 = (t3 * V(0.5f));
        t4.store(out + i, count);
    }
}

extern "C" void tp_native_b7b473738fe8bd1d(long n, const float* __restrict__ in0, float* __restrict__ out) {
    TP_Ctx ctx{in0, out};
    tp_parallel_for_c(0, n, 32768, tp_body, &ctx);
}
