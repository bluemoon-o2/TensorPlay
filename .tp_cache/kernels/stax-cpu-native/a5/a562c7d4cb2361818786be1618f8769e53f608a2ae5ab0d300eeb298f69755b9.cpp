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
        V t00 = x00.log();
        V t10 = (V(1.0f) - x00);
        V t20 = t10.log();
        V t30 = (t00 - t20);
        V t01 = x01.log();
        V t11 = (V(1.0f) - x01);
        V t21 = t11.log();
        V t31 = (t01 - t21);
        V t02 = x02.log();
        V t12 = (V(1.0f) - x02);
        V t22 = t12.log();
        V t32 = (t02 - t22);
        V t03 = x03.log();
        V t13 = (V(1.0f) - x03);
        V t23 = t13.log();
        V t33 = (t03 - t23);
        t30.store(out + i, W);
        t31.store(out + i + 1 * W, W);
        t32.store(out + i + 2 * W, W);
        t33.store(out + i + 3 * W, W);
    }
    #pragma GCC ivdep
    for (; i + W <= e; i += W) {
        V x0 = V::loadu(in0 + i, W);
        V t0 = x0.log();
        V t1 = (V(1.0f) - x0);
        V t2 = t1.log();
        V t3 = (t0 - t2);
        t3.store(out + i, W);
    }
    if (i < e) {
        const long count = e - i;
        V x0 = V::loadu(in0 + i, count);
        V t0 = x0.log();
        V t1 = (V(1.0f) - x0);
        V t2 = t1.log();
        V t3 = (t0 - t2);
        t3.store(out + i, count);
    }
}

extern "C" void tp_native_89580bf3e0a3216b(long n, const float* __restrict__ in0, float* __restrict__ out) {
    TP_Ctx ctx{in0, out};
    tp_parallel_for_c(0, n, 32768, tp_body, &ctx);
}
