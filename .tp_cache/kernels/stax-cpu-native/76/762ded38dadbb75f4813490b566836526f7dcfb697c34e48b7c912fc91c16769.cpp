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
    const V c0 = V(0.5f);
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
        V t00 = x00.gt(x10);
        V t10 = (x00 * V(0.5f));
        V t20 = (-x10);
        V t30 = t10;
        V t40 = V::blendv(t20, t10, (t00 > V(0.0f)));
        V t01 = x01.gt(x11);
        V t11 = (x01 * V(0.5f));
        V t21 = (-x11);
        V t31 = t11;
        V t41 = V::blendv(t21, t11, (t01 > V(0.0f)));
        V t02 = x02.gt(x12);
        V t12 = (x02 * V(0.5f));
        V t22 = (-x12);
        V t32 = t12;
        V t42 = V::blendv(t22, t12, (t02 > V(0.0f)));
        V t03 = x03.gt(x13);
        V t13 = (x03 * V(0.5f));
        V t23 = (-x13);
        V t33 = t13;
        V t43 = V::blendv(t23, t13, (t03 > V(0.0f)));
        t40.store(out + i, W);
        t41.store(out + i + 1 * W, W);
        t42.store(out + i + 2 * W, W);
        t43.store(out + i + 3 * W, W);
    }
    #pragma GCC ivdep
    for (; i + W <= e; i += W) {
        V x0 = V::loadu(in0 + i, W);
        V x1 = V::loadu(in1 + i, W);
        V t0 = x0.gt(x1);
        V t1 = (x0 * V(0.5f));
        V t2 = (-x1);
        V t3 = t1;
        V t4 = V::blendv(t2, t1, (t0 > V(0.0f)));
        t4.store(out + i, W);
    }
    if (i < e) {
        const long count = e - i;
        V x0 = V::loadu(in0 + i, count);
        V x1 = V::loadu(in1 + i, count);
        V t0 = x0.gt(x1);
        V t1 = (x0 * V(0.5f));
        V t2 = (-x1);
        V t3 = t1;
        V t4 = V::blendv(t2, t1, (t0 > V(0.0f)));
        t4.store(out + i, count);
    }
}

extern "C" void tp_native_a4d9af596c43ccd3(long n, const float* __restrict__ in0, const float* __restrict__ in1, float* __restrict__ out) {
    TP_Ctx ctx{in0, in1, out};
    tp_parallel_for_c(0, n, 32768, tp_body, &ctx);
}
