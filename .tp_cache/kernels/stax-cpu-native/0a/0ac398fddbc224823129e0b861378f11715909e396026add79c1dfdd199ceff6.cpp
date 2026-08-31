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
    const V c0 = V(1.0f);
    const V c1 = V(1.0f);
    const V c2 = V(2.0f);
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
        V t00 = x00.abs();
        V t10 = (-x10);
        V t20 = t10.exp();
        V t30 = (t20 + V(1.0f));
        V t40 = (V(1.0f) / t30);
        V t50 = (t00 + t40);
        V t60 = t50.tanh();
        V t70 = x00.cos();
        V t80 = (t70 + V(2.0f));
        V t90 = (t60 / t80);
        V t100 = tensorplay::vec::maximum(t90, V(0.0f));
        V t01 = x01.abs();
        V t11 = (-x11);
        V t21 = t11.exp();
        V t31 = (t21 + V(1.0f));
        V t41 = (V(1.0f) / t31);
        V t51 = (t01 + t41);
        V t61 = t51.tanh();
        V t71 = x01.cos();
        V t81 = (t71 + V(2.0f));
        V t91 = (t61 / t81);
        V t101 = tensorplay::vec::maximum(t91, V(0.0f));
        V t02 = x02.abs();
        V t12 = (-x12);
        V t22 = t12.exp();
        V t32 = (t22 + V(1.0f));
        V t42 = (V(1.0f) / t32);
        V t52 = (t02 + t42);
        V t62 = t52.tanh();
        V t72 = x02.cos();
        V t82 = (t72 + V(2.0f));
        V t92 = (t62 / t82);
        V t102 = tensorplay::vec::maximum(t92, V(0.0f));
        V t03 = x03.abs();
        V t13 = (-x13);
        V t23 = t13.exp();
        V t33 = (t23 + V(1.0f));
        V t43 = (V(1.0f) / t33);
        V t53 = (t03 + t43);
        V t63 = t53.tanh();
        V t73 = x03.cos();
        V t83 = (t73 + V(2.0f));
        V t93 = (t63 / t83);
        V t103 = tensorplay::vec::maximum(t93, V(0.0f));
        t100.store(out + i, W);
        t101.store(out + i + 1 * W, W);
        t102.store(out + i + 2 * W, W);
        t103.store(out + i + 3 * W, W);
    }
    #pragma GCC ivdep
    for (; i + W <= e; i += W) {
        V x0 = V::loadu(in0 + i, W);
        V x1 = V::loadu(in1 + i, W);
        V t0 = x0.abs();
        V t1 = (-x1);
        V t2 = t1.exp();
        V t3 = (t2 + V(1.0f));
        V t4 = (V(1.0f) / t3);
        V t5 = (t0 + t4);
        V t6 = t5.tanh();
        V t7 = x0.cos();
        V t8 = (t7 + V(2.0f));
        V t9 = (t6 / t8);
        V t10 = tensorplay::vec::maximum(t9, V(0.0f));
        t10.store(out + i, W);
    }
    if (i < e) {
        const long count = e - i;
        V x0 = V::loadu(in0 + i, count);
        V x1 = V::loadu(in1 + i, count);
        V t0 = x0.abs();
        V t1 = (-x1);
        V t2 = t1.exp();
        V t3 = (t2 + V(1.0f));
        V t4 = (V(1.0f) / t3);
        V t5 = (t0 + t4);
        V t6 = t5.tanh();
        V t7 = x0.cos();
        V t8 = (t7 + V(2.0f));
        V t9 = (t6 / t8);
        V t10 = tensorplay::vec::maximum(t9, V(0.0f));
        t10.store(out + i, count);
    }
}

extern "C" void tp_native_bdd316089295260d(long n, const float* __restrict__ in0, const float* __restrict__ in1, float* __restrict__ out) {
    TP_Ctx ctx{in0, in1, out};
    tp_parallel_for_c(0, n, 32768, tp_body, &ctx);
}
