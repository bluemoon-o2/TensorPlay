"""Diagnose per-call CPU overhead of the stax compiled path vs raw launch."""
import os
import sys
import cProfile
import pstats
import io

os.environ.setdefault("TP_CACHE_DIR", "/tmp/tpcache_prof")
sys.path.insert(0, "/tmp/TensorPlay")

import torch  # noqa: E402
import tensorplay as tp  # noqa: E402
from tensorplay.compiler import compile as tp_compile  # noqa: E402

FNS = {
    "dims_sum3_1": lambda x: ((x * 2.0).sigmoid()).sum(dim=1) * 3.0 + 1.0,
    "full_sum_16M": lambda x: x.sum(),
}


def wall_per_call(fn, x, n=300):
    import time

    for _ in range(20):
        fn(x)
    tp.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            fn(x)
        t1 = time.perf_counter()
        tp.cuda.synchronize()
        best = min(best, (t1 - t0) * 1e6 / n)
    return best


def gpu_per_call(fn, x, n=100):
    for _ in range(10):
        fn(x)
    tp.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        s = tp.cuda.Event(enable_timing=True)
        e = tp.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn(x)
        e.record()
        tp.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / n)
    return best


def gpu_per_iter_sync(fn, x, n=100):
    for _ in range(10):
        fn(x)
    tp.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        times = []
        for _ in range(n):
            s = tp.cuda.Event(enable_timing=True)
            e = tp.cuda.Event(enable_timing=True)
            s.record()
            fn(x)
            e.record()
            tp.cuda.synchronize()
            times.append(s.elapsed_time(e))
        best = min(best, min(times))
    return best


def main():
    x = tp.rand(4096, 4096, device="cuda")
    xt = torch.from_numpy(x.detach().cpu().numpy()).cuda()

    for name, fn in FNS.items():
        print(f"\n================ {name} ================", flush=True)
        compiled = tp_compile(fn, backend="stax")
        out = compiled(x)
        ref = fn(xt)
        diff = (out.detach().cpu().numpy() - ref.detach().cpu().numpy())
        rel = abs(diff).max() / max(abs(ref.detach().cpu().numpy()).max(), 1e-6)
        print(f"max_abs_diff={abs(diff).max():.3e} rel={rel:.3e}")

        # dig out the lowering + raw launch
        inner = compiled
        for attr in ("last_compiled_fn",):
            pass
        # compiled is the frontend closure; call it, then find the lowering
        # via the specialization cache is private — instead recompile path:
        print(f"compiled wall/cpu-bound us/call: {wall_per_call(compiled, x):.1f}")
        print(f"compiled gpu pipelined us/call : {gpu_per_call(compiled, x):.1f}")
        print(f"compiled gpu per-iter-sync us  : {gpu_per_iter_sync(compiled, x):.1f}")

        pr = cProfile.Profile()
        pr.enable()
        for _ in range(2000):
            compiled(x)
        pr.disable()
        tp.cuda.synchronize()
        buf = io.StringIO()
        ps = pstats.Stats(pr, stream=buf)
        ps.sort_stats("cumulative")
        ps.print_stats(25)
        print(buf.getvalue()[:4000])


if __name__ == "__main__":
    main()
