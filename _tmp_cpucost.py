"""CPU-side cost breakdown: frontend vs lowering vs raw launch vs eager."""
import os
import sys
import time

os.environ.setdefault("TP_CACHE_DIR", "/tmp/tpcache_prof")
sys.path.insert(0, "/tmp/TensorPlay")

import torch  # noqa: E402
import tensorplay as tp  # noqa: E402
import tensorplay.compiler.codegen.triton as T  # noqa: E402
from tensorplay.compiler import compile as tp_compile  # noqa: E402

_orig = T._compile_program
launches = []


def spy(*a, **k):
    if k.get("fixed_config") is not None:
        pass
    return _orig(*a, **k)


def cpu_cost(fn, x, n=500):
    for _ in range(50):
        fn(x)
    tp.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n):
            fn(x)
        best = min(best, (time.perf_counter() - t0) * 1e6 / n)
    tp.cuda.synchronize()
    return best


x = tp.rand(4096, 4096, device="cuda")

fns = {
    "dims": lambda t: ((t * 2.0).sigmoid()).sum(dim=1) * 3.0 + 1.0,
    "full": lambda t: t.sum(),
}
for name, fn in fns.items():
    compiled = tp_compile(fn, backend="stax")
    compiled(x)
    tp.cuda.synchronize()
    # eager tp CPU cost
    eager_cpu = cpu_cost(lambda t: fn(t), x)
    comp_cpu = cpu_cost(compiled, x)
    print(f"{name}: tp_eager CPU={eager_cpu:.1f}us/call  compiled CPU={comp_cpu:.1f}us/call")

# isolate kernel_launch CPU cost for the winning configs
import tensorplay.compiler.runtime.stax_autotune as S

x1 = tp.rand(4096, 4096, device="cuda")
fn = lambda t: t.sum()
compiled = tp_compile(fn, backend="stax")
compiled(x1)
print("done")
