"""Profile compiled() overhead on cc."""
import tensorplay as tp
from tensorplay.compiler.codegen.triton import _launch_memo

fn = tp.compile(lambda x: x * x + x, backend="stax")
x = tp.randn(4096, 4096, device="cuda")

iters = 500
start = tp.cuda.Event(enable_timing=True)
end = tp.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters):
    _ = fn(x)
end.record()
tp.cuda.synchronize()
t_compiled = start.elapsed_time(end) / iters
print(f"compiled() path: {t_compiled*1000:.3f} ms/iter")

keys = list(_launch_memo.keys())
print(f"memo keys count: {len(keys)}")
for k in keys[:5]:
    launch = _launch_memo[k]
    import inspect
    try:
        src = inspect.getsource(launch)
        has_lambda = "lambda" in src
        print(f"  key={k[:50]}... has_lambda={has_lambda} lines={len(src.splitlines())}")
    except Exception as e:
        print(f"  key={k[:50]}... error={e}")

if keys:
    launch = _launch_memo[keys[0]]
    start.record()
    for _ in range(iters):
        _ = launch([x])
    end.record()
    tp.cuda.synchronize()
    t_launch = start.elapsed_time(end) / iters
    print(f"kernel_launch direct: {t_launch*1000:.3f} ms/iter")
    print(f"overhead ratio compiled/direct: {t_compiled/t_launch:.2f}x")
