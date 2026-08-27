"""Isolate cache-policy effect on repeated 16M full-sum (L2 residency)."""
import sys
sys.path.insert(0, "/tmp/TensorPlay")
import torch
import triton
import triton.language as tl

print("L2 bytes:", torch.cuda.get_device_properties(0).L2_cache_size if hasattr(torch.cuda.get_device_properties(0), "L2_cache_size") else "?")


@triton.jit
def k_cg(ptr, ws, n, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * XBLOCK + tl.arange(0, XBLOCK)
    v = tl.load(ptr + offs, cache_modifier=".cg")
    tl.store(ws + pid, tl.sum(v, axis=0))


@triton.jit
def k_ca(ptr, ws, n, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * XBLOCK + tl.arange(0, XBLOCK)
    v = tl.load(ptr + offs)
    tl.store(ws + pid, tl.sum(v, axis=0))


@triton.jit
def k_ef(ptr, ws, n, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * XBLOCK + tl.arange(0, XBLOCK)
    v = tl.load(ptr + offs, eviction_policy="evict_first")
    tl.store(ws + pid, tl.sum(v, axis=0))


@triton.jit
def k_fin(ws, out, wsn, FB: tl.constexpr):
    acc = tl.zeros([FB], dtype=tl.float32)
    offs = tl.arange(0, FB)
    for b in range(0, wsn, FB):
        i = b + offs
        acc += tl.where(i < wsn, tl.load(ws + i, mask=i < wsn, other=0.0), 0.0)
    tl.store(out, tl.sum(acc, axis=0))


x = torch.rand(4096 * 4096, device="cuda")
out = torch.zeros((), device="cuda")
n = x.numel()


def run(kern, xb, wp):
    wsn = n // xb
    ws = torch.empty(wsn, device="cuda")
    kern[(wsn,)](x, ws, n, XBLOCK=xb, num_warps=wp)
    k_fin[(1,)](ws, out, wsn, FB=1024, num_warps=4)
    return ws


def timeit(kern, xb, wp, iters=200):
    ws = run(kern, xb, wp)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        wsn = n // xb
        s.record()
        for _ in range(iters):
            kern[(wsn,)](x, ws, n, XBLOCK=xb, num_warps=wp)
            k_fin[(1,)](ws, out, wsn, FB=1024, num_warps=4)
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / iters)
    return best


for name, kern in (("cg", k_cg), ("ca", k_ca), ("evict_first", k_ef)):
    for xb, wp in ((2048, 8), (4096, 8), (1024, 4)):
        print(f"{name:12s} XBLOCK={xb:5d} w={wp}: {timeit(kern, xb, wp):7.1f}us")

# eager reference
best = float("inf")
for _ in range(3):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(200):
        x.sum()
    s.record()
    for _ in range(200):
        x.sum()
    e.record()
    torch.cuda.synchronize()
    best = min(best, s.elapsed_time(e) * 1000 / 200)
print(f"torch eager sum: {best:7.1f}us")
