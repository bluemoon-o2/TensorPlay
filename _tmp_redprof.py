"""Reduction autotune diagnostic: tuner metric vs realistic metric per candidate."""
import os
import sys

os.environ.setdefault("TP_CACHE_DIR", "/tmp/tpcache_prof")
sys.path.insert(0, "/tmp/TensorPlay")

import torch  # noqa: E402
import tensorplay as tp  # noqa: E402
import tensorplay.compiler.codegen.triton as T  # noqa: E402
import tensorplay.compiler.runtime.stax_autotune as S  # noqa: E402
from tensorplay.compiler import compile as tp_compile  # noqa: E402

CASES = {
    "dims_sum3_1": lambda x: ((x * 2.0).sigmoid()).sum(dim=1) * 3.0 + 1.0,
    "full_sum_sigmoid": lambda x: (x.sigmoid()).sum(),
    "full_sum_16M": lambda x: x.sum(),
}

spy = {"calls": [], "failures": [], "mode": None, "cur_cfg": None,
       "sweep_kwargs": None}
_orig_compile = T._compile_program


def _compile_spy(*a, **k):
    cfg = k.get("fixed_config")
    spy["cur_cfg"] = cfg
    try:
        return _orig_compile(*a, **k)
    except Exception as exc:  # noqa: BLE001
        spy["failures"].append((cfg, repr(exc)[:200]))
        raise


_orig_bench = S.bench_launch


def _bench_spy(launch, args, **k):
    t = _orig_bench(launch, args, **k)
    spy["calls"].append((spy["cur_cfg"], t, launch))
    return t


_orig_dims = T._autotune_dims_program
_orig_split = T._autotune_split_program


def _dims_spy(role, program, constants, output_refs, example_inputs, **k):
    spy["mode"] = "dims"
    spy["sweep_kwargs"] = (program, constants, output_refs, example_inputs, k)
    return _orig_dims(role, program, constants, output_refs, example_inputs, **k)


def _split_spy(role, program, constants, output_refs, example_inputs, **k):
    spy["mode"] = "split"
    spy["sweep_kwargs"] = (program, constants, output_refs, example_inputs, k)
    return _orig_split(role, program, constants, output_refs, example_inputs, **k)


def _real_time(call, x, iters=50):
    for _ in range(5):
        call(x)
    tp.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        times = []
        for _ in range(iters):
            s = tp.cuda.Event(enable_timing=True)
            e = tp.cuda.Event(enable_timing=True)
            s.record()
            call(x)
            e.record()
            tp.cuda.synchronize()
            times.append(s.elapsed_time(e))
        best = min(best, min(times))
    return best


def _torch_time(fn, xt, iters=50):
    for _ in range(5):
        fn(xt)
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn(xt)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        best = min(best, min(times))
    return best


def main():
    T._compile_program = _compile_spy
    S.bench_launch = _bench_spy
    T._autotune_dims_program = _dims_spy
    T._autotune_split_program = _split_spy

    x = tp.rand(4096, 4096, device="cuda")
    xt = torch.rand(4096, 4096, device="cuda")

    for name, fn in CASES.items():
        print(f"\n================ {name} ================", flush=True)
        spy["calls"].clear()
        spy["failures"].clear()
        spy["mode"] = None
        compiled = tp_compile(fn, backend="stax")
        out = compiled(x)
        ref = fn(xt)
        try:
            ok = torch.allclose(
                torch.as_tensor(out.detach().cpu().numpy()),
                torch.as_tensor(ref.detach().cpu()),
            )
        except Exception as exc:  # noqa: BLE001
            ok = f"? {exc!r}"
        print(f"correct={ok}  tuner_path={spy.get('mode')}")
        for cfg, t, launch in spy["calls"]:
            rt = _real_time(lambda a, l=launch: l([a]), x)
            print(f"  cand {str(cfg):22s} tuner={t*1000:8.1f}us  realistic={rt*1000:8.1f}us")
        for cfg, err in spy["failures"]:
            print(f"  FAIL cand {cfg}: {err}")
        te = _real_time(lambda a: fn(a), x)
        print(f"  tp eager realistic={te*1000:8.1f}us")
        print(f"  torch eager realistic={_torch_time(fn, xt)*1000:8.1f}us")
        ct = torch.compile(fn)
        print(f"  torch inductor realistic={_torch_time(ct, xt)*1000:8.1f}us")


if __name__ == "__main__":
    main()
