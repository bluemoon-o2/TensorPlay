"""Compile-time autotuner for Stax Triton kernels (L5-M2).

Modeled on ``torch/_inductor/runtime/triton_heuristics.py`` (CachingAutotuner):
instead of emitting ``@triton.autotune`` — which benchmarks candidate configs
at every new runtime key with per-launch overhead and keeps no persistent
record — we benchmark the candidates once at compile time, pick the winner,
and emit a fixed-config kernel.  Decisions are stored in the kernel codecache
keyed by ``(program digest, xnumel bucket, device)``, so later processes skip
benchmarking entirely, mirroring Inductor's persistent autotune cache.

The benchmarking itself uses :class:`tensorplay.cuda.Event` timings around a
warmup + timed-iteration loop, equivalent to Inductor's ``do_bench``.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

# (XBLOCK, num_warps) candidates.  Beyond the classic Inductor table this
# probes the 8-wide geometry (2048 elements / 256 threads = two vectorized
# 16B accesses per thread) and the 16-wide extreme — transcendental-heavy
# pointwise chains gain ILP from fewer, busier threads even at halved
# occupancy; the tuner discards them wherever spills dominate.
CANDIDATE_CONFIGS: Tuple[Tuple[int, int], ...] = (
    (128, 4),
    (256, 4),
    (512, 4),
    (512, 8),
    (1024, 4),
    (1024, 8),
    (2048, 8),
    (2048, 4),
)

# Salt folded into every persisted decision (pointwise ``decision_key`` and
# the dims/split reduction keys in ``codegen.triton``).  ``program_digest``
# hashes only the kernel program — it cannot see emitter or candidate-table
# changes — so without this bump a decision cached by an older compiler
# generation short-circuits benchmarking forever and pins yesterday's
# geometry.
TUNING_VERSION = "t8-redsplit"

_DISABLE_ENV = "TP_DISABLE_STAX_AUTOTUNE"


def disabled() -> bool:
    """True when autotuning is switched off via environment."""

    return os.environ.get(_DISABLE_ENV, "") not in ("", "0")


def program_digest(program: Sequence[int], constants: Sequence[float],
                   output_refs: Sequence[int]) -> str:
    """Content hash of a postfix program, independent of launch shapes."""

    h = hashlib.sha256()
    h.update(repr((tuple(program), tuple(constants), tuple(output_refs))).encode())
    return h.hexdigest()[:16]


def xnumel_bucket(xnumel: int) -> int:
    """Power-of-two bucket for an element count (min = smallest XBLOCK).

    Decisions generalize within a bucket because XBLOCK only changes the
    grid/block geometry; any xnumel in the same bucket sees the same ranking
    in practice (Inductor keys its autotune cache similarly by shape).
    """

    if xnumel <= 0:
        return CANDIDATE_CONFIGS[0][0]
    bucket = 1
    while bucket < xnumel:
        bucket *= 2
    return max(bucket, CANDIDATE_CONFIGS[0][0])


def _decision_cache():
    from ..codecache import default_cache

    return default_cache("triton-autotune")


def decision_key(digest: str, bucket: int, device: str) -> str:
    h = hashlib.sha256(
        f"{TUNING_VERSION}|{digest}|{bucket}|{device}".encode()
    )
    return h.hexdigest()[:24]


def load_decision(digest: str, bucket: int,
                  device: str) -> Optional[Tuple[int, int]]:
    """Return ``(xblock, num_warps)`` previously chosen for this key."""

    payload = _decision_cache().load(decision_key(digest, bucket, device),
                                     ext="json")
    if payload is None:
        return None
    try:
        record = json.loads(payload.decode())
        config = (int(record["xblock"]), int(record["warps"]))
        if config not in CANDIDATE_CONFIGS:
            return None
        return config
    except (ValueError, KeyError, TypeError):
        return None


def store_decision(digest: str, bucket: int, device: str,
                   config: Tuple[int, int]) -> None:
    payload = json.dumps({"xblock": config[0], "warps": config[1]}).encode()
    _decision_cache().store(decision_key(digest, bucket, device), payload,
                            ext="json")


def bench_launch(launch: Callable[[list], Any], args: list,
                 *, warmup_ms: float = 3.0, iters: int = 20) -> float:
    """Minimum per-iteration latency (ms) over warmup + timed launches.

    Mirrors the benchmark harness (``_time_tp``): CUDA events around EACH
    launch with a device sync, best-of.  Two Inductor-benchmarker lessons
    (``torch/_inductor/runtime/benchmarking.py::benchmark_gpu``):

    * A pipelined average measures the Python launch floor (~25-35us) once
      kernels drop below it, flattening the ranking; per-iteration latency
      is what callers actually pay.
    * Candidates are JIT-compiled immediately before their window, so the
      GPU idles into a down-clock and a short timed loop never ramps back
      (Inductor runs ``memory_warmup_iters=100`` busy iterations first).
      We warm up by running the kernel itself for ``warmup_ms`` wall time
      before recording, which also settles L2 into the steady state the
      harness measures.
    """

    import time

    import tensorplay as tp

    best = float("inf")
    for _ in range(3):
        deadline = time.perf_counter() + warmup_ms / 1000.0
        while time.perf_counter() < deadline:
            launch(args)
        tp.cuda.synchronize()
        for _ in range(iters):
            start = tp.cuda.Event(enable_timing=True)
            end = tp.cuda.Event(enable_timing=True)
            start.record()
            launch(args)
            end.record()
            tp.cuda.synchronize()
            best = min(best, start.elapsed_time(end))
    return best


def pick_config(
    digest: str,
    xnumel: int,
    device_key: str,
    build_launch: Callable[[Tuple[int, int]], Any],
    sample_args: list,
    *,
    bench_fn: Optional[Callable[[Any, list], float]] = None,
) -> Tuple[Tuple[int, int], Any]:
    """Benchmark candidates and return ``(config, launch_callable)``.

    ``build_launch(config)`` must compile and return a launch callable for a
    fixed-config kernel; it may raise, which disqualifies that candidate.
    A cached decision short-circuits benchmarking entirely (only one compile
    runs).  ``bench_fn`` is injectable for tests.
    """

    if bench_fn is None:
        bench_fn = bench_launch
    bucket = xnumel_bucket(xnumel)

    cached = load_decision(digest, bucket, device_key)
    if cached is not None:
        return cached, build_launch(cached)

    best_config: Optional[Tuple[int, int]] = None
    best_launch: Any = None
    best_time = float("inf")
    timings: Dict[Tuple[int, int], float] = {}
    for config in CANDIDATE_CONFIGS:
        try:
            launch = build_launch(config)
            timing = bench_fn(launch, sample_args)
        except Exception:  # noqa: BLE001 - candidate disqualification
            continue
        timings[config] = timing
        if timing < best_time:
            best_config, best_launch, best_time = config, launch, timing
    if best_config is None:
        raise RuntimeError(
            "stax autotune: all candidate configs failed to compile or run"
        )
    store_decision(digest, bucket, device_key, best_config)
    return best_config, best_launch
