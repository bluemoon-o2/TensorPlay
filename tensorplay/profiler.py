"""Native op-level profiler, mirroring :mod:`torch.profiler` (subset).

Records every dispatched operator exactly once at the below-autograd
redispatch funnel -- composite inner calls show up individually, matching
upstream's CompositeImplicitAutograd behavior.  The autograd engine emits a
``__backward__`` span around every backward()/grad() execution.

Typical use::

    with tp.profiler.profile() as prof:
        y = model(x)
        loss = criterion(y)
    print(prof.key_averages())
    prof.export_chrome_trace("trace.json")

The exported file uses Chrome Trace JSON (torch's format) and loads
directly in ``chrome://tracing`` / Perfetto.
"""

from __future__ import annotations

import collections
import contextlib
import json

import tensorplay._C as _C

__all__ = ["profile", "record_function", "EventList"]


class _FunctionsTable:
    """Aggregated per-op statistics (torch.profiler.key_averages analog).

    Provides the upstream summary surface including self-CPU time: each
    event's duration minus the time spent inside its same-thread children,
    computed with a per-thread sweep over the (properly nested) spans.
    """

    def __init__(self, events, group_by_input_shape=False):
        self_ns = _self_times(events)
        agg = collections.OrderedDict()
        for ev, self_dur in zip(events, self_ns):
            name, kind, start_ns, end_ns = ev[0], ev[1], ev[2], ev[3]
            shapes = ev[5]
            if end_ns <= start_ns:
                continue
            key = (name, kind)
            if group_by_input_shape and shapes is not None:
                key = key + (tuple(tuple(s) for s in shapes),)
            cnt, total, mn, mx, stotal = agg.get(key, (0, 0, None, None, 0))
            dur = end_ns - start_ns
            agg[key] = (cnt + 1, total + dur,
                        dur if mn is None else min(mn, dur),
                        dur if mx is None else max(mx, dur),
                        stotal + self_dur)
        grand_total = sum(r[1] for r in agg.values()) or 1
        self.rows = []
        for key, (cnt, total, mn, mx, stotal) in sorted(
                agg.items(), key=lambda kv: -kv[1][4]):
            name, kind = key[0], key[1]
            shapes_sig = key[2] if len(key) > 2 else None
            self.rows.append(_Row(name, kind, cnt, total // cnt, mn, mx,
                                  shapes_sig, stotal,
                                  stotal / grand_total * 100.0))
        self.total_ns = grand_total

    def __str__(self):
        header = (f"{'Name':<28}{'Calls':>7}{'Self us':>10}"
                  f"{'Self %':>8}{'Total us':>10}")
        lines = [header, "-" * len(header)]
        for r in self.rows:
            lines.append(
                f"{r.name:<28}{r.count:>7}{r.self_us:>10.2f}"
                f"{r.self_pct:>7.1f}%{r.total_us:>10.2f}")
        lines.append("-" * len(header))
        lines.append(f"Self CPU time total: {self.total_ns/1e3:.2f} us")
        return "\n".join(lines)

    def __repr__(self):
        return str(self)


def _self_times(events):
    """Per-event self duration (ns): own span minus same-thread children.

    Spans from one thread nest properly (RAII/LIFO), so a sweep with a stack
    attributes every child's duration to its immediate parent exactly once.
    """
    by_tid = collections.defaultdict(list)
    for idx, ev in enumerate(events):
        s, e, tid = ev[2], ev[3], ev[4]
        by_tid[tid].append((s, e, idx))
    self_ns = [0] * len(events)
    for lst in by_tid.values():
        lst.sort()
        stack = []  # (start, end, idx)
        for s, e, idx in lst:
            while stack and stack[-1][1] <= s:
                stack.pop()
            dur = e - s
            self_ns[idx] += dur
            if stack:
                # charge the full duration to whoever is on top; their own
                # self-time was pre-credited above and is reduced here
                self_ns[stack[-1][2]] -= dur
            stack.append((s, e, idx))
    return self_ns


# Factory ops whose output allocation volume is derivable from
# shapes x dtype (the basis of the memory-snapshot view).
_FACTORY_PREFIXES = (
    "empty", "zeros", "ones", "full", "rand", "randn", "randint",
    "randperm", "arange", "linspace", "logspace", "eye", "tensor",
    "scalar_tensor",
)

_DTYPE_ITEMSIZE = {
    "float32": 4, "float64": 8, "float16": 2, "bfloat16": 2,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "bool": 1, "complex64": 8, "complex128": 16,
}


def _dtype_name(v):
    # dtypes ride as raw DType ints from the binding; map via tensorplay.
    try:
        import tensorplay as tp
        return str(tp.DType(v)).rsplit(".", 1)[-1]
    except Exception:
        return None


class _Row:
    def __init__(self, name, kind, count, avg_ns, min_ns, max_ns, shapes,
                 self_ns=0, self_pct=0.0):
        self.name = name
        self.kind = kind
        self.count = count
        self.input_shapes = shapes
        self.avg_ns = avg_ns
        self.min_ns = min_ns
        self.max_ns = max_ns
        self.self_ns = self_ns
        self.self_pct = self_pct

    @property
    def avg_us(self):
        return self.avg_ns / 1e3

    @property
    def min_us(self):
        return self.min_ns / 1e3

    @property
    def max_us(self):
        return self.max_ns / 1e3

    @property
    def self_us(self):
        return self.self_ns / 1e3

    @property
    def total_us(self):
        return self.avg_ns * self.count / 1e3


# profiler action constants (torch.profiler.profiler_action parity subset)
class _Action:
    NONE = "none"
    WARMUP = "warmup"
    RECORD = "record"
    WARMUP_AND_RECORD = "warmup_and_record"


def schedule(*, wait, warmup, active, repeat=0, skip_first=0):
    """Returns a step-driven schedule callable (torch parity subset).

    The returned callable maps a monotonically increasing ``step`` number to
    one of the actions NONE / WARMUP / RECORD; ``profile.step()`` consults it
    to start and stop capture cycles: wait steps do nothing, warmup steps run
    uncaptured (allocator/cache warm-up), active steps record.
    """
    def schedule_fn(step):
        if step < skip_first:
            return _Action.NONE
        cycle = warmup + active
        pos = (step - skip_first)
        if repeat > 0 and pos >= repeat * (wait + cycle):
            return _Action.NONE
        pos %= wait + cycle if (wait + cycle) > 0 else 1
        if pos < wait:
            return _Action.NONE
        pos -= wait
        if pos < warmup:
            return _Action.WARMUP
        return _Action.RECORD
    return schedule_fn


class profile:
    """Context manager recording ops executed inside the block.

    Args (torch.profiler subset): ``record_shapes`` captures per-op input
    shapes+dtypes; ``with_stack`` captures the Python call site of each op;
    ``schedule`` enables step-driven capture cycles via ``step()``.
    Extra keyword arguments are accepted and ignored so common torch call
    patterns keep working.  ``gpu_timing`` arms pool-backed cudaEvent pairs
    around dispatched CUDA work (validated on CUDA 12.x, sm_89).
    """

    def __init__(self, *args, record_shapes=False, with_stack=False,
                 schedule=None, gpu_timing=False, **kwargs):
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.gpu_timing = gpu_timing
        self.schedule = schedule
        self.step_num = 0
        self.events = None
        self._recording = False
        self._t0 = 0
        # Stop-time accounting is intentionally kept outside the event tuple:
        # it describes the profiler itself, not a user operation.  In CUDA
        # mode this is the complete event-query/reclaim path, so callers can
        # detect a regression without inferring it from wall-clock noise in
        # the profiled workload.
        self.stop_ms = 0.0
        self.gpu_timed_events = 0
        self.gpu_resolved_events = 0

    def __enter__(self):
        import time
        self._t0 = time.perf_counter_ns()
        if self.schedule is None:
            self._start_session()
        return self

    def _start_session(self):
        _C._profiler_start(self.record_shapes, self.with_stack,
                           self.gpu_timing)
        self._recording = True

    def _stop_session(self):
        import time
        stop_t0 = time.perf_counter_ns()
        raw = _C._profiler_stop()
        self.stop_ms += (time.perf_counter_ns() - stop_t0) / 1e6
        self.gpu_timed_events += sum(1 for ev in raw if ev[8] >= 0)
        # Keep the name explicit: the native binding returns all events, and
        # gpu_ms >= 0 is the success bit for a resolved CUDA event pair.
        self.gpu_resolved_events = self.gpu_timed_events
        self._recording = False
        if self.events is None:
            self.events = EventList([], base_ns=self._t0)
        self.events.extend(raw)

    def step(self):
        """Advance the schedule; called once per training iteration."""
        if self.schedule is None:
            return None
        action = self.schedule(self.step_num)
        self.step_num += 1
        if action in (_Action.RECORD, _Action.WARMUP_AND_RECORD):
            if not self._recording:
                self._start_session()
        else:  # NONE / WARMUP
            if self._recording:
                self._stop_session()
        return action

    def __exit__(self, exc_type, exc, tb):
        if self._recording:
            self._stop_session()
        if self.events is None:
            self.events = EventList([], base_ns=self._t0)
        return False

    def key_averages(self, group_by_input_shape=False, sort_by=None):
        if group_by_input_shape and not any(
                ev[5] is not None for ev in self.events):
            raise ValueError(
                "No shapes recorded: pass record_shapes=True to profile()")
        table = _FunctionsTable(self.events,
                                group_by_input_shape=group_by_input_shape)
        if sort_by:
            keys = {
                "self_cpu_time": lambda r: r.self_ns,
                "cpu_time": lambda r: r.total_us,
                "calls": lambda r: r.count,
                "name": lambda r: r.name,
            }
            if sort_by not in keys:
                raise ValueError(f"unsupported sort_by: {sort_by}")
            table.rows.sort(key=keys[sort_by], reverse=(sort_by != "name"))
        return table

    def events(self):
        """Per-event records (torch.profiler FunctionEvent subset)."""
        return [_FunctionEvent(ev) for ev in (self.events or [])]

    @property
    def current_action(self):
        """Action the schedule currently prescribes (torch parity)."""
        if self.schedule is None:
            return None
        return self.schedule(self.step_num)

    def export_chrome_trace(self, path):
        if self.events is None:
            self.events = EventList([], base_ns=self._t0)
        self.events.export_chrome_trace(path)

    def memory_summary(self):
        """Allocation-volume view derived from recorded factory ops.

        Requires ``record_shapes=True``.  For each factory op
        (empty/zeros/rand*/arange/...) the output allocation is estimated as
        numel x itemsize from its captured input shapes and dtype.  Returns
        (total_allocated_bytes, peak_live_estimate, [(ts_ns, bytes, name)]).
        This is an approximation: in-place resizes and view aliases are not
        tracked (upstream tracks them inside the allocator).
        """
        if self.events is None:
            self.events = EventList([], base_ns=self._t0)
        timeline = []
        for ev in self.events:
            name, start_ns, nbytes = ev[0], ev[2], ev[9]
            base = name.split(".")[0]
            if base not in _FACTORY_PREFIXES and not base.endswith("_like"):
                continue
            if nbytes is None or nbytes <= 0:
                continue
            timeline.append((start_ns, nbytes, name))
        timeline.sort()
        total = sum(b for _, b, _ in timeline)
        peak = live = 0
        for _ts, b, _n in timeline:
            live += b
            peak = max(peak, live)
        return total, peak, timeline



class _FunctionEvent:
    """Read-only view over one collected tuple (FunctionEvent subset)."""

    def __init__(self, ev):
        (self.name, self.kind, self.start_ns, self.end_ns, self.tid,
         self.shapes, self.dtypes, self.site, self.gpu_ms,
         self.out_bytes) = ev
        self.cpu_interval = (start_ns, end_ns) if False else \
            type("Interval", (), {"start": start_ns / 1e3,
                                  "end": end_ns / 1e3})()

    @property
    def cpu_time(self):
        return (self.end_ns - self.start_ns) / 1e3

    @property
    def input_shapes(self):
        return self.shapes

    def __repr__(self):
        return f"<FunctionEvent {self.name} {(self.end_ns-self.start_ns)/1e3:.2f}us>"


class EventList(list):
    """Collected ``(name, kind, start_ns, end_ns, tid, shapes, dtypes,
    site, gpu_ms, out_bytes)`` tuples."""

    def __init__(self, raw, base_ns=0):
        super().__init__(raw)
        self.base_ns = base_ns

    def export_chrome_trace(self, path):
        # Chrome Trace JSON -- the exact format torch.profiler exports, so
        # chrome://tracing and Perfetto render it natively.
        pid = 0
        tids = {}
        cat_of = {"o": "cpu_op", "u": "user_annotation", "b": "backward"}
        trace_events = []
        for ev in self:
            name, kind, start_ns, end_ns, tid = ev[0], ev[1], ev[2], ev[3], ev[4]
            shapes, dtypes, site, gpu_ms = ev[5], ev[6], ev[7], ev[8]
            tid_key = tids.setdefault(tid, len(tids))
            args = {}
            if shapes is not None:
                args["Input Dims"] = [list(s) for s in shapes]
                if dtypes is not None:
                    args["Input type"] = [
                        _dtype_name(d) for d in dtypes]
            if site:
                args["Call site"] = site
            if gpu_ms is not None and gpu_ms >= 0:
                args["GPU duration (ms)"] = round(gpu_ms, 4)
            trace_events.append({
                "name": name,
                "cat": cat_of.get(kind, "cpu_op"),
                "ph": "X",
                "ts": (start_ns - self.base_ns) / 1e3,  # us, relative
                "dur": max((end_ns - start_ns) / 1e3, 0.001),
                "pid": pid,
                "tid": tid_key,
                "args": args,
            })
        doc = {"traceEvents": trace_events, "displayTimeUnit": "ns"}
        with open(path, "w") as fh:
            json.dump(doc, fh)


@contextlib.contextmanager
def record_function(name):
    """Annotates a user span inside a profiling session (torch parity)."""
    _C._profiler_user_begin(name)
    try:
        yield
    finally:
        _C._profiler_user_end()


@contextlib.contextmanager
def emit_nvtx():
    """torch.autograd.profiler.emit_nvtx parity.

    Makes every dispatched op (and every backward node, via the engine)
    emit a matching NVTX range so ``nsys profile`` timelines show
    TensorPlay operator names.  Works with or without a surrounding
    ``profile()`` session; requires libnvtx at runtime (CUDA toolkit),
    otherwise ranges are silently skipped.
    """
    _C._profiler_emit_nvtx(True)
    try:
        yield
    finally:
        _C._profiler_emit_nvtx(False)


@contextlib.contextmanager
def emit_itt():
    """Intel VTune/Advisor annotation (torch kineto ITT parity).

    Every dispatched op / backward node emits an ITT task under the
    "tensorplay" domain while active.  Requires libittnotify at runtime;
    silently skipped otherwise.
    """
    _C._profiler_emit_itt(True)
    try:
        yield
    finally:
        _C._profiler_emit_itt(False)


__all__ = ["profile", "record_function", "EventList", "schedule",
           "emit_nvtx", "emit_itt"]
