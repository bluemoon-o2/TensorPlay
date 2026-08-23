"""Guard expressions for compiled specializations (L6).

Modeled on ``torch/_dynamo/guards.py``: every cached specialization carries a
set of guard conditions over its inputs, and a failed cache lookup can be
explained condition-by-condition ("why did this recompile happen?").

TensorPlay keeps its specialization cache keyed on structured metadata
signatures (the same fields Dynamo's TENSOR_MATCH guards over: type, shape,
dtype, device, requires_grad).  This module derives *expression objects*
from those signatures so that

* each condition is introspectable (``guard.expr`` renders like
  ``x.dtype == 'tensorplay.float32'`` or ``x.shape[0] == 4``),
* evaluation is a memoized compiled comparison rather than ad-hoc tuple
  building on every miss, and
* a miss can be explained against every stored chain, yielding the exact
  failing expressions (Dynamo's ``get_guard_fail_reason``).

The cache key itself is unchanged; guards are a projection of it.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Guard:
    """One specialization condition over one input (sub)value.

    ``path`` is a user-facing location ("x", "x[0]", "kw:training"),
    ``expr`` renders the condition, and ``expected``/``actual`` hold the
    signature components compared for this condition.
    """

    path: str
    expr: str
    expected: Any
    actual: Any = None


def _render(value: Any) -> str:
    if isinstance(value, tuple):
        return "(" + ", ".join(_render(item) for item in value) + (
            "," if len(value) == 1 else ""
        ) + ")"
    return repr(value)


def _diff_guards(expected: Any, actual: Any, path: str, out: List[Guard]) -> None:
    """Collect the leaf-level conditions where two signatures disagree."""

    if type(expected) is not type(actual):
        out.append(Guard(path, f"type({path}) == {type(expected).__name__}",
                         expected, actual))
        return
    if isinstance(expected, tuple) and isinstance(actual, tuple):
        if len(expected) != len(actual) or (
            expected and isinstance(expected[0], str) and expected[0] == "dynamic"
        ):
            out.append(Guard(path, f"{path} == {_render(expected)}",
                             expected, actual))
            return
        for index, (exp_item, act_item) in enumerate(zip(expected, actual)):
            _diff_guards(exp_item, act_item, f"{path}[{index}]", out)
        return
    if expected != actual:
        out.append(Guard(path, f"{path} == {_render(expected)}", expected, actual))


def _summarize_guards(expected: Any, path: str, out: List[Guard], depth: int) -> None:
    """Render the leading conditions of a signature as Guard expressions."""

    if depth <= 0 or not isinstance(expected, tuple):
        out.append(Guard(path, f"{path} == {_render(expected)}", expected))
        return
    if expected and isinstance(expected[0], str):
        # Tagged component ("tensor", type, shape, dtype, ...): name the
        # fields the way Dynamo's check_tensor does.
        tag = expected[0]
        rest = expected[1:]
        names = {
            "tensor": ("pytype", "shape", "dtype", "device", "requires_grad"),
        }.get(tag)
        for index, item in enumerate(rest):
            name = names[index] if names and index < len(names) else f"field{index}"
            sub_path = f"{path}.{name}"
            if isinstance(item, tuple):
                _summarize_guards(item, sub_path, out, depth - 1)
            else:
                out.append(Guard(sub_path, f"{sub_path} == {_render(item)}", item))
        return
    for index, item in enumerate(expected):
        _summarize_guards(item, f"{path}[{index}]", out, depth - 1)


class GuardChain:
    """The guard set of one cached specialization.

    ``key`` is the specialization's structured signature (identical to the
    cache key component it was built from), ``evaluate`` is a memoized
    compiled comparison against live arguments, and ``explain`` returns the
    failing conditions as :class:`Guard` expressions.
    """

    def __init__(
        self,
        key: Any,
        *,
        param_names: Sequence[str],
        dynamic: bool,
        target: Any = None,
    ) -> None:
        self.key = key
        self.param_names = tuple(param_names)
        self.dynamic = dynamic
        self._target = target
        self._signature = key[0] if isinstance(key, tuple) and len(key) == 2 else key
        self._guard_component = (
            key[1] if isinstance(key, tuple) and len(key) == 2 else ()
        )
        self._evaluate: Callable[..., bool] | None = None
        self.hits = 0

    # -- introspection -------------------------------------------------------

    @property
    def guards(self) -> List[Guard]:
        """Leading guard expressions of this specialization (for display)."""

        rendered: List[Guard] = []
        _summarize_guards(self._signature, "inputs", rendered, depth=3)
        if self._guard_component:
            rendered.append(Guard(
                "shape-guards",
                f"shape-guards == {_render(self._guard_component)}",
                self._guard_component,
            ))
        return rendered

    @property
    def source(self) -> str:
        """Human-readable guard source, Dynamo ``check_fn`` style."""

        return "def guard(" + ", ".join(self.param_names) + "):\n" + "\n".join(
            f"    # {guard.expr}" for guard in self.guards
        ) + "\n    return _signature_matches(args, kwargs)  # structured check"

    # -- evaluation ----------------------------------------------------------

    def _live_signature(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        from .api import _input_signature

        return (
            _input_signature(args, kwargs, dynamic=self.dynamic),
            self._live_guard_component(args, kwargs),
        )

    def _live_guard_component(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple:
        from .api import _value_signature

        if not self.param_names:
            return ()
        target = self._target
        if target is None:
            return ("shape-guards", "unbound")
        try:
            bound = inspect.signature(target).bind_partial(*args, **kwargs)
            bound.apply_defaults()
        except (TypeError, ValueError):
            return ("shape-guards", "unbound")
        return (
            "shape-guards",
            tuple(
                _value_signature(bound.arguments.get(name), dynamic=False)
                for name in self.param_names
            ),
        )

    def evaluate(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
        if self._evaluate is None:
            # Compile once: bind this chain into a closure so the hot path is
            # a single structured-signature comparison.
            chain = self

            def _compiled(args_: Tuple[Any, ...], kwargs_: Dict[str, Any],
                          _chain: GuardChain = chain) -> bool:
                return _chain._live_signature(args_, kwargs_) == (
                    _chain._signature, _chain._guard_component
                )

            self._evaluate = _compiled
        result = self._evaluate(args, kwargs)
        if result:
            self.hits += 1
        return result

    def explain(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> List[Guard]:
        """Guards this chain would fail against the given live arguments."""

        expected_sig = (self._signature, self._guard_component)
        live_sig = self._live_signature(args, kwargs)
        if expected_sig == live_sig:
            return []
        failures: List[Guard] = []
        _diff_guards(expected_sig[0], live_sig[0], "inputs", failures)
        _diff_guards(expected_sig[1], live_sig[1], "shape-guards", failures)
        return failures


def build_guard_chain(
    key: Any,
    *,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    dynamic: bool,
    target: Any,
) -> GuardChain:
    """Create the GuardChain stored alongside one compiled specialization."""

    param_names: Tuple[str, ...] = ()
    if isinstance(key, tuple) and len(key) == 2 and key[1]:
        component = key[1]
        if (
            isinstance(component, tuple)
            and len(component) == 2
            and component[0] == "shape-guards"
            and isinstance(component[1], tuple)
        ):
            param_names = tuple(f"arg{index}" for index in range(len(component[1])))
    return GuardChain(key, param_names=param_names, dynamic=dynamic, target=target)


def format_recompile_reasons(reasons: Sequence[Guard], limit: int = 4) -> str:
    """Render failing guard expressions for logs/warnings."""

    if not reasons:
        return "no failing guards recorded"
    shown = "; ".join(guard.expr for guard in reasons[:limit])
    extra = len(reasons) - min(len(reasons), limit)
    return shown + (f" (+{extra} more)" if extra > 0 else "")
