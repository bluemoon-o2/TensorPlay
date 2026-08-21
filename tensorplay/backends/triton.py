"""Internal Triton code generation selected by the Stax backend.

Triton is deliberately not a second public TensorPlay compiler backend.  The
frontend and backend contract remain ``tensorplay.compile(..., backend='stax')``;
Stax selects this code generator for eligible CUDA pointwise groups.  Training
uses the same AOT-style split as Inductor: Stax receives a forward program and
a separately compiled reverse-mode program, then the runtime Function only
assembles those two compiled artifacts for autograd.
"""

from __future__ import annotations

import hashlib
import linecache
import textwrap
from typing import Any

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised on CPU-only installs
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False

from ..compiler.graph import GraphModule
from .stax import (
    _CPU_FUSED_AUTOGRAD_OPS,
    _build_fused_gradient_graphs,
    _build_pointwise_program,
    _normalize_pointwise_grad_output,
)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def _scalar_source(value: Any) -> str:
    if not _is_scalar(value):
        raise TypeError(f"Triton pointwise scalar must be numeric, got {type(value)!r}")
    if isinstance(value, bool):
        return "1.0" if value else "0.0"
    return repr(float(value))


def _supports_runtime_inputs(
    example_inputs: list[Any], *, allow_grad: bool = False
) -> bool:
    if not example_inputs:
        return False
    try:
        import tensorplay

        tensor_type = tensorplay.Tensor
    except (AttributeError, ImportError):
        return False
    if any(not isinstance(value, tensor_type) for value in example_inputs):
        return False
    if any(
        not value.device.is_cuda()
        or not value.is_contiguous()
        or (value.requires_grad and not allow_grad)
        for value in example_inputs
    ):
        return False
    first = example_inputs[0]
    return all(
        value.shape == first.shape
        and value.dtype == first.dtype
        and value.device == first.device
        for value in example_inputs[1:]
    )


class TritonProgramCodegen:
    """Generate explicit Triton source for Stax's shared postfix program.

    The same representation is used by the CPU fused path and by the CUDA
    forward/backward kernels.  Instructions are expanded into source at
    compile time, so the kernel has no per-element opcode dispatch loop.
    """

    _OP_NAMES = {
        1: "add",
        2: "sub",
        3: "mul",
        4: "div",
        5: "pow",
        6: "neg",
        7: "pos",
        8: "abs",
        9: "sin",
        10: "cos",
        11: "exp",
        12: "log",
        13: "sigmoid",
        14: "sqrt",
        15: "square",
        16: "tanh",
        17: "relu",
        18: "relu_grad",
        19: "abs_grad",
    }

    def __init__(
        self,
        program: list[int],
        constants: list[float],
        output_refs: tuple[int, ...],
        input_count: int,
    ) -> None:
        if len(program) % 3:
            raise ValueError("Triton Stax program must contain triples")
        self.program = program
        self.constants = constants
        self.output_refs = output_refs
        self.input_count = input_count

    def _ref(self, ref: int) -> str:
        if ref < 0:
            index = -ref - 1
            if index < 0 or index >= len(self.constants):
                raise ValueError(f"invalid Triton Stax constant reference: {ref}")
            return _scalar_source(self.constants[index])
        if ref < self.input_count:
            return f"in{ref}"
        return f"tmp{ref - self.input_count}"

    def _expression(self, opcode: int, lhs_ref: int, rhs_ref: int) -> str:
        try:
            name = self._OP_NAMES[opcode]
        except KeyError as exc:
            raise ValueError(f"unsupported Triton Stax opcode: {opcode}") from exc
        lhs = self._ref(lhs_ref)
        if name in {"add", "sub", "mul", "div", "pow"}:
            rhs = self._ref(rhs_ref)
            return {
                "add": f"{lhs} + {rhs}",
                "sub": f"{lhs} - {rhs}",
                "mul": f"{lhs} * {rhs}",
                "div": f"{lhs} / {rhs}",
                "pow": f"{lhs} ** {rhs}",
            }[name]
        return {
            "neg": f"-{lhs}",
            "pos": lhs,
            "abs": f"tl.abs({lhs})",
            "sin": f"tl.sin({lhs})",
            "cos": f"tl.cos({lhs})",
            "exp": f"tl.exp({lhs})",
            "log": f"tl.log({lhs})",
            "sigmoid": f"(1.0 / (1.0 + tl.exp(-{lhs})))",
            "sqrt": f"tl.sqrt({lhs})",
            "square": f"{lhs} * {lhs}",
            "tanh": f"libdevice.tanh({lhs})",
            "relu": f"tl.maximum({lhs}, 0.0)",
            "relu_grad": f"tl.where({lhs} > 0.0, 1.0, 0.0)",
            "abs_grad": (
                f"tl.where({lhs} > 0.0, 1.0, "
                f"tl.where({lhs} < 0.0, -1.0, 0.0))"
            ),
        }[name]

    def generate(self, kernel_name: str) -> str:
        body: list[str] = [
            "xoffset = tl.program_id(0) * XBLOCK",
            "xindex = xoffset + tl.arange(0, XBLOCK)",
            "xmask = xindex < xnumel",
        ]
        for index in range(self.input_count):
            body.append(
                f"in{index} = tl.load(in_ptr{index} + xindex, mask=xmask, other=0.0)"
            )
        for instruction, offset in enumerate(range(0, len(self.program), 3)):
            opcode, lhs_ref, rhs_ref = self.program[offset : offset + 3]
            expression = self._expression(opcode, lhs_ref, rhs_ref)
            body.append(f"tmp{instruction} = {expression}")
        for output_index, output_ref in enumerate(self.output_refs):
            body.append(
                f"tl.store(out_ptr{output_index} + xindex, {self._ref(output_ref)}, mask=xmask)"
            )

        signature = [
            *(f"in_ptr{index}" for index in range(self.input_count)),
            *(f"out_ptr{index}" for index in range(len(self.output_refs))),
            "xnumel",
            "XBLOCK: tl.constexpr",
        ]
        source = (
            "import triton\n"
            "import triton.language as tl\n"
            "import triton.language.extra.cuda.libdevice as libdevice\n\n"
        )
        source += "@triton.autotune(\n"
        source += "    configs=[\n"
        source += "        triton.Config({'XBLOCK': 128}, num_warps=4),\n"
        source += "        triton.Config({'XBLOCK': 256}, num_warps=4),\n"
        source += "        triton.Config({'XBLOCK': 512}, num_warps=8),\n"
        source += "        triton.Config({'XBLOCK': 1024}, num_warps=8),\n"
        source += "    ],\n"
        source += "    key=['xnumel'],\n"
        source += ")\n"
        source += "@triton.jit\n"
        source += f"def {kernel_name}({', '.join(signature)}):\n"
        source += textwrap.indent("\n".join(body), "    ") + "\n\n"
        source += "def kernel_launch(inputs):\n"
        source += "    import tensorplay as tp\n"
        source += "    xnumel = inputs[0].numel()\n"
        source += "    outputs = [tp.empty_like(inputs[0], requires_grad=False) for _ in range(" \
            f"{len(self.output_refs)})]\n"
        source += "    grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)\n"
        call_args = [
            *(f"inputs[{index}]" for index in range(self.input_count)),
            *(f"outputs[{index}]" for index in range(len(self.output_refs))),
            "xnumel",
        ]
        source += f"    {kernel_name}[grid]({', '.join(call_args)})\n"
        if len(self.output_refs) == 1:
            source += "    return outputs[0]\n"
        else:
            source += "    return outputs\n"
        return source


def _compile_program(
    program: list[int],
    constants: list[float],
    output_refs: tuple[int, ...],
    example_inputs: list[Any],
):
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    if not _supports_runtime_inputs(example_inputs, allow_grad=True):
        raise NotImplementedError("Triton requires matching contiguous CUDA tensors")
    digest = hashlib.sha256(
        (
            repr((program, constants, output_refs))
            + repr(
                [
                    (tuple(value.shape), repr(value.dtype), repr(value.device))
                    for value in example_inputs
                ]
            )
        ).encode()
    ).hexdigest()[:16]
    kernel_name = f"stax_triton_program_{digest}"
    source = TritonProgramCodegen(
        program, constants, output_refs, len(example_inputs)
    ).generate(kernel_name)
    fake_file = f"<tensorplay-stax-triton-program-{digest}>"
    linecache.cache[fake_file] = (
        len(source),
        None,
        source.splitlines(True),
        fake_file,
    )
    namespace: dict[str, Any] = {"triton": triton, "tl": tl}
    exec(compile(source, fake_file, "exec"), namespace, namespace)
    return namespace["kernel_launch"]


def compile_graph_module(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    mode: str | None = None,
    strict_native: bool = False,
    **kwargs: Any,
):
    del mode, kwargs
    if not HAS_TRITON or not _supports_runtime_inputs(
        example_inputs, allow_grad=True
    ):
        return None
    pointwise = _build_pointwise_program(graph_module)
    if pointwise is None:
        return None
    placeholders, forward_program, forward_constants, instructions, output_ref = pointwise
    if len(placeholders) != len(example_inputs):
        return None

    # This is the Stax equivalent of Inductor's forward inner compiler.  The
    # backward program is compiled separately below, just as compile_fx calls
    # its ``bw_compiler`` after AOTAutograd partitions the joint graph.
    forward_launch = _compile_program(
        forward_program,
        forward_constants,
        (output_ref,),
        example_inputs,
    )
    backward_launch = None
    autograd_function: Any | None = None
    if any(value.requires_grad for value in example_inputs):
        if any(
            op_name not in _CPU_FUSED_AUTOGRAD_OPS
            for op_name, *_ in instructions
        ):
            return None
        gradient_plan = _build_fused_gradient_graphs(
            len(placeholders),
            instructions,
            forward_program,
            forward_constants,
            len(forward_program) // 3,
            output_ref,
        )
        if gradient_plan is None:
            return None
        backward_program, backward_constants, backward_outputs = gradient_plan
        # The extra input is the tangent/grad-output supplied by autograd.
        backward_launch = _compile_program(
            backward_program,
            backward_constants,
            backward_outputs,
            [*example_inputs, example_inputs[0]],
        )
        from ..autograd import Function

        class _StaxTritonAutograd(Function):
            @staticmethod
            def forward(ctx: Any, *forward_inputs: Any) -> Any:
                ctx.save_for_backward(*forward_inputs)
                return forward_launch(list(forward_inputs))

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
                grad_output = grad_outputs[0] if grad_outputs else None
                if grad_output is None:
                    return (None,) * len(ctx.saved_tensors)
                grad_output = _normalize_pointwise_grad_output(
                    grad_output, ctx.saved_tensors[0]
                )
                return tuple(
                    backward_launch([*ctx.saved_tensors, grad_output])
                )

        autograd_function = _StaxTritonAutograd

    placeholders = graph_module.graph.placeholders
    fallback = None if strict_native else graph_module.recompile()

    def compiled(*args: Any, **call_kwargs: Any) -> Any:
        if not call_kwargs and len(args) == len(placeholders):
            inputs = list(args)
        else:
            bound = graph_module.signature.bind_partial(*args, **call_kwargs)
            bound.apply_defaults()
            inputs = [bound.arguments[node.name] for node in placeholders]
        if not _supports_runtime_inputs(inputs, allow_grad=True):
            if strict_native:
                raise RuntimeError(
                    "Stax strict_native Triton lowering received inputs outside "
                    "its compiled specialization"
                )
            assert fallback is not None
            return fallback(*args, **call_kwargs)
        if backward_launch is not None and any(
            value.requires_grad for value in inputs
        ):
            if autograd_function is None:
                raise RuntimeError("Stax Triton autograd function is missing")
            return autograd_function.apply(*inputs)
        return forward_launch(inputs)

    compiled._tensorplay_codegen = "triton"  # type: ignore[attr-defined]
    compiled._tensorplay_backward_codegen = (  # type: ignore[attr-defined]
        "triton" if backward_launch is not None else None
    )
    return compiled
