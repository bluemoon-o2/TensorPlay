# tensorplay.compiler

TensorPlay ships an in-process graph compiler used by {mod}`tensorplay.stax`
and the CUDA graphs workflow. This module is TensorPlay-specific and has no
direct upstream torch counterpart (it is conceptually closest to
`torch.compile` internals plus `torch.cuda.graphs`).

The reference below is a static listing: the underlying extension modules are
under active development, so this page intentionally does not use autodoc.

## Graph IR

- `Graph`, `Node`, `GraphModule` — graph representation and module wrapper
- `PassBase`, `DecomposePass`, `ConstFold`, `DeadCodeElimination`,
  `NormalizeOperators` — compiler passes
- `NodePathTracer` — attribution helper

## CUDA graphs

- `CudaGraphManager`, `CudaGraphError` — capture/replay management
- `Guard`, `GuardChain` — shape/guard caching for replay validation

## Caching and errors

- `CodeCache` — compiled artifact cache
- `AOTError`, `GraphCaptureError` — error types

## Related modules

- {mod}`tensorplay.jit` — tracing/scripting entry points (`script`, `trace`,
  `export`, `ignore`, `is_tracing`)
- {mod}`tensorplay.export` — `ExportedProgram` / `GraphSignature` export format
- `tensorplay.cuda.CUDAGraph` — low-level stream capture API
