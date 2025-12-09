# TensorPlay Operations (Ops)

It is intended to contain:
1.  **Device-agnostic operator implementations**: Logic that is common across all backends.
2.  **Composite operators**: Operators implemented in terms of other operators (e.g., `softmax` using `exp` and `sum`).
3.  **Dispatch logic**: If manual dispatch beyond `DispatchStub` is required.

Currently, most operator kernels are located in `src/backend/cpu`. As the project grows, backend-agnostic logic should be moved here.
