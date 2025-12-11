# Operator Registration in TensorPlay

TensorPlay provides a flexible operator registration system similar to PyTorch. This allows you to register implementations (kernels) for different backends (CPU, CUDA, etc.) and for different operator variants.

## Registration Methods Comparison

TensorPlay supports two ways to register operators. Choose the one that fits your scenario:

| Feature | **In-Tree Registration** (Internal) | **Out-of-Tree Registration** (Custom Op) |
| :--- | :--- | :--- |
| **Guide** | **This Document** | `CUSTOM_OP_SYSTEM.md` |
| **Scope** | Core library (`tensorplay_core`) | External extension module |
| **Config** | `config/native_functions.yaml` | Custom YAML file |
| **Best For** | Official operators, Core contributions | Experiments, Project-specific kernels |
| **Rebuild** | Requires full rebuild | Builds independently |

This document focuses on **In-Tree Registration**.

## Overview

Operators are registered using the `Dispatcher` mechanism. Each operator has a name (e.g., "add", "mm") and can have multiple implementations keyed by `DispatchKey` (e.g., `CPU`, `CUDA`).

## How to Register a New Operator

### 1. Define the Kernel Function

First, implement your kernel function. The function signature should match the operator's expected signature.

```cpp
// MyKernels.cpp
#include "tensorplay/core/Tensor.h"
#include "tensorplay/core/Dispatcher.h"

namespace myops {

using namespace tensorplay;

Tensor my_op_cpu_kernel(const Tensor& self, double scale) {
    // Implementation...
    return self;
}

}
```

### 2. Register the Kernel

Use the `TENSORPLAY_REGISTER_KERNEL` or `TENSORPLAY_REGISTER_KERNEL_STR` macro to register the kernel.

- `TENSORPLAY_REGISTER_KERNEL(OpName, Backend, KernelFunc)`: Registers a kernel for an operator where the name is a valid C++ identifier.
- `TENSORPLAY_REGISTER_KERNEL_STR("OpName", Backend, KernelFunc)`: Registers a kernel for an operator with a custom name string (e.g., "add.Tensor").

```cpp
// Registering 'my_op' for CPU backend
TENSORPLAY_REGISTER_KERNEL(my_op, CPU, myops::my_op_cpu_kernel)

// Registering a variant with a dot in the name
TENSORPLAY_REGISTER_KERNEL_STR("my_op.variant", CPU, myops::my_op_variant_kernel)
```

### 3. Expose to Python (Optional)

To expose the operator to Python, you typically need to:

1.  Add the operator definition to `native_functions.yaml`.
2.  The build system will automatically generate the `Tensor` method or function binding.

Example `native_functions.yaml` entry:

```yaml
- func: my_op(Tensor self, double scale) -> Tensor
  variants: function, method
  dispatch:
    CPU: my_op_cpu_kernel
```

**Note:** The `dispatch` section in `native_functions.yaml` is currently used for code generation reference, but the actual dynamic dispatch happens at runtime via the `Dispatcher`. Ensure the registered name matches what the `Dispatcher` expects (usually the function name).

## Registering Backward (Autograd)

To enable autograd for your operator, you need to define its derivative in `derivatives.yaml`.

1.  **Open `derivatives.yaml`**.
2.  **Add an entry** matching your operator's signature.
3.  **Define gradients** for each differentiable input.

Example:
```yaml
- name: my_op(Tensor self, double scale) -> Tensor
  self: grad * scale
```

The build system will automatically:
1.  Generate a Backward Node class (e.g., `MyOpBackward`).
2.  Connect the forward pass to this node.
3.  Generate the `apply` method to compute gradients using the formulas provided.

**Supported syntax in formulas:**
- `grad`: The incoming gradient (gradient of the output).
- `self`, `other`, etc.: Input arguments (saved automatically).
- Method calls: `self.cos()`, `grad.mm(...)`.
- Operators: `+`, `-`, `*`, `/`.
- `neg()`: Unary negation.

**Note:** If your operator name has a dot (e.g., `add.Tensor`), the generated node name will sanitize it (e.g., `AddTensorBackward`).

## Building and Linking

Ensure your `.cpp` file containing the registration is compiled and linked into the `tensorplay_core` library. If you are adding a new file, update `CMakeLists.txt` to include it.

## Dispatch Keys

Available dispatch keys (defined in `Dispatcher.h`):

- `CPU`: For CPU implementations.
- `CUDA`: For CUDA implementations.
- `Autograd`: For autograd logic (usually handled automatically).

## Example: Custom Operator

1.  **Create `src/backend/cpu/MyOps.cpp`**:
    ```cpp
    #include "tensorplay/core/Tensor.h"
    #include "tensorplay/core/Dispatcher.h"

    namespace tensorplay {
    namespace cpu {

    Tensor custom_add(const Tensor& a, const Tensor& b) {
        return a + b; // Simplistic example
    }

    TENSORPLAY_REGISTER_KERNEL(custom_add, CPU, custom_add)

    }
    }
    ```

2.  **Add to `CMakeLists.txt`**:
    Add `src/backend/cpu/MyOps.cpp` to the `tensorplay_core` sources.

3.  **Add to `native_functions.yaml`**:
    ```yaml
    - func: custom_add(Tensor a, Tensor b) -> Tensor
      variants: function
      dispatch:
        CPU: custom_add
    ```

4.  **Add to `derivatives.yaml` (for Autograd)**:
    ```yaml
    - name: custom_add(Tensor a, Tensor b) -> Tensor
      a: grad
      b: grad
    ```

5.  **Rebuild**.
