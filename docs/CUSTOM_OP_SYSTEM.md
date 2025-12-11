# TensorPlay Custom Operator System (Out-of-Tree)

This document explains the role of `tensorplay/cmake` and `tensorplay/tools/codegen/tensorplaygen.py`. This system is designed to support users in developing independent custom operator extensions outside of the TensorPlay core library (similar to PyTorch's Custom C++ Extensions).

## 1. Core Components

This system consists of two main parts:

### 1.1 `tools/codegen/tensorplaygen.py` (Code Generator)

This is the core code generation tool for custom operators. It reads user-defined YAML configuration files and automatically generates C++ boilerplate code and Python bindings.

*   **Input**: A YAML file containing operator definitions (e.g., `my_ops.yaml`).
*   **Function**:
    1.  Parses function signatures in YAML (e.g., `func: my_op(Tensor a) -> Tensor`).
    2.  Generates **C++ Header (`OpsGenerated.h`)**: Declares the C++ function interfaces that the user needs to implement.
    3.  Generates **Python Binding Implementation (`OpsBinding.cpp`)**: Generates Python module entries based on Nanobind and automatically handles device dispatch logic, forwarding Python calls to CPU or CUDA implementations.
*   **Purpose**: Allows users to focus on core algorithm implementation without manually writing complex Python C++ bindings and device dispatch logic.

### 1.2 `cmake/TensorPlayCustomOp.cmake` (CMake Build Helper)

This is a CMake module that provides a convenient function `add_tensorplay_op` to simplify the build configuration for custom operator projects.

*   **Function**:
    *   Encapsulates commands to call `tensorplaygen.py`.
    *   Configures Nanobind module builds.
    *   Automatically handles header include paths and compiler options.
*   **Usage**: Users simply `include` this file in their own `CMakeLists.txt` and call `add_tensorplay_op` to generate an extension module with one click.

---

## 2. Comparison: In-Tree vs. Out-of-Tree Registration

TensorPlay supports two ways to register operators. Choose the one that fits your scenario:

| Feature | **In-Tree Registration** (Internal) | **Out-of-Tree Registration** (Custom Op) |
| :--- | :--- | :--- |
| **Documentation** | See `OPERATOR_REGISTRATION.md` | **This document** |
| **Config File** | `config/native_functions.yaml` | Your own `.yaml` file |
| **Build Artifact** | Compiled into `tensorplay_core` library | Compiled as a standalone Python extension (`.pyd` / `.so`) |
| **Rebuild Requirement** | Requires rebuilding the entire TensorPlay project | Only builds your extension; links against installed TensorPlay |
| **Use Case** | Adding standard operators to the core library; Contributing to TensorPlay | Creating private kernels; Fast prototyping; Third-party plugins |
| **Autograd** | Fully supported via `config/derivatives.yaml` | Currently limited (manual backward implementation may be required) |

## 3. Workflow Example

Suppose a user wants to create an extension package named `my_extension`.

### Step 1: Define Operators (my_ops.yaml)
```yaml
- func: custom_add(Tensor a, Tensor b) -> Tensor
  dispatch:
    CPU: custom_add_cpu
    CUDA: custom_add_cuda
```

### Step 2: Configure Build (CMakeLists.txt)
```cmake
cmake_minimum_required(VERSION 3.18)
project(MyExtension)

# Include TensorPlay helper script
# Assumes TENSORPLAY_ROOT is set to the TensorPlay source directory
include(${TENSORPLAY_ROOT}/cmake/TensorPlayCustomOp.cmake)

# Add extension module
add_tensorplay_op(
    NAME my_extension          # Generated Python module name
    YAML my_ops.yaml           # Operator definition
    SOURCES src/MyKernel.cpp   # User C++ implementation
)
```

### Step 3: Implement Kernels (src/MyKernel.cpp)
Include the generated header and implement the corresponding functions:
```cpp
#include "OpsGenerated.h" 

namespace impl {
    tensorplay::Tensor custom_add_cpu(const tensorplay::Tensor& a, const tensorplay::Tensor& b) {
        // ... Implementation ...
    }
}
```

### Step 4: Compile and Use
After compilation, use it directly in Python:
```python
import my_extension
result = my_extension.custom_add(tensor_a, tensor_b)
```

## 4. Summary

*   **`tensorplaygen.py`** is the "translator" behind the scenes, responsible for translating simple YAML definitions into compilable C++ binding code.
*   **`TensorPlayCustomOp.cmake`** is the "glue" of the build system, allowing users to easily compile generated code and their own implementation code into a Python extension library.

Together, they form TensorPlay's **Out-of-Tree Custom Operator Development Framework**.
