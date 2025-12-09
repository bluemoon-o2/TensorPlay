# TensorPlay Matrix Multiplication (mm/matmul) Benchmark Report

## 1. Test Overview
### 1.1 Purpose
Quantify the performance of TensorPlay's built-in matrix multiplication operators (`mm`/`matmul`) 
under a **unified 1000-iteration** setup. Compare with PyTorch and NumPy to identify efficiency and 
applicable scenarios for optimization and business selection.

### 1.2 Test Objects
| Type          | Operator/Interface               | Description                                 |
|---------------|----------------------------------|---------------------------------------------|
| Test Subject  | `tensorplay.Tensor.mm`/`matmul`  | TensorPlay's original matrix multiplication |
| Baseline 1    | `torch.mm`                       | PyTorch's square matrix multiplication      |
| Baseline 2    | `numpy.matmul`                   | NumPy's general matrix multiplication       |

### 1.3 Test Scenarios
All sizes use the same test strategy, covering key application dimensions:
- Small: 32x32, 64x64 (lightweight inference, embedded computing)
- Medium: 128x128, 256x256, 512x512 (general numerical computing, model layers)
- Large: 1024x1024, 2048x2048 (HPC, large-model inference)

## 2. Test Environment
### 2.1 TensorPlay Build Config
```
TensorPlay built with:
  - C++ Compiler: MSVC 19.44.35221.0
  - Build type: Release
  - BLAS Info: MKL
  - BLAS Found: YES
  - OpenMP: TRUE
```

### 2.2 Hardware & System
- OS: Windows 10/11 (MSVC-compatible)
- CPU: AVX2/AVX512-supported
- Memory: ≥32GB (avoid 2048x2048 iteration bottlenecks)
- Python Version: 3.8+

### 2.3 Dependencies
- TensorPlay: Custom build (MKL/OpenMP enabled)
- PyTorch: 2.0.0+ (MKL backend accelerated)
- NumPy: 2.0.0+ (MKL accelerated, float32 optimized)

## 3. Test Results Summary
### Metrics Explanation
- Execution Time (ms): Average of 1000 iterations (eliminates random fluctuations)
- GFLOPS: Floating-point operations per second (Formula: `2*N³ / Execution Time (s)`)
- Speedup: `Baseline Time / TensorPlay Time` (>1 = TensorPlay is faster)

| Matrix Size | Metric                  | TensorPlay | PyTorch   | NumPy     |
|-------------|-------------------------|------------|-----------|-----------|
| 32x32       | Execution Time (ms)     | 0.0022     | 0.0031    | 0.0030    |
|             | GFLOPS                  | 30.27      | 20.81     | 21.78     |
|             | Speedup (vs PyTorch)    | 1.45x      | -         | -         |
|             | Speedup (vs NumPy)      | 1.39x      | -         | -         |
| 64x64       | Execution Time (ms)     | 0.0224     | 0.0107    | 0.0111    |
|             | GFLOPS                  | 23.37      | 49.15     | 47.20     |
|             | Speedup (vs PyTorch)    | 0.48x      | -         | -         |
|             | Speedup (vs NumPy)      | 0.50x      | -         | -         |
| 128x128     | Execution Time (ms)     | 0.0182     | 0.0202    | 0.0854    |
|             | GFLOPS                  | 230.53     | 207.60    | 49.12     |
|             | Speedup (vs PyTorch)    | 1.11x      | -         | -         |
|             | Speedup (vs NumPy)      | 4.69x      | -         | -         |
| 256x256     | Execution Time (ms)     | 0.0595     | 0.0341    | 0.1353    |
|             | GFLOPS                  | 564.08     | 984.96    | 247.91    |
|             | Speedup (vs PyTorch)    | 0.57x      | -         | -         |
|             | Speedup (vs NumPy)      | 2.28x      | -         | -         |
| 512x512     | Execution Time (ms)     | 0.4626     | 0.3133    | 0.5354    |
|             | GFLOPS                  | 580.23     | 856.75    | 501.39    |
|             | Speedup (vs PyTorch)    | 0.68x      | -         | -         |
|             | Speedup (vs NumPy)      | 1.16x      | -         | -         |
| 1024x1024   | Execution Time (ms)     | 2.2578     | 1.6505    | 2.6189    |
|             | GFLOPS                  | 951.14     | 1301.13   | 820.00    |
|             | Speedup (vs PyTorch)    | 0.73x      | -         | -         |
|             | Speedup (vs NumPy)      | 1.16x      | -         | -         |
| 2048x2048   | Execution Time (ms)     | 24.5792    | 16.6043   | 15.9139   |
|             | GFLOPS                  | 698.96     | 1034.66   | 1079.55   |
|             | Speedup (vs PyTorch)    | 0.68x      | -         | -         |
|             | Speedup (vs NumPy)      | 0.65x      | -         | -         |

## 4. Test Methodology
### 4.1 Core Strategy
- Unified Parameters: 100 warmup iterations (eliminate cold start) + 1000 test iterations (stable mean)
- Precision Consistency: float32 across all frameworks (NumPy explicitly set `dtype=np.float32`)
- Call Path: TensorPlay directly binds to C++ backend (no system-level overhead)

### 4.2 Data Preparation
- TensorPlay: `tp.rand([size, size])` (contiguous memory allocation)
- PyTorch: `torch.rand(size, size)` (framework-optimized generation)
- NumPy: `np.random.rand(size, size).astype(np.float32)` (precision alignment)

### 4.3 Environment Isolation
- Close high-resource processes (browsers, compilers) for exclusive CPU/memory access
- Verify MKL/OpenMP enablement via TensorPlay config

## 5. Performance Analysis
### 5.1 Strengths
#### 32x32 Small Matrix
- Performance: 30.27 GFLOPS (1.45x vs PyTorch, 1.39x vs NumPy)
- Reason: C++ binding avoids framework overhead; MKL+OpenMP minimizes small-matrix scheduling latency

#### 128x128 Medium Matrix
- Performance: 230.53 GFLOPS (1.11x vs PyTorch, 4.69x vs NumPy)
- Reason: Blocking strategy aligns with CPU L1/L2 cache; near-peak cache hit rate

### 5.2 Weaknesses
#### 64x64 Medium Matrix
- Performance: 23.37 GFLOPS (48% of PyTorch, 50% of NumPy)
- Reason: No size-specific blocking optimization; PyTorch/NumPy have MKL local optima here

#### 2048x2048 Large Matrix
- Performance: 698.96 GFLOPS (32% slower than PyTorch, 35% slower than NumPy)
- Reason: Lack of dynamic blocking/memory prefetch; memory bandwidth saturation

### 5.3 General Scenarios (256x256~1024x1024)
- Pattern: Consistently outperforms NumPy (1.16x~2.28x speedup) but lags PyTorch (60%~73% GFLOPS)
- Key Gap: PyTorch uses adaptive blocking + CPU core binding for better parallel efficiency

## 6. Conclusions & Recommendations
### 6.1 Core Conclusions
- Optimal Scenarios: 32x32 (lightweight inference) and 128x128 (general computing) – outperforms industry tools
- Weak Scenarios: 64x64 and 2048x2048 – not recommended for HPC
- Overall Position: Replace NumPy in pure numerical computing (except 2048x2048); PyTorch remains better for deep learning/ultra-large matrices

### 6.2 Operator Optimization Recommendations
1. Dynamic Blocking: Add size-specific blocking for 64x64/2048x2048 (reference PyTorch's adaptive logic)
2. Multi-thread Scheduling: Adjust thread count by matrix size; bind cores for large matrices to reduce switching overhead
3. Memory Optimization: Add blocked computation + prefetch for 2048x2048 to alleviate bandwidth bottlenecks

## Appendix: Test Code
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import tensorplay as tp
import torch
import numpy as np

def benchmark_mm(size, warmup=100, num_iters=1000):
    print(f"Benchmarking [{size}x{size}] matrix multiplication")
    ops = 2.0 * size * size * size  # Floating-point operations
    
    # TensorPlay Test
    try:
        a, b = tp.rand([size, size]), tp.rand([size, size])
        [tp.mm(a, b) for _ in range(warmup)]
        start = time.time()
        [tp.matmul(a, b) for _ in range(num_iters)]
        tp_time = (time.time() - start) / num_iters
        tp_gflops = (ops / tp_time) / 1e9
        print(f" TensorPlay: {tp_time*1000:.4f} ms | {tp_gflops:.2f} GFLOPS")
    except Exception as e:
        print(f" TensorPlay: Failed ({e})")
        tp_time = float('inf')

    # PyTorch Test
    try:
        ta, tb = torch.rand(size, size), torch.rand(size, size)
        [torch.mm(ta, tb) for _ in range(warmup)]
        start = time.time()
        [torch.mm(ta, tb) for _ in range(num_iters)]
        torch_time = (time.time() - start) / num_iters
        torch_gflops = (ops / torch_time) / 1e9
        print(f" PyTorch   : {torch_time*1000:.4f} ms | {torch_gflops:.2f} GFLOPS")
    except Exception as e:
        print(f" PyTorch   : Failed ({e})")
        torch_time = float('inf')

    # NumPy Test (float32)
    try:
        na, nb = np.random.rand(size, size).astype(np.float32), np.random.rand(size, size).astype(np.float32)
        [np.matmul(na, nb) for _ in range(warmup)]
        start = time.time()
        [np.matmul(na, nb) for _ in range(num_iters)]
        np_time = (time.time() - start) / num_iters
        np_gflops = (ops / np_time) / 1e9
        print(f" NumPy     : {np_time*1000:.4f} ms | {np_gflops:.2f} GFLOPS")
    except Exception as e:
        print(f" NumPy     : Failed ({e})")
        np_time = float('inf')

    # Print Speedup
    if torch_time > 0:
        print(f" Speedup vs PyTorch: {torch_time / tp_time:.2f}x")
    if np_time > 0:
        print(f" Speedup vs NumPy: {np_time / tp_time:.2f}x")
    print("-" * 40)

if __name__ == "__main__":
    print(f"TensorPlay Config:\n{tp.__config__.show()}")
    # Unified test for all sizes
    for size in [32, 64, 128, 256, 512, 1024, 2048]:
        benchmark_mm(size, warmup=100, num_iters=1000)
```