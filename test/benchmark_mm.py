import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import tensorplay as tp
import torch
import numpy as np

def benchmark_mm(size, warmup=100, num_iters=1000):
    print(f"Benchmarking matrix multiplication: [{size}, {size}] x [{size}, {size}]")
    
    # Calculate GFLOPS constant: 2 * M * N * K
    ops = 2.0 * size * size * size
    
    # TensorPlay
    try:
        a = tp.rand([size, size])
        b = tp.rand([size, size])
        
        # Warmup
        for _ in range(warmup):
            tp.mm(a, b)
        
        start = time.time()
        for _ in range(num_iters):
            tp.matmul(a, b)
        end = time.time()
        tp_time = (end - start) / num_iters
        tp_gflops = (ops / tp_time) / 1e9
        print(f" TensorPlay: {tp_time*1000:.4f} ms | {tp_gflops:.2f} GFLOPS")
    except Exception as e:
        print(f" TensorPlay: Failed ({e})")
        tp_time = float('inf')

    # PyTorch
    try:
        ta = torch.rand(size, size)
        tb = torch.rand(size, size)
        
        # Warmup
        for _ in range(warmup):
            torch.mm(ta, tb)
        
        start = time.time()
        for _ in range(num_iters):
            torch.mm(ta, tb)
        end = time.time()
        torch_time = (end - start) / num_iters
        torch_gflops = (ops / torch_time) / 1e9
        print(f" PyTorch   : {torch_time*1000:.4f} ms | {torch_gflops:.2f} GFLOPS")
    except Exception as e:
        print(f" PyTorch   : Failed ({e})")
        torch_time = float('inf')

    # NumPy
    try:
        na = np.random.rand(size, size).astype(np.float32)
        nb = np.random.rand(size, size).astype(np.float32)
        
        # Warmup
        for _ in range(warmup):
            np.matmul(na, nb)
        
        start = time.time()
        for _ in range(num_iters):
            np.matmul(na, nb)
        end = time.time()
        np_time = (end - start) / num_iters
        np_gflops = (ops / np_time) / 1e9
        print(f" NumPy     : {np_time*1000:.4f} ms | {np_gflops:.2f} GFLOPS")
    except Exception as e:
        print(f" NumPy     : Failed ({e})")
        np_time = float('inf')

    if torch_time > 0:
        print(f" Speedup vs PyTorch: {torch_time / tp_time:.2f}x")
    if np_time > 0:
        print(f" Speedup vs NumPy: {np_time / tp_time:.2f}x")
    print("-" * 40)

if __name__ == "__main__":
    print(f"TensorPlay Config:\n{tp.__config__.show()}")
    # Small
    benchmark_mm(32, warmup=100, num_iters=1000)
    benchmark_mm(64, warmup=100, num_iters=1000)
    # Medium
    benchmark_mm(128, warmup=100, num_iters=1000)
    benchmark_mm(256, warmup=100, num_iters=1000)
    benchmark_mm(512, warmup=100, num_iters=100)
    # Large
    benchmark_mm(1024, warmup=100, num_iters=100)
    benchmark_mm(2048, warmup=100, num_iters=100)
