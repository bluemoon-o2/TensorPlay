#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#define TP_CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace tensorplay {
namespace cuda {

namespace {

template <typename T>
__device__ inline bool topk_is_nan_device(T value) {
  if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
    return isnan(value);
  } else if constexpr (std::is_same<T, Half>::value ||
                       std::is_same<T, BFloat16>::value) {
    return isnan(static_cast<float>(value));
  } else {
    return false;
  }
}

template <typename T>
__device__ inline bool topk_before_device(T lhs, int64_t lhs_index, T rhs,
                                           int64_t rhs_index, bool largest) {
  const bool lhs_nan = topk_is_nan_device(lhs);
  const bool rhs_nan = topk_is_nan_device(rhs);
  if (lhs_nan != rhs_nan) return largest ? lhs_nan : !lhs_nan;
  if (lhs_nan) return lhs_index < rhs_index;
  if (lhs < rhs) return !largest;
  if (rhs < lhs) return largest;
  return lhs_index < rhs_index;
}

template <typename T>
__device__ inline T topk_padding_device(bool largest) {
  if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
    return largest ? static_cast<T>(-INFINITY) : static_cast<T>(NAN);
  } else if constexpr (std::is_same<T, Half>::value ||
                       std::is_same<T, BFloat16>::value) {
    return T(largest ? -INFINITY : NAN);
  } else {
    return largest ? std::numeric_limits<T>::lowest()
                   : std::numeric_limits<T>::max();
  }
}

template <typename T>
__device__ inline void bitonic_swap(T* vals, int64_t* idxs, int i, int j,
                                    bool up, bool largest) {
  const bool should_swap = (up == !largest)
      ? topk_before_device(vals[j], idxs[j], vals[i], idxs[i], largest)
      : topk_before_device(vals[i], idxs[i], vals[j], idxs[j], largest);
  if (should_swap) {
    T value = vals[i];
    vals[i] = vals[j];
    vals[j] = value;
    int64_t index = idxs[i];
    idxs[i] = idxs[j];
    idxs[j] = index;
  }
}

template <typename T>
__global__ void bitonic_topk_kernel(
    const T* __restrict__ in,
    T* __restrict__ out_vals,
    int64_t* __restrict__ out_idxs,
    int64_t rows, int64_t cols, int64_t k, int64_t inner, bool largest,
    int64_t n) {
  extern __shared__ unsigned char raw[];
  T* smem_vals = reinterpret_cast<T*>(raw);
  const size_t index_offset = (n * sizeof(T) + 7u) & ~size_t(7u);
  int64_t* smem_idxs = reinterpret_cast<int64_t*>(raw + index_offset);
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const int64_t outer_index = row / inner;
  const int64_t inner_index = row % inner;
  const int64_t input_base = outer_index * cols * inner + inner_index;
  const int64_t output_base = outer_index * k * inner + inner_index;
  const T padding = topk_padding_device<T>(largest);

  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    smem_vals[i] = i < cols ? in[input_base + i * inner] : padding;
    smem_idxs[i] = i;
  }
  __syncthreads();

  const bool ascending = !largest;
  for (int64_t size = 2; size <= n; size <<= 1) {
    for (int64_t stride = size >> 1; stride > 0; stride >>= 1) {
      for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t j = i ^ stride;
        if (j > i) {
          const bool up = ((i & size) == 0) == ascending;
          bitonic_swap(smem_vals, smem_idxs, static_cast<int>(i),
                       static_cast<int>(j), up, largest);
        }
      }
      __syncthreads();
    }
  }

  for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
    out_vals[output_base + i * inner] = smem_vals[i];
    out_idxs[output_base + i * inner] = smem_idxs[i];
  }
}

template <typename T>
__global__ void iterative_topk_kernel(
    T* __restrict__ scratch,
    uint8_t* __restrict__ selected,
    const T* __restrict__ in,
    T* __restrict__ out_vals,
    int64_t* __restrict__ out_idxs,
    int64_t rows, int64_t cols, int64_t k, int64_t inner, bool largest) {
  extern __shared__ unsigned char raw[];
  T* best_values = reinterpret_cast<T*>(raw);
  const size_t index_offset = (blockDim.x * sizeof(T) + 7u) & ~size_t(7u);
  int64_t* best_indices = reinterpret_cast<int64_t*>(raw + index_offset);
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const int64_t outer_index = row / inner;
  const int64_t inner_index = row % inner;
  const int64_t input_base = outer_index * cols * inner + inner_index;
  const int64_t output_base = outer_index * k * inner + inner_index;
  T* row_scratch = scratch + input_base;
  uint8_t* row_selected = selected + input_base;

  for (int64_t column = threadIdx.x; column < cols; column += blockDim.x) {
    row_scratch[column * inner] = in[input_base + column * inner];
    row_selected[column * inner] = 0;
  }
  __syncthreads();

  for (int64_t result = 0; result < k; ++result) {
    bool found = false;
    T best_value{};
    int64_t best_index = -1;
    for (int64_t column = threadIdx.x; column < cols; column += blockDim.x) {
      if (row_selected[column * inner] != 0) continue;
      const T value = row_scratch[column * inner];
      if (!found || topk_before_device(value, column, best_value, best_index, largest)) {
        found = true;
        best_value = value;
        best_index = column;
      }
    }
    best_values[threadIdx.x] = best_value;
    best_indices[threadIdx.x] = found ? best_index : -1;
    __syncthreads();
    if (threadIdx.x == 0) {
      found = false;
      for (int64_t thread = 0; thread < blockDim.x; ++thread) {
        const int64_t candidate = best_indices[thread];
        if (candidate < 0) continue;
        if (!found || topk_before_device(best_values[thread], candidate,
                                         best_value, best_index, largest)) {
          found = true;
          best_value = best_values[thread];
          best_index = candidate;
        }
      }
      out_vals[output_base + result * inner] = best_value;
      out_idxs[output_base + result * inner] = best_index;
      if (found) row_selected[best_index * inner] = 1;
    }
    __syncthreads();
  }
}

template <typename T>
void launch_topk_cuda(const Tensor& input, Tensor& values, Tensor& indices,
                      int64_t rows, int64_t cols, int64_t k, int64_t inner,
                      bool largest, int64_t impl) {
  constexpr int kThreads = 256;
  int64_t padded = 1;
  while (padded < cols) padded <<= 1;
  const size_t bitonic_index_offset =
      (static_cast<size_t>(padded) * sizeof(T) + 7u) & ~size_t(7u);
  const size_t bitonic_smem = bitonic_index_offset +
                              static_cast<size_t>(padded) * sizeof(int64_t);
  const bool use_bitonic = impl == 0 && padded <= 4096 && bitonic_smem <= 48 * 1024;
  if (impl != 0 && impl != 1) {
    TP_THROW(RuntimeError, "topk: unknown impl " + std::to_string(impl));
  }

  if (use_bitonic) {
    const int threads = static_cast<int>(std::min<int64_t>(padded, 512));
    bitonic_topk_kernel<T><<<static_cast<unsigned>(rows), threads, bitonic_smem,
                             getCurrentCUDAStream().stream()>>>(
        input.data_ptr<T>(), values.data_ptr<T>(), indices.data_ptr<int64_t>(),
        rows, cols, k, inner, largest, padded);
    TP_CUDA_CHECK(cudaGetLastError());
    return;
  }

  Tensor scratch = Tensor::empty(input.shape(), input.dtype(), input.device());
  Tensor selected = Tensor::zeros(input.shape(), DType::UInt8, input.device());
  const size_t iterative_index_offset =
      (static_cast<size_t>(kThreads) * sizeof(T) + 7u) & ~size_t(7u);
  const size_t iterative_smem = iterative_index_offset +
                                static_cast<size_t>(kThreads) * sizeof(int64_t);
  iterative_topk_kernel<T><<<static_cast<unsigned>(rows), kThreads, iterative_smem,
                             getCurrentCUDAStream().stream()>>>(
      scratch.data_ptr<T>(), selected.data_ptr<uint8_t>(), input.data_ptr<T>(),
      values.data_ptr<T>(), indices.data_ptr<int64_t>(), rows, cols, k, inner,
      largest);
  TP_CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::tuple<Tensor, Tensor> topk_kernel_cuda(const Tensor& self, int64_t k, int64_t dim,
                                             bool largest, bool sorted, int64_t impl) {
  (void)sorted;
  Tensor input = self.contiguous();
  const int64_t ndim = input.dim();
  if (ndim == 0) {
    TP_CHECK(dim == 0 || dim == -1, "topk: dimension out of range");
  } else {
    if (dim < 0) dim += ndim;
    TP_CHECK(dim >= 0 && dim < ndim, "topk: dimension out of range");
  }
  const int64_t dim_size = ndim == 0 ? 1 : input.size(dim);
  if (k < 0 || k > dim_size) {
    TP_THROW(RuntimeError, "topk: k must be in [0, dimension size]");
  }

  std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(input.shape());
  if (ndim != 0) shape[static_cast<size_t>(dim)] = k;
  Tensor values = Tensor::empty(shape, input.dtype(), input.device());
  Tensor indices = Tensor::empty(shape, DType::Int64, input.device());
  if (k == 0 || input.numel() == 0) return {values, indices};

  int64_t outer = 1;
  int64_t inner = 1;
  for (int64_t axis = 0; axis < (ndim == 0 ? 0 : dim); ++axis) outer *= input.size(axis);
  for (int64_t axis = ndim == 0 ? 0 : dim + 1; axis < ndim; ++axis) inner *= input.size(axis);
  const int64_t rows = outer * inner;

  switch (input.dtype()) {
    case DType::UInt8:
      launch_topk_cuda<uint8_t>(input, values, indices, rows, dim_size, k, inner, largest, impl);
      break;
    case DType::Int8:
      launch_topk_cuda<int8_t>(input, values, indices, rows, dim_size, k, inner, largest, impl);
      break;
    case DType::Int16:
      launch_topk_cuda<int16_t>(input, values, indices, rows, dim_size, k, inner, largest, impl);
      break;
    case DType::Int32:
      launch_topk_cuda<int32_t>(input, values, indices, rows, dim_size, k, inner, largest, impl);
      break;
    case DType::Int64:
      launch_topk_cuda<int64_t>(input, values, indices, rows, dim_size, k, inner, largest, impl);
      break;
    case DType::UInt16:
      launch_topk_cuda<uint16_t>(input, values, indices, rows, dim_size, k, inner, largest, impl);
      break;
    case DType::UInt32:
      launch_topk_cuda<uint32_t>(input, values, indices, rows, dim_size, k, inner, largest, impl);
      break;
    case DType::UInt64:
      launch_topk_cuda<uint64_t>(input, values, indices, rows, dim_size, k, inner, largest, impl);
      break;
    case DType::Float16:
      launch_topk_cuda<Half>(input, values, indices, rows, dim_size, k, inner, largest, impl);
      break;
    case DType::BFloat16:
      launch_topk_cuda<BFloat16>(input, values, indices, rows, dim_size, k, inner, largest, impl);
      break;
    case DType::Float32:
      launch_topk_cuda<float>(input, values, indices, rows, dim_size, k, inner, largest, impl);
      break;
    case DType::Float64:
      launch_topk_cuda<double>(input, values, indices, rows, dim_size, k, inner, largest, impl);
      break;
    case DType::Bool:
    case DType::ComplexFloat:
    case DType::ComplexDouble:
    case DType::ComplexHalf:
    case DType::BComplex32:
    case DType::Float8_e4m3fn:
    case DType::Float8_e5m2:
    case DType::Undefined:
    case DType::NumOptions:
      TP_THROW(NotImplementedError, "topk: unsupported dtype");
  }
  return {values, indices};
}

TENSORPLAY_LIBRARY_IMPL(CUDA, TopKKernels) {
  m.impl("topk", topk_kernel_cuda);
}

} // namespace cuda
} // namespace tensorplay
