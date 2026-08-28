// Tier 2-4 operators part 2 - CUDA kernels: reductions + shape ops.
// Companion to TierOpsKernels.cu (which owns arithmetic/comparisons/math/
// clamp/activations). Same ATen anchors as cpu/TierOpsKernels.cpp.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <utility>
#include <type_traits>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace {

constexpr int kThreads = 256;

inline dim3 make_grid(int64_t work) {
    return dim3(static_cast<unsigned>((work + kThreads - 1) / kThreads));
}

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    // c10::maybe_wrap_dim (WrapDimMinimal.cpp): IndexError, message reports
    // the original (unwrapped) dim.
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 min, ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

// c10::maybe_wrap_dim with wrap_scalar=true: rank-0 accepts dims [-1, 0]
// (both wrap to 0).  Used by flip's dim_list_to_bitset (WrapDimUtilsMulti.h).
inline int64_t wrap_dim_scalar(int64_t dim, int64_t ndim) {
    return wrap_dim(dim, ndim == 0 ? 1 : ndim);
}

inline void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& inner) {
    outer = 1; inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
}

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

Tensor pack_i64(const std::vector<int64_t>& v, const Device& dev) {
    Tensor t = Tensor::empty({static_cast<int64_t>(std::max<size_t>(v.size(), 1))},
                             DType::Int64, dev);
    if (!v.empty())
        cudaMemcpy(t.data_ptr<int64_t>(), v.data(), v.size() * sizeof(int64_t),
                   cudaMemcpyHostToDevice);
    return t;
}

// ---------------------------------------------------------------------------
// Reduction slice kernels (one thread per output slice)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void slice_max_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                 const T* in, double* out) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        double acc = -std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < d_size; ++j) {
            double v = static_cast<double>(sp[j * inner]);
            acc = (v != v || v > acc) ? v : acc;  // NaN propagates
        }
        out[si] = acc;
    }
}

template <typename T>
__global__ void slice_min_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                 const T* in, double* out) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        double acc = std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < d_size; ++j) {
            double v = static_cast<double>(sp[j * inner]);
            acc = (v != v || v < acc) ? v : acc;
        }
        out[si] = acc;
    }
}

template <typename T>
__global__ void slice_nansum_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                    const T* in, double* out) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        double acc = 0;
        for (int64_t j = 0; j < d_size; ++j) {
            double v = static_cast<double>(sp[j * inner]);
            if (v == v) acc += v;
        }
        out[si] = acc;
    }
}

template <typename T>
__global__ void slice_count_nonzero_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                           const T* in, double* out) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        int64_t c = 0;
        for (int64_t j = 0; j < d_size; ++j)
            if (sp[j * inner] != static_cast<T>(0)) ++c;
        out[si] = static_cast<double>(c);
    }
}

template <typename T>
__global__ void slice_logsumexp_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                       const T* in, double* out) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        double m = -std::numeric_limits<double>::infinity();
        bool has_nan = false;
        for (int64_t j = 0; j < d_size; ++j) {
            double v = static_cast<double>(sp[j * inner]);
            if (v != v) { has_nan = true; break; }
            if (v > m) m = v;
        }
        if (has_nan) { out[si] = ::nan(""); continue; }
        if (m == -std::numeric_limits<double>::infinity()) { out[si] = m; continue; }
        double s2 = 0;
        for (int64_t j = 0; j < d_size; ++j)
            s2 += ::exp(static_cast<double>(sp[j * inner]) - m);
        out[si] = m + ::log(s2);
    }
}

template <typename T>
__global__ void slice_mean_f64_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                      const T* in, double* out, bool squares) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        double s2 = 0;
        for (int64_t j = 0; j < d_size; ++j) {
            double v = static_cast<double>(sp[j * inner]);
            s2 += squares ? v * v : v;
        }
        out[si] = s2 / static_cast<double>(d_size);
    }
}

template <typename T>
__global__ void cummaxmin_scan_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                      const T* in, T* vals, int64_t* idxs, bool is_max) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* s2p = in + o * d_size * inner + in2;
        T* vp = vals + o * d_size * inner + in2;
        int64_t* ip = idxs + o * d_size * inner + in2;
        T best = s2p[0];
        int64_t bi = 0;
        vp[0] = best;
        ip[0] = 0;
        for (int64_t j = 1; j < d_size; ++j) {
            double cur = static_cast<double>(s2p[j * inner]);
            double b = static_cast<double>(best);
            if ((is_max && cur > b) || (!is_max && cur < b)) { best = s2p[j * inner]; bi = j; }
            vp[j * inner] = best;
            ip[j * inner] = bi;
        }
    }
}

template <typename T>
__global__ void renorm_slice_kernel(int64_t n_slices, int64_t d_size, int64_t inner,
                                    const T* in, T* out, double p, double maxnorm) {
    int64_t si = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; si < n_slices; si += stride) {
        int64_t o = si / inner, in2 = si % inner;
        const T* sp = in + o * d_size * inner + in2;
        T* dp = out + o * d_size * inner + in2;
        double norm = 0;
        if (p == std::numeric_limits<double>::infinity()) {
            for (int64_t j = 0; j < d_size; ++j)
                norm = ::fmax(norm, ::fabs(static_cast<double>(sp[j * inner])));
        } else {
            double s2 = 0;
            for (int64_t j = 0; j < d_size; ++j)
                s2 += ::pow(::fabs(static_cast<double>(sp[j * inner])), p);
            norm = ::pow(s2, 1.0 / p);
        }
        double factor = norm > maxnorm ? maxnorm / norm : 1.0;
        for (int64_t j = 0; j < d_size; ++j)
            dp[j * inner] = static_cast<T>(static_cast<double>(sp[j * inner]) * factor);
    }
}

// Multi-dim iterative reduction driver.
// which: 0=max, 1=min, 2=nansum, 3=count_nonzero
Tensor reduce_iterative(const Tensor& self, std::vector<int64_t> dims, bool keepdim,
                        int which, DType out_dtype_override = DType::Undefined) {
    Tensor cur = self.contiguous();
    int64_t nd = cur.dim();
    std::vector<bool> reduced(nd, false);
    for (auto& d : dims) reduced[wrap_dim(d, nd)] = true;
    auto stream = getCurrentCUDAStream().stream();
    // With keepdim=false each pass erases the reduced axis, so later original
    // dims shift down; track the current position of original dim `dim`.
    int64_t shift = 0;
    for (int64_t dim = 0; dim < nd; ++dim) {
        if (!reduced[dim]) continue;
        const int64_t cur_dim = dim - shift;
        int64_t d_size = cur.size(cur_dim);
        int64_t outer = 1, inner = 1;
        outer_inner(shape_of(cur), cur_dim, outer, inner);
        int64_t slices = outer * inner;
        Tensor accs = Tensor::zeros({slices}, DType::Float64,
                                    self.device());
        if (slices > 0 && d_size > 0) {
            dim3 grid = make_grid(slices), block(kThreads);
#define TP_RI(ctype, name_) \
    case DType::name_: \
        if (which == 0) \
            slice_max_kernel<ctype><<<grid, block, 0, stream>>>( \
                slices, d_size, inner, cur.data_ptr<ctype>(), accs.data_ptr<double>()); \
        else if (which == 1) \
            slice_min_kernel<ctype><<<grid, block, 0, stream>>>( \
                slices, d_size, inner, cur.data_ptr<ctype>(), accs.data_ptr<double>()); \
        else if (which == 2) \
            slice_nansum_kernel<ctype><<<grid, block, 0, stream>>>( \
                slices, d_size, inner, cur.data_ptr<ctype>(), accs.data_ptr<double>()); \
        else \
            slice_count_nonzero_kernel<ctype><<<grid, block, 0, stream>>>( \
                slices, d_size, inner, cur.data_ptr<ctype>(), accs.data_ptr<double>()); \
        break;
            switch (cur.dtype()) {
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_RI)
                default: TP_THROW(TypeError, "reduce: unsupported dtype");
            }
#undef TP_RI
            CUDA_CHECK(cudaGetLastError());
        }
        std::vector<int64_t> ns = shape_of(cur);
        if (keepdim) {
            ns[cur_dim] = 1;
        } else {
            ns.erase(ns.begin() + cur_dim);
            ++shift;
        }
        cur = accs.reshape(ns);
    }
    DType final_dt = out_dtype_override == DType::Undefined ? self.dtype() : out_dtype_override;
    return cur.to(final_dt);
}

// ---------------------------------------------------------------------------
// Shape-op device kernels
// ---------------------------------------------------------------------------

__global__ void trace_batch_kernel(int64_t batch, int64_t rows, int64_t cols,
                                   const double* sp, double* dp) {
    int64_t bi = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; bi < batch; bi += stride) {
        double s2 = 0;
        int64_t d = rows < cols ? rows : cols;
        for (int64_t i = 0; i < d; ++i) s2 += sp[bi * rows * cols + i * cols + i];
        dp[bi] = s2;
    }
}

template <typename T>
__global__ void index_gather_kernel(int64_t n, const T* src, const int64_t* idx, T* out) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) out[i] = src[idx[i]];
}

template <typename T>
__global__ void diag_scatter_kernel(int64_t n, int64_t size, int64_t diagonal,
                                    const T* src, T* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        int64_t r = diagonal >= 0 ? i : i - diagonal;
        int64_t c = diagonal >= 0 ? i + diagonal : i;
        dst[r * size + c] = src[i];
    }
}

template <typename T>
__global__ void row_copy_kernel(int64_t total_rows, int64_t inner,
                                const T* src, T* dst,
                                int64_t src_row_stride, int64_t dst_row_stride) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < total_rows * inner; i += stride) {
        int64_t r = i / inner, c = i % inner;
        dst[r * dst_row_stride + c] = src[r * src_row_stride + c];
    }
}

template <typename T>
__global__ void flip_map_kernel(int64_t n, int64_t nd, const T* src, T* dst,
                                const int64_t* sizes, const int64_t* flips) {
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < n; li += stride) {
        int64_t r2 = li, src_off = 0, mult = 1;
        for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
            int64_t c = r2 % sizes[d2];
            r2 /= sizes[d2];
            int64_t sc3 = flips[d2] ? (sizes[d2] - 1 - c) : c;
            src_off += sc3 * mult;
            mult *= sizes[d2];
        }
        dst[li] = src[src_off];
    }
}

template <typename T>
__global__ void roll_map_kernel(int64_t n, int64_t nd, const T* src, T* dst,
                                const int64_t* sizes, const int64_t* shifts) {
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < n; li += stride) {
        int64_t r2 = li, src_off = 0, mult = 1;
        for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
            int64_t c = r2 % sizes[d2];
            r2 /= sizes[d2];
            int64_t sc3 = c - shifts[d2];
            if (sc3 < 0) sc3 += sizes[d2];
            src_off += sc3 * mult;
            mult *= sizes[d2];
        }
        dst[li] = src[src_off];
    }
}

template <typename T>
__global__ void broadcast_map_kernel(int64_t n, int64_t nd, const T* src, T* dst,
                                     const int64_t* out_sizes,
                                     const int64_t* in_sizes_padded,
                                     const int64_t* in_strides_padded) {
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < n; li += stride) {
        int64_t r2 = li, src_off = 0;
        for (int64_t d2 = nd - 1; d2 >= 0; --d2) {
            int64_t c = r2 % out_sizes[d2];
            r2 /= out_sizes[d2];
            src_off += (in_sizes_padded[d2] == 1 ? 0 : c) * in_strides_padded[d2];
        }
        dst[li] = src[src_off];
    }
}

template <typename T>
__global__ void repeat_interleave_kernel(int64_t total_rows, int64_t inner, int64_t out_d,
                                         int64_t d_size, int64_t repeats,
                                         const T* src, T* dst) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < total_rows; t += stride) {
        int64_t o = t / out_d, j = t % out_d;
        int64_t src_j = j / repeats;
        const T* s = src + (o * d_size + src_j) * inner;
        T* d = dst + (o * out_d + j) * inner;
        for (int64_t c = 0; c < inner; ++c) d[c] = s[c];
    }
}

} // anonymous namespace

// ===========================================================================
// Reduction entry points
// ===========================================================================

// ATen parity (ReduceOps.cpp meta funcs + ReduceOpsUtils.h
// zero_numel_check_dims): reducing an empty tensor is only valid along an
// explicitly given non-empty dim; a full reduction has no identity.
static void zero_numel_check_dims(const Tensor& self, const std::vector<int64_t>& dims,
                                  const char* fn_name) {
    if (dims.empty()) {
        TP_THROW(RuntimeError, fn_name,
                 ": Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    const int64_t nd = self.dim();
    for (int64_t d : dims) {
        if (d < 0) d += nd;
        TP_CHECK_INDEX(self.size(d) != 0, fn_name,
                       ": Expected reduction dim ", d, " to have non-zero size.");
    }
}

Tensor amax_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim, "amax()");
    return reduce_iterative(self, dim.empty()
                                       ? [&]{ std::vector<int64_t> a;
                                              for (int64_t i = 0; i < self.dim(); ++i) a.push_back(i);
                                              return a; }()
                                       : dim,
                            keepdim, 0);
}
Tensor amin_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
    if (self.numel() == 0) zero_numel_check_dims(self, dim, "amin()");
    return reduce_iterative(self, dim.empty()
                                       ? [&]{ std::vector<int64_t> a;
                                              for (int64_t i = 0; i < self.dim(); ++i) a.push_back(i);
                                              return a; }()
                                       : dim,
                            keepdim, 1);
}
std::tuple<Tensor, Tensor> aminmax_cuda(const Tensor& self, std::vector<int64_t> dim,
                                        bool keepdim) {
    if (self.numel() == 0) {
        if (dim.empty()) {
            TP_THROW(RuntimeError, "aminmax(): cannot compute aminmax over an empty dimension as "
                     "the operation has no identity.");
        }
        zero_numel_check_dims(self, dim, "aminmax");
    }
    return {amin_cuda2(self, dim, keepdim), amax_cuda2(self, dim, keepdim)};
}
Tensor logsumexp_cuda2(const Tensor& self, int64_t dim, bool keepdim) {
    if (!isFloatingType(self.dtype()))
        TP_THROW(RuntimeError, "logsumexp(): Expected floating point type");
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous().to(DType::Float64);
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(sc), dim, outer, inner);
    int64_t slices = outer * inner;
    Tensor accs = Tensor::zeros({slices}, DType::Float64, self.device());
    if (slices > 0) {
        // The kernel handles d_size==0 itself (writes -inf per slice, matching
        // torch's log(0) result for reducing an empty dim).
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid = make_grid(slices), block(kThreads);
        slice_logsumexp_kernel<double><<<grid, block, 0, stream>>>(
            slices, d_size, inner, sc.data_ptr<double>(), accs.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<int64_t> ns = shape_of(sc);
    ns[dim] = keepdim ? 1 : 0;
    if (!keepdim) ns.erase(ns.begin() + dim);
    DType out_dt = self.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    return accs.reshape(ns).to(out_dt);
}
Tensor nansum_cuda2(const Tensor& self, const std::vector<int64_t>& dim_in, bool keepdim) {
    DType out_dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    std::vector<int64_t> dim = dim_in;
    if (dim.empty()) {
        // torch: dim omitted (or empty) reduces over every dimension
        for (int64_t i = 0; i < self.dim(); ++i) dim.push_back(i);
    }
    return reduce_iterative(self, dim, keepdim, 2, out_dt);
}
Tensor count_nonzero_cuda2(const Tensor& self, const std::vector<int64_t>& dim) {
    return reduce_iterative(self, dim, false, 3, DType::Int64);
}

std::tuple<Tensor, Tensor> cummax_cuda(const Tensor& self, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor vals = Tensor::empty(shape_of(sc), sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(shape_of(sc), DType::Int64, sc.device());
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(sc), dim, outer, inner);
    int64_t slices = outer * inner;
    if (slices > 0 && d_size > 0) {
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid = make_grid(slices), block(kThreads);
#define TP_CM(ctype, name_) \
    case DType::name_: \
        cummaxmin_scan_kernel<ctype><<<grid, block, 0, stream>>>( \
            slices, d_size, inner, sc.data_ptr<ctype>(), vals.data_ptr<ctype>(), \
            idxs.data_ptr<int64_t>(), true); \
        break;
        switch (sc.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_CM)
            default: TP_THROW(TypeError, "cummax: unsupported dtype");
        }
#undef TP_CM
        CUDA_CHECK(cudaGetLastError());
    }
    return {vals, idxs};
}
std::tuple<Tensor, Tensor> cummin_cuda(const Tensor& self, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor vals = Tensor::empty(shape_of(sc), sc.dtype(), sc.device());
    Tensor idxs = Tensor::empty(shape_of(sc), DType::Int64, sc.device());
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(sc), dim, outer, inner);
    int64_t slices = outer * inner;
    if (slices > 0 && d_size > 0) {
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid = make_grid(slices), block(kThreads);
#define TP_CMIN(ctype, name_) \
    case DType::name_: \
        cummaxmin_scan_kernel<ctype><<<grid, block, 0, stream>>>( \
            slices, d_size, inner, sc.data_ptr<ctype>(), vals.data_ptr<ctype>(), \
            idxs.data_ptr<int64_t>(), false); \
        break;
        switch (sc.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_CMIN)
            default: TP_THROW(TypeError, "cummin: unsupported dtype");
        }
#undef TP_CMIN
        CUDA_CHECK(cudaGetLastError());
    }
    return {vals, idxs};
}

std::tuple<Tensor, Tensor> var_mean_cuda(const Tensor& self, std::vector<int64_t> dim,
                                         bool unbiased, bool keepdim) {
    // var = E[x^2] - E[x]^2 corrected by n/(n-ddof); Float64 accumulation.
    Tensor xf = self.to(DType::Float64).contiguous();
    Tensor x2f = xf.mul(xf).contiguous();
    Tensor mean = xf, msq = x2f;
    std::vector<int64_t> dims = dim;
    bool any = false;
    for (auto& d : dims) { d = wrap_dim(d, self.dim()); any = true; }
    if (!any) { dims.clear(); for (int64_t i = 0; i < self.dim(); ++i) dims.push_back(i); }
    auto stream = getCurrentCUDAStream().stream();
    // Ascend over the original dims; with keepdim=false each pass erases the
    // reduced axis, so later dims shift down by the number already removed.
    std::vector<bool> red(static_cast<size_t>(self.dim()), false);
    for (int64_t d2 : dims) red[static_cast<size_t>(d2)] = true;
    int64_t shift = 0;
    for (int64_t dd = 0; dd < self.dim(); ++dd) {
        if (!red[static_cast<size_t>(dd)]) continue;
        const int64_t cur_d = dd - shift;
        int64_t dsz = mean.size(cur_d);
        int64_t outer = 1, inner = 1;
        outer_inner(shape_of(mean), cur_d, outer, inner);
        int64_t slices = outer * inner;
        Tensor m1 = Tensor::zeros({slices}, DType::Float64, self.device());
        Tensor m2 = Tensor::zeros(shape_of(m1), DType::Float64, self.device());
        if (slices > 0) {
            // The kernel divides by dsz, so dsz==0 slices become NaN, matching
            // torch's var/mean over an empty reduction dim.
            dim3 grid = make_grid(slices), block(kThreads);
            slice_mean_f64_kernel<double><<<grid, block, 0, stream>>>(
                slices, dsz, inner, mean.data_ptr<double>(), m1.data_ptr<double>(), false);
            slice_mean_f64_kernel<double><<<grid, block, 0, stream>>>(
                slices, dsz, inner, msq.data_ptr<double>(), m2.data_ptr<double>(), true);
            CUDA_CHECK(cudaGetLastError());
        }
        std::vector<int64_t> ns = shape_of(mean);
        if (keepdim) {
            ns[cur_d] = 1;
        } else {
            ns.erase(ns.begin() + cur_d);
            ++shift;
        }
        mean = m1.reshape(ns);
        msq = m2.reshape(ns);
    }
    int64_t n_red = 1;
    for (int64_t d2 : dims) n_red *= self.size(d2);
    double ddof = (unbiased && n_red > 1) ? 1.0 : 0.0;
    Tensor var = msq.sub(mean.mul(mean));
    Tensor corr = var.mul(Tensor::full({}, Scalar(n_red / (n_red - ddof)),
                                       DType::Float64, self.device()));
    DType out_dt = self.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    return {corr.to(out_dt), mean.to(out_dt)};
}
std::tuple<Tensor, Tensor> std_mean_cuda(const Tensor& self, std::vector<int64_t> dim,
                                         bool unbiased, bool keepdim) {
    auto vm = var_mean_cuda(self, dim, unbiased, keepdim);
    return {std::get<0>(vm).sqrt(), std::get<1>(vm)};
}

Tensor nanmedian_cuda(const Tensor& self) {
    // Host-staged reference implementation (rare op).
    Tensor host = self.to(DType::Float64).contiguous().to(Device(DeviceType::CPU));
    std::vector<double> vals;
    const double* pp = host.data_ptr<double>();
    for (int64_t i = 0; i < host.numel(); ++i)
        if (!(pp[i] != pp[i])) vals.push_back(pp[i]);
    DType out_dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Int64;
    double med = 0;
    if (!vals.empty()) {
        std::sort(vals.begin(), vals.end());
        med = vals[(vals.size() - 1) / 2];
    }
    return Tensor::zeros({}, out_dt, self.device()).fill_(Scalar(med));
}

std::tuple<Tensor, Tensor> mode_cuda(const Tensor& self, int64_t dim, bool keepdim) {
    // Host-staged reference implementation (rare op).
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor hv = self.contiguous().to(Device(DeviceType::CPU));
    Tensor hi = Tensor::zeros(shape_of(hv), DType::Int64, Device(DeviceType::CPU));
    int64_t d_size = hv.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(hv), dim, outer, inner);
#define TP_MODEH(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = hv.data_ptr<ctype>(); \
        ctype* vp = hv.data_ptr<ctype>(); \
        int64_t* ip = hi.data_ptr<int64_t>(); \
        for (int64_t si = 0; si < outer * inner; ++si) { \
            int64_t o = si / inner, in2 = si % inner; \
            std::vector<std::pair<ctype, int64_t>> buf(d_size); \
            for (int64_t j = 0; j < d_size; ++j) buf[j] = {sp[(o*d_size+j)*inner+in2], j}; \
            std::sort(buf.begin(), buf.end(), [](const std::pair<ctype,int64_t>& a2, \
                                                 const std::pair<ctype,int64_t>& b2){ \
                if (!(a2.first<b2.first) && !(b2.first<a2.first)) return a2.second<b2.second; \
                return a2.first<b2.first; }); \
            ctype bv = buf[0].first; int64_t bc = 0, bi2 = buf[0].second, run = 0; \
            for (int64_t j = 0; j < d_size; ++j) { \
                bool same = j > 0 && !(buf[j].first<buf[j-1].first) && !(buf[j-1].first<buf[j].first); \
                run = same ? run + 1 : 1; \
                if (run > bc) { bc = run; bv = buf[j].first; bi2 = buf[j].second; } } \
            vp[si] = bv; ip[si] = bi2; \
        } \
        break; }
    switch (hv.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_MODEH)
        default: TP_THROW(TypeError, "mode: unsupported dtype");
    }
#undef TP_MODEH
    std::vector<int64_t> out_shape = shape_of(hv);
    out_shape[dim] = keepdim ? 1 : 0;
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    return {hv.reshape(out_shape).to(self.device()),
            hi.reshape(out_shape).to(self.device())};
}

std::tuple<Tensor, Tensor> kthvalue_cuda(const Tensor& self, int64_t k, int64_t dim,
                                         bool keepdim) {
    // Host-staged reference implementation (rare op).
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    int64_t d_size = self.size(dim);
    if (k < 1 || k > d_size)
        TP_THROW(RuntimeError, "kthvalue(): selected number k out of range for dim ", dim);
    Tensor host = self.contiguous().to(Device(DeviceType::CPU));
    Tensor vals = Tensor::empty(shape_of(host), host.dtype(), Device(DeviceType::CPU));
    Tensor idxs = Tensor::empty(shape_of(host), DType::Int64, Device(DeviceType::CPU));
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(host), dim, outer, inner);
#define TP_KTHH(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = host.data_ptr<ctype>(); \
        ctype* vp = vals.data_ptr<ctype>(); \
        int64_t* ip = idxs.data_ptr<int64_t>(); \
        for (int64_t si = 0; si < outer * inner; ++si) { \
            int64_t o = si/inner, in2 = si%inner; \
            std::vector<std::pair<ctype, int64_t>> buf(d_size); \
            for (int64_t j = 0; j < d_size; ++j) buf[j] = {sp[(o*d_size+j)*inner+in2], j}; \
            std::stable_sort(buf.begin(), buf.end(), [](const std::pair<ctype,int64_t>& a2, \
                                                        const std::pair<ctype,int64_t>& b2){ \
                return a2.first<b2.first; }); \
            vp[si] = buf[k-1].first; ip[si] = buf[k-1].second; \
        } \
        break; }
    switch (host.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_KTHH)
        default: TP_THROW(TypeError, "kthvalue: unsupported dtype");
    }
#undef TP_KTHH
    std::vector<int64_t> out_shape = shape_of(host);
    out_shape[dim] = keepdim ? 1 : 0;
    if (!keepdim) out_shape.erase(out_shape.begin() + dim);
    return {vals.reshape(out_shape).to(self.device()),
            idxs.reshape(out_shape).to(self.device())};
}

Tensor dist_cuda(const Tensor& self, const Tensor& other, Scalar p) {
    // Host-staged reference implementation (rare op).
    std::vector<int64_t> bshape = broadcast_shapes(shape_of(self), shape_of(other));
    Tensor a = self.to(DType::Float64).expand(bshape).contiguous().to(Device(DeviceType::CPU));
    Tensor b = other.to(DType::Float64).expand(bshape).contiguous().to(Device(DeviceType::CPU));
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    int64_t n = a.numel();
    double pd = p.toDouble();
    double result = 0;
    if (pd == std::numeric_limits<double>::infinity()) {
        for (int64_t i = 0; i < n; ++i) result = std::max(result, std::fabs(ap[i] - bp[i]));
    } else if (pd == -std::numeric_limits<double>::infinity()) {
        result = std::numeric_limits<double>::infinity();
        for (int64_t i = 0; i < n; ++i) result = std::min(result, std::fabs(ap[i] - bp[i]));
    } else if (pd == 0.0) {
        for (int64_t i = 0; i < n; ++i) if (ap[i] != bp[i]) result += 1;
    } else {
        double s2 = 0;
        for (int64_t i = 0; i < n; ++i) s2 += std::pow(std::fabs(ap[i] - bp[i]), pd);
        result = std::pow(s2, 1.0 / pd);
    }
    return Tensor::zeros({}, DType::Float64, self.device()).fill_(Scalar(result));
}

Tensor renorm_cuda(const Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(sc), sc.dtype(), sc.device());
    int64_t d_size = sc.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(sc), dim, outer, inner);
    int64_t slices = outer * inner;
    if (slices > 0 && d_size > 0) {
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid = make_grid(slices), block(kThreads);
        double pd = p.toDouble(), mn = maxnorm.toDouble();
#define TP_REN(ctype, name_) \
    case DType::name_: \
        renorm_slice_kernel<ctype><<<grid, block, 0, stream>>>( \
            slices, d_size, inner, sc.data_ptr<ctype>(), out.data_ptr<ctype>(), pd, mn); \
        break;
        switch (sc.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_REN)
            default: TP_THROW(TypeError, "renorm: unsupported dtype");
        }
#undef TP_REN
        CUDA_CHECK(cudaGetLastError());
    }
    return out;
}

// ===========================================================================
// Shape ops
// ===========================================================================

namespace {

__global__ void meshgrid_coord_kernel(int64_t total, size_t k, size_t j,
                                      const int64_t* sizes, int64_t* out) {
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < total; li += stride) {
        int64_t r2 = li;
        int64_t coord = 0;
        for (size_t d2 = k; d2-- > 0;) {
            coord = r2 % sizes[d2];
            r2 /= sizes[d2];
            if (d2 == j) break;
        }
        out[li] = coord;
    }
}

template <typename T>
__global__ void pixel_shuffle_map_kernel(int64_t n, int64_t C, int64_t H, int64_t W,
                                         int64_t r, const T* src, T* dst) {
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < n; li += stride) {
        int64_t w = li % W; int64_t rem = li / W;
        int64_t h = rem % H; rem /= H;
        int64_t c = rem % C; int64_t bn = rem / C;
        int64_t ih = h % r, iw = w % r;
        int64_t src_off = ((((bn * C * r * r) + c * r * r + ih * r + iw) * H) + h / r) * W + w / r;
        dst[li] = src[src_off];
    }
}

template <typename T>
__global__ void pixel_unshuffle_map_kernel(int64_t n, int64_t C, int64_t H, int64_t W,
                                           int64_t r, const T* src, T* dst) {
    // out: (N, C*r^2, H, W); in: (N, C, H*r, W*r)
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t Wr = W * r;
    for (; li < n; li += stride) {
        int64_t w = li % W; int64_t rem = li / W;
        int64_t h = rem % H; rem /= H;
        int64_t cc = rem % (C * r * r); rem /= (C * r * r);
        int64_t bn = rem;
        int64_t c = cc / (r * r);
        int64_t ij = cc % (r * r);
        int64_t ih = ij / r, iw = ij % r;
        int64_t src_off = (((bn * C + c) * (H * r)) + h * r + ih) * Wr + w * r + iw;
        dst[li] = src[src_off];
    }
}

template <typename T>
__global__ void channel_shuffle_map_kernel(int64_t n, int64_t outer, int64_t C, int64_t inner,
                                           int64_t cg, const T* src, T* dst) {
    // li layout: (outer * C + c) * inner + tail; channel shuffle swaps the
    // group index and the within-group index.
    int64_t li = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; li < n; li += stride) {
        int64_t tail = li % inner;
        int64_t rest = li / inner;
        int64_t c = rest % C;
        int64_t o = rest / C;
        int64_t j = c / cg, gi = c % cg;
        int64_t src_c = gi * cg + j;
        dst[li] = src[(o * C + src_c) * inner + tail];
    }
}



} // anonymous namespace

Tensor trace_cuda(const Tensor& self) {
    if (self.dim() < 2) TP_THROW(RuntimeError, "trace: input must have at least 2 dimensions");
    int64_t rows = self.size(-2), cols = self.size(-1);
    int64_t batch = self.numel() / (rows * cols);
    std::vector<int64_t> out_shape = shape_of(self);
    out_shape.resize(out_shape.size() - 2);
    Tensor out64 = Tensor::zeros({std::max<int64_t>(batch, 1)}, DType::Float64, self.device());
    if (batch > 0) {
        Tensor sc = self.contiguous().to(DType::Float64);
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid = make_grid(batch), block(kThreads);
        trace_batch_kernel<<<grid, block, 0, stream>>>(
            batch, rows, cols, sc.data_ptr<double>(), out64.data_ptr<double>());
        CUDA_CHECK(cudaGetLastError());
    }
    return out64.reshape(out_shape).to(self.dtype());
}

Tensor diag_cuda(const Tensor& self, int64_t diagonal) {
    int64_t nd = self.dim();
    Tensor sc = self.contiguous();
    if (nd == 1) {
        int64_t n = sc.size(0);
        int64_t size = n + std::abs(diagonal);
        Tensor out = Tensor::zeros({size, size}, sc.dtype(), sc.device());
        auto stream = getCurrentCUDAStream().stream();
        dim3 grid = make_grid(std::max<int64_t>(n, 1)), block(kThreads);
#define TP_DGS(ctype, name_) \
    case DType::name_: \
        diag_scatter_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, size, diagonal, sc.data_ptr<ctype>(), out.data_ptr<ctype>()); \
        break;
        switch (sc.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_DGS)
            default: TP_THROW(TypeError, "diag: unsupported dtype");
        }
#undef TP_DGS
        CUDA_CHECK(cudaGetLastError());
        return out;
    }
    if (nd == 2) {
        int64_t rows = sc.size(0), cols = sc.size(1);
        std::vector<int64_t> idx;
        if (diagonal >= 0) {
            for (int64_t i = 0; i + diagonal < cols && i < rows; ++i)
                idx.push_back(i * cols + i + diagonal);
        } else {
            for (int64_t i = 0; i - diagonal < rows && i < cols; ++i)
                idx.push_back((i - diagonal) * cols + i);
        }
        Tensor out = Tensor::empty({static_cast<int64_t>(idx.size())}, sc.dtype(), sc.device());
        if (!idx.empty()) {
            Tensor d_idx = pack_i64(idx, sc.device());
            auto stream = getCurrentCUDAStream().stream();
            dim3 grid = make_grid(static_cast<int64_t>(idx.size())), block(kThreads);
#define TP_DGE(ctype, name_) \
    case DType::name_: \
        index_gather_kernel<ctype><<<grid, block, 0, stream>>>( \
            static_cast<int64_t>(idx.size()), sc.data_ptr<ctype>(), \
            d_idx.data_ptr<int64_t>(), out.data_ptr<ctype>()); \
        break;
            switch (sc.dtype()) {
                TENSORPLAY_FORALL_SCALAR_TYPES(TP_DGE)
                default: TP_THROW(TypeError, "diag: unsupported dtype");
            }
#undef TP_DGE
            CUDA_CHECK(cudaGetLastError());
        }
        return out;
    }
    TP_THROW(RuntimeError, "diag: input must be 1-D or 2-D");
}

Tensor diag_embed_cuda(const Tensor& self, int64_t offset, int64_t dim1_, int64_t dim2_) {
    // Host-staged reference implementation (rare op), mirrors the structure
    // of TensorShape.cpp:1272.
    int64_t nDims = self.dim() + 1;
    int64_t dim1 = wrap_dim(dim1_, nDims);
    int64_t dim2 = wrap_dim(dim2_, nDims);
    if (dim1 == dim2) TP_THROW(RuntimeError, "diagonal dimensions cannot be identical");
    int64_t new_dim_len = std::abs(offset) + self.size(-1);
    std::vector<int64_t> sizes = shape_of(self);
    sizes.pop_back();
    sizes.insert(sizes.begin() + std::min(dim1, dim2), new_dim_len);
    sizes.insert(sizes.begin() + std::max(dim1, dim2), new_dim_len);
    Tensor rc = Tensor::zeros(sizes, self.dtype(), Device(DeviceType::CPU));
    Tensor sc = self.contiguous().to(Device(DeviceType::CPU));
    int64_t mid = std::max(dim1, dim2);
    int64_t lowdim = std::min(dim1, dim2);
    int64_t n = self.numel();
#define TP_DEW(ctype, name_) \
    case DType::name_: { \
        const ctype* s2 = sc.data_ptr<ctype>(); \
        ctype* d2 = rc.data_ptr<ctype>(); \
        for (int64_t li = 0; li < n; ++li) { \
            int64_t rem = li; \
            std::vector<int64_t> sc3(self.dim(), 0); \
            for (int64_t d3 = static_cast<int64_t>(self.dim()) - 1; d3 >= 0; --d3) { \
                sc3[d3] = rem % self.size(d3); rem /= self.size(d3); } \
            int64_t t = sc3.back(); \
            int64_t i = offset >= 0 ? t : t - offset; \
            int64_t j = offset >= 0 ? t + offset : t; \
            std::vector<int64_t> rc2(sizes.size(), 0); \
            int64_t sk = 0; \
            for (int64_t d3 = 0; d3 < static_cast<int64_t>(sizes.size()); ++d3) { \
                if (d3 == lowdim) rc2[d3] = i; \
                else if (d3 == mid) rc2[d3] = j; \
                else rc2[d3] = sc3[sk++]; } \
            int64_t lin = 0; \
            for (int64_t d3 = 0; d3 < static_cast<int64_t>(sizes.size()); ++d3) \
                lin = lin * sizes[d3] + rc2[d3]; \
            d2[lin] = s2[li]; \
        } \
        break; }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_DEW)
        default: TP_THROW(TypeError, "diag_embed: unsupported dtype");
    }
#undef TP_DEW
    return rc.to(self.device());
}

Tensor narrow_cuda(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
    // TensorShape.cpp narrow: a slice view with torch's exact checks.
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "narrow() cannot be applied to a 0-dim tensor.");
    }
    if (length < 0) {
        TP_THROW(RuntimeError, "narrow(): length must be non-negative.");
    }
    dim = wrap_dim(dim, self.dim());
    const int64_t cur_size = self.size(dim);
    if (start < -cur_size || start > cur_size) {
        TP_THROW(IndexError, "start out of range (expected to be in range of [",
                 -cur_size, ", ", cur_size, "], but got ", start, ")");
    }
    if (start < 0) start += cur_size;
    if (start > cur_size - length) {
        TP_THROW(RuntimeError, "start (", start, ") + length (", length,
                 ") exceeds dimension size (", cur_size, ").");
    }
    return self.slice(dim, start, start + length, 1);
}

std::vector<Tensor> split_with_sizes_cuda(const Tensor& self, std::vector<int64_t> split_sizes,
                                          int64_t dim) {
    // Torch split_with_sizes (TensorShape.cpp).
    if (self.dim() == 0) {
        TP_THROW(RuntimeError, "split expects at least a 1-dimensional tensor");
    }
    const int64_t nd = self.dim();
    if (dim < -nd || dim >= nd) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 -nd, ", ", nd - 1, "], but got ", dim, ")");
    }
    if (dim < 0) dim += nd;
    const int64_t dim_size = self.size(dim);
    std::vector<Tensor> outs;
    outs.reserve(split_sizes.size());
    int64_t start = 0;
    for (const int64_t len : split_sizes) {
        if (len < 0) {
            TP_THROW(RuntimeError, "split_with_sizes expects split_sizes have only non-negative "
                     "entries, but got split_sizes=[", [&] {
                         std::string s;
                         for (size_t i = 0; i < split_sizes.size(); ++i) {
                             if (i) s += ", ";
                             s += std::to_string(split_sizes[i]);
                         }
                         return s;
                     }(), "]");
        }
        outs.push_back(self.slice(dim, start, start + len));
        start += len;
    }
    if (start != dim_size) {
        TP_THROW(RuntimeError, "split_with_sizes expects split_sizes to sum exactly to ",
                 dim_size, " (input tensor's size at dimension ", dim, "), but got split_sizes=[",
                 [&] {
                     std::string s;
                     for (size_t i = 0; i < split_sizes.size(); ++i) {
                         if (i) s += ", ";
                         s += std::to_string(split_sizes[i]);
                     }
                     return s;
                 }(), "]");
    }
    return outs;
}

std::vector<Tensor> tensor_split_cuda(const Tensor& self, int64_t sections, int64_t dim) {
    int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (sections <= 0) TP_THROW(RuntimeError, "tensor_split: number of sections must be larger than 0");
    int64_t size = self.size(dim);
    int64_t base = size / sections, rem = size % sections;
    std::vector<Tensor> outs;
    int64_t start = 0;
    for (int64_t i = 0; i < sections; ++i) {
        int64_t len = base + (i < rem ? 1 : 0);
        if (len > 0) outs.push_back(narrow_cuda(self, dim, start, len));
        else outs.emplace_back();
        start += len;
    }
    return outs;
}

Tensor flip_cuda(const Tensor& self, const std::vector<int64_t>& dims) {
    // TensorTransformations.cpp:36 flip: dim_list_to_bitset (WrapDimUtilsMulti.h)
    // wraps with wrap_scalar=true and rejects duplicate dims.
    int64_t nd = self.dim();
    std::vector<bool> seen(nd > 0 ? nd : 1, false);
    std::vector<int64_t> h_flips(nd, 0);
    for (auto d : dims) {
        int64_t w = wrap_dim_scalar(d, nd);
        if (nd > 0) {
            if (seen[w]) {
                TP_THROW(RuntimeError, "dim ", w,
                         " appears multiple times in the list of dims");
            }
            seen[w] = true;
            h_flips[w] = 1;
        }
    }
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(sc), sc.dtype(), sc.device());
    int64_t n = sc.numel();
    if (n == 0) return out;
    Tensor d_sizes = pack_i64(shape_of(sc), sc.device());
    Tensor d_flips = pack_i64(h_flips, sc.device());
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(n), block(kThreads);
#define TP_FL2(ctype, name_) \
    case DType::name_: \
        flip_map_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, nd, sc.data_ptr<ctype>(), out.data_ptr<ctype>(), \
            d_sizes.data_ptr<int64_t>(), d_flips.data_ptr<int64_t>()); \
        break;
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_FL2)
        default: TP_THROW(TypeError, "flip: unsupported dtype");
    }
#undef TP_FL2
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor roll_cuda(const Tensor& self, const std::vector<int64_t>& shifts, const std::vector<int64_t>& dims) {
    // TensorTransformations.cpp:110 roll + TensorTransformations.h roll_common.
    if (dims.size() != 1 || shifts.size() != 1) {
        if (shifts.empty()) TP_THROW(RuntimeError, "`shifts` required");
        if (dims.empty() && shifts.size() == 1) {
            // Flatten-roll: roll the flattened tensor and view back.
            Tensor flat = self.contiguous().reshape({self.numel()});
            Tensor rolled = roll_cuda(flat, {shifts[0]}, {0});
            return rolled.reshape(shape_of(self));
        }
        if (shifts.size() != dims.size()) {
            TP_THROW(RuntimeError, "shifts and dimensions must align. shifts: ",
                     shifts.size(), ", dims:", dims.size());
        }
        Tensor cur = self;
        for (size_t i = 0; i < dims.size(); ++i) {
            cur = roll_cuda(cur, {shifts[i]}, {dims[i]});
        }
        return cur;
    }
    // Avoid a div zero error below (upstream comment); empty input rolls to
    // itself.
    if (self.numel() == 0) return self.clone();
    const int64_t nd = self.dim();
    if (nd == 0) {
        // torch reaches size(dim) on the 0-d tensor: maybe_wrap_dim with
        // wrap_scalar=false rejects any dim.
        TP_THROW(IndexError, "Dimension specified as ", dims[0],
                 " but tensor has no dimensions");
    }
    const int64_t dim = wrap_dim(dims[0], nd);
    const int64_t size = self.size(dim);
    // roll_map_kernel reads src[c - sh] so a normalized positive shift keeps
    // the mapping identical to cat({narrow(start), narrow(0, start)}).
    std::vector<int64_t> sh(nd, 0);
    sh[dim] = ((shifts[0] % size) + size) % size;
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(sc), sc.dtype(), sc.device());
    int64_t n = sc.numel();
    Tensor d_sizes = pack_i64(shape_of(sc), sc.device());
    Tensor d_sh = pack_i64(sh, sc.device());
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(n), block(kThreads);
#define TP_RL2(ctype, name_) \
    case DType::name_: \
        roll_map_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, nd, sc.data_ptr<ctype>(), out.data_ptr<ctype>(), \
            d_sizes.data_ptr<int64_t>(), d_sh.data_ptr<int64_t>()); \
        break;
    switch (sc.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_RL2)
        default: TP_THROW(TypeError, "roll: unsupported dtype");
    }
#undef TP_RL2
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor rot90_cuda(const Tensor& self, int64_t k, const std::vector<int64_t>& dims) {
    // TensorTransformations.cpp:127 rot90.
    const int64_t total_dims = self.dim();
    const int64_t total_rot_dims = static_cast<int64_t>(dims.size());
    if (total_rot_dims != 2) {
        TP_THROW(RuntimeError, "expected total rotation dims == 2, but got dims = ",
                 total_rot_dims);
    }
    if (total_dims < 2) {
        TP_THROW(RuntimeError, "expected total dims >= 2, but got total dims = ",
                 total_dims);
    }
    // Validate range first so out-of-range dims raise IndexError, then
    // normalize before checking for duplicates (e.g. [1, -1] on a 2D tensor).
    const int64_t dim0 = wrap_dim(dims[0], total_dims);
    const int64_t dim1 = wrap_dim(dims[1], total_dims);
    if (dim0 == dim1) {
        TP_THROW(RuntimeError, "expected rotation dims to be different, but got dim0 = ",
                 dims[0], " and dim1 = ", dims[1]);
    }
    // handle modulo with negative k
    k = (4 + (k % 4)) % 4;
    // transpose_ on the fresh flip result: a view with swapped sizes/strides.
    auto transpose_view = [](const Tensor& x, int64_t a, int64_t b) {
        std::vector<int64_t> sizes(x.dim()), strides(x.dim());
        for (int64_t i = 0; i < x.dim(); ++i) {
            sizes[i] = x.size(i);
            strides[i] = x.stride(i);
        }
        std::swap(sizes[a], sizes[b]);
        std::swap(strides[a], strides[b]);
        return x.as_strided(sizes, strides);
    };
    switch (k) {
        case 1: return transpose_view(flip_cuda(self, {dim1}), dim0, dim1);
        case 2: return flip_cuda(self, {dim0, dim1});
        case 3: return transpose_view(flip_cuda(self, {dim0}), dim0, dim1);
        default: return detail::contiguous_clone(self);
    }
}

Tensor repeat_interleave_cuda(const Tensor& self, int64_t repeats, int64_t dim) {
    int64_t nd = self.dim();
    if (nd == 0) TP_THROW(RuntimeError, "repeat_interleave: dimension required for scalar");
    dim = wrap_dim(dim, nd);
    if (repeats < 0) TP_THROW(RuntimeError, "repeat_interleave: repeats can not be negative");
    std::vector<int64_t> out_shape = shape_of(self);
    out_shape[dim] *= repeats;
    Tensor out = Tensor::empty(out_shape, self.dtype(), self.device());
    int64_t d_size = self.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(shape_of(self), dim, outer, inner);
    Tensor sc = self.contiguous();
    int64_t out_d = out_shape[dim];
    int64_t total_rows = outer * out_d;
    if (total_rows == 0 || inner == 0) return out;
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(total_rows * inner), block(kThreads);
#define TP_RI4(ctype, name_) \
    case DType::name_: \
        ew_unary_index_copy<ctype><<<grid, block, 0, stream>>>( \
            total_rows, inner, out_d, d_size, repeats, sc.data_ptr<ctype>(), \
            out.data_ptr<ctype>()); \
        break;
    (void)0;
#undef TP_RI4
#define TP_RI5(ctype, name_) \
    case DType::name_: \
        repeat_interleave_kernel<ctype><<<grid, block, 0, stream>>>( \
            total_rows, inner, out_d, d_size, repeats, sc.data_ptr<ctype>(), \
            out.data_ptr<ctype>()); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_RI5)
        default: TP_THROW(TypeError, "repeat_interleave: unsupported dtype");
    }
#undef TP_RI5
    CUDA_CHECK(cudaGetLastError());
    return out;
}

std::vector<Tensor> meshgrid_cuda(const std::vector<Tensor>& tensors, const std::string& indexing) {
    // ij-indexing semantics.
    size_t k = tensors.size();
    if (k == 0) return {};
    std::vector<int64_t> sizes;
    sizes.reserve(k);
    for (auto& t : tensors) sizes.push_back(static_cast<int64_t>(t.numel()));
    int64_t total = 1;
    for (int64_t s2 : sizes) total *= s2;
    Tensor d_sizes = pack_i64(sizes, tensors[0].device());
    std::vector<Tensor> outs;
    auto stream = getCurrentCUDAStream().stream();
    for (size_t j = 0; j < k; ++j) {
        Tensor g = Tensor::empty(sizes, DType::Int64, tensors[0].device());
        if (total > 0) {
            dim3 grid = make_grid(total), block(kThreads);
            meshgrid_coord_kernel<<<grid, block, 0, stream>>>(
                total, k, j, d_sizes.data_ptr<int64_t>(), g.data_ptr<int64_t>());
            CUDA_CHECK(cudaGetLastError());
        }
        outs.push_back(g);
    }
    return outs;
}

std::vector<Tensor> broadcast_tensors_cuda(const std::vector<Tensor>& tensors) {
    // CompositeImplicit mirror of broadcast_tensors_cpu: device-generic
    // expand views; no CUDA-specific code required.
    std::vector<int64_t> shape;
    for (auto& t : tensors) shape = broadcast_shapes(shape, shape_of(t));
    std::vector<Tensor> outs;
    outs.reserve(tensors.size());
    for (auto& t : tensors) {
        std::vector<int64_t> ts = shape_of(t);
        outs.push_back(ts == shape ? t : t.expand(shape));
    }
    return outs;
}

Tensor block_diag_cuda(const std::vector<Tensor>& tensors) {
    // CompositeImplicit mirror of block_diag_cpu (device-generic members).
    if (tensors.empty()) return Tensor::empty({1, 0}, DType::Float32);
    const Device& device = tensors[0].device();
    DType out_dtype = tensors[0].dtype();
    int64_t rows = 0, cols = 0;
    std::vector<Tensor> blocks2d;
    blocks2d.reserve(tensors.size());
    for (size_t idx = 0; idx < tensors.size(); ++idx) {
        const Tensor& t = tensors[idx];
        if (!(t.device() == device)) {
            TP_THROW(RuntimeError,
                     "torch.block_diag: input tensors must all be on the same device.");
        }
        out_dtype = promoteTypes(out_dtype, t.dtype());
        const int64_t nd = t.dim();
        if (nd > 2) {
            TP_THROW(RuntimeError,
                     "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input ",
                     static_cast<int64_t>(idx), " has ", nd, " dimensions");
        }
        Tensor b2 = t;
        if (nd == 1) b2 = t.expand({1, t.size(0)});
        else if (nd == 0) b2 = t.expand({1, 1});
        blocks2d.push_back(b2);
        rows += b2.size(0);
        cols += b2.size(1);
    }
    Tensor out = Tensor::zeros({rows, cols}, out_dtype, device);
    int64_t off0 = 0, off1 = 0;
    for (const auto& b : blocks2d) {
        out.slice(0, off0, off0 + b.size(0))
           .slice(1, off1, off1 + b.size(1))
           .copy_(b);
        off0 += b.size(0);
        off1 += b.size(1);
    }
    return out;
}

Tensor pixel_shuffle_cuda(const Tensor& self, int64_t upscale_factor) {
    // PixelShuffle.cpp:23 semantics: (N, C*r^2, H, W) -> (N, C, H*r, W*r)
    if (self.dim() != 4) TP_THROW(RuntimeError, "pixel_shuffle expects 4D input");
    int64_t r = upscale_factor;
    int64_t N = self.size(0);
    int64_t C = self.size(1) / (r * r);
    int64_t H = self.size(2), W = self.size(3);
    if (C * r * r != self.size(1))
        TP_THROW(RuntimeError, "pixel_shuffle: channel dim must be divisible by r^2");
    Tensor out = Tensor::empty({N, C, H * r, W * r}, self.dtype(), self.device());
    Tensor sc = self.contiguous();
    int64_t n = out.numel();
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(n), block(kThreads);
#define TP_PS(ctype, name_) \
    case DType::name_: \
        pixel_shuffle_map_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, C, H, W, r, sc.data_ptr<ctype>(), out.data_ptr<ctype>()); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_PS)
        default: TP_THROW(TypeError, "pixel_shuffle: unsupported dtype");
    }
#undef TP_PS
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor pixel_unshuffle_cuda(const Tensor& self, int64_t downscale_factor) {
    if (self.dim() != 4) TP_THROW(RuntimeError, "pixel_unshuffle expects 4D input");
    int64_t r = downscale_factor;
    int64_t N = self.size(0);
    int64_t C = self.size(1);
    int64_t H = self.size(2) / r, W = self.size(3) / r;
    if (H * r != self.size(2) || W * r != self.size(3))
        TP_THROW(RuntimeError, "pixel_unshuffle: spatial dims must be divisible by r");
    Tensor out = Tensor::empty({N, C * r * r, H, W}, self.dtype(), self.device());
    Tensor sc = self.contiguous();
    int64_t n = out.numel();
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(n), block(kThreads);
#define TP_PU(ctype, name_) \
    case DType::name_: \
        pixel_unshuffle_map_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, C, H, W, r, sc.data_ptr<ctype>(), out.data_ptr<ctype>()); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_PU)
        default: TP_THROW(TypeError, "pixel_unshuffle: unsupported dtype");
    }
#undef TP_PU
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor channel_shuffle_cuda(const Tensor& self, int64_t groups) {
    // ChannelShuffle: view(N, g, C/g, ...) -> transpose(1,2). Channel dim is
    // dim 1 for >=2D input, dim 0 for 1-D input.
    if (self.dim() < 1) TP_THROW(RuntimeError, "channel_shuffle expects >= 1D input");
    int64_t cdim = self.dim() >= 2 ? 1 : 0;
    int64_t C = self.size(cdim);
    int64_t outer = 1;
    for (int64_t i = 0; i < cdim; ++i) outer *= self.size(i);
    int64_t inner = 1;
    for (int64_t i = cdim + 1; i < self.dim(); ++i) inner *= self.size(i);
    if (C % groups) TP_THROW(RuntimeError, "channel_shuffle: channel dim not divisible by groups");
    int64_t cg = C / groups;
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    int64_t n = self.numel();
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(n), block(kThreads);
#define TP_CS(ctype, name_) \
    case DType::name_: \
        channel_shuffle_map_kernel<ctype><<<grid, block, 0, stream>>>( \
            n, outer, C, inner, cg, sc.data_ptr<ctype>(), out.data_ptr<ctype>()); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_CS)
        default: TP_THROW(TypeError, "channel_shuffle: unsupported dtype");
    }
#undef TP_CS
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor unfold_cuda(const Tensor& self, int64_t dimension, int64_t size, int64_t step) {
    // TensorShape.cpp unfold: an as_strided view.  wrap_scalar=true allows
    // dimension == 0 on 0-d tensors (max_size becomes 1).
    const int64_t nd = self.dim();
    dimension = wrap_dim_scalar(dimension, nd);

    std::vector<int64_t> sizes = shape_of(self);
    std::vector<int64_t> strides = self.strides();
    const int64_t max_size = nd == 0 ? 1 : sizes[dimension];
    if (size < 0) TP_THROW(RuntimeError, "size is ", size, " but must be >= 0");
    if (size > max_size) {
        TP_THROW(RuntimeError, "maximum size for tensor at dimension ", dimension,
                 " is ", max_size, " but size is ", size);
    }
    if (step <= 0) TP_THROW(RuntimeError, "step is ", step, " but must be > 0");
    sizes.push_back(size);
    strides.push_back(nd == 0 ? 1 : strides[dimension]);
    // The if handles the self.dim() == 0 case
    if (dimension < nd) {
        sizes[dimension] = (sizes[dimension] - size) / step + 1;
        strides[dimension] *= step;
    }
    return self.as_strided(sizes, strides);
}

namespace {

template <typename T>
__global__ void unfold_backward_gather_kernel(int64_t total, int64_t input_dim_size,
                                              int64_t count, int64_t inner,
                                              int64_t size, int64_t step,
                                              const T* grad, T* grad_input) {
    // Gather over grad_input elements (race-free); each element accumulates the
    // folds whose window covers it (ATen cpu/UnfoldBackwardKernel.cpp).
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; t < total; t += stride) {
        int64_t inner_idx = t % inner;
        int64_t rest = t / inner;
        int64_t idx_dim = rest % input_dim_size;
        int64_t outer_idx = rest / input_dim_size;
        int64_t left = (idx_dim > size) ? (idx_dim - size) / step : 0;
        if (!(left * step <= idx_dim && idx_dim < left * step + size)) ++left;
        int64_t right = idx_dim / step;
        if (right >= count) right = count - 1;
        T acc{};
        for (int64_t fold = left; fold <= right; ++fold) {
            int64_t j = idx_dim - fold * step;
            acc += grad[((outer_idx * count + fold) * inner + inner_idx) * size + j];
        }
        grad_input[t] = acc;
    }
}

} // anonymous namespace

Tensor unfold_backward_cuda(const Tensor& grad, const std::vector<int64_t>& input_sizes,
                            int64_t dim, int64_t size, int64_t step) {
    // ATen UnfoldBackward.cpp: scatter each window's gradient back onto `dim`,
    // accumulating where windows overlap (step < size).
    if (step <= 0) TP_THROW(RuntimeError, "step is ", step, " but must be > 0");
    Tensor grad_input = Tensor::zeros(input_sizes, grad.dtype(), grad.device());
    const int64_t nd = static_cast<int64_t>(input_sizes.size());
    if (nd == 0) {
        if (size > 0) grad_input.copy_(grad.select(0, 0));
        return grad_input;
    }
    dim = wrap_dim(dim, nd);
    const int64_t input_dim_size = input_sizes[dim];
    const int64_t count = grad.size(dim);
    int64_t outer = 1, inner = 1;
    outer_inner(input_sizes, dim, outer, inner);
    Tensor gc = grad.contiguous();
    const int64_t total = outer * input_dim_size * inner;
    if (total == 0) return grad_input;
    auto stream = getCurrentCUDAStream().stream();
    dim3 grid = make_grid(total), block(kThreads);
#define TP_UFB(ctype, name_) \
    case DType::name_: \
        unfold_backward_gather_kernel<ctype><<<grid, block, 0, stream>>>( \
            total, input_dim_size, count, inner, size, step, \
            gc.data_ptr<ctype>(), grad_input.data_ptr<ctype>()); \
        break;
    switch (grad.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_UFB)
        default: TP_THROW(TypeError, "unfold_backward: unsupported dtype");
    }
#undef TP_UFB
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, TierReduceOpsKernels) {
    m.impl("amax", amax_cuda2);
    m.impl("amin", amin_cuda2);
    m.impl("aminmax", aminmax_cuda);
    m.impl("logsumexp", logsumexp_cuda2);
    m.impl("nansum", nansum_cuda2);
    m.impl("nanmedian", nanmedian_cuda);
    m.impl("count_nonzero", count_nonzero_cuda2);
    m.impl("cummax", cummax_cuda);
    m.impl("cummin", cummin_cuda);
    m.impl("var_mean", var_mean_cuda);
    m.impl("std_mean", std_mean_cuda);
    m.impl("mode", mode_cuda);
    m.impl("kthvalue", kthvalue_cuda);
    m.impl("dist", dist_cuda);
    m.impl("renorm", renorm_cuda);
    m.impl("trace", trace_cuda);
    m.impl("diag", diag_cuda);
    m.impl("diag_embed", diag_embed_cuda);
    m.impl("narrow", narrow_cuda);
    m.impl("split_with_sizes", split_with_sizes_cuda);
    // tensor_split.sections: owned by cpu/ShapeAlignKernels.cpp (torch-exact
    // view semantics + indices/tensor overloads); duplicate removed.
    m.impl("flip", flip_cuda);
    m.impl("roll", roll_cuda);
    m.impl("rot90", rot90_cuda);
    m.impl("repeat_interleave.self_int", repeat_interleave_cuda);
    m.impl("meshgrid", meshgrid_cuda);
    m.impl("broadcast_tensors", broadcast_tensors_cuda);
    m.impl("block_diag", block_diag_cuda);
    m.impl("pixel_shuffle", pixel_shuffle_cuda);
    m.impl("pixel_unshuffle", pixel_unshuffle_cuda);
    m.impl("channel_shuffle", channel_shuffle_cuda);
    m.impl("unfold", unfold_cuda);
    m.impl("unfold_backward", unfold_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay
