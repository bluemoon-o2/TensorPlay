// Tier 5 operators - CUDA side.
//
// Strategy: linear-algebra decompositions (cholesky family / triangular_solve
// / svd), pdist/pairwise_distance and the RNN cells are HOST-STAGED through
// the CPU reference implementations (extern-linked from cpu/TierOpsKernels /
// cpu/Tier5OpsKernels). This guarantees identical semantics while the
// high-throughput paths remain on CPU-backed LAPACK-free reference kernels.
// Elementwise-friendly factories run natively on device.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

// CPU reference implementations (cpu/Tier5OpsKernels.cpp).
namespace cpu {
Tensor addbmm_cpu(const Tensor&, const Tensor&, const Tensor&, Scalar, Scalar);
Tensor addmv_cpu(const Tensor&, const Tensor&, const Tensor&, Scalar, Scalar);
Tensor addr_cpu(const Tensor&, const Tensor&, const Tensor&, Scalar, Scalar);
Tensor vdot_cpu(const Tensor&, const Tensor&);
Tensor cholesky_cpu(const Tensor&, bool);
Tensor cholesky_inverse_cpu(const Tensor&, bool);
Tensor cholesky_solve_cpu(const Tensor&, const Tensor&, bool);
std::tuple<Tensor, Tensor> triangular_solve_cpu(const Tensor&, const Tensor&, bool, bool, bool);
std::tuple<Tensor, Tensor, Tensor> svd_cpu(const Tensor&, bool, bool);
Tensor pairwise_distance_cpu(const Tensor&, const Tensor&, double, double, bool);
Tensor pdist_cpu(const Tensor&, double);
Tensor hinge_embedding_loss_cpu(const Tensor&, const Tensor&, Scalar);
Tensor margin_ranking_loss_cpu(const Tensor&, const Tensor&, const Tensor&, Scalar);
std::tuple<Tensor, Tensor, Tensor> lstm_cpu(const Tensor&, const std::vector<Tensor>&,
                                            const std::vector<Tensor>&, bool, int64_t,
                                            float, bool, bool, bool);
std::tuple<Tensor, Tensor> gru_cpu(const Tensor&, const std::vector<Tensor>&,
                                   const std::vector<Tensor>&, bool, int64_t,
                                   float, bool, bool, bool);
std::tuple<Tensor, Tensor> rnn_relu_cpu(const Tensor&, const std::vector<Tensor>&,
                                        const std::vector<Tensor>&, bool, int64_t,
                                        float, bool, bool, bool);
std::tuple<Tensor, Tensor> rnn_tanh_cpu(const Tensor&, const std::vector<Tensor>&,
                                        const std::vector<Tensor>&, bool, int64_t,
                                        float, bool, bool, bool);
} // namespace cpu

namespace {

constexpr int kThreads = 256;

inline dim3 make_grid(int64_t work) {
    return dim3(static_cast<unsigned>((work + kThreads - 1) / kThreads));
}

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

inline Tensor to_host(const Tensor& t) { return t.to(Device(DeviceType::CPU)); }
inline Tensor to_device(const Tensor& t, const Device& d) { return t.to(d); }

template <typename F>
Tensor staged1(const Tensor& a, F&& fn) {
    Device dev = a.device();
    return to_device(fn(to_host(a)), dev);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Native device kernels (window / complex helpers)
// ---------------------------------------------------------------------------

namespace {

__global__ void hann_window_kernel(int64_t n, int64_t denom, double two_pi,
                                   float* fp, double* dp) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        double v = 0.5 * (1.0 - ::cos(two_pi * i / static_cast<double>(denom)));
        if (fp) fp[i] = static_cast<float>(v);
        if (dp) dp[i] = v;
    }
}

template <typename CT, typename RT>
__global__ void cplx_real_kernel(int64_t n, const CT* src, RT* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) dst[i] = src[i].real();
}
template <typename CT, typename RT>
__global__ void cplx_imag_kernel(int64_t n, const CT* src, RT* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) dst[i] = src[i].imag();
}
template <typename CT>
__global__ void cplx_conj_kernel(int64_t n, CT* data) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) data[i] = std::conj(data[i]);
}
template <typename CT, typename RT>
__global__ void cplx_pack_kernel(int64_t n, const RT* re, const RT* im, CT* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) dst[i] = CT(re[i], im[i]);
}
template <typename CT, typename RT>
__global__ void polar_kernel(int64_t n, const RT* ab, const RT* ang, CT* dst) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) dst[i] = std::polar(ab[i], ang[i]);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

Tensor addbmm_cuda(const Tensor& self, const Tensor& batch1, const Tensor& batch2,
                   Scalar beta, Scalar alpha) {
    Device dev = self.device();
    return to_device(cpu::addbmm_cpu(to_host(self), to_host(batch1), to_host(batch2),
                                     beta, alpha),
                     dev);
}
Tensor addmv_cuda(const Tensor& self, const Tensor& mat, const Tensor& vec,
                  Scalar beta, Scalar alpha) {
    Device dev = self.device();
    return to_device(cpu::addmv_cpu(to_host(self), to_host(mat), to_host(vec), beta, alpha),
                     dev);
}
Tensor addr_cuda(const Tensor& self, const Tensor& vec1, const Tensor& vec2,
                 Scalar beta, Scalar alpha) {
    Device dev = self.device();
    return to_device(cpu::addr_cpu(to_host(self), to_host(vec1), to_host(vec2), beta, alpha),
                     dev);
}
Tensor vdot_cuda(const Tensor& a, const Tensor& b) {
    Device dev = a.device();
    return to_device(cpu::vdot_cpu(to_host(a), to_host(b)), dev);
}
Tensor cholesky_cuda(const Tensor& self, bool upper) {
    Device dev = self.device();
    return to_device(cpu::cholesky_cpu(to_host(self), upper), dev);
}
Tensor cholesky_inverse_cuda(const Tensor& self, bool upper) {
    Device dev = self.device();
    return to_device(cpu::cholesky_inverse_cpu(to_host(self), upper), dev);
}
Tensor cholesky_solve_cuda(const Tensor& self, const Tensor& input2, bool upper) {
    Device dev = self.device();
    return to_device(cpu::cholesky_solve_cpu(to_host(self), to_host(input2), upper), dev);
}
std::tuple<Tensor, Tensor> triangular_solve_cuda(const Tensor& self, const Tensor& A,
                                                 bool upper, bool transpose,
                                                 bool unitriangular) {
    Device dev = self.device();
    auto r = cpu::triangular_solve_cpu(to_host(self), to_host(A), upper, transpose,
                                       unitriangular);
    return {to_device(std::get<0>(r), dev), to_device(std::get<1>(r), dev)};
}
std::tuple<Tensor, Tensor, Tensor> svd_cuda(const Tensor& self, bool some, bool compute_uv) {
    Device dev = self.device();
    auto r = cpu::svd_cpu(to_host(self), some, compute_uv);
    return {to_device(std::get<0>(r), dev), to_device(std::get<1>(r), dev),
            to_device(std::get<2>(r), dev)};
}
Tensor pairwise_distance_cuda(const Tensor& x1, const Tensor& x2, double p, double eps,
                              bool keepdim) {
    Device dev = x1.device();
    return to_device(cpu::pairwise_distance_cpu(to_host(x1), to_host(x2), p, eps, keepdim),
                     dev);
}
Tensor pdist_cuda(const Tensor& self, double p) {
    Device dev = self.device();
    return to_device(cpu::pdist_cpu(to_host(self), p), dev);
}
std::tuple<Tensor, Tensor, Tensor> lstm_cuda(const Tensor& input,
                                             const std::vector<Tensor>& hx,
                                             const std::vector<Tensor>& params,
                                             bool has_biases, int64_t num_layers,
                                             float dropout_p, bool training,
                                             bool bidirectional, bool batch_first) {
    Device dev = input.device();
    auto r = cpu::lstm_cpu(to_host(input), hx, params, has_biases, num_layers, dropout_p,
                           training, bidirectional, batch_first);
    return {to_device(std::get<0>(r), dev), to_device(std::get<1>(r), dev),
            to_device(std::get<2>(r), dev)};
}
std::tuple<Tensor, Tensor> gru_cuda(const Tensor& input, const std::vector<Tensor>& hx,
                                    const std::vector<Tensor>& params, bool has_biases,
                                    int64_t num_layers, float dropout_p, bool training,
                                    bool bidirectional, bool batch_first) {
    Device dev = input.device();
    auto r = cpu::gru_cpu(to_host(input), hx, params, has_biases, num_layers, dropout_p,
                          training, bidirectional, batch_first);
    return {to_device(std::get<0>(r), dev), to_device(std::get<1>(r), dev)};
}
std::tuple<Tensor, Tensor> rnn_relu_cuda(const Tensor& input, const std::vector<Tensor>& hx,
                                         const std::vector<Tensor>& params, bool has_biases,
                                         int64_t num_layers, float dropout_p, bool training,
                                         bool bidirectional, bool batch_first) {
    Device dev = input.device();
    auto r = cpu::rnn_relu_cpu(to_host(input), hx, params, has_biases, num_layers,
                               dropout_p, training, bidirectional, batch_first);
    return {to_device(std::get<0>(r), dev), to_device(std::get<1>(r), dev)};
}
std::tuple<Tensor, Tensor> rnn_tanh_cuda(const Tensor& input, const std::vector<Tensor>& hx,
                                         const std::vector<Tensor>& params, bool has_biases,
                                         int64_t num_layers, float dropout_p, bool training,
                                         bool bidirectional, bool batch_first) {
    Device dev = input.device();
    auto r = cpu::rnn_tanh_cpu(to_host(input), hx, params, has_biases, num_layers,
                               dropout_p, training, bidirectional, batch_first);
    return {to_device(std::get<0>(r), dev), to_device(std::get<1>(r), dev)};
}

Tensor hann_window_cuda(int64_t window_length, bool periodic, std::optional<DType> dtype) {
    DType dt = dtype.value_or(DType::Float32);
    if (!isFloatingType(dt)) dt = DType::Float32;
    Tensor out = Tensor::empty({window_length}, dt, Device(DeviceType::CUDA));
    int64_t n = window_length;
    if (n == 0) return out;
    int64_t denom = periodic ? window_length : window_length - 1;
    if (denom <= 0) denom = 1;
    auto stream = getCurrentCUDAStream();
    dim3 grid = make_grid(n), block(kThreads);
    if (dt == DType::Float32) {
        hann_window_kernel<<<grid, block, 0, stream.stream()>>>
            n, denom, 2.0 * M_PI, out.data_ptr<float>(), nullptr);
    } else {
        hann_window_kernel<<<grid, block, 0, stream.stream()>>>
            n, denom, 2.0 * M_PI, nullptr, out.data_ptr<double>());
    }
    CUDA_CHECK(cudaGetLastError());
    return out;
}

namespace {

bool is_cplx(DType d) { return d == DType::ComplexFloat || d == DType::ComplexDouble; }

} // anonymous namespace

Tensor real_cuda(const Tensor& self) {
    if (!is_cplx(self.dtype())) return self.clone();
    DType rt = self.dtype() == DType::ComplexDouble ? DType::Float64 : DType::Float32;
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(sc), rt, self.device());
    int64_t n = sc.numel();
    auto stream = getCurrentCUDAStream();
    dim3 grid = make_grid(n), block(kThreads);
    if (self.dtype() == DType::ComplexFloat)
        cplx_real_kernel<std::complex<float>, float><<<grid, block, 0, stream.stream()>>>
            n, sc.data_ptr<std::complex<float>>(), out.data_ptr<float>());
    else
        cplx_real_kernel<std::complex<double>, double><<<grid, block, 0, stream.stream()>>>
            n, sc.data_ptr<std::complex<double>>(), out.data_ptr<double>());
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor imag_cuda(const Tensor& self) {
    if (!is_cplx(self.dtype())) {
        return Tensor::zeros(shape_of(self), self.dtype(), self.device());
    }
    DType rt = self.dtype() == DType::ComplexDouble ? DType::Float64 : DType::Float32;
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(shape_of(sc), rt, self.device());
    int64_t n = sc.numel();
    auto stream = getCurrentCUDAStream();
    dim3 grid = make_grid(n), block(kThreads);
    if (self.dtype() == DType::ComplexFloat)
        cplx_imag_kernel<std::complex<float>, float><<<grid, block, 0, stream.stream()>>>
            n, sc.data_ptr<std::complex<float>>(), out.data_ptr<float>());
    else
        cplx_imag_kernel<std::complex<double>, double><<<grid, block, 0, stream.stream()>>>
            n, sc.data_ptr<std::complex<double>>(), out.data_ptr<double>());
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor conj_cuda(const Tensor& self) {
    if (!is_cplx(self.dtype())) return self.clone();
    Tensor out = self.clone();
    int64_t n = out.numel();
    if (n == 0) return out;
    auto stream = getCurrentCUDAStream();
    dim3 grid = make_grid(n), block(kThreads);
    if (self.dtype() == DType::ComplexFloat)
        cplx_conj_kernel<std::complex<float>><<<grid, block, 0, stream.stream()>>>
            n, out.data_ptr<std::complex<float>>());
    else
        cplx_conj_kernel<std::complex<double>><<<grid, block, 0, stream.stream()>>>
            n, out.data_ptr<std::complex<double>>());
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor complex_cuda(const Tensor& re, const Tensor& im) {
    DType fdt = promoteTypes(re.dtype(), im.dtype());
    DType cdt = fdt == DType::Float64 ? DType::ComplexDouble : DType::ComplexFloat;
    std::vector<int64_t> shape = broadcast_shapes(shape_of(re), shape_of(im));
    Tensor rc = re.expand(shape).contiguous().to(fdt == DType::Float64 ? DType::Float64
                                                                       : DType::Float32);
    Tensor ic = im.expand(shape).contiguous().to(rc.dtype());
    Tensor out = Tensor::empty(shape, cdt, re.device());
    int64_t n = out.numel();
    if (n == 0) return out;
    auto stream = getCurrentCUDAStream();
    dim3 grid = make_grid(n), block(kThreads);
    if (cdt == DType::ComplexFloat)
        cplx_pack_kernel<std::complex<float>, float><<<grid, block, 0, stream.stream()>>>
            n, rc.data_ptr<float>(), ic.data_ptr<float>(),
            out.data_ptr<std::complex<float>>());
    else
        cplx_pack_kernel<std::complex<double>, double><<<grid, block, 0, stream.stream()>>>
            n, rc.data_ptr<double>(), ic.data_ptr<double>(),
            out.data_ptr<std::complex<double>>());
    CUDA_CHECK(cudaGetLastError());
    return out;
}

Tensor polar_cuda(const Tensor& abs_, const Tensor& angle_) {
    DType fdt = promoteTypes(abs_.dtype(), angle_.dtype());
    if (fdt != DType::Float64) fdt = DType::Float32;
    DType cdt = fdt == DType::Float64 ? DType::ComplexDouble : DType::ComplexFloat;
    std::vector<int64_t> shape = broadcast_shapes(shape_of(abs_), shape_of(angle_));
    Tensor a = abs_.expand(shape).contiguous().to(fdt);
    Tensor th = angle_.expand(shape).contiguous().to(fdt);
    Tensor out = Tensor::empty(shape, cdt, abs_.device());
    int64_t n = out.numel();
    if (n == 0) return out;
    auto stream = getCurrentCUDAStream();
    dim3 grid = make_grid(n), block(kThreads);
    if (fdt == DType::Float64)
        polar_kernel<std::complex<double>, double><<<grid, block, 0, stream.stream()>>>
            n, a.data_ptr<double>(), th.data_ptr<double>(),
            out.data_ptr<std::complex<double>>());
    else
        polar_kernel<std::complex<float>, float><<<grid, block, 0, stream.stream()>>>
            n, a.data_ptr<float>(), th.data_ptr<float>(),
            out.data_ptr<std::complex<float>>());
    CUDA_CHECK(cudaGetLastError());
    return out;
}


TENSORPLAY_LIBRARY_IMPL(CUDA, Tier5OpsKernels) {
    m.impl("addbmm", addbmm_cuda);
    m.impl("addmv", addmv_cuda);
    m.impl("addr", addr_cuda);
    m.impl("vdot", vdot_cuda);
    m.impl("cholesky", cholesky_cuda);
    m.impl("cholesky_inverse", cholesky_inverse_cuda);
    m.impl("cholesky_solve", cholesky_solve_cuda);
    m.impl("triangular_solve", triangular_solve_cuda);
    m.impl("svd", svd_cuda);
    m.impl("pairwise_distance", pairwise_distance_cuda);
    m.impl("pdist", pdist_cuda);
    m.impl("lstm", lstm_cuda);
    m.impl("gru", gru_cuda);
    m.impl("rnn_relu", rnn_relu_cuda);
    m.impl("rnn_tanh", rnn_tanh_cuda);
    m.impl("hann_window", hann_window_cuda);
    m.impl("real", real_cuda);
    m.impl("imag", imag_cuda);
    m.impl("conj", conj_cuda);
    m.impl("complex", complex_cuda);
    m.impl("polar", polar_cuda);
}

} // namespace cuda
} // namespace tensorplay
