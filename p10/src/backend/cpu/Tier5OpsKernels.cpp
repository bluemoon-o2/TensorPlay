// Tier 5 operators - CPU kernels: linear algebra (addbmm/addmv/addr/vdot/
// cholesky family/triangular_solve/svd/pdist/pairwise_distance), mean-reduced
// losses (binary_cross_entropy_with_logits / hinge_embedding_loss /
// margin_ranking_loss), RNN cells (lstm/gru/rnn_relu/rnn_tanh) and
// window/complex factories (hann_window/real/imag/conj/complex/polar).
//
// ATen anchors (third_party/pytorch 2.15.0a0):
//   Blas.cpp addbmm/addmv/addr; LinearAlgebra.cpp triangular_solve;
//   BatchLinearAlgebra.cpp cholesky_*; DistanceKernels.cpp pdist/cdist;
//   Loss.cpp hinge_embedding_loss / margin_ranking_loss;
//   LossNLL2d/BCE: binary_cross_entropy_with_logits stable form
//     softplus(y) = max(y,0) + log1p(exp(-|y|)),
//     l = pw*t*softplus(-x) + (1-t)*softplus(x)   [LossBCE2d.cpp];
//   Rnn.cpp param layout per direction [w_ih, w_hh, b_ih?, b_hh?],
//     gate order i,f,g,o for LSTM / i,r,n for GRU;
//   TensorFactories.cpp hann_window; ComplexHelper complex/real/imag/conj.
//
// Deviation note: svd uses a self-contained one-sided Jacobi iteration
// (always "reduced") instead of LAPACK gesdd.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Utils.h"
#include "Exception.h"
#include "Parallel.h"
#include "TypePromotion.h"
#include "tensorplay/ops/TensorRedispatchGenerated.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <complex>
#include <utility>

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

// Defined in LinearAlgebraKernels.cpp:301.
Tensor mm_kernel(const Tensor& self, const Tensor& mat2);

namespace {

inline void require_float(const Tensor& t, const char* who) {
    if (!isFloatingType(t.dtype()))
        TP_THROW(TypeError, who, ": only floating-point tensors are supported");
}

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) TP_THROW(RuntimeError, "Dimension out of range");
    return dim;
}

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

inline bool is_cplx(DType d) {
    return d == DType::ComplexFloat || d == DType::ComplexDouble;
}

// Copies a contiguous rows x cols block starting at flat element `offset`.
Tensor slice_matrix_copy(const Tensor& src, int64_t offset, int64_t rows, int64_t cols) {
    Tensor out = Tensor::empty({rows, cols}, src.dtype(), src.device());
    int64_t count = rows * cols;
#define TP_SLICE(ctype, name_) \
    case DType::name_: \
        std::memcpy(out.data_ptr<ctype>(), src.data_ptr<ctype>() + offset, \
                    count * sizeof(ctype)); \
        break;
    switch (src.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_SLICE)
        default: TP_THROW(TypeError, "slice_matrix_copy: unsupported dtype");
    }
#undef TP_SLICE
    return out;
}

inline double softplus(double y) { return std::max(y, 0.0) + std::log1p(std::exp(-std::fabs(y))); }

inline void axpy_add(Tensor& acc, const Tensor& prod) {
    int64_t n = acc.numel();
    const double* pp = prod.data_ptr<double>();
    double* ap = acc.data_ptr<double>();
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) ap[i] += pp[i];
    });
}

// fp32 twin of axpy_add: the Blas.cpp-style accumulator dtype for
// Float32/Float16/BFloat16 batches.
inline void axpy_add_f32(Tensor& acc, const Tensor& prod) {
    int64_t n = acc.numel();
    const float* pp = prod.data_ptr<float>();
    float* ap = acc.data_ptr<float>();
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) ap[i] += pp[i];
    });
}

std::vector<double> batch_block_f64(const Tensor& t, int64_t b, int64_t elems) {
    Tensor tc = t.contiguous().to(DType::Float64);
    std::vector<double> out(static_cast<size_t>(elems));
    std::memcpy(out.data(), tc.data_ptr<double>() + b * elems, elems * sizeof(double));
    return out;
}

void put_batch_block(Tensor& dst, const std::vector<double>& blkHost, int64_t bidx,
                     int64_t elems) {
    Tensor host = Tensor::empty({elems}, DType::Float64, Device(DeviceType::CPU));
    std::memcpy(host.data_ptr<double>(), blkHost.data(), elems * sizeof(double));
    Tensor piece = host.to(dst.dtype()).to(dst.device());
    Tensor dc = dst.contiguous();
    std::memcpy(reinterpret_cast<char*>(dc.data_ptr()) + bidx * elems * dst.itemsize(),
                reinterpret_cast<const char*>(piece.contiguous().data_ptr()),
                elems * dst.itemsize());
    dst = dc;
}

// Solve triangular M x = xvec (M row-major n x n values, not modified).
void triangular_solve_vec(std::vector<double> M, int64_t n, std::vector<double>& xvec,
                          bool upper, bool transposed, bool unitriangular) {
    if (transposed) {
        std::vector<double> Mt(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j) Mt[i * n + j] = M[j * n + i];
        M.swap(Mt);
        upper = !upper;
    }
    auto at = [&](int64_t i, int64_t j) { return M[i * n + j]; };
    if (!upper) {
        for (int64_t i = 0; i < n; ++i) {
            double s2 = xvec[i];
            for (int64_t j = 0; j < i; ++j) s2 -= at(i, j) * xvec[j];
            xvec[i] = s2 / (unitriangular ? 1.0 : at(i, i));
        }
    } else {
        for (int64_t i = n - 1; i >= 0; --i) {
            double s2 = xvec[i];
            for (int64_t j = i + 1; j < n; ++j) s2 -= at(i, j) * xvec[j];
            xvec[i] = s2 / (unitriangular ? 1.0 : at(i, i));
        }
    }
}

Tensor transpose_matrix_copy(const Tensor& m) {
    int64_t r = m.size(0), c = m.size(1);
    Tensor out = Tensor::empty({c, r}, m.dtype(), m.device());
    Tensor mc = m.contiguous();
#define TP_TMC(ctype, name_) \
    case DType::name_: { \
        const ctype* sp = mc.data_ptr<ctype>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        for (int64_t i = 0; i < r; ++i) \
            for (int64_t j = 0; j < c; ++j) dp[j * r + i] = sp[i * c + j]; \
        break; }
    switch (m.dtype()) {
        TP_TMC(float, Float32)
        TP_TMC(double, Float64)
        default: TP_THROW(TypeError, "transpose_matrix_copy: unsupported dtype");
    }
#undef TP_TMC
    return out;
}

} // anonymous namespace

// ===========================================================================
// Linear algebra
// ===========================================================================

Tensor addbmm_cpu(const Tensor& self, const Tensor& batch1, const Tensor& batch2,
                  Scalar beta, Scalar alpha) {
    // Blas.cpp addbmm: beta*self + alpha * sum_i batch1[i] @ batch2[i].
    // GEMMs run in the operands' dtype (oneDNN bf16/fp16 paths keep their
    // 2x throughput); the cross-batch sum accumulates in fp32 -- fp64 only
    // for Float64 inputs -- matching torch's Blas.cpp accumulation contract.
    require_float(batch1, "addbmm");
    require_float(batch2, "addbmm");
    int64_t b = batch1.size(0), n = batch1.size(1), p = batch1.size(2), m = batch2.size(2);
    DType dt = promoteTypes(batch1.dtype(), batch2.dtype());
    DType acc_dt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor work = Tensor::zeros({n, m}, acc_dt, self.device());
    for (int64_t bi = 0; bi < b; ++bi) {
        Tensor s1 = slice_matrix_copy(batch1.contiguous(), bi * n * p, n, p);
        Tensor s2 = slice_matrix_copy(batch2.contiguous(), bi * p * m, p, m);
        Tensor prod = mm_kernel(s1, s2).to(acc_dt);
        if (acc_dt == DType::Float64) axpy_add(work, prod);
        else axpy_add_f32(work, prod);
    }
    Tensor self_b = self.expand({n, m}).contiguous().to(acc_dt);
    Tensor out = Tensor::empty({n, m}, dt, self.device());
    double beta_d = beta.toDouble(), alpha_d = alpha.toDouble();
#define TP_ABMM_ACC(ctype, name_, acct) \
    case DType::name_: { \
        ctype* dp = out.data_ptr<ctype>(); \
        const acct* wp = work.data_ptr<acct>(); \
        const acct* sp = self_b.data_ptr<acct>(); \
        parallel_for(0, n * m, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) \
                dp[i] = static_cast<ctype>(beta_d * sp[i] + alpha_d * wp[i]); \
        }); \
        break; }
    switch (dt) {
        TP_ABMM_ACC(float, Float32, float)
        TP_ABMM_ACC(double, Float64, double)
        TP_ABMM_ACC(BFloat16, BFloat16, float)
        TP_ABMM_ACC(Half, Float16, float)
        default: TP_THROW(TypeError, "addbmm: unsupported dtype");
    }
#undef TP_ABMM_ACC
#undef TP_ABMM
    return out;
}

Tensor addmv_cpu(const Tensor& self, const Tensor& mat, const Tensor& vec,
                 Scalar beta, Scalar alpha) {
    // Blas.cpp addmv: beta*self + alpha*(mat @ vec).  Low-precision inputs
    // are upcast to fp32 once so the product rounds a single time (the ATen
    // CPU contract); Float64 stays in double precision.
    require_float(mat, "addmv");
    int64_t m = mat.size(0), k = mat.size(1);
    if (vec.numel() != k) TP_THROW(RuntimeError, "addmv: both args should have matching shapes");
    DType dt = promoteTypes(promoteTypes(mat.dtype(), vec.dtype()), self.dtype());
    DType cdt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor mc = mat.contiguous().to(cdt);
    Tensor vc = vec.contiguous().to(cdt);
    Tensor self_b = self.reshape({-1}).to(cdt).contiguous();
    bool scalar_self = self_b.numel() == 1;
    Tensor out = Tensor::empty({m}, dt, mat.device());
    double beta_d = beta.toDouble(), alpha_d = alpha.toDouble();
#define TP_ADMV_ACC(ctype, name_, acct) \
    case DType::name_: { \
        const acct* mp = mc.data_ptr<acct>(); \
        const acct* vp = vc.data_ptr<acct>(); \
        const acct* sp = self_b.data_ptr<acct>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, m, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) { \
                acct s2 = 0; \
                for (int64_t j = 0; j < k; ++j) s2 += mp[i * k + j] * vp[j]; \
                acct base = scalar_self ? sp[0] : sp[i]; \
                dp[i] = static_cast<ctype>(beta_d * base + alpha_d * s2); \
            } \
        }); \
        break; }
    switch (dt) {
        TP_ADMV_ACC(float, Float32, float)
        TP_ADMV_ACC(double, Float64, double)
        TP_ADMV_ACC(BFloat16, BFloat16, float)
        TP_ADMV_ACC(Half, Float16, float)
        default: TP_THROW(TypeError, "addmv: unsupported dtype");
    }
#undef TP_ADMV_ACC
#undef TP_ADMV
    return out;
}

Tensor addr_cpu(const Tensor& self, const Tensor& vec1, const Tensor& vec2,
                Scalar beta, Scalar alpha) {
    // Blas.cpp addr: beta*self + alpha*vec1⊗vec2.  fp32 compute for
    // Float32/Float16/BFloat16 (single rounding), fp64 for Float64.
    require_float(vec1, "addr");
    int64_t m = vec1.numel(), k = vec2.numel();
    DType dt = promoteTypes(promoteTypes(vec1.dtype(), vec2.dtype()), self.dtype());
    DType cdt = (dt == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor v1 = vec1.contiguous().to(cdt);
    Tensor v2 = vec2.contiguous().to(cdt);
    Tensor self_b = self.expand({m, k}).contiguous().to(cdt);
    Tensor out = Tensor::empty({m, k}, dt, self.device());
    double beta_d = beta.toDouble(), alpha_d = alpha.toDouble();
#define TP_ADDR_ACC(ctype, name_, acct) \
    case DType::name_: { \
        const acct* a = v1.data_ptr<acct>(); \
        const acct* bv = v2.data_ptr<acct>(); \
        const acct* sp = self_b.data_ptr<acct>(); \
        ctype* dp = out.data_ptr<ctype>(); \
        parallel_for(0, m * k, GRAIN_SIZE, [&](int64_t begin, int64_t end) { \
            for (int64_t i = begin; i < end; ++i) { \
                int64_t r = i / k, c = i % k; \
                dp[i] = static_cast<ctype>(beta_d * sp[i] + alpha_d * a[r] * bv[c]); \
            } \
        }); \
        break; }
    switch (dt) {
        TP_ADDR_ACC(float, Float32, float)
        TP_ADDR_ACC(double, Float64, double)
        TP_ADDR_ACC(BFloat16, BFloat16, float)
        TP_ADDR_ACC(Half, Float16, float)
        default: TP_THROW(TypeError, "addr: unsupported dtype");
    }
#undef TP_ADDR_ACC
#undef TP_ADDR
    return out;
}

Tensor vdot_cpu(const Tensor& a_in, const Tensor& b_in) {
    Tensor a = a_in.contiguous().reshape({a_in.numel()});
    Tensor b = b_in.contiguous().reshape({b_in.numel()});
    if (a.numel() != b.numel()) TP_THROW(RuntimeError, "vdot: sizes don't match");
    int64_t n = a.numel();
    if (is_cplx(a.dtype())) {
        if (a.dtype() == DType::ComplexFloat) {
            const std::complex<float>* ap = a.data_ptr<std::complex<float>>();
            const std::complex<float>* bp =
                b.to(a.dtype()).data_ptr<std::complex<float>>();
            std::complex<double> acc = 0;
            for (int64_t i = 0; i < n; ++i)
                acc += static_cast<std::complex<double>>(std::conj(ap[i]) * bp[i]);
            return Tensor::full({}, Scalar(std::complex<float>(
                                     static_cast<float>(acc.real()),
                                     static_cast<float>(acc.imag()))),
                                a.dtype(), a.device());
        }
        const std::complex<double>* ap = a.data_ptr<std::complex<double>>();
        const std::complex<double>* bp =
            b.to(a.dtype()).data_ptr<std::complex<double>>();
        std::complex<double> acc = 0;
        for (int64_t i = 0; i < n; ++i) acc += std::conj(ap[i]) * bp[i];
        return Tensor::full({}, Scalar(acc), a.dtype(), a.device());
    }
    Tensor ad = a.to(DType::Float64), bd = b.to(DType::Float64);
    const double* ap = ad.data_ptr<double>();
    const double* bp = bd.data_ptr<double>();
    double acc = 0;
    for (int64_t i = 0; i < n; ++i) acc += ap[i] * bp[i];
    DType out_dt = a_in.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    return Tensor::full({}, Scalar(acc), out_dt, a.device());
}

Tensor cholesky_cpu(const Tensor& self, bool upper) {
    require_float(self, "cholesky");
    int64_t nd = self.dim();
    if (nd < 2 || self.size(-1) != self.size(-2))
        TP_THROW(RuntimeError, "cholesky: input must be batches of square matrices");
    int64_t n = self.size(-1);
    int64_t batch = self.numel() / (n * n);
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    for (int64_t bidx = 0; bidx < batch; ++bidx) {
        std::vector<double> A = batch_block_f64(self, bidx, n * n);
        // Cholesky-Banachiewicz on lower triangle.
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j <= i; ++j) {
                double s2 = A[i * n + j];
                for (int64_t kk = 0; kk < j; ++kk) s2 -= A[i * n + kk] * A[j * n + kk];
                if (i == j) {
                    if (!(s2 > 0))
                        TP_THROW(RuntimeError,
                                 "cholesky: matrix is not positive definite");
                    A[i * n + i] = std::sqrt(s2);
                } else {
                    A[i * n + j] = s2 / A[j * n + j];
                }
            }
            for (int64_t j = i + 1; j < n; ++j) A[i * n + j] = 0.0;
        }
        if (upper) {
            std::vector<double> U(n * n);
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < n; ++j) U[i * n + j] = A[j * n + i];
            A.swap(U);
        }
        put_batch_block(out, A, bidx, n * n);
    }
    return out;
}

Tensor cholesky_inverse_cpu(const Tensor& self, bool upper) {
    // A = L L^T -> A^{-1} = L^{-T} L^{-1}. Input is the factor.
    require_float(self, "cholesky_inverse");
    int64_t n = self.size(-1);
    int64_t batch = self.numel() / (n * n);
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    for (int64_t bidx = 0; bidx < batch; ++bidx) {
        std::vector<double> L = batch_block_f64(self, bidx, n * n);
        if (upper) {
            std::vector<double> Lt(n * n);
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < n; ++j) Lt[i * n + j] = L[j * n + i];
            L.swap(Lt);
        }
        std::vector<double> Linv(n * n, 0.0);
        for (int64_t j = 0; j < n; ++j) {
            std::vector<double> e(n, 0.0);
            e[j] = 1.0;
            triangular_solve_vec(L, n, e, false, false, false);
            for (int64_t i = 0; i < n; ++i) Linv[i * n + j] = e[i];
        }
        std::vector<double> Ainv(n * n, 0.0);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j) {
                double s2 = 0;
                for (int64_t kk = 0; kk < n; ++kk)
                    s2 += Linv[kk * n + i] * Linv[kk * n + j];
                Ainv[i * n + j] = s2;
            }
        put_batch_block(out, Ainv, bidx, n * n);
    }
    return out;
}

Tensor cholesky_solve_cpu(const Tensor& self, const Tensor& input2, bool upper) {
    // self = B (..., n, rhs); input2 = factor; solve (L L^T) X = B.
    require_float(self, "cholesky_solve");
    require_float(input2, "cholesky_solve");
    int64_t n = self.size(-2), rhs = self.size(-1);
    int64_t batchB = self.numel() / (n * rhs);
    int64_t batchL = input2.numel() / (n * n);
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    for (int64_t bidx = 0; bidx < batchB; ++bidx) {
        int64_t li = batchL == 1 ? 0 : bidx;
        std::vector<double> L = batch_block_f64(input2, li, n * n);
        if (upper) {
            std::vector<double> Lt(n * n);
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < n; ++j) Lt[i * n + j] = L[j * n + i];
            L.swap(Lt);
        }
        std::vector<double> B = batch_block_f64(self, bidx, n * rhs);
        for (int64_t c = 0; c < rhs; ++c) {
            std::vector<double> xcol(n);
            for (int64_t i = 0; i < n; ++i) xcol[i] = B[i * rhs + c];
            triangular_solve_vec(L, n, xcol, false, false, false);   // L y = b
            triangular_solve_vec(L, n, xcol, true, false, false);    // L^T x = y
            for (int64_t i = 0; i < n; ++i) B[i * rhs + c] = xcol[i];
        }
        put_batch_block(out, B, bidx, n * rhs);
    }
    return out;
}

std::tuple<Tensor, Tensor> triangular_solve_cpu(const Tensor& self, const Tensor& A,
                                                bool upper, bool transpose,
                                                bool unitriangular) {
    require_float(A, "triangular_solve");
    int64_t n = A.size(-1);
    int64_t rhs = self.size(-1);
    int64_t batchB = self.numel() / (n * rhs);
    int64_t batchA = A.numel() / (n * n);
    Tensor X = Tensor::empty(shape_of(self), self.dtype(), self.device());
    for (int64_t bidx = 0; bidx < batchB; ++bidx) {
        int64_t ai = batchA == 1 ? 0 : bidx;
        std::vector<double> M = batch_block_f64(A, ai, n * n);
        std::vector<double> B = batch_block_f64(self, bidx, n * rhs);
        for (int64_t c = 0; c < rhs; ++c) {
            std::vector<double> xcol(n);
            for (int64_t i = 0; i < n; ++i) xcol[i] = B[i * rhs + c];
            triangular_solve_vec(M, n, xcol, upper, transpose, unitriangular);
            for (int64_t i = 0; i < n; ++i) B[i * rhs + c] = xcol[i];
        }
        put_batch_block(X, B, bidx, n * rhs);
    }
    return {X, A.clone()};
}

std::tuple<Tensor, Tensor, Tensor> svd_cpu(const Tensor& self, bool some, bool compute_uv) {
    // One-sided Jacobi reference SVD, always reduced ("some" accepted for
    // signature compatibility). ATen uses LAPACK gesdd.
    (void)some;
    require_float(self, "svd");
    int64_t nd = self.dim();
    int64_t m = self.size(-2), k = self.size(-1);
    int64_t r = std::min(m, k);
    int64_t batch = self.numel() / (m * k);
    bool flip = m < k;
    int64_t wm = flip ? k : m;  // working-matrix rows
    int64_t wn = flip ? m : k;  // working-matrix cols == rank

    Tensor U = Tensor::empty(flip ? [&]{auto v=shape_of(self); v[nd-2]=m; v[nd-1]=r; return v;}()
                                  : shape_of(self),
                             self.dtype(), self.device());
    Tensor S = Tensor::empty([&]{auto v=shape_of(self); v.pop_back(); v.push_back(r); return v;}(),
                             self.dtype(), self.device());
    std::vector<int64_t> vh_shape = shape_of(self);
    vh_shape[nd - 2] = r;
    vh_shape[nd - 1] = k;
    Tensor Vh = Tensor::empty(vh_shape, self.dtype(), self.device());

    const double kTol = 1e-14;
    const int kMaxSweeps = 60;
    for (int64_t bidx = 0; bidx < batch; ++bidx) {
        std::vector<double> W = batch_block_f64(self, bidx, m * k);
        if (flip) {
            // work on the transpose (wm x wn)
            std::vector<double> Wt(wm * wn);
            for (int64_t i = 0; i < m; ++i)
                for (int64_t j = 0; j < k; ++j) Wt[j * wm + i] = W[i * k + j];
            W.swap(Wt);
        }
        std::vector<double> V(wn * wn, 0.0);
        for (int64_t i = 0; i < wn; ++i) V[i * wn + i] = 1.0;
        for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
            double offmax = 0;
            for (int64_t pp = 0; pp < wn; ++pp) {
                for (int64_t q = pp + 1; q < wn; ++q) {
                    double al = 0, be = 0, ga = 0;
                    for (int64_t i2 = 0; i2 < wm; ++i2) {
                        double xp = W[i2 * wn + pp], xq = W[i2 * wn + q];
                        al += xp * xp;
                        be += xq * xq;
                        ga += xp * xq;
                    }
                    double rel = ga / std::sqrt(std::max(al * be, 1e-300));
                    offmax = std::max(offmax, std::fabs(rel));
                    if (ga == 0.0 || std::fabs(rel) <= kTol) continue;
                    double zeta = (be - al) / (2.0 * ga);
                    double t = (zeta >= 0 ? 1.0 : -1.0) /
                               (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                    double c = 1.0 / std::sqrt(1.0 + t * t);
                    double sv = c * t;
                    for (int64_t i2 = 0; i2 < wm; ++i2) {
                        double xp = W[i2 * wn + pp], xq = W[i2 * wn + q];
                        W[i2 * wn + pp] = c * xp - sv * xq;
                        W[i2 * wn + q] = sv * xp + c * xq;
                    }
                    for (int64_t i2 = 0; i2 < wn; ++i2) {
                        double vp = V[i2 * wn + pp], vq = V[i2 * wn + q];
                        V[i2 * wn + pp] = c * vp - sv * vq;
                        V[i2 * wn + q] = sv * vp + c * vq;
                    }
                }
            }
            if (offmax <= kTol) break;
        }
        std::vector<double> sv(wn, 0.0);
        for (int64_t j = 0; j < wn; ++j) {
            double s2 = 0;
            for (int64_t i2 = 0; i2 < wm; ++i2) s2 += W[i2 * wn + j] * W[i2 * wn + j];
            sv[j] = std::sqrt(s2);
        }
        std::vector<int64_t> ord(wn);
        for (int64_t j = 0; j < wn; ++j) ord[j] = j;
        std::sort(ord.begin(), ord.end(),
                  [&](int64_t a2, int64_t b2) { return sv[a2] > sv[b2]; });

        // Non-flip: A = U S V^T with U = normalized W cols (wm x r),
        // Vh = V cols transposed (r x wn).
        // Flip: worked on A^T: U_A = V cols (m x r), Vh_A = W cols/sv (r x k).
        std::vector<double> Uout(static_cast<size_t>((flip ? m : wm) * r), 0.0);
        std::vector<double> Vhout(static_cast<size_t>(r * (flip ? k : wn)), 0.0);
        for (int64_t jj = 0; jj < r; ++jj) {
            int64_t j = ord[jj];
            double denom = sv[j] > 0 ? sv[j] : 1.0;
            if (!flip) {
                for (int64_t i2 = 0; i2 < wm; ++i2)
                    Uout[i2 * r + jj] = W[i2 * wn + j] / denom;
                for (int64_t i2 = 0; i2 < wn; ++i2)
                    Vhout[jj * wn + i2] = V[i2 * wn + j];
            } else {
                for (int64_t i2 = 0; i2 < m; ++i2)
                    Uout[i2 * r + jj] = V[i2 * wn + j];
                for (int64_t i2 = 0; i2 < k; ++i2)
                    Vhout[jj * k + i2] = W[i2 * wn + j] / denom;
            }
        }
        put_batch_block(S, sv, bidx, r);
        put_batch_block(U, Uout, bidx, (flip ? m : wm) * r);
        put_batch_block(Vh, Vhout, bidx, r * (flip ? k : wn));
    }
    if (!compute_uv) {
        Tensor zero = Tensor::zeros({}, self.dtype(), self.device());
        return {zero, S, zero};
    }
    return {U, S, Vh};
}

Tensor pairwise_distance_cpu(const Tensor& x1, const Tensor& x2, double p, double eps,
                             bool keepdim) {
    // ATen Distance.cpp: pairwise_distance is a composite --
    //   norm(x1 - x2 + eps, p, last_dim, keepdim)
    // with full broadcasting; the old hand loop assumed (N, D) inputs and
    // silently produced zeros for 1-D pairs.
    Tensor diff = x1 - x2 + eps;
    if (diff.dim() == 0) {
        TP_THROW(RuntimeError, "pairwise_distance: inputs must be at least 1-dimensional");
    }
    const int64_t dim = diff.dim() - 1;
    return detail::redispatch_norm_function(diff,
                                            std::vector<int64_t>{dim}, p, keepdim);
}

Tensor pdist_cpu(const Tensor& self, double p) {
    // DistanceKernels pdist: condensed upper-triangle distances, len n(n-1)/2.
    require_float(self, "pdist");
    int64_t n = self.size(0), D = self.size(1);
    int64_t outn = n * (n - 1) / 2;
    Tensor a = self.contiguous().to(DType::Float64);
    const double* ap = a.data_ptr<double>();
    Tensor out = Tensor::empty({std::max<int64_t>(outn, 1)}, DType::Float64, self.device());
    double* op = out.data_ptr<double>();
    parallel_for(0, std::max<int64_t>(outn, 1), GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t li = begin; li < end; ++li) {
            // linear -> (i, j), i < j (numpy condensed-index trick)
            int64_t i = static_cast<int64_t>(
                n - 2 - std::floor(std::sqrt(-8.0 * li + 4.0 * n * (n - 1) - 7) / 2.0 - 0.5));
            int64_t j = li + i + 1 - (n * (n - 1)) / 2 +
                        ((n - i) * (n - i - 1)) / 2;
            double d2 = 0;
            if (p == std::numeric_limits<double>::infinity()) {
                for (int64_t c = 0; c < D; ++c)
                    d2 = std::max(d2, std::fabs(ap[i * D + c] - ap[j * D + c]));
            } else if (p == 0.0) {
                int64_t cnt = 0;
                for (int64_t c = 0; c < D; ++c)
                    if (ap[i * D + c] != ap[j * D + c]) ++cnt;
                d2 = static_cast<double>(cnt);
            } else if (p == 2.0) {
                for (int64_t c = 0; c < D; ++c) {
                    double diff = ap[i * D + c] - ap[j * D + c];
                    d2 += diff * diff;
                }
                d2 = std::sqrt(d2);
            } else if (p == 1.0) {
                for (int64_t c = 0; c < D; ++c)
                    d2 += std::fabs(ap[i * D + c] - ap[j * D + c]);
            } else {
                for (int64_t c = 0; c < D; ++c)
                    d2 += std::pow(std::fabs(ap[i * D + c] - ap[j * D + c]), p);
                d2 = std::pow(d2, 1.0 / p);
            }
            op[li] = d2;
        }
    });
    DType out_dt = self.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    Tensor res = out.to(out_dt);
    return outn == 0 ? res.reshape({0}) : res;
}

// ===========================================================================
// Losses (mean reduction)
// ===========================================================================

Tensor binary_cross_entropy_with_logits_cpu(const Tensor& self, const Tensor& target,
                                            const std::optional<Tensor>& weight_opt,
                                            const std::optional<Tensor>& pos_weight_opt) {
    // Stable form (LossBCE2d): l = w*(pw*t*softplus(-x) + (1-t)*softplus(x)).
    Tensor weight = weight_opt.value_or(Tensor());
    Tensor pos_weight = pos_weight_opt.value_or(Tensor());
    Tensor x = self.contiguous().to(DType::Float64);
    Tensor t = target.contiguous().to(DType::Float64).expand(shape_of(x)).contiguous();
    bool has_w = weight.defined() && weight.numel() > 0;
    bool has_pw = pos_weight.defined() && pos_weight.numel() > 0;
    Tensor w = has_w ? weight.to(DType::Float64).expand(shape_of(x)).contiguous()
                     : Tensor::zeros({}, DType::Float64, Device(DeviceType::CPU));
    Tensor pw = has_pw ? pos_weight.to(DType::Float64).expand(shape_of(x)).contiguous()
                       : Tensor::zeros({}, DType::Float64, Device(DeviceType::CPU));
    int64_t n = x.numel();
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    const double* wp = has_w ? w.data_ptr<double>() : nullptr;
    const double* pwp = has_pw ? pw.data_ptr<double>() : nullptr;
    double total = 0;
    for (int64_t i = 0; i < n; ++i) {
        double xv = xp[i], tv = tp[i];
        double wi = wp ? wp[i] : 1.0;
        double pi = pwp ? pwp[i] : 1.0;
        total += wi * (pi * tv * softplus(-xv) + (1.0 - tv) * softplus(xv));
    }
    double mean = n > 0 ? total / static_cast<double>(n) : 0.0;
    DType out_dt = self.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    return Tensor::full({}, Scalar(mean), out_dt, self.device());
}

Tensor hinge_embedding_loss_cpu(const Tensor& input, const Tensor& target, Scalar margin) {
    // Loss.cpp: target == 1 -> x ; else relu(margin - x); mean.
    Tensor x = input.contiguous().to(DType::Float64);
    Tensor t = target.contiguous().to(DType::Float64).expand(shape_of(x)).contiguous();
    double mg = margin.toDouble();
    int64_t n = x.numel();
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i)
        total += (tp[i] == 1.0) ? xp[i] : std::max(0.0, mg - xp[i]);
    double mean = n > 0 ? total / static_cast<double>(n) : 0.0;
    DType out_dt = input.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    return Tensor::full({}, Scalar(mean), out_dt, input.device());
}

Tensor margin_ranking_loss_cpu(const Tensor& input1, const Tensor& input2,
                               const Tensor& target, Scalar margin) {
    // mean(relu(margin - target*(x1 - x2)))
    Tensor a = input1.contiguous().to(DType::Float64);
    Tensor b = input2.contiguous().to(DType::Float64).expand(shape_of(a)).contiguous();
    Tensor tg = target.contiguous().to(DType::Float64).expand(shape_of(a)).contiguous();
    double mg = margin.toDouble();
    int64_t n = a.numel();
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    const double* gp = tg.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i)
        total += std::max(0.0, mg - gp[i] * (ap[i] - bp[i]));
    double mean = n > 0 ? total / static_cast<double>(n) : 0.0;
    DType out_dt = input1.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    return Tensor::full({}, Scalar(mean), out_dt, input1.device());
}

namespace {

} // anonymous namespace

// ---------------------------------------------------------------------------
// Window / complex factories
// ---------------------------------------------------------------------------

Tensor real_cpu(const Tensor& self) {
    if (!is_cplx(self.dtype())) return self.clone();
    Tensor out = Tensor::empty(shape_of(self),
                               self.dtype() == DType::ComplexDouble ? DType::Float64 : DType::Float32,
                               self.device());
    Tensor sc = self.contiguous();
    int64_t n = self.numel();
    if (self.dtype() == DType::ComplexFloat) {
        const std::complex<float>* sp = sc.data_ptr<std::complex<float>>();
        float* dp = out.data_ptr<float>();
        for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].real();
    } else {
        const std::complex<double>* sp = sc.data_ptr<std::complex<double>>();
        double* dp = out.data_ptr<double>();
        for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].real();
    }
    return out;
}

Tensor imag_cpu(const Tensor& self) {
    if (!is_cplx(self.dtype()))
        return Tensor::zeros(shape_of(self), self.dtype(), self.device());
    Tensor out = Tensor::empty(shape_of(self),
                               self.dtype() == DType::ComplexDouble ? DType::Float64 : DType::Float32,
                               self.device());
    Tensor sc = self.contiguous();
    int64_t n = self.numel();
    if (self.dtype() == DType::ComplexFloat) {
        const std::complex<float>* sp = sc.data_ptr<std::complex<float>>();
        float* dp = out.data_ptr<float>();
        for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].imag();
    } else {
        const std::complex<double>* sp = sc.data_ptr<std::complex<double>>();
        double* dp = out.data_ptr<double>();
        for (int64_t i = 0; i < n; ++i) dp[i] = sp[i].imag();
    }
    return out;
}

Tensor conj_cpu(const Tensor& self) {
    // ATen alignment: conj over real tensors is a zero-copy VIEW -- a new
    // TensorImpl sharing storage, exactly like torch's conjugate-bit views.
    // Returning `self` itself would hand the same impl back to autograd
    // wrappers, which then re-tag the input's grad_fn and corrupt graphs.
    if (!is_cplx(self.dtype())) {
        return self.as_strided(static_cast<std::vector<int64_t>>(self.shape()),
                               static_cast<std::vector<int64_t>>(self.strides()));
    }
    Tensor out = detail::contiguous_clone(self);
    int64_t n = out.numel();
    if (self.dtype() == DType::ComplexFloat) {
        std::complex<float>* dp = out.data_ptr<std::complex<float>>();
        for (int64_t i = 0; i < n; ++i) dp[i] = std::conj(dp[i]);
    } else {
        std::complex<double>* dp = out.data_ptr<std::complex<double>>();
        for (int64_t i = 0; i < n; ++i) dp[i] = std::conj(dp[i]);
    }
    return out;
}

// ATen native_functions.yaml: adjoint(Tensor(a) self) -> Tensor(a) is
// transpose(-2, -1) composed with conj(); ndim <= 1 is plain conj.
Tensor adjoint_cpu(const Tensor& self) {
    if (self.dim() <= 1) return conj_cpu(self);
    return conj_cpu(self.transpose(-2, -1));
}

Tensor complex_cpu(const Tensor& real, const Tensor& imag) {
    DType fdt = promoteTypes(real.dtype(), imag.dtype());
    if (fdt == DType::Float64) fdt = DType::ComplexDouble;
    else fdt = DType::ComplexFloat;
    std::vector<int64_t> shape = broadcast_shapes(shape_of(real), shape_of(imag));
    Tensor rc = real.expand(shape).contiguous().to(fdt == DType::ComplexDouble ? DType::Float64 : DType::Float32);
    Tensor ic = imag.expand(shape).contiguous().to(rc.dtype());
    Tensor out = Tensor::empty(shape, fdt, real.device());
    int64_t n = out.numel();
    if (fdt == DType::ComplexFloat) {
        const float* rp = rc.data_ptr<float>();
        const float* ip = ic.data_ptr<float>();
        std::complex<float>* dp = out.data_ptr<std::complex<float>>();
        for (int64_t i = 0; i < n; ++i) dp[i] = std::complex<float>(rp[i], ip[i]);
    } else {
        const double* rp = rc.data_ptr<double>();
        const double* ip = ic.data_ptr<double>();
        std::complex<double>* dp = out.data_ptr<std::complex<double>>();
        for (int64_t i = 0; i < n; ++i) dp[i] = std::complex<double>(rp[i], ip[i]);
    }
    return out;
}

Tensor polar_cpu(const Tensor& abs_, const Tensor& angle_) {
    DType fdt = promoteTypes(abs_.dtype(), angle_.dtype());
    if (fdt != DType::Float64) fdt = DType::Float32;
    std::vector<int64_t> shape = broadcast_shapes(shape_of(abs_), shape_of(angle_));
    Tensor a = abs_.expand(shape).contiguous().to(fdt);
    Tensor th = angle_.expand(shape).contiguous().to(fdt);
    DType cdt = fdt == DType::Float64 ? DType::ComplexDouble : DType::ComplexFloat;
    Tensor out = Tensor::empty(shape, cdt, abs_.device());
    int64_t n = out.numel();
    if (fdt == DType::Float64) {
        const double* ap = a.data_ptr<double>();
        const double* tp = th.data_ptr<double>();
        std::complex<double>* dp = out.data_ptr<std::complex<double>>();
        for (int64_t i = 0; i < n; ++i)
            dp[i] = std::polar(ap[i], tp[i]);
    } else {
        const float* ap = a.data_ptr<float>();
        const float* tp = th.data_ptr<float>();
        std::complex<float>* dp = out.data_ptr<std::complex<float>>();
        for (int64_t i = 0; i < n; ++i)
            dp[i] = std::polar(ap[i], tp[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// RNN cells (Rnn.cpp layout: [w_ih, w_hh, b_ih?, b_hh?] per direction)
// ---------------------------------------------------------------------------

namespace {

// Fetch one parameter from the params list with bounds checking.
const Tensor& param_at(const std::vector<Tensor>& params, size_t idx, bool has_biases) {
    (void)has_biases;
    if (idx >= params.size()) TP_THROW(RuntimeError, "rnn: missing parameter ", idx);
    return params[idx];
}

} // anonymous namespace

static std::tuple<Tensor, Tensor, Tensor> rnn_impl(
    int kind,  // 0=lstm, 1=gru, 2=tanh, 3=relu
    const Tensor& input, const std::vector<Tensor>& hx,
    const std::vector<Tensor>& params, bool has_biases, int64_t num_layers,
    bool bidirectional, bool batch_first) {
    // Vectorized port of at::native RNN loops (aten/src/ATen/native/RNN.cpp).
    // Compute happens in the input dtype (fp32 stays fp32); input-side gates
    // for a whole direction are produced by a single GEMM and gate math is
    // expressed with tensor ops so the pointwise kernels do the element work.
    const DType dt = input.dtype();
    if (dt != DType::Float32 && dt != DType::Float64) {
        TP_THROW(RuntimeError, "rnn: only Float32/Float64 inputs are supported");
    }
    Tensor x = batch_first ? input.transpose(0, 1).contiguous() : input.contiguous();
    const int64_t T = x.size(0), N = x.size(1);
    if (hx.empty()) TP_THROW(RuntimeError, "rnn: hx required");
    const int64_t L = num_layers;
    const int64_t dirs = bidirectional ? 2 : 1;
    const int64_t H = hx[0].size(-1);

    Tensor hn_out = Tensor::zeros({L * dirs, N, H}, hx[0].dtype(), input.device());
    Tensor cn_out = kind == 0
        ? Tensor::zeros({L * dirs, N, H}, hx[0].dtype(), input.device())
        : Tensor();

    size_t ppi = 0;  // params cursor: per layer/direction w_ih, w_hh[, b_ih, b_hh]
    auto param_at = [&](void) -> const Tensor& {
        if (ppi >= params.size()) TP_THROW(RuntimeError, "rnn: missing parameter ", ppi);
        return params[ppi++];
    };

    for (int64_t layer = 0; layer < L; ++layer) {
        // Per-direction sequence outputs concatenated along the feature dim;
        // writes go through Tensor::slice/select views (narrow is a copying
        // op on this backend and must not be used as an assignment target).
        std::vector<Tensor> dir_outs;
        for (int64_t dir = 0; dir < dirs; ++dir) {
            const int64_t state_idx = layer * dirs + dir;
            Tensor h = hx[0].select(0, state_idx).contiguous();
            Tensor c = kind == 0 ? hx[1].select(0, state_idx).contiguous() : h;
            Tensor dir_out = Tensor::zeros({T, N, H}, dt, x.device());

            const Tensor& w_ih = param_at();
            const Tensor& w_hh = param_at();
            Tensor b_ih, b_hh;
            if (has_biases) {
                b_ih = param_at();
                b_hh = param_at();
                if (!(b_ih.numel() > 0)) b_ih = Tensor();
                if (!(b_hh.numel() > 0)) b_hh = Tensor();
            }

            // Input-side gates for the whole sequence in one GEMM:
            // (T*N, feat) @ (feat, G)^T + b_ih  ->  (T*N, G).
            Tensor x2d = x.reshape({T * N, x.size(2)});
            Tensor in_gates = x2d.mm(w_ih.t());
            if (b_ih.defined()) in_gates = in_gates.add(b_ih);
            const int64_t G = in_gates.size(1);

            const Tensor w_hh_t = w_hh.t();  // (H, G)

            for (int64_t t = 0; t < T; ++t) {
                const int64_t tt = dir == 0 ? t : (T - 1 - t);
                const Tensor ig = in_gates.narrow(0, tt * N, N);   // (N, G)
                Tensor hg = h.mm(w_hh_t);                          // (N, G)
                // lstm / simple cells fold b_hh linearly into every gate;
                // gru handles the three bias segments separately below.
                if (kind != 1 && b_hh.defined()) hg = hg.add(b_hh);

                if (kind == 0) {
                    auto gate = [&](int64_t off, Tensor (*fn)(const Tensor&)) -> Tensor {
                        return fn(ig.narrow(1, off, H).add(hg.narrow(1, off, H)));
                    };
                    Tensor i_ = gate(0, [](const Tensor& v) { return v.sigmoid(); });
                    Tensor f_ = gate(H, [](const Tensor& v) { return v.sigmoid(); });
                    Tensor g_ = gate(2 * H, [](const Tensor& v) { return v.tanh(); });
                    Tensor o_ = gate(3 * H, [](const Tensor& v) { return v.sigmoid(); });
                    c = f_.mul(c).add(i_.mul(g_));
                    h = o_.mul(c.tanh());
                } else if (kind == 1) {
                    // torch GRUCell:
                    //   r = sigmoid(ir + hr), z = sigmoid(iz + hz)
                    //   n = tanh(in + r * (hn + b_hn))
                    //   h' = (1 - z) * n + z * h
                    Tensor b_r, b_z, b_n;
                    if (b_hh.defined()) {
                        b_r = b_hh.narrow(0, 0, H);
                        b_z = b_hh.narrow(0, H, H);
                        b_n = b_hh.narrow(0, 2 * H, H);
                    }
                    Tensor r_ = ig.narrow(1, 0, H)
                                    .add(b_r.defined() ? b_r.add(hg.narrow(1, 0, H))
                                                       : hg.narrow(1, 0, H))
                                    .sigmoid();
                    Tensor z_ = ig.narrow(1, H, H)
                                    .add(b_z.defined() ? b_z.add(hg.narrow(1, H, H))
                                                       : hg.narrow(1, H, H))
                                    .sigmoid();
                    Tensor hn_lin = hg.narrow(1, 2 * H, H);
                    if (b_n.defined()) hn_lin = hn_lin.add(b_n);
                    Tensor n_ = ig.narrow(1, 2 * H, H).add(r_.mul(hn_lin)).tanh();
                    Tensor one_minus_z = z_.neg().add(Scalar(1));
                    h = one_minus_z.mul(n_).add(z_.mul(h));
                } else {
                    Tensor pre = ig.add(hg);
                    h = (kind == 2) ? pre.tanh() : pre.relu();
                }

                dir_out.select(0, tt).copy_(h);
                hn_out.select(0, state_idx).copy_(h);
                if (kind == 0) cn_out.select(0, state_idx).copy_(c);
            }
            dir_outs.push_back(dir_out);
        }
        Tensor layer_out;
        if (dirs == 1) {
            layer_out = dir_outs[0];
        } else {
            extern Tensor cat_kernel(const std::vector<Tensor>& tensors, int64_t dim);
            layer_out = cat_kernel({dir_outs[0], dir_outs[1]}, 2);
        }
        x = layer_out;
    }
    Tensor y = batch_first ? x.transpose(0, 1).contiguous() : x;
    return {y, hn_out, cn_out};
}

std::tuple<Tensor, Tensor> rnn_relu_cpu(const Tensor& input, const std::vector<Tensor>& hx,
                                        const std::vector<Tensor>& params, bool has_biases,
                                        int64_t num_layers, float dropout_p, bool training,
                                        bool bidirectional, bool batch_first) {
    (void)dropout_p; (void)training;
    auto r = rnn_impl(3, input, hx, params, has_biases, num_layers, bidirectional, batch_first);
    return {std::get<0>(r), std::get<1>(r)};
}
std::tuple<Tensor, Tensor> rnn_tanh_cpu(const Tensor& input, const std::vector<Tensor>& hx,
                                        const std::vector<Tensor>& params, bool has_biases,
                                        int64_t num_layers, float dropout_p, bool training,
                                        bool bidirectional, bool batch_first) {
    (void)dropout_p; (void)training;
    auto r = rnn_impl(2, input, hx, params, has_biases, num_layers, bidirectional, batch_first);
    return {std::get<0>(r), std::get<1>(r)};
}
std::tuple<Tensor, Tensor> gru_cpu(const Tensor& input, const std::vector<Tensor>& hx,
                                   const std::vector<Tensor>& params, bool has_biases,
                                   int64_t num_layers, float dropout_p, bool training,
                                   bool bidirectional, bool batch_first) {
    (void)dropout_p; (void)training;
    auto r = rnn_impl(1, input, hx, params, has_biases, num_layers, bidirectional, batch_first);
    return {std::get<0>(r), std::get<1>(r)};
}
std::tuple<Tensor, Tensor, Tensor> lstm_cpu(const Tensor& input, const std::vector<Tensor>& hx,
                                            const std::vector<Tensor>& params, bool has_biases,
                                            int64_t num_layers, float dropout_p, bool training,
                                            bool bidirectional, bool batch_first) {
    (void)dropout_p; (void)training;
    return rnn_impl(0, input, hx, params, has_biases, num_layers, bidirectional, batch_first);
}


// ===========================================================================
// Remaining nn losses (mean reduction). ATen anchors: Loss.cpp families
// (l1/smooth_l1/huber/kl_div/bce/cosine_embedding/soft_margin/
//  triplet_margin/poisson_nll/multi_margin/multilabel_*).
// ctc_loss / gaussian_nll_loss intentionally omitted (heavy DP / different
// signature surface) — noted for a later pass.
// ===========================================================================

namespace {

inline std::pair<Tensor, Tensor> bcast2(const Tensor& a, const Tensor& b) {
    auto shape = broadcast_shapes(shape_of(a), shape_of(b));
    return {a.expand(shape).contiguous().to(DType::Float64),
            b.expand(shape).contiguous().to(DType::Float64)};
}

inline Tensor scalar_from(double v, DType dt, const Device& dev) {
    return Tensor::full({}, Scalar(v),
                        dt == DType::Float64 ? DType::Float64 : DType::Float32, dev);
}

} // anonymous namespace

Tensor l1_loss_cpu(const Tensor& input, const Tensor& target) {
    auto pr = bcast2(input, target);
    Tensor a = pr.first, b = pr.second;
    int64_t n = a.numel();
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i) total += std::fabs(ap[i] - bp[i]);
    return scalar_from(n ? total / n : 0.0, input.dtype(), input.device());
}

Tensor smooth_l1_loss_cpu(const Tensor& input, const Tensor& target, Scalar beta) {
    auto pr = bcast2(input, target);
    Tensor a = pr.first, b = pr.second;
    double bt = beta.toDouble();
    int64_t n = a.numel();
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i) {
        double d = std::fabs(ap[i] - bp[i]);
        total += d < bt ? 0.5 * d * d / bt : d - 0.5 * bt;
    }
    return scalar_from(n ? total / n : 0.0, input.dtype(), input.device());
}

Tensor huber_loss_cpu(const Tensor& input, const Tensor& target, Scalar delta) {
    auto pr = bcast2(input, target);
    Tensor a = pr.first, b = pr.second;
    double dl = delta.toDouble();
    int64_t n = a.numel();
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i) {
        double d = std::fabs(ap[i] - bp[i]);
        total += d < dl ? 0.5 * d * d : dl * (d - 0.5 * dl);
    }
    return scalar_from(n ? total / n : 0.0, input.dtype(), input.device());
}

Tensor kl_div_cpu(const Tensor& input, const Tensor& target) {
    // input log-probs; target probs; mean(t*(log t - x)).
    auto pr = bcast2(input, target);
    Tensor x = pr.first, t = pr.second;
    int64_t n = x.numel();
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i)
        if (tp[i] > 0) total += tp[i] * (std::log(tp[i]) - xp[i]);
    return scalar_from(n ? total / n : 0.0, input.dtype(), input.device());
}

Tensor binary_cross_entropy_cpu(const Tensor& input, const Tensor& target) {
    auto pr = bcast2(input, target);
    Tensor x = pr.first, t = pr.second;
    constexpr double kEps = 1e-12;
    int64_t n = x.numel();
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i) {
        double xv = std::min(std::max(xp[i], kEps), 1.0 - kEps);
        total += -(tp[i] * std::log(xv) + (1.0 - tp[i]) * std::log(1.0 - xv));
    }
    return scalar_from(n ? total / n : 0.0, input.dtype(), input.device());
}

Tensor cosine_embedding_loss_cpu(const Tensor& x1, const Tensor& x2, const Tensor& target,
                                 Scalar margin) {
    Tensor a = x1.contiguous().to(DType::Float64);
    Tensor b = x2.contiguous().to(DType::Float64);
    Tensor tg = target.contiguous().to(DType::Float64);
    int64_t N = a.size(0), D = a.size(1);
    if (tg.dim() == 0) tg = tg.expand({N}).contiguous();
    const double* ap = a.data_ptr<double>();
    const double* bp = b.data_ptr<double>();
    const double* gp = tg.data_ptr<double>();
    double mg = margin.toDouble();
    double total = 0;
    for (int64_t i = 0; i < N; ++i) {
        double dot = 0, na = 0, nbv = 0;
        for (int64_t j = 0; j < D; ++j) {
            dot += ap[i * D + j] * bp[i * D + j];
            na += ap[i * D + j] * ap[i * D + j];
            nbv += bp[i * D + j] * bp[i * D + j];
        }
        double cosv = dot / (std::sqrt(na) * std::sqrt(nbv) + 1e-12);
        total += (gp[i] == 1.0) ? 1.0 - cosv : std::max(0.0, cosv - mg);
    }
    return scalar_from(N ? total / N : 0.0, x1.dtype(), x1.device());
}

Tensor soft_margin_loss_cpu(const Tensor& input, const Tensor& target) {
    auto pr = bcast2(input, target);
    Tensor x = pr.first, t = pr.second;
    int64_t n = x.numel();
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i)
        total += softplus(-tp[i] * xp[i]);
    return scalar_from(n ? total / n : 0.0, input.dtype(), input.device());
}

Tensor triplet_margin_loss_cpu(const Tensor& anchor, const Tensor& positive,
                               const Tensor& negative, Scalar margin, double p) {
    int64_t N = anchor.size(0), D = anchor.size(1);
    Tensor a = anchor.contiguous().to(DType::Float64);
    Tensor pp2 = positive.contiguous().to(DType::Float64);
    Tensor nn2 = negative.contiguous().to(DType::Float64);
    const double* ap = a.data_ptr<double>();
    const double* ppos = pp2.data_ptr<double>();
    const double* pneg = nn2.data_ptr<double>();
    double mg = margin.toDouble();
    auto dist = [&](const double* u, const double* v) {
        if (p == std::numeric_limits<double>::infinity()) {
            double mx = 0;
            for (int64_t j = 0; j < D; ++j) mx = std::max(mx, std::fabs(u[j] - v[j]));
            return mx;
        }
        double s2 = 0;
        for (int64_t j = 0; j < D; ++j) s2 += std::pow(std::fabs(u[j] - v[j]), p);
        return std::pow(s2, 1.0 / p);
    };
    double total = 0;
    for (int64_t i = 0; i < N; ++i)
        total += std::max(0.0, dist(ap + i * D, ppos + i * D) -
                               dist(ap + i * D, pneg + i * D) + mg);
    return scalar_from(N ? total / N : 0.0, anchor.dtype(), anchor.device());
}

Tensor poisson_nll_loss_cpu(const Tensor& input, const Tensor& target, bool log_input,
                            bool full, double eps) {
    auto pr = bcast2(input, target);
    Tensor x = pr.first, z = pr.second;
    int64_t n = x.numel();
    const double* xp = x.data_ptr<double>();
    const double* zp = z.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < n; ++i) {
        double xv = xp[i], zv = zp[i];
        double l2 = log_input ? (std::exp(xv) - zv * xv)
                              : (xv - zv * std::log(std::exp(xv) + eps));
        if (full && zv > 0) l2 += zv * std::log(zv) - std::lgamma(zv + 1.0);
        total += l2;
    }
    return scalar_from(n ? total / n : 0.0, input.dtype(), input.device());
}

Tensor multi_margin_loss_cpu(const Tensor& input, const Tensor& target, Scalar margin,
                             double p) {
    Tensor x = input.contiguous().to(DType::Float64);
    Tensor tg = target.contiguous().to(DType::Float64);
    int64_t N = x.size(0), C = x.size(1);
    const double* xp = x.data_ptr<double>();
    const double* gp = tg.data_ptr<double>();
    double mg = margin.toDouble();
    double total = 0;
    for (int64_t i = 0; i < N; ++i) {
        int64_t y = static_cast<int64_t>(gp[i]);
        if (y < 0 || y >= C)
            TP_THROW(RuntimeError, "multi_margin_loss: label out of range");
        double row = 0;
        for (int64_t k2 = 0; k2 < C; ++k2) {
            if (k2 == y) continue;
            double v = mg - xp[i * C + y] + xp[i * C + k2];
            if (v > 0) row += (p == 2) ? v * v : v;
        }
        total += row / C;
    }
    return scalar_from(N ? total / N : 0.0, input.dtype(), input.device());
}

Tensor multilabel_soft_margin_loss_cpu(const Tensor& input, const Tensor& target) {
    auto pr = bcast2(input, target);
    Tensor x = pr.first, t = pr.second;
    int64_t N = x.size(0), C = x.size(1);
    const double* xp = x.data_ptr<double>();
    const double* tp = t.data_ptr<double>();
    double total = 0;
    for (int64_t i = 0; i < N; ++i) {
        double row = 0;
        for (int64_t c = 0; c < C; ++c) {
            double xv = xp[i * C + c], tv = tp[i * C + c];
            row += tv * -softplus(-xv) + (1.0 - tv) * -softplus(xv);
        }
        total += -row / C;
    }
    return scalar_from(N ? total / N : 0.0, input.dtype(), input.device());
}

Tensor multilabel_margin_loss_cpu(const Tensor& input, const Tensor& target, Scalar margin) {
    Tensor x = input.contiguous().to(DType::Float64);
    Tensor tg = target.contiguous().to(DType::Float64);
    int64_t N = x.size(0), C = x.size(1);
    const double* xp = x.data_ptr<double>();
    const double* tp = tg.data_ptr<double>();
    double mg = margin.toDouble();
    double total = 0;
    for (int64_t i = 0; i < N; ++i) {
        double row = 0;
        for (int64_t c = 0; c < C; ++c) {
            if (tp[i * C + c] != 1.0) continue;
            for (int64_t k2 = 0; k2 < C; ++k2) {
                if (tp[i * C + k2] != 0.0) continue;
                double v = mg - xp[i * C + c] + xp[i * C + k2];
                if (v > 0) row += v;
            }
        }
        total += row / C;
    }
    return scalar_from(N ? total / N : 0.0, input.dtype(), input.device());
}

TENSORPLAY_LIBRARY_IMPL(CPU, Tier5OpsKernels) {
    m.impl("addbmm", addbmm_cpu);
    m.impl("addmv", addmv_cpu);
    m.impl("addr", addr_cpu);
    m.impl("vdot", vdot_cpu);
    m.impl("cholesky", cholesky_cpu);
    m.impl("cholesky_inverse", cholesky_inverse_cpu);
    m.impl("cholesky_solve", cholesky_solve_cpu);
    m.impl("triangular_solve", triangular_solve_cpu);
    m.impl("svd", svd_cpu);
    m.impl("pairwise_distance", pairwise_distance_cpu);
    m.impl("pdist", pdist_cpu);
    m.impl("binary_cross_entropy_with_logits", binary_cross_entropy_with_logits_cpu);
    m.impl("hinge_embedding_loss", hinge_embedding_loss_cpu);
    m.impl("margin_ranking_loss", margin_ranking_loss_cpu);
}

TENSORPLAY_LIBRARY_IMPL(CPU, Tier5OpsKernelsB) {
    m.impl("real", real_cpu);
    m.impl("imag", imag_cpu);
    m.impl("conj", conj_cpu);
    m.impl("adjoint", adjoint_cpu);
    m.impl("complex", complex_cpu);
    m.impl("polar", polar_cpu);
    m.impl("lstm", lstm_cpu);
    m.impl("gru", gru_cpu);
    m.impl("rnn_relu", rnn_relu_cpu);
    m.impl("rnn_tanh", rnn_tanh_cpu);
}

TENSORPLAY_LIBRARY_IMPL(CPU, Tier5LossesKernels) {
    m.impl("l1_loss", l1_loss_cpu);
    m.impl("smooth_l1_loss", smooth_l1_loss_cpu);
    m.impl("huber_loss", huber_loss_cpu);
    m.impl("kl_div", kl_div_cpu);
    m.impl("binary_cross_entropy", binary_cross_entropy_cpu);
    m.impl("cosine_embedding_loss", cosine_embedding_loss_cpu);
    m.impl("soft_margin_loss", soft_margin_loss_cpu);
    m.impl("triplet_margin_loss", triplet_margin_loss_cpu);
    m.impl("poisson_nll_loss", poisson_nll_loss_cpu);
    m.impl("multi_margin_loss", multi_margin_loss_cpu);
    m.impl("multilabel_soft_margin_loss", multilabel_soft_margin_loss_cpu);
    m.impl("multilabel_margin_loss", multilabel_margin_loss_cpu);
}

} // namespace cpu
} // namespace tensorplay
