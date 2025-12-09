#include "tensorplay/core/Tensor.h"
#include "tensorplay/core/Dispatcher.h"
#include "tensorplay/core/Scalar.h"
#include "tensorplay/core/Exception.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef USE_MKL
#include <mkl.h>
#elif defined(USE_BLAS)
#include <cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorplay {
namespace cpu {

namespace {

// Naive matrix multiplication implementation (Optimized Loop Order M-K-N)
void gemm_naive(int64_t M, int64_t N, int64_t K, float alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float beta, float* C, int64_t ldc) {
    // Scale C by beta first
    if (beta == 0.0f) {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int64_t m = 0; m < M; ++m) {
            std::memset(C + m * ldc, 0, N * sizeof(float));
        }
    } else if (beta != 1.0f) {
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                C[m * ldc + n] *= beta;
            }
        }
    }

    // Accumulate alpha * A * B
    // Loop order M-K-N optimizes cache access for RowMajor matrices B and C
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = alpha * A[m * lda + k];
            // Vectorization friendly inner loop
            for (int64_t n = 0; n < N; ++n) {
                C[m * ldc + n] += a_val * B[k * ldb + n];
            }
        }
    }
}

} // anonymous namespace

Tensor mm_kernel(const Tensor& self, const Tensor& mat2) {
    if (self.dim() != 2 || mat2.dim() != 2) TP_THROW(RuntimeError, "mm: expected 2D tensors");
    if (self.size(1) != mat2.size(0)) TP_THROW(RuntimeError, "mm: shape mismatch");
    
    int64_t M = self.size(0);
    int64_t K = self.size(1);
    int64_t N = mat2.size(1);
    
    Tensor result = Tensor::empty({M, N}, self.dtype(), self.device());
    
    if (self.dtype() == DType::Float32 && mat2.dtype() == DType::Float32) {
        
        bool transA = false;
        bool transB = false;
        int64_t lda = 0;
        int64_t ldb = 0;
        
        // Helper to check for transposed layout (stride(0)=1, stride(1)=size(0))
        // This corresponds to a Column-Major layout of a matrix of size (size(0), size(1))
        // or a Transposed view of a Row-Major matrix.
        auto is_transposed = [](const Tensor& t) {
            return t.stride(0) == 1 && t.stride(1) == t.size(0);
        };
        
        Tensor a_input = self;
        if (self.is_contiguous()) {
            lda = K;
        } else if (is_transposed(self)) {
            transA = true;
            lda = M; 
        } else {
            a_input = self.clone();
            lda = K;
        }
        
        Tensor b_input = mat2;
        if (mat2.is_contiguous()) {
            ldb = N;
        } else if (is_transposed(mat2)) {
            transB = true;
            ldb = K;
        } else {
            b_input = mat2.clone();
            ldb = N;
        }
        
        const float* A = a_input.data_ptr<float>();
        const float* B = b_input.data_ptr<float>();
        float* C = result.data_ptr<float>();
        
        #if defined(USE_MKL) || defined(USE_BLAS)
            CBLAS_TRANSPOSE TransA = transA ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE TransB = transB ? CblasTrans : CblasNoTrans;
            
            #ifdef USE_MKL
            cblas_sgemm(CblasRowMajor, TransA, TransB, 
                        M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, N);
            #else
            cblas_sgemm(CblasRowMajor, TransA, TransB, 
                        (int)M, (int)N, (int)K, 1.0f, A, (int)lda, B, (int)ldb, 0.0f, C, (int)N);
            #endif
            
        #else
            // Fallback to naive (no transpose support in naive yet, force clone)
            if (transA || transB) {
                 Tensor a_contig = self.is_contiguous() ? self : self.clone();
                 Tensor b_contig = mat2.is_contiguous() ? mat2 : mat2.clone();
                 gemm_naive(M, N, K, 1.0f, a_contig.data_ptr<float>(), K, b_contig.data_ptr<float>(), N, 0.0f, C, N);
            } else {
                 gemm_naive(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
            }
        #endif
        
    } else {
         TP_THROW(NotImplementedError, "mm: only Float32 supported for now");
    }
    
    return result;
}

TENSORPLAY_REGISTER_KERNEL(mm, CPU, mm_kernel)

} // namespace cpu
} // namespace tensorplay
