#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
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

Tensor matmul_kernel(const Tensor& self, const Tensor& other) {
    int64_t dim1 = self.dim();
    int64_t dim2 = other.dim();

    if (dim1 == 2 && dim2 == 2) {
        return mm_kernel(self, other);
    }
    
    // Simple broadcasting implementation for > 2D
    // E.g. (B, N, M) @ (B, M, K) -> (B, N, K)
    // Or (B, N, M) @ (M, K) -> (B, N, K)
    
    if (dim1 >= 2 && dim2 >= 2) {
                // Broadcast batch dimensions
                std::vector<int64_t> self_shape_vec = static_cast<std::vector<int64_t>>(self.shape());
                std::vector<int64_t> other_shape_vec = static_cast<std::vector<int64_t>>(other.shape());

                std::vector<int64_t> batch_shape1(self_shape_vec.begin(), self_shape_vec.end() - 2);
                std::vector<int64_t> batch_shape2(other_shape_vec.begin(), other_shape_vec.end() - 2);
                
                std::vector<int64_t> batch_shape = broadcast_shapes(batch_shape1, batch_shape2);
                
                std::vector<int64_t> shape1 = batch_shape;
                shape1.push_back(self.size(dim1 - 2));
                shape1.push_back(self.size(dim1 - 1));
                
                std::vector<int64_t> shape2 = batch_shape;
                shape2.push_back(other.size(dim2 - 2));
                shape2.push_back(other.size(dim2 - 1));
                
                Tensor self_broadcasted = self.expand(shape1);
                Tensor other_broadcasted = other.expand(shape2);
                
                // Flatten batch dims
                int64_t batch_size = 1;
                for (auto s : batch_shape) batch_size *= s;
                
                int64_t M = self.size(dim1 - 2);
                int64_t K = self.size(dim1 - 1); // == other.size(dim2 - 2)
                int64_t N = other.size(dim2 - 1);
                
                // Fix result shape: last dim should be N
                std::vector<int64_t> res_shape = batch_shape;
                res_shape.push_back(M);
                res_shape.push_back(N);
                
                Tensor result = Tensor::empty(res_shape, self.dtype(), self.device());
                
                Tensor self_reshaped = self_broadcasted.reshape({batch_size, M, K});
                Tensor other_reshaped = other_broadcasted.reshape({batch_size, K, N});
                Tensor result_reshaped = result.reshape({batch_size, M, N});
                
                for (int64_t i = 0; i < batch_size; ++i) {
            Tensor s = self_reshaped.select(0, i);
            Tensor o = other_reshaped.select(0, i);
            Tensor r = mm_kernel(s, o);
            result_reshaped.select(0, i).copy_(r);
        }
        
        return result;
    }
    
    TP_THROW(NotImplementedError, "matmul: only 2D or batched 2D supported for now");
}

TENSORPLAY_REGISTER_KERNEL(mm, CPU, mm_kernel)
TENSORPLAY_REGISTER_KERNEL(matmul, CPU, matmul_kernel)

} // namespace cpu
} // namespace tensorplay
