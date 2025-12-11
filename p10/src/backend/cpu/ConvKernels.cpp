#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include <vector>

namespace tensorplay {
namespace cpu {

Tensor conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, int64_t groups) {
    if (input.dim() != 4 || weight.dim() != 4) TP_THROW(RuntimeError, "conv2d: Expected 4D input and weight");
    
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    int64_t C_out = weight.size(0);
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);
    
    int64_t sH = stride.size() > 0 ? stride[0] : 1;
    int64_t sW = stride.size() > 1 ? stride[1] : sH;
    
    int64_t pH = padding.size() > 0 ? padding[0] : 0;
    int64_t pW = padding.size() > 1 ? padding[1] : pH;
    
    int64_t dH = dilation.size() > 0 ? dilation[0] : 1;
    int64_t dW = dilation.size() > 1 ? dilation[1] : dH;
    
    int64_t H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    int64_t W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    
    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "conv2d: Calculated output size is too small");

    Tensor out = Tensor::empty({N, C_out, H_out, W_out}, input.dtype(), input.device());
    
    if (input.dtype() == DType::Float32) {
        float* out_ptr = out.data_ptr<float>();
        const float* in_ptr = input.data_ptr<float>();
        const float* w_ptr = weight.data_ptr<float>();
        const float* b_ptr = (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;
        
        // Naive implementation
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c_out = 0; c_out < C_out; ++c_out) {
                float b_val = b_ptr ? b_ptr[c_out] : 0.0f;
                for (int64_t h_out = 0; h_out < H_out; ++h_out) {
                    for (int64_t w_out = 0; w_out < W_out; ++w_out) {
                        float sum = 0.0f;
                        for (int64_t c_in = 0; c_in < C_in; ++c_in) {
                            for (int64_t kh = 0; kh < kH; ++kh) {
                                for (int64_t kw = 0; kw < kW; ++kw) {
                                    int64_t h_in = h_out * sH - pH + kh * dH;
                                    int64_t w_in = w_out * sW - pW + kw * dW;
                                    
                                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                        // Input: (n, c_in, h_in, w_in)
                                        int64_t in_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                                        // Weight: (c_out, c_in, kh, kw)
                                        int64_t w_idx = ((c_out * C_in + c_in) * kH + kh) * kW + kw;
                                        
                                        sum += in_ptr[in_idx] * w_ptr[w_idx];
                                    }
                                }
                            }
                        }
                        // Output: (n, c_out, h_out, w_out)
                        int64_t out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                        out_ptr[out_idx] = sum + b_val;
                    }
                }
            }
        }
    } else {
        TP_THROW(NotImplementedError, "conv2d only supports Float32");
    }
    
    return out;
}

TENSORPLAY_REGISTER_KERNEL(conv2d, CPU, conv2d_kernel)

} // namespace cpu
} // namespace tensorplay
