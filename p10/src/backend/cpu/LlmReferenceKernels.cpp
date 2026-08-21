#include "Tensor.h"
#include "Dispatcher.h"
#include "Generator.h"
#include "Exception.h"
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>

// CPU reference implementations of the LLM hot-path operators. These are the
// canonical serial implementations; the CUDA kernels mirror their semantics
// (the dtype conversions below keep this path useful as a CPU oracle).
// (last-dim topk, [B, V] sample, [B, H, T, D] attention, etc).

namespace tensorplay {
namespace cpu {

namespace {

// Discrete sampling from a row of (unnormalized) weights in [0, cols).
int64_t sample_from_weights(const float* row, int64_t cols) {
  double total = 0.0;
  for (int64_t c = 0; c < cols; ++c) total += row[c];
  if (total <= 0.0) return 0;
  std::discrete_distribution<int64_t> dist(row, row + cols);
  return dist(default_generator().engine());
}

} // namespace

Tensor multinomial_kernel_cpu(const Tensor& self, int64_t num_samples, bool replacement, int64_t impl) {
  if (num_samples < 0) {
    TP_THROW(RuntimeError, "multinomial: num_samples must be >= 0");
  }
  Tensor prob = self.contiguous();
  if (prob.dim() > 2) {
    TP_THROW(RuntimeError, "multinomial: input must be 1D or 2D");
  }
  bool is_1d = prob.dim() == 1;
  int64_t rows = is_1d ? 1 : prob.size(0);
  int64_t cols = is_1d ? prob.numel() : prob.size(1);
  if (cols == 0 && num_samples > 0) {
    TP_THROW(RuntimeError, "multinomial: input must have at least one category");
  }
  std::vector<double> work;
  std::vector<double> wrow;
  Tensor f32;
  const float* pdata = nullptr;
  if (prob.dtype() == DType::Float32) {
    pdata = prob.data_ptr<float>();
  } else {
    f32 = prob.to(DType::Float32);
    pdata = f32.data_ptr<float>();
  }

  std::vector<int64_t> out_shape = is_1d ? std::vector<int64_t>{num_samples}
                                         : std::vector<int64_t>{rows, num_samples};
  Tensor result = Tensor::empty(out_shape, DType::Int64, prob.device());
  int64_t* rdata = result.data_ptr<int64_t>();
  if (num_samples == 0) return result;

  auto& gen = default_generator().engine();
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = pdata + r * cols;
    if (replacement) {
      std::discrete_distribution<int64_t> dist(row, row + cols);
      for (int64_t s = 0; s < num_samples; ++s) {
        rdata[is_1d ? s : r * num_samples + s] = dist(gen);
      }
    } else {
      // Without replacement: repeatedly sample from the remaining mass,
      // zeroing the chosen category (torch CPU semantics).
      std::vector<double> remaining(cols);
      for (int64_t c = 0; c < cols; ++c) remaining[c] = row[c];
      for (int64_t s = 0; s < num_samples; ++s) {
        std::discrete_distribution<int64_t> dist(remaining.begin(), remaining.end());
        int64_t pick = dist(gen);
        rdata[is_1d ? s : r * num_samples + s] = pick;
        remaining[pick] = 0.0;
      }
    }
  }
  return result;
}

std::tuple<Tensor, Tensor> topk_kernel_cpu(const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted, int64_t impl) {
  Tensor input = self.contiguous();
  int64_t ndim = input.dim();
  if (dim < 0) dim += ndim;
  if (dim != ndim - 1) {
    TP_THROW(NotImplementedError, "topk: only the last dim is supported for now");
  }
  int64_t rows = input.numel() / input.size(ndim - 1);
  int64_t cols = input.size(ndim - 1);
  if (k < 0 || k > cols) {
    TP_THROW(RuntimeError, "topk: k must be in [0, cols]");
  }
  if (input.dtype() != DType::Float32) {
    TP_THROW(NotImplementedError, "topk: only float32 is supported for now");
  }

  std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(input.shape());
  shape[ndim - 1] = k;
  Tensor values = Tensor::empty(shape, input.dtype(), input.device());
  Tensor indices = Tensor::empty(shape, DType::Int64, input.device());
  if (k == 0) return {values, indices};

  const float* idata = input.data_ptr<float>();
  float* vdata = values.data_ptr<float>();
  int64_t* idxdata = indices.data_ptr<int64_t>();

  std::vector<int64_t> perm(cols);
  std::iota(perm.begin(), perm.end(), 0);
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = idata + r * cols;
    std::partial_sort(perm.begin(), perm.begin() + k, perm.end(),
                      [row, largest](int64_t a, int64_t b) {
                        return largest ? (row[a] > row[b]) : (row[a] < row[b]);
                      });
    for (int64_t i = 0; i < k; ++i) {
      vdata[r * k + i] = row[perm[i]];
      idxdata[r * k + i] = perm[i];
    }
  }
  return {values, indices};
}

Tensor sample_kernel_cpu(const Tensor& logits, double temperature, int64_t top_k, double top_p, int64_t impl) {
  if (temperature <= 0) {
    TP_THROW(RuntimeError, "sample: temperature must be > 0");
  }
  if (top_p <= 0 || top_p > 1.0) {
    TP_THROW(RuntimeError, "sample: top_p must be in (0, 1]");
  }
  Tensor input = logits.contiguous();
  if (input.dim() != 2) {
    TP_THROW(RuntimeError, "sample: logits must be 2D [batch, vocab]");
  }
  if (input.dtype() != DType::Float32) {
    input = input.to(DType::Float32);
  }
  int64_t rows = input.size(0);
  int64_t cols = input.size(1);
  if (cols == 0) {
    TP_THROW(RuntimeError, "sample: vocab size must be > 0");
  }

  Tensor result = Tensor::empty({rows}, DType::Int64, input.device());
  const float* idata = input.data_ptr<float>();
  int64_t* rdata = result.data_ptr<int64_t>();
  auto& gen = default_generator().engine();

  std::vector<float> probs(cols);
  std::vector<int64_t> order(cols);
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = idata + r * cols;
    // temperature + softmax (double precision, like the torch reference)
    double mx = -INFINITY;
    for (int64_t c = 0; c < cols; ++c) mx = std::max(mx, (double)row[c]);
    double total = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      double e = std::exp(((double)row[c] - mx) / temperature);
      probs[c] = (float)e;
      total += e;
    }
    for (int64_t c = 0; c < cols; ++c) probs[c] = (float)(probs[c] / total);

    // top-k filter (exact, ties by lower index)
    if (top_k > 0 && top_k < cols) {
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                        [&probs](int64_t a, int64_t b) { return probs[a] > probs[b]; });
      std::vector<char> keep(cols, 0);
      for (int64_t i = 0; i < top_k; ++i) keep[order[i]] = 1;
      for (int64_t c = 0; c < cols; ++c)
        if (!keep[c]) probs[c] = 0.f;
    }
    // top-p filter (nucleus, ties kept together)
    if (top_p < 1.0) {
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&probs](int64_t a, int64_t b) {
                  return probs[a] > probs[b] || (probs[a] == probs[b] && a < b);
                });
      double cum = 0.0;
      int64_t keep_n = 0;
      for (int64_t i = 0; i < cols; ++i) {
        cum += probs[order[i]];
        ++keep_n;
        if (cum >= top_p) break;
      }
      std::vector<char> keep(cols, 0);
      for (int64_t i = 0; i < keep_n; ++i) keep[order[i]] = 1;
      for (int64_t c = 0; c < cols; ++c)
        if (!keep[c]) probs[c] = 0.f;
    }

    std::discrete_distribution<int64_t> dist(probs.begin(), probs.end());
    rdata[r] = dist(gen);
  }
  return result;
}

Tensor sdpa_kernel_cpu(const Tensor& query, const Tensor& key, const Tensor& value, bool is_causal, int64_t impl) {
  Tensor q = query.contiguous();
  Tensor k = key.contiguous();
  Tensor v = value.contiguous();
  if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) {
    TP_THROW(RuntimeError, "sdpa: query/key/value must be 4D [B, H, T, D]");
  }
  int64_t B = q.size(0), H = q.size(1), T = q.size(2), D = q.size(3);
  if (k.size(0) != B || k.size(1) != H || v.size(0) != B || v.size(1) != H ||
      k.size(2) != T || v.size(2) != T || k.size(3) != D || v.size(3) != D) {
    TP_THROW(RuntimeError, "sdpa: key/value shapes must match [B, H, T, D]");
  }
  if (q.dtype() != DType::Float32 && q.dtype() != DType::Float16 &&
      q.dtype() != DType::BFloat16) {
    TP_THROW(NotImplementedError,
             "sdpa cpu: expected float32, float16 or bfloat16");
  }
  const DType original_dtype = q.dtype();
  if (original_dtype != DType::Float32) {
    q = q.to(DType::Float32);
    k = k.to(DType::Float32);
    v = v.to(DType::Float32);
  }
  Tensor out = Tensor::empty({B, H, T, D}, DType::Float32, q.device());
  const float* qd = q.data_ptr<float>();
  const float* kd = k.data_ptr<float>();
  const float* vd = v.data_ptr<float>();
  float* od = out.data_ptr<float>();
  float scale = 1.f / sqrtf((float)D);

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      const float* qh = qd + ((b * H + h) * T) * D;
      const float* kh = kd + ((b * H + h) * T) * D;
      const float* vh = vd + ((b * H + h) * T) * D;
      float* oh = od + ((b * H + h) * T) * D;
      std::vector<float> scores(T);
      for (int64_t t = 0; t < T; ++t) {
        const float* qrow = qh + t * D;
        float mx = -INFINITY;
        for (int64_t kk = 0; kk < T; ++kk) {
          float s = (is_causal && kk > t) ? -INFINITY : 0.f;
          if (s != -INFINITY) {
            const float* krow = kh + kk * D;
            for (int64_t d = 0; d < D; ++d) s += qrow[d] * krow[d];
            s *= scale;
          }
          scores[kk] = s;
          mx = std::max(mx, s);
        }
        double total = 0.0;
        for (int64_t kk = 0; kk < T; ++kk) {
          double e = std::exp((double)scores[kk] - mx);
          scores[kk] = (float)e;
          total += e;
        }
        float* orow = oh + t * D;
        for (int64_t d = 0; d < D; ++d) {
          double acc = 0.0;
          for (int64_t kk = 0; kk < T; ++kk)
            acc += scores[kk] * vh[kk * D + d];
          orow[d] = (float)(acc / total);
        }
      }
    }
  }
  return original_dtype == DType::Float32 ? out : out.to(original_dtype);
}

std::tuple<Tensor, Tensor, Tensor> sdpa_backward_kernel_cpu(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, bool is_causal, int64_t impl) {
  (void)impl;
  Tensor q = query.contiguous();
  Tensor k = key.contiguous();
  Tensor v = value.contiguous();
  Tensor go = grad_output.contiguous();
  if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4 || go.dim() != 4) {
    TP_THROW(RuntimeError, "sdpa backward: q/k/v/grad_output must be 4D");
  }
  const int64_t B = q.size(0), H = q.size(1), T = q.size(2), D = q.size(3);
  if (k.size(0) != B || k.size(1) != H || k.size(2) != T || k.size(3) != D ||
      v.size(0) != B || v.size(1) != H || v.size(2) != T || v.size(3) != D ||
      go.size(0) != B || go.size(1) != H || go.size(2) != T || go.size(3) != D) {
    TP_THROW(RuntimeError, "sdpa backward: q/k/v/grad_output shapes must match");
  }
  if (q.dtype() != DType::Float32 && q.dtype() != DType::Float16 &&
      q.dtype() != DType::BFloat16) {
    TP_THROW(NotImplementedError,
             "sdpa backward CPU: expected float32, float16 or bfloat16");
  }
  if (k.dtype() != q.dtype() || v.dtype() != q.dtype() || go.dtype() != q.dtype()) {
    TP_THROW(RuntimeError, "sdpa backward: q/k/v/grad_output dtypes must match");
  }

  const DType original_dtype = q.dtype();
  if (original_dtype != DType::Float32) {
    q = q.to(DType::Float32);
    k = k.to(DType::Float32);
    v = v.to(DType::Float32);
    go = go.to(DType::Float32);
  }
  const float* qd = q.data_ptr<float>();
  const float* kd = k.data_ptr<float>();
  const float* vd = v.data_ptr<float>();
  const float* god = go.data_ptr<float>();
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int64_t rows = B * H;

  Tensor probs = Tensor::empty({B, H, T, T}, DType::Float32, q.device());
  Tensor dprob = Tensor::empty({B, H, T, T}, DType::Float32, q.device());
  Tensor dscore = Tensor::empty({B, H, T, T}, DType::Float32, q.device());
  float* pd = probs.data_ptr<float>();
  float* dpd = dprob.data_ptr<float>();
  float* dsd = dscore.data_ptr<float>();

  for (int64_t bh = 0; bh < rows; ++bh) {
    const float* qh = qd + bh * T * D;
    const float* kh = kd + bh * T * D;
    for (int64_t t = 0; t < T; ++t) {
      float max_score = -INFINITY;
      for (int64_t kk = 0; kk < T; ++kk) {
        float score = -INFINITY;
        if (!is_causal || kk <= t) {
          score = 0.0f;
          for (int64_t d = 0; d < D; ++d) score += qh[t * D + d] * kh[kk * D + d];
          score *= scale;
        }
        pd[(bh * T + t) * T + kk] = score;
        max_score = std::max(max_score, score);
      }
      float total = 0.0f;
      for (int64_t kk = 0; kk < T; ++kk) {
        float p = (pd[(bh * T + t) * T + kk] == -INFINITY)
            ? 0.0f : std::exp(pd[(bh * T + t) * T + kk] - max_score);
        pd[(bh * T + t) * T + kk] = p;
        total += p;
      }
      for (int64_t kk = 0; kk < T; ++kk) pd[(bh * T + t) * T + kk] /= total;
    }
  }

  for (int64_t bh = 0; bh < rows; ++bh) {
    const float* gh = god + bh * T * D;
    const float* vh = vd + bh * T * D;
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t kk = 0; kk < T; ++kk) {
        float dot = 0.0f;
        for (int64_t d = 0; d < D; ++d) dot += gh[t * D + d] * vh[kk * D + d];
        dpd[(bh * T + t) * T + kk] = dot;
      }
      float row_dot = 0.0f;
      for (int64_t kk = 0; kk < T; ++kk) {
        row_dot += dpd[(bh * T + t) * T + kk] * pd[(bh * T + t) * T + kk];
      }
      for (int64_t kk = 0; kk < T; ++kk) {
        dsd[(bh * T + t) * T + kk] =
            pd[(bh * T + t) * T + kk] * (dpd[(bh * T + t) * T + kk] - row_dot) * scale;
      }
    }
  }

  Tensor d_q = Tensor::zeros({B, H, T, D}, DType::Float32, q.device());
  Tensor d_k = Tensor::zeros({B, H, T, D}, DType::Float32, q.device());
  Tensor d_v = Tensor::zeros({B, H, T, D}, DType::Float32, q.device());
  float* dqd = d_q.data_ptr<float>();
  float* dkd = d_k.data_ptr<float>();
  float* dvd = d_v.data_ptr<float>();
  for (int64_t bh = 0; bh < rows; ++bh) {
    const float* qh = qd + bh * T * D;
    const float* kh = kd + bh * T * D;
    const float* gh = god + bh * T * D;
    const float* vh = vd + bh * T * D;
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t d = 0; d < D; ++d) {
        float q_acc = 0.0f;
        for (int64_t kk = 0; kk < T; ++kk) q_acc += dsd[(bh * T + t) * T + kk] * kh[kk * D + d];
        dqd[(bh * T + t) * D + d] = q_acc;
      }
    }
    for (int64_t kk = 0; kk < T; ++kk) {
      for (int64_t d = 0; d < D; ++d) {
        float k_acc = 0.0f, v_acc = 0.0f;
        for (int64_t t = 0; t < T; ++t) {
          k_acc += dsd[(bh * T + t) * T + kk] * qh[t * D + d];
          v_acc += pd[(bh * T + t) * T + kk] * gh[t * D + d];
        }
        dkd[(bh * T + kk) * D + d] = k_acc;
        dvd[(bh * T + kk) * D + d] = v_acc;
      }
    }
  }
  if (original_dtype != DType::Float32) {
    return {d_q.to(original_dtype), d_k.to(original_dtype), d_v.to(original_dtype)};
  }
  return {d_q, d_k, d_v};
}

TENSORPLAY_LIBRARY_IMPL(CPU, LlmReferenceKernels) {
  m.impl("multinomial", multinomial_kernel_cpu);
  m.impl("topk", topk_kernel_cpu);
  m.impl("sample", sample_kernel_cpu);
  m.impl("scaled_dot_product_attention", sdpa_kernel_cpu);
  m.impl("scaled_dot_product_attention_backward", sdpa_backward_kernel_cpu);
}

} // namespace cpu
} // namespace tensorplay
