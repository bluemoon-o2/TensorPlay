# TensorPlay Python 侧 vs torch 差距报告

- 日期: 2026-08-21
- 对照基准: `third_party/pytorch`(2.15.0a0,裁剪树,无 `nn/`、`cuda/`、`autograd/` 目录)+ 本机完整 torch 2.13.0+cu130 运行时
- TensorPlay: 1.0.0rc0,AST 静态枚举(报告当日 `import tensorplay` 因 `_C.so` 落后于 Python 层而失败,见文末)

## 总览

| 命名空间 | torch | tensorplay | 覆盖评价 |
|---|---|---|---|
| 顶层符号 | 1025 | ~348 | 核心可用,缺域子包与全局配置 |
| nn.functional | 160 | 252 | 核心 op 超前覆盖(含 tp 特有 backward 函数),缺 loss/pool/dropout 家族 |
| nn 导出类 | 182 | 49 | 最大差距区 |
| Tensor 方法 | 603 | ~170(144 绑定 + 15 property + 12 Python) | 缺多个整方法族 |
| optim | 19 | 14 优化器 + 16/16 scheduler + swa_utils | 基本持平 |
| cuda | 147 | 38 | 基础够用,缺 RNG/graph/memory 细分 |
| utils.data | 36 | 完整 DataLoader/Sampler/DistributedSampler/collate | 接近持平 |

## 1. nn 层(最大缺口)

- RNN 族全缺:`RNN/LSTM/GRU` + 3 个 Cell、`RNNBase`;Transformer 全家(Encoder/Decoder/Layer ×2)、`MultiheadAttention`
- Padding 模块 15 个全缺:ConstantPad*/ReflectionPad*/ReplicationPad*/ZeroPad*/CircularPad*
- loss 只有 3 个(MSELoss/NLLLoss/CrossEntropy),torch 约 20 个:BCE(_WithLogits)/KLDiv/L1/Huber/SmoothL1/CosineEmbedding/MarginRanking/TripletMargin/HingeEmbedding/MultiMargin/MultiLabel*/PoissonNLL/GaussianNLL/CTC
- 激活模块缺一半:ELU/SELU/CELU/LeakyReLU/ReLU6/Hardtanh/Hardswish/Hardsigmoid/Softplus/Softmin/LogSoftmax/Softmax/GLU/Mish/RReLU 等(GELU/SiLU/PReLU/Sigmoid/Tanh/Threshold 已有)
- 结构类:Fold/Unfold、Upsample(+UpsamplingNearest2d/Bilinear2d)、PixelShuffle/Unshuffle、ChannelShuffle、EmbeddingBag、AdaptiveLogSoftmaxWithLoss、CosineSimilarity、PairwiseDistance、DataParallel、SyncBatchNorm
- Lazy 系列:只有 LazyBatchNorm*(LazyModuleMixin 已有,LazyLinear/LazyConv* 成本低);InstanceNorm1d/2d/3d 已实现未导出
- ConvTranspose1d 缺(2d/3d 有)

## 2. functional 层

缺 122 个,按族归类:

- loss 函数族:l1_loss、kl_div、binary_cross_entropy(_with_logits)、huber_loss、smooth_l1_loss、cosine_embedding_loss、margin_ranking_loss、triplet_margin_loss(_with_distance)、hinge_embedding_loss、multi_margin_loss、multilabel_(margin|soft_margin)_loss、poisson_nll_loss、gaussian_nll_loss、soft_margin_loss、ctc_loss、cross_entropy 组合入口
- pool:avg_pool1d/3d、adaptive_*(1d/3d 及 *_with_indices)、max_pool1d/3d(+with_indices)、max_unpool*
- 结构:unfold/fold、grid_sample、affine_grid、interpolate、pad(多模式)、pixel_shuffle/unshuffle、channel_shuffle
- dropout 函数族:dropout/1d/2d/3d/alpha_dropout/feature_dropout
- 其他:`F.linear`(下游强依赖)、normalize、rms_norm、local_response_norm、embedding_bag、multi_head_attention_forward、gumbel_softmax、logsigmoid、softshrink/softsign/tanhshrink/hardshrink、cosine_similarity、pairwise_distance、pdist、inplace 激活变体(elu_/selu_/celu_...)
- SDPA 有正向+backward,无 backend 选择参数(flash/mem_efficient/math)

## 3. Tensor 方法面(603 vs ~170)

整族缺失:

- index/scatter-gather 族:index_select/index_add/index_copy/index_fill/index_put/index_reduce/scatter/scatter_add/scatter_reduce/gather/take/take_along_dim
- 累积规约:cumsum/cumprod/cummax/cummin/logsumexp/nansum/nanmean/mode/kthvalue/quantile/unique/bincount/histogram
- bitwise/logical 全族(~30 个)
- linalg 方法:cholesky/eig/svd/qr/solve/inverse/det/logdet/lstsq/lu/matrix_power/matrix_exp
- 复数支持方法:real/imag/conj/view_as_complex(F 层有,Tensor 方法无)
- new_* 工厂(new_full/new_zeros/new_empty/new_tensor...)、resize_/resize_as_、expand_as/repeat_interleave、flip/fliplr/rot90/tile/unravel_index
- autograd hook:register_hook/register_post_accumulate_grad_hook/retain_grad
- 属性:strides/storage_offset/nbytes/element_size(numel()/itemsize() 有)

## 4. 顶层与其他域

- 域子包缺:linalg、fft、special、distributions(sparse 有部分)
- 全局配置缺:set_default_dtype/device、use_deterministic_algorithms、set_float32_matmul_precision、inference_mode 未挂顶层、get_default_device
- 工具缺:from_numpy/frombuffer、broadcast_tensors/broadcast_shapes、atleast_*d、hstack/vstack/dsplit/column_stack、tensor_split/unravel_index/block_diag/tensordot
- autograd:gradcheck/gradgradcheck、detect_anomaly、autograd.functional(jacobian/hessian/vjp/jvp)缺;核心 grad 语义已齐
- cuda:~~缺 RNG 入口、CUDAGraph 整族、memory_stats 细分/snapshot/reset、nccl/nvtx/profiler 子包、can_device_access_peer、OOM 异常类型~~ **2026-08-21 已对齐**(见下节)
- distributed:c10d 集合通信 + TCPStore/FileStore 可用;缺 DDP/FSDP/RPC/Pipeline/DTensor/checkpoint
- utils.data:仅缺 DataPipe(torch 已弃用,低优先级)

## 6. cuda 模块对齐(2026-08-21 完成)

以三方源码 `third_party/pytorch/torch/cuda/`(commit 893b640, 2026-08-17)为基准逐文件照抄适配,`tensorplay/cuda.py` 重构为包:

| 文件 | 来源 | 说明 |
|---|---|---|
| `__init__.py` | torch/cuda/\_\_init\_\_.py | lazy-init/_lazy_call/device guard/NVML 设备数与指标/StreamContext/arch 兼容检查全套保留 |
| `_utils.py` | torch/_utils.py + cuda/_utils.py | `_dummy_type/_ClassPropertyDescriptor/classproperty/_LazySeedTracker` 原样移植 |
| `streams.py` | cuda/streams.py | Stream/Event/ExternalStream,补齐 stream_id/device_index/device_type 属性与 `\_\_repr__` 格式 |
| `memory.py` | cuda/memory.py | memory_stats 全键矩阵(未跟踪项报 0,同 torch cudaMallocAsync 先例)、summary 格式器原样 |
| `random.py` | cuda/random.py | manual_seed(_all)/seed(_all)/initial_seed(initial_seed 由 Python 侧跟踪);state API 报诚实错误 |
| `graphs.py`/`graph_annotations.py`/`nvtx.py`/`nccl.py`/`profiler.py`/`tunable.py`/`jiterator.py`/`gds.py`/`green_contexts.py`/`sparse.py` | 同名文件 | 公开名全在;依赖 native 的按 torch"无 CUDA 支持"模式降级 |
| ~~`amp/`~~ | — | **已删除**(2026-08-21):顶层 `tensorplay.amp` 已覆盖,无引用方;与 torch 的对齐在此处有意收窄 |

验证:`__all__` 与 torch.cuda 128 项对齐(仅排除 22 个 legacy Storage/Tensor 类,tp 无 typed storage);桩注入冒烟 9 项全过(import/init/device ctx/stream+ctx/Event/memory 家族/random 家族/降级 API)。

**待 native 补齐后点亮**(Python 已就位):per-device CUDA generator state(get/set_rng_state)、CUDA graph capture、caching allocator 扩展接口、NVRTC/jiterator、NVML 需 `pip install nvidia-ml-py`。
**注意**:legacy Storage/Tensor 类不提供(tp 无 typed storage);`PYTORCH_RELEASES_CODE_CC` 为 torch 发布渠道专用,未移植。

## 5. TP 反超/特色(非差距)

自研 compiler 栈(`tp.compile`)、backward 算子直接暴露(conv2d_grad_input 等)、`_serialization_torch` 与 torch checkpoint 互操作、vision/audio/hub 扩展、`bcomplex32` dtype、DepthwiseConv2d。

## 建议补齐优先级

1. `F.linear`(2026-08-21 已完成:`tensorplay/nn/functional.py` 逐分支对齐 `aten/src/ATen/native/Linear.cpp`,含 addmm 融合、`_flatten_nd_linear`、`TORCH_LINEAR_FLATTEN_3D`、sparse weight 分支;`Tensor.contiguous()` 经 native_functions.yaml 补齐)
2. RNN/LSTM(2026-08-21 已完成:`tensorplay/nn/modules/rnn.py` 移植 `aten/src/ATen/native/RNN.cpp` 的 CPU cell/层堆叠机制与 `torch/nn/modules/rnn.py` 模块 API(RNN/LSTM/GRU/RNNBase + 三 Cell,含 bidirectional/batch_first/dropout/proj_size/PackedSequence);配套新增 `tensorplay/nn/utils/rnn.py`(PackedSequence/pack_padded_sequence 等)与缺失算子 `scatter_.src/scatter_.value/scatter_add_`(CPU+CUDA)、`narrow`、顶层 `as_tensor`)。Transformer 仍缺
3. loss 模块族
4. index/scatter 方法族
5. padding/Upsample/Fold
6. linalg 域
7. **cuda 模块对齐**(2026-08-21 已立项,见 git history)
