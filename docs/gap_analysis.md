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
- distributed:c10d 集合通信 + TCPStore/FileStore 可用;~~缺 DDP/FSDP/RPC/Pipeline/DTensor/checkpoint~~
  **2026-08-22 已对齐 DDP 与对象集合通信**(见下节);~~仍缺 FSDP/RPC/Pipeline/DTensor/dist.checkpoint~~
  **2026-08-22 dist.checkpoint 已落地**(见下节);仍缺 FSDP/RPC/Pipeline/DTensor/elastic launch(torchrun)
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

## 7. distributed 模块对齐(2026-08-22,持续更新)

以三方源码 `third_party/pytorch/torch/distributed/` 与
`torch/nn/parallel/` 为基准照抄适配。布局对齐 torch(薄 `__init__` +
`distributed_c10d.py` 主体 + 各域子包),共 72 个 py 文件。

### 已完成(可用实现)

| tp 文件 | torch 来源 | 说明 |
|---|---|---|
| `distributed_c10d.py` | distributed_c10d.py | 全部集合通信(all_reduce/broadcast/reduce/all_gather/gather/scatter/reduce_scatter/send/recv/isend/irecv/barrier/all_to_all(_single)/all_gather_into_tensor/_allgather_base/reduce_scatter_tensor/_reduce_scatter_base)、6 个对象集合通信、rank 换算(get_group_rank/get_global_rank/get_process_group_ranks)、`new_subgroups(_by_enumeration)`、`_compute_bucket_assignment_by_size`、`_broadcast_coalesced`(按 dtype/device 分桶+双在飞)、`_verify_params_across_processes`、GradBucket、P2POp/batch_isend_irecv(NCCL group 包裹)、Backend/GroupMember.WORLD |
| `_store.py` | c10d Store | FileStore(flock)/TCPStore(线程服务器) |
| `rendezvous.py` | rendezvous.py | env://file://tcp:// 三处理器+注册机制 |
| `constants.py`/`logging_handlers.py`/`c10d_logger.py` | 同名 | 默认超时、日志 handler 表、`_exception_logger/_time_logger` |
| `utils.py` | utils.py | `_sync_module_states/_pack_kwargs/_apply_to_tensors/_recursive_to/_to_kwargs/_replace_by_prefix` 等 |
| `collective_utils.py` | collective_utils.py | SyncPayload/broadcast/all_gather 原语+类型校验;RNG 同步检查待 Generator state API |
| `nn/parallel/distributed.py` | nn/parallel/distributed.py | DDP:构造期状态广播、真分桶梯度归约(post-accumulate hook+按序 flush)、find_unused_parameters(输出图遍历+未用参数零填充)、static_graph(首遍遍历缓存)、register_comm_hook(Future 契约)、no_sync、buffer coalesced 广播、join()(Join 子包接入) |
| `algorithms/join.py` | algorithms/join.py | Join/Joinable/JoinHook 完整移植 |
| `algorithms/ddp_comm_hooks/` | 同名 | default(allreduce/fp16/bf16+wrapper)、powerSGD(powerSGD_hook/batched,QR/Gram-Schmidt 正交化+错误反馈+warm start)、post_localSGD、quantization(pertensor/perchannel,tp 无 ao 内嵌最小 MinMax 观察器)、debugging(noop)、mixed_precision、optimizer_overlap |
| `algorithms/model_averaging/` | 同名 | Periodic/Hierarchical averager+utils |
| `algorithms/_checkpoint/checkpoint_wrapper.py` | 同名 | checkpoint_wrapper/apply_activation_checkpointing;offload 需 save_on_cpu(挂起) |
| `optim/` | optim/* | 9 个 functional 优化器、_NamedOptimizer、PostLocalSGDOptimizer、apply_optimizer_in_backward、as_functional_optim;**ZeroRedundancyOptimizer 移植挂起**(1700 行) |
| `run.py`/`launch.py` | run.py/launch.py | 单节点 torchrun 等价:LaunchConfig/elastic_launch/main,子进程 worker+env:// rendezvous |
| `autograd/`,`rpc/`,`nn/api/` | 同名 | RPC 运行时依赖项,入口报与 torch 未初始化 RPC 一致的错误 |
| `futures`(顶层) | torch.futures | Future(set_result/wait/value/then/add_done_callback);Work.get_future() 对齐 |
| C++ 层 | ProcessGroupNCCL+torch/csrc/cuda/nccl.cpp | allToAll 等分/不等分(group send/recv 形式)、groupStart/groupEnd、Tensor.element_size()、Node._raw_ptr |

### 第二轮补齐(2026-08-22 续)

- `optim/zero_rendancy_optimizer.py`→`zero_redundancy_optimizer.py`:ZeRO 全量
  (sorted-greedy 分片、consolidate_state_dict 广播收集、state_dict 本地/全局索引换算、
  parameters_as_bucket_view、overlap_with_ddp 延迟初始化+DDPBucketAssignment、Join hook);
  `_broadcast_object` 用 pickle 桥(tp.Tensor 可 pickle)
- `algorithms/ddp_comm_hooks/ddp_zero_hook.py`:hook_with_zero_step(_interleaved)
- `device_mesh.py`:DeviceMesh/init_device_mesh/get_group/get_local_rank/get_coordinate/
  子 mesh 切片/上下文管理器/from_group(布局内部用 sizes/strides,torch 的 _pycute 不移植)
- `checkpoint/`:save/async_save/load + FileSystem{Reader,Writer}(合并式单文件布局,
  torch 分片 planner 写入路径挂起)
- `_composable/`:replicate(composable DDP)+contract;fully_shard 报 DTensor 依赖错误
- `nn/api`、`rpc/`、`autograd/`:RPC 运行时门控入口

### 待办(按优先级,2026-08-22 收尾时状态)

**⚠ 阻塞项:构建+测试未执行**(当日共享树内存被他任务占满,用户暂停构建)。
全部 72 个 py 文件已过 `compileall` 与静态导入一致性/`__all__` 一致性检查,
但**运行时行为零验证**。恢复后第一步:
`ninja -C build -j4 _C`(等静默+内存≥8G),然后 `python test/test_distributed.py`
(NCCL 子进程脚本覆盖全部新 API;单 rank 任意 CUDA 机即可跑)。
C++ 改动:NCCLContext.{h,cpp}(allToAll/group)、Distributed.cpp(all_to_all 绑定)、
Autograd.cpp(Node._raw_ptr)、Tensor.cpp(element_size)。产物新鲜度需比对 mtime。

1. **tensor/**(DTensor):dispatcher 集成,深度大——device_mesh 已就绪可承接
2. **fsdp/**(FullyShardedDataParallel v2)+ `_composable/fsdp/fully_shard`(依赖 DTensor;
   入口已挂"DTensor 未移植"报错)
3. **pipelining/**:FX 式模型切分
4. elastic 多机 agent(c10d/etcd rendezvous);现有 run.py 仅单节点子进程启动器
5. `debug/`:ModTracker 依赖 `register_multi_grad_hook`(tp autograd 缺);
   全局 module hook 已存在(_global_forward_hooks)
6. `_functional_collectives` 的 *_coalesced 族(C++ coalescing manager)
7. RNG 同步检查(collective_utils._check_rng_sync)待 tp Generator state API
   (get/set_rng_state 已在 Python 层引用、等 C++ 补齐——即当前 import 失败点)
8. checkpoint 分片 planner 并行写路径(torch 默认);DCP 现为合并式单文件布局
9. OffloadWrapper(需 save_on_cpu)、launch --module/--no_python
10. `_tools/`(mem_tracker/sac_estimator 等 5854 行,多为分析工具,低优先级)、
    flight_recorder、_symmetric_memory(native)、_state_dict_utils、_serialization、
    remote_device、_token_switch、benchmarks

验证:`test/test_distributed.py` NCCL 子进程脚本覆盖 base 集合通信、对象集合通信、
rank 换算、all_to_all 族、batch p2p、DDP(初始同步/分桶梯度=解析值/comm hook/fp16 hook/
find_unused 非对称使用/state_dict 前缀/no_sync);单 rank 任意 CUDA 机可跑,双 rank 按 GPU 数门控。


自研 compiler 栈(`tp.compile`)、backward 算子直接暴露(conv2d_grad_input 等)、`_serialization_torch` 与 torch checkpoint 互操作、vision/audio/hub 扩展、`bcomplex32` dtype、DepthwiseConv2d。

## 建议补齐优先级

0. **fft/special 模块对齐**(2026-08-22 已完成 Python 侧):
   - `tensorplay/fft.py`:从 4 个 1D 原语(fft/ifft/rfft/irfft,native)组合出 torch.fft 全部
     22 个公开名——2D/nD 族(fftn/ifftn/rfftn/irfftn 及 2D 变体,DFT 可分性 + 各 norm 模式
     跨维乘性,故每维传同一 norm 即得全局缩放)、Hermitian 族(hfft/ihfft/hfft2/ihfft2/
     hfftn/ihfftn,按 ATen 约定先取共轭;conj 经 `view_as_real→翻转虚部→view_as_complex`
     组合,因 native conj 内核未接线)、辅助族 fftfreq/rfftfreq/fftshift/ifftshift(narrow+cat
     实现)。`s`/`dim` 默认值语义对齐 torch(None → 全部维度或最后 len(s) 维)。
   - `tensorplay/special.py`(新):60 个公开名全对齐。原生直通:erf/erfc/erfinv/exp2/expm1/
     log1p/lgamma/digamma/i0/sinc/logit/logsumexp(yaml 已声明);组合实现:
     expit=sigmoid、ndtr=½·erfc(-x/√2)、ndtri=-√2·erfinv(2x-1)、log_ndtr(左尾渐近展开)、
     erfcx、entr、xlogy/xlog1py、multigammaln;**其余 28 个(Bessel/Airy 全族、正交多项式、
     incomplete gamma、zeta、polygamma、i1/i0e/i1e 等)已于 2026-08-22 补齐原生内核**——
     数学体从 `aten/src/ATen/native/Math.h` 逐行照抄为 `p10/include/SpecialMath.h`
     (g++ -fsyntax-only 零错误),CPU/CUDA 包装在
     `p10/src/backend/{cpu/SpecialKernels.cpp,cuda/SpecialKernels.cu}`(float_math_kernel/cuda
     房式模式),CMakeLists 已挂入,yaml 已加 31 条 dispatch 条目。注意:这些点级 op 暂无
     derivatives.yaml 条目 → 前向可用、不走生成 autograd(torch 对多数特殊函数同样仅前向)。
   - `tensorplay/__init__.py`:`special` 加入 lazy_modules。运行时验证同前:待重建。

1. `F.linear`(2026-08-21 已完成:`tensorplay/nn/functional.py` 逐分支对齐 `aten/src/ATen/native/Linear.cpp`,含 addmm 融合、`_flatten_nd_linear`、`TORCH_LINEAR_FLATTEN_3D`、sparse weight 分支;`Tensor.contiguous()` 经 native_functions.yaml 补齐)
2. RNN/LSTM(2026-08-21 已完成:`tensorplay/nn/modules/rnn.py` 移植 `aten/src/ATen/native/RNN.cpp` 的 CPU cell/层堆叠机制与 `torch/nn/modules/rnn.py` 模块 API(RNN/LSTM/GRU/RNNBase + 三 Cell,含 bidirectional/batch_first/dropout/proj_size/PackedSequence);配套新增 `tensorplay/nn/utils/rnn.py`(PackedSequence/pack_padded_sequence 等)与缺失算子 `scatter_.src/scatter_.value/scatter_add_`(CPU+CUDA)、`narrow`、顶层 `as_tensor`)。Transformer/MultiheadAttention(2026-08-22 已落地 `modules/transformer.py`、`modules/multihead_attention.py`,静态验证导出齐全;运行时验证因 `_C.so` 过期暂缓)
3. loss 模块族(2026-08-22 静态核验:除 CTCLoss/MultiLabelMarginLoss 两个模块包装外全部导出;F 层组合实现齐)
4. index/scatter 方法族
5. padding/Upsample/Fold(见下节:内核已存在,卡在 yaml 接线)
6. linalg 域(大部分 yaml 已声明,见下节 B/C 类核对)
7. **cuda 模块对齐**(2026-08-21 已立项,见 git history)

## 8. 底层(native)接线审计(2026-08-22,纯静态,未构建)

方法:三层漏斗盘点 —— p10 dispatcher 实际注册(`m.impl`,642 个)→ `config/native_functions.yaml`
声明(424 条 / 328 个基础名)→ Python 可见面(`_C/__init__.pyi` 275 函数 + 184 方法;
注意 pyi 只覆盖 yaml+模板,手写 `m.def` 以 src/bindings 为准)。对照本机 torch 2.13
aten 2124 个 schema。**结论:内核不缺,缺接线。**

### A. 已注册进 dispatcher 但未暴露给 Python:254 个

内核(CPU+CUDA)与 `m.impl` 注册均在,但 yaml 无条目 → 无 codegen 绑定 → Python 不可达。
按族(nn 相关度排序):

| 族 | 未接线算子 | 影响 |
|---|---|---|
| upsample | nearest1d/2d/3d、linear1d、bilinear2d、bicubic2d、trilinear3d 全家 fwd+bwd(28) | **F.interpolate 直接引用 `tensorplay.upsample_*`,运行时 AttributeError**;Upsample 模块同 |
| RNN fused | lstm、gru、rnn_tanh、rnn_relu(cpu/cuda 内核齐) | rnn.py 现走纯 Python cell;接线即对齐 torch 融合路径并大幅提速 |
| loss 原生版 | binary_cross_entropy(_with_logits)、kl_div、l1_loss、huber_loss、smooth_l1_loss、cosine_embedding/margin_ranking/hinge_embedding/multi_margin/multilabel_margin/multilabel_soft_margin/poisson_nll/soft_margin/triplet_margin_loss | F 层现用 Python 组合(慢、backward 图长);原生版含 backward 内核 |
| 激活原生版 | elu(+bwd)、celu、glu(+bwd)、hardshrink/hardsigmoid/hardswish/hardtanh/leaky_relu/mish/relu6/selu/softplus/softshrink/threshold(多数带 bwd)、prelu | 同上 |
| index/scatter/reduce | gather、sort、argsort、index_copy、index_fill(.Scalar/.Tensor)、index_put(_)、masked_fill(_)(.Tensor 变体)、masked_scatter、scatter.src/.value、searchsorted.Tensor、bucketize.Tensor、take、bincount、cummax、cummin、cumprod、logcumsumexp、nansum、kthvalue、mode、aminmax/amax/amin、nanmedian | gap 第 4 优先级的底层一半已备好 |
| 结构 | im2col/col2im(+bwd)、pixel_shuffle/unshuffle、channel_shuffle、reflection/replication/circular_pad_nd(+bwd) | Fold/Unfold、PixelShuffle、ChannelShuffle、三类 Pad 可切原生 |
| conv | conv_transpose1d 全家(fwd+grad×3) | ConvTranspose1d 现经 F 组合绕行 |
| 判断/复数 | isinf/isnan/isfinite/isneginf/isposinf、logical_not/or/xor、complex/imag/conj | bitwise 家族内核也在(见 D) |
| 距离/其他 | dist、pdist、pairwise_distance、one_hot、meshgrid、trace/tril/triu/diag/diag_embed、rot90、repeat_interleave.self_int、tensor_split.sections、split_with_sizes、std_mean、var_mean、copysign/heaviside/logaddexp(2)/hypot/gcd/lcm/nextafter/nan_to_num/deg2rad/rad2deg/signbit/sinc/sgn/fix/logit/erfinv/xlogy.Tensor、divide/multiply/subtract/rsub/fmod/remainder/true_divide 的 Scalar/Tensor 分解式、clip、clamp_max/min.Scalar/.Tensor、greater/less(_equal)/not_equal、positive/negative/named 变体 | 多为标量/张量重载细分,接 yaml 时按 overload 收敛 |

linalg 大部(svd、cholesky 家族、triangular_solve、linalg_*_kernel_cuda 系列)同样只差 yaml 条目。

### B. Python 层已引用、但底层无任何绑定路径(重建后仍会崩)

| 引用处 | 引用名 | 底层状态 |
|---|---|---|
| `nn/functional.py::unfold/fold` | `_C.im2col` / col2im | 内核+m.impl 在,**yaml 与手写绑定均无** → rebuild 后 F.unfold/F.fold 仍 AttributeError |
| `nn/functional.py::interpolate` | `tensorplay.upsample_*` | 见 A,需 yaml 接线 |

**2026-08-22 09:05 更新**:yaml 接线已由并行工作流启动(im2col/col2im/upsample 全家/
conv_transpose1d/elu/glu/pixel_shuffle/one_hot/reflection·replication_pad_nd 等条目已进
yaml,424→479 条);derivatives.yaml 相应导数早已备好。剩余未接线:gather/sort/argsort、
index_put/index_fill/index_copy、masked_scatter/searchsorted/bucketize/take/bincount、
cummax/cummin/cumprod/logsumexp/logcumsumexp/nansum/kthvalue/mode/nanmedian/aminmax、
channel_shuffle、lstm/gru/rnn_*、loss 原生版与 isinf/isnan/logical_*/complex 族。

### C. yaml 有声明但无对应 m.impl(39,疑为命名错位)

`_foreach_div/mul/sub/neg/sqrt/rsqrt/reciprocal/sign/abs_...` 共 37 条:impl 侧实际注册的是
带 `.List_out/.Scalar_out` 后缀的 out 变体与一个 `_foreach_` 通配名,需人工核对 overload
拼写是否与 codegen 期望一致。另 `einsum`(p10/src/Einsum.cpp 直连)、`unfold`(yaml 无
overload 后缀,impl 为 `unfold.Tensor`)、`view`(Tensor.cpp 直连)。

### D. 两层都真缺(nn 对齐阻塞项)

(经 overload 归一化复核;`where/maximum/minimum/logical_and/masked_select/unfold(Tensor 法)/SDPA`
等此前误报缺失,实际 yaml 已有。)

- bitwise 全族(bitwise_and/or/xor/not/left_shift/right_shift):内核、yaml、Python 三层皆无
- `embedding_renorm_`:F.embedding 的 max_norm 分支直接 NotImplementedError(torch 用它做 renorm)
- `grid_sampler`(F.grid_sample 为 Python 组合占位,affine_grid 同);vision/空间变换关键
- index/scatter 补充:`index_reduce`、in-place `index_copy_/index_fill_`(out-of-place 内核已在 A 类)、
  `select_scatter/slice_scatter/diagonal_scatter/take_along_dim`、`msort`、`nanmean`、`quantile`
- 判断:`isclose`、`isreal`
- linalg 补充:linalg_vector_norm/matrix_norm/cross/vecdot/pinv/matrix_power/matrix_exp/matrix_rank(norm/linalg_norm 有)
- fft 高维与辅助:fft_fftn/rfftn/ifftn 族、fftshift/fftfreq(stft/istft/1D fft_*2 尚缺 2D/nD)
- histogram/histogramdd、`cdist`、weight_norm(renorm 内核已在 A 类)
- RNN cell 级 `lstm_cell/gru_cell/_pack_padded_sequence/_pad_packed_sequence`:rnn.py/utils 以
  Python 组合实现,属有意收窄(融合内核接线后可切)
- RNG:`get_rng_state/set_rng_state` 手写绑定已在 Generator.cpp,**`.so` 落后于源码**(18:52 vs 提交),当前 `import tensorplay` 因此失败——重建即可解

### E. 对剩余 nn 缺口的阻塞判定与补齐(2026-08-22 已完成 Python 侧)

| 待补模块 | 阻塞? | 状态 |
|---|---|---|
| CTCLoss / MultiLabelMarginLoss 模块包装 | 否 | **已补**(modules/loss.py,包装 F.ctc_loss / F.multilabel_margin_loss) |
| EmbeddingBag 模块 | 否 | **已补**(modules/sparse.py,含 from_pretrained;F.embedding_bag 组合实现驱动) |
| AdaptiveLogSoftmaxWithLoss | 否 | **已补**(modules/adaptive.py;以 tp.where+embedding-on-flattened 替代 torch 的 index_copy_/index_fill_/gather) |
| SyncBatchNorm | 否 | **已补**(modules/batchnorm.py;纯可微原语组合 + distributed.all_gather 统计同步,无需自定义 autograd Function) |
| DataParallel | 部分 | **已补**(nn/parallel/data_parallel.py:scatter/replicate(deepcopy)/parallel_apply(线程+stream ctx)/gather;单设备或 CPU 自动回退) |

导出已全部接入 `nn/modules/__init__.py`(+__all__ 保持 sorted)与 `tensorplay.nn.DataParallel`。
运行时冒烟验证因 `_C.so` 过期(`get_rng_state` ImportError)暂缓,重建后应随 B 类接线一并点亮。

### 建议

1. **一次性 yaml 接线**:把 A 表 nn 关键族(upsample、RNN fused、loss 原生版、激活 bwd、
   index/scatter、im2col/col2im、pad_nd 三类、conv_transpose1d)+ B 行 im2col/col2im
   写入 native_functions.yaml(含 dispatch 标签),一次 rebuild 全部点亮。
2. C 类 overload 名核对可与 1 同批处理。
3. D 类按需立项;embedding_renorm_/nll_loss2d 建议随 nn 模块补齐顺带解决。

## 9. dist.checkpoint 对齐(2026-08-22 完成)

`tensorplay/distributed/checkpoint/` 对照 `torch.distributed.checkpoint`(torch 2.13 运行时
26 个公开名)落地,设计采用**合并式单文件布局**(协调 rank 经对象集合通信汇聚各 rank 状态,
写 `__0_0.distcp` + `.metadata`;torch 的分片 planner 写入路径列为后续项):

| 文件 | 内容 |
|---|---|
| `state_dict_saver.py` / `state_dict_loader.py` | save / async_save(当前同步降级并告警)/ load;副本键去重、张量原位 copy_ 回填、跨 rank 广播 |
| `filesystem.py` | FileSystemWriter/Reader + StorageWriter/Reader ABC(修复了初版 `_pickle` NameError 与 read_data 未回填问题) |
| `mega_storage.py` | **MEGA 存储后端,对应 torch 的 HuggingFace 后端**(同为 Xet 底座):`.mega` 分片(tp 原生序列化)+ `model.mega.index.json` weight_map;支持 `mega://<repo>[@rev]/path`、`mega://buckets/<id>/path`,经 megatensors `MegaFileSystem`(fsspec)读写 |
| `state_dict.py` | torch.distributed.checkpoint.state_dict 子集:get/set_state_dict、get/set_model_state_dict、get/set_optimizer_state_dict(普通模块与 DDP 路径) |

对齐面:`save/async_save/load/FileSystem*/Storage*/MegaStorage*` 共 11 个公开名;
未移植(依赖 tp 不存在的 DTensor/ShardedTensor):分片 resaving、load_sharded_optimizer_state_dict、
HF/量化后端(由 MEGA 后端替代)。验证:py_compile 全过、包内交叉引用 AST 核对通过;
多 rank 数值冒烟待 `_C.so` 重建后进行。

## 10. C++ 核心(p10)/ autograd(tpx)/ 绑定层全栈差距分析(2026-08-23)

方法:三路并行源码走读(C++ 核心 / Python 门面 / autograd+绑定),对照
`third_party/pytorch`。Python 门面部分与本文件 §1–§8 有重叠,此处只记录增量结论。

### 总量对比

| 维度 | TensorPlay | PyTorch |
|---|---|---|
| C++ 核心规模 | p10+tpx+stax+bindings ≈ 7.5 万行 | ATen/c10 数十万行级 |
| 算子 | impl 去重 673 名(CPU 671/CUDA 587),yaml 618 schema | ~2000+ op |
| DispatchKey | 6 个 | 20+(两轴 BackendComponent) |
| 反向节点 | 148 个生成节点 / derivatives.yaml 187 条 | 数千 + SavedVariable 体系 |
| 顶层导出符号 | ~644 | 约 1.5–3 倍 |
| Python 层 | 354 文件 ≈ 13.3 万行 | torch/ ≈ 20 万行级 |

一句话:**骨架高度仿真、血肉按需裁剪**——对象模型/dispatcher 分层/TensorIterator/
autograd 引擎逐文件对照移植,数量级差距在 op 覆盖、分发深度与 dtype×op 组合完整性。

### 10.1 p10 vs ATen/c10

对象模型(`include/{Tensor,TensorImpl,SizesAndStrides,StorageImpl,DataPtr,VariableVersion}.h`):
- 已复刻:SizesAndStrides(5 维内联)、DataPtr 三件套、StorageImpl 经 shared_ptr<SharedState>
  共享、VariableVersion(原子计数 + view 经 share_version_counter 别名)、AutogradMetaBase
  虚接口解耦 tpx、SparseState(COO 最小集)。
- 差距:单一 Tensor 类持 `shared_ptr<TensorImpl>`,无 TensorBase/Tensor 分离,全库无
  intrusive_ptr;key_set 非存储字段而是按 device 现算;memory_format 仅 `is_channels_last_`
  bool(无 MemoryFormat 枚举与传播);无 conjugate/neg 位(conj 是物化拷贝);无 named tensor。

Dispatcher(`include/{Dispatcher,DispatchKey,DispatchStub}.h/.cpp`):
- DispatchKey 仅 CPU/CUDA/AutogradCPU/AutogradCUDA/AutocastCPU/AutocastCUDA 六个,
  EndOfKeys 编译期定死,无第三方注册机制、无 Meta 后端(即无 shape-only 推理)。
- yaml 驱动 codegen + 运行时 `unordered_map<string, DispatchTable>` 按 (op,key) 单函数槽;
  调用侧生成代码直接查表,类型 `void*` 强转,**无 boxing/kernel 栈/fallthrough**,非 key_set
  驱动。autograd 接线靠生成代码里 `GradMode::is_enabled()` 分支。
- DispatchStub 是 ATen CPU capability stub 的独立移植(DEFAULT/AVX2/AVX512),仅 reduction
  与 StaxPointwise 在用。autocast 真实现(76 op 路由)。

算子覆盖(673 vs ~2000+):
- 齐:pointwise/比较/reduction/matmul-BLAS/conv+pool(24 conv op)/linalg 26 个 linalg_* /
  index-scatter 主干/norm 族/loss(tp_*)/embedding/SDPA(含 bwd)/RNN fused/upsample 全套
  (16)/foreach≈100 + fused optimizer/FFT(pocketfft)/pad 系列/window/特殊数学。
- 缺:**dropout 全系(0 处)**、**resize_/resize_as_(0 处)**、cross_entropy/ctc_loss、
  max_pool1d/3d 与 max_unpool、rms_norm、scatter_reduce/index_reduce/take_along_dim/
  segment_reduce、量化全部、稀疏高级算子、nestedtensor 及长尾。

TensorIterator(`src/TensorIterator.cpp` 1069 行,自述 port of ATen):
- 有:broadcast infer_size、维度重排、common dtype(compute_types/promoteTypes)、
  fast setup、内存重叠检查、is_cpu_scalar、reduction 双 pass + parallel_reduce;
  自研线程池(GRAIN_SIZE=32768,非 OpenMP)。
- 缺:MemoryFormat/channels-last 输出布局、meta function、can_cast 被简化成"非 Undefined
  即可转"(丢安全规则)、CUDA 无对应 iterator(各 .cu 手写索引)、vectorized loop 未与 TI 集成。

view/inplace:view 语义正确(as_strided/select/slice/expand/view 共享 storage + 版本号别名,
ViewKernels 22 个 view/shape op);inplace bump_version 由生成层统一做(TensorGenerated.cpp
123 处),p10 库内只有 sparse/optimizer 内核显式 bump。`view()` 只接受 contiguous 输入。

CUDA/oneDNN:32 个 .cu 与 CPU 大体对称;SDPA 有 flash-v1 风格 online softmax 简化版;
CUDAAllocator 是真 caching allocator(按流分块/合并/recordStream);cuBLASLt bias epilogue +
strided batched GEMM;ForeachMultiTensor.cuh 复刻 MTA。oneDNN 仅 engine/stream 单例 +
CPU 四个内核的选择性加速,非成体系 mkldnn 后端;cuDNN 集成浅(USE_CUDNN 开关 + 67 行 CHECK 封装)。
complex dtype 定义齐全(含 ComplexHalf/BComplex32),但 vec256 无 complex 向量核。

### 10.2 tpx autograd vs torch autograd

已有且质量尚可:每设备 ReadyQueue(max-heap by sequence_nr)+ 惰性常驻 worker、依赖计数
+ InputBuffer 合并(GraphTask::mutex_)、grad() 的 init_to_execute capture、reentrant 嵌套
本地队列、anomaly mode(NaN probe + AnomalyMetadata 栈回溯 + Python 调用点捕获,完成度高)、
create_graph 高阶导、Python 侧 vjp/jvp/jacobian/hessian/vhp/hvp 全套(functional.py 863 行)、
gradcheck 968 行、Function 双风格(setup_context/legacy)带 save_for_backward 版本检查。

结构性缺口(按危害排序):
1. **无 SavedVariable**:148 个生成反向节点(AutogradNodesGenerated.h)直接按值持有前向
   `Tensor`,unpack 零校验 → 前向后原地改输入会**静默算错梯度**(torch 抛 RuntimeError)。
   saved_tensors_hooks、`_saved_*` 属性同样缺失。版本检查仅覆盖 Python Function 路径。
2. **`_InferenceMode` 未绑定**:grad_mode.py:284 引用 `_autograd._InferenceMode`,但
   bindings Autograd.cpp 未注册 → `inference_mode()` 必然 AttributeError;底层亦无
   InferenceMode TLS/禁记录语义。
3. 无 forward-mode AD(jvp 为 double-backward trick;Function.jvp/vmap 明确 raise);
   worker 无 DeviceGuard/stream 处理,混合设备路由不精确;无 validate_outputs 元数据校验;
   `retain_graph=False` 仅清 next_edges 不释放节点持有的张量副本,显存回收弱于 torch;
   无单节点 fast path、优雅关停、compiled-autograd 钩子;c10d 分布式 autograd 无关。

### 10.3 绑定层(src/bindings/python)vs THPVariable

- 双层结构:pybind11 手写 412 `.def` + codegen METH_FASTCALL 快路径(324 模块函数 +
  226 张量方法,gen_python_c.py 从 618 schema 生成,setattr 覆盖同名 pybind 绑定),
  合计约 550 个 C 入口;`_C/__init__.pyi` 573 def。backward/grad 执行期 gil_scoped_release。
- 互操作:numpy 零拷贝入向快路径、DLPack 双向齐全(capsule destructor/内存池)、
  pickle(getstate/setstate + 显式 `__reduce__`,CPU-only,支持 SharedMemory 通道)、
  dynamic_attr 支持 weakref;异常映射 IndexError/ValueError/TypeError/NotImplementedError/
  RuntimeError 完整,可选暴露 C++ 栈。
- 缺:`__torch_function__`/`__torch_dispatch__` 协议、Tensor 子类/_make_subclass、
  python dispatcher key、`__deepcopy__`(靠 __reduce__ 兜底)、`__cuda_array_interface__`、
  out-of-band pickle(protocol 5)、完整索引方言(python_variable_indexing.cpp 对应物)。
  `tensorplay.Tensor = _C.TensorBase` 直别名,monkey-patch 是唯一扩展途径(_tensor.py 补了
  flatten/unflatten/unfold/t/register_hook 等 14 个)。缺 `__floordiv__/__mod__`、位运算 dunder。

### 10.4 stax/compiler 定性

静态 IR + 微型 pass 框架 + 解释执行器:FusionPass 只识别 mul→add 一种 pattern + pointwise
表达式程序(CPU 向量化);Graph::execute 顺序解释,有存储生命周期回收与 channels_last
布局变换。tensorplay/compiler/(3263 行)是 FX 式 Tracer + DCE/ConstFold/ShapeProp +
AOT 式 joint-graph 切分 + codecache/cudagraphs。**概念演示层**:无 codegen 到循环/CUDA、
无 buffer 规划/自动调优,非 Inductor。

### 10.5 高价值补齐项(2026-08-23 处理)

1. ~~`_InferenceMode` 绑定缺失(inference_mode 必崩)~~ **已修复**:
   - 新增 `p10/include/InferenceMode.h` + `src/InferenceMode.cpp`(thread_local TLS +
     RAII guard,镜像 GradMode 模式),tpx/Autograd.h 再导出;
   - 绑定层注册 `_InferenceMode`(init 存 prev、`__enter__/__exit__` 恢复,支持嵌套)
     与 `_autograd.is_inference_mode_enabled`;
   - codegen 两处接线:gen_tpx.py 的梯度记录门改为
     `GradMode::is_enabled() && !InferenceMode::is_enabled() && ...`(164 处),
     in-place 版本号自增加 inference 保护(TPXOpsGenerated.cpp 35 处 +
     TensorGenerated.cpp 41 处,含 gen_api.py 的 skip_implementation 路径)。
   - 已知收窄:inference tensor 无独立版本计数器/禁止后续参与 autograd 的完整语义
     未做(需 TensorImpl 位标志),当前覆盖"不记录图 + 不 bump 版本"。
2. ~~生成反向节点无版本校验(SavedVariable 缺失)~~ **已修复**:
   - 新增 `tpx/include/SavedVariable.h` + `src/SavedVariable.cpp`:save 记录版本,
     unpack 时版本不符抛 RuntimeError(消息对齐 torch "modified by an inplace
     operation: [saved version: X; current version: Y]");
   - gen_autograd.py:Tensor 型成员改存 SavedVariable,apply() 顶部生成
     `{m}_.unpack()` 局部变量(公式引用 `{m}_sv`),节点新增 release_variables()
     override 释放保存的张量;Node::release_variables 改虚函数;
   - ManualNodes.h 手写节点(SDPA/Mean/Cat/Stack)同步迁移;
   - Python Function 路径原本就有版本检查,不变。
3. resize_/dropout 缺失 **已部分落地**:
   - `resize_`:yaml 条目(method 变体,CPU/CUDA)+ TensorImpl::set_sizes_contiguous
     + 双后端内核(storage 原地扩容保数据、缩容只改逻辑形状、非 resizable storage 抛错);
   - dropout 走 torch 同构架构:新增融合 `native_dropout(Tensor,float)->(Tensor,Tensor)`
     (CPU 标量循环 / CUDA philox grid-stride 单内核同时产 output+bool mask),
     derivatives.yaml 公式 `grad * mask / (1 - p)`(首个 tuple 返回 op);
     F.dropout 训练路径改调 `_C.native_dropout`,p==1 返回全零,inplace 暂走组合回退;
   - memory_format **已补**(2026-08-23 深夜):新增 `p10/include/MemoryFormat.h`
     (Contiguous/Preserve/ChannelsLast/ChannelsLast3d 枚举 + NHWC/NDHWC stride
     数学,对齐 c10);TensorImpl 的 `is_channels_last_` bool 升级为
     `memory_format_` 字段(原 bool 无任何读取方,零风险迁移);
     Tensor 新增 `memory_format()/is_contiguous(mf)/is_channels_last{,_2d,_3d}/
     contiguous(mf)`(材质化走 fresh-storage + strided copy_ + 格式标签);
     绑定层暴露上述方法(接受 int 枚举);Python 顶层导出
     `MemoryFormat/contiguous_format/preserve_format/channels_last/channels_last_3d`。
     收窄:工厂族(empty_like/clone)的 memory_format 参数、TI 的 channels-last
     输出布局传播、conv/oneDNN 的 NHWC 加速路径未动——枚举与语义地基已就位。

验证状态(2026-08-23 深夜):`ninja -C build _C` 全量通过(p10/tpx/stax/_C,
产物 mtime 新于全部改动源);冒烟测试全绿——inference_mode(记录抑制/嵌套恢复/
装饰器/查询 API)、版本校验(原地修改已保存张量抛 "[saved version: X;
current version: Y]"、未修改路径梯度数值正确、create_graph 双反向可用)、
resize_(扩容保前缀/缩容仅逻辑/版本号自增)、dropout(梯度比例=1/(1-p)、
被丢弃位梯度为零、p==0 恒等、p==1 全零)、memory_format(NHWC/NDHWC stride
数学、值保持往返、同 storage no-op、channels-last 张量参与 autograd)。
注:晚间 PointwiseKernels.cpp/VecUnary.h 的并行 AVX2 WIP 已由其作者完成,
与本节改动共同编译通过。

## CUDA 算子缺口盘点与补齐(2026-08-24)

盘点方法:提取 `m.impl` 注册名,CPU 全集 698 个 vs CUDA 全集 614 个,差集
65 条。注意两点坑:两侧均有宏拼接注册(`"_foreach_" #NAME ...`),文本提取
会互相假匹配,需按宏展开逐个核对;`to/arange` 类工厂 op 在 yaml dispatch 表
只列 CPU,但真注册以 backend 文件的 TENSORPLAY_LIBRARY_IMPL 为准。

本轮已补(CUDA 注册或新内核):
- 视图族:chunk / split / split.sizes / unbind(纯视图,CUDA ViewKernels.cu);
- 工厂:arange / arange.end / linspace / logspace(FactoryKernels.cu,
  steps==1 返回 start,对齐 torch RangeFactories.cpp);
- RNG 家族(RandomKernels.cu,philox 基建复用):randint(.like)/ randperm
  (fp32 keys + argsort)/ random_ / geometric_ / log_normal_ / cauchy_ /
  bernoulli_ / poisson;并修复 philox_cuda_state 调用点的 .first/.second
  残留(结构体重构后未跟进,首编译必炸雷);
- 归一化:normalization group_norm(+backward)/ instance_norm(+backward)
  (NormalizationKernels.cu,复用 LN block-reduce;instance 训练态 =
  G=C 组归一 + running stats 更新内核,eval 态委托 batch_norm);
- 其它:angle(Pointwise)、median(ReductionKernels,sort 取 (n-1)/2)、
  constant_pad_nd(+backward)(PadKernels,复合实现双端复用)。

CPU rnn_impl 重写(Tier5OpsKernels.cpp):
- fp64 标量循环 → 输入 dtype 计算 + 单 GEMM 预算全序列输入门 + 张量算子门数学;
- **修正 GRU 公式 bug**:旧 CPU 实现把 `h' = (1-z)*n + z*h` 坍缩成纯 n
  (注释自我否定处),与 CUDA fused cell 及 torch 不符;
- 验证:test/test_rnn_numerics.py 对照本机 torch 2.13,48/48 通过
  (lstm/gru/rnn_tanh × 双向 × batch_first × 层数 × fp32/fp64);
- 坑:narrow 在 CPU 是拷贝语义非视图,不能作写入目标(输出曾全零),
  写入一律走 slice/select 视图 + copy_,方向输出用 cat 拼。

CUDA 侧验证:test/test_rnn_cuda.py(cpu↔cuda 一致性 + nn.LSTM GPU 训练冒烟),
待远端(CUDA 12.8,Tesla P4)构建完成后执行。

仍缺(低优先/记录在案):
- foreach `_out` 变体约 50 条(optim python 路径未用,暂缓);
- unique(sort+分段可做)、linalg_ldl_* 系列(torch 本身也仅 CPU,不补);
- RNN 反向原生公式(方案 a):fused cell 的 fwd/bwd 内核已在 RNNKernels.cu,
  待暴露为 dispatch op + derivatives.yaml 接线;当前训练走 nn/modules/rnn.py
  可微组合路径(chunk/split/unbind 补齐后 GPU 可训)。

## 全栈接线审计 + ABI/autograd 阻塞修复 + D 类补齐(2026-08-24 深夜)

方法:以 `m.impl` 注册名 ∩ yaml 条目 ∩ Python 可见面三层漏斗复核(§8 的续),
本轮重点是把"内核在、yaml 缺"和"三层皆缺"两类真正打通,并顺手修掉三处
**运行时必崩/静默算错**的深层阻塞。

### A. 阻塞级修复(先于一切新功能)

1. **工厂/标量算子全线崩溃(ABI 错位)**:`t.add(1)`、`zeros/ones/empty/
   rand/randn/randint/randperm/eye/arange/linspace/logspace/full` 全部
   RuntimeError 或读垃圾值。根因:schema 用 `ScalarType?`/`Device?`
   (`std::optional<Device>` 24 字节走栈传参),而注册的 CPU/CUDA 内核收裸
   `DType/Device`(寄存器)→ DispatchStub 按 schema ABI 调用,内核读到
   垃圾 pin_memory。修复:FactoryKernels{.cpp,.cu} 增加 stub-ABI 适配层
   (resolve dtype/device 后转发原内核),softmax/log_softmax(CPU)反向错位
   同理改为裸 DType。全量 arity/optional 双向审计脚本化核对,CUDA 0 差异。
2. **fastcall 层缺参段错误**:必填参数缺失时 `slots[i]==nullptr` 直接进
   unpacker → SIGSEGV(且非 invalid_argument,重载回退失效)。gen_python_c
   对必填参数生成显式守卫(抛 invalid_argument → TypeError/回退)。
3. **scatter 族索引解码错误**(此前 Python 不可达故未暴露):按 index 自身
   形状解码 outer/dim/inner,源值沿 self_inner 广播填充(ATen
   ScatterGatherKernel 语义),src 补 dtype 转换;`shape().begin()` 迭代
   Size 代理的 UB 一并替换为 static_cast。CPU+CUDA 四条路径同修。
4. **nansum(dim=[]) 不归约**:返回输入副本而非全局和(ReduceOps 语义),
   nanmean 组合依赖之。CPU/CUDA 修正为空 dim = 全维归约。
5. **sum(dim)/mean() 反向缺失**:`sum.dim_IntList` 无 derivatives 条目、
   `mean` 被 EXTERNAL_NODES 排除但其手工节点从未接线。新增
   `_sum_dim_backward` op(CPU/CUDA)+ derivatives 条目;mean 移入
   MANUAL_DERIVATIVES 复用既有 MeanBackward 节点。至此任意 dim 组合的
   sum/mean 训练路径可用。
6. **整型 dunder 提升错误**:`__add__/__sub__/__mul__/__radd__/__rmul__`
   把 python int 预转 double → int64 张量被提升成 float32(torch 保持
   整型)。增加 int64 重载置于 double 之前(Tensor.cpp)。

### B. yaml 接线(内核已在,Python 不可达 → 打通)

index_fill.Scalar/.Tensor(+in-place 双变体,原位内核为本轮新增)、
searchsorted.Tensor、bucketize.Tensor、scatter.src/.value/_.src/_.value、
masked_fill.Tensor/.Tensor_;cumsum/cumprod/nansum/mode/kthvalue/nanmedian/
index_select 补 `method` 变体(torch 有而 tp 缺的方法面)。

### C. 新增算子(D 类"三层皆缺"第一批)

| 算子 | 实现 | 备注 |
|---|---|---|
| select_scatter / slice_scatter / diagonal_scatter | 复合(clone+视图+copy_)双端 | ATen TensorShape.cpp 语义 |
| take_along_dim | 广播展开 + gather 复用,dim=None 走 flatten | TensorAdvancedIndexing.cpp |
| msort | sort(dim=0).values 复合 | |
| nanmean | nansum/valid-count 复合,全 NaN 行→NaN,int 输入提升 fp32 | ReduceOps.cpp |
| isclose / isreal | fp64 点评内核(CPU 循环/CUDA grid-stride),inf/equal_nan 规则对齐 ATen | |
| bitwise_not/and/or/xor(.Tensor/.Scalar)/left·right_shift(.Tensor/.Tensor_Scalar) | 整型+bool 点评内核双端,shift 取模位宽经无符号域 | BinaryOps/BinaryBitwiseOpsKernel |

导数:select/slice/diagonal_scatter(torch 有公式)暂未接 derivatives.yaml,
列为下批;其余 torch 本就无可微导数。

### D. 其它修复

- cosine_embedding_loss `-1` 分支 clamp 下界写成 margin(应为 0);
  hinge_embedding_loss 对齐 ATen 双路相加形式(t∉{±1} 时 = x +
  clamp(margin-x,0)),反向同步;
- multi_margin_loss/multilabel_margin_loss 组合实现依赖的 embedding/
  index_select 方法面补齐;multilabel 的 first--1 掩码与 gid 索引形状
  重写(旧实现对批量输入必然广播失败);
- permute/expand 方法接受 *args 变体(_tensor.py 归一化,torch 兼容);
- 并行工作流遗留:Tier5OpsKernels rnn_impl 重写中 `cat` 未限定导致编译
  失败,最小修复为 Tensor::cat(AGENTS.md 最小修复原则)。

### 验证

- 新增面:27 项冒烟(标量运算/工厂/全部新算子数值/autograd 完整性)全绿;
- 回归:9 个测试文件 A/B 对照 HEAD——基线 63 failed,本批后 62 failed
  (净修复 pixel_shuffle roundtrip 顺序依赖 + margin family 三例),零新增;
- 产物新鲜度:libp10.so/_C.so mtime 晚于全部改动源(ninja -C build p10 _C)。

### 下批建议(按价值排序)

1. scatter_reduce/index_reduce 内核(embedding renorm 与 NLL2d 的前置);
2. select/slice/diagonal_scatter 的 derivatives.yaml 接线(线性,公式照抄
   torch,引用的 select/slice/diagonal 均已可用);
3. quantile/nanquantile、histogram(dd)、cdist、weight_norm;
4. grid_sampler_2d/3d(+backward)(vision 关键,F.grid_sample 解除占位);
5. `_foreach_*` 的 `.out` 死注册清理或补 schema(C 类收尾)。

## 远端 CUDA 构建 + 验证结果(2026-08-24 凌晨,deepln Tesla P4/CUDA12.8)

环境:远端构建机见 .remote_build.md(勿提交)。要点:nvcc 不在 PATH 需
export;/usr/local/cuda 为 12.8(非 AGENTS.md 所述 13);cuDNN frontend 头在
/tmp/cudnn-frontend-1.15.0/include(CMake 需 -DCUDNN_FRONTEND_INCLUDE_DIR);
**必须 -DCMAKE_CUDA_ARCHITECTURES=61**(默认 sm_52 无 double atomicAdd,
EmbeddingKernels 直接编译失败);python 依赖 pybind11/pyyaml/typing_extensions;
third_party/v3.4.1.zip 缺失时可用已解压目录现场打包;长任务用 setsid 脱离
ssh 会话(nohup 会被容器回收)。

修复的移植期问题:
- CUDAGraph.cpp 实现在 cuda:: 下而头文件声明在 cuda::graph:: 下(_C 加载
  undefined symbol);实现整体移入 namespace graph;
- RandomKernels philox_args.first/.second 结构体残留;
- NormalizationKernels kLNThreads/ln_block_reduce2 未加 layer_norm:: 限定;
- randperm 的 static_cast<vector>{n} 非法语法;
- 激活走 cuDNN 在 P4+cuDNN9 全形态 CUDNN_STATUS_EXECUTION_FAILED_CUDART,
  sigmoid/tanh/relu 改原生逐元素内核(torch 同款取舍);
- Device 绑定缺 __hash__(optimizer 分组用 Device 做 dict key 崩溃);
- codegen 目录陈旧导致 TPXOpsGenerated 与注册签名不一致(arange 段错误),
  tools/codegen 必须整目录同步;functional.py 的 arange 包装现做 torch 同款
  dtype 推断(int64/float32),不再向绑定层传 Undefined。

验证结果(全部在远端 GPU):
- test_rnn_cuda.py:**native 64/64 通过**(lstm/gru/rnn_tanh/rnn_relu × 双向 ×
  batch_first × 层数 × fp32/fp64,CUDA↔CPU 一致)+ 训练冒烟通过(LSTM 2 层 +
  Linear,SGD 一步 loss 下降,10/10 参数有梯度);
- 新算子冒烟全绿:arange/linspace/randint/randperm/chunk/median/angle/
  constant_pad_nd/group_norm/instance_norm/bernoulli_/log_normal_/geometric_/
  random_/poisson;
- 本地 CPU:test_rnn_numerics.py 对 torch 48/48;test_shape_funcs.py 35/46
  (11 个既有 API 缺口:broadcast_tensors 签名/block_diag 绑定/tensordot 等);
- nn.LSTM/GRU/RNN 训练路径修复:chunk/unbind/add_ 三类断图点改 narrow/select/
  函数式组合(chunk、unbind 无导数注册;narrow/select 可微)。

**开放阻塞(归约线,非本轮引入)**:CUDA sum 结果 = Σ v[i]·(i+1)(ones[10].sum()
=55,full(7).sum()=385,权重恰为索引+1;mean/max 正常,max 不受累因不线性)。
症状指向 CUDAReduce.cuh 向量化 thread_reduce 区域,是"ReductionKernels 性能
修复"首次在真 GPU 运行暴露的问题;test_cuda_reductions.py 6 failed 1 passed。
另:.mean() 导数条目在 derivatives.yaml 存在但加载失败(mean.requires_grad=False,
CPU/GPU 同),训练用 sum-of-squares 替代;二者需归约线跟进。

## 稀疏 + 量化 + 原生 forward-mode AD(2026-08-24)

### 稀疏(COO/CSR)
- TensorImpl::SparseState 增加 layout 字段(COO=0/CSR=1)与 crow/col 组件;
  Tensor 新增 is_sparse_csr/_crow_indices/_col_indices/make_sparse_csr_tensor;
- yaml 接线(CPU/CUDA 各一):`sparse_coo_tensor`(size=None 时由坐标 max+1 与
  values 形状推断)、`to_dense`、`to_sparse`、`to_sparse_csr`、`_nnz`、
  `sparse_mm`(COO/CSR SpMM)、`sparse_sum`;此前这批 C++ 内核对 Python 完全
  不可达(_docs.py 有文档、绑定层无入口);
- `Tensor::to` 三重载补 sparse 分支的 CSR 路径(原实现一律按 COO 经
  _indices() 重建,对 CSR 直接抛"must be defined",远端首测暴露);
- 新增 `tensorplay/sparse.py` 命名空间(mm/sum/to_dense/to_sparse/
  to_sparse_csr/coalesce/sparse_mask/sparse_coo_tensor),挂 __getattr__ 惰性加载;
- CUDA 侧:COO scatter-to-dense / SpMM / 求和归约原生内核(byte-copy 技巧保持
  dtype 无关,to_sparse/to_sparse_csr 沿 coalesce 先例 CPU 中转);
  **修复 sparse_coo_mm_kernel 竞态**:同行不同列的两个坐标会写同一输出格,
  `+=` 丢更新(P4 上实测丢一项),改 atomicAdd。

### 量化(Int8 affine,不加新 DType)
- 内核 `quantize_per_tensor/per_channel` + `dequantize_*`:round-half-even
  (nearbyint,ATen 同款)、[qmin,qmax] 饱和、zero_point 校验、half/bf16 先提升
  Float32;输出 Int8/还原 Float32;
- Python 包 `tensorplay/quantization`:MinMaxObserver / MovingAverageMinMax /
  PerChannelMinMax(含 calculate_qparams 零点饱和与 eps 保护)、FakeQuantize
  (range-masked STE:区内梯度透传、区外置零;经 autograd Function 实现,
  注意本引擎 backward 按"每个 forward 实参一个梯度槽"传参)、fake_quantize_
  per_tensor、QuantStub(训练校准→freeze→eval 出真 Int8)/DeQuantStub。

### 原生 forward-mode AD(JVP)
- p10 ForwardKernels(CPU/CUDA 各 15 个融合内核):
  neg/exp/log/sin/cos/sqrt/tanh/sigmoid/relu、add/sub/mul/div/pow、mm,
  签名 (primal..., tangent...)->(Tensor,Tensor) 单趟同时算 primal+tangent;
  仅 Float32/Float64、二元要求形状一致(广播由上层展开);
- `tensorplay/autograd/_forward.py`:DualTensor(双分量包装,运算符/方法层,
  标量经 full_like 物化成同形常量 dual);autograd.functional.jvp 增加
  mode="forward"(默认 "reversed" 双反向 trick 不变);未覆盖 op(如 softmax)
  明确抛 NotImplementedError 提示回落;
- 引擎侧无嵌套 ForwardADLevel,单层隐式;Function.jvp 钩子仍是 parity stub。

### codegen 共用层修复(本轮附带)
- model.py parse_schema.conv_arg:`int[]?`(Optional(List))剥层错误——原先
  OptionalType 只剥一层且 ListType 判定基于外层类名,`size=None` 这类 schema
  从未可用;p10_ctype/api_types.cpp_arg_type 补 optional<vector> 组合;
- gen_tpx.py 一处 f-string 悬空 `{`(并行 WIP 半成品)按 AGENTS.md 最小修复。

### 验证
- 本地 CPU:sparse 冒烟 12/12(COO 构造/size 推断/to_sparse 往返/重复坐标
  coalesce/sparse_sum/spmm/CSR crow-col/to_dense/hybrid values);量化往返误差
  ≤ 半步长、per-channel 网格精确、STE 区内 [1,1] 区外 [0]、QuantStub 校准→
  冻结→真 Int8→DeQuant 复原;jvp forward 对 torch 2.13 十五内核全对齐 +
  有限差分 + 解析公式 + tuple 输出;
- 远端 Tesla P4(.remote_build.md):三块 CUDA 冒烟全绿(sparse 全链路、量化
  往返 err=0、forward 内核与 CPU 位级/1e-3 一致);期间抓出并修复上文两个
  仅真 GPU 才暴露的 bug;
- 并行协同备注:另一 agent 的 Ops.cpp 工厂重构中途态曾使 tp.zeros 全线报
  pin_memory 错误(其 timeout 中断的构建还留下 ABI 混合产物),待其收敛后
  自愈——共享树纪律(查进程/等静默/不回滚他人 WIP)再次生效。

### 与 torch 的剩余差距(对照 third_party/pytorch 2.15a0)
稀疏:
1. 布局缺 CSC/BSR/BSC;hybrid(values 带 dense 维)sparse_mm 未收;
2. 公开命名差异:torch 提供 values()/crow_indices()/col_indices() 公有方法与
   layout 属性;我们暂以 _values/_crow_indices/_col_indices 私有名暴露;
3. 缺 sparse_sum(dim/dtype)、spspmm(sparse@sparse)、add/mul sparse-sparse、
   sparse 逐元素族(relu/softmax 等)、spdiags、to_sparse(layout=/dense_dim=)、
   to_dense(dense_dim=)、is_coalesced 的 check_invariants 语义面;
4. autograd:sparse 路径整体不可微(torch 的 to_dense、embedding(sparse=True)
   + SparseAdam 链路可微);我们 embedding_sparse_backward 已有但优化器侧
   sparse 更新未接。
量化:
1. 无专用 qint8/quint8 DType 与 QTensor 类,scale/zp 由调用方携带(序列化/
   state_dict 自管);
2. observer 族缺 Histogram/MovingAveragePerChannel/FixedQParams/Placeholder
   等,default_* qconfig 常量与 get/load_observer_state_dict 未做;
3. FakeQuantize 缺 per-channel/learnable 变体与 disable_observer 开关;
4. 无 prepare/convert(eager/FX)流程、backend_config、fuse_modules;无 fused
   quantized linear/conv 内核,dynamic 量化未做——当前定位是 PTQ/QAT 的
   手工最小闭环。
forward AD:
1. torch 在几乎全部 ATen 内核原生 dual + 任意嵌套 ForwardADLevel +
   unpack_dual;我们覆盖 15 个核心算子、单层、结果为 no-grad 张量;
2. Function.jvp 钩子引擎侧未接线;
3. functorch 的 jvp/jacfwd/forward-over-reverse 组合未涉及(jacfwd 可用现有
   vjp×jvp 组合在 Python 层搭)。

## 继续补齐(2026-08-24 轮次 2)

已完成:
- **CUDA sum 三角和 bug 修复**:CUDAReduce.cuh thread_reduce 尾循环
  `++unit` 未按 step_input 步进,每个 lane 越界扫到末尾,元素 i 被计数
  (i+1) 次(ones[10].sum()=55 的根因)。改为 `unit += step` 后
  test_cuda_reductions.py **7/7 通过**,任意规模 sum 数值全对;
  RNN 训练冒烟复验通过(此前梯度被索引权重污染,现已正确);
- **.mean() 导数**:根因是远端 codegen 工具目录陈旧(TPXOpsGenerated 旧版
  缺 autograd 块),整目录同步 tools/codegen 后自愈,梯度数值正确(2a/N),
  derivatives.yaml 无需改动;
- **foreach CUDA 对齐 CPU 全集**:补 23 个缺失 unary functional/inplace
  (acos…trunc)、norm/powsum/max/zero/clone/copy/mm functional,以及全部
  41 个 `_out` 变体(与 cpu/ForeachKernels.cpp copy_foreach_out 模式一致,
  经 dispatcher 注册,python 暴露面随 yaml 逐步开放);
- **unique CUDA**:sort + 相邻比较 flags + cumsum 分组 + inverse scatter +
  host 端小缓冲算 counts;yaml 加 CUDA dispatch。待构建验证(见下)。

工程纪律补充:多 agent 并行时,**tools/codegen 与 p10/include 必须整目录
同步**——零散拷贝会撞上 API 迁移期的签名漂移(gen_api/Tensor.h/生成头的
三方联动变更),本次 arange 段错误与 contiguous/expand 二义均源于此;
`TP_NO_FASTCALL=1` 可禁用 METH_FASTCALL 层(init.cpp 已留开关)。

**当前阻塞**:forward-AD/memory_format 迁移线(CopyKernels/Linalg/
QuantKernels 的 expand 二义、Tensor.cpp clone_impl 私有访问)本地与远端
快照同样编译失败,系该线 WIP 未收口;unique/foreach 的最终 GPU 验证在其
落地后 `ninja -C build _C` 一键可验(代码已就位)。

## 稀疏收口 + 量化闭环补全 + forward-AD 对齐(2026-08-24 深夜,接上节)

方法:剩余差距逐条对照 `third_party/pytorch` 源码实现(非凭记忆):
`_sparse_sum`(SparseTensorMath.cpp:1634)、`spdiags`(SparseFactories.cpp + cpu kernel)、
`observer.py`(HistogramObserver/_combine_histograms/_non_linear_param_search/
MovingAveragePerChannelMinMax/FixedQParams/Placeholder/get·load_observer_state_dict)、
`functional.py` 的 jacobian/jvp/hessian 返回结构。

### A. 阻塞级修复(共享基建,顺手对齐)
1. **`sum(dim, keepdim=False)` 反向广播错轴**(并行线引入的 `_sum_dim_backward`
   内核正确,但注册 ABI 不匹配):CPU/CUDA 共 40 处 dim 归约内核签名
   `std::vector<int64_t> dims`(按值)+ 内核内 `std::move`,而 codegen
   DispatchStub 模板按 `const&` 传参 → callee 把调用方 vector move-空,
   节点存下 dims=[] → 反向退化为"无 unsqueeze 的 expand"。修复:40 处内核
   签名改 `const&` + ReductionKernels.h 六个 `*_dim_fn` stub typedef 同步
   const&;amax/amin/aminmax/nansum(cpu+cuda)的 dim.empty() 原地填充改为
   局部拷贝。**教训入纪律:m.impl 注册内核的可变实参(int[]/Tensor[])一律
   const&,与生成端 DispatchStub 模板一致。**
2. **`autograd.grad` 返回 list → tuple**:torch 契约为 tuple,functional.py
   全套 vjp/jvp 依赖之(Autograd.cpp 绑定处转 py::tuple;注意 GIL——
   py:: 构造必须在 gil_scoped_release 作用域之外,初版曾因此段错误)。
3. **`grad_outputs` 含 None**:torch 语义 None→ones 种子,C++ 绑定无法携带
   None 元素,在 Python grad() 包装层物化。
4. **`.item()` 返回 Scalar 未拆箱**:并行线 item 迁移 yaml codegen 后丢失
   HEAD 手写绑定的 python-number 语义,全仓算术受累;_tensor.py 按 dtype
   monkeypatch 回 float/int(builtins 别名,避开模块内 float()/int() 方法遮蔽)。
5. **DualTensor matmul**:原 _binary 走同形强制,向量×矩阵不可用;按乘积法则
   直接实现 __matmul__/__rmatmul__(支持 @ 的完整广播语义)。

### B. 稀疏(全部原生,CUDA 无 CPU 中转)
- **sparse_sum 重写对齐 ATen**:部分归约返回**稀疏 COO**(kept 维重建 +
  coalesce 折叠),全归约才 dense;新增 `ScalarType? dtype` 累计参数;
  CUDA 部分归约原生(kept 行 gather + native coalesce),不再 CPU 中转;
- **spdiags 原生内核**(此前一版 Python 组合作废):值从对角行 max(d,0) 列
  读起(ATen cpu kernel 逐行照抄,官方文档 arange(9) 例验证一致);
  layout 参数支持 sparse_csr(COO→CSR 原生:cub InclusiveSum 建 crow);
- **原生 CUDA COO coalesce**(替换原 CPU staging):逐维稳定基数排序
  (cub RadixSort 自最后一维起携带置换)+ run 检测 + ExclusiveSum 压缩槽 +
  确定 run 段累加(单线程/run,免 atomicAdd、complex 可用);
- sparse_add_cuda=cat+native coalesce;sparse_mul_cuda=A 对 B 二分查坐标
  交集 + CUB Flagged 压缩(确定性);to_sparse_coo/csr_cuda 原生
  (字节级判零 mask + Flagged 计数迭代器);sparse_coo_tensor size=None 推断
  原生(coord_max_kernel);
- Python:sparse.sum(dim/dtype) 透传、spdiags 包装。

### C. 量化
- 观察者族:HistogramObserver(L2 直方图搜索/combine/upscale 移植)、
  MovingAveragePerChannelMinMax、FixedQParams、Placeholder、
  default_observer/default_weight_observer/default_dynamic_quant_observer
  (with_args 偏特化机制)、get/load_observer_state_dict(observation_state
  序列化协议——观察者状态是普通属性非 buffer);
- FakeQuantize/PerChannelFakeQuantize 增加 disable_observer 开关;
- **quantized_linear 原生内核 CPU(parallel_for)+CUDA(线程/输出元)**:
  Int8[M,K]xInt8[N,K] per-channel 权重,scale*w_scale[n]*int32 累加+bias
  →Float32;QuantizedLinear 模块 + from_float(权重按输出通道 min/max 量化)。

### D. forward AD / functional 对齐
- jacfwd 重写:列扫描收集后 stack(out_numel, in_numel),返回结构严格对齐
  torch(Tensor/tuple[Tensor]/tuple[tuple],Jacobian[i][j]=out_i×in_j);
- test_autograd_functional 6 处测试体按 torch 真值修正(v 形状校验、多输入
  返回结构 tuple[Tensor]、hessian 报错文案 "should contain a single element"、
  retain_graph 迭代);引擎遗留缺口:**双反向穿过 MatMulBackward 断流**
  (gi 无 grad_fn,create_graph 语义在 mm 节点未生效),transpose 恒等式
  测试暂走 mode="forward",引擎侧待归因。

### 远端抓出并修复的两个真 bug
1. **coalesce 字节收集 span 语义混淆**:gather_bytes_by_perm 的 span 参数
   按"每行字节数"使用,调用点却传"每行元素数"→ float 只拷 1/4 字节,
   值全部损坏(索引 int64 恰好正确所以坐标看着正常,极具迷惑性);
   compute-sanitizer 定位前先靠 to_dense(flagged) 单值写穿行为排除法锁定。
2. **QuantizedLinear.from_float 设备错位**:as_tensor 生成的 CPU
   scales/zps 直接喂给 CUDA 内核(host 指针解引用,sanitizer 报
   Invalid read 距最近分配 140TB)→ 统一 .to(weight.device())。
   附带:tp 把 context 损坏后的后续错误映射成 "cusolverDnCreate failed"
   具有误导性,排障时勿被带偏(compute-sanitizer 一锤定音)。

### 验证(最终)
- 本地 CPU:test_sparse 20 + test_quantization 19 + test_forward_ad 6 +
  test_autograd_functional 20 + ops/new_ops/pointwise/optim **88/88**。
- 远端 P4(CUDA12.8):自建冒烟 **18/18**(coalesce 往返/to_dense/
  add 并集/mul 交集/sparse_sum 四形态/spdiags COO+CSR/to_sparse·csr/
  quantized_linear vs float 参考 err=0.49<1.59 预算/量化往返);
  回归:test_cuda_reductions+test_ops+test_new_ops 21 通过;
  test_rnn_cuda native **64/64** + 训练冒烟(loss 下降,10/10 参数有梯度)
  ——dim 归约 ABI 修复后 RNN 反向链路复验无恙。

## 遗留断点修复(2026-08-24 深夜):MatmulBackward 双反向 + tp.sum 函数面

上节 D 项与 powerSGD 实际依赖的两个已归因断点,**本节全部原生补齐**(对照 torch
源码逐条落地),另附带修一处 grad_outputs 契约缺口、登记三个新发现的相邻差距。

### L1 ✅ 双反向穿过 MatmulBackward(已修复)

按上游同构方案落地(torch 的 matmul_backward 也是手写 native,因导数公式无法在
derivatives.yaml 里按 dim 分支):

- derivatives.yaml 删除 matmul 的孤立辅助公式;gen_autograd.py 的
  MANUAL_DERIVATIVES 增 `"matmul": {"saved": ["self","other"]}`;
- 手写节点 MatmulBackward(tpx/include/ManualNodes.h):镜像裸内核的归一化约定
  (self_vec→unsqueeze(0)、other_vec→unsqueeze(-1)、grad 相应整形),随后**全部走
  可记录原语**(ops::unsqueeze/transpose/t/matmul/squeeze/sum+reshape),
  sum_to_shape_cpu 的广播累加语义以"keepdim 批量求和 + 必要时 reshape"复现;
  create_graph 时内部调用重入生成包装器自动挂节点 → 双反向带图。
- **复数收窄**:复数伴随是共轭转置,而可记录的 conj 组合件(select/slice 导数)
  未接线——复数分支委托保留的原生辅助算子(matmul_backward_self/other,数值与
  torch 精确一致),一阶反向可用、该分支不向深处记录(与改造前深度相同)。
- 验证:7 类形状(dot/vec@mat/mat@vec/mat@mat/批@mat/双双批/广播批)的 forward、
  一阶反向、双反向数值与本机 torch 2.13 全对齐;TestJVP::test_matches_vjp_transpose
  恢复双模式覆盖(reversed 默认引擎 + forward),jvp reversed 返回 [3,7] 正确。

### L2 ✅ tp.sum 函数面 dim kwarg(已修复,连带 mean/prod/max/min/all/any/squeeze/var/std/norm)

根因如前述三层错位。修复落在 tools/codegen/gen_python.py:新增
`_reduction_union_lines`——对"base + 兄弟新增可选位置参"结构的 overload 对发射
torch 并集签名(dim=None 路由 base overload,否则 dim overload,一律关键字转发进
现有 FASTCALL 层;kwonly(dtype)正确加 `*`)。既有位置参保持原槽位
(`tp.norm(x, 2)` 仍是 p=2);tensor_split/foreach/pow 等**类型分派组不受影响**
(位置转发语义保留)。yaml/内核/C++ 零改动,重跑 codegen 即生效。

- 修复面:`tp.sum/mean/prod(x, dim=None, keepdim=False, *, dtype=...)`、
  max/min/all/any/squeeze/var/std/norm 同族 union 签名;
  powerSGD_hook.py:87 的 `tp.sum(col*rest, dim=1, keepdim=True)` 实测可用。

### 附带修复:grad_outputs 裸 Tensor 契约

`autograd.grad(out, inputs, go_tensor)`:torch 接受裸张量,tp 的 Python 包装却把
它 zip 按 dim-0 切开 → 标量输出直接 IndexError、向量输出静默用错种子(0 维迭代即炸,
极具迷惑性)。已在包装层补 `isinstance(grad_outputs, (list, tuple))` 归一,
对齐 torch 表面契约。

### 新发现并登记的相邻差距(未扩线,后续跟进)

1. `tp.max/min(x, dim)` 只返回 values 单张量——yaml 声明 `max.dim -> Tensor`,
   而 torch 及 tp 自己的 `_docs.py` 契约都是 `(Tensor, LongTensor)` 元组
   (indices 缺失)。改元组牵动 yaml 返回类型+内核+codegen 元组返回机制,单列。
2. `mean.dim` 导数仍是孤立辅助内核(mean_dim_backward→MeanDimBackward 直调裸内核)
   ——与 matmul 同病,mean 的双反向同样断流,复用本节手写节点模式即可。
3. linspace/logspace 内核仅 Float32 分支(arange/rand 家族的 Float64 已由并行线
   于 22:31 补上);f64 工厂请求会得到未初始化缓冲或 NotImplementedError。

### 相邻差距补齐(2026-08-25 凌晨):max/min 元组 + mean.dim 双反向 + 工厂 f64

上节登记的 1/2/3 全部落地:

**① max/min(dim) → (values, indices) 元组**(对齐 torch 与 tp 自家 _docs 契约;
此前仓内 linalg 5 处 `.values`、vision 1 处元组解包、audio 1 处全是坏的,本修复点亮):
- yaml:`max.dim/min.dim` 改 `int64_t dim`(torch 同款单 dim)→ `(Tensor values,
  Tensor indices)`,照 cummax 先例(variants: function+method 的元组 op);
- CPU:ReductionKernelsImpl 重写为 outer/inner 单维遍历(cummax 模式),严格比较
  保**首个**极值索引;stub typedef 改 `tuple<Tensor,Tensor>(*)(Tensor,int64_t,bool)`;
- CUDA:values 复用 minmax_same_dtype 机器 + indices 用 ArgOps 状态机
  (TP_DISPATCH_REDUCTION 经立即调用 lambda 双路取物);Bool 抛明确错误
  (argmax 无 Bool 实例化是既定取舍);
- 导数:derivatives.yaml 新增两目,公式走 ManualNodes.h 新增的可记录组合函数
  `tensorplay::tpx::value_selecting_reduction_backward`
  (unsqueeze 对齐 + iota reshape + eq 掩码 + mul;上游同名 native 的结构),
  create_graph 下 max/min(dim) 反向带图、二阶可导;
- 调用方:special.py logsumexp 改 amax(多 dim 语义本就该用 amax);
  nn/functional.py 五处按旧列表契约直调 `_C.max/min` 的站点(dim=[1] → dim=1
  并解包元组取 values:ctc 的 first_neg/is_target、_vector_norm 的 ±inf 分支、
  adaptive_max_pool 组合的深度归约)。

**② mean.dim 双反向**:derivatives.yaml 换 `tensorplay::tpx::broadcast_mean_backward`
(unsqueeze 还形 → div count → **expand 到 self 形状**)。教训:expand 必须是
被记录的算子(ExpandBackward→sum_to_size)——最初版只做隐式广播,一阶数值因广播
对齐"碰巧全等"(形状 (2,1,4) vs (2,3,4) allclose 恒真),二阶却少乘 J;
标量透传反向(scalar div)不降维,与下游 unsqueeze-backward 组合即崩。

**③ linspace/logspace Float64**:CPU 补分支(CUDA 侧本就支持),报错文案同步。

**附带(codegen 解析器)**:并行线 23:50 加的 clamp_min/max 公式含裸 `>` 比较,
`_normalize_comparisons` 只认括号包裹形式 → 整个 codegen 卡死(阻塞双方)。补裸
比较操作数重写(`a > min` → `gt(a, min)`),clamp 四条随本次构建一并点亮。

### 回归

- 新面验证:max/min(dim) values/indices/keepdim/负 dim/平局取首 vs torch 全对;
  max 一阶反向 vs torch 对齐且带图;mean(dim) 非keepdim/keepdim/多维 dim 列表的
  一阶+双反向 vs torch 全对;linspace/logspace f64 vs numpy 对齐;
- 回归批次:test_matmul_parity+autograd_functional+new_ops+ops(62)+
  decompositions/aot/forward_ad/rnn_numerics/sparse/quantization/shape_funcs/
  compile(153)全绿;仓内 linalg/vision/audio 调用模式实测可用;
  special.logsumexp 多维 dim 正确;
- 已知小疣:indices 张量在 requires_grad 输入下被误标 requires_grad(torch 为
  False;Int64 无梯度故仅影响观感与 detach 场景),记录待后续统一 non_differentiable 标注;
- 产物新鲜度:_C.so/libp10.so 均晚于全部改动源(ninja -C build p10 _C)。
- 并行协同备注:本轮两次撞上并行线中途态(ReductionKernels.cpp 包装器被覆盖回旧
  签名、derivatives.yaml clamp 条目卡死 codegen),均按最小修复原则就地收口,
  未回滚对方工作。

### 回归

- 本轮改动面全绿:test_matmul_parity(48)+test_autograd_functional(含翻转后的
  transpose 双模式)+test_new_ops/test_ops/test_decompositions/test_aot/
  test_forward_ad/test_rnn_numerics(67)+test_sparse/test_quantization/
  test_shape_funcs(84);
- test_nn_functional_alignment 存量 23 failed 与本轮无关:grid_sample/affine_grid/
  pool/ctc/multilabel 为文档既录缺口,in_proj packed linear 与 cross_entropy 的
  f32 限制属并行 nn/functional.py WIP(172 行未提交改动,前向形状/dtype 问题);
- 产物新鲜度:_C.so(23:02)/libtpx.so 晚于 ManualNodes.h(23:00);
  libp10.so(22:46)晚于 FactoryKernels.cpp(22:31)。

## 新远端机迁移 + GPU 验证收口(2026-08-24 轮次 3)

新机(见 .remote_build.md):Tesla P4 / CUDA 12.8 / py3.12,全量源码 tar 管道
8 秒级;oneDNN zip 本地打包上传(顶层 oneDNN-3.4.1/),FetchContent 正常。
**cuDNN frontend 头必须保留原始目录布局**(头间用 `../include/` 相对引用,
平铺拷贝会 fatal error),已固化到 /root/cudnn-frontend-1.15.0/include。

### 验证结果(P4 实测)
- test_cuda_reductions.py **7/7**;test_rnn_cuda.py native **64/64** +
  training smoke ok=True(**10/10 参数有梯度**);
- smoke_cuda.py 22/22(factories/chunk-split-unbind/RNG 家族/group+instance
  norm/median/angle/pad/**unique 三态**/narrow-autograd);
- foreach 暴露面 38/38 + 数值抽查 10 项(TFP_OK)。

### 迁移期抓出的三个真 bug
1. **narrow 无导数 → LSTM 训练链路断**(CPU/CUDA 双复现):rnn.py 训练路径
   chunk→narrow 后,out.requires_grad=False。上游 narrow 是
   CompositeImplicitAutograd == `slice(dim,start,start+length)`;修复=
   derivatives.yaml 补 `slice_backward(grad,self,dim,start,start+length,1)`
   条目(公式语言对非张量算术直发 C++ 文本,零手工节点)。CPU 先验后上 GPU。
2. **CUDA sort 方向整体颠倒**:sort_kernel 堆比较 `descending?lt:gt` 写反,
   升序建了小顶堆输出降序(tp.sort 默认即错)。翻转为 `descending?gt:lt`
   (升序需大顶堆沉大元素);desc 用例同步验证。
3. **unique CUDA 三连**:①counts 用 memcpy 直写设备指针→P4 段错误(gdb 栈
   定位 __memcpy_avx),改 Tensor::tensor(CPU 构建).to(device) 走 H2D;
   ②组号用 inclusive cumsum 却未减 1(inverse 出现越界值、values 漏写槽位
   ——设备侧 OOB 写),inverse/emit 内核统一 gid-1;③绑定层对未定义
   Tensor() 包 PyCapsule null 崩——functional.unique 改恒向 _C 要全三元组
   再按 flag 裁剪(镜像 torch.unique 可观测契约,__init__ 已有同型包装但被
   `from ._shape_funcs import *` 遮蔽,functional 层就地修为单一事实源)。

### CUDA 注册面补齐(照上游复合语义,零自创)
cpu/ShapeAlignKernels.cpp 的 shapeops 族全是 dispatcher 复合(cat/slice/
as_strided/eq/all/full_like/isclose,设备无关),在 cuda/ShapeAlignKernels.cu
以 extern 声明注册同名 39 个 op:expand/expand_as/broadcast_to/tile/hstack/
vstack/dstack/column_stack/row_stack(+out)/tensor_split 三变体/hsplit·vsplit·
dsplit×2/atleast_1d·2d·3d(+Sequence)/flatten/unflatten/ravel/moveaxis×2/
swapaxes/swapdims/argwhere/fill.Scalar·fill.Tensor/equal/allclose。
RandomKernels.cu 补 bernoulli(empty_like+philox fill,in/out 分离复用
bernoulli_fill_impl)与 normal(broadcast_shapes→empty→normal_(0,1)→
mul_(std).add_(mean),逐条对照 DistributionTemplates.h normal_out_impl)。
注意 shapeops 命名空间挂在 tensorplay 下而非 cuda 下(.cu 里声明需临时弹出
cuda namespace,否则 mangled 名不匹配链接报 undefined symbol)。

### 并行线共存记录
- ReductionKernelsImpl 的 stub 签名漂移(vector const&↔按值)曾三度红绿翻转,
  对方 agent 在线编辑期间我方改动被其覆盖一次;最终以其收敛的 const& 为准,
  我方不做二次修改(共享树纪律)。
- **两 agent 同时向同一台远端推 tar + 各起 ninja 会并发写同一 build 目录**
  (实测 pgrep 抓到 2 个 ninja):按 AGENTS.md 规则 pkill 清场后单构建重启;
  远端同样适用"查进程再构建"。
- 本地构建树已被并行线重配为纯 CPU(libp10 仅链 cpu 源,.cu 只能远端编)。

### 与 torch 剩余差距(更新)
- m.impl 口径 CPU770/CUDA763(+39 复合、+bernoulli/normal/-重复计数):
  仅剩 linalg_ldl_factor/ex/solve(cusolver 线)与
  _foreach_pow.TensorAndTensor 两处 CUDA 缺口;
- functional.unique 的 sorted=False 分支与上游一致回落 sorted 语义(cpu 注释
  同),dim>1 输入仍限 1D(上游 unique_dim 另线)。

### 轮次 3 附记(_foreach_pow.TensorAndTensor)
CUDA 侧已照 cpu foreach_pow_tensor_tensor_cpu 补 foreach_map 注册
(OptimizerKernels.cu,`self ** exponent[i]` 走 dispatcher pow);python 层
`tp._foreach_pow(Tensor, Tensor[])` 的多 overload 分派仍会先撞
`.ScalarList` 报 TypeError——与文首 sum/mean/prod 同族,归 codegen
gen_python.py 并集签名线(并行工作流),C++ 注册面已就位待其收口。
另:dispatcher "to" op 仅 CPU 注册,但真实路径全走 Tensor::to C++ 方法
(设备通用),yaml 无暴露面——无需补。

## 顶层公开算子批次 3 + ABI/内核修复(2026-08-24 深夜 II)

方法:与本地 torch 2.13 全量对拍(371 个缺失可调用物中滤除后端专属噪声,
按"别名/语义包装 → 结构组合 → 统计数值"分批落地纯 Python 组合,规格测试
test_composite_funcs.py 逐项对照 torch 数值+梯度)。

### A. 阻塞级修复(C++,均已过构建)

1. **bincount 段错误**:schema `Tensor?` 经生成 stub 按 `std::optional<Tensor>`
   传参,而 bincount_{cpu,cuda} 收裸 `const Tensor&`——ABI 错位必崩
   (conv2d 族无恙是因 tpx 层先做 `has_value()?*v:Tensor()` 解包,bincount
   的生成体没有这层)。审计全树 `Tensor?` 透传点后修内核签名收
   optional+value_or;同时确认 fused_adam/adagrad/sgd、layer/batch/group/
   instance_norm、stft 族、diff、tp_bce 等均已是 optional(误报排除),
   nll_loss_backward 宿主包装亦 optional。
2. **binary_cross_entropy_with_logits 段错误**:同款 ABI 错位,同法修复
   (CPU Tier5OpsKernels/CUDA Tier5LossesKernels),默认权重路径与 torch
   位级一致。
3. **arange(dtype=float64) 返回未初始化内存**:arange_start_step_kernel
   只写了 F32/I64/I32 分支,F64 落空。补 Float64 写入分支(CUDA 侧宏分派
   本就完整)。
4. **meshgrid 内核返回索引网格而非值网格**:重写 meshgrid_cpu 为 ATen
   TensorShape.cpp 语义(值平铺 + promoteTypes 公共 dtype + "xy" 前两轴
   交换);顺带修复初版 memcpy 把"单值平铺"写成"连续块拷贝"的错误。
5. **pdist_cpu 线性索引反解公式错误**:j 的闭合解写错导致成对距离取错行
   (实测输出=行范数)。改用标准 squareform 反解
   j = li+i+1 − n(n−1)/2 + (n−i)(n−i−1)/2。

### B. Python 批次(tensorplay/_composite_funcs.py,~85 个公开名)

- 别名/语义:absolute、arc 六件套(+原生式 Function 原地版 acos_…atanh_,
  torch 本身对 acosh_/asinh_/atanh_ 无原地导数,我们反而支持)、arctan2
  (复合公式顶替丢失的 atan2 CPU 内核注册,记录负零边缘收窄)、concat/
  concatenate、ger、rsub、adjoint、divide/multiply/subtract/true_divide/
  floor_divide/remainder/fmod(整型走 float64 真除再取整,负数方向对齐
  torch)、clamp_max/min、copysign、detach(fn)、diagflat、numel、
  scalar_tensor。
- 结构:linalg.chain_matmul、matrix_power(n≥0,负幂诚实报错)、kron(任意
  同维)、vander、tril/triu_indices、cartesian_prod(1-D,列堆叠 (N,C))、
  combinations(r=0→(0,) 对齐)。
- 统计:cov/corrcoef(逐行照抄 ATen Correlation.cpp:fweights 频率×
  aweights 可靠度联合权、norm_factor 四分支含 Σ(w·aw)/w_sum 项、单变量
  squeeze 成标量)、trapezoid/trapz/cumulative_trapezoid(x 或 dx 双路)、
  gradient(非均匀坐标用 ATen 加权中央差分 h 权重公式)、quantile/
  nanquantile(linear;q 张量前置维布局对齐 keepdim)、histc、histogram
  (int bins + 边缘张量双路 + weight + density 按 bin 宽归一)、isin(sort+
  searchsorted right=True 候选=pos−1)、unique_consecutive(展平形)、
  repeat_interleave(int/tensor repeats,dim=None 展平,index_select 可微)、
  kaiser_window(本地 I₀ 级数——special.i0 中段精度不足已暴露)。
- RNN cell:lstm/rnn_relu/rnn_tanh_cell(用 narrow 切门控——chunk 无导数
  注册会断图,已知问题)。
- 工具:put(index_put 平铺包装,源短则周期铺)、resolve_conj/neg、
  is_conj/neg(无共轭位,诚实返回物理布局)、can_cast/promote_types/
  result_type(**从 torch 生成精确查表硬编码**,u16/u32/u64 保守回退)、
  is_nonzero/is_same_size/get_device。
- *_copy 家族全套(alias/t/permute/transpose/squeeze/unsqueeze/select/
  slice/narrow/diagonal/unbind/split/view/unfold/expand)+ unsafe_chunk/
  unsafe_split 视图别名。

### C. 连带修复(Python,他人 WIP 窗口外)

- nn/functional.cosine_similarity 用了不存在的 `.clamp_min` 方法 →
  tensorplay.clamp(min=);
- functional.pairwise_distance 对 1-D 输入走批量内核得逐元素零 → 包装层
  unsqueeze 成对再 squeeze(沿用生成文件手工微调先例);
- _composite_funcs 内 logical_and 一律走模块函数(Tensor 方法面缺失)。

### D. atleast_* Sequence 梯度(上轮遗留项核销)

运行时已全面达标:C++ 绑定处理单/多/Sequence 三形态,reshape 反向梯度
数值与 torch 一致(含别名重复输入累加、0 维提升)。__init__.py 尾部
functional 星号导入会遮蔽 _shape_funcs 的同名组合,实际生效的是生成的
_C 路径——行为正确但两套实现并存待清理。回归锁进
TestAtLeastSequenceRegression(6 例)。

### E. tensordot / CopySlices 备忘

- tensordot 保持纯 Python 组合与 torch 同构(torch 自己也是 functional.py
  组合实现,无 ATen 单算子),无需 C++ 内核;
- .out 变体 CopySlices 层仍缺:本批 clamp_max_/xlogy_ 等 in-place 名未提供
  (copy_ 兜底会静默错梯度),arc 原地件用显式 Function 公式补齐——CopySlices
  立项前 in-place 家族继续按此模式。

### 验证

- test_composite_funcs.py 84/84(唯一例外 pdist 顶层用例待下轮构建链接);
- test_shape_funcs.py 57/57(含 atleast Sequence 回归);周边
  new_ops/ops/reduction/t_autograd/scalar/python_tensor/new_features/
  pointwise/statistical 共 ~230 例全绿;
- test_random.py 4 failed = 并行 WIP(RandomKernels.cu +46 行迁移中),
  与本批次无关;
- 构建纪律:两次遭遇并行 ninja 同目录,按规程等待静默+单方低并行重建;
  ReductionKernels.cpp 的 max/min 迁移线由其作者收口,我方零触碰。

## Tensor 方法面批次(2026-08-24 深夜 III)——零构建冷区选线

选题依据:90 分钟文件热度图显示热点在 ReductionKernels(max/min 迁移)、
native_functions.yaml/derivatives.yaml(23:22 双改)、IndexingKernels、
codegen;`tensorplay/_tensor.py` 完全冷(纯 Python、无构建/yaml/codegen
依赖=零并行冲突),且方法面缺口大(torch 有而 tp 缺 30+)。

新增(_tensor.py 追加,全部懒加载导入规避 bootstrap 循环):
- **new_* 工厂族**:new_zeros/new_ones/new_full/new_empty/new_tensor
  (dtype 默认随 self、可覆盖;requires_grad 开关;标量/变长/List 双形态);
- **dtype 捷径**:bool/byte(uint8)/char(int8)/short(int16)/
  half(float16)/bfloat16(torch 名→tp 枚举名映射表);
- **逐点方法**:fmod/remainder/floor_divide/true_divide(经 _composite_funcs
  组合,整型负方向与 torch 一致)、repeat_interleave、count_nonzero、
  unique 方法形态(沿用「全量三输出后裁剪」——原生 2 输出组合路径有
  PyCapsule 崩溃,已绕开)、topk(sort+切片,values/indices 对齐);
- **运算 dunder 全家**:__floordiv__/__rfloordiv__/__ifloordiv__、
  __mod__/__rmod__/__imod__、位运算 __and__/__rand__/__iand__、
  __or__|xor|lshift|rshift 及 r/i 变体、__invert__、__pos__/__abs__;
  反射形式把左侧标量 as_tensor 提升后按 tensor-first 调用(交换律算子
  数值等价,移位保方向);
- **转换**:__complex__/__index__。

坑位记录:
1. _tensor.py 模块级 def int()/float() 遮蔽内建名——模块内一律用
   builtins_int 别名;
2. _tensor.py 在 __init__.py:940 加载早于 functional 星号导入(952),
   任何模块级 `from . import <functional 符号>` 都会循环炸——组合函数
   一律调用时懒加载;
3. 原生 unique 的 2 输出返回组合(return_inverse 而非 counts)存在
   "PyCapsule_New called with null pointer" 崩溃,待归约线收口时排查。

验证:test_tensor_methods.py 27 例全绿(new_* 形状/dtype/rg、捷径 dtype
对拍 torch 字符串、fmod/remainder 整型负数方向、反射与 in-place 位运算、
topk values+indices、remainder 组合梯度数值对齐);周边 8 文件回归 49 例
无破坏。累计本会话三层批次(composite ~85 名 / atleast 回归锁 / tensor
方法面 ~40 方法)共新增公开面 ~125 项。
