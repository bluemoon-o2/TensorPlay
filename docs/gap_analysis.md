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
