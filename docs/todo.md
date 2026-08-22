# 待办 / Handoff — 2026-08-22

> 本轮会话收尾记录。代码已写完但**未编译验证**（按指示只写码不构建）；接手者请按 AGENTS.md 构建纪律执行。

## P0 — 立即验证（代码已落盘，未过编译）

1. **全量重建**：`ninja -C build -j16 _C`（构建前查进程、看内存）。
   产物新鲜度校验：`tensorplay/lib/libp10.so` 与 `tensorplay/_C/*.so` mtime 需新于
   `p10/src/backend/cuda/{Tier5OpsKernels,RNNKernels}.cu`。
2. **RNN 原生 CUDA 移植首编译**（预期会有首轮报错，逐个修）：
   - 新文件 `p10/src/backend/cuda/RNNKernels.cu` + `p10/include/RNNCudaKernels.h`
     —— 照抄 ATen `RNN.cu` 四个 fused cell kernel（lstm/gru × fwd/bwd），
     TensorInfo 寻址改为连续指针，Half/BFloat16 以 fp32 累加。
   - `p10/src/backend/cuda/Tier5OpsKernels.cu`：`rnn_cuda_impl(kind,…)` 层循环
     （mm/addmm + fused cell），替换原先 cuda 套 CPU 的 to_host/to_device 委托；
     `Tensor::tanh()/relu()/mm()` 为 generated 成员，可直接用。
   - 已知风险点：`Tensor()` 未定义张量的 `.numel()/.defined()` 行为、generated
     方法在 p10 内部 TU 的链接可见性、`AccTraits` 特化拼写。
3. **数值验证**：同种子下 `lstm_cuda/gru_cuda/rnn_*_cuda` 对照 CPU 参考与
   本机 torch（fp32 容差）；双向/batch_first/多层各一例。

## P1 — 功能缺口

4. **RNN 反向完全缺失**：`lstm/gru/rnn_tanh/rnn_relu` 在 derivatives.yaml 无条目，
   训练在任何设备都拿不到梯度（先于本轮就存在）。方案二选一：
   - a) derivatives.yaml 加公式 → 调 full-replay backward helper（需照抄
     ATen RNN.cpp `lstm_backward` 结构，fused backward kernel 已移植好待接线）；
   - b) python 层 nn/modules/rnn.py 改为可微算子组合（torch 旧 variable-rnn 路径），
     零原生改动。建议 b 先行解锁训练，a 作为性能项。
5. **CPU rnn_impl 是 fp64 标量参考实现**（Tier5OpsKernels.cpp rnn_impl），
   性能项：换向量化/至少 fp32。
6. **shape 工具函数测试**：`test/test_shape_funcs.py` 已写好从未运行
   （当时 import 被 stale _C 挡住）。构建通过后 `pytest test/test_shape_funcs.py`。
   注意 `unravel_index` 的 float64 取模路径对 |idx|≥2^53 不精确（文档已注明）。
7. **gap_analysis.md 更新**（本轮遗留任务）：Transformer/loss/padding/激活/
   Lazy/Fold 等大块已完成；剩余真实缺口 = Tensor native 方法族
   （index_put/fill/reduce、bitwise/logical 全族、linalg 方法、new_* 工厂、
   cummax/min/prod、resize_ 等）、fft/special/distributions 子包、
   顶层全局配置（另一 agent 正在做 Context.h）、gradcheck/autograd.functional。

## P2 — 备忘

8. **并发协作事故复盘**（共享树纪律）：本轮经历孤儿 nvcc×N、双 ninja 同目录、
   native_functions.yaml 被对方旧缓冲覆盖一次（09:22）、多处半成品 WIP 编译失败。
   编辑 config/*.yaml 与 p10/src/backend/cuda/* 前先确认无人在改。
9. **ReductionKernels 性能修复**已落地（vec8 上限移除→torch 对齐 vec4；
   PTX 94MB→23MB，单文件编译 6min+假死→74s；`thread_reduce` 按 torch
   Reduce.cuh 结构重排，div/mod 移出展开区）。跑一遍
   `test_cuda_reductions.py` 回归确认数值不变。
10. **本机环境差异备忘**：CUDA 13 toolkit（cusolver getrf 签名变更、
    Syevd 大小写、cuDNN9 移除 TF32 旧常量/CUDART_INF_F）——后续新内核
    直接按 CUDA13/cuDNN9 写，勿抄 v8 写法。

## 本轮已完成（摘要）

- shape 工具族落地 `tensorplay/_shape_funcs.py`（broadcast/atleast/hstack 族/
  tensor_split 族/tensordot/block_diag/unravel_index，broadcast_tensors 带
  ExpandBackward 式反向）+ `__init__.py` 接线 + 测试。
- 修复十余处他人 WIP 的编译/链接断点（详见 git diff：CUDAContext/Tier5/Linalg/
  NCCL/Pooling/MiscKernels ODR 去重/ManualNodes/Autocast/init.cpp/Context.h）。
- codegen 三处 bug：binding_default 支持 `[]` 默认值、AutogradNodes 重复
  `variable_list grads;`、Autocast 跳过 out 变体与重载变体、gen_tpx 非 Tensor 返回。
- derivatives.yaml 五处公式修正（embedding/index_add/cumsum/pixel_shuffle 族/gelu）
  全部改为 tpx::ops:: 显式调用或 torch 同款复合式。
