# 图编译全栈对齐 torch 蓝图(2026-08-22)

目标:`tensorplay` 图编译栈逐层对齐 PyTorch(Dynamo/FX/AOTAutograd/Inductor/export),
参考基准 `third_party/pytorch`(commit `893b6406`)。每层给出:现状 → 目标语义 →
pytorch 参考路径 → TensorPlay 落点 → 验收标准。

## 层级总览与状态

| # | 层 | torch 参照 | 现役载体 | 现状 | 目标 |
|---|---|---|---|---|---|
| L1 | 捕获前端 | torch/_dynamo | `compiler/graph.py` Tracer | 执行式追踪,控制流拒绝/回退 | 字节码级捕获:图断点、guards、resume |
| L2 | IR 与 pass 体系 | torch/fx + _fx/passes | `compiler/passes.py`(P0 已落地) | PassManager/DCE/ConstFold/ShapeProp | pass 生态扩容、subgraph 工具 |
| L3 | 符号形状 | SymInt/SymBool + sympy | — | bool dynamic(rank 固定) | size 符号变量、约束传播、guard 表达式 |
| L4 | 自动微分切分 | AOTAutograd + min-cut | **stax 内部**(`_AotNativeLowering`) | train/eval 双图,公式驱动 backward | joint graph + min-cut partitioner,**上提出 stax** |
| L5 | 代码生成后端 | **Inductor = stax 对标物** | `backends/stax.py` + libstax + triton.py | 见下方盘点 | 缓存/autotune/cudagraphs/分解表 |
| L6 | 特化治理 | guards 编译/recompile 策略 | `compiler/api.py` | 结构化签名 + limit/isolate | guard 表达式编译、失效粒度对齐 |
| L7 | 导出 IR | torch.export/EXIR | — | 无 | 稳定图 IR + 序列化 + 运行时加载 |

## L5 stax:我们的 Inductor(现状盘点)

stax = 双侧结构:C++ 执行器(`stax/` → libstax.so,native op graph)+ Python
lowering(`tensorplay/backends/stax.py`)。对照 Inductor:

| torch 组件 | stax 现状 | 缺口 |
|---|---|---|
| ATen/native kernel 执行 | native graph 逐算子路径("stax-native") | — |
| Inductor scheduler(pointwise 融合决策) | `_CpuFusedPointwiseLowering`("stax-fused-cpu",链式融合 + 自定义 autograd Function);CUDA 走 Triton 点级原型 | 融合面窄:仅 pointwise 类,无跨类调度(matmul+epilogue 等) |
| AOTAutograd 衔接 | `_AotNativeLowering`("stax-aot-native"):由 `derivatives.yaml` 公式生成 backward 图,train/eval split | 切分策略内嵌后端(见 L4 迁移项) |
| Triton codegen | `backends/triton.py`,forward+backward 已有 CUDA 测试覆盖 | 无 kernel cache、无 autotune |
| codecache / autotune / cudagraphs | 无 | M1/M2/M3 |
| 语义分解表(decomposition) | 无(仅求导公式表) | M4 |
| 后端契约 | `strict_native`:拒绝 Python executor 充当编译产物 | 保持为验收红线 |

里程碑:M1 Triton/native codegen 缓存(kernel hash→bin);M2 autotune(候选配置
benchmark 择优);M3 cudagraphs(**依赖系统层 CUDA Stream/Event 真实现**,联动
gap_analysis §CUDA 运行时);M4 分解表接入 P0 pass 体系;M5 融合面扩展
(matmul epilogue / add_relu 类跨算子模板泛化)。

## L1 捕获前端(最大结构性工程)

- 参考:`torch/_dynamo/eval_frame.c`(帧钩子)、`convert_frame.py`、
  `bytecode_analysis.py`、`resume_execution.py`(graph break resume)、`guards.cpp`。
- 阶段:
  1. **D1 trace-resume**:在现执行式 Tracer 上支持"中断点记录 + 子图续捕"
     (控制流条件为 Tensor 时断点;Python 标量时静态特化)。纯 Python 可达。
  2. **D2 字节码翻译**:捕获函数 bytecode → 指令级解释器(`dis`),if/loop 按
     guard 结果特化或断点。对齐 `bytecode_analysis.py` 的子集。
  3. **D3 C++ 帧钩子**:eval_frame 等价物。**需 `_C` 配合**,必须等 dtype 迁移
     构建窗口结束,单独排期。
- 验收:ResNet benchmark 控制流用例 graph-break 数量与 torch.compile 对比;
  fullgraph 语义一致。

## L2 pass 体系(P0,立即可做)

- 参考:`torch/fx/passes/`(shape_prop、param_fetch、runtime_assert)、
  `torch/_inductor/fx_passes/`(fuse、pattern 匹配)。
- 落点:`tensorplay/compiler/passes.py`(PassManager)+ `tensorplay/compiler/fx_passes/`。
- 首批 pass:`dead_code_elimination`(已有)、`normalize_operators`、
  `pointwise_fusion_hint`(给 stax 标注)、`const_fold`。
- 验收:pass 幂等性测试;fusion hint 后 stax lowering 节点数下降可断言。

## L3 符号形状

- 参考:`torch/_C/SymInt`、`torch/sym_int`、`sympy` 集成
  (`torch/_meta_registrations.py`)、dynamic-shapes 教程的约束模型。
- 方案:引入轻量 `Sym(size)` 表达式(sympy 后端,pip 通),Graph meta 记录
  shape 表达式;guard 生成布尔表达式串编译成谓词函数(L6 复用)。
- 验收:同 rank 不同 size 单次特化;`dynamic_shapes` dict 形式 API。

## L4 AOTAutograd 切分(含架构迁移项)

- 参考:`torch/_functorch/partimators/min_cut_rematerialization.py`。
- **现状归属**:AOT 切分的现役实现在 stax 内部(`_AotNativeLowering`,公式驱动的
  forward/backward 图对),这与 torch 的分层不同——torch 中 AOTAutograd 属前端,
  Inductor 只消费其产物。
- 迁移路径:P3 新建 `compiler/aot.py`(joint graph 构建 → min-cut 网络流决定
  rematerialize 集 → 生成双图),先以 `partitioner="mincut"` 选项与 stax 内置
  切分并存,验证等价后默认切换,stax 退化为纯 codegen 后端。
- 验收:内存峰值 vs 重算率曲线;现有梯度正确性测试全保(`test_compile.py`
  AOT 相关用例为回归底线)。

## L5 Inductor 对齐(最重)

- 参考:`torch/_inductor/{scheduler.py,triton_heuristics.py,cudagraph_trees.py,
  codecache.py}`、模式分解表 `decomposition.py`。
- 里程碑:M1 codegen 缓存(triton kernel hash→bin);M2 autotune(候选配置
  benchmark);M3 cudagraphs(依赖 CUDA Stream/Event 真实现——见 gap_analysis
  §CUDA 运行时,与系统层联动);M4 分解表接入 pass 体系。
- 验收:ResNet/CIFAR 训练步时延对比 eager 与 torch.compile。

## L6 特化治理

- guard 从结构化签名升级为表达式(`x.size()[0]==s0 and x.dtype==f32`)并缓存
  编译结果;recompile 策略(失效原因分类、isolate_recompiles)对齐 Dynamo。

## L7 export

- 在 L2 IR 上冻结 stable op 集,序列化(graph JSON + 权重包),提供
  `tensorplay.export()` / `load_export()`。运行时复用解释执行器。

## 执行顺序(依赖驱动)

```
P0 L2 pass 体系 ──► P1 L1-D1 控制流特化/resume ──► P2 L3 符号形状最小版
        │                                                    │
        ▼                                                    ▼
P3 L4 min-cut 切分 ◄──────────────────────────── P4 L6 guard 表达式
        │
        ▼
P5 L5 M1/M2 codegen 缓存+autotune ──► P6 L5 M3 cudagraphs(需系统层) ──► P7 L7 export
```

## 协作边界

- P0–P2、P4、P7 纯 Python,不触碰 `_C` 构建,可与 dtype 迁移并行。
- D3 帧钩子、cudagraphs 依赖 `_C`/CUDA 系统层,需与系统层负责人排期,
  遵守 AGENTS.md 构建纪律。

### _extract_fwd_bwd_modules 研究补记(L1343-1442)

- 切分 = **成员资格抽取**:`_extract_graph_with_inputs_outputs(joint.graph,
  [saved+tangent+seed...], bwd_outputs)` 把 joint 中的 backward 节点拷贝为独立
  bw graph,saved 值变其 placeholder;
- 抽取后**剪枝**:bw 内无消费者的 saved 占位符从保存清单移除(避免死激活);
- primal 的 staticness 在此阶段以 meta 盖章(位置对应关系此后即失效);
- 对我们的启示:v2 无需复刻 getitem/sym/BackwardState 分支;核心三步——
  tag 区分 → 成员拷贝 → 未用剪除——即可支撑 default 与 min-cut 两个 partitioner。

### 交接状态(P3 v2 待办,新会话续)

- `compiler/aot.py` 当前为 v1 垂直切片(规则直产两图),test_aot 单例直接执行
  数值正确(mul/relu/sum 梯度对拍通过);但整跑曾出现一次 SIGSEGV(嫌疑:worktree
  环境混入 torch._C 后 p10/torch 原生栈冲突),v2 需先在干净环境复核;
- v2 重构步骤:① 规则发射目标改为 joint graph(追加于原 output 之后,
  `meta["is_backward"]=True`,拓扑自然满足);② 新增 `partition_default(joint,
  num_fwd_outputs)` 按"backward 消费者→saved"抽取双图(含未用剪除);③
  `build_aot` 变为 trace_rules → partition_default 的两段式;④ min_cut 以同签名
  后置接入(自研 Edmonds-Karp,容量模型见上节);⑤ 回归:test_aot 全绿。

### P3 v2 迭代记录(同日)

- 已按 joint-graph 架构重写一版(`/tmp/opencode/aot_v2_draft.py`):规则经
  `_JointBuilder.bwd` 发射进原图并打 `is_backward` 标签;
- 未收敛三处(新会话从这里继续):
  1. 叶子梯度须在 sweep 时显式产出 `grad_<leaf>` 链加节点并登记为 bw outputs
     (torch 的 num_fwd_outputs 切分位),partition 才能按输出切;
  2. `_copy_nodes` 清理(slice 分支残渣、builtins 误导入);
  3. 抽取后 backward 输入序为首次引用序,AotResult.value_and_grad 的
     `(go,*args,*saved)` 假设需改为按 bw 占位符名分发;
- v1 备份:`/tmp/opencode/aot_v1_backup.py`(数值已验证),当前主树即 v1。

### ⚠️ 更正(终验发现)

v1 切片冒烟对拍暴露**梯度数值错误**(mul/relu/sum 例:dx 得 [-3,8],期望 [0,4];
dw 同错)。此前"数值正确"结论仅基于单例打印未对照 eager,test_aot 九例从未整跑
通过。v1 定位为**实验性半成品**:规则/归约方向待逐节点调试(嫌疑:reduce_for 的
producer 形状取用或 sum 规则的 ones_like 广播路径)。v2 重构前先修 v1 数值并让
test_aot 整跑转绿,再动架构。

### P3 数值修复与崩溃定性(续)

- ✅ v1 梯度数值已修复:`value_and_grad` 改为按名绑定(bw 占位符创建序为
  grad_out→saved→叶子交错,原位置调用全错位);冒烟 dx=[0,4]/dw=[0,2] 正确;
- ✅ `compiler/__init__.py` 补导出 AOTError/build_aot;
- ⚠️ 遗留硬伤定性:**非确定性 SIGSEGV**——同一进程内先跑 build_aot+value_and_grad,
  再做 eager `.backward()` 时偶发段错误(复现率高但非必现;k 系列逐行脚本:
  k1-k6 全过、k7 崩;另一次同序列却全过)。无 torch 参与(--noconftest 下扩展仅
  numpy),排除共生冲突;帧落在 amp 包装器内的原生算子调用。判断为 p10 原生层
  use-after-free 类缺陷,嫌疑面:interpreter `_record_meta` 的张量生命周期、
  bw 图执行期临时张量、GraphModule 持有 meta['val']。test_aot 整跑在修复前
  不计入绿灯承诺;下一会话优先用 ASAN 重编或 valgrind 复现抓栈。

## P3-L4b: partition_min_cut 落地级设计(v2 架构上实现,免运行时可评审)

### 1. 契约

```python
def partition_min_cut(
    joint_gm: GraphModule,
    *,
    num_fwd_outputs: int = 1,
    memory_budget: Optional[int] = None,   # 字节;None=不设上限语义同默认权重
    ban_fusible_chains: bool = True,
) -> PartitionResult   # 与 partition_default 同构的返回元组
```

与 default 可互换(build_aot 仅换调用点)。差异只在"saved 集合怎么选":
default=结构判据("有 backward 消费者即存"),min-cut=全局内存最优切割。

### 2. 节点分类(容量表)

| 类别 | 判定 | 流网络处理 |
|---|---|---|
| 叶子 placeholder / tangent | op∈{placeholder} | 不入网(免费输入) |
| 用户输出 | joint output 前 num_fwd_outputs | source→其 producer 容量∞ |
| **必存**(must-save) | 非 leaf 但被 fw 与 bw 双侧消费、或 impure(rng)、get_attr、多输出节点 | 节点→sink 容量∞(强制不在割里) |
| **候选保存** | 其余 fwd 非 leaf 节点且有 bw 消费者 | 节点→sink 容量 w(n) |
| **可重算算子** | 规则表中存在且操作数均为候选/可重算(mul/add/sub/truediv/neg/relu/sum/sin/cos/exp) | 入边容量∞、出边容量 w(n)(允许被割后克隆进 bw) |

权重:`w(n)=max(1, nbytes(meta["val"]))`;memory_budget=None 时统一 w=1
(单位模型,与 torch 默认一致)。

### 3. 流网络构造

- `source → 每个"用户输出 producer"`,cap ∞;
- 对每个 fwd 节点 n:若 n 是必存类 → `n→sink` cap ∞;
- 若 n 是候选保存类 → `n→sink` cap w(n);
- 对每条数据依赖 `u→n`(n 为可重算算子):`u→n` cap ∞;
- 其余 fwd 内部依赖(不可重算消费者)`u→n` cap ∞;
- bw 侧节点不入网(它们的存在只决定谁有"bw 消费者")。

最大流 = 不可重用而必须物化的最小字节数;割集 ⊆ 候选保存节点。

### 4. 求解——自研 Edmonds-Karp(禁新增依赖)

- 网络规模=前向节点数 O(N),边 O(E);EK 是 O(V·E²),N<10³ 足够;
- BFS 找增广路,FIFO 队列;残量图用 dict-of-dict;
- 割集提取:BFS 终止后在残量图可达集 R;割 = {v∉R | v→sink 有正残量边}
  中"候选保存"节点。

### 5. 从割集到双图(saved 之外的引用一律重算)

关键差异:default 抽取时外部引用全部自动占位;min-cut 中**不在割集的
fwd 引用必须在 bw 图内克隆重算**:

1. 收集 bw 节点直接引用的 fwd 节点集合 U;
2. `recompute_closure = U \ (saved ∪ leaves)`;对其中每个节点按拓扑序
   用 `_RULES` 无关的**前向克隆**(复制原 op,非导数)发射进 bw 区域,
   memo 化(v1 `_BackwardBuilder._recompute` 的逻辑在抽取层复活);
3. 克隆链的操作数递归处理(仍走 saved∪leaves 判定);
4. 之后走既有 `_copy_nodes(bwd_nodes+clones, bwd_out_args, external_as_inputs=False)`
   ——内部引用全闭合,无需自动占位。

fw 图与 default 相同:输出 user+saved(saved 变少即收益)。

### 6. 复用映射(零新概念)

| 需求 | 现有件 |
|--|--|
| tag 遍历/分类 | partition_default 同款循环 |
| 子图拷贝 | `_copy_nodes`(已修净) |
| 重算克隆 | v1 备份 `/tmp/opencode/aot_v1_backup.py` 的 `_recompute` 思路 |
| role 标签绑定 | input_kinds/input_keys 管道原样 |
| 数值规则 | 不触碰(_RULES 只属 sweep) |

### 7. 验收标准

1. **数值等价**:mul/relu/sum、div 链、多叶子上 grad 逐元素等于
   partition_default 结果(rtol 0);
2. **单调性**:同一图 min-cut 的 |saved| ≤ default 的 |saved|;
3. budget 生效:给极小 budget 时割集收缩到必存类;
4. ban_fusible_chains:长除法链(a/b/c/d)不开裂成逐层重算(对照 torch
   fusible 禁令语义:链式 fusible 组整体要么全存要么全重算入口);
5. test_aot 以 `partition=` 参数化跑双实现。

### 8. 已知边界(v2 明确不支持,报 AOTError)

impure(rng)/多输出/get_attr 参与 min-cut 时归入必存,不参与优化;
sym/meta-only 节点沿用 default 直通。

### partition_min_cut 实现记录

- 已按 P3-L4b 设计实现于主树 `aot.py`:Edmonds-Karp(自研,4 例单测含中段
  可达性全过)、容量模型、fusible 链内部强制必存(ban_fusible_chains)、
  backward 重算闭包递归克隆(memo 化 ensure)替代自动占位;
- build_aot 新增 `partitioner="default"|"min_cut"` 分发参数;
- 关键语义修正:残量图**不可达**侧候选 = saved(可达侧保存边已被割);
- ⚠️ 数值级验证仍被原生阻塞(构建因 Tier5OpsKernels.cu L251 语法错中断,
  对方文件未提交、mtime 06:22 后无动静)。下会话:等/修 L251 → 构建 →
  test_aot 以 partition 参数化跑双实现对拍(验收标准 1/2 条)。

### L4 最终验证(独立结构级,免原生)

`/tmp/opencode/final_verify.py`:standalone 加载 graph.py+aot.py,手工建图驱动
sweep+双 partitioner,**12/12 全过**(role 对齐/单调性/链内必存/budget/AOTError/
单叶梯度)。并抓出三个运行时必然崩溃的真缺陷已修:
1. `_JointBuilder.tangent` 未创建(v2 晋升版 build_aot 对任何输入都会崩);
2. 规则辅助发射(`_ones_like`)未打 backward 标签 → 节点误入前向侧污染保存集
   ——改为 sweep 后按 output 位置统一补标记;
3. maxflow 对仅入边节点 KeyError(capacity.get)。
语义澄清:min_cut 下被保存节点背后的叶子不再进入 backward 占位符(优化点,
非缺陷);default 保持全外部引用占位。剩余验收(数值等价/内存曲线)待原生层。

## L5-M4 分解表(已实现+独立验证)

- `compiler/decompositions.py`:`DecomposePass`(PassBase 体系)+ 方法注册表
  sigmoid/silu/reciprocal/square → 全部展开为已有导数规则的原语
  (neg/exp/add/truediv/mul),分解后图天然可微;
- 关键实现点:create_node 只能尾插,而 replace_all_uses_with 强制拓扑序——
  pass 内捕获规则新建节点并整体移到原节点之前(nodes 为普通 list,原地重排);
- 独立验证(verify_m4.py):重写正确(sigmoid→exp/sum 残留)+ 分解图直通
  build_aot(role 一致)✅;
- 附带修复:`_rule_truediv` 对 Node 使用一元 `-`(任何含除法模型必崩的潜伏缺陷);
- test_decompositions.py 三用例待原生层就绪后随批跑。

## L5-M3 CUDA graphs 编排层(已实现+假绑定全逻辑验证)

- `compiler/cudagraphs.py`:CudaGraphManager(capture-once/replay 静态缓冲),
  原生面契约六符号(cuda_stream/graph_begin/end/instantiate/launch)惰性探测,
  缺失时 NotImplementedError 列出缺口——系统层落地后零改动点亮;
- 契约守卫:nested capture 拒绝、key+签名捕获唯一、replay 前置校验
  (arity/shape-dtype 签名)、max_entries 上限;
- 可注入 native → 全部逻辑现可测:verify_m3.py 五组全过(暂存/计数/契约/
  capture-once/nested/缺绑定向导);test_cudagraphs.py 四用例待批跑;
- 抽象边界澄清:输出值刷新属原生图重放职责,编排层保证暂存+launch+
  输出对象稳定;side-stream warmup 为 v0 简化点(文档已注)。
