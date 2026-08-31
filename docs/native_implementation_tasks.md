# TensorPlay 原生实现与包结构任务清单

更新时间：2026-08-31

这份清单是当前任务的持续记录。每次继续工作前先读取本文件；完成一项时同时补上实现位置、验证命令和结果，避免只创建目录或只补入口就误判完成。

状态标记：

- `[x]` 已实现并完成对应验证
- `[~]` 已有代码，但实现边界、原生接线或验证尚未完成
- `[ ]` 尚未完成
- `[!]` 当前阻塞项

## 一、验收红线

- [ ] 所有公开和内部命名空间按参考源码树拆分；参考源码为子包的地方必须使用子目录，不能用同名单文件长期承载。
- [ ] 每个目录中的模块都必须有真实实现；禁止用转发文件、空入口、占位异常或缩减版接口掩盖缺口。
- [ ] Python 层与 C++ 层必须共同完成：声明、生成代码、调度注册、CPU 实现、CUDA 实现、自动求导和绑定不能只完成其中一层。
- [ ] 旧路径不保留；迁移后的公共名字使用 TensorPlay 自身命名，不带外部框架前缀。
- [ ] 不使用 `skip_implementation` 规避重复定义或未实现问题。生成代码与手写代码冲突时，迁移所有权并补齐真实内核、注册和测试。
- [ ] `_stax` 是 TensorPlay 的编译、追踪和脚本化执行入口。媒体与视觉代码中的脚本化判断必须恢复为 `_stax` 语义，不能为了消除依赖而删除行为分支。
- [ ] 过时接口必须删除实现、导出、绑定、文档和测试，不以保留警告代替删除。
- [ ] 不引入不属于 TensorPlay 范围的编译后端命名空间；范围内的前端、图变换、函数变换、导出和分布式扩展必须逐步原生完成。
- [ ] 本机编译和测试全部通过后，才允许启动远端构建；所有正式文件提交前清理临时文件、缓存和生成副产物。

## 二、当前事实状态

- `[~]` `tensorplay/fft/`、`tensorplay/sparse/`、`tensorplay/special/`、`tensorplay/export/`、`tensorplay/linalg/` 已采用目录形式，但仍需逐文件确认实现是否完整、是否存在转发或缩减逻辑。
- `[~]` `tensorplay/graph/` 已包含 graph、node、proxy、graph module、tracer、symbolic trace、passes 和 experimental 等目录；仍需继续拆解真实实现并清除对旧实现中心的依赖。
- `[~]` `tensorplay/compiler/` 已有前端入口和配置；需要把编译前端、图捕获、后端注册、缓存、调度和错误语义逐项验证。
- `[~]` `tensorplay/distributed/` 已有 c10d、DDP、checkpoint、device mesh、tensor、elastic 等扩展目录；仍有 DTensor/FSDP/pipelining、分片 planner、弹性多机和若干工具链缺口。
- `[~]` `_stax` Python/C++ 入口已存在；媒体、视觉、模块和生成代码中的脚本化/追踪语义仍需全树审计。
- `[~]` codegen 已接入本地 schema 引擎的完整函数记录、后端索引、变体、别名、结构化字段、自动生成声明和源码位置；TP 自有 ABI 投影仍保留。完整生成校验已覆盖所有 2792 条配置记录，且不允许解析器丢记录。一次本地 `ninja -C build p10 _C -j2` 构建和相关 CPU 回归已通过；CUDA 代码已同步但本机构建配置关闭 CUDA。
- `[~]` 仍需按目录和算子族继续扩展；已知未完成项不得以生成成功、目录存在或单个 smoke test 作为完成标准。

## 三、包结构迁移

### 3.1 顶层域包

- `[~]` `fft/`：保留完整变换、频率、辅助函数和内部工具的子模块边界；核对默认维度、归一化、复数共轭、空维度和异常语义。
- `[~]` `sparse/`：拆分构造、检查、梯度、半结构化和内部算子；补齐公开入口、布局检查、稀疏梯度和 CPU/CUDA 原生实现。
- `[~]` `special/`：拆分误差函数、gamma、Bessel、多项式和公共数学工具；逐个核对 dtype、极值、NaN/Inf、复数和梯度行为。
- `[~]` `export/`：保留导出程序、图签名、动态形状、序列化、unflatten、passes、归档和实验入口；补齐状态字典、别名、约束和错误消息。
- `[~]` `linalg/`：补齐公共矩阵分解、求解、范数和矩阵函数；每个入口都要落到实际 CPU/CUDA 内核或明确的原生复合算子链。
- `[ ]` 核对顶层 `__init__`、各域 `__init__`、`__all__`、类型存根和文档索引，确保目录迁移后没有隐式旧文件优先级或循环导入。

### 3.2 graph 域

- `[~]` `graph/graph.py`、`node.py`、`proxy.py`、`graph_module.py`、`tracer.py`、`symbolic_trace.py`：逐文件建立真实依赖关系，禁止再以旧中心模块作为实现来源。
- `[~]` 补齐 `_compatibility.py`、`_graph_pickler.py`、`_lazy_graph_module.py`、`_pytree.py`、`_utils.py`、`annotate.py`、`config.py`、`immutable_collections.py`、`interpreter.py`、`operator_schemas.py`、`subgraph_rewriter.py`、`tensor_type.py`、`traceback.py` 的完整行为。
- `[~]` `passes/`：逐个完成 canonicalize、decompose、shape propagation、fake tensor、constant folding、dead code、split、partition、rewrite、runtime assert、reinplace、operator support、graph drawer 和公共工具。
- `[~]` `experimental/`：逐模块确认符号形状、元追踪、渐进类型、重写器、统一匹配、分区器和调试工具不是空壳或转发层。
- `[ ]` 统一图对象的用户可见名字；删除残留的外部前缀命名，图绘制器等类别使用 TensorPlay 自身名称。
- `[ ]` 为 graph 建立独立测试矩阵：节点拓扑、代码生成、模块执行、参数/缓冲区、控制流、pickle、重写、子图切分、形状传播和错误栈。

### 3.3 编译前端、函数变换和导出

- `[~]` `compiler/`：核对编译入口、配置、注解、后端注册、缓存、重置、错误传播和编译后对象的生命周期。
- `[~]` `_stax/`：核对 eager、AOT、代码缓存、向量 ISA、调度、守卫、运行时、C++ 执行器、图融合和自定义算子路径。
- `[~]` `_transforms/`：补齐 functional call、vmap、批归一化替换、eager transform 和公共工具的嵌套变换、随机性、别名及反向语义。
- `[~]` `export/`：让导出、验证、序列化和恢复都使用 graph 的真实实现；不能通过降低输入约束或跳过节点来“通过”导出。
- `[ ]` 建立编译前端到 `_stax` C++ 执行器的端到端测试，并覆盖 CPU、CUDA、动态形状、别名、autograd 和异常回溯。

### 3.4 分布式扩展

- `[~]` c10d、store、rendezvous、集合通信、P2P、Future、DDP 和 comm hooks：先完成本机编译，再做单进程和多进程行为验证。
- `[~]` checkpoint：补齐分片 planner、并行读写、状态字典重建、异步保存、恢复失败回滚和多 rank 一致性。
- `[~]` tensor：补齐 DeviceMesh、placement、分片传播、重分布、随机算子、规则注册、调试可视化和算子覆盖检查。
- `[ ]` FSDP、pipelining、弹性多机 agent、功能集合通信 coalesced 族和对称内存原生路径。
- `[ ]` 完成 RPC、远程模块、组合式并行入口的真实初始化、关闭、异常传播和资源回收测试。

## 四、原生 C++ 实现主线

### 4.1 生成代码与 Tensor 所有权

- `[x]` codegen 已补充字符串默认值解析、可选参数默认值和 out 参数默认值抑制逻辑，并通过本地编译验证。
- `[x]` `dim`、`numel`、`is_contiguous`、`retains_grad`、`detach`、`as_strided`、`select` 以及稀疏属性和访问器已完成生成/手写所有权拆分，并通过本地编译。
- `[x]` 生成成员包装器、原生自由函数、dispatcher 注册和底层 TensorImpl 访问的当前冲突点已移除；未使用 skip 标记掩盖本轮实现。
- `[ ]` 为 metadata、view、sparse、autograd 属性和工厂类方法补齐原生 kernel 声明、CPU/CUDA/Composite 注册和 Python 绑定。
- `[~]` `SymBool`、`SymInt`、`SymFloat` 已有独立 C++ 表示、表达式节点、生成返回签名、绑定转换、元数据算子注册和 CPU/composite 执行路径；仍需继续补齐输入 ABI、完整运算族与图捕获语义。
- `[ ]` 检查所有 out/inplace/view/multi-output 算子的别名标注、版本计数、视图元数据和反向节点连接。

### 4.4 本轮原生接线

- `[x]` redispatch helper 改为完整 schema 名的确定性标识：基础名和 overload 用单个下划线组合；不再生成哈希、`_overload_` 或 `_signature_` 后缀，并加入重复 operator schema 校验。
- `[x]` `full` 的隐式 dtype 在 CPU/CUDA 工厂源中按 fill value 分类推断：浮点跟随默认浮点类型，整数使用 Int64，布尔使用 Bool，复数跟随默认浮点精度；显式 dtype 不受影响。
- `[x]` `to.dtype_layout`、`to.device`、`to.dtype`、`to.other` 与 `_to_copy` 已共用原生 copy/options/layout/stride 链，CPU 注册不再保留旧的独立 `to` kernel。
- `[x]` Bernoulli 概率张量、标量概率、out 和 inplace 入口已完成原生 CPU/CUDA 注册；Python 入口保留省略概率与显式概率的 schema 区分。
- `[x]` `normal_functional`、`normal_` 的 Generator 传递已接通；`randn.generator` 已完成 CPU 原生 Generator 填充和 CUDA 源同步注册。
- `[x]` codegen 使用完整函数解析、后端索引和生成变体校验；重复签名只在 TP 配置中保留独立命名记录，不会覆盖或静默删除条目。新增 `test_codegen_model.py` 覆盖函数记录、后端元数据、别名列表和固定长度列表。
- `[x]` 可微视图保存根基底、版本号、尺寸/步长/存储偏移和可重放调用；视图原地修改通过 `CopySlices` 重建根基底历史，多级视图合成重放链；普通同类型视图使用几何路径，元数据变化视图保存函数路径。
- `[x]` 编译前端、原生后端、图解释器、导出绑定和三套原生 lowering 均按占位符的 schema 目标名取样例值，避免合法化名称与函数签名不一致时丢失输入。
- `[~]` 符号标量核心已落在 `p10/include/Sym*.h`、`p10/src/Sym*.cpp`、`p10/src/backend/cpu/SymScalarKernels.cpp` 和 Python 绑定；`sym_size`、`sym_numel`、`sym_stride`、`sym_storage_offset`、`sym_is_contiguous` 已返回原生符号类型，并通过 `test/test_sym_scalar_native.py`。
- `[~]` 其余工厂 Generator 入口、CUDA 实机编译和全树算子覆盖继续按失败证据逐项补齐。

### 4.2 算子和自动求导

- `[ ]` Tensor 方法族：index/scatter/gather、累积规约、bitwise/logical、linalg、复数、new 工厂、resize、repeat/flip/tile、统计和唯一化。
- `[ ]` fft、sparse、special、linalg 的核心数学内核；CPU 使用正确的 dtype/累加精度，CUDA 使用设备内核，不能以主机往返替代。
- `[ ]` RNN/LSTM/GRU 正向 CUDA 内核完成后，补齐 CPU/CUDA 反向、双向、多层、batch-first、packed 输入和 dropout 状态。
- `[ ]` 补齐 derivatives 配置中的缺失条目，逐项验证一阶、二阶、复数共轭和 view/inplace 版本计数。
- `[ ]` 复核 reduction、pooling、convolution、linear algebra、随机数和特殊函数的数值稳定性与向量化路径。
- `[ ]` 删除已确认过时的函数及其所有入口、绑定、声明、测试和文档；未完成迁移的旧函数不能以 warning 形式残留。

### 4.3 C++ 构建和绑定

- `[ ]` 逐个检查 p10、tpx、stax、Python binding 的头文件依赖、符号可见性、ODR、静态注册初始化顺序和 CUDA 编译宏。
- `[ ]` CPU 构建通过后再编译 CUDA；验证 `libp10`、`libtpx`、`libstax`、Python 扩展和生成文件的 mtime 都新于改动源文件。
- `[ ]` 本机验证结束前不启动远端构建；本机使用受内存约束的高并行度，远端构建只在本机绿灯后执行。
- `[ ]` 每次构建前检查并清理同目录重复构建和孤儿编译进程；构建失败后先确认没有残留进程，再进行下一轮。

## 五、`_stax` 语义恢复清单

- `[ ]` 全树搜索已经删除的脚本化、追踪和编译判断，逐处恢复 `_stax` 等价语义。
- `[ ]` 检查 audio、vision、nn/modules、functional、export、graph 和生成代码中的条件分支，保留 eager 与编译路径的行为差异。
- `[ ]` 检查装饰器、属性排除列表、trace 入口、脚本化常量、动态形状守卫和错误类型，删除真正过时项但保留仍由 `_stax` 使用的能力。
- `[ ]` 新增回归测试：eager、trace、compile、嵌套模块、媒体预处理和反向执行各一组。

## 六、验证门禁

- `[ ]` Python 静态检查：`compileall`、循环导入、`__all__`、类型存根、包路径唯一性、禁用旧路径扫描。
- `[ ]` graph/export：捕获、执行、重写、序列化、恢复、动态形状和错误回溯。
- `[ ]` autograd：一阶/二阶、view/inplace、复数、随机数、异常和 hook。
- `[ ]` CPU：dtype、布局、非连续张量、空张量、极端形状、NaN/Inf 和性能基线。
- `[ ]` CUDA：多流、事件、设备守卫、混合精度、CUDA graph、内存统计、RNG、通信和 kernel 数值。
- `[ ]` distributed：单 rank smoke、多 rank 集合通信、DDP、checkpoint、device mesh、分片和失败恢复。
- `[ ]` 运行改动文件的注释/描述文本扫描，确保不出现外部来源、品牌、路径和对照性措辞。
- `[ ]` 只保留正式源码、测试、配置和文档；删除临时脚本、缓存、日志和未登记生成物。

## 七、下一轮固定顺序

1. `[ ]` 读取本清单并确认构建进程处于静默状态。
2. `[ ]` 先处理生成 Tensor 方法与手写方法的所有权冲突，不添加任何 skip 标记。
3. `[ ]` 补齐 metadata、view、sparse、autograd 属性的 C++ 原生实现与注册。
4. `[~]` 继续补齐符号标量输入、完整运算族和绑定语义，再做本机 `-j8` 增量构建。
5. `[ ]` 逐个修复编译/链接错误；每轮确认产物新鲜度。
6. `[ ]` 运行最小 Python、graph、autograd、CPU、CUDA、distributed 回归集。
7. `[ ]` 根据失败证据继续拆分下一个参考模块；不得为了绿灯裁剪接口、跳过实现或加入降级路径。
8. `[ ]` 本机全绿后，才登记远端高并行构建及其结果。

## 八、完成记录

| 日期 | 范围 | 实现文件/测试 | 结果 | 后续 |
|---|---|---|---|---|
| 2026-08-30 | codegen 参数解析与 out 签名 | `tools/codegen/` | 生成成功，本地构建未通过 | 解决 Tensor 方法所有权和原生注册 |
| 2026-08-31 | 生成命名、工厂 dtype、copy 链、随机 Generator | `tools/codegen/model.py`、`p10/src/backend/cpu/FactoryKernels.cpp`、`p10/src/backend/cuda/RandomKernels.cu`、`p10/src/RegisterComposites.cpp` | 本地 `ninja -C build p10 _C -j2` 通过；相关测试 `129 passed, 20 skipped` | 继续补齐剩余 Generator 入口、CUDA 实机和目录内原生模块 |
| 2026-08-31 | codegen 完整函数/后端元数据接入 | `tools/codegen/model.py`、`tools/codegen/main.py`、`test/test_codegen_model.py` | 2792 条记录完成严格解析与保留；测试 `131 passed, 20 skipped` | 继续将结构化、视图和就地生成逻辑接入真实 C++ 所有权 |
| 2026-08-31 | schema 标签依赖与视图输入元数据 | `tools/codegen/model.py`、`CMakeLists.txt`、`test/test_codegen_model.py` | 视图输入回填覆盖可能返回视图的算子；标签注册表纳入增量依赖；本地 `ninja -C build p10 _C -j2` 通过；测试 `131 passed, 20 skipped` | 继续实现结构化 kernel emitter 与完整 view replay 所有权 |
| 2026-08-31 | 视图历史重基与签名占位符绑定 | `tpx/include/AutogradMeta.h`、`tpx/include/Autograd.h`、`tpx/include/ManualNodes.h`、`tpx/src/Autograd.cpp`、`tools/codegen/gen_tpx.py`、`tensorplay/_stax/`、`tensorplay/graph/graph_module.py`、`tensorplay/export/_trace.py`、`test/test_view_rebase.py` | 本地 `ninja -C build p10 _C -j2` 119/119；视图/自动求导/编译/codegen 回归 `39 passed, 1 skipped`；产物时间戳晚于改动源文件 | 继续处理符号标量、结构化内核和剩余算子族的完整 C++ 接线 |
| 2026-08-31 | 原生符号标量核心与 metadata 返回 | `p10/include/Sym*.h`、`p10/src/Sym*.cpp`、`p10/src/backend/cpu/SymScalarKernels.cpp`、`src/bindings/python/SymInt.cpp`、`tools/codegen/`、`test/test_sym_scalar_native.py` | 本地 `ninja -C build _C -j2` 通过；符号标量回归 `4 passed`；修复空输入折叠越界和自同一比较恒等化 | 继续补齐符号输入 ABI、完整魔术方法、图捕获和其余原生算子接线 |
