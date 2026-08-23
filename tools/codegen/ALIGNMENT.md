# Codegen 对齐状态(torchgen)

## 已完成
- schema 引擎 = vendored `third_party/pytorch/torchgen.model`(强制加载,防 site-packages 旧版遮蔽);
  339 native + 159 derivatives 全部通过其 FunctionSchema.parse。
- yaml 方言清零:`migrate_schemas.py --canonicalize --apply`(int/SymInt/(Tensor,...)/ScalarType/True-False/
  Tensor? x=None);非空列表默认值迁至条目级 `python_defaults`。
- types 层 = p10 以 torchgen.api.types CType 代数注册(`StdOptionalCType/StdVectorCType` 适配 std 命名)。
- 编排层 = `main.py::run_gen()` + `@register_generator`,支持 `--targets`;双入口(包/脚本)可用。
- `tensorplaygen.py`:自定义算子生成器(TORCH_LIBRARY_IMPL 风格注册到 p10 Dispatcher + 纯 CPython `METH_FASTCALL|KEYWORDS` 绑定,经 `python_c` 桥接面,无 pybind11;对齐 torch 的 torch_python 布局)。
- Float8_e4m3fn/e5m2:DType 枚举 + `TENSORPLAY_FORALL_FP8_TYPES` 分层宏 + copy/item 接入 + float 桥接运算符。
- conv 系 bias / nll_loss_backward.total_weight 的 optional→`const Tensor&` 解包边界(`UNWRAP_OPT_TENSOR`)。

## TensorIterator 补齐(进行中)
- 现状:TIBase(ATen 同款 reorder/coalesce/fast-setup/for_each)已存在,reduce_op 有工厂;
  新增 `TensorIterator::binary_op` 工厂(TensorIterator.h/.cpp)。
- 已迁移:`ArithmeticKernels.cpp::binary_op_kernel_impl` fallback 由串行递归
  (apply_op_recursive)改为 TI for_each(字节 stride 双路径:连续快速内层+通用跨步),
  获得维度重排+合并+并行化;g++ -fsyntax-only 通过。
- 迁移完成(cpu 全量):Arithmetic(binary impl/add_out/add_-mul_-div_ inplace,
  inplace 经 TI 的 overlap 检查显式放行 out==self)、TierOps、Comparison(含
  maximum/minimum)、Pointwise(pow);共享 helper 头 p10/include/TensorIteratorOps.h
  (ti_apply_binary / ti_apply_compare);apply_op_recursive 调用点清零;
  全部 -fsyntax-only 通过。
- CUDA 审计结论(不迁移,记录理由):cuda/ArithmeticKernels 已是定制 GPU 迭代器
  ——TensorDesc 偏移机制 + try_binary_vectorized 快速路径 + bf16/half 全覆盖,
  与 ATen 的 gpu_kernel(CUDA 专用 TI)同构。强行复用 CPU TI 是倒退。
  遗留小项:switch 的 default 分支报错信息统一、fp8 档位补入 broadcast 内核;
  CUDA 二进制内核仍手写 grid-stride。

## 未完成(按序)
1. [骨架完成] InplaceOrView 真实现已入 gen_tpx.py:inplace 算子 bump_version、
   view 算子 share_version_counter(41 处生成点);gen_inplace_or_view.py 为声明面。
   余:TraceType 切片、view-replay 链式重放(依赖 saved-view 元数据)。
2. [代码生成侧完成] gen_structured.py:基于 torchgen structured_delegate 字段做
   生成期一致性校验 + 'Structured' 目标占位;运行时侧待 p10 增加 Device::Meta。
3. [切片1完成] gen_python_c.py:'PythonCAPI' 目标生成 METH_FASTCALL|KEYWORDS
   入口(295 个算子;奇异类型跳过计数),经 CPythonBridge.h(src/bindings/python/)
   解包/打包,绕过 pybind11 分发。CPythonBridge.cpp 已实现并通过
   g++ -std=c++20 -fsyntax-only(p10+pybind includes);keep_alive 为有据 no-op
   (view 共享 Storage/VariableVersion)。
   集成三步(待构建解锁):① CMakeLists _C 源加 src/bindings/python/CPythonBridge.cpp;
   ② init.cpp 调 python_c::register_generated_cpython(m.ptr());
   ③ 生成头 include 路径进 _C target_include_dirs。

## 【最后一步】验证
等构建静默 → 单次 `ninja -C build _C -j4` → 校验 `tensorplay/lib/libp10.so` 与
`tensorplay/_C/*.so` mtime 新于改动源码 → 测试子集:test_ops / test_grad /
test_op_parity / test_dtype_alignment。

## 注意
- 多方并发编辑同一棵树:每次 make/ninja 前必须查进程;出现两个以上同目录构建按 AGENTS.md 清树。
- 近期他人新增文件遗留的编译错误已最小修复:TierOps(cpu/cu remainder-fmod/heaviside/gcd/lcm)、
  ForeachMultiTensor.cuh(.template)、IndexingKernels(Utils.h/.stream()/fp8 atomic CAS)、
  SpectralKernels(<functional>/is_complex)、OptimizerKernels 占位 SpectralKernels.cu、
  TierOpsReduceKernels(tuple get/删除重复 kernel 模板)。
