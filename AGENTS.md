# Agent 工作规则

## 开发红线（最高优先级，严禁违反）
任何代码落地必须先看到第三方目录的pytorch对应，缺少检索的代码修改一律违规。禁止凭借自己的看法修改。
必须严格遵守git提交规则，频繁提交，避免丢失。

## 注释红线

本仓库是独立实现。**严禁在任何源码的注释或描述性文本中出现对上游框架
（PyTorch / ATen / torch 系）的引用、出处声明或"对齐/对照"表述**——这是抄袭/侵权风险，
属于硬性红线，违反即打回。

- **适用范围**：一切 `//`、`/* */`、`#`、Python docstring、以及字符串形式的描述性文本
  （错误消息、警告、`print`/日志输出、生成代码里内嵌的注释模板）。
- **禁止内容**（大小写不敏感）：
  - 品牌词与命名空间：`pytorch`、`torch.`、`torch/`、`aten`、`ATen`、`at::`、
    `torchvision/torchaudio/torchgen/TorchScript` 等；
  - 上游路径与文件名：`aten/src/ATen/...`、`native_functions.yaml`、`torch/csrc/...`、
    `third_party/pytorch/...`（`third_party/` 目录内的 vendored 文件本身除外）；
  - 出处/对照表述：`port of`、`ported from`、`mirrors`、`mirroring`、`aligned with`、
    `parity`、`copied from`、`same as upstream`、`as in <上游文件>:<行号>` 等任何把上游
    实现当出处、基准或对照物的说法。
- **写法要求**：注释只描述本项目的语义、数学公式、复杂度与设计取舍。行为约定写成
  中性的规范描述（例如"除零返回 inf"），不要写成"与某某一致"。
- **唯一豁免**：功能性互操作标识——序列化格式标识、外部环境变量名、第三方权重下载
  URL 等**运行所必需**的代码级字符串。它们只能出现在代码里，严禁出现在注释中；
  新增此类标识须在 PR 描述中说明必要性。
- **提交前自检**（对所有自己改动的文件执行，命中注释/描述性文本必须删改后再提交）：

```bash
grep -rn -i -E 'pytorch|\baten\b|\bat::|mirrors|ported from|parity' <改动的文件>
```

- 本仓库部分历史文件仍由 root 身份的进程创建，需要sudo密码123。改动这些文件时同样
  遵守本红线；发现他人新代码违反红线，顺手清理即可（最小改动，不回滚其功能）。

## 编译纪律（必须遵守）

**编译前必须查进程。** 本仓库常有多个 agent / 开发者同时工作，且共享同一棵源码树。

每次执行 `make` / `ninja` / `cmake --build` 之前：

```bash
ps -eo pid,etime,args | grep -E "ninja|cmake --build|make.*-j|nvcc|cc1plus|cicc" | grep -v grep
```

1. **有活跃构建 → 禁止启动自己的构建。** 等待静默（连续 ~120 秒无编译进程）。
   - 共享同一棵树意味着对方的构建也会编译你的改动；多数情况下等即可。
2. **重复构建（例外：等待无效）。** 若查到 **两个及以上** `cmake --build`/`ninja`
   同时跑在同一个 build 目录（无论是否同一人发起），不要等——并发 ninja 写同一批
   `.o`/`.so` 会互相覆盖产物并互相触发重编，**无限卡死且产出损坏的库**。处理：
   把这些构建的进程树全部停掉（先 `kill -TERM` 顶层的 `sh -c`/`cmake --build`，
   残留的 `nvcc`/`cc1plus` 再 `kill -KILL`），等静默后由一方起 **单个** 低并行度构建。
   **自己被中断/取消的构建同样会留下孤儿 nvcc/ninja**：每次构建命令异常退出后，
   必须重新查进程、确认无残留（必要时清理自己的孤儿）才允许做下一次操作。
3. **多个二进制目录写同一个包目录。** 根目录和 `build/` 都是 CMake 二进制目录，
   `TP_PACKAGE_DIR` 固定指向 `<repo>/tensorplay/`（CMakeLists.txt 的 `TP_PACKAGE_DIR`）。
   并发链接会产出损坏的 `.so`。
4. **构建后必须验证产物新鲜度**：比较 `tensorplay/lib/libp10.so` 与
   `tensorplay/_C/*.so` 的 mtime 是否新于本次改动的源文件。过期说明没编进去。
5. **只编需要的目标**（如 `make p10`、`ninja -C build _C`），避免全量重编抢占 CPU/OOM
   （本机 30GB 内存，`-j` 全开 + nvcc 会 OOM kill，报错 137 即此因）。
6. 大 `-j` 前先 `free -g` 确认可用内存。

## 提交规范（Conventional Commits）

- 提交标题格式：`type(scope): subject`（≤100 字符，结尾不加句号）。
  type ∈ build | chore | ci | docs | feat | fix | perf | refactor | revert | style | test。
- **feat/fix 必须带 scope**；scope ∈ frontend | autograd | compiler | kernels | cuda | build | docs，
  与 `release notes: *` 标签族 1:1 对应（见 .github/labels.yml）。
- 破坏性变更：标题加 `!`（如 `feat(frontend)!: ...`），并在 footer 写 `BREAKING CHANGE: ...`。
- 校验器是 tools/commit_schema.py（commit-msg 钩子、pr-title 工作流共用）；
  本地启用：`pre-commit install --hook-type pre-commit --hook-type commit-msg`。
- PR 标题同样按此规范校验（合并后成为 squash commit 标题）。
- **版本不用 cz bump**：版本号遵循 `X.Y.Z` 语义化规则（version.txt +
  tools/generate_tensorplay_version.py；nightly 为 X.Y.0.dev<UTC日期>[+cuXXX|+cpu]）。
  commitizen 仅用于 `cz changelog` 草稿。
- 合并提交、`Revert ...`、fixup!/squash! 提交不受规范约束。

## 其他注意事项

- 本仓库常有多方未提交改动并存。遇到与己无关的文件编译失败（缩进、缺 include、
  命名空间错位），做最小修复即可，不要回滚别人的工作。
- Python 层（tensorplay/ 下手写文件）可能领先于已编译的 `_C` 扩展；
  ImportError/AttributeError 先怀疑版本不一致，而不是代码错误。
