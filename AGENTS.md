# TensorPlay Agent 工作规则

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

## 其他注意事项

- 本仓库常有多方未提交改动并存。遇到与己无关的文件编译失败（缩进、缺 include、
  命名空间错位），做最小修复即可，不要回滚别人的工作。
- Python 层（tensorplay/ 下手写文件）可能领先于已编译的 `_C` 扩展；
  ImportError/AttributeError 先怀疑版本不一致，而不是代码错误。
