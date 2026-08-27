"""Show why non-winning reduction candidates are disqualified."""
import os
import sys
import traceback

os.environ.setdefault("TP_CACHE_DIR", "/tmp/tpcache_prof2")
sys.path.insert(0, "/tmp/TensorPlay")

import torch  # noqa: E402
import tensorplay as tp  # noqa: E402
import tensorplay.compiler.codegen.triton as T  # noqa: E402
from tensorplay.compiler import compile as tp_compile  # noqa: E402

captured = {}
_orig = T._compile_program


def spy(*a, **k):
    captured.setdefault("calls", []).append((a, k))
    return _orig(*a, **k)


T._compile_program = spy

x = tp.rand(4096, 4096, device="cuda")
for fn in (lambda x: ((x * 2.0).sigmoid()).sum(dim=1) * 3.0 + 1.0,
           lambda x: x.sum()):
    captured["calls"] = []
    compiled = tp_compile(fn, backend="stax")
    compiled(x)

T._compile_program = _orig

for a, k in captured["calls"]:
    cfg = k.get("fixed_config")
    if cfg is None:
        continue
    try:
        launch = _orig(*a, **k)
        # try one launch to surface OOR
        launch([a[3][0]])
        tp.cuda.synchronize()
        status = "OK"
    except Exception as exc:  # noqa: BLE001
        status = f"{type(exc).__name__}: {str(exc)[-500:]}"
    print(f"cfg={str(cfg):22s} {status}")
