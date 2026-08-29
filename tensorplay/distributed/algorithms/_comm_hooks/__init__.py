# deprecated re-export shim pointing at ddp_comm_hooks.
import warnings

from tensorplay.distributed.algorithms.ddp_comm_hooks.default_hooks import (
    allreduce_hook,
    bf16_compress_hook,
    fp16_compress_hook,
)


def __getattr__(name):
    warnings.warn(
        "tensorplay.distributed.algorithms._comm_hooks is deprecated; "
        "use tensorplay.distributed.algorithms.ddp_comm_hooks instead.",
        stacklevel=2,
    )
    import tensorplay.distributed.algorithms.ddp_comm_hooks as mod

    return getattr(mod, name)
