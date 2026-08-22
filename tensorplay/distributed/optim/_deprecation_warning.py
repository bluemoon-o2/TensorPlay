# Ported from torch/distributed/optim/_deprecation_warning.py.
import warnings


def _scripted_functional_optimizer_deprecation_warning(stacklevel=2):
    warnings.warn(
        "Functional optimizers and scripting them is deprecated. "
        "Please use tensorplay.distributed.optim directly.",
        stacklevel=stacklevel,
    )
