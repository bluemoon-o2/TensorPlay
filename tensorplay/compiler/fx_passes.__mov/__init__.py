"""Graph passes beyond the core infra, mirroring ``torch/_inductor/fx_passes``."""

from .fusion_hint import POINTWISE_FUSED_OP_NAMES, PointwiseFusionHint
from .normalize import NormalizeOperators

__all__ = [
    "POINTWISE_FUSED_OP_NAMES",
    "NormalizeOperators",
    "PointwiseFusionHint",
]
