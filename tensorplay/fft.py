"""torch.fft-compatible namespace.

The transforms live natively in p10 (p10/src/backend/cpu/SpectralKernels.cpp
with vendored pocketfft, and p10/src/backend/cuda/SpectralKernels.cu on
cuFFT); this module only re-exports them under the torch.fft names so
torchaudio-derived code calling ``torch.fft.rfft`` works verbatim.
"""
from tensorplay import (
    fft_fft as fft,
    fft_ifft as ifft,
    fft_rfft as rfft,
    fft_irfft as irfft,
)

__all__ = ["fft", "ifft", "rfft", "irfft"]
