"""NumPy-only training-sampling plans used by the optional PyTorch loader."""

from .sampling import SamplingWeights, inverse_frequency_weights

__all__ = ["SamplingWeights", "inverse_frequency_weights"]
