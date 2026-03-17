"""Small math primitives shared by vector kernels."""

import numpy as np

from dojiwick.domain.type_aliases import FloatVector


def clamp01(values: FloatVector) -> FloatVector:
    """Clamp vector values to [0, 1]."""

    return np.clip(values, 0.0, 1.0)


def safe_divide(numerator: FloatVector, denominator: FloatVector, default: float = 0.0) -> FloatVector:
    """Elementwise safe division with fallback for zero denominators."""

    out = np.full_like(numerator, default, dtype=np.float64)
    valid = denominator != 0.0
    out[valid] = numerator[valid] / denominator[valid]
    return out
