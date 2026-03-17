"""Math kernel edge case tests."""

import numpy as np

from dojiwick.compute.kernels.math import clamp01, safe_divide


def test_clamp01_clips_high_values() -> None:
    values = np.array([0.5, 1.5, -0.5, 0.0, 1.0], dtype=np.float64)
    result = clamp01(values)
    expected = np.array([0.5, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)


def test_clamp01_handles_nan() -> None:
    values = np.array([float("nan"), 0.5], dtype=np.float64)
    result = clamp01(values)
    assert np.isnan(result[0])
    assert result[1] == 0.5


def test_clamp01_handles_inf() -> None:
    values = np.array([float("inf"), float("-inf")], dtype=np.float64)
    result = clamp01(values)
    assert result[0] == 1.0
    assert result[1] == 0.0


def test_safe_divide_normal_case() -> None:
    num = np.array([10.0, 20.0], dtype=np.float64)
    den = np.array([2.0, 5.0], dtype=np.float64)
    result = safe_divide(num, den)
    np.testing.assert_array_almost_equal(result, [5.0, 4.0])


def test_safe_divide_zero_denominator() -> None:
    num = np.array([10.0, 20.0], dtype=np.float64)
    den = np.array([0.0, 5.0], dtype=np.float64)
    result = safe_divide(num, den)
    assert result[0] == 0.0
    assert result[1] == 4.0


def test_safe_divide_custom_default() -> None:
    num = np.array([10.0], dtype=np.float64)
    den = np.array([0.0], dtype=np.float64)
    result = safe_divide(num, den, default=-1.0)
    assert result[0] == -1.0


def test_safe_divide_all_zeros() -> None:
    num = np.zeros(3, dtype=np.float64)
    den = np.zeros(3, dtype=np.float64)
    result = safe_divide(num, den)
    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])
