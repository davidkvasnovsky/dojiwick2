"""Tests for CSCV / PBO estimation kernel."""

import numpy as np
import pytest

from dojiwick.compute.kernels.validation.cscv import compute_pbo


class TestComputePbo:
    def test_pbo_zero_for_good_strategy(self) -> None:
        """A strongly positive-mean strategy should have PBO near zero."""
        rng = np.random.default_rng(42)
        returns = rng.normal(loc=0.05, scale=0.01, size=320)
        pbo = compute_pbo(returns, n_partitions=8)
        assert pbo == 0.0

    def test_pbo_high_for_random(self) -> None:
        """A zero-mean random walk should produce elevated PBO."""
        rng = np.random.default_rng(99)
        returns = rng.normal(loc=0.0, scale=1.0, size=320)
        pbo = compute_pbo(returns, n_partitions=8)
        assert pbo > 0.0

    def test_pbo_bounded(self) -> None:
        """PBO must be in [0, 1]."""
        rng = np.random.default_rng(7)
        returns = rng.normal(loc=0.0, scale=1.0, size=160)
        pbo = compute_pbo(returns, n_partitions=8)
        assert 0.0 <= pbo <= 1.0

    def test_pbo_rejects_odd_partitions(self) -> None:
        returns = np.ones(100)
        with pytest.raises(ValueError, match="even"):
            compute_pbo(returns, n_partitions=7)

    def test_pbo_rejects_too_few_partitions(self) -> None:
        returns = np.ones(100)
        with pytest.raises(ValueError, match=">= 4"):
            compute_pbo(returns, n_partitions=2)

    def test_min_is_sharpe_filters_marginal(self) -> None:
        """With min_is_sharpe=0.1, marginal IS performance is not counted as overfitting."""
        rng = np.random.default_rng(42)
        returns = rng.normal(loc=0.05, scale=0.01, size=320)
        pbo_default = compute_pbo(returns, n_partitions=8, min_is_sharpe=0.0)
        pbo_strict = compute_pbo(returns, n_partitions=8, min_is_sharpe=0.1)
        # A higher min_is_sharpe threshold can only reduce or maintain the overfit count
        assert pbo_strict <= pbo_default

    def test_min_is_sharpe_zero_matches_default(self) -> None:
        """min_is_sharpe=0.0 is backward-compatible with the original behavior."""
        rng = np.random.default_rng(99)
        returns = rng.normal(loc=0.0, scale=1.0, size=320)
        pbo_default = compute_pbo(returns, n_partitions=8)
        pbo_explicit = compute_pbo(returns, n_partitions=8, min_is_sharpe=0.0)
        assert pbo_default == pbo_explicit
