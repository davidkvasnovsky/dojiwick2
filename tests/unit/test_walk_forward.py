"""Tests for walk-forward validation splitter."""

import numpy as np
import pytest

from dojiwick.compute.kernels.validation.walk_forward import walk_forward_splits


class TestWalkForwardSplits:
    def test_rolling_windows_no_overlap(self) -> None:
        """Test windows within each split don't overlap."""
        splits = walk_forward_splits(n_samples=300, train_size=100, test_size=50)
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_expanding_windows_grow(self) -> None:
        """In expanding mode, training window must grow with each step."""
        splits = walk_forward_splits(n_samples=300, train_size=100, test_size=50, expanding=True)
        assert len(splits) >= 2
        for i in range(1, len(splits)):
            assert len(splits[i][0]) > len(splits[i - 1][0])

    def test_coverage(self) -> None:
        """All OOS windows combined should cover a contiguous range."""
        splits = walk_forward_splits(n_samples=300, train_size=100, test_size=50)
        all_oos = np.concatenate([test_idx for _, test_idx in splits])
        # OOS windows should be contiguous and non-overlapping
        assert np.all(np.diff(all_oos) == 1)

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            walk_forward_splits(n_samples=10, train_size=100, test_size=50)

    def test_oos_follows_is(self) -> None:
        """OOS window must start immediately after the IS window ends."""
        splits = walk_forward_splits(n_samples=300, train_size=100, test_size=50)
        for train_idx, test_idx in splits:
            assert int(test_idx[0]) == int(train_idx[-1]) + 1

    def test_rolling_train_size_constant(self) -> None:
        """In rolling mode, all training windows should be the same size."""
        splits = walk_forward_splits(n_samples=500, train_size=200, test_size=50)
        for train_idx, _test_idx in splits:
            assert len(train_idx) == 200

    def test_at_least_one_window(self) -> None:
        """Minimum viable case: exactly train_size + test_size samples."""
        splits = walk_forward_splits(n_samples=150, train_size=100, test_size=50)
        assert len(splits) == 1
