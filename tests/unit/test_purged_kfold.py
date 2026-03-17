"""Tests for purged K-fold cross-validation splitter."""

import numpy as np
import pytest

from dojiwick.compute.kernels.validation.purged_kfold import purged_kfold_splits


class TestPurgedKfoldSplits:
    def test_fold_count_matches(self) -> None:
        splits = purged_kfold_splits(n_samples=100, n_folds=5, purge_bars=3, embargo_bars=2)
        assert len(splits) == 5

    def test_no_overlap_between_train_and_test(self) -> None:
        splits = purged_kfold_splits(n_samples=100, n_folds=5, purge_bars=3, embargo_bars=2)
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, f"train/test overlap: {overlap}"

    def test_purge_gap_exists(self) -> None:
        """Training indices must not include bars immediately before the test fold."""
        purge_bars = 5
        splits = purged_kfold_splits(n_samples=100, n_folds=5, purge_bars=purge_bars, embargo_bars=0)
        for train_idx, test_idx in splits:
            test_start = int(test_idx[0])
            purge_zone = set(range(max(0, test_start - purge_bars), test_start))
            train_set = set(train_idx.tolist())
            assert purge_zone.isdisjoint(train_set), (
                f"purge zone {purge_zone} overlaps train at test_start={test_start}"
            )

    def test_embargo_applied(self) -> None:
        """Training indices must not include bars immediately after the test fold."""
        embargo_bars = 4
        splits = purged_kfold_splits(n_samples=100, n_folds=5, purge_bars=0, embargo_bars=embargo_bars)
        for train_idx, test_idx in splits:
            test_end = int(test_idx[-1])
            embargo_zone = set(range(test_end + 1, min(100, test_end + 1 + embargo_bars)))
            train_set = set(train_idx.tolist())
            assert embargo_zone.isdisjoint(train_set), (
                f"embargo zone {embargo_zone} overlaps train at test_end={test_end}"
            )

    def test_all_samples_covered(self) -> None:
        """Union of all test folds should cover all samples."""
        splits = purged_kfold_splits(n_samples=100, n_folds=5, purge_bars=2, embargo_bars=2)
        all_test = np.concatenate([test_idx for _, test_idx in splits])
        assert set(all_test.tolist()) == set(range(100))

    def test_small_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            purged_kfold_splits(n_samples=5, n_folds=10, purge_bars=3, embargo_bars=3)
