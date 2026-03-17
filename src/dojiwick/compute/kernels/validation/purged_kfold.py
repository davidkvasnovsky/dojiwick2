"""Purged K-fold cross-validation splitter.

Produces contiguous test folds with purge gaps before each test fold and
embargo gaps after, preventing information leakage from the training set
into the test set (de Prado, *Advances in Financial Machine Learning*).
"""

import numpy as np

from dojiwick.domain.type_aliases import IntVector


def purged_kfold_splits(
    n_samples: int,
    n_folds: int,
    purge_bars: int,
    embargo_bars: int,
) -> list[tuple[IntVector, IntVector]]:
    """Return (train_indices, test_indices) for each fold.

    Parameters
    ----------
    n_samples:
        Total number of sequential bars.
    n_folds:
        Number of contiguous test folds.
    purge_bars:
        Rows purged *before* the test fold (removes look-ahead leak).
    embargo_bars:
        Rows embargoed *after* the test fold (removes autocorrelation leak).

    Raises
    ------
    ValueError
        If ``n_samples`` is too small to create the requested folds.
    """

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    min_fold_size = 1
    min_required = n_folds * min_fold_size + (purge_bars + embargo_bars)
    if n_samples < min_required:
        raise ValueError(
            f"n_samples ({n_samples}) too small for {n_folds} folds "
            f"with purge={purge_bars}, embargo={embargo_bars} "
            f"(need >= {min_required})"
        )

    indices = np.arange(n_samples)
    fold_boundaries = np.array_split(indices, n_folds)

    splits: list[tuple[IntVector, IntVector]] = []

    for fold_idx in range(n_folds):
        test_idx = fold_boundaries[fold_idx]
        test_start = int(test_idx[0])
        test_end = int(test_idx[-1])

        purge_start = max(0, test_start - purge_bars)
        embargo_end = min(n_samples - 1, test_end + embargo_bars)

        mask = np.ones(n_samples, dtype=np.bool_)
        mask[purge_start : embargo_end + 1] = False
        train_idx = indices[mask]

        splits.append((train_idx, test_idx.astype(np.int64)))

    return splits
