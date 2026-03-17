"""Walk-forward validation splitter.

Produces rolling (fixed-size) or expanding training windows followed by
contiguous out-of-sample test windows, marching forward through time.
"""

import numpy as np

from dojiwick.domain.type_aliases import IntVector


def walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    *,
    expanding: bool = False,
) -> list[tuple[IntVector, IntVector]]:
    """Return (train_indices, test_indices) per walk-forward window.

    Parameters
    ----------
    n_samples:
        Total number of sequential bars.
    train_size:
        Minimum training window size (fixed for rolling, initial for expanding).
    test_size:
        Out-of-sample window size per step.
    expanding:
        If ``True``, the training window grows from the beginning of the
        series.  If ``False`` (default), a rolling fixed-size window is used.

    Raises
    ------
    ValueError
        If ``n_samples`` is insufficient for at least one train+test window.
    """

    if train_size < 1:
        raise ValueError(f"train_size must be >= 1, got {train_size}")
    if test_size < 1:
        raise ValueError(f"test_size must be >= 1, got {test_size}")

    min_required = train_size + test_size
    if n_samples < min_required:
        raise ValueError(
            f"n_samples ({n_samples}) too small for train_size={train_size}, "
            f"test_size={test_size} (need >= {min_required})"
        )

    splits: list[tuple[IntVector, IntVector]] = []
    test_start = train_size

    while test_start + test_size <= n_samples:
        test_end = test_start + test_size

        if expanding:
            train_idx = np.arange(0, test_start, dtype=np.int64)
        else:
            train_idx = np.arange(test_start - train_size, test_start, dtype=np.int64)

        test_idx = np.arange(test_start, test_end, dtype=np.int64)
        splits.append((train_idx, test_idx))

        test_start = test_end

    return splits
