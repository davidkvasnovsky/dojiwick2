"""Combinatorially Symmetric Cross-Validation (CSCV) for PBO estimation.

Implements Bailey et al. "Probability of Backtest Overfitting" (2014).
Given a single returns vector, partitions into S contiguous blocks and
enumerates C(S, S/2) train/test combinations.  PBO = fraction where
in-sample Sharpe > 0 but out-of-sample Sharpe <= 0.
"""

from itertools import combinations

import numpy as np

from dojiwick.domain.type_aliases import FloatVector


def compute_pbo(returns: FloatVector, n_partitions: int = 16, *, min_is_sharpe: float = 0.0) -> float:
    """Estimate Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    returns:
        Bar-level returns vector (1-D).
    n_partitions:
        Number of contiguous blocks (must be even, >= 4).

    Returns
    -------
    float
        PBO in [0, 1].  Values > 0.5 indicate likely overfitting.
    """

    if n_partitions < 4:
        raise ValueError(f"n_partitions must be >= 4, got {n_partitions}")
    if n_partitions % 2 != 0:
        raise ValueError(f"n_partitions must be even, got {n_partitions}")
    if n_partitions > 20:
        raise ValueError(f"n_partitions must be <= 20, got {n_partitions}")

    blocks = np.array_split(returns, n_partitions)
    half = n_partitions // 2

    overfit_count = 0
    total = 0

    for is_indices in combinations(range(n_partitions), half):
        is_set = frozenset(is_indices)
        oos_indices = tuple(i for i in range(n_partitions) if i not in is_set)

        is_returns = np.concatenate([blocks[i] for i in is_indices])
        oos_returns = np.concatenate([blocks[i] for i in oos_indices])

        is_sharpe = _sharpe(is_returns)
        oos_sharpe = _sharpe(oos_returns)

        if is_sharpe > min_is_sharpe and oos_sharpe <= 0:
            overfit_count += 1
        total += 1

    return overfit_count / total if total > 0 else 0.0


def _sharpe(returns: FloatVector) -> float:
    """Annualization-free Sharpe ratio (mean / std)."""

    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(returns)) / std
