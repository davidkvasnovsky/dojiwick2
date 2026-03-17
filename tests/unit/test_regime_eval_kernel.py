"""Regime evaluation kernel tests."""

import numpy as np

from fixtures.factories.infrastructure import default_settings
from dojiwick.compute.kernels.regime.evaluate import evaluate_regimes


def test_regime_evaluation_report_has_core_fields() -> None:
    settings = default_settings()
    prices = np.array([100.0, 101.0, 102.0, 103.0, 102.5], dtype=np.float64)
    predicted = np.array([1, 1, 1, 3, 3], dtype=np.int64)

    report = evaluate_regimes(
        prices=prices,
        predicted_states=predicted,
        settings=settings.regime.params,
        horizons=(1, 2),
    )

    assert report.total_points == 7
    assert 0.0 <= report.coarse_macro_f1 <= 1.0
    assert report.mean_run_length >= 1.0
