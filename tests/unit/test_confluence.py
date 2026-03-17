"""Tests for confluence scoring kernel — HOLD row gating."""

import numpy as np
import pytest

from dojiwick.compute.kernels.strategy.confluence import compute_confluence_score
from dojiwick.domain.enums import TradeAction
from dojiwick.domain.indicator_schema import INDICATOR_INDEX
from fixtures.factories.compute import make_indicator_matrix
from fixtures.factories.infrastructure import default_strategy_params

# Defaults matching the old _make_indicators helper (base fill was 0.0).
_BUY_DEFAULTS: dict[str, float] = {
    "rsi": 30.0,
    "adx": 40.0,
    "macd_histogram": 1.0,
    "volume_ema_ratio": 2.0,
    "ema_fast": 30.0,
    "ema_slow": 20.0,
    "ema_base": 10.0,
}


@pytest.mark.unit
class TestConfluenceHoldGating:
    def test_hold_rows_score_zero(self) -> None:
        """HOLD rows must receive score=0 regardless of indicator values."""
        n = 3
        indicators = make_indicator_matrix(n, **_BUY_DEFAULTS)
        prices = np.full(n, 100.0)
        action = np.full(n, TradeAction.HOLD.value, dtype=np.int64)
        settings = default_strategy_params()

        scores = compute_confluence_score(indicators, prices, action, settings)
        np.testing.assert_array_equal(scores, 0.0)

    def test_buy_rows_score_all_dimensions(self) -> None:
        """BUY rows accumulate scores from all 5 dimensions."""
        n = 1
        indicators = make_indicator_matrix(n, **_BUY_DEFAULTS)
        prices = np.full(n, 100.0)
        action = np.array([TradeAction.BUY.value], dtype=np.int64)
        settings = default_strategy_params()

        scores = compute_confluence_score(indicators, prices, action, settings)
        assert scores[0] > 0

    def test_short_rows_score_all_dimensions(self) -> None:
        """SHORT rows accumulate scores from all 5 dimensions."""
        n = 1
        indicators = make_indicator_matrix(
            n,
            **{
                **_BUY_DEFAULTS,
                "rsi": 70.0,
                "macd_histogram": -1.0,
                "ema_fast": 10.0,
                "ema_slow": 20.0,
                "ema_base": 30.0,
            },
        )
        prices = np.full(n, 100.0)
        action = np.array([TradeAction.SHORT.value], dtype=np.int64)
        settings = default_strategy_params()

        scores = compute_confluence_score(indicators, prices, action, settings)
        assert scores[0] > 0

    def test_mixed_actions_hold_gets_zero(self) -> None:
        """In a mixed array, only HOLD rows get zero."""
        n = 3
        indicators = make_indicator_matrix(n, **_BUY_DEFAULTS)
        prices = np.full(n, 100.0)
        action = np.array(
            [TradeAction.BUY.value, TradeAction.HOLD.value, TradeAction.SHORT.value],
            dtype=np.int64,
        )
        # Adjust SHORT row indicators
        indicators[2, INDICATOR_INDEX["rsi"]] = 70.0
        indicators[2, INDICATOR_INDEX["macd_histogram"]] = -1.0
        indicators[2, INDICATOR_INDEX["ema_fast"]] = 10.0
        indicators[2, INDICATOR_INDEX["ema_slow"]] = 20.0
        indicators[2, INDICATOR_INDEX["ema_base"]] = 30.0
        settings = default_strategy_params()

        scores = compute_confluence_score(indicators, prices, action, settings)
        assert scores[0] > 0, "BUY row should have positive score"
        assert scores[1] == 0.0, "HOLD row should have zero score"
        assert scores[2] > 0, "SHORT row should have positive score"
