"""Domain model validation tests."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from dojiwick.domain.indicator_schema import INDICATOR_COUNT
from dojiwick.domain.models.value_objects.batch_models import (
    BatchMarketSnapshot,
    BatchPortfolioSnapshot,
    BatchRegimeProfile,
)
from dojiwick.domain.models.entities.bot_state import BotState
from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.type_aliases import CandleInterval
from dojiwick.domain.models.value_objects.health import HealthStatus
from dojiwick.domain.models.entities.pair_state import PairTradingState
from fixtures.factories.infrastructure import default_strategy_params
from dojiwick.domain.models.value_objects.performance import PerformanceSnapshot
from dojiwick.domain.models.value_objects.signal import Signal


# --- Batch models (existing) ---


def test_mismatched_vector_sizes_raises() -> None:
    with pytest.raises(ValueError, match="price length mismatch"):
        BatchMarketSnapshot(
            pairs=("BTC/USDC", "ETH/USDC"),
            observed_at=datetime.now(UTC),
            price=np.array([100.0], dtype=np.float64),
            indicators=np.zeros((2, INDICATOR_COUNT), dtype=np.float64),
        )


def test_wrong_dtype_raises() -> None:
    bad_price: Any = np.array([100], dtype=np.int64)
    with pytest.raises(ValueError, match="price must be float64"):
        BatchMarketSnapshot(
            pairs=("BTC/USDC",),
            observed_at=datetime.now(UTC),
            price=bad_price,
            indicators=np.zeros((1, INDICATOR_COUNT), dtype=np.float64),
        )


def test_non_finite_values_raises() -> None:
    with pytest.raises(ValueError, match="price must be finite"):
        BatchMarketSnapshot(
            pairs=("BTC/USDC",),
            observed_at=datetime.now(UTC),
            price=np.array([float("inf")], dtype=np.float64),
            indicators=np.zeros((1, INDICATOR_COUNT), dtype=np.float64),
        )


def test_empty_pairs_raises() -> None:
    with pytest.raises(ValueError, match="pairs must not be empty"):
        BatchMarketSnapshot(
            pairs=(),
            observed_at=datetime.now(UTC),
            price=np.array([], dtype=np.float64),
            indicators=np.zeros((0, INDICATOR_COUNT), dtype=np.float64),
        )


def test_timezone_naive_observed_at_raises() -> None:
    with pytest.raises(ValueError, match="observed_at must be timezone-aware"):
        BatchMarketSnapshot(
            pairs=("BTC/USDC",),
            observed_at=datetime.now(),
            price=np.array([100.0], dtype=np.float64),
            indicators=np.zeros((1, INDICATOR_COUNT), dtype=np.float64),
        )


def test_regime_confidence_size_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="confidence length mismatch"):
        BatchRegimeProfile(
            coarse_state=np.array([3, 3], dtype=np.int64),
            confidence=np.array([0.8], dtype=np.float64),
            valid_mask=np.array([True, True], dtype=np.bool_),
        )


def test_portfolio_valid_construction() -> None:
    portfolio = BatchPortfolioSnapshot(
        equity_usd=np.array([1000.0], dtype=np.float64),
        day_start_equity_usd=np.array([1000.0], dtype=np.float64),
        open_positions_total=np.array([0], dtype=np.int64),
        has_open_position=np.array([False], dtype=np.bool_),
        unrealized_pnl_usd=np.array([0.0], dtype=np.float64),
    )
    assert len(portfolio.equity_usd) == 1


# --- Candle validation ---


def test_candle_empty_pair_raises() -> None:
    with pytest.raises(ValueError, match="pair must not be empty"):
        Candle(
            pair="",
            interval=CandleInterval("1h"),
            open_time=datetime.now(UTC),
            open=Decimal(100),
            high=Decimal(105),
            low=Decimal(95),
            close=Decimal(102),
            volume=Decimal(1000),
        )


def test_candle_empty_interval_raises() -> None:
    with pytest.raises(ValueError, match="interval must not be empty"):
        Candle(
            pair="BTC/USDC",
            interval=CandleInterval(""),
            open_time=datetime.now(UTC),
            open=Decimal(100),
            high=Decimal(105),
            low=Decimal(95),
            close=Decimal(102),
            volume=Decimal(1000),
        )


def test_candle_naive_open_time_raises() -> None:
    with pytest.raises(ValueError, match="open_time must be timezone-aware"):
        Candle(
            pair="BTC/USDC",
            interval=CandleInterval("1h"),
            open_time=datetime.now(),
            open=Decimal(100),
            high=Decimal(105),
            low=Decimal(95),
            close=Decimal(102),
            volume=Decimal(1000),
        )


def test_candle_zero_open_raises() -> None:
    with pytest.raises(ValueError, match="open must be positive"):
        Candle(
            pair="BTC/USDC",
            interval=CandleInterval("1h"),
            open_time=datetime.now(UTC),
            open=Decimal(0),
            high=Decimal(105),
            low=Decimal(95),
            close=Decimal(102),
            volume=Decimal(1000),
        )


def test_candle_negative_close_raises() -> None:
    with pytest.raises(ValueError, match="close must be positive"):
        Candle(
            pair="BTC/USDC",
            interval=CandleInterval("1h"),
            open_time=datetime.now(UTC),
            open=Decimal(100),
            high=Decimal(105),
            low=Decimal(95),
            close=Decimal(-1),
            volume=Decimal(1000),
        )


def test_candle_high_less_than_low_raises() -> None:
    with pytest.raises(ValueError, match="high must be >= low"):
        Candle(
            pair="BTC/USDC",
            interval=CandleInterval("1h"),
            open_time=datetime.now(UTC),
            open=Decimal(100),
            high=Decimal(90),
            low=Decimal(95),
            close=Decimal(102),
            volume=Decimal(1000),
        )


def test_candle_negative_volume_raises() -> None:
    with pytest.raises(ValueError, match="volume must be non-negative"):
        Candle(
            pair="BTC/USDC",
            interval=CandleInterval("1h"),
            open_time=datetime.now(UTC),
            open=Decimal(100),
            high=Decimal(105),
            low=Decimal(95),
            close=Decimal(102),
            volume=Decimal(-1),
        )


def test_candle_negative_quote_volume_raises() -> None:
    with pytest.raises(ValueError, match="quote_volume must be non-negative"):
        Candle(
            pair="BTC/USDC",
            interval=CandleInterval("1h"),
            open_time=datetime.now(UTC),
            open=Decimal(100),
            high=Decimal(105),
            low=Decimal(95),
            close=Decimal(102),
            volume=Decimal(1000),
            quote_volume=Decimal(-1),
        )


# --- Signal validation ---


def test_signal_empty_pair_raises() -> None:
    with pytest.raises(ValueError, match="pair must not be empty"):
        Signal(pair="", target_id="test", signal_type="breakout")


def test_signal_empty_type_raises() -> None:
    with pytest.raises(ValueError, match="signal_type must not be empty"):
        Signal(pair="BTC/USDC", target_id="test", signal_type="")


def test_signal_naive_detected_at_raises() -> None:
    with pytest.raises(ValueError, match="detected_at must be timezone-aware"):
        Signal(pair="BTC/USDC", target_id="test", signal_type="breakout", detected_at=datetime.now())


def test_signal_none_detected_at_ok() -> None:
    s = Signal(pair="BTC/USDC", target_id="test", signal_type="breakout", detected_at=None)
    assert s.detected_at is None


# --- BotState validation ---


def test_bot_state_negative_consecutive_errors_raises() -> None:
    with pytest.raises(ValueError, match="consecutive_errors must be non-negative"):
        BotState(consecutive_errors=-1)


def test_bot_state_negative_consecutive_losses_raises() -> None:
    with pytest.raises(ValueError, match="consecutive_losses must be non-negative"):
        BotState(consecutive_losses=-1)


def test_bot_state_negative_daily_trade_count_raises() -> None:
    with pytest.raises(ValueError, match="daily_trade_count must be non-negative"):
        BotState(daily_trade_count=-1)


def test_bot_state_zero_values_ok() -> None:
    state = BotState()
    assert state.consecutive_errors == 0


# --- PairTradingState validation ---


def test_pair_state_empty_pair_raises() -> None:
    with pytest.raises(ValueError, match="pair must not be empty"):
        PairTradingState(pair="", target_id="test", venue="binance", product="usd_c")


def test_pair_state_negative_wins_raises() -> None:
    with pytest.raises(ValueError, match="wins must be non-negative"):
        PairTradingState(pair="BTC/USDC", target_id="test", venue="binance", product="usd_c", wins=-1)


def test_pair_state_negative_losses_raises() -> None:
    with pytest.raises(ValueError, match="losses must be non-negative"):
        PairTradingState(pair="BTC/USDC", target_id="test", venue="binance", product="usd_c", losses=-1)


def test_pair_state_negative_consecutive_losses_raises() -> None:
    with pytest.raises(ValueError, match="consecutive_losses must be non-negative"):
        PairTradingState(pair="BTC/USDC", target_id="test", venue="binance", product="usd_c", consecutive_losses=-1)


# --- PerformanceSnapshot validation ---


def test_performance_naive_observed_at_raises() -> None:
    with pytest.raises(ValueError, match="observed_at must be timezone-aware"):
        PerformanceSnapshot(observed_at=datetime.now(), equity_usd=Decimal(1000))


def test_performance_zero_equity_raises() -> None:
    with pytest.raises(ValueError, match="equity_usd must be positive"):
        PerformanceSnapshot(observed_at=datetime.now(UTC), equity_usd=Decimal(0))


def test_performance_negative_open_positions_raises() -> None:
    with pytest.raises(ValueError, match="open_positions must be non-negative"):
        PerformanceSnapshot(observed_at=datetime.now(UTC), equity_usd=Decimal(1000), open_positions=-1)


def test_performance_negative_drawdown_raises() -> None:
    with pytest.raises(ValueError, match="drawdown_pct must be non-negative"):
        PerformanceSnapshot(observed_at=datetime.now(UTC), equity_usd=Decimal(1000), drawdown_pct=Decimal("-0.01"))


# --- HealthStatus validation ---


def test_health_negative_errors_raises() -> None:
    with pytest.raises(ValueError, match="consecutive_errors must be non-negative"):
        HealthStatus(healthy=True, db_connected=True, last_tick_at=None, consecutive_errors=-1, details={})


def test_health_naive_last_tick_raises() -> None:
    with pytest.raises(ValueError, match="last_tick_at must be timezone-aware"):
        HealthStatus(
            healthy=True,
            db_connected=True,
            last_tick_at=datetime.now(),
            consecutive_errors=0,
            details={},
        )


def test_health_none_last_tick_ok() -> None:
    h = HealthStatus(healthy=True, db_connected=True, last_tick_at=None, consecutive_errors=0, details={})
    assert h.last_tick_at is None


# --- StrategyParams new fields ---


def test_max_hold_bars_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_hold_bars must be >= 1"):
        default_strategy_params(max_hold_bars=0)


def test_max_hold_bars_negative_raises() -> None:
    with pytest.raises(ValueError, match="max_hold_bars must be >= 1"):
        default_strategy_params(max_hold_bars=-1)


def test_max_hold_bars_valid() -> None:
    p = default_strategy_params(max_hold_bars=10)
    assert p.max_hold_bars == 10


def test_max_hold_bars_none_ok() -> None:
    p = default_strategy_params()
    assert p.max_hold_bars is None


def test_trend_pullback_adx_min_zero_raises() -> None:
    with pytest.raises(ValueError, match="trend_pullback_adx_min must be > 0"):
        default_strategy_params(trend_pullback_adx_min=0.0)


def test_trend_pullback_adx_min_negative_raises() -> None:
    with pytest.raises(ValueError, match="trend_pullback_adx_min must be > 0"):
        default_strategy_params(trend_pullback_adx_min=-5.0)


def test_trend_pullback_adx_min_valid() -> None:
    p = default_strategy_params(trend_pullback_adx_min=20.0)
    assert p.trend_pullback_adx_min == 20.0


def test_trend_pullback_adx_min_none_ok() -> None:
    p = default_strategy_params()
    assert p.trend_pullback_adx_min is None
