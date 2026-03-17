"""Trend-follow strategy kernel tests."""

import numpy as np

from dojiwick.domain.enums import MarketState
from dojiwick.compute.kernels.strategy.trend_follow import trend_follow_signal
from fixtures.factories.compute import make_indicator_matrix
from fixtures.factories.infrastructure import default_strategy_params


def test_trending_up_pullback_emits_buy() -> None:
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    ind = make_indicator_matrix(
        size,
        rsi=40.0,
        adx=28.0,
        atr=0.5,
        ema_fast=101.0,
        ema_slow=100.5,
        ema_base=99.0,
        bb_upper=104.0,
        bb_lower=96.0,
        ema_trend=90.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.TRENDING_UP.value], dtype=np.int64)
    settings = default_strategy_params()

    buy, short = trend_follow_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert buy[0]
    assert not short[0]


def test_trending_down_pullback_emits_short() -> None:
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    ind = make_indicator_matrix(
        size,
        rsi=65.0,
        adx=28.0,
        atr=0.5,
        ema_fast=99.0,
        ema_slow=99.5,
        ema_base=101.0,
        bb_upper=104.0,
        bb_lower=96.0,
        ema_trend=110.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.TRENDING_DOWN.value], dtype=np.int64)
    settings = default_strategy_params()

    buy, short = trend_follow_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert not buy[0]
    assert short[0]


def test_ranging_regime_emits_silence() -> None:
    size = 2
    prices = np.full(size, 100.0, dtype=np.float64)
    ind = make_indicator_matrix(size, rsi=50.0, adx=15.0, atr=0.3)
    states = np.full(size, MarketState.RANGING.value, dtype=np.int64)
    settings = default_strategy_params()

    buy, short = trend_follow_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert not np.any(buy)
    assert not np.any(short)


def test_volatile_with_ema_alignment_emits_silence_when_disabled() -> None:
    """VOLATILE bars don't produce trend_follow signals when trend_volatile_ema_enabled=False."""
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    ind = make_indicator_matrix(
        size,
        rsi=40.0,
        adx=28.0,
        atr=1.2,
        ema_fast=101.0,
        ema_slow=100.5,
        ema_base=99.0,
        bb_upper=104.0,
        bb_lower=96.0,
        ema_trend=90.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.VOLATILE.value], dtype=np.int64)
    settings = default_strategy_params(trend_volatile_ema_enabled=False)

    buy, short = trend_follow_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert not buy[0]
    assert not short[0]

    # Also verify bearish EMA alignment in VOLATILE is silent
    ind_bear = make_indicator_matrix(
        size,
        rsi=65.0,
        adx=28.0,
        atr=1.2,
        ema_fast=99.0,
        ema_slow=99.5,
        ema_base=101.0,
        bb_upper=104.0,
        bb_lower=96.0,
        ema_trend=110.0,
        volume_ema_ratio=2.0,
    )

    buy2, short2 = trend_follow_signal(states=states, indicators=ind_bear, prices=prices, settings=settings)
    assert not buy2[0]
    assert not short2[0]


def test_volatile_with_ema_alignment_emits_buy_when_enabled() -> None:
    """VOLATILE bars with EMA alignment produce trend_follow buy when enabled (default)."""
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    ind = make_indicator_matrix(
        size,
        rsi=40.0,
        adx=28.0,
        atr=1.2,
        ema_fast=101.0,
        ema_slow=100.5,
        ema_base=99.0,
        bb_upper=104.0,
        bb_lower=96.0,
        ema_trend=90.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.VOLATILE.value], dtype=np.int64)
    settings = default_strategy_params()  # trend_volatile_ema_enabled=True by default

    buy, short = trend_follow_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert buy[0]
    assert not short[0]


def test_breakdown_short_in_trending_down() -> None:
    """Price <= bb_lower + high ADX in TRENDING_DOWN → short breakdown signal."""
    size = 1
    prices = np.array([95.0], dtype=np.float64)
    ind = make_indicator_matrix(
        size,
        rsi=50.0,
        adx=45.0,
        atr=0.5,
        ema_fast=99.0,
        ema_slow=100.0,
        ema_base=101.0,
        bb_upper=104.0,
        bb_lower=96.0,
        ema_trend=110.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.TRENDING_DOWN.value], dtype=np.int64)
    settings = default_strategy_params()

    buy, short = trend_follow_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert not buy[0]
    assert short[0]


def test_breakout_signal_in_trending_up() -> None:
    size = 1
    prices = np.array([104.0], dtype=np.float64)
    ind = make_indicator_matrix(
        size,
        rsi=55.0,
        adx=45.0,
        atr=0.5,
        ema_fast=103.0,
        ema_slow=102.0,
        ema_base=101.0,
        bb_upper=104.0,
        bb_lower=96.0,
        ema_trend=90.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.TRENDING_UP.value], dtype=np.int64)
    settings = default_strategy_params()

    buy, _ = trend_follow_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert buy[0]


def test_high_confidence_trending_up_filtered() -> None:
    """Signals are suppressed when regime confidence exceeds the cap."""
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    ind = make_indicator_matrix(
        size,
        rsi=40.0,
        adx=28.0,
        atr=0.5,
        ema_fast=101.0,
        ema_slow=100.5,
        ema_base=99.0,
        bb_upper=104.0,
        bb_lower=96.0,
        ema_trend=90.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.TRENDING_UP.value], dtype=np.int64)
    confidence = np.array([0.90], dtype=np.float64)
    settings = default_strategy_params(trend_max_regime_confidence=0.82)

    buy, short = trend_follow_signal(
        states=states,
        indicators=ind,
        prices=prices,
        settings=settings,
        regime_confidence=confidence,
    )
    assert not buy[0]
    assert not short[0]


def test_moderate_confidence_trending_up_passes() -> None:
    """Signals pass through when regime confidence is below the cap."""
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    ind = make_indicator_matrix(
        size,
        rsi=40.0,
        adx=28.0,
        atr=0.5,
        ema_fast=101.0,
        ema_slow=100.5,
        ema_base=99.0,
        bb_upper=104.0,
        bb_lower=96.0,
        ema_trend=90.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.TRENDING_UP.value], dtype=np.int64)
    confidence = np.array([0.70], dtype=np.float64)
    settings = default_strategy_params(trend_max_regime_confidence=0.82)

    buy, short = trend_follow_signal(
        states=states,
        indicators=ind,
        prices=prices,
        settings=settings,
        regime_confidence=confidence,
    )
    assert buy[0]
    assert not short[0]


def test_high_confidence_shorts_pass_with_separate_cap() -> None:
    """Shorts pass when buy cap is exceeded but short cap is not set (None = no cap)."""
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    ind = make_indicator_matrix(
        size,
        rsi=65.0,
        adx=28.0,
        atr=0.5,
        ema_fast=99.0,
        ema_slow=99.5,
        ema_base=101.0,
        bb_upper=104.0,
        bb_lower=96.0,
        ema_trend=110.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.TRENDING_DOWN.value], dtype=np.int64)
    confidence = np.array([0.90], dtype=np.float64)
    # Buy cap at 0.82 would suppress buys, but short_max_confidence is None (no cap)
    settings = default_strategy_params(trend_max_regime_confidence=0.82, trend_short_max_regime_confidence=None)

    buy, short = trend_follow_signal(
        states=states,
        indicators=ind,
        prices=prices,
        settings=settings,
        regime_confidence=confidence,
    )
    assert not buy[0]
    assert short[0]


def test_high_confidence_shorts_filtered_with_short_cap() -> None:
    """Shorts are filtered when short-specific confidence cap is exceeded."""
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    ind = make_indicator_matrix(
        size,
        rsi=65.0,
        adx=28.0,
        atr=0.5,
        ema_fast=99.0,
        ema_slow=99.5,
        ema_base=101.0,
        bb_upper=104.0,
        bb_lower=96.0,
        ema_trend=110.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.TRENDING_DOWN.value], dtype=np.int64)
    confidence = np.array([0.95], dtype=np.float64)
    settings = default_strategy_params(trend_max_regime_confidence=0.99, trend_short_max_regime_confidence=0.90)

    buy, short = trend_follow_signal(
        states=states,
        indicators=ind,
        prices=prices,
        settings=settings,
        regime_confidence=confidence,
    )
    assert not buy[0]
    assert not short[0]
