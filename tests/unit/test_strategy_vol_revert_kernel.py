"""Volatility-reversion strategy kernel tests."""

import numpy as np

from dojiwick.domain.enums import MarketState
from dojiwick.compute.kernels.strategy.vol_revert import vol_revert_signal
from fixtures.factories.compute import make_indicator_matrix
from fixtures.factories.infrastructure import default_strategy_params


def test_volatile_extreme_oversold_emits_buy() -> None:
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    # Triple EMA stack bullish: ema_slow(105) > ema_base(100) > ema_trend(90)
    ind = make_indicator_matrix(
        size,
        rsi=25.0,
        bb_lower=105.0,
        ema_slow=105.0,
        ema_base=100.0,
        ema_trend=90.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.VOLATILE.value], dtype=np.int64)
    settings = default_strategy_params()

    buy, short = vol_revert_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert buy[0]
    assert not short[0]


def test_volatile_extreme_overbought_emits_short() -> None:
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    # Triple EMA stack bearish: ema_slow(85) < ema_base(90) < ema_trend(110)
    ind = make_indicator_matrix(
        size,
        rsi=80.0,
        bb_upper=95.0,
        ema_slow=85.0,
        ema_base=90.0,
        ema_trend=110.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.VOLATILE.value], dtype=np.int64)
    settings = default_strategy_params()

    buy, short = vol_revert_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert not buy[0]
    assert short[0]


def test_ranging_regime_emits_silence() -> None:
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    ind = make_indicator_matrix(size, rsi=25.0)
    states = np.array([MarketState.RANGING.value], dtype=np.int64)
    settings = default_strategy_params()

    buy, short = vol_revert_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert not buy[0]
    assert not short[0]


def test_volatile_oversold_without_triple_ema_emits_buy() -> None:
    """ema_slow > ema_trend but ema_base NOT between them — should still fire after relaxation."""
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    # ema_slow(105) > ema_trend(90) but ema_base(110) breaks triple alignment
    ind = make_indicator_matrix(
        size,
        rsi=25.0,
        bb_lower=105.0,
        ema_slow=105.0,
        ema_base=110.0,
        ema_trend=90.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.VOLATILE.value], dtype=np.int64)
    settings = default_strategy_params()

    buy, short = vol_revert_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert buy[0]
    assert not short[0]


def test_volatile_oversold_with_ema_slow_below_trend_no_buy() -> None:
    """ema_slow < ema_trend — directional filter still blocks buy."""
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    ind = make_indicator_matrix(
        size,
        rsi=25.0,
        bb_lower=105.0,
        ema_slow=85.0,
        ema_base=100.0,
        ema_trend=90.0,
        volume_ema_ratio=2.0,
    )
    states = np.array([MarketState.VOLATILE.value], dtype=np.int64)
    settings = default_strategy_params()

    buy, short = vol_revert_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert not buy[0]
    assert not short[0]


def test_volatile_neutral_rsi_emits_silence() -> None:
    size = 1
    prices = np.array([100.0], dtype=np.float64)
    ind = make_indicator_matrix(size, rsi=50.0)
    states = np.array([MarketState.VOLATILE.value], dtype=np.int64)
    settings = default_strategy_params()

    buy, short = vol_revert_signal(states=states, indicators=ind, prices=prices, settings=settings)
    assert not buy[0]
    assert not short[0]
