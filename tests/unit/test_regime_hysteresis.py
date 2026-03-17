"""Regime hysteresis state machine tests."""

import numpy as np

from dojiwick.domain.enums import MarketState
from dojiwick.application.orchestration.regime_hysteresis import RegimeHysteresis


def test_no_change_stays_stable() -> None:
    hyst = RegimeHysteresis()
    pairs = ("BTC/USDC",)
    state = np.array([MarketState.RANGING.value], dtype=np.int64)

    result1 = hyst.apply(pairs, state, bars=2)
    result2 = hyst.apply(pairs, state, bars=2)

    assert result1[0] == MarketState.RANGING.value
    assert result2[0] == MarketState.RANGING.value


def test_confirmed_transition_after_n_bars() -> None:
    hyst = RegimeHysteresis()
    pairs = ("BTC/USDC",)

    hyst.apply(pairs, np.array([MarketState.RANGING.value], dtype=np.int64), bars=2)

    result1 = hyst.apply(pairs, np.array([MarketState.TRENDING_UP.value], dtype=np.int64), bars=2)
    assert result1[0] == MarketState.RANGING.value

    result2 = hyst.apply(pairs, np.array([MarketState.TRENDING_UP.value], dtype=np.int64), bars=2)
    assert result2[0] == MarketState.TRENDING_UP.value


def test_interrupted_pending_resets() -> None:
    hyst = RegimeHysteresis()
    pairs = ("BTC/USDC",)

    hyst.apply(pairs, np.array([MarketState.RANGING.value], dtype=np.int64), bars=3)

    hyst.apply(pairs, np.array([MarketState.TRENDING_UP.value], dtype=np.int64), bars=3)

    result = hyst.apply(pairs, np.array([MarketState.VOLATILE.value], dtype=np.int64), bars=3)
    assert result[0] == MarketState.RANGING.value


def test_new_pair_initializes_immediately() -> None:
    hyst = RegimeHysteresis()
    pairs = ("NEW/PAIR",)
    state = np.array([MarketState.VOLATILE.value], dtype=np.int64)

    result = hyst.apply(pairs, state, bars=5)
    assert result[0] == MarketState.VOLATILE.value


def test_bars_one_transitions_immediately() -> None:
    hyst = RegimeHysteresis()
    pairs = ("BTC/USDC",)

    hyst.apply(pairs, np.array([MarketState.RANGING.value], dtype=np.int64), bars=1)

    result = hyst.apply(pairs, np.array([MarketState.TRENDING_DOWN.value], dtype=np.int64), bars=1)
    assert result[0] == MarketState.TRENDING_DOWN.value
