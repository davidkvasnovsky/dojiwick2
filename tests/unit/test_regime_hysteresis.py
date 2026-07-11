"""Regime hysteresis state machine tests."""

import numpy as np

from dojiwick.application.orchestration.regime_hysteresis import RegimeHysteresis
from dojiwick.domain.enums import MarketState


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


def test_revert_to_original_during_pending_window() -> None:
    """A -> pending B -> A again: pending clears and the stable state never flips."""
    hyst = RegimeHysteresis()
    pairs = ("BTC/USDC",)

    hyst.apply(pairs, np.array([MarketState.RANGING.value], dtype=np.int64), bars=3)
    hyst.apply(pairs, np.array([MarketState.TRENDING_UP.value], dtype=np.int64), bars=3)

    back = hyst.apply(pairs, np.array([MarketState.RANGING.value], dtype=np.int64), bars=3)
    assert back[0] == MarketState.RANGING.value

    # The interrupted pending run must not resume: two more TRENDING bars is
    # a fresh count, still below the 3-bar threshold
    hyst.apply(pairs, np.array([MarketState.TRENDING_UP.value], dtype=np.int64), bars=3)
    again = hyst.apply(pairs, np.array([MarketState.TRENDING_UP.value], dtype=np.int64), bars=3)
    assert again[0] == MarketState.RANGING.value


def test_multi_pair_pending_state_is_independent() -> None:
    """One pair's pending transition never leaks into another pair's state."""
    hyst = RegimeHysteresis()
    pairs = ("BTC/USDC", "ETH/USDC")
    ranging = MarketState.RANGING.value
    trend = MarketState.TRENDING_UP.value

    hyst.apply(pairs, np.array([ranging, ranging], dtype=np.int64), bars=2)

    mixed = np.array([trend, ranging], dtype=np.int64)
    step1 = hyst.apply(pairs, mixed, bars=2)
    assert step1[0] == ranging and step1[1] == ranging

    step2 = hyst.apply(pairs, mixed, bars=2)
    assert step2[0] == trend, "BTC confirms after 2 bars"
    assert step2[1] == ranging, "ETH never saw a transition"
