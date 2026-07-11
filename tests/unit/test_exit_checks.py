"""Gap-aware exit fill tests for the backtest exit engine."""

import pytest

from dojiwick.application.use_cases.run_backtest import _check_exit, _OpenPosition  # pyright: ignore[reportPrivateUsage]
from dojiwick.domain.enums import CloseReason, TradeAction


def _long(stop: float = 95.0, tp: float = 110.0, max_hold: int = 0) -> _OpenPosition:
    return _OpenPosition(
        entry_price=100.0,
        entry_bar=0,
        quantity=1.0,
        action=TradeAction.BUY,
        stop_price=stop,
        take_profit_price=tp,
        notional_usd=100.0,
        strategy_name="test",
        pair="BTC/USDC",
        original_stop=stop,
        max_hold_bars=max_hold,
        observed_bars=1,
    )


def _short(stop: float = 105.0, tp: float = 90.0) -> _OpenPosition:
    pos = _long(stop=stop, tp=tp)
    pos.action = TradeAction.SHORT
    return pos


def test_long_stop_fills_at_trigger_without_gap() -> None:
    result = _check_exit(_long(), bar_open=100.0, bar_high=101.0, bar_low=94.0)
    assert result == (95.0, CloseReason.STOP_LOSS)


def test_long_stop_gap_through_fills_at_open() -> None:
    result = _check_exit(_long(), bar_open=90.0, bar_high=92.0, bar_low=88.0)
    assert result == (90.0, CloseReason.STOP_LOSS)


def test_long_tp_favorable_gap_fills_at_open() -> None:
    result = _check_exit(_long(), bar_open=112.0, bar_high=115.0, bar_low=109.0)
    assert result == (112.0, CloseReason.TAKE_PROFIT)


def test_short_stop_gap_through_fills_at_open() -> None:
    result = _check_exit(_short(), bar_open=108.0, bar_high=109.0, bar_low=107.0)
    assert result == (108.0, CloseReason.STOP_LOSS)


def test_short_tp_favorable_gap_fills_at_open() -> None:
    result = _check_exit(_short(), bar_open=88.0, bar_high=89.0, bar_low=86.0)
    assert result == (88.0, CloseReason.TAKE_PROFIT)


def test_time_exit_fills_at_open() -> None:
    pos = _long(max_hold=1)
    result = _check_exit(pos, bar_open=101.5, bar_high=102.0, bar_low=99.0)
    assert result == (101.5, CloseReason.TIME_EXIT)


def test_no_exit_when_untouched() -> None:
    assert _check_exit(_long(), bar_open=100.0, bar_high=105.0, bar_low=96.0) is None


def test_stop_checked_before_tp_on_same_bar() -> None:
    # Bar touches both: conservative stop-first semantics
    result = _check_exit(_long(), bar_open=100.0, bar_high=111.0, bar_low=94.0)
    assert result is not None
    assert result[1] is CloseReason.STOP_LOSS
    assert result[0] == pytest.approx(95.0)  # pyright: ignore[reportUnknownMemberType]
