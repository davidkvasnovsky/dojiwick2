"""Parameter boundary violation tests."""

import pytest

from fixtures.factories.infrastructure import default_regime_params, default_risk_params, default_strategy_params


def test_regime_adx_ordering() -> None:
    with pytest.raises(ValueError, match="adx_strong_trend_min"):
        default_regime_params(adx_trend_min=30.0, adx_strong_trend_min=20.0)


def test_regime_ema_spread_ordering() -> None:
    with pytest.raises(ValueError, match="ema_spread_strong_bps"):
        default_regime_params(ema_spread_weak_bps=25.0, ema_spread_strong_bps=10.0)


def test_regime_atr_ordering() -> None:
    with pytest.raises(ValueError, match="atr_high_pct must be > atr_low_pct"):
        default_regime_params(atr_low_pct=1.0, atr_high_pct=0.5)


def test_regime_min_confidence_bounds() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        default_regime_params(min_confidence=1.5)


def test_strategy_rr_ratio_must_exceed_one() -> None:
    with pytest.raises(ValueError, match="rr_ratio must be > 1"):
        default_strategy_params(rr_ratio=0.5)


def test_strategy_stop_atr_mult_positive() -> None:
    with pytest.raises(ValueError, match="stop_atr_mult must be > 0"):
        default_strategy_params(stop_atr_mult=0.0)


def test_strategy_mean_rsi_ordering() -> None:
    with pytest.raises(ValueError, match="mean_rsi_oversold must be < mean_rsi_overbought"):
        default_strategy_params(mean_rsi_oversold=80.0, mean_rsi_overbought=30.0)


def test_risk_max_positions_minimum() -> None:
    with pytest.raises(ValueError, match="max_open_positions must be >= 1"):
        default_risk_params(max_open_positions=0)


def test_risk_daily_loss_positive() -> None:
    with pytest.raises(ValueError, match="max_daily_loss_pct must be > 0"):
        default_risk_params(max_daily_loss_pct=0.0)


def test_risk_min_rr_must_exceed_one() -> None:
    with pytest.raises(ValueError, match="min_rr_ratio must be > 1"):
        default_risk_params(min_rr_ratio=0.5)


def test_risk_cooldown_negative() -> None:
    with pytest.raises(ValueError, match="trade_cooldown_sec must be >= 0"):
        default_risk_params(trade_cooldown_sec=-1)


def test_risk_consecutive_losses_minimum() -> None:
    with pytest.raises(ValueError, match="max_consecutive_losses must be >= 1"):
        default_risk_params(max_consecutive_losses=0)


def test_risk_win_rate_floor_bounds() -> None:
    with pytest.raises(ValueError, match="pair_win_rate_floor must be in"):
        default_risk_params(pair_win_rate_floor=1.5)


def test_risk_sector_exposure_minimum() -> None:
    with pytest.raises(ValueError, match="max_sector_exposure must be >= 1"):
        default_risk_params(max_sector_exposure=0)
