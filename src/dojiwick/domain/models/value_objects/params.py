"""Pure domain parameter types for kernel computation.

Kernels depend on these types, not on config.schema. Config settings
expose a ``.params`` property returning these types. Because params are
frozen Pydantic models, kernels receive validated immutable data with
zero conversion overhead.
"""

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from dojiwick.domain.enums import RegimeExitProfile


class RegimeParams(BaseModel):
    """Kernel-facing regime classification parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    adx_trend_min: float
    adx_strong_trend_min: float
    ema_spread_weak_bps: float
    ema_spread_strong_bps: float
    atr_low_pct: float
    atr_high_pct: float
    atr_extreme_pct: float
    min_confidence: float
    truth_trend_return_pct: float
    truth_volatile_return_pct: float
    trend_weight: float
    spread_weight: float
    vol_weight: float
    volume_clip_lo: float
    volume_clip_hi: float

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.adx_strong_trend_min < self.adx_trend_min:
            raise ValueError("adx_strong_trend_min must be >= adx_trend_min")
        if self.ema_spread_strong_bps < self.ema_spread_weak_bps:
            raise ValueError("ema_spread_strong_bps must be >= ema_spread_weak_bps")
        if self.atr_low_pct <= 0:
            raise ValueError("atr_low_pct must be > 0")
        if self.atr_high_pct <= self.atr_low_pct:
            raise ValueError("atr_high_pct must be > atr_low_pct")
        if self.atr_extreme_pct <= self.atr_high_pct:
            raise ValueError("atr_extreme_pct must be > atr_high_pct")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        if abs(self.trend_weight + self.spread_weight + self.vol_weight - 1.0) > 1e-9:
            raise ValueError("trend_weight + spread_weight + vol_weight must sum to 1.0")
        if self.volume_clip_lo >= self.volume_clip_hi:
            raise ValueError("volume_clip_lo must be < volume_clip_hi")
        return self


class StrategyParams(BaseModel):
    """Kernel-facing strategy signal parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_variant: str
    stop_atr_mult: float
    rr_ratio: float
    min_stop_distance_pct: float
    trend_pullback_rsi_max: float
    trend_overbought_rsi_min: float
    trend_breakout_adx_min: float
    mean_rsi_oversold: float
    mean_rsi_overbought: float
    vol_extreme_oversold: float
    vol_extreme_overbought: float
    trailing_stop_activation_rr: float | None = None
    trailing_stop_atr_mult: float | None = None
    breakeven_after_rr: float | None = None
    min_volume_ratio: float
    max_hold_bars: int | None = None
    trend_pullback_adx_min: float | None = None
    trend_max_regime_confidence: float | None = None
    trend_short_max_regime_confidence: float | None = None
    trend_volatile_ema_enabled: bool
    partial_tp_enabled: bool
    partial_tp1_rr: float
    partial_tp1_fraction: float
    mean_revert_use_bb_mid_tp: bool
    mean_revert_disable_breakeven: bool
    mean_revert_disable_ema_filter: bool
    macd_filter_enabled: bool
    confluence_filter_enabled: bool
    min_confluence_score: float
    regime_exit_profile: RegimeExitProfile
    adaptive_volatile_stop_scale: float
    adaptive_volatile_rr_mult: float
    adaptive_trending_trail_scale: float
    adaptive_volatile_max_bars: int
    adaptive_ranging_max_bars: int
    confluence_rsi_midpoint: float
    confluence_rsi_range: float
    confluence_volume_baseline: float
    confluence_volume_multiplier: float
    confluence_adx_baseline: float
    confluence_adx_range: float
    partial_tp_stop_ratio: float

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.default_variant:
            raise ValueError("default_variant must not be empty")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be > 0")
        if self.rr_ratio <= 1:
            raise ValueError("rr_ratio must be > 1")
        if self.min_stop_distance_pct <= 0:
            raise ValueError("min_stop_distance_pct must be > 0")
        if self.mean_rsi_oversold >= self.mean_rsi_overbought:
            raise ValueError("mean_rsi_oversold must be < mean_rsi_overbought")
        if self.vol_extreme_oversold >= self.vol_extreme_overbought:
            raise ValueError("vol_extreme_oversold must be < vol_extreme_overbought")
        if self.trailing_stop_atr_mult is not None and self.trailing_stop_atr_mult >= self.stop_atr_mult:
            raise ValueError("trailing_stop_atr_mult must be < stop_atr_mult")
        if self.min_volume_ratio <= 0:
            raise ValueError("min_volume_ratio must be > 0")
        if self.max_hold_bars is not None and self.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be >= 1")
        if self.trend_pullback_adx_min is not None and self.trend_pullback_adx_min <= 0:
            raise ValueError("trend_pullback_adx_min must be > 0")
        if self.trend_max_regime_confidence is not None and not 0 < self.trend_max_regime_confidence <= 1:
            raise ValueError("trend_max_regime_confidence must be in (0, 1]")
        if self.trend_short_max_regime_confidence is not None and not 0 < self.trend_short_max_regime_confidence <= 1:
            raise ValueError("trend_short_max_regime_confidence must be in (0, 1]")
        if self.partial_tp1_rr <= 0:
            raise ValueError("partial_tp1_rr must be > 0")
        if not 0 < self.partial_tp1_fraction < 1:
            raise ValueError("partial_tp1_fraction must be in (0, 1)")
        if not 0 <= self.min_confluence_score <= 100:
            raise ValueError("min_confluence_score must be in [0, 100]")
        if self.adaptive_volatile_stop_scale <= 0:
            raise ValueError("adaptive_volatile_stop_scale must be > 0")
        if self.adaptive_volatile_rr_mult <= 0:
            raise ValueError("adaptive_volatile_rr_mult must be > 0")
        if self.adaptive_trending_trail_scale <= 0:
            raise ValueError("adaptive_trending_trail_scale must be > 0")
        if self.adaptive_volatile_max_bars < 1:
            raise ValueError("adaptive_volatile_max_bars must be >= 1")
        if self.adaptive_ranging_max_bars < 1:
            raise ValueError("adaptive_ranging_max_bars must be >= 1")
        if self.confluence_rsi_range <= 0:
            raise ValueError("confluence_rsi_range must be > 0")
        if self.confluence_volume_multiplier <= 0:
            raise ValueError("confluence_volume_multiplier must be > 0")
        if self.confluence_adx_range <= 0:
            raise ValueError("confluence_adx_range must be > 0")
        if not 0.0 <= self.partial_tp_stop_ratio <= 1.0:
            raise ValueError("partial_tp_stop_ratio must be in [0, 1]")
        return self


class RiskParams(BaseModel):
    """Kernel-facing risk gate and sizing parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_daily_loss_pct: float
    max_open_positions: int
    min_rr_ratio: float
    risk_per_trade_pct: float
    min_notional_usd: float
    max_notional_pct_of_equity: float
    max_notional_usd: float
    max_loss_per_trade_pct: float
    max_portfolio_risk_pct: float
    trade_cooldown_sec: int
    max_consecutive_losses: int
    pair_win_rate_floor: float
    max_sector_exposure: int
    max_risk_inflation_mult: float
    drawdown_halt_pct: float
    drawdown_risk_scale_enabled: bool
    drawdown_risk_scale_max_dd: float
    drawdown_risk_scale_floor: float
    equity_curve_filter_enabled: bool
    equity_curve_filter_period: int
    portfolio_risk_baseline_pairs: int

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.max_daily_loss_pct <= 0:
            raise ValueError("max_daily_loss_pct must be > 0")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be >= 1")
        if self.min_rr_ratio <= 1:
            raise ValueError("min_rr_ratio must be > 1")
        if self.risk_per_trade_pct <= 0:
            raise ValueError("risk_per_trade_pct must be > 0")
        if self.min_notional_usd <= 0:
            raise ValueError("min_notional_usd must be > 0")
        if self.max_notional_pct_of_equity <= 0:
            raise ValueError("max_notional_pct_of_equity must be > 0")
        if self.max_notional_usd <= 0:
            raise ValueError("max_notional_usd must be > 0")
        if self.max_loss_per_trade_pct <= 0:
            raise ValueError("max_loss_per_trade_pct must be > 0")
        if self.max_portfolio_risk_pct <= 0:
            raise ValueError("max_portfolio_risk_pct must be > 0")
        if self.trade_cooldown_sec < 0:
            raise ValueError("trade_cooldown_sec must be >= 0")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be >= 1")
        if not 0.0 <= self.pair_win_rate_floor <= 1.0:
            raise ValueError("pair_win_rate_floor must be in [0, 1]")
        if self.max_sector_exposure < 1:
            raise ValueError("max_sector_exposure must be >= 1")
        if self.max_risk_inflation_mult < 1.0:
            raise ValueError("max_risk_inflation_mult must be >= 1.0")
        if self.drawdown_halt_pct <= 0:
            raise ValueError("drawdown_halt_pct must be > 0")
        if self.drawdown_risk_scale_max_dd <= 0:
            raise ValueError("drawdown_risk_scale_max_dd must be > 0")
        if not 0.0 < self.drawdown_risk_scale_floor <= 1.0:
            raise ValueError("drawdown_risk_scale_floor must be in (0, 1]")
        if self.equity_curve_filter_period < 2:
            raise ValueError("equity_curve_filter_period must be >= 2")
        if self.portfolio_risk_baseline_pairs < 1:
            raise ValueError("portfolio_risk_baseline_pairs must be >= 1")
        return self


def extract_param_vector(settings: tuple[StrategyParams, ...], field: str) -> np.ndarray:
    """Extract a single field from per-pair StrategyParams into a float64 vector."""
    values = [getattr(s, field) for s in settings]
    if any(v is None for v in values):
        raise ValueError(f"extract_param_vector: field '{field}' contains None values; use explicit handling")
    return np.array(values, dtype=np.float64)


def resolve_param_vector(
    pre_extracted: dict[str, np.ndarray] | None,
    per_pair: tuple[StrategyParams, ...],
    field: str,
) -> np.ndarray:
    """Resolve a param vector from pre-extracted cache or per-pair settings."""
    if pre_extracted is not None and field in pre_extracted:
        return pre_extracted[field]
    return extract_param_vector(per_pair, field)


def resolve_optional_param_vector(
    pre_extracted: dict[str, np.ndarray] | None,
    per_pair: tuple[StrategyParams, ...],
    field: str,
    default: float = 0.0,
) -> np.ndarray | None:
    """Like resolve_param_vector but returns None if all values are None, or fills None->default."""
    if pre_extracted is not None and field in pre_extracted:
        return pre_extracted[field]
    values = [getattr(s, field) for s in per_pair]
    if all(v is None for v in values):
        return None
    return np.array([v if v is not None else default for v in values], dtype=np.float64)
