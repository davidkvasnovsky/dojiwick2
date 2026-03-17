"""Optuna search space definitions for deterministic strategy tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from dojiwick.application.registry.strategy_registry import (
    STRATEGY_MEAN_REVERT,
    STRATEGY_TREND_FOLLOW,
    STRATEGY_VOL_REVERT,
)
from dojiwick.domain.models.value_objects.params import RegimeParams

if TYPE_CHECKING:
    from dojiwick.application.models.pipeline_settings import PipelineSettings


type ParamValue = int | float | str | bool

type ParamSet = dict[str, ParamValue]


class TrialPort(Protocol):
    """Minimal Optuna-like trial interface."""

    def suggest_float(self, name: str, low: float, high: float) -> float:
        """Suggest float value in bounds."""
        ...

    def suggest_int(self, name: str, low: int, high: int) -> int:
        """Suggest integer value in bounds."""
        ...


_BASE_INT_PARAMS: frozenset[str] = frozenset(
    {"max_hold_bars", "adaptive_volatile_max_bars", "adaptive_ranging_max_bars"}
)

REGIME_PARAMS: frozenset[str] = frozenset(
    {"adx_trend_min", "atr_high_pct", "min_confidence", "ema_spread_weak_bps", "atr_low_pct"}
)
assert REGIME_PARAMS <= frozenset(RegimeParams.model_fields), "sync REGIME_PARAMS with RegimeParams"

_BASE_BOUNDS: dict[str, tuple[float, float]] = {
    # Exit / risk params
    "stop_atr_mult": (2.0, 4.5),
    "rr_ratio": (1.2, 4.0),
    "min_stop_distance_pct": (0.20, 0.50),
    "trailing_stop_activation_rr": (0.5, 1.5),
    "trailing_stop_atr_mult": (0.5, 1.3),
    "breakeven_after_rr": (0.3, 1.5),
    "max_hold_bars": (8, 120),
    # Regime params
    "adx_trend_min": (14.0, 25.0),
    "atr_high_pct": (0.9, 2.0),
    "min_confidence": (0.10, 0.50),
    "ema_spread_weak_bps": (4.0, 15.0),
    "atr_low_pct": (0.15, 0.50),
    # Signal-generation params (wide bounds to allow moderate thresholds)
    "trend_pullback_rsi_max": (40.0, 65.0),
    "trend_overbought_rsi_min": (45.0, 65.0),
    "trend_breakout_adx_min": (25.0, 40.0),
    "mean_rsi_oversold": (25.0, 45.0),
    "mean_rsi_overbought": (55.0, 80.0),
    "vol_extreme_oversold": (25.0, 45.0),
    "vol_extreme_overbought": (55.0, 80.0),
    "min_volume_ratio": (0.1, 1.2),
    "trend_max_regime_confidence": (0.70, 0.98),
    # Regime-adaptive exit params
    "adaptive_volatile_stop_scale": (1.0, 2.0),
    "adaptive_volatile_rr_mult": (1.0, 3.0),
    "adaptive_trending_trail_scale": (0.8, 2.0),
    "adaptive_volatile_max_bars": (8, 30),
    "adaptive_ranging_max_bars": (10, 40),
    # Leverage / risk sizing
    "leverage": (1.0, 3.0),
    "risk_per_trade_pct": (1.5, 4.0),
    "max_loss_per_trade_pct": (2.0, 8.0),
    "max_portfolio_risk_pct": (3.0, 25.0),
}

REGIME_SCOPE_PREFIX = "scope_"

REGIME_SCOPE_FIELDS: dict[str, dict[str, tuple[float, float]]] = {
    "ranging": {
        "stop_atr_mult": (1.0, 2.5),
        "rr_ratio": (1.2, 2.5),
        "max_hold_bars": (15, 60),
        "trailing_stop_atr_mult": (0.3, 1.0),
        "breakeven_after_rr": (0.3, 1.5),
    },
    "trending": {
        "stop_atr_mult": (2.0, 4.5),
        "rr_ratio": (2.5, 4.0),
        "max_hold_bars": (50, 200),
        "trailing_stop_atr_mult": (0.5, 2.0),
        "breakeven_after_rr": (0.5, 3.0),
    },
    "volatile": {
        "stop_atr_mult": (1.5, 4.0),
        "rr_ratio": (1.5, 3.5),
        "max_hold_bars": (8, 60),
        "trailing_stop_atr_mult": (0.3, 1.5),
        "breakeven_after_rr": (0.3, 2.0),
    },
}

REGIME_SCOPE_BOUNDS: dict[str, tuple[float, float]] = {
    f"scope_{regime}__{field}": bounds
    for regime, fields in REGIME_SCOPE_FIELDS.items()
    for field, bounds in fields.items()
}

INT_PARAMS: frozenset[str] = _BASE_INT_PARAMS | frozenset(
    key for key in REGIME_SCOPE_BOUNDS if key.split("__", 1)[1] in _BASE_INT_PARAMS
)


NON_STRATEGY_PARAMS: frozenset[str] = frozenset(
    {
        "leverage",
        "risk_per_trade_pct",
        "max_loss_per_trade_pct",
        "max_portfolio_risk_pct",
    }
)


def extend_baseline_with_regime_scope(baseline: dict[str, float]) -> None:
    """Populate ``scope_<regime>__<field>`` baseline entries from global param values."""
    for key in REGIME_SCOPE_BOUNDS:
        field = key.split("__", 1)[1]
        baseline[key] = baseline.get(field, 0.0)


def extract_regularization_baseline(settings: PipelineSettings) -> dict[str, float]:
    """Extract baseline for regularization from settings.

    Used by both the application-layer objectives and the config-layer
    warm-start generator.  Accepts the ``PipelineSettings`` Protocol so
    it works from either layer.
    """
    s = settings.strategy
    r = settings.regime
    baseline: dict[str, float] = {}
    for name in SearchSpace(
        partial_tp_enabled=s.partial_tp_enabled,
        confluence_filter_enabled=s.confluence_filter_enabled,
        enabled_strategies=settings.trading.enabled_strategies,
    ).strategy_param_names():
        val = getattr(s, name)
        if val is None:
            raise ValueError(f"optimization requires settings.strategy.{name} to be set (got None)")
        baseline[name] = float(val)
    for name in REGIME_PARAMS:
        baseline[name] = float(getattr(r, name))
    baseline["leverage"] = float(settings.backtest.leverage)
    baseline["risk_per_trade_pct"] = float(settings.risk.risk_per_trade_pct)
    baseline["max_loss_per_trade_pct"] = float(settings.risk.max_loss_per_trade_pct)
    baseline["max_portfolio_risk_pct"] = float(settings.risk.max_portfolio_risk_pct)
    extend_baseline_with_regime_scope(baseline)
    return baseline


_STRATEGY_SIGNAL_PARAMS: dict[str, frozenset[str]] = {
    STRATEGY_MEAN_REVERT: frozenset({"mean_rsi_oversold", "mean_rsi_overbought"}),
    STRATEGY_TREND_FOLLOW: frozenset(
        {
            "trend_pullback_rsi_max",
            "trend_overbought_rsi_min",
            "trend_breakout_adx_min",
            "trend_max_regime_confidence",
        }
    ),
    STRATEGY_VOL_REVERT: frozenset({"vol_extreme_oversold", "vol_extreme_overbought"}),
}

_BASE_STRATEGY_PARAMS: frozenset[str] = frozenset(
    {
        "stop_atr_mult",
        "rr_ratio",
        "min_stop_distance_pct",
        "trailing_stop_activation_rr",
        "trailing_stop_atr_mult",
        "breakeven_after_rr",
        "max_hold_bars",
        "trend_pullback_rsi_max",
        "trend_overbought_rsi_min",
        "trend_breakout_adx_min",
        "mean_rsi_oversold",
        "mean_rsi_overbought",
        "vol_extreme_oversold",
        "vol_extreme_overbought",
        "min_volume_ratio",
        "trend_max_regime_confidence",
        "adaptive_volatile_stop_scale",
        "adaptive_volatile_rr_mult",
        "adaptive_trending_trail_scale",
        "adaptive_volatile_max_bars",
        "adaptive_ranging_max_bars",
    }
)


@dataclass(slots=True, frozen=True, kw_only=True)
class SearchSpace:
    """Deterministic search ranges for key trading thresholds."""

    partial_tp_enabled: bool = False
    confluence_filter_enabled: bool = False
    enabled_strategies: tuple[str, ...] | None = None
    _cached_bounds: dict[str, tuple[float, float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_cached_bounds", self._compute_bounds())

    def _regime_active(self, regime_key: str) -> bool:
        """Check if any enabled strategy fires in this regime."""
        if self.enabled_strategies is None:
            return True
        if regime_key == "ranging":
            return STRATEGY_MEAN_REVERT in self.enabled_strategies
        if regime_key == "volatile":
            return STRATEGY_TREND_FOLLOW in self.enabled_strategies or STRATEGY_VOL_REVERT in self.enabled_strategies
        if regime_key == "trending":
            return STRATEGY_TREND_FOLLOW in self.enabled_strategies
        return False  # unreachable for known REGIME_SCOPE_FIELDS keys

    def _compute_bounds(self) -> dict[str, tuple[float, float]]:
        b = dict(_BASE_BOUNDS)
        # Skip strategy-specific signal params when their strategy is disabled
        if self.enabled_strategies is not None:
            for strategy, params in _STRATEGY_SIGNAL_PARAMS.items():
                if strategy not in self.enabled_strategies:
                    for k in params:
                        b.pop(k, None)
        # Only include regime scope params for regimes with active strategies
        for regime_key, regime_fields in REGIME_SCOPE_FIELDS.items():
            if self._regime_active(regime_key):
                for field_name, bounds in regime_fields.items():
                    b[f"scope_{regime_key}__{field_name}"] = bounds
        if self.partial_tp_enabled:
            b["partial_tp1_rr"] = (0.5, 2.0)
            b["partial_tp1_fraction"] = (0.3, 0.7)
        if self.confluence_filter_enabled:
            b["min_confluence_score"] = (20.0, 60.0)
        return b

    def bounds(self) -> dict[str, tuple[float, float]]:
        """Return (low, high) bounds for all search space parameters."""
        return self._cached_bounds

    def strategy_param_names(self) -> frozenset[str]:
        """Strategy-level param names in the search space, including conditional additions."""
        names = set(_BASE_STRATEGY_PARAMS)
        if self.enabled_strategies is not None:
            for strategy, params in _STRATEGY_SIGNAL_PARAMS.items():
                if strategy not in self.enabled_strategies:
                    names -= params
        if self.partial_tp_enabled:
            names |= {"partial_tp1_rr", "partial_tp1_fraction"}
        if self.confluence_filter_enabled:
            names |= {"min_confluence_score"}
        return frozenset(names)

    def sample(self, trial: TrialPort) -> ParamSet:
        """Sample optimization parameters from one Optuna trial."""
        params: ParamSet = {}
        for name, (low, high) in self._cached_bounds.items():
            if name in INT_PARAMS:
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                params[name] = trial.suggest_float(name, low, high)
        return params
