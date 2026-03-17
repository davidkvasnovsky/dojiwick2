"""Fluent builders for infrastructure/config test data."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Self

from dojiwick.config.schema import (
    AISettings,
    AdaptiveSettings,
    BacktestSettings,
    ExchangeSettings,
    OptimizationSettings,
    RegimeSettings,
    ResearchGateSettings,
    RiskSettings,
    Settings,
    SystemSettings,
    TargetConfig,
    TradingSettings,
    UniverseSettings,
)
from dojiwick.domain.enums import (
    AdaptiveMode,
    EntryPriceModel,
    MissingBarPolicy,
    ObjectiveMode,
    PositionMode,
    WFMode,
)
from dojiwick.domain.enums import RegimeExitProfile
from dojiwick.domain.models.value_objects.params import RegimeParams, RiskParams, StrategyParams
from dojiwick.domain.numerics import to_money
from dojiwick.domain.models.value_objects.performance import PerformanceSnapshot
from dojiwick.domain.type_aliases import ProductCode, VenueCode


# --- Factory functions for models that lost defaults ---


def default_regime_params(**overrides: Any) -> RegimeParams:
    """Test defaults for RegimeParams."""
    defaults: dict[str, Any] = {
        "adx_trend_min": 20.0,
        "adx_strong_trend_min": 35.0,
        "ema_spread_weak_bps": 8.0,
        "ema_spread_strong_bps": 25.0,
        "atr_low_pct": 0.30,
        "atr_high_pct": 0.90,
        "atr_extreme_pct": 1.50,
        "min_confidence": 0.55,
        "truth_trend_return_pct": 0.40,
        "truth_volatile_return_pct": 1.00,
        "trend_weight": 0.4,
        "spread_weight": 0.3,
        "vol_weight": 0.3,
        "volume_clip_lo": 0.8,
        "volume_clip_hi": 1.2,
    }
    defaults.update(overrides)
    return RegimeParams(**defaults)


def default_strategy_params(**overrides: Any) -> StrategyParams:
    """Test defaults for StrategyParams."""
    defaults: dict[str, Any] = {
        "default_variant": "baseline",
        "stop_atr_mult": 1.5,
        "rr_ratio": 2.0,
        "min_stop_distance_pct": 0.3,
        "trend_pullback_rsi_max": 45.0,
        "trend_overbought_rsi_min": 60.0,
        "trend_breakout_adx_min": 30.0,
        "mean_rsi_oversold": 35.0,
        "mean_rsi_overbought": 70.0,
        "vol_extreme_oversold": 25.0,
        "vol_extreme_overbought": 78.0,
        "min_volume_ratio": 1.0,
        "trend_volatile_ema_enabled": True,
        "partial_tp_enabled": False,
        "partial_tp1_rr": 1.0,
        "partial_tp1_fraction": 0.5,
        "mean_revert_use_bb_mid_tp": True,
        "mean_revert_disable_breakeven": True,
        "mean_revert_disable_ema_filter": False,
        "mean_revert_max_bb_width": 0.0,
        "macd_filter_enabled": False,
        "confluence_filter_enabled": False,
        "min_confluence_score": 50.0,
        "regime_exit_profile": RegimeExitProfile.DEFAULT,
        "adaptive_volatile_stop_scale": 0.7,
        "adaptive_volatile_rr_mult": 1.5,
        "adaptive_trending_trail_scale": 1.3,
        "adaptive_volatile_max_bars": 8,
        "adaptive_ranging_max_bars": 12,
        "confluence_rsi_midpoint": 50.0,
        "confluence_rsi_range": 20.0,
        "confluence_volume_baseline": 1.0,
        "confluence_volume_multiplier": 40.0,
        "confluence_adx_baseline": 20.0,
        "confluence_adx_range": 20.0,
        "partial_tp_stop_ratio": 0.5,
    }
    defaults.update(overrides)
    return StrategyParams(**defaults)


def default_risk_params(**overrides: Any) -> RiskParams:
    """Test defaults for RiskParams."""
    defaults: dict[str, Any] = {
        "max_daily_loss_pct": 5.0,
        "max_open_positions": 8,
        "min_rr_ratio": 1.3,
        "risk_per_trade_pct": 1.0,
        "min_notional_usd": 10.0,
        "max_notional_pct_of_equity": 10.0,
        "max_notional_usd": 500_000.0,
        "max_loss_per_trade_pct": 5.0,
        "max_portfolio_risk_pct": 5.0,
        "trade_cooldown_sec": 300,
        "max_consecutive_losses": 5,
        "pair_win_rate_floor": 0.0,
        "max_sector_exposure": 3,
        "max_risk_inflation_mult": 2.0,
        "drawdown_halt_pct": 30.0,
        "drawdown_risk_scale_enabled": False,
        "drawdown_risk_scale_max_dd": 30.0,
        "drawdown_risk_scale_floor": 0.25,
        "equity_curve_filter_enabled": False,
        "equity_curve_filter_period": 20,
        "portfolio_risk_baseline_pairs": 2,
    }
    defaults.update(overrides)
    return RiskParams(**defaults)


def default_system_settings(**overrides: Any) -> SystemSettings:
    """Test defaults for SystemSettings."""
    defaults: dict[str, Any] = {
        "tick_interval_sec": 90,
        "max_consecutive_errors": 5,
        "missing_bar_policy": MissingBarPolicy.SKIP,
        "recon_degraded_timeout_sec": 300,
        "recon_uncertain_timeout_sec": 900,
    }
    defaults.update(overrides)
    return SystemSettings(**defaults)


def default_trading_settings(**overrides: Any) -> TradingSettings:
    """Test defaults for TradingSettings."""
    defaults: dict[str, Any] = {
        "primary_pair": "BTC/USDC",
        "active_pairs": ("BTC/USDC", "ETH/USDC"),
        "candle_interval": "1h",
        "candle_lookback": 60,
        "rsi_period": 14,
        "ema_fast_period": 12,
        "ema_slow_period": 26,
        "ema_base_period": 50,
        "ema_trend_period": 200,
        "atr_period": 14,
        "adx_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "volume_ema_period": 20,
    }
    defaults.update(overrides)
    return TradingSettings(**defaults)


def default_regime_settings(**overrides: Any) -> RegimeSettings:
    """Test defaults for RegimeSettings."""
    defaults: dict[str, Any] = {
        "adx_trend_min": 20.0,
        "adx_strong_trend_min": 35.0,
        "ema_spread_weak_bps": 8.0,
        "ema_spread_strong_bps": 25.0,
        "atr_low_pct": 0.30,
        "atr_high_pct": 0.90,
        "atr_extreme_pct": 1.50,
        "min_confidence": 0.55,
        "truth_trend_return_pct": 0.40,
        "truth_volatile_return_pct": 1.00,
        "trend_weight": 0.4,
        "spread_weight": 0.3,
        "vol_weight": 0.3,
        "volume_clip_lo": 0.8,
        "volume_clip_hi": 1.2,
        "hysteresis_bars": 2,
    }
    defaults.update(overrides)
    return RegimeSettings(**defaults)


def default_risk_settings(**overrides: Any) -> RiskSettings:
    """Test defaults for RiskSettings."""
    defaults: dict[str, Any] = {
        "max_daily_loss_pct": 5.0,
        "max_open_positions": 8,
        "min_rr_ratio": 1.3,
        "risk_per_trade_pct": 1.0,
        "min_notional_usd": 10.0,
        "max_notional_pct_of_equity": 10.0,
        "max_notional_usd": 500_000.0,
        "max_loss_per_trade_pct": 5.0,
        "max_portfolio_risk_pct": 5.0,
        "trade_cooldown_sec": 300,
        "max_consecutive_losses": 5,
        "pair_win_rate_floor": 0.0,
        "max_sector_exposure": 3,
        "max_risk_inflation_mult": 2.0,
        "drawdown_halt_pct": 30.0,
        "drawdown_risk_scale_enabled": False,
        "drawdown_risk_scale_max_dd": 30.0,
        "drawdown_risk_scale_floor": 0.25,
        "equity_curve_filter_enabled": False,
        "equity_curve_filter_period": 20,
        "portfolio_risk_baseline_pairs": 2,
        "daily_loss_precedence": 10,
        "max_positions_precedence": 20,
        "zero_stop_precedence": 30,
        "min_rr_precedence": 40,
        "daily_loss_severity": 0.9,
        "max_positions_severity": 0.7,
        "zero_stop_severity": 0.6,
        "min_rr_severity": 0.5,
    }
    defaults.update(overrides)
    return RiskSettings(**defaults)


def default_ai_settings(**overrides: Any) -> AISettings:
    """Test defaults for AISettings."""
    defaults: dict[str, Any] = {
        "enabled": True,
        "veto_enabled": True,
        "fail_open_on_error": True,
        "regime_enabled": False,
        "regime_fail_open_on_error": True,
        "regime_agreement_boost": 1.25,
        "regime_disagreement_penalty": 0.6,
        "veto_model": "claude-sonnet-4-6",
        "regime_model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_retries": 3,
        "timeout_sec": 10.0,
        "daily_budget_usd": 1.0,
        "veto_confidence_threshold": 0.85,
        "input_cost_per_million": 3.0,
        "output_cost_per_million": 15.0,
        "max_response_tokens": 200,
        "batch_timeout_sec": 30.0,
    }
    defaults.update(overrides)
    return AISettings(**defaults)


def default_backtest_settings(**overrides: Any) -> BacktestSettings:
    """Test defaults for BacktestSettings."""
    defaults: dict[str, Any] = {
        "fee_bps": 4.0,
        "slippage_bps": 2.0,
        "fee_multiplier": 2.0,
        "leverage": 1.0,
        "funding_rate_per_bar": 0.0,
        "impact_bps": 0.0,
        "max_volume_pct": 0.1,
        "equity_usd": 10_000.0,
        "warmup_bars": 200,
        "entry_price_model": EntryPriceModel.CLOSE,
        "partial_fill_enabled": False,
        "partial_fill_threshold_pct": 0.05,
        "partial_fill_min_ratio": 0.1,
        "maintenance_margin_rate": 0.0,
        "simulated_execution": False,
        "use_candle_cache": True,
    }
    defaults.update(overrides)
    return BacktestSettings(**defaults)


def default_exchange_settings(**overrides: Any) -> ExchangeSettings:
    """Test defaults for ExchangeSettings."""
    defaults: dict[str, Any] = {
        "api_key_env": "BINANCE_API_KEY",
        "api_secret_env": "BINANCE_API_SECRET",
        "venue": VenueCode("binance"),
        "product": ProductCode("usd_c"),
        "position_mode": PositionMode.ONE_WAY,
        "testnet": True,
        "ws_enabled": True,
        "rest_fallback_enabled": True,
        "recv_window_ms": 5000,
        "connect_timeout_sec": 10.0,
        "read_timeout_sec": 10.0,
        "retry_max_attempts": 3,
        "retry_base_delay_sec": 0.5,
        "backoff_factor": 2.0,
        "rate_limit_per_sec": 10,
    }
    defaults.update(overrides)
    return ExchangeSettings(**defaults)


def default_optimization_settings(**overrides: Any) -> OptimizationSettings:
    """Test defaults for OptimizationSettings."""
    defaults: dict[str, Any] = {
        "enabled": False,
        "study_name": "dojiwick_v2",
        "storage_url": "",
        "trials": 100,
        "trial_timeout_sec": 60.0,
        "objective_return_weight": 1.0,
        "objective_sharpe_weight": 1.0,
        "objective_win_rate_weight": 0.5,
        "objective_drawdown_penalty": 2.0,
        "objective_profit_factor_weight": 0.0,
        "objective_regularization_strength": 1.0,
        "objective_expectancy_weight": 0.0,
        "objective_consecutive_loss_penalty": 0.0,
        "objective_min_trades": 0,
        "objective_max_drawdown_threshold": 25.0,
        "objective_drawdown_cliff_penalty": 50.0,
        "objective_trade_freq_weight": 0.3,
        "objective_min_density_threshold": 0.005,
        "objective_high_winrate_bonus": 10.0,
        "objective_high_winrate_threshold": 0.55,
        "objective_payoff_ratio_weight": 2.0,
        "objective_payoff_ratio_cap": 3.0,
        "multivariate_sampler": True,
        "constant_liar": True,
        "pruning_enabled": True,
        "pruning_percentile": 25,
        "pruning_startup_trials": 30,
        "train_fraction": 0.7,
        "objective_mode": ObjectiveMode.IS_OOS,
        "objective_cv_folds": 5,
        "objective_consistency_penalty": 0.5,
        "objective_min_trades_penalty": -20.0,
        "objective_min_trades_penalty_start": -2.0,
        "objective_profit_factor_cap": 5.0,
        "objective_consecutive_loss_threshold": 3,
        "objective_pnl_weight": 0.5,
        "objective_pnl_cap": 3.0,
    }
    defaults.update(overrides)
    return OptimizationSettings(**defaults)


def _default_targets() -> tuple[TargetConfig, ...]:
    """Standard two-pair targets for tests."""
    return (
        TargetConfig(
            target_id="btc_usdc",
            display_pair="BTC/USDC",
            execution_instrument="BTCUSDC",
            market_data_instrument="BTCUSDC",
        ),
        TargetConfig(
            target_id="eth_usdc",
            display_pair="ETH/USDC",
            execution_instrument="ETHUSDC",
            market_data_instrument="ETHUSDC",
        ),
    )


def default_universe_settings(**overrides: Any) -> UniverseSettings:
    """Test defaults for UniverseSettings."""
    defaults: dict[str, Any] = {
        "account": "default",
        "quote_asset": "USDC",
        "settle_asset": "USDC",
        "targets": _default_targets(),
    }
    defaults.update(overrides)
    return UniverseSettings(**defaults)


def default_instrument_map(settings: Settings | None = None) -> dict[str, Any]:
    """Build a test instrument_map from settings targets."""
    from dojiwick.config.targets import resolve_instrument_map

    if settings is None:
        settings = default_settings()
    return resolve_instrument_map(settings)


def default_adaptive_settings(**overrides: Any) -> AdaptiveSettings:
    """Test defaults for AdaptiveSettings."""
    defaults: dict[str, Any] = {
        "mode": AdaptiveMode.DISABLED,
        "exploration_rate": 0.1,
        "min_samples": 30,
    }
    defaults.update(overrides)
    return AdaptiveSettings(**defaults)


def default_research_gate_settings(**overrides: Any) -> ResearchGateSettings:
    """Test defaults for ResearchGateSettings."""
    defaults: dict[str, Any] = {
        "enabled": False,
        "min_cv_sharpe": 0.3,
        "max_pbo": 0.5,
        "min_oos_degradation_ratio": 0.5,
        "cv_folds": 5,
        "purge_bars": 5,
        "embargo_bars": 3,
        "wf_train_size": 200,
        "wf_test_size": 50,
        "wf_expanding": False,
        "wf_mode": WFMode.RATIO,
        "min_wf_oos_sharpe": 0.0,
        "wf_min_trades": 0,
        "continuous_validation_enabled": False,
        "min_continuous_trades_per_1000_bars": 5,
        "max_continuous_drawdown_pct": 30.0,
        "pbo_min_trade_returns": 8,
        "pbo_max_partitions": 16,
        "shock_test_enabled": False,
        "shock_test_tp_shift_pct": -10.0,
        "shock_test_sl_shift_pct": 10.0,
        "shock_test_min_pf": 1.05,
        "per_regime_pf_enabled": False,
        "per_regime_min_pf": 0.95,
        "per_regime_min_trades": 30,
        "concentration_check_enabled": False,
        "concentration_max_month_pct": 40.0,
        "concentration_max_trade_pct": 15.0,
        "pair_robustness_enabled": False,
        "pair_robustness_min_pairs": 3,
        "pair_robustness_min_pf_threshold": 1.0,
    }
    defaults.update(overrides)
    return ResearchGateSettings(**defaults)


class PerformanceSnapshotBuilder:
    """Fluent builder for PerformanceSnapshot."""

    def __init__(self) -> None:
        self._observed_at = datetime(2026, 1, 1, tzinfo=UTC)
        self._equity_usd = Decimal("10000")
        self._unrealized_pnl_usd = Decimal(0)
        self._realized_pnl_usd = Decimal(0)
        self._open_positions = 0
        self._drawdown_pct = Decimal(0)

    def with_equity(self, equity: Decimal | str | int | float) -> Self:
        self._equity_usd = to_money(equity)
        return self

    def with_drawdown(self, pct: Decimal | str | int | float) -> Self:
        self._drawdown_pct = Decimal(str(pct)) if not isinstance(pct, Decimal) else pct
        return self

    def with_observed_at(self, at: datetime) -> Self:
        self._observed_at = at
        return self

    def build(self) -> PerformanceSnapshot:
        return PerformanceSnapshot(
            observed_at=self._observed_at,
            equity_usd=self._equity_usd,
            unrealized_pnl_usd=self._unrealized_pnl_usd,
            realized_pnl_usd=self._realized_pnl_usd,
            open_positions=self._open_positions,
            drawdown_pct=self._drawdown_pct,
        )


def default_settings(**overrides: Any) -> Settings:
    """Create Settings with all sections for test use.

    Production code must load from TOML via load_settings().
    """
    defaults: dict[str, Any] = {
        "system": default_system_settings(),
        "trading": default_trading_settings(),
        "regime": default_regime_settings(),
        "strategy": default_strategy_params(),
        "risk": default_risk_settings(),
        "ai": default_ai_settings(),
        "backtest": default_backtest_settings(),
        "exchange": default_exchange_settings(),
        "optimization": default_optimization_settings(),
        "universe": default_universe_settings(),
        "adaptive": default_adaptive_settings(),
        "research": default_research_gate_settings(),
    }
    defaults.update(overrides)
    return Settings(**defaults)


class SettingsBuilder:
    """Convenience wrapper for creating Settings with overrides."""

    def __init__(self) -> None:
        self._settings = default_settings()

    def with_active_pairs(self, pairs: tuple[str, ...]) -> Self:
        primary = pairs[0] if pairs else self._settings.trading.primary_pair
        # Rebuild matching targets for new pairs
        targets = tuple(
            TargetConfig(
                target_id=pair.replace("/", "_").lower(),
                display_pair=pair,
                execution_instrument=pair.replace("/", ""),
                market_data_instrument=pair.replace("/", ""),
            )
            for pair in pairs
        )
        self._settings = self._settings.model_copy(
            update={
                "trading": self._settings.trading.model_copy(update={"active_pairs": pairs, "primary_pair": primary}),
                "universe": self._settings.universe.model_copy(update={"targets": targets}),
            }
        )
        return self

    def with_max_daily_loss_pct(self, pct: float) -> Self:
        self._settings = self._settings.model_copy(
            update={"risk": self._settings.risk.model_copy(update={"max_daily_loss_pct": pct})}
        )
        return self

    def with_ai_regime(
        self,
        *,
        enabled: bool = True,
        fail_open_on_error: bool = True,
        agreement_boost: float = 1.25,
        disagreement_penalty: float = 0.6,
    ) -> Self:
        self._settings = self._settings.model_copy(
            update={
                "ai": self._settings.ai.model_copy(
                    update={
                        "regime_enabled": enabled,
                        "regime_fail_open_on_error": fail_open_on_error,
                        "regime_agreement_boost": agreement_boost,
                        "regime_disagreement_penalty": disagreement_penalty,
                    }
                )
            }
        )
        return self

    def with_ai_veto(
        self,
        *,
        enabled: bool = True,
        veto_enabled: bool = True,
        fail_open_on_error: bool = True,
        veto_model: str = "claude-sonnet-4-6",
        daily_budget_usd: float = 1.0,
        veto_confidence_threshold: float = 0.85,
    ) -> Self:
        self._settings = self._settings.model_copy(
            update={
                "ai": self._settings.ai.model_copy(
                    update={
                        "enabled": enabled,
                        "veto_enabled": veto_enabled,
                        "fail_open_on_error": fail_open_on_error,
                        "veto_model": veto_model,
                        "daily_budget_usd": daily_budget_usd,
                        "veto_confidence_threshold": veto_confidence_threshold,
                    }
                )
            }
        )
        return self

    def with_regime_weights(self, trend: float, spread: float, vol: float) -> Self:
        self._settings = self._settings.model_copy(
            update={
                "regime": self._settings.regime.model_copy(
                    update={"trend_weight": trend, "spread_weight": spread, "vol_weight": vol}
                )
            }
        )
        return self

    def with_risk_precedence(self, daily_loss: int, max_positions: int, zero_stop: int, min_rr: int) -> Self:
        self._settings = self._settings.model_copy(
            update={
                "risk": self._settings.risk.model_copy(
                    update={
                        "daily_loss_precedence": daily_loss,
                        "max_positions_precedence": max_positions,
                        "zero_stop_precedence": zero_stop,
                        "min_rr_precedence": min_rr,
                    }
                )
            }
        )
        return self

    def with_risk_severity(self, daily_loss: float, max_positions: float, zero_stop: float, min_rr: float) -> Self:
        self._settings = self._settings.model_copy(
            update={
                "risk": self._settings.risk.model_copy(
                    update={
                        "daily_loss_severity": daily_loss,
                        "max_positions_severity": max_positions,
                        "zero_stop_severity": zero_stop,
                        "min_rr_severity": min_rr,
                    }
                )
            }
        )
        return self

    def with_ai_pricing(self, input_cost: float, output_cost: float, max_tokens: int) -> Self:
        self._settings = self._settings.model_copy(
            update={
                "ai": self._settings.ai.model_copy(
                    update={
                        "input_cost_per_million": input_cost,
                        "output_cost_per_million": output_cost,
                        "max_response_tokens": max_tokens,
                    }
                )
            }
        )
        return self

    def with_system_errors(self, max_consecutive: int, missing_bar_policy: str) -> Self:
        self._settings = self._settings.model_copy(
            update={
                "system": self._settings.system.model_copy(
                    update={
                        "max_consecutive_errors": max_consecutive,
                        "missing_bar_policy": missing_bar_policy,
                    }
                )
            }
        )
        return self

    def with_backtest_fees(self, fee_multiplier: float) -> Self:
        self._settings = self._settings.model_copy(
            update={"backtest": self._settings.backtest.model_copy(update={"fee_multiplier": fee_multiplier})}
        )
        return self

    def with_reconciliation_settings(
        self,
        interval_ticks: int = 10,
        degraded_timeout_sec: int = 300,
        uncertain_timeout_sec: int = 900,
    ) -> Self:
        self._settings = self._settings.model_copy(
            update={
                "system": self._settings.system.model_copy(
                    update={
                        "reconciliation_interval_ticks": interval_ticks,
                        "recon_degraded_timeout_sec": degraded_timeout_sec,
                        "recon_uncertain_timeout_sec": uncertain_timeout_sec,
                    }
                )
            }
        )
        return self

    def with_flags(self, **kwargs: object) -> Self:
        self._settings = self._settings.model_copy(update={"flags": self._settings.flags.model_copy(update=kwargs)})
        return self

    def with_global_halt(self) -> Self:
        return self.with_flags(global_halt=True)

    def with_disable_llm(self) -> Self:
        return self.with_flags(disable_llm=True)

    def with_exits_only(self) -> Self:
        return self.with_flags(exits_only_mode=True)

    def with_shadow_veto(self) -> Self:
        return self.with_flags(ai_veto_shadow_mode=True)

    def with_shadow_regime(self) -> Self:
        return self.with_flags(ai_regime_shadow_mode=True)

    def with_halted_pairs(self, pairs: tuple[str, ...]) -> Self:
        return self.with_flags(halted_pairs=pairs)

    def with_research(self, **kwargs: object) -> Self:
        self._settings = self._settings.model_copy(
            update={"research": self._settings.research.model_copy(update=kwargs)}
        )
        return self

    def build(self) -> Settings:
        return self._settings
