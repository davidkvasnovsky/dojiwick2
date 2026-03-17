"""Typed settings schema."""

from collections.abc import Mapping
from typing import Annotated, Any, Self, cast

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from dojiwick.domain.enums import (
    AdaptiveMode,
    BacktestGapPolicy,
    BenchmarkMode,
    EntryPriceModel,
    HistoryAlignment,
    MissingBarPolicy,
    ObjectiveMode,
    PositionMode,
    WFMode,
)
from dojiwick.domain.type_aliases import ProductCode, VenueCode
from dojiwick.domain.errors import ConfigurationError
from dojiwick.domain.models.value_objects.cost_model import CostModel
from dojiwick.domain.models.value_objects.params import RegimeParams, RiskParams, StrategyParams

from .risk_scope import RiskScopeResolver
from .scope import StrategyScopeResolver


def _list_to_tuple(v: object) -> object:
    if isinstance(v, list):
        return tuple(str(item) for item in cast(list[object], v))
    return v


_StrTuple = Annotated[tuple[str, ...], BeforeValidator(_list_to_tuple)]


class SystemSettings(BaseModel):
    """Runtime-level settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tick_interval_sec: int
    log_level: str = "INFO"
    shutdown_timeout_sec: int = 30
    max_ticks: int = 0
    max_consecutive_errors: int
    missing_bar_policy: MissingBarPolicy
    reconciliation_interval_ticks: int = 10
    recon_degraded_timeout_sec: int
    recon_uncertain_timeout_sec: int

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.tick_interval_sec < 1:
            raise ValueError("tick_interval_sec must be >= 1")
        if self.log_level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"invalid log_level: {self.log_level}")
        if self.shutdown_timeout_sec < 1:
            raise ValueError("shutdown_timeout_sec must be >= 1")
        if self.max_ticks < 0:
            raise ValueError("max_ticks must be >= 0")
        if self.max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be >= 1")
        if self.reconciliation_interval_ticks < 1:
            raise ValueError("reconciliation_interval_ticks must be >= 1")
        if self.recon_degraded_timeout_sec < 1:
            raise ValueError("recon_degraded_timeout_sec must be >= 1")
        if self.recon_uncertain_timeout_sec <= self.recon_degraded_timeout_sec:
            raise ValueError("recon_uncertain_timeout_sec must be > recon_degraded_timeout_sec")
        return self


class TradingSettings(BaseModel):
    """Pair selection settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary_pair: str
    active_pairs: _StrTuple
    enabled_strategies: Annotated[tuple[str, ...] | None, BeforeValidator(_list_to_tuple)] = None
    candle_interval: str
    candle_lookback: int
    rsi_period: int
    ema_fast_period: int
    ema_slow_period: int
    ema_base_period: int
    ema_trend_period: int
    atr_period: int
    adx_period: int
    bb_period: int
    bb_std: float
    volume_ema_period: int

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.primary_pair:
            raise ValueError("primary_pair must not be empty")
        if not self.active_pairs:
            raise ValueError("active_pairs must not be empty")
        return self


class RegimeSettings(BaseModel):
    """Regime settings with service-level fields on top of kernel params."""

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
    hysteresis_bars: int

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
        if self.hysteresis_bars < 1:
            raise ValueError("hysteresis_bars must be >= 1")
        return self

    @property
    def params(self) -> RegimeParams:
        d = vars(self)
        cached = d.get("_cached_params")
        if cached is not None:
            return cached
        result = RegimeParams(
            adx_trend_min=self.adx_trend_min,
            adx_strong_trend_min=self.adx_strong_trend_min,
            ema_spread_weak_bps=self.ema_spread_weak_bps,
            ema_spread_strong_bps=self.ema_spread_strong_bps,
            atr_low_pct=self.atr_low_pct,
            atr_high_pct=self.atr_high_pct,
            atr_extreme_pct=self.atr_extreme_pct,
            min_confidence=self.min_confidence,
            truth_trend_return_pct=self.truth_trend_return_pct,
            truth_volatile_return_pct=self.truth_volatile_return_pct,
            trend_weight=self.trend_weight,
            spread_weight=self.spread_weight,
            vol_weight=self.vol_weight,
            volume_clip_lo=self.volume_clip_lo,
            volume_clip_hi=self.volume_clip_hi,
        )
        d["_cached_params"] = result
        return result

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        vars(copied).pop("_cached_params", None)
        return copied


class RiskSettings(BaseModel):
    """Risk config with rule precedence and severity tuning."""

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
    daily_loss_precedence: int
    max_positions_precedence: int
    zero_stop_precedence: int
    min_rr_precedence: int
    daily_loss_severity: float
    max_positions_severity: float
    zero_stop_severity: float
    min_rr_severity: float

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
        for name in ("daily_loss_precedence", "max_positions_precedence", "zero_stop_precedence", "min_rr_precedence"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        for name in ("daily_loss_severity", "max_positions_severity", "zero_stop_severity", "min_rr_severity"):
            val = getattr(self, name)
            if not (0.0 < val <= 1.0):
                raise ValueError(f"{name} must be in (0, 1]")
        return self

    @property
    def params(self) -> RiskParams:
        d = vars(self)
        cached = d.get("_cached_params")
        if cached is not None:
            return cached
        result = RiskParams(
            max_daily_loss_pct=self.max_daily_loss_pct,
            max_open_positions=self.max_open_positions,
            min_rr_ratio=self.min_rr_ratio,
            risk_per_trade_pct=self.risk_per_trade_pct,
            min_notional_usd=self.min_notional_usd,
            max_notional_pct_of_equity=self.max_notional_pct_of_equity,
            max_notional_usd=self.max_notional_usd,
            max_loss_per_trade_pct=self.max_loss_per_trade_pct,
            max_portfolio_risk_pct=self.max_portfolio_risk_pct,
            trade_cooldown_sec=self.trade_cooldown_sec,
            max_consecutive_losses=self.max_consecutive_losses,
            pair_win_rate_floor=self.pair_win_rate_floor,
            max_sector_exposure=self.max_sector_exposure,
            max_risk_inflation_mult=self.max_risk_inflation_mult,
            drawdown_halt_pct=self.drawdown_halt_pct,
            drawdown_risk_scale_enabled=self.drawdown_risk_scale_enabled,
            drawdown_risk_scale_max_dd=self.drawdown_risk_scale_max_dd,
            drawdown_risk_scale_floor=self.drawdown_risk_scale_floor,
            equity_curve_filter_enabled=self.equity_curve_filter_enabled,
            equity_curve_filter_period=self.equity_curve_filter_period,
            portfolio_risk_baseline_pairs=self.portfolio_risk_baseline_pairs,
        )
        d["_cached_params"] = result
        return result

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        vars(copied).pop("_cached_params", None)
        return copied


class AISettings(BaseModel):
    """AI veto and regime classifier settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    veto_enabled: bool
    fail_open_on_error: bool
    regime_enabled: bool
    regime_fail_open_on_error: bool
    regime_agreement_boost: float
    regime_disagreement_penalty: float
    veto_model: str
    regime_model: str
    api_key_env: str
    max_retries: int
    timeout_sec: float
    daily_budget_usd: float
    veto_confidence_threshold: float
    input_cost_per_million: float
    output_cost_per_million: float
    max_response_tokens: int
    batch_timeout_sec: float

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.enabled and self.veto_enabled and not self.veto_model:
            raise ConfigurationError("ai.veto_model must be set when ai.enabled and ai.veto_enabled")
        if self.enabled and self.regime_enabled and not self.regime_model:
            raise ConfigurationError("ai.regime_model must be set when ai.enabled and ai.regime_enabled")
        if self.regime_agreement_boost < 1.0:
            raise ValueError("ai.regime_agreement_boost must be >= 1.0")
        if not (0.0 < self.regime_disagreement_penalty <= 1.0):
            raise ValueError("ai.regime_disagreement_penalty must be in (0, 1]")
        if self.max_retries < 0:
            raise ValueError("ai.max_retries must be >= 0")
        if self.timeout_sec <= 0:
            raise ValueError("ai.timeout_sec must be > 0")
        if self.daily_budget_usd <= 0:
            raise ValueError("ai.daily_budget_usd must be > 0")
        if not (0.0 < self.veto_confidence_threshold <= 1.0):
            raise ValueError("ai.veto_confidence_threshold must be in (0, 1]")
        if self.input_cost_per_million <= 0:
            raise ValueError("ai.input_cost_per_million must be > 0")
        if self.output_cost_per_million <= 0:
            raise ValueError("ai.output_cost_per_million must be > 0")
        if self.max_response_tokens < 1:
            raise ValueError("ai.max_response_tokens must be >= 1")
        if self.batch_timeout_sec <= 0:
            raise ValueError("ai.batch_timeout_sec must be > 0")
        return self


class BacktestSettings(BaseModel):
    """Backtest execution settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fee_bps: float
    slippage_bps: float
    fee_multiplier: float
    leverage: float
    funding_rate_per_bar: float
    impact_bps: float
    max_volume_pct: float
    equity_usd: float
    warmup_bars: int
    entry_price_model: EntryPriceModel
    partial_fill_enabled: bool
    partial_fill_threshold_pct: float
    partial_fill_min_ratio: float
    maintenance_margin_rate: float
    simulated_execution: bool
    use_candle_cache: bool
    history_alignment: HistoryAlignment
    inactive_gap_policy: BacktestGapPolicy
    benchmark_mode: BenchmarkMode

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be >= 0")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")
        if self.fee_multiplier <= 0:
            raise ValueError("fee_multiplier must be > 0")
        if self.leverage < 1.0:
            raise ValueError("leverage must be >= 1.0")
        if self.funding_rate_per_bar < 0:
            raise ValueError("funding_rate_per_bar must be >= 0")
        if self.impact_bps < 0:
            raise ValueError("impact_bps must be >= 0")
        if not (0.0 < self.max_volume_pct <= 1.0):
            raise ValueError("max_volume_pct must be in (0, 1]")
        if self.equity_usd <= 0:
            raise ValueError("equity_usd must be > 0")
        if self.warmup_bars < 1:
            raise ValueError("warmup_bars must be >= 1")
        if not (0.0 < self.partial_fill_threshold_pct <= 1.0):
            raise ValueError("partial_fill_threshold_pct must be in (0, 1]")
        if not (0.0 < self.partial_fill_min_ratio <= 1.0):
            raise ValueError("partial_fill_min_ratio must be in (0, 1]")
        if self.maintenance_margin_rate < 0:
            raise ValueError("maintenance_margin_rate must be >= 0")
        if self.leverage > 1.0 and self.maintenance_margin_rate >= 1.0 / self.leverage:
            raise ValueError("maintenance_margin_rate must be < 1/leverage (initial margin rate)")
        if (
            self.history_alignment == HistoryAlignment.ROLLING_JOINED
            and self.benchmark_mode == BenchmarkMode.STATIC_FULL_WINDOW
        ):
            raise ValueError(
                "benchmark_mode must be 'rolling_joined' when history_alignment is 'rolling_joined' "
                "(static benchmark is invalid with zero-padded initial prices)"
            )
        return self

    @property
    def cost_model(self) -> CostModel:
        return CostModel(
            fee_bps=self.fee_bps,
            fee_multiplier=self.fee_multiplier,
            slippage_bps=self.slippage_bps,
            impact_bps=self.impact_bps,
            funding_rate_per_bar=self.funding_rate_per_bar,
            leverage=self.leverage,
            maintenance_margin_rate=self.maintenance_margin_rate,
        )


class DatabaseSettings(BaseModel):
    """PostgreSQL connection settings for persistence adapters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dsn: str = "postgresql://dojiwick:dojiwick@postgres:5432/dojiwick"
    app_name: str = "dojiwick"
    connect_timeout_sec: float = 5.0
    statement_timeout_ms: int = 5_000
    min_connections: int = 2
    max_connections: int = 5

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.dsn:
            raise ValueError("database.dsn must not be empty")
        if self.connect_timeout_sec <= 0:
            raise ValueError("database.connect_timeout_sec must be > 0")
        if self.statement_timeout_ms < 100:
            raise ValueError("database.statement_timeout_ms must be >= 100")
        if self.min_connections < 2:
            raise ValueError("database.min_connections must be >= 2")
        if self.max_connections < self.min_connections:
            raise ValueError("database.max_connections must be >= min_connections")
        return self


class OptimizationSettings(BaseModel):
    """Optimization settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    study_name: str
    storage_url: str
    trials: int
    trial_timeout_sec: float
    objective_return_weight: float
    objective_sharpe_weight: float
    objective_win_rate_weight: float
    objective_drawdown_penalty: float
    objective_profit_factor_weight: float
    objective_regularization_strength: float
    objective_expectancy_weight: float
    objective_consecutive_loss_penalty: float
    objective_min_trades: int
    # Drawdown cliff
    objective_max_drawdown_threshold: float
    objective_drawdown_cliff_penalty: float
    # Trade frequency bonus
    objective_trade_freq_weight: float
    # Trade density penalty
    objective_min_density_threshold: float
    # Win rate bonus
    objective_high_winrate_bonus: float
    objective_high_winrate_threshold: float
    # Payoff ratio reward
    objective_payoff_ratio_weight: float
    objective_payoff_ratio_cap: float
    # TPE sampler settings
    multivariate_sampler: bool
    constant_liar: bool
    sampler_seed: int | None = None
    # Pruning settings
    pruning_enabled: bool
    pruning_percentile: int
    pruning_startup_trials: int
    # IS/OOS split
    train_fraction: float
    # Objective mode
    objective_mode: ObjectiveMode
    objective_cv_folds: int
    objective_consistency_penalty: float
    # Objective tuning
    objective_min_trades_penalty: float
    objective_min_trades_penalty_start: float
    objective_profit_factor_cap: float
    objective_consecutive_loss_threshold: int
    # PnL return term weight and cap
    objective_pnl_weight: float
    objective_pnl_cap: float

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.trials < 1:
            raise ValueError("trials must be >= 1")
        if not (0.0 < self.train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1)")
        if self.trial_timeout_sec <= 0:
            raise ValueError("trial_timeout_sec must be > 0")
        if self.objective_min_trades < 0:
            raise ValueError("objective_min_trades must be >= 0")
        if self.objective_expectancy_weight < 0:
            raise ValueError("objective_expectancy_weight must be >= 0")
        if self.objective_consecutive_loss_penalty < 0:
            raise ValueError("objective_consecutive_loss_penalty must be >= 0")
        if not (0 < self.objective_max_drawdown_threshold <= 100):
            raise ValueError("objective_max_drawdown_threshold must be in (0, 100]")
        if self.objective_drawdown_cliff_penalty < 0:
            raise ValueError("objective_drawdown_cliff_penalty must be >= 0")
        if self.objective_high_winrate_threshold < 0 or self.objective_high_winrate_threshold > 1:
            raise ValueError("objective_high_winrate_threshold must be in [0, 1]")
        if not (1 <= self.pruning_percentile <= 99):
            raise ValueError("pruning_percentile must be in [1, 99]")
        if self.pruning_startup_trials < 0:
            raise ValueError("pruning_startup_trials must be >= 0")
        if self.objective_cv_folds < 2:
            raise ValueError("objective_cv_folds must be >= 2")
        if self.objective_consistency_penalty < 0:
            raise ValueError("objective_consistency_penalty must be >= 0")
        if self.objective_min_trades_penalty >= self.objective_min_trades_penalty_start:
            raise ValueError("objective_min_trades_penalty must be < objective_min_trades_penalty_start")
        return self


class ExchangeSettings(BaseModel):
    """Exchange connection settings (config-driven adapter selection)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key_env: str
    api_secret_env: str
    venue: VenueCode
    product: ProductCode
    position_mode: PositionMode
    testnet: bool
    ws_enabled: bool
    rest_fallback_enabled: bool
    recv_window_ms: int
    connect_timeout_sec: float
    read_timeout_sec: float
    retry_max_attempts: int
    retry_base_delay_sec: float
    backoff_factor: float
    rate_limit_per_sec: int

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.recv_window_ms < 1000:
            raise ValueError("exchange.recv_window_ms must be >= 1000")
        if self.connect_timeout_sec <= 0:
            raise ValueError("exchange.connect_timeout_sec must be > 0")
        if self.read_timeout_sec <= 0:
            raise ValueError("exchange.read_timeout_sec must be > 0")
        if self.retry_max_attempts < 0:
            raise ValueError("exchange.retry_max_attempts must be >= 0")
        if self.retry_base_delay_sec < 0:
            raise ValueError("exchange.retry_base_delay_sec must be >= 0")
        if self.backoff_factor < 1.0:
            raise ValueError("exchange.backoff_factor must be >= 1.0")
        if self.rate_limit_per_sec < 1:
            raise ValueError("exchange.rate_limit_per_sec must be >= 1")
        return self


class TargetConfig(BaseModel):
    """Per-pair instrument mapping for multi-venue support.

    ``target_id`` is the stable identity key used in persistence, scope resolution,
    and all internal lookups. ``execution_instrument`` is the exchange symbol for
    order submission. ``market_data_instrument`` is the exchange symbol for candle fetch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_id: str
    display_pair: str
    execution_instrument: str
    market_data_instrument: str

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.target_id:
            raise ValueError("target_id must be non-empty")
        if not self.execution_instrument:
            raise ValueError("execution_instrument must be non-empty")
        if not self.display_pair:
            raise ValueError("display_pair must be non-empty")
        if not self.market_data_instrument:
            raise ValueError("market_data_instrument must be non-empty")
        return self


class UniverseSettings(BaseModel):
    """Symbol normalization and quote asset preferences."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    account: str
    quote_asset: str
    settle_asset: str
    symbol_separator: str = ""
    pair_separator: str = "/"
    targets: tuple[TargetConfig, ...]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.quote_asset:
            raise ValueError("universe.quote_asset must not be empty")
        if not self.settle_asset:
            raise ValueError("universe.settle_asset must not be empty")
        if not self.targets:
            raise ValueError("universe.targets must be non-empty — declare [[universe.targets]] for every traded pair")
        target_ids = [t.target_id for t in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("universe.targets: target_id must be unique across all targets")
        display_pairs = [t.display_pair for t in self.targets]
        if len(display_pairs) != len(set(display_pairs)):
            raise ValueError("universe.targets: display_pair must be unique across all targets")
        return self


class AdaptiveSettings(BaseModel):
    """Adaptive policy configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: AdaptiveMode
    exploration_rate: float
    min_samples: int

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not 0.0 <= self.exploration_rate <= 1.0:
            raise ValueError("adaptive.exploration_rate must be in [0, 1]")
        if self.min_samples < 1:
            raise ValueError("adaptive.min_samples must be >= 1")
        return self


class FeatureFlags(BaseModel):
    """Runtime feature flags for staged rollout and emergency kill switches."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ai_veto_shadow_mode: bool = False
    ai_regime_shadow_mode: bool = False
    exits_only_mode: bool = False
    global_halt: bool = False
    disable_llm: bool = False
    halted_pairs: _StrTuple = ()


class ResearchGateSettings(BaseModel):
    """Anti-overfitting research gate for strategy validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool
    min_cv_sharpe: float
    max_pbo: float
    min_oos_degradation_ratio: float
    cv_folds: int
    purge_bars: int
    embargo_bars: int
    wf_train_size: int
    wf_test_size: int
    wf_expanding: bool
    wf_mode: WFMode
    min_wf_oos_sharpe: float
    wf_min_trades: int
    continuous_validation_enabled: bool
    min_continuous_trades_per_1000_bars: int
    max_continuous_drawdown_pct: float
    pbo_min_trade_returns: int
    pbo_max_partitions: int
    # Shock test (TP/SL perturbation)
    shock_test_enabled: bool
    shock_test_tp_shift_pct: float
    shock_test_sl_shift_pct: float
    shock_test_min_pf: float
    # Per-regime minimum profit factor
    per_regime_pf_enabled: bool
    per_regime_min_pf: float
    per_regime_min_trades: int
    # Concentration checks
    concentration_check_enabled: bool
    concentration_max_month_pct: float
    concentration_max_trade_pct: float
    # Pair robustness
    pair_robustness_enabled: bool
    pair_robustness_min_pairs: int
    pair_robustness_min_pf_threshold: float

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.min_cv_sharpe < 0:
            raise ValueError("research.min_cv_sharpe must be >= 0")
        if not (0.0 <= self.max_pbo <= 1.0):
            raise ValueError("research.max_pbo must be in [0, 1]")
        if not (0.0 < self.min_oos_degradation_ratio <= 1.0):
            raise ValueError("research.min_oos_degradation_ratio must be in (0, 1]")
        if self.cv_folds < 2:
            raise ValueError("research.cv_folds must be >= 2")
        if self.purge_bars < 0:
            raise ValueError("research.purge_bars must be >= 0")
        if self.embargo_bars < 0:
            raise ValueError("research.embargo_bars must be >= 0")
        if self.wf_train_size < 1:
            raise ValueError("research.wf_train_size must be >= 1")
        if self.wf_test_size < 1:
            raise ValueError("research.wf_test_size must be >= 1")
        if self.min_wf_oos_sharpe < 0:
            raise ValueError("research.min_wf_oos_sharpe must be >= 0")
        if self.wf_min_trades < 0:
            raise ValueError("research.wf_min_trades must be >= 0")
        if not (0.0 < self.concentration_max_month_pct <= 100.0):
            raise ValueError("research.concentration_max_month_pct must be in (0, 100]")
        if not (0.0 < self.concentration_max_trade_pct <= 100.0):
            raise ValueError("research.concentration_max_trade_pct must be in (0, 100]")
        return self


class Settings(BaseModel):
    """Top-level settings bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    # All behavior-bearing sections are mandatory — must be in TOML
    trading: TradingSettings
    regime: RegimeSettings
    strategy: StrategyParams
    risk: RiskSettings
    exchange: ExchangeSettings
    universe: UniverseSettings

    # Behavior-bearing — must be in TOML
    system: SystemSettings
    ai: AISettings
    backtest: BacktestSettings
    optimization: OptimizationSettings
    adaptive: AdaptiveSettings
    research: ResearchGateSettings

    # Infrastructure/tooling — keep defaults
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    flags: FeatureFlags = Field(default_factory=FeatureFlags)
    strategy_scope: StrategyScopeResolver = Field(default_factory=StrategyScopeResolver.empty)
    risk_scope: RiskScopeResolver = Field(default_factory=RiskScopeResolver.empty)

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.trading.primary_pair not in self.trading.active_pairs:
            raise ValueError("primary_pair must be included in active_pairs")
        if self.strategy.rr_ratio < self.risk.min_rr_ratio:
            raise ValueError(
                f"strategy.rr_ratio ({self.strategy.rr_ratio}) must be >= risk.min_rr_ratio ({self.risk.min_rr_ratio})"
            )
        derived = tuple(t.display_pair for t in self.universe.targets)
        if self.trading.active_pairs != derived:
            raise ValueError(
                f"trading.active_pairs {self.trading.active_pairs} must match universe.targets display_pairs {derived}"
            )
        if self.trading.primary_pair != derived[0]:
            raise ValueError(f"trading.primary_pair must be first target display_pair ({derived[0]})")
        return self
