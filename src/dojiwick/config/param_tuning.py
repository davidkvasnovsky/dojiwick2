"""Settings-aware parameter tuning helpers for optimization.

Moved from ``application/use_cases/optimization/objective.py`` to keep
Pydantic ``model_copy`` calls in the config layer (not application).
"""

from dataclasses import replace as dc_replace

from dojiwick.application.use_cases.optimization.search_space import (
    INT_PARAMS,
    NON_STRATEGY_PARAMS,
    ParamSet,
    REGIME_PARAMS,
    REGIME_SCOPE_FIELDS,
    REGIME_SCOPE_PREFIX,
    SearchSpace,
    extract_regularization_baseline,
)
from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.params import StrategyParams

from .risk_scope import RiskScopeResolver, RiskScopeRule
from .schema import RiskSettings, Settings
from .scope import ScopeSelector, StrategyOverrideValues, StrategyScopeResolver, StrategyScopeRule


_SCOPE_CLAMP_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "mean_revert": {
        "stop_atr_mult": (1.5, 4.0),
        "rr_ratio": (1.2, 3.0),
        "max_hold_bars": (15, 80),
        "breakeven_after_rr": (0.3, 2.0),
        "trailing_stop_activation_rr": (0.3, 1.5),
        "trailing_stop_atr_mult": (0.3, 1.5),
    },
    "volatility_revert": {
        "stop_atr_mult": (1.5, 4.5),
        "rr_ratio": (1.2, 3.5),
        "max_hold_bars": (15, 200),
        "breakeven_after_rr": (0.3, 2.5),
        "trailing_stop_activation_rr": (0.3, 1.5),
        "trailing_stop_atr_mult": (0.3, 1.5),
    },
    "trend_follow": {
        "stop_atr_mult": (2.0, 6.0),
        "rr_ratio": (2.0, 6.0),
        "max_hold_bars": (50, 1000),
        "breakeven_after_rr": (0.5, 4.0),
        "trailing_stop_activation_rr": (0.5, 3.0),
        "trailing_stop_atr_mult": (0.5, 3.0),
    },
}
assert all(field in StrategyParams.model_fields for bounds in _SCOPE_CLAMP_BOUNDS.values() for field in bounds), (
    "_SCOPE_CLAMP_BOUNDS field names must match StrategyParams"
)


def _clamp_scope_value(strategy: str | None, field_name: str, value: float) -> float:
    """Clamp a scaled scope value to strategy-specific bounds."""
    if strategy is None:
        return value
    bounds = _SCOPE_CLAMP_BOUNDS.get(strategy, {}).get(field_name)
    if bounds is None:
        return value
    lo, hi = bounds
    return max(lo, min(hi, value))


def _scale_scope_rules(
    baseline: Settings,
    tuned_strategy: StrategyParams,
    search_param_names: frozenset[str],
) -> StrategyScopeResolver:
    """Scale scope rule overrides proportionally to optimizer changes, with clamping."""
    scaled_rules: list[StrategyScopeRule] = []
    for rule in baseline.strategy_scope.rules:
        overrides: dict[str, float | int] = {}
        strategy_name = rule.selector.strategy
        for field_name in search_param_names:
            rule_val = getattr(rule.values, field_name, None)
            if rule_val is None:
                continue
            baseline_val = getattr(baseline.strategy, field_name)
            tuned_val = getattr(tuned_strategy, field_name)
            if baseline_val is None or baseline_val == 0:
                scaled = float(tuned_val) if tuned_val is not None else float(rule_val)
            else:
                ratio = float(tuned_val) / float(baseline_val)
                scaled = float(rule_val) * ratio
            scaled = _clamp_scope_value(strategy_name, field_name, scaled)
            if field_name in INT_PARAMS:
                overrides[field_name] = round(scaled)
            else:
                overrides[field_name] = scaled
        if overrides:
            new_values = dc_replace(rule.values, **overrides)
            scaled_rules.append(dc_replace(rule, values=new_values))
        else:
            scaled_rules.append(rule)
    return StrategyScopeResolver(rules=tuple(scaled_rules))


def _scale_risk_scope_rules(
    baseline: Settings,
    tuned_risk: RiskSettings,
    risk_search_params: frozenset[str],
) -> RiskScopeResolver:
    """Scale risk scope rule overrides proportionally to optimizer changes."""
    scaled_rules: list[RiskScopeRule] = []
    for rule in baseline.risk_scope.rules:
        overrides: dict[str, float | int] = {}
        for field_name in risk_search_params:
            rule_val = getattr(rule.values, field_name, None)
            if rule_val is None:
                continue
            baseline_val = getattr(baseline.risk, field_name, None)
            tuned_val = getattr(tuned_risk, field_name, None)
            if tuned_val is None:
                continue
            if baseline_val is None or baseline_val == 0:
                scaled = float(tuned_val)
            else:
                ratio = float(tuned_val) / float(baseline_val)
                scaled = float(rule_val) * ratio
            overrides[field_name] = scaled
        if overrides:
            new_values = dc_replace(rule.values, **overrides)
            scaled_rules.append(dc_replace(rule, values=new_values))
        else:
            scaled_rules.append(rule)
    return RiskScopeResolver(rules=tuple(scaled_rules))


_REGIME_SELECTOR_MAP: dict[str, tuple[MarketState, ...]] = {
    "ranging": (MarketState.RANGING,),
    "trending": (MarketState.TRENDING_UP, MarketState.TRENDING_DOWN),
    "volatile": (MarketState.VOLATILE,),
}
assert frozenset(_REGIME_SELECTOR_MAP) == frozenset(REGIME_SCOPE_FIELDS), (
    "_REGIME_SELECTOR_MAP keys must match REGIME_SCOPE_FIELDS keys"
)


def _extract_regime_scope_params(params: ParamSet) -> dict[str, dict[str, float | int]]:
    """Group ``scope_<regime>__<field>`` params by regime key."""
    result: dict[str, dict[str, float | int]] = {}
    for regime_key, fields in REGIME_SCOPE_FIELDS.items():
        for field_name in fields:
            param_name = f"scope_{regime_key}__{field_name}"
            if param_name not in params:
                continue
            group = result.setdefault(regime_key, {})
            if param_name in INT_PARAMS:
                group[field_name] = int(params[param_name])
            else:
                group[field_name] = float(params[param_name])
    return result


def _build_regime_scope_rules(
    regime_params: dict[str, dict[str, float | int]],
) -> list[StrategyScopeRule]:
    """Build per-regime scope rules from optimizer-tuned regime params."""
    rules: list[StrategyScopeRule] = []
    for regime_key, fields in regime_params.items():
        if not fields:
            continue
        regimes = _REGIME_SELECTOR_MAP.get(regime_key, ())
        for regime in regimes:
            values = dc_replace(StrategyOverrideValues(), **fields)
            rule = StrategyScopeRule(
                id=f"auto_{regime.name.lower()}",
                priority=10,
                selector=ScopeSelector(regime=regime),
                values=values,
            )
            rules.append(rule)
    return rules


def apply_params(
    settings: Settings,
    params: ParamSet,
    *,
    baseline: Settings | None = None,
    search_names: frozenset[str] | None = None,
) -> Settings:
    """Apply optimization params to settings, returning a new Settings.

    Dynamic: iterates over ``params`` keys so callers may pass any subset
    (9-param reduced space or legacy 15-param dicts from old studies).

    When *search_names* is provided, skips ``SearchSpace`` construction
    (optimization hot path).
    """
    strategy_updates: dict[str, float | int] = {}
    for k, v in params.items():
        if k in REGIME_PARAMS or k in NON_STRATEGY_PARAMS or k.startswith(REGIME_SCOPE_PREFIX):
            continue
        if k in INT_PARAMS:
            strategy_updates[k] = int(v)
        else:
            strategy_updates[k] = float(v)
    strategy = settings.strategy.model_copy(update=strategy_updates)

    regime_updates: dict[str, float] = {}
    for k in REGIME_PARAMS:
        if k in params:
            regime_updates[k] = float(params[k])
    if "atr_high_pct" in regime_updates:
        atr_high = regime_updates["atr_high_pct"]
        regime_updates["atr_extreme_pct"] = max(settings.regime.atr_extreme_pct, atr_high + 0.1)
    regime = settings.regime.model_copy(update=regime_updates) if regime_updates else settings.regime

    update: dict[str, object] = {"strategy": strategy, "regime": regime}
    if baseline is not None:
        resolved_names = (
            search_names
            if search_names is not None
            else (
                SearchSpace(
                    partial_tp_enabled=settings.strategy.partial_tp_enabled,
                    confluence_filter_enabled=settings.strategy.confluence_filter_enabled,
                    enabled_strategies=settings.trading.enabled_strategies,
                ).strategy_param_names()
                | REGIME_PARAMS
            )
        )
        update["strategy_scope"] = _scale_scope_rules(baseline, strategy, resolved_names)

    # Apply backtest-level params (leverage)
    if "leverage" in params:
        update["backtest"] = settings.backtest.model_copy(update={"leverage": float(params["leverage"])})

    # Apply risk-level params (derived from NON_STRATEGY_PARAMS minus leverage)
    risk_updates: dict[str, float] = {}
    for rk in NON_STRATEGY_PARAMS - {"leverage"}:
        if rk in params:
            risk_updates[rk] = float(params[rk])
    tuned_risk = settings.risk
    if risk_updates:
        tuned_risk = settings.risk.model_copy(update=risk_updates)
        update["risk"] = tuned_risk

    # Scale risk scope rules proportionally (mirrors strategy scope scaling)
    if baseline is not None and risk_updates:
        risk_search_keys = frozenset(risk_updates.keys())
        update["risk_scope"] = _scale_risk_scope_rules(baseline, tuned_risk, risk_search_keys)

    # Generate per-regime scope rules from optimizer params
    regime_scope_params = _extract_regime_scope_params(params)
    if regime_scope_params:
        auto_rules = _build_regime_scope_rules(regime_scope_params)
        auto_ids = {r.id for r in auto_rules}
        existing_scope = update.get("strategy_scope")
        if not isinstance(existing_scope, StrategyScopeResolver):
            existing_scope = settings.strategy_scope
        # Remove stale auto rules from config before merging fresh ones
        kept_rules = tuple(r for r in existing_scope.rules if r.id not in auto_ids)
        update["strategy_scope"] = StrategyScopeResolver(rules=kept_rules + tuple(auto_rules))

    return settings.model_copy(update=update)


def generate_warm_start_params(settings: Settings) -> tuple[ParamSet, ...]:
    """Generate warm-start trials: baseline + 4 perturbations at +/-10%."""
    import numpy as np

    baseline = extract_regularization_baseline(settings)
    space = SearchSpace(
        partial_tp_enabled=settings.strategy.partial_tp_enabled,
        confluence_filter_enabled=settings.strategy.confluence_filter_enabled,
        enabled_strategies=settings.trading.enabled_strategies,
    )
    space_bounds = space.bounds()

    def _clip(k: str, v: float) -> float | int:
        if k in space_bounds:
            lo, hi = space_bounds[k]
            v = max(lo, min(hi, v))
        return round(v) if k in INT_PARAMS else v

    clipped_baseline: ParamSet = {k: _clip(k, float(v)) for k, v in baseline.items()}
    trials: list[ParamSet] = [dict(clipped_baseline)]

    rng = np.random.default_rng(42)
    for _ in range(4):
        perturbed: ParamSet = {}
        for k, v in clipped_baseline.items():
            fv = float(v)
            perturbed[k] = _clip(k, fv + fv * 0.10 * rng.uniform(-1.0, 1.0))
        trials.append(perturbed)

    return tuple(trials)
