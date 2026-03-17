"""Deterministic strategy-scope resolution with priority guardrails."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields
from typing import Protocol, TypeVar

from dojiwick.domain.enums import MarketState, RegimeExitProfile, SQL_TO_MARKET_STATE
from dojiwick.domain.models.value_objects.params import StrategyParams

_MAX_PRIORITY = 1_000_000


class ScopeRuleLike(Protocol):
    """Protocol for scope rules used in ranked matching."""

    @property
    def id(self) -> str: ...
    @property
    def priority(self) -> int: ...
    @property
    def selector(self) -> ScopeSelector: ...


_ScopeRuleT = TypeVar("_ScopeRuleT", bound=ScopeRuleLike)


@dataclass(slots=True, frozen=True, kw_only=True)
class ScopeSelector:
    """Target selector for strategy rules."""

    pair: str | None = None
    regime: MarketState | None = None
    strategy: str | None = None

    def __post_init__(self) -> None:
        if self.pair is not None and not self.pair:
            raise ValueError("selector.pair must not be empty")
        if self.strategy is not None and not self.strategy:
            raise ValueError("selector.strategy must not be empty")

    @property
    def specificity(self) -> int:
        """Return selector specificity rank."""
        score = 0
        if self.pair is not None:
            score += 2
        if self.regime is not None:
            score += 1
        if self.strategy is not None:
            score += 1
        return score

    @property
    def canonical_key(self) -> str:
        """Stable selector key used in deterministic tie-breakers."""
        pair = self.pair if self.pair is not None else "*"
        regime = regime_name(self.regime) if self.regime is not None else "*"
        strategy = self.strategy if self.strategy is not None else "*"
        return f"pair={pair}|regime={regime}|strategy={strategy}"


@dataclass(slots=True, frozen=True, kw_only=True)
class StrategyOverrideValues:
    """Strategy fields that a scope rule may override."""

    stop_atr_mult: float | None = None
    rr_ratio: float | None = None
    min_stop_distance_pct: float | None = None
    default_variant: str | None = None
    trend_pullback_rsi_max: float | None = None
    trend_overbought_rsi_min: float | None = None
    trend_breakout_adx_min: float | None = None
    mean_rsi_oversold: float | None = None
    mean_rsi_overbought: float | None = None
    vol_extreme_oversold: float | None = None
    vol_extreme_overbought: float | None = None
    trailing_stop_activation_rr: float | None = None
    trailing_stop_atr_mult: float | None = None
    breakeven_after_rr: float | None = None
    min_volume_ratio: float | None = None
    max_hold_bars: int | None = None
    trend_pullback_adx_min: float | None = None
    trend_max_regime_confidence: float | None = None
    trend_short_max_regime_confidence: float | None = None
    trend_volatile_ema_enabled: bool | None = None
    partial_tp_enabled: bool | None = None
    partial_tp1_rr: float | None = None
    partial_tp1_fraction: float | None = None
    mean_revert_use_bb_mid_tp: bool | None = None
    mean_revert_disable_breakeven: bool | None = None
    mean_revert_disable_ema_filter: bool | None = None
    macd_filter_enabled: bool | None = None
    confluence_filter_enabled: bool | None = None
    min_confluence_score: float | None = None
    regime_exit_profile: RegimeExitProfile | None = None
    adaptive_volatile_stop_scale: float | None = None
    adaptive_volatile_rr_mult: float | None = None
    adaptive_trending_trail_scale: float | None = None
    adaptive_volatile_max_bars: int | None = None
    adaptive_ranging_max_bars: int | None = None
    confluence_rsi_midpoint: float | None = None
    confluence_rsi_range: float | None = None
    confluence_volume_baseline: float | None = None
    confluence_volume_multiplier: float | None = None
    confluence_adx_baseline: float | None = None
    confluence_adx_range: float | None = None
    partial_tp_stop_ratio: float | None = None

    def has_any(self) -> bool:
        """Return True when at least one override field is set."""
        return any(getattr(self, f.name) is not None for f in fields(StrategyOverrideValues))


STRATEGY_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(StrategyOverrideValues))


@dataclass(slots=True, frozen=True, kw_only=True)
class StrategyScopeRule:
    """One strategy scope rule with deterministic resolution metadata."""

    id: str
    priority: int
    selector: ScopeSelector
    values: StrategyOverrideValues

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("scope.strategy.id must not be empty")
        if self.priority < 0 or self.priority > _MAX_PRIORITY:
            raise ValueError(f"scope.strategy.priority must be in [0, {_MAX_PRIORITY}]")
        if not self.values.has_any():
            raise ValueError("scope.strategy must set at least one override field")


@dataclass(slots=True, frozen=True, kw_only=True)
class MatchedRule:
    """Rule metadata captured in explain traces."""

    rule_id: str
    priority: int
    specificity: int
    selector: str


@dataclass(slots=True, frozen=True, kw_only=True)
class FieldWinner:
    """Per-field winning rule captured in explain traces."""

    field_name: str
    value: float | str | bool
    rule_id: str
    priority: int
    specificity: int
    selector: str
    reason: str


@dataclass(slots=True, frozen=True, kw_only=True)
class ResolutionTrace:
    """Deterministic strategy resolution trace for one pair/regime."""

    pair: str
    regime: MarketState | None
    strategy: str | None = None
    matched_rules: tuple[MatchedRule, ...]
    field_winners: tuple[FieldWinner, ...]
    resolved: StrategyParams

    def as_json(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the trace."""
        return {
            "pair": self.pair,
            "regime": regime_name(self.regime) if self.regime is not None else None,
            "strategy": self.strategy,
            "matched_rules": [
                {
                    "rule_id": rule.rule_id,
                    "priority": rule.priority,
                    "specificity": rule.specificity,
                    "selector": rule.selector,
                }
                for rule in self.matched_rules
            ],
            "field_winners": [
                {
                    "field_name": winner.field_name,
                    "value": winner.value,
                    "rule_id": winner.rule_id,
                    "priority": winner.priority,
                    "specificity": winner.specificity,
                    "selector": winner.selector,
                    "reason": winner.reason,
                }
                for winner in self.field_winners
            ],
            "resolved": self.resolved.model_dump(),
        }


@dataclass(slots=True, frozen=True, kw_only=True)
class StrategyScopeResolver:
    """Deterministic resolver for strategy scope rules."""

    rules: tuple[StrategyScopeRule, ...] = ()

    def __post_init__(self) -> None:
        seen_ids: set[str] = set()
        seen_scope_priority: set[tuple[str, int]] = set()

        for rule in self.rules:
            if rule.id in seen_ids:
                raise ValueError(f"duplicate scope.strategy.id: {rule.id}")
            seen_ids.add(rule.id)

            selector_key = rule.selector.canonical_key
            composite = (selector_key, rule.priority)
            if composite in seen_scope_priority:
                raise ValueError(
                    f"duplicate scope.strategy selector+priority: selector={selector_key} priority={rule.priority}"
                )
            seen_scope_priority.add(composite)

    @property
    def has_strategy_rules(self) -> bool:
        """True when at least one rule targets a specific strategy name."""
        return any(r.selector.strategy is not None for r in self.rules)

    @classmethod
    def empty(cls) -> StrategyScopeResolver:
        """Return an empty scope resolver."""
        return cls(rules=())

    def resolve(
        self,
        pair: str,
        regime: MarketState | None,
        default: StrategyParams,
        strategy: str | None = None,
    ) -> StrategyParams:
        """Resolve strategy params for a pair/regime with deterministic precedence."""
        ranked_rules = self._ranked_matches(pair=pair, regime=regime, strategy=strategy)
        if not ranked_rules:
            return default

        replacements: dict[str, float | str | bool] = {}
        for field_name in STRATEGY_FIELDS:
            for rule in ranked_rules:
                value = getattr(rule.values, field_name)
                if value is not None:
                    replacements[field_name] = value
                    break

        return default if not replacements else default.model_copy(update=replacements)

    def explain(
        self,
        pair: str,
        regime: MarketState | None,
        default: StrategyParams,
        strategy: str | None = None,
    ) -> ResolutionTrace:
        """Return full explain trace for deterministic strategy resolution."""
        ranked_rules = self._ranked_matches(pair=pair, regime=regime, strategy=strategy)

        replacements: dict[str, float | str | bool] = {}
        winners: list[FieldWinner] = []

        for field_name in STRATEGY_FIELDS:
            for rule in ranked_rules:
                value = getattr(rule.values, field_name)
                if value is None:
                    continue
                replacements[field_name] = value
                winners.append(
                    FieldWinner(
                        field_name=field_name,
                        value=value,
                        rule_id=rule.id,
                        priority=rule.priority,
                        specificity=rule.selector.specificity,
                        selector=rule.selector.canonical_key,
                        reason="priority > specificity > selector_lexicographic",
                    )
                )
                break

        resolved = default if not replacements else default.model_copy(update=replacements)

        return ResolutionTrace(
            pair=pair,
            regime=regime,
            strategy=strategy,
            matched_rules=tuple(
                MatchedRule(
                    rule_id=rule.id,
                    priority=rule.priority,
                    specificity=rule.selector.specificity,
                    selector=rule.selector.canonical_key,
                )
                for rule in ranked_rules
            ),
            field_winners=tuple(winners),
            resolved=resolved,
        )

    def _ranked_matches(
        self,
        pair: str,
        regime: MarketState | None,
        strategy: str | None = None,
    ) -> tuple[StrategyScopeRule, ...]:
        return ranked_scope_matches(self.rules, pair, regime, strategy)


def parse_regime(value: str) -> MarketState:
    """Parse TOML regime literal into MarketState."""
    normalized = value.strip().lower()
    if normalized not in SQL_TO_MARKET_STATE:
        valid = ", ".join(sorted(SQL_TO_MARKET_STATE.keys()))
        raise ValueError(f"regime must be one of: {valid}; got '{value}'")
    return SQL_TO_MARKET_STATE[normalized]


def regime_name(value: MarketState) -> str:
    """Return canonical lower-case regime name."""
    return value.name.lower()


def selector_matches(
    selector: ScopeSelector,
    pair: str,
    regime: MarketState | None,
    strategy: str | None = None,
) -> bool:
    if selector.pair is not None and selector.pair != pair:
        return False
    if selector.regime is not None and selector.regime != regime:
        return False
    if selector.strategy is not None:
        if strategy is None or selector.strategy != strategy:
            return False
    return True


def ranked_scope_matches(
    rules: Iterable[_ScopeRuleT],
    pair: str,
    regime: MarketState | None,
    strategy: str | None = None,
) -> tuple[_ScopeRuleT, ...]:
    """Filter and rank scope rules by priority/specificity for deterministic resolution."""
    matched = tuple(rule for rule in rules if selector_matches(rule.selector, pair, regime, strategy))
    return tuple(
        sorted(
            matched,
            key=lambda rule: (
                -rule.priority,
                -rule.selector.specificity,
                rule.selector.canonical_key,
                rule.id,
            ),
        )
    )
