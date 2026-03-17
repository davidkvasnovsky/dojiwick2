"""Deterministic risk-scope resolution with priority guardrails.

Parallel to scope.py for strategy overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.params import RiskParams

from .scope import FieldWinner, MatchedRule, ScopeSelector, ranked_scope_matches, regime_name


@dataclass(slots=True, frozen=True, kw_only=True)
class RiskOverrideValues:
    """Risk fields that a scope rule may override."""

    max_daily_loss_pct: float | None = None
    max_open_positions: int | None = None
    min_rr_ratio: float | None = None
    risk_per_trade_pct: float | None = None
    min_notional_usd: float | None = None
    max_notional_pct_of_equity: float | None = None
    max_notional_usd: float | None = None
    max_loss_per_trade_pct: float | None = None
    max_portfolio_risk_pct: float | None = None
    trade_cooldown_sec: int | None = None
    max_consecutive_losses: int | None = None
    pair_win_rate_floor: float | None = None
    max_sector_exposure: int | None = None
    max_risk_inflation_mult: float | None = None
    drawdown_halt_pct: float | None = None
    drawdown_risk_scale_enabled: bool | None = None
    drawdown_risk_scale_max_dd: float | None = None
    drawdown_risk_scale_floor: float | None = None
    equity_curve_filter_enabled: bool | None = None
    equity_curve_filter_period: int | None = None
    portfolio_risk_baseline_pairs: int | None = None

    def has_any(self) -> bool:
        """Return True when at least one override field is set."""
        return any(getattr(self, f.name) is not None for f in fields(RiskOverrideValues))


RISK_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(RiskOverrideValues))


_MAX_PRIORITY = 1_000_000


@dataclass(slots=True, frozen=True, kw_only=True)
class RiskScopeRule:
    """One risk scope rule with deterministic resolution metadata."""

    id: str
    priority: int
    selector: ScopeSelector
    values: RiskOverrideValues

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("scope.risk.id must not be empty")
        if self.priority < 0 or self.priority > _MAX_PRIORITY:
            raise ValueError(f"scope.risk.priority must be in [0, {_MAX_PRIORITY}]")
        if not self.values.has_any():
            raise ValueError("scope.risk must set at least one override field")


@dataclass(slots=True, frozen=True, kw_only=True)
class RiskResolutionTrace:
    """Deterministic risk resolution trace for one pair/regime."""

    pair: str
    regime: MarketState | None
    matched_rules: tuple[MatchedRule, ...]
    field_winners: tuple[FieldWinner, ...]
    resolved: RiskParams

    def as_json(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the trace."""
        return {
            "pair": self.pair,
            "regime": regime_name(self.regime) if self.regime is not None else None,
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
class RiskScopeResolver:
    """Deterministic resolver for risk scope rules."""

    rules: tuple[RiskScopeRule, ...] = ()

    def __post_init__(self) -> None:
        seen_ids: set[str] = set()
        seen_scope_priority: set[tuple[str, int]] = set()

        for rule in self.rules:
            if rule.id in seen_ids:
                raise ValueError(f"duplicate scope.risk.id: {rule.id}")
            seen_ids.add(rule.id)

            selector_key = rule.selector.canonical_key
            composite = (selector_key, rule.priority)
            if composite in seen_scope_priority:
                raise ValueError(
                    f"duplicate scope.risk selector+priority: selector={selector_key} priority={rule.priority}"
                )
            seen_scope_priority.add(composite)

    @classmethod
    def empty(cls) -> RiskScopeResolver:
        """Return an empty scope resolver."""
        return cls(rules=())

    def resolve(self, pair: str, regime: MarketState | None, default: RiskParams) -> RiskParams:
        """Resolve risk params for a pair/regime with deterministic precedence."""
        return self.explain(pair=pair, regime=regime, default=default).resolved

    def explain(self, pair: str, regime: MarketState | None, default: RiskParams) -> RiskResolutionTrace:
        """Return full explain trace for deterministic risk resolution."""
        ranked_rules = self._ranked_matches(pair=pair, regime=regime)

        replacements: dict[str, float | int] = {}
        winners: list[FieldWinner] = []

        for field_name in RISK_FIELDS:
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

        return RiskResolutionTrace(
            pair=pair,
            regime=regime,
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

    def _ranked_matches(self, pair: str, regime: MarketState | None) -> tuple[RiskScopeRule, ...]:
        return ranked_scope_matches(self.rules, pair, regime)
