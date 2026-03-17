"""Unit tests for the risk scope resolver."""

import pytest

from dojiwick.config.risk_scope import (
    RiskOverrideValues,
    RiskScopeResolver,
    RiskScopeRule,
)
from dojiwick.config.scope import ScopeSelector
from dojiwick.domain.enums import MarketState
from fixtures.factories.infrastructure import default_risk_params


_DEFAULT = default_risk_params()


class TestRiskOverrideValues:
    def test_has_any_false_when_empty(self) -> None:
        assert not RiskOverrideValues().has_any()

    def test_has_any_true_with_float(self) -> None:
        assert RiskOverrideValues(risk_per_trade_pct=2.0).has_any()

    def test_has_any_true_with_int(self) -> None:
        assert RiskOverrideValues(max_open_positions=4).has_any()


class TestRiskScopeResolver:
    def test_no_rules_returns_default(self) -> None:
        resolver = RiskScopeResolver.empty()
        result = resolver.resolve("BTC/USDC", None, _DEFAULT)
        assert result == _DEFAULT

    def test_priority_and_specificity_resolution(self) -> None:
        resolver = RiskScopeResolver(
            rules=(
                RiskScopeRule(
                    id="global_loss",
                    priority=10,
                    selector=ScopeSelector(),
                    values=RiskOverrideValues(max_daily_loss_pct=3.0),
                ),
                RiskScopeRule(
                    id="btc_loss",
                    priority=20,
                    selector=ScopeSelector(pair="BTC/USDC"),
                    values=RiskOverrideValues(max_daily_loss_pct=2.0),
                ),
            )
        )
        result = resolver.resolve("BTC/USDC", None, _DEFAULT)
        assert result.max_daily_loss_pct == 2.0

    def test_explain_trace_contains_winners(self) -> None:
        resolver = RiskScopeResolver(
            rules=(
                RiskScopeRule(
                    id="base",
                    priority=10,
                    selector=ScopeSelector(),
                    values=RiskOverrideValues(risk_per_trade_pct=0.5),
                ),
                RiskScopeRule(
                    id="btc_pair",
                    priority=20,
                    selector=ScopeSelector(pair="BTC/USDC"),
                    values=RiskOverrideValues(risk_per_trade_pct=0.8, max_open_positions=4),
                ),
            )
        )
        trace = resolver.explain("BTC/USDC", None, _DEFAULT)
        winner_fields = {w.field_name for w in trace.field_winners}
        assert "risk_per_trade_pct" in winner_fields
        assert "max_open_positions" in winner_fields
        assert trace.resolved.risk_per_trade_pct == 0.8
        assert trace.resolved.max_open_positions == 4

    def test_duplicate_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate scope.risk.id"):
            RiskScopeResolver(
                rules=(
                    RiskScopeRule(
                        id="dup",
                        priority=10,
                        selector=ScopeSelector(),
                        values=RiskOverrideValues(max_daily_loss_pct=3.0),
                    ),
                    RiskScopeRule(
                        id="dup",
                        priority=20,
                        selector=ScopeSelector(pair="BTC/USDC"),
                        values=RiskOverrideValues(max_daily_loss_pct=2.0),
                    ),
                )
            )

    def test_duplicate_selector_priority_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"duplicate scope.risk selector\+priority"):
            RiskScopeResolver(
                rules=(
                    RiskScopeRule(
                        id="a",
                        priority=10,
                        selector=ScopeSelector(pair="BTC/USDC"),
                        values=RiskOverrideValues(max_daily_loss_pct=3.0),
                    ),
                    RiskScopeRule(
                        id="b",
                        priority=10,
                        selector=ScopeSelector(pair="BTC/USDC"),
                        values=RiskOverrideValues(risk_per_trade_pct=0.5),
                    ),
                )
            )

    def test_empty_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one override field"):
            RiskScopeRule(
                id="empty",
                priority=10,
                selector=ScopeSelector(),
                values=RiskOverrideValues(),
            )

    def test_regime_matching(self) -> None:
        resolver = RiskScopeResolver(
            rules=(
                RiskScopeRule(
                    id="volatile_risk",
                    priority=10,
                    selector=ScopeSelector(regime=MarketState.VOLATILE),
                    values=RiskOverrideValues(risk_per_trade_pct=0.3),
                ),
            )
        )
        matched = resolver.resolve("BTC/USDC", MarketState.VOLATILE, _DEFAULT)
        unmatched = resolver.resolve("BTC/USDC", MarketState.TRENDING_UP, _DEFAULT)
        assert matched.risk_per_trade_pct == 0.3
        assert unmatched.risk_per_trade_pct == _DEFAULT.risk_per_trade_pct

    def test_as_json(self) -> None:
        resolver = RiskScopeResolver(
            rules=(
                RiskScopeRule(
                    id="test",
                    priority=10,
                    selector=ScopeSelector(),
                    values=RiskOverrideValues(max_daily_loss_pct=3.0),
                ),
            )
        )
        trace = resolver.explain("BTC/USDC", None, _DEFAULT)
        json_output = trace.as_json()
        assert json_output["pair"] == "BTC/USDC"
        assert "resolved" in json_output
        assert "matched_rules" in json_output
