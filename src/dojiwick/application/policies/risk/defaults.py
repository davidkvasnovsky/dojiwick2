"""Default risk engine factory with built-in rules."""

from typing import Protocol

from dojiwick.application.policies.risk.engine import RiskPolicyEngine


class RiskRuleConfig(Protocol):
    """Rule precedence and severity configuration for the default risk engine.

    Satisfied by ``RiskSettings`` from config.schema via structural subtyping.
    """

    @property
    def daily_loss_precedence(self) -> int: ...
    @property
    def daily_loss_severity(self) -> float: ...
    @property
    def max_positions_precedence(self) -> int: ...
    @property
    def max_positions_severity(self) -> float: ...
    @property
    def zero_stop_precedence(self) -> int: ...
    @property
    def zero_stop_severity(self) -> float: ...
    @property
    def min_rr_precedence(self) -> int: ...
    @property
    def min_rr_severity(self) -> float: ...


def build_default_risk_engine(settings: RiskRuleConfig) -> RiskPolicyEngine:
    """Register the 4 built-in risk rules and return a ready engine.

    Rules are registered in order of precedence (highest priority first),
    though the engine sorts by precedence at evaluation time regardless
    of registration order. Precedence and severity values are injected
    from ``settings``.
    """
    from dojiwick.compute.kernels.risk.rules.daily_loss import DailyLossRule
    from dojiwick.compute.kernels.risk.rules.max_positions import MaxPositionsRule
    from dojiwick.compute.kernels.risk.rules.min_rr import MinRRRule
    from dojiwick.compute.kernels.risk.rules.zero_stop import ZeroStopRule

    engine = RiskPolicyEngine()
    engine.register(DailyLossRule(precedence=settings.daily_loss_precedence, risk_score=settings.daily_loss_severity))
    engine.register(
        MaxPositionsRule(precedence=settings.max_positions_precedence, risk_score=settings.max_positions_severity)
    )
    engine.register(ZeroStopRule(precedence=settings.zero_stop_precedence, risk_score=settings.zero_stop_severity))
    engine.register(MinRRRule(precedence=settings.min_rr_precedence, risk_score=settings.min_rr_severity))
    return engine
