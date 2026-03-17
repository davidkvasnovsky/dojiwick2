"""Risk rule protocol for composable risk evaluation."""

from typing import Protocol

from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchTradeCandidate,
    RiskRuleDecision,
)
from dojiwick.domain.models.value_objects.params import RiskParams


class RiskRule(Protocol):
    """A pluggable risk evaluation rule.

    Implementations must expose a ``name`` property and an ``evaluate``
    method that returns a :class:`RiskRuleDecision` with a boolean
    blocked mask over the batch.
    """

    @property
    def name(self) -> str: ...

    def evaluate(
        self,
        *,
        context: BatchDecisionContext,
        candidate: BatchTradeCandidate,
        risk_params: tuple[RiskParams, ...],
    ) -> RiskRuleDecision: ...


class ConfigurableRiskRule:
    """Base for risk rules with configurable precedence and severity."""

    def __init__(self, *, precedence: int, risk_score: float) -> None:
        self._precedence = precedence
        self._risk_score = risk_score
