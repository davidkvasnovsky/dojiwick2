"""Typed AI evaluation artifact — replaces boolean veto with structured result."""

from dataclasses import dataclass, field
from typing import Any

from dojiwick.domain.numerics import Confidence


@dataclass(slots=True, frozen=True, kw_only=True)
class AIEvaluationResult:
    """Structured result from the AI evaluation / veto filter.

    Fields
    ------
    approval : bool
        True if the candidate is approved, False if vetoed.
    confidence : Confidence
        Model confidence in the evaluation (0.0 – 1.0).
    reason_code : str
        Machine-readable code (e.g., "AI_VETO", "AI_APPROVE", "AI_ERROR").
    rationale : str
        Human-readable explanation of the decision.
    policy_flags : dict[str, Any]
        Arbitrary key-value pairs for policy metadata (e.g., risk tags, model version).
    """

    approval: bool
    confidence: Confidence
    reason_code: str
    rationale: str
    policy_flags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")
        if not self.reason_code:
            raise ValueError("reason_code must not be empty")
