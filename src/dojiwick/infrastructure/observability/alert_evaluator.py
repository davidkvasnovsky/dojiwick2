"""Log-based alert evaluator for operational thresholds."""

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

_DEFAULT_FAILURE_THRESHOLD = 3
_DEFAULT_VETO_RATE_THRESHOLD = 0.5
_DEFAULT_BUDGET_WARNING_THRESHOLD = 0.8


@dataclass(slots=True, frozen=True)
class AlertEvaluator:
    """Evaluates operational metrics and emits log warnings when thresholds are breached."""

    failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD
    veto_rate_threshold: float = _DEFAULT_VETO_RATE_THRESHOLD
    budget_warning_threshold: float = _DEFAULT_BUDGET_WARNING_THRESHOLD

    def evaluate_tick_failure(self, consecutive_failures: int) -> None:
        """Log warning when consecutive tick failures reach the threshold."""
        if consecutive_failures >= self.failure_threshold:
            log.warning(
                "consecutive tick failures reached threshold: %d >= %d",
                consecutive_failures,
                self.failure_threshold,
            )

    def evaluate_veto_rate(self, vetoed: int, total: int) -> None:
        """Log warning when veto rate exceeds the threshold."""
        if total <= 0:
            return
        rate = vetoed / total
        if rate > self.veto_rate_threshold:
            log.warning(
                "veto rate %.1f%% exceeds threshold %.1f%% (%d/%d)",
                rate * 100,
                self.veto_rate_threshold * 100,
                vetoed,
                total,
            )

    def evaluate_budget(self, spent_usd: float, daily_budget_usd: float) -> None:
        """Log warning when daily budget usage exceeds the warning threshold."""
        if daily_budget_usd <= 0:
            return
        usage = spent_usd / daily_budget_usd
        if usage > self.budget_warning_threshold:
            log.warning(
                "daily AI budget usage %.1f%% exceeds warning threshold %.1f%% ($%.4f / $%.2f)",
                usage * 100,
                self.budget_warning_threshold * 100,
                spent_usd,
                daily_budget_usd,
            )
