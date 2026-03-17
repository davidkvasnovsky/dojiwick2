"""Unit tests for AlertEvaluator threshold-based log warnings."""

import logging

import pytest

from dojiwick.infrastructure.observability.alert_evaluator import AlertEvaluator


@pytest.fixture
def evaluator() -> AlertEvaluator:
    return AlertEvaluator()


def test_consecutive_failures_alert(evaluator: AlertEvaluator, caplog: pytest.LogCaptureFixture) -> None:
    """3+ consecutive failures triggers a warning."""
    with caplog.at_level(logging.WARNING):
        evaluator.evaluate_tick_failure(3)
    assert any("consecutive tick failures" in r.message for r in caplog.records)


def test_high_veto_rate_alert(evaluator: AlertEvaluator, caplog: pytest.LogCaptureFixture) -> None:
    """Veto rate >50% triggers a warning."""
    with caplog.at_level(logging.WARNING):
        evaluator.evaluate_veto_rate(vetoed=6, total=10)
    assert any("veto rate" in r.message for r in caplog.records)


def test_budget_warning(evaluator: AlertEvaluator, caplog: pytest.LogCaptureFixture) -> None:
    """>80% budget usage triggers a warning."""
    with caplog.at_level(logging.WARNING):
        evaluator.evaluate_budget(spent_usd=0.85, daily_budget_usd=1.0)
    assert any("budget usage" in r.message for r in caplog.records)


def test_no_alert_below_threshold(evaluator: AlertEvaluator, caplog: pytest.LogCaptureFixture) -> None:
    """Below all thresholds, no warnings emitted."""
    with caplog.at_level(logging.WARNING):
        evaluator.evaluate_tick_failure(1)
        evaluator.evaluate_veto_rate(vetoed=2, total=10)
        evaluator.evaluate_budget(spent_usd=0.5, daily_budget_usd=1.0)
    assert len(caplog.records) == 0
