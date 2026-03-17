"""Unit tests for reconciliation health state machine."""

from datetime import UTC, datetime, timedelta

from dojiwick.domain.enums import ReconciliationHealth
from dojiwick.domain.models.value_objects.health_state import HealthState
from dojiwick.domain.models.value_objects.reconciliation import PositionMismatch, ReconciliationResult
from dojiwick.domain.reconciliation_health import compute_health


def _clean_result() -> ReconciliationResult:
    return ReconciliationResult(orphaned_db=(), orphaned_exchange=(), mismatches=())


def _orphaned_exchange_result(pairs: tuple[str, ...] = ("BTC/USDC",)) -> ReconciliationResult:
    return ReconciliationResult(orphaned_db=(), orphaned_exchange=pairs, mismatches=())


def _mismatch_result(pairs: tuple[str, ...] = ("ETH/USDC",)) -> ReconciliationResult:
    mismatches = tuple(PositionMismatch(pair=p, order_id="o1", db_state="open", exchange_state="closed") for p in pairs)
    return ReconciliationResult(orphaned_db=(), orphaned_exchange=(), mismatches=mismatches)


_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_NORMAL = HealthState()


def test_normal_clean_stays_normal() -> None:
    result = compute_health(_NORMAL, _clean_result(), _NOW)
    assert result.health is ReconciliationHealth.NORMAL
    assert result.frozen_symbols == ()


def test_normal_orphaned_exchange_transitions_to_uncertain() -> None:
    result = compute_health(_NORMAL, _orphaned_exchange_result(("BTC/USDC",)), _NOW)
    assert result.health is ReconciliationHealth.UNCERTAIN
    assert "BTC/USDC" in result.frozen_symbols


def test_normal_qty_mismatch_transitions_to_degraded() -> None:
    result = compute_health(_NORMAL, _mismatch_result(("ETH/USDC",)), _NOW)
    assert result.health is ReconciliationHealth.DEGRADED
    assert "ETH/USDC" in result.frozen_symbols


def test_degraded_clean_recovers_to_normal() -> None:
    degraded = HealthState(
        health=ReconciliationHealth.DEGRADED,
        health_since=_NOW - timedelta(seconds=10),
        frozen_symbols=("ETH/USDC",),
    )
    result = compute_health(degraded, _clean_result(), _NOW)
    assert result.health is ReconciliationHealth.NORMAL
    assert result.frozen_symbols == ()


def test_degraded_timeout_escalates_to_uncertain() -> None:
    degraded = HealthState(
        health=ReconciliationHealth.DEGRADED,
        health_since=_NOW - timedelta(seconds=301),
        frozen_symbols=("ETH/USDC",),
    )
    result = compute_health(degraded, _mismatch_result(), _NOW, degraded_timeout_sec=300)
    assert result.health is ReconciliationHealth.UNCERTAIN


def test_degraded_before_timeout_stays() -> None:
    degraded = HealthState(
        health=ReconciliationHealth.DEGRADED,
        health_since=_NOW - timedelta(seconds=100),
        frozen_symbols=("ETH/USDC",),
    )
    result = compute_health(degraded, _mismatch_result(), _NOW, degraded_timeout_sec=300)
    assert result.health is ReconciliationHealth.DEGRADED


def test_uncertain_clean_recovers_to_normal() -> None:
    uncertain = HealthState(
        health=ReconciliationHealth.UNCERTAIN,
        health_since=_NOW - timedelta(seconds=10),
        frozen_symbols=("BTC/USDC",),
    )
    result = compute_health(uncertain, _clean_result(), _NOW)
    assert result.health is ReconciliationHealth.NORMAL
    assert result.frozen_symbols == ()


def test_uncertain_timeout_escalates_to_halt() -> None:
    uncertain = HealthState(
        health=ReconciliationHealth.UNCERTAIN,
        health_since=_NOW - timedelta(seconds=901),
        frozen_symbols=("BTC/USDC",),
    )
    result = compute_health(uncertain, _orphaned_exchange_result(), _NOW, uncertain_timeout_sec=900)
    assert result.health is ReconciliationHealth.HALT


def test_halt_clean_stays_halt() -> None:
    halt = HealthState(
        health=ReconciliationHealth.HALT,
        health_since=_NOW - timedelta(seconds=10),
        frozen_symbols=("BTC/USDC",),
    )
    result = compute_health(halt, _clean_result(), _NOW)
    assert result.health is ReconciliationHealth.HALT


def test_halt_divergence_stays_halt() -> None:
    halt = HealthState(
        health=ReconciliationHealth.HALT,
        health_since=_NOW - timedelta(seconds=10),
        frozen_symbols=("BTC/USDC",),
    )
    result = compute_health(halt, _orphaned_exchange_result(), _NOW)
    assert result.health is ReconciliationHealth.HALT


def test_health_since_set_on_transition() -> None:
    result = compute_health(_NORMAL, _orphaned_exchange_result(), _NOW)
    assert result.health_since == _NOW

    degraded = HealthState(
        health=ReconciliationHealth.DEGRADED,
        health_since=_NOW - timedelta(seconds=301),
        frozen_symbols=("ETH/USDC",),
    )
    result2 = compute_health(degraded, _mismatch_result(), _NOW, degraded_timeout_sec=300)
    assert result2.health_since == _NOW


def test_frozen_symbols_contains_orphaned_exchange_pairs() -> None:
    result = compute_health(_NORMAL, _orphaned_exchange_result(("BTC/USDC", "SOL/USDC")), _NOW)
    assert result.health is ReconciliationHealth.UNCERTAIN
    assert set(result.frozen_symbols) == {"BTC/USDC", "SOL/USDC"}


def _orphaned_db_result(pairs: tuple[str, ...] = ("BTC/USDC",)) -> ReconciliationResult:
    return ReconciliationResult(orphaned_db=pairs, orphaned_exchange=(), mismatches=())


def test_orphaned_db_triggers_degraded() -> None:
    """orphaned_db positions cause divergence detection and transition to DEGRADED."""
    result = compute_health(_NORMAL, _orphaned_db_result(), _NOW)
    assert result.health is ReconciliationHealth.DEGRADED


def test_orphaned_db_merged_into_frozen_symbols() -> None:
    """orphaned_db symbols are included in frozen symbols during merge."""
    degraded = HealthState(
        health=ReconciliationHealth.DEGRADED,
        health_since=_NOW - timedelta(seconds=301),
        frozen_symbols=("ETH/USDC",),
    )
    combined = ReconciliationResult(orphaned_db=("BTC/USDC",), orphaned_exchange=(), mismatches=())
    result = compute_health(degraded, combined, _NOW, degraded_timeout_sec=300)
    assert result.health is ReconciliationHealth.UNCERTAIN
    assert "BTC/USDC" in result.frozen_symbols
    assert "ETH/USDC" in result.frozen_symbols
