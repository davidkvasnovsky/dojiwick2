"""Pure state machine for reconciliation health transitions."""

from __future__ import annotations

from datetime import datetime

from dojiwick.domain.enums import ReconciliationHealth
from dojiwick.domain.models.value_objects.health_state import HealthState
from dojiwick.domain.models.value_objects.reconciliation import ReconciliationResult


def _has_divergence(result: ReconciliationResult) -> bool:
    """True when the result contains any divergence (exchange or DB side)."""
    return bool(result.orphaned_exchange or result.orphaned_db or result.mismatches)


def compute_health(
    current: HealthState,
    result: ReconciliationResult,
    observed_at: datetime,
    *,
    degraded_timeout_sec: int = 300,
    uncertain_timeout_sec: int = 900,
) -> HealthState:
    """Compute the next health state from the current state and reconciliation result.

    State machine transitions:
    - NORMAL + clean → NORMAL
    - NORMAL + orphaned_exchange → UNCERTAIN (freeze those symbols)
    - NORMAL + qty_mismatch → DEGRADED
    - DEGRADED + clean → NORMAL (clear frozen)
    - DEGRADED + elapsed >= degraded_timeout_sec → UNCERTAIN
    - UNCERTAIN + clean → NORMAL (clear frozen)
    - UNCERTAIN + elapsed >= uncertain_timeout_sec → HALT
    - HALT → HALT always (manual-only recovery)
    """
    health = current.health

    if health is ReconciliationHealth.HALT:
        return current

    if health is ReconciliationHealth.NORMAL:
        if not _has_divergence(result):
            return current
        if result.orphaned_exchange:
            return HealthState(
                health=ReconciliationHealth.UNCERTAIN,
                health_since=observed_at,
                frozen_symbols=result.orphaned_exchange,
            )
        return HealthState(
            health=ReconciliationHealth.DEGRADED,
            health_since=observed_at,
            frozen_symbols=tuple(m.pair for m in result.mismatches),
        )

    elapsed = _elapsed_sec(current.health_since, observed_at)

    if health is ReconciliationHealth.DEGRADED:
        if not _has_divergence(result):
            return HealthState(health=ReconciliationHealth.NORMAL, health_since=observed_at)
        if elapsed >= degraded_timeout_sec:
            frozen = _merge_frozen(current.frozen_symbols, result)
            return HealthState(
                health=ReconciliationHealth.UNCERTAIN,
                health_since=observed_at,
                frozen_symbols=frozen,
            )
        return current

    # UNCERTAIN
    if not _has_divergence(result):
        return HealthState(health=ReconciliationHealth.NORMAL, health_since=observed_at)
    if elapsed >= uncertain_timeout_sec:
        frozen = _merge_frozen(current.frozen_symbols, result)
        return HealthState(
            health=ReconciliationHealth.HALT,
            health_since=observed_at,
            frozen_symbols=frozen,
        )
    return current


def _elapsed_sec(since: datetime | None, now: datetime) -> float:
    """Seconds elapsed since a health state was entered."""
    if since is None:
        return 0.0
    return (now - since).total_seconds()


def _merge_frozen(existing: tuple[str, ...], result: ReconciliationResult) -> tuple[str, ...]:
    """Merge existing frozen symbols with new divergences."""
    symbols: set[str] = set(existing)
    symbols.update(result.orphaned_exchange)
    symbols.update(result.orphaned_db)
    symbols.update(m.pair for m in result.mismatches)
    return tuple(sorted(symbols))
