"""Reconciliation health state value object."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from dojiwick.domain.enums import ReconciliationHealth

if TYPE_CHECKING:
    from dojiwick.domain.models.entities.bot_state import BotState


@dataclass(slots=True, frozen=True, kw_only=True)
class HealthState:
    """Immutable snapshot of reconciliation health."""

    health: ReconciliationHealth = ReconciliationHealth.NORMAL
    health_since: datetime | None = None
    frozen_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.health_since is not None and self.health_since.tzinfo is None:
            raise ValueError("health_since must be timezone-aware")

    @classmethod
    def from_bot_state(cls, state: BotState) -> HealthState:
        """Extract health fields from a BotState."""
        return cls(
            health=state.recon_health,
            health_since=state.recon_health_since,
            frozen_symbols=state.recon_frozen_symbols,
        )

    def apply_to(self, state: BotState) -> None:
        """Write health fields back onto a mutable BotState."""
        state.recon_health = self.health
        state.recon_health_since = self.health_since
        state.recon_frozen_symbols = self.frozen_symbols
