"""Position exit-state repository protocol."""

from typing import Protocol

from dojiwick.domain.models.entities.position_exit_state import PositionExitState


class PositionExitStateRepositoryPort(Protocol):
    """Persistence for live exit-management state, keyed by position leg."""

    async def upsert(self, state: PositionExitState) -> None:
        """Insert or replace the exit state for a leg."""
        ...

    async def get(self, position_leg_id: int) -> PositionExitState | None:
        """Return the exit state for a leg, or None."""
        ...

    async def list_active(self, account: str) -> tuple[PositionExitState, ...]:
        """Return exit states for all open legs of *account*."""
        ...

    async def delete(self, position_leg_id: int) -> None:
        """Remove the exit state for a closed leg."""
        ...
