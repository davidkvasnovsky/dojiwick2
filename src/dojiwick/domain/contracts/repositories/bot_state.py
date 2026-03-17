"""Bot state repository protocol."""

from typing import Protocol

from dojiwick.domain.models.entities.bot_state import BotState


class BotStateRepositoryPort(Protocol):
    """Circuit breaker and operational state persistence."""

    async def get_state(self) -> BotState:
        """Return current bot state (singleton row)."""
        ...

    async def update_state(self, state: BotState) -> None:
        """Persist the full bot state."""
        ...
