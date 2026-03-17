"""Bot state repository test doubles."""

from dataclasses import dataclass, field

from dojiwick.domain.models.entities.bot_state import BotState


@dataclass(slots=True)
class InMemoryBotStateRepo:
    """In-memory bot state for tests."""

    _state: BotState = field(default_factory=BotState)

    async def get_state(self) -> BotState:
        return BotState(
            consecutive_errors=self._state.consecutive_errors,
            consecutive_losses=self._state.consecutive_losses,
            daily_trade_count=self._state.daily_trade_count,
            daily_pnl_usd=self._state.daily_pnl_usd,
            circuit_breaker_active=self._state.circuit_breaker_active,
            circuit_breaker_until=self._state.circuit_breaker_until,
            last_tick_at=self._state.last_tick_at,
            last_decay_at=self._state.last_decay_at,
            daily_reset_at=self._state.daily_reset_at,
            recon_health=self._state.recon_health,
            recon_health_since=self._state.recon_health_since,
            recon_frozen_symbols=self._state.recon_frozen_symbols,
        )

    async def update_state(self, state: BotState) -> None:
        self._state = state


class FailingBotStateRepo:
    """Raises on all operations."""

    async def get_state(self) -> BotState:
        raise RuntimeError("bot state repo failure")

    async def update_state(self, state: BotState) -> None:
        del state
        raise RuntimeError("bot state repo failure")
