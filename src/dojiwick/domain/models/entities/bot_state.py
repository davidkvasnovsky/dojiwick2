"""Bot state domain model."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from dojiwick.domain.enums import ReconciliationHealth
from dojiwick.domain.numerics import Money


@dataclass(slots=True, kw_only=True)
class BotState:
    """Mutable circuit breaker and operational state — changes per tick."""

    consecutive_errors: int = 0
    consecutive_losses: int = 0
    daily_trade_count: int = 0
    daily_pnl_usd: Money = Decimal(0)
    circuit_breaker_active: bool = False
    circuit_breaker_until: datetime | None = None
    last_tick_at: datetime | None = None
    last_decay_at: datetime | None = None
    daily_reset_at: datetime | None = None
    recon_health: ReconciliationHealth = ReconciliationHealth.NORMAL
    recon_health_since: datetime | None = None
    recon_frozen_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.consecutive_errors < 0:
            raise ValueError("consecutive_errors must be non-negative")
        if self.consecutive_losses < 0:
            raise ValueError("consecutive_losses must be non-negative")
        if self.daily_trade_count < 0:
            raise ValueError("daily_trade_count must be non-negative")
