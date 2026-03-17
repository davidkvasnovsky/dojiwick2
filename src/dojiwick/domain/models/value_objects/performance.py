"""Performance snapshot domain model."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from dojiwick.domain.numerics import Money


@dataclass(slots=True, frozen=True, kw_only=True)
class PerformanceSnapshot:
    """Periodic equity and PnL snapshot."""

    observed_at: datetime
    equity_usd: Money
    unrealized_pnl_usd: Money = Decimal(0)
    realized_pnl_usd: Money = Decimal(0)
    open_positions: int = 0
    drawdown_pct: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if self.equity_usd <= 0:
            raise ValueError("equity_usd must be positive")
        if self.open_positions < 0:
            raise ValueError("open_positions must be non-negative")
        if self.drawdown_pct < 0:
            raise ValueError("drawdown_pct must be non-negative")
