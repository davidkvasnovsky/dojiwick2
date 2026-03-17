"""Per-pair trading state for cooldown and win-rate cutoff risk gates."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, kw_only=True)
class PairTradingState:
    """Mutable per-pair performance tracker."""

    pair: str
    target_id: str
    venue: str
    product: str
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    last_trade_at: datetime | None = None
    blocked: bool = False

    def __post_init__(self) -> None:
        if not self.pair:
            raise ValueError("pair must not be empty")
        if not self.target_id:
            raise ValueError("target_id must not be empty")
        if not self.venue:
            raise ValueError("venue must not be empty")
        if not self.product:
            raise ValueError("product must not be empty")
        if self.wins < 0:
            raise ValueError("wins must be non-negative")
        if self.losses < 0:
            raise ValueError("losses must be non-negative")
        if self.consecutive_losses < 0:
            raise ValueError("consecutive_losses must be non-negative")

    @property
    def total_trades(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades
