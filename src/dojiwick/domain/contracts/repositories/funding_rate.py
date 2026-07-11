"""Funding rate repository protocol."""

from datetime import datetime
from typing import Protocol

from dojiwick.domain.models.value_objects.funding_rate import FundingRate


class FundingRateRepositoryPort(Protocol):
    """Settled funding history persistence."""

    async def upsert_rates(
        self,
        symbol: str,
        rates: tuple[FundingRate, ...],
        *,
        venue: str,
        product: str,
    ) -> int:
        """Insert funding rows (idempotent — settled funding never changes)."""
        ...

    async def get_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        venue: str,
        product: str,
    ) -> tuple[FundingRate, ...]:
        """Return funding events for the symbol/time range, scoped by venue/product."""
        ...
