"""Binance historical funding rate provider."""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import cast

from dojiwick.domain.models.value_objects.funding_rate import FundingRate

from .boundary import ms_to_utc
from .http_client import BinanceHttpClient

log = logging.getLogger(__name__)

_MAX_ROWS_PER_REQUEST = 1000


def _parse_funding_entry(entry: dict[str, object]) -> FundingRate | None:
    """Parse one /fapi/v1/fundingRate row, or None on bad data."""
    symbol = entry.get("symbol")
    funding_time_ms = entry.get("fundingTime")
    rate_str = entry.get("fundingRate")
    if not isinstance(symbol, str) or not isinstance(funding_time_ms, int | float) or not isinstance(rate_str, str):
        return None
    try:
        rate = Decimal(rate_str)
    except InvalidOperation:
        return None
    return FundingRate(
        symbol=symbol,
        funding_time=ms_to_utc(int(funding_time_ms)),
        rate=rate,
    )


@dataclass(slots=True)
class BinanceFundingRateProvider:
    """Fetches settled funding events from the Binance Futures REST API."""

    client: BinanceHttpClient

    async def fetch_funding_range(self, symbol: str, start: datetime, end: datetime) -> tuple[FundingRate, ...]:
        """Fetch funding events for a date range with automatic pagination."""
        rates: list[FundingRate] = []
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        while start_ms < end_ms:
            raw = await self.client.request_list(
                "GET",
                "/fapi/v1/fundingRate",
                params={
                    "symbol": symbol,
                    "startTime": str(start_ms),
                    "endTime": str(end_ms),
                    "limit": str(_MAX_ROWS_PER_REQUEST),
                },
            )
            batch: list[FundingRate] = []
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                parsed = _parse_funding_entry(cast(dict[str, object], entry))
                if parsed is not None:
                    batch.append(parsed)

            if not batch:
                break
            rates.extend(batch)
            start_ms = int(batch[-1].funding_time.timestamp() * 1000) + 1
            if len(batch) < _MAX_ROWS_PER_REQUEST:
                break

        return tuple(rates)
