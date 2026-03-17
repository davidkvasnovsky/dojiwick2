"""Context provider that reads from an ExchangeCache for the tick cycle.

Replaces direct REST/WS calls with atomic cache reads, guaranteeing
consistent point-in-time snapshots with no torn reads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from dojiwick.domain.errors import AdapterError, DataQualityError
from dojiwick.domain.indicator_schema import INDICATOR_COUNT
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchMarketSnapshot,
    BatchPortfolioSnapshot,
)
from dojiwick.domain.symbols import pair_to_symbol
from dojiwick.infrastructure.exchange.cache import ExchangeCache

if TYPE_CHECKING:
    from dojiwick.infrastructure.exchange.indicator_enricher import IndicatorEnricher

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CachedContextProvider:
    """Reads atomic snapshots from the exchange cache for the tick loop.

    The cache is populated by the ``ExchangeDataFeed`` (WS or REST).
    This provider guarantees that market + portfolio data are captured
    at the same point in time — no torn reads.
    """

    cache: ExchangeCache
    pair_separator: str = "/"
    indicator_enricher: IndicatorEnricher | None = None
    _day_start_equity: float | None = None

    async def fetch_context_batch(
        self,
        pairs: tuple[str, ...],
        at: datetime,
    ) -> BatchDecisionContext:
        """Return a batch context by reading the cache snapshot atomically."""
        snap = await self.cache.snapshot()
        size = len(pairs)
        symbols = tuple(pair_to_symbol(pair, self.pair_separator) for pair in pairs)

        raw_prices = [snap.prices.get(symbol) for symbol in symbols]
        missing = [s for s, p in zip(symbols, raw_prices) if p is None or float(p) == 0.0]
        if missing:
            raise DataQualityError(f"zero or missing prices for: {', '.join(missing)}")
        prices = np.array([float(p) for p in raw_prices if p is not None], dtype=np.float64)

        indicators = np.zeros((size, INDICATOR_COUNT), dtype=np.float64)
        if self.indicator_enricher is not None:
            try:
                indicators = await self.indicator_enricher.compute_for_pairs(symbols)
            except (OSError, AdapterError):  # fmt: skip
                log.warning("indicator enrichment failed, falling back to zeros", exc_info=True)

        account = snap.account
        if account is not None:
            equity = float(account.total_wallet_balance)
            unrealized = float(account.total_unrealized_pnl)

            position_map: dict[str, float] = {}
            for leg in account.positions:
                position_map[leg.instrument_id.symbol] = float(leg.quantity)

            if self._day_start_equity is None:
                self._day_start_equity = equity

            equity_arr = np.full(size, equity, dtype=np.float64)
            day_start_arr = np.full(size, self._day_start_equity, dtype=np.float64)
            total_open = len(position_map)
            open_total = np.full(size, total_open, dtype=np.int64)
            has_open = np.array(
                [symbol in position_map for symbol in symbols],
                dtype=np.bool_,
            )
            pnl_arr = np.full(size, unrealized, dtype=np.float64)
        else:
            equity_arr = np.zeros(size, dtype=np.float64)
            day_start_arr = np.zeros(size, dtype=np.float64)
            open_total = np.zeros(size, dtype=np.int64)
            has_open = np.zeros(size, dtype=np.bool_)
            pnl_arr = np.zeros(size, dtype=np.float64)

        return BatchDecisionContext(
            market=BatchMarketSnapshot(
                pairs=pairs,
                observed_at=at,
                price=prices,
                indicators=indicators,
                asof_timestamp=snap.captured_at,
            ),
            portfolio=BatchPortfolioSnapshot(
                equity_usd=equity_arr,
                day_start_equity_usd=day_start_arr,
                open_positions_total=open_total,
                has_open_position=has_open,
                unrealized_pnl_usd=pnl_arr,
            ),
        )
