"""Context provider test doubles."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from dojiwick.domain.contracts.gateways.context_provider import ContextProviderPort
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchMarketSnapshot,
    BatchPortfolioSnapshot,
)


class FailingContextProvider(ContextProviderPort):
    """Raises on fetch_context_batch."""

    async def fetch_context_batch(self, pairs: tuple[str, ...], at: datetime) -> BatchDecisionContext:
        del pairs, at
        raise RuntimeError("context provider failure")


class TimeoutContextProvider(ContextProviderPort):
    """Raises TimeoutError on fetch_context_batch."""

    async def fetch_context_batch(self, pairs: tuple[str, ...], at: datetime) -> BatchDecisionContext:
        del pairs, at
        raise TimeoutError("context provider timed out")


@dataclass(slots=True)
class StaticBatchContextProvider(ContextProviderPort):
    """Returns one static batch context, reordered by requested pairs."""

    context: BatchDecisionContext

    async def fetch_context_batch(self, pairs: tuple[str, ...], at: datetime) -> BatchDecisionContext:
        """Return subset context matching requested pair order."""

        del at
        index_map = {pair: index for index, pair in enumerate(self.context.market.pairs)}
        try:
            indices = np.array([index_map[pair] for pair in pairs], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"pair not found in static context: {exc}") from exc

        market = self.context.market
        portfolio = self.context.portfolio
        return BatchDecisionContext(
            market=BatchMarketSnapshot(
                pairs=pairs,
                observed_at=market.observed_at,
                price=market.price[indices],
                indicators=market.indicators[indices, :],
            ),
            portfolio=BatchPortfolioSnapshot(
                equity_usd=portfolio.equity_usd[indices],
                day_start_equity_usd=portfolio.day_start_equity_usd[indices],
                open_positions_total=portfolio.open_positions_total[indices],
                has_open_position=portfolio.has_open_position[indices],
                unrealized_pnl_usd=portfolio.unrealized_pnl_usd[indices],
            ),
        )
