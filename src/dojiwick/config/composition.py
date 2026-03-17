"""Config-driven adapter composition — dispatches on VenueCode."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dojiwick.infrastructure.exchange.binance.http_client import BinanceHttpClient

from dojiwick.application.orchestration.execution_planner import DefaultExecutionPlanner
from dojiwick.config.schema import Settings
from dojiwick.domain.contracts.gateways.account_state import AccountStatePort
from dojiwick.domain.contracts.gateways.context_provider import ContextProviderPort
from dojiwick.domain.contracts.gateways.execution import ExecutionGatewayPort
from dojiwick.domain.contracts.gateways.execution_planner import ExecutionPlannerPort
from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.exchange_metadata import ExchangeMetadataPort
from dojiwick.domain.contracts.gateways.open_order import OpenOrderPort
from dojiwick.domain.contracts.gateways.order_event_stream import OrderEventStreamPort
from dojiwick.domain.type_aliases import VenueCode
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_VENUE
from dojiwick.domain.errors import ConfigurationError

from dojiwick.config.targets import resolve_execution_symbols

from dojiwick.domain.contracts.gateways.historical_candle_source import HistoricalCandleSourcePort
from dojiwick.domain.contracts.gateways.market_data_feed import MarketDataFeedPort

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ComposedAdapters:
    """Bundle of adapter instances selected by configuration."""

    context_provider: ContextProviderPort
    execution_gateway: ExecutionGatewayPort
    execution_planner: ExecutionPlannerPort
    exchange_metadata: ExchangeMetadataPort
    account_state: AccountStatePort
    open_order_port: OpenOrderPort | None = None
    feed: MarketDataFeedPort | None = None
    order_stream: OrderEventStreamPort | None = None
    _cleanup: Callable[[], Awaitable[None]] | None = field(default=None, repr=False)

    async def close(self) -> None:
        """Gracefully shut down feed and exchange resources."""
        if self.feed is not None:
            await self.feed.stop()
        if self._cleanup is not None:
            await self._cleanup()


AdapterBuilder = Callable[[Settings, ClockPort], ComposedAdapters]


def _build_http_client(
    settings: Settings,
    clock: ClockPort,
    *,
    api_key: str,
    api_secret: str,
) -> BinanceHttpClient:
    """Construct a ``BinanceHttpClient`` from *settings* — single source of truth."""
    from dojiwick.infrastructure.exchange.binance.http_client import BinanceHttpClient

    return BinanceHttpClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=settings.exchange.testnet,
        recv_window_ms=settings.exchange.recv_window_ms,
        connect_timeout_sec=settings.exchange.connect_timeout_sec,
        read_timeout_sec=settings.exchange.read_timeout_sec,
        retry_max_attempts=settings.exchange.retry_max_attempts,
        retry_base_delay_sec=settings.exchange.retry_base_delay_sec,
        backoff_factor=settings.exchange.backoff_factor,
        rate_limit_per_sec=settings.exchange.rate_limit_per_sec,
        clock=clock,
    )


def _build_binance_adapters(settings: Settings, clock: ClockPort) -> ComposedAdapters:
    """Build Binance adapter instances with HTTP client and read-only adapters."""
    from dojiwick.infrastructure.exchange.binance.readiness import assert_binance_ready

    api_key, api_secret = assert_binance_ready(
        api_key_env=settings.exchange.api_key_env,
        api_secret_env=settings.exchange.api_secret_env,
    )

    from dojiwick.infrastructure.exchange.binance.account_state import BinanceAccountStateProvider
    from dojiwick.infrastructure.exchange.binance.execution import BinanceExecutionGateway
    from dojiwick.infrastructure.exchange.binance.open_order import BinanceOpenOrderAdapter
    from dojiwick.infrastructure.exchange.binance.exchange_metadata import BinanceExchangeMetadataProvider
    from dojiwick.infrastructure.exchange.binance.market_data import BinanceMarketDataProvider
    from dojiwick.infrastructure.exchange.binance.order_event_stream import BinanceOrderEventStream
    from dojiwick.infrastructure.exchange.cached_context_provider import CachedContextProvider
    from dojiwick.infrastructure.exchange.cache import ExchangeCache
    from dojiwick.infrastructure.exchange.feed import ExchangeDataFeed

    http_client = _build_http_client(settings, clock, api_key=api_key, api_secret=api_secret)

    execution_planner: ExecutionPlannerPort = DefaultExecutionPlanner(
        position_mode=settings.exchange.position_mode,
    )

    symbols = resolve_execution_symbols(settings)
    cache = ExchangeCache(clock=clock)
    account_state = BinanceAccountStateProvider(client=http_client)
    market_data = BinanceMarketDataProvider(client=http_client)
    order_stream = BinanceOrderEventStream(client=http_client, clock=clock)

    async def _cleanup() -> None:
        await http_client.close()

    from dojiwick.infrastructure.exchange.indicator_enricher import IndicatorEnricher
    from dojiwick.domain.type_aliases import CandleInterval

    indicator_enricher = IndicatorEnricher(
        market_data=market_data,
        candle_interval=CandleInterval(settings.trading.candle_interval),
        candle_lookback=settings.trading.candle_lookback,
        rsi_period=settings.trading.rsi_period,
        ema_fast_period=settings.trading.ema_fast_period,
        ema_slow_period=settings.trading.ema_slow_period,
        ema_base_period=settings.trading.ema_base_period,
        ema_trend_period=settings.trading.ema_trend_period,
        atr_period=settings.trading.atr_period,
        adx_period=settings.trading.adx_period,
        bb_period=settings.trading.bb_period,
        bb_std=settings.trading.bb_std,
        volume_ema_period=settings.trading.volume_ema_period,
    )

    return ComposedAdapters(
        context_provider=CachedContextProvider(
            cache=cache,
            pair_separator=settings.universe.pair_separator,
            indicator_enricher=indicator_enricher,
        ),
        execution_gateway=BinanceExecutionGateway(client=http_client),
        execution_planner=execution_planner,
        exchange_metadata=BinanceExchangeMetadataProvider(client=http_client),
        account_state=account_state,
        open_order_port=BinanceOpenOrderAdapter(client=http_client),
        order_stream=order_stream,
        feed=ExchangeDataFeed(
            cache=cache,
            market_data=market_data,
            account_state=account_state,
            order_stream=order_stream,
            pairs=symbols,
            account=settings.universe.account,
        ),
        _cleanup=_cleanup,
    )


# --- Venue builder registry ---
# To add a new exchange:
# 1. Create infrastructure/exchange/<venue>/ with adapters implementing existing ports
# 2. Create infrastructure/exchange/<venue>/readiness.py with assert_<venue>_ready()
# 3. Write _build_<venue>_adapters() in this module
# 4. Register it in _VENUE_BUILDERS below
_VENUE_BUILDERS: dict[VenueCode, AdapterBuilder] = {
    BINANCE_VENUE: _build_binance_adapters,
}


async def build_market_data_fetcher(
    settings: Settings,
    clock: ClockPort | None = None,
    *,
    use_cache: bool = True,
) -> tuple[HistoricalCandleSourcePort, Callable[[], Awaitable[None]]]:
    """Build a market data fetcher + cleanup callback for backtest CLIs.

    Returns ``(provider, cleanup)`` where *provider* satisfies
    ``MarketDataFetcher`` and *cleanup* closes the HTTP client.

    When *use_cache* is True, wraps the provider with a Postgres-backed
    ``CachingCandleFetcher`` for DB-first candle retrieval.
    """
    from dojiwick.infrastructure.exchange.binance.readiness import assert_binance_ready
    from dojiwick.infrastructure.exchange.binance.market_data import BinanceMarketDataProvider
    from dojiwick.infrastructure.system.clock import SystemClock

    effective_clock = clock or SystemClock()
    api_key, api_secret = assert_binance_ready(
        api_key_env=settings.exchange.api_key_env,
        api_secret_env=settings.exchange.api_secret_env,
    )
    http_client = _build_http_client(settings, effective_clock, api_key=api_key, api_secret=api_secret)
    provider: HistoricalCandleSourcePort = BinanceMarketDataProvider(client=http_client)

    if use_cache:
        db_conn = None
        try:
            from dojiwick.infrastructure.postgres.connection import connect
            from dojiwick.infrastructure.postgres.repositories.candle import PgCandleRepository
            from dojiwick.application.services.caching_candle_fetcher import CachingCandleFetcher

            db_conn = await connect(settings.database)
            candle_repo = PgCandleRepository(connection=db_conn)
            provider = CachingCandleFetcher(
                candle_repo=candle_repo,
                fetcher=provider,
                venue=str(settings.exchange.venue),
                product=str(settings.exchange.product),
            )
            log.info("postgres candle cache enabled")

            async def _cleanup_with_db() -> None:
                await db_conn.close()
                await http_client.close()

            return provider, _cleanup_with_db
        except Exception:
            if db_conn is not None:
                await db_conn.close()
            log.warning("postgres cache unavailable, falling back to direct exchange fetch", exc_info=True)

    return provider, http_client.close


def build_adapters(settings: Settings, *, clock: ClockPort) -> ComposedAdapters:
    """Build adapter instances for the configured exchange venue."""
    venue = settings.exchange.venue
    builder = _VENUE_BUILDERS.get(venue)
    if builder is None:
        supported = ", ".join(_VENUE_BUILDERS)
        raise ConfigurationError(f"unsupported exchange venue: {venue} (supported: {supported})")
    return builder(settings, clock)
