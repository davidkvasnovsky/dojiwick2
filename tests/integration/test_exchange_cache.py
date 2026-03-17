"""Integration tests for ExchangeCache, ExchangeDataFeed, and CachedContextProvider."""

from datetime import UTC, datetime
from decimal import Decimal

from dojiwick.domain.enums import PositionSide
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.contracts.gateways.order_event_stream import StreamCursor, StreamGap
from dojiwick.domain.models.value_objects.account_state import (
    AccountBalance,
    AccountSnapshot,
    ExchangePositionLeg,
)
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.order_event import OrderEvent
from dojiwick.infrastructure.exchange.cache import ExchangeCache
from dojiwick.infrastructure.exchange.cached_context_provider import CachedContextProvider
from dojiwick.infrastructure.exchange.feed import ExchangeDataFeed, FeedStatus
from fixtures.fakes.account_state import FakeAccountState
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.market_data_provider import InMemoryMarketDataProvider
from fixtures.fakes.order_event_stream import InMemoryOrderEventStream


def _btc_instrument() -> InstrumentId:
    return InstrumentId(
        venue=BINANCE_VENUE,
        product=BINANCE_USD_C,
        symbol="BTCUSDC",
        base_asset="BTC",
        quote_asset="USDC",
        settle_asset="USDC",
    )


def _sample_snapshot(account: str = "default") -> AccountSnapshot:
    return AccountSnapshot(
        account=account,
        balances=(AccountBalance(asset="USDC", wallet_balance=Decimal(10_000), available_balance=Decimal(5_000)),),
        positions=(
            ExchangePositionLeg(
                instrument_id=_btc_instrument(),
                position_side=PositionSide.LONG,
                quantity=Decimal("0.5"),
                entry_price=Decimal("95000"),
                unrealized_pnl=Decimal("250"),
            ),
        ),
        total_wallet_balance=Decimal(10_000),
        available_balance=Decimal(5_000),
        total_unrealized_pnl=Decimal("250"),
    )


def _make_market_data(prices: dict[str, float]) -> InMemoryMarketDataProvider:
    provider = InMemoryMarketDataProvider()
    provider.set_prices(prices)
    return provider


# ExchangeCache snapshot consistency tests


async def test_cache_snapshot_returns_consistent_prices() -> None:
    """Cache snapshot returns all prices atomically."""
    cache = ExchangeCache(clock=FixedClock())
    await cache.update_prices({"BTCUSDC": 95_000.0, "ETHUSDC": 3_400.0})

    snap = await cache.snapshot()
    assert snap.prices == {"BTCUSDC": 95_000.0, "ETHUSDC": 3_400.0}


async def test_cache_snapshot_returns_account_state() -> None:
    """Cache snapshot returns account snapshot atomically."""
    cache = ExchangeCache(clock=FixedClock())
    account = _sample_snapshot()
    await cache.update_account(account)

    snap = await cache.snapshot()
    assert snap.account is not None
    assert snap.account.account == "default"
    assert len(snap.account.positions) == 1


async def test_cache_snapshot_is_immutable_copy() -> None:
    """Modifying the source dict after snapshot does not affect the snapshot."""
    cache = ExchangeCache(clock=FixedClock())
    await cache.update_prices({"BTCUSDC": 95_000.0})
    snap = await cache.snapshot()

    await cache.update_prices({"BTCUSDC": 96_000.0})
    snap2 = await cache.snapshot()

    assert snap.prices["BTCUSDC"] == 95_000.0
    assert snap2.prices["BTCUSDC"] == 96_000.0


async def test_cache_clear_resets_state() -> None:
    """Clear resets cache to empty."""
    cache = ExchangeCache(clock=FixedClock())
    await cache.update_prices({"BTCUSDC": 95_000.0})
    assert cache.has_data

    await cache.clear()
    assert not cache.has_data

    snap = await cache.snapshot()
    assert snap.prices == {}
    assert snap.account is None


# CachedContextProvider reads from cache


async def test_cached_context_provider_reads_prices_from_cache() -> None:
    """CachedContextProvider reads prices from cache snapshot."""
    cache = ExchangeCache(clock=FixedClock())
    await cache.update_prices({"BTCUSDC": 95_000.0, "ETHUSDC": 3_400.0})
    await cache.update_account(_sample_snapshot())

    provider = CachedContextProvider(cache=cache)
    context = await provider.fetch_context_batch(("BTCUSDC", "ETHUSDC"), datetime.now(UTC))

    assert context.size == 2
    assert context.market.price[0] == 95_000.0
    assert context.market.price[1] == 3_400.0


async def test_cached_context_provider_normalizes_pair_keys_to_symbols() -> None:
    """Configured pairs (BASE/QUOTE) map to symbol-keyed cache prices."""
    cache = ExchangeCache(clock=FixedClock())
    await cache.update_prices({"BTCUSDC": 95_000.0})
    await cache.update_account(_sample_snapshot())

    provider = CachedContextProvider(cache=cache, pair_separator="/")
    context = await provider.fetch_context_batch(("BTC/USDC",), datetime.now(UTC))

    assert context.market.price[0] == 95_000.0


async def test_cached_context_provider_includes_portfolio_from_account() -> None:
    """CachedContextProvider builds portfolio from cached account state."""
    cache = ExchangeCache(clock=FixedClock())
    await cache.update_prices({"BTCUSDC": 95_000.0})
    await cache.update_account(_sample_snapshot())

    provider = CachedContextProvider(cache=cache)
    context = await provider.fetch_context_batch(("BTCUSDC",), datetime.now(UTC))

    assert context.portfolio.equity_usd[0] == 10_000.0
    assert context.portfolio.has_open_position[0]


async def test_cached_context_provider_works_without_account() -> None:
    """CachedContextProvider returns zero portfolio when no account is cached."""
    cache = ExchangeCache(clock=FixedClock())
    await cache.update_prices({"BTCUSDC": 95_000.0})

    provider = CachedContextProvider(cache=cache)
    context = await provider.fetch_context_batch(("BTCUSDC",), datetime.now(UTC))

    assert context.portfolio.equity_usd[0] == 0.0
    assert not context.portfolio.has_open_position[0]


# ExchangeDataFeed lifecycle tests


async def test_feed_bootstrap_populates_cache_via_rest() -> None:
    """Bootstrap populates cache from REST (market data + account state)."""
    cache = ExchangeCache(clock=FixedClock())
    market_data = _make_market_data({"BTCUSDC": 95_000.0, "ETHUSDC": 3_400.0})

    account_state = FakeAccountState()
    account_state.set_snapshot("default", _sample_snapshot())

    feed = ExchangeDataFeed(
        cache=cache,
        market_data=market_data,
        account_state=account_state,
        order_stream=InMemoryOrderEventStream(),
        pairs=("BTCUSDC", "ETHUSDC"),
        account="default",
    )

    await feed.bootstrap()

    assert cache.has_data
    snap = await cache.snapshot()
    assert snap.prices["BTCUSDC"] == 95_000.0
    assert snap.account is not None
    assert feed.status == FeedStatus.BOOTSTRAPPING


async def test_feed_start_connects_ws_when_available() -> None:
    """Start connects WS and sets status to WS_ACTIVE."""
    cache = ExchangeCache(clock=FixedClock())
    market_data = InMemoryMarketDataProvider()
    account_state = FakeAccountState()
    account_state.set_snapshot("default", _sample_snapshot())

    order_stream = InMemoryOrderEventStream()

    feed = ExchangeDataFeed(
        cache=cache,
        market_data=market_data,
        account_state=account_state,
        order_stream=order_stream,
        pairs=("BTCUSDC",),
        account="default",
    )

    await feed.start()

    assert feed.status == FeedStatus.WS_ACTIVE
    assert order_stream.is_connected


async def test_feed_falls_back_to_rest_when_ws_unavailable() -> None:
    """Start falls back to REST when WS connect fails."""

    class FailingStream:
        _connected = False

        @property
        def stream_name(self) -> str:
            return "fail"

        async def connect(self) -> None:
            raise ConnectionError("WS unavailable")

        async def disconnect(self) -> None:
            self._connected = False

        @property
        def is_connected(self) -> bool:
            return self._connected

        async def detect_gaps(self, since: StreamCursor) -> tuple[StreamGap, ...]:
            _ = since
            return ()

        async def recover_gap(self, gap: StreamGap) -> tuple[OrderEvent, ...]:
            _ = gap
            return ()

        async def get_cursor(self) -> StreamCursor:
            return StreamCursor(stream_name="fail", sequence=0, timestamp_ms=0)

    cache = ExchangeCache(clock=FixedClock())
    market_data = InMemoryMarketDataProvider()
    account_state = FakeAccountState()
    account_state.set_snapshot("default", _sample_snapshot())

    feed = ExchangeDataFeed(
        cache=cache,
        market_data=market_data,
        account_state=account_state,
        order_stream=FailingStream(),
        pairs=("BTCUSDC",),
        account="default",
    )

    await feed.start()

    assert feed.status == FeedStatus.REST_FALLBACK


async def test_feed_ensure_fresh_refreshes_on_rest_fallback() -> None:
    """ensure_fresh() refreshes via REST when in REST_FALLBACK mode."""
    cache = ExchangeCache(clock=FixedClock())
    market_data = _make_market_data({"BTCUSDC": 95_000.0})

    account_state = FakeAccountState()
    account_state.set_snapshot("default", _sample_snapshot())

    feed = ExchangeDataFeed(
        cache=cache,
        market_data=market_data,
        account_state=account_state,
        order_stream=InMemoryOrderEventStream(),
        pairs=("BTCUSDC",),
        account="default",
    )
    feed._status = FeedStatus.REST_FALLBACK  # pyright: ignore[reportPrivateUsage]

    await feed.ensure_fresh()

    snap = await cache.snapshot()
    assert snap.prices["BTCUSDC"] == 95_000.0


async def test_feed_stop_disconnects_ws() -> None:
    """Stop disconnects WS and sets status to DISCONNECTED."""
    cache = ExchangeCache(clock=FixedClock())
    market_data = InMemoryMarketDataProvider()
    account_state = FakeAccountState()
    account_state.set_snapshot("default", _sample_snapshot())

    order_stream = InMemoryOrderEventStream()

    feed = ExchangeDataFeed(
        cache=cache,
        market_data=market_data,
        account_state=account_state,
        order_stream=order_stream,
        pairs=("BTCUSDC",),
        account="default",
    )

    await feed.start()
    assert order_stream.is_connected

    await feed.stop()
    assert feed.status == FeedStatus.DISCONNECTED
    assert not order_stream.is_connected


# Gap detection and recovery


async def test_feed_gap_recovery_returns_zero_when_no_gaps() -> None:
    """check_and_recover_gaps returns 0 when no gaps detected."""
    cache = ExchangeCache(clock=FixedClock())
    market_data = _make_market_data({"BTCUSDC": 95_000.0})

    account_state = FakeAccountState()
    account_state.set_snapshot("default", _sample_snapshot())

    order_stream = InMemoryOrderEventStream()

    feed = ExchangeDataFeed(
        cache=cache,
        market_data=market_data,
        account_state=account_state,
        order_stream=order_stream,
        pairs=("BTCUSDC",),
        account="default",
    )

    await feed.start()
    recovered = await feed.check_and_recover_gaps()
    assert recovered == ()
