"""Tests for Postgres cache wiring in build_market_data_fetcher."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fixtures.factories.infrastructure import SettingsBuilder
from psycopg import OperationalError

from dojiwick.config.composition import build_market_data_fetcher
from dojiwick.config.schema import Settings

_READINESS = "dojiwick.infrastructure.exchange.binance.readiness.assert_binance_ready"
_CONNECT = "dojiwick.infrastructure.postgres.connection.connect"


@pytest.fixture
def settings() -> Settings:
    return SettingsBuilder().build()


def _patch_exchange():
    """Patch Binance readiness + HTTP client so no real network calls are made."""
    return (
        patch(_READINESS, return_value=("key", "secret")),
        patch("dojiwick.config.composition._build_http_client"),
    )


@pytest.mark.asyncio
async def test_cache_wrapping_on_success(settings: Settings) -> None:
    """When DB connects successfully, providers are wrapped in caching fetchers."""
    mock_conn = AsyncMock()
    mock_conn.close = AsyncMock()
    p_ready, p_http = _patch_exchange()

    with p_ready, p_http as mock_http, patch(_CONNECT, return_value=mock_conn) as mock_connect:
        mock_http.return_value = AsyncMock()
        mock_http.return_value.close = AsyncMock()

        fetchers, cleanup = await build_market_data_fetcher(settings, use_cache=True)

        mock_connect.assert_awaited_once_with(settings.database)

        from dojiwick.application.services.caching_candle_fetcher import CachingCandleFetcher

        assert isinstance(fetchers.candles, CachingCandleFetcher)

        await cleanup()
        mock_conn.close.assert_awaited_once()
        mock_http.return_value.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cache_fallback_on_db_failure(settings: Settings) -> None:
    """When DB connect fails with a connection error, falls back to raw provider."""
    p_ready, p_http = _patch_exchange()

    with p_ready, p_http as mock_http, patch(_CONNECT, side_effect=OperationalError("db down")):
        mock_http.return_value = AsyncMock()
        mock_http.return_value.close = AsyncMock()

        fetchers, cleanup = await build_market_data_fetcher(settings, use_cache=True)

        from dojiwick.application.services.caching_candle_fetcher import CachingCandleFetcher

        assert not isinstance(fetchers.candles, CachingCandleFetcher)

        await cleanup()
        mock_http.return_value.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_programming_errors_surface(settings: Settings) -> None:
    """Non-connection errors during cache setup must propagate, not silently disable caching."""
    p_ready, p_http = _patch_exchange()

    with p_ready, p_http as mock_http, patch(_CONNECT, side_effect=TypeError("bug")):
        mock_http.return_value = AsyncMock()
        mock_http.return_value.close = AsyncMock()

        with pytest.raises(TypeError, match="bug"):
            await build_market_data_fetcher(settings, use_cache=True)


@pytest.mark.asyncio
async def test_no_cache_flag_skips_db(settings: Settings) -> None:
    """When use_cache=False, no DB connection is attempted."""
    p_ready, p_http = _patch_exchange()

    with p_ready, p_http as mock_http, patch(_CONNECT) as mock_connect:
        mock_http.return_value = AsyncMock()
        mock_http.return_value.close = AsyncMock()

        _fetchers, cleanup = await build_market_data_fetcher(settings, use_cache=False)

        mock_connect.assert_not_awaited()

        await cleanup()
