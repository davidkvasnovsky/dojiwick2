"""Unit tests for BinanceHttpClient — signing, URL selection, error mapping, retries."""

import hashlib
import hmac
from unittest.mock import AsyncMock

import pytest

from dojiwick.domain.errors import AuthenticationError, ExchangeError, NetworkError
from dojiwick.infrastructure.exchange.binance.http_client import BinanceHttpClient
from fixtures.fakes.clock import FixedClock


def _make_client(*, testnet: bool = True, retry_max_attempts: int = 3) -> BinanceHttpClient:
    return BinanceHttpClient(
        api_key="test-key",
        api_secret="test-secret",
        testnet=testnet,
        retry_max_attempts=retry_max_attempts,
        retry_base_delay_sec=0.0,
        clock=FixedClock(),
    )


def _inject_session(client: BinanceHttpClient, session: object) -> None:
    object.__setattr__(client, "_session", session)


class TestSigning:
    def test_hmac_signature_matches_known_vector(self) -> None:
        client = _make_client()
        qs = "symbol=BTCUSDT&side=BUY&timestamp=1234567890"
        expected = hmac.new(b"test-secret", qs.encode(), hashlib.sha256).hexdigest()
        assert client.sign(qs) == expected

    def test_sign_empty_string(self) -> None:
        client = _make_client()
        expected = hmac.new(b"test-secret", b"", hashlib.sha256).hexdigest()
        assert client.sign("") == expected


class TestBaseUrl:
    def test_testnet_url(self) -> None:
        client = _make_client(testnet=True)
        assert "testnet" in client.base_url

    def test_prod_url(self) -> None:
        client = _make_client(testnet=False)
        assert "fapi.binance.com" in client.base_url


class TestRequest:
    async def test_successful_request(self) -> None:
        client = _make_client()
        mock_resp = AsyncMock()
        mock_resp.ok = True
        mock_resp.json = AsyncMock(return_value={"result": "ok"})

        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_resp)
        _inject_session(client, mock_session)

        result = await client.request("GET", "/fapi/v1/ping")
        assert result == {"result": "ok"}

    async def test_auth_error_no_retry(self) -> None:
        client = _make_client()
        mock_resp = AsyncMock()
        mock_resp.ok = False
        mock_resp.status = 401
        mock_resp.json = AsyncMock(return_value={"code": -2015, "msg": "Invalid API key"})

        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_resp)
        _inject_session(client, mock_session)

        with pytest.raises(AuthenticationError, match="-2015"):
            await client.request("GET", "/fapi/v2/account", signed=True)
        assert mock_session.request.call_count == 1

    async def test_backoff_error_retries(self) -> None:
        client = _make_client(retry_max_attempts=2)
        mock_resp = AsyncMock()
        mock_resp.ok = False
        mock_resp.status = 500
        mock_resp.json = AsyncMock(return_value={"code": -1000, "msg": "Unknown error"})

        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_resp)
        _inject_session(client, mock_session)

        with pytest.raises(ExchangeError, match="-1000"):
            await client.request("GET", "/fapi/v1/ping")
        assert mock_session.request.call_count == 2

    async def test_network_error_raises(self) -> None:
        client = _make_client(retry_max_attempts=1)
        mock_session = AsyncMock()

        import aiohttp

        mock_session.request = AsyncMock(side_effect=aiohttp.ClientError("connection lost"))  # pyright: ignore[reportUnknownMemberType]
        _inject_session(client, mock_session)

        with pytest.raises(NetworkError, match="ClientError"):
            await client.request("GET", "/fapi/v1/ping")

    async def test_close_session(self) -> None:
        client = _make_client()
        mock_session = AsyncMock()
        _inject_session(client, mock_session)
        await client.close()
        mock_session.close.assert_awaited_once()


class TestRequestList:
    async def test_successful_list_response(self) -> None:
        client = _make_client()
        mock_resp = AsyncMock()
        mock_resp.ok = True
        mock_resp.json = AsyncMock(return_value=[{"symbol": "BTCUSDT", "price": "50000.00"}])

        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_resp)
        _inject_session(client, mock_session)

        result = await client.request_list("GET", "/fapi/v2/ticker/price")
        assert len(result) == 1

    async def test_non_list_response_raises(self) -> None:
        client = _make_client()
        mock_resp = AsyncMock()
        mock_resp.ok = True
        mock_resp.json = AsyncMock(return_value={"not": "a list"})

        mock_session = AsyncMock()
        mock_session.request = AsyncMock(return_value=mock_resp)
        _inject_session(client, mock_session)

        with pytest.raises(NetworkError, match="expected list"):
            await client.request_list("GET", "/fapi/v2/ticker/price")
