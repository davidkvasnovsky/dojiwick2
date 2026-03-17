"""Binance Futures REST HTTP client with HMAC signing, rate limiting, and retry.

Lazy-imports aiohttp (optional dependency) inside methods to match the
pattern used by anthropic_client.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.errors import AdapterError, NetworkError

from .error_mapping import RetryCategory, RetryPolicy, map_binance_error

if TYPE_CHECKING:
    import aiohttp

log = logging.getLogger(__name__)

_PROD_BASE_URL = "https://fapi.binance.com"
_TESTNET_BASE_URL = "https://testnet.binancefuture.com"


@dataclass(slots=True)
class _TokenBucket:
    """Simple async token-bucket rate limiter."""

    rate: float
    clock: ClockPort
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _tokens: float = field(default=0.0, init=False)
    _last_refill_ns: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._tokens = self.rate
        self._last_refill_ns = self.clock.monotonic_ns()

    async def acquire(self) -> None:
        async with self._lock:
            now_ns = self.clock.monotonic_ns()
            elapsed = (now_ns - self._last_refill_ns) / 1_000_000_000
            self._tokens = min(self.rate, self._tokens + elapsed * self.rate)
            self._last_refill_ns = now_ns
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self.rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
                self._last_refill_ns = self.clock.monotonic_ns()
            else:
                self._tokens -= 1.0


@dataclass(slots=True)
class BinanceHttpClient:
    """Async HTTP client for Binance Futures REST API."""

    api_key: str
    api_secret: str
    clock: ClockPort
    testnet: bool = True
    recv_window_ms: int = 5000
    connect_timeout_sec: float = 10.0
    read_timeout_sec: float = 10.0
    retry_max_attempts: int = 3
    retry_base_delay_sec: float = 0.5
    backoff_factor: float = 2.0
    rate_limit_per_sec: int = 10

    _session: aiohttp.ClientSession | None = field(default=None, init=False, repr=False)
    _bucket: _TokenBucket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._bucket = _TokenBucket(rate=float(self.rate_limit_per_sec), clock=self.clock)

    @property
    def base_url(self) -> str:
        return _TESTNET_BASE_URL if self.testnet else _PROD_BASE_URL

    def sign(self, query_string: str) -> str:
        """HMAC-SHA256 sign a query string with the API secret."""
        return hmac.new(self.api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

    async def ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            import aiohttp as _aiohttp

            timeout = _aiohttp.ClientTimeout(
                sock_connect=self.connect_timeout_sec,
                sock_read=self.read_timeout_sec,
            )
            self._session = _aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _prepare_request(
        self,
        path: str,
        params: dict[str, str] | None,
        signed: bool,
    ) -> tuple[str, dict[str, str], dict[str, str]]:
        """Build URL, query params, and headers for a Binance API request."""
        query_params = dict(params) if params else {}
        headers: dict[str, str] = {}
        if signed:
            query_params["timestamp"] = str(self.clock.epoch_ms())
            query_params["recvWindow"] = str(self.recv_window_ms)
            qs = "&".join(f"{k}={v}" for k, v in query_params.items())
            query_params["signature"] = self.sign(qs)
            headers["X-MBX-APIKEY"] = self.api_key
        url = f"{self.base_url}{path}"
        return url, query_params, headers

    @staticmethod
    def _parse_error_body(raw: object) -> tuple[AdapterError, RetryPolicy, int] | None:
        """Extract a Binance error from a raw JSON response body."""
        if not isinstance(raw, dict):
            return None
        body = cast(dict[str, object], raw)
        code = body.get("code", -1)
        if not isinstance(code, int):
            return None
        msg = str(body.get("msg", ""))
        exc, policy = map_binance_error(code, msg)
        return exc, policy, code

    async def _with_retry(
        self,
        method: str,
        url: str,
        query_params: dict[str, str],
        headers: dict[str, str],
    ) -> object:
        """Execute HTTP request with retry and exponential backoff."""
        import aiohttp as _aiohttp

        session = await self.ensure_session()
        last_exc: Exception | None = None

        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                resp = await session.request(method, url, params=query_params, headers=headers)
                raw: object = await resp.json()

                if resp.ok:
                    return raw

                result = self._parse_error_body(raw)
                if result is not None:
                    exc, policy, code = result
                    if policy.category == RetryCategory.NO_RETRY:
                        raise exc
                    if attempt < self.retry_max_attempts and policy.category in (
                        RetryCategory.BACKOFF_RETRY,
                        RetryCategory.RATE_LIMIT_BACKOFF,
                        RetryCategory.IMMEDIATE_RETRY,
                    ):
                        delay = self.retry_base_delay_sec * (self.backoff_factor ** (attempt - 1))
                        log.warning(
                            "retryable error code=%d attempt=%d/%d delay=%.1fs",
                            code,
                            attempt,
                            self.retry_max_attempts,
                            delay,
                        )
                        last_exc = exc
                        await asyncio.sleep(delay)
                        continue
                    raise exc

                raise NetworkError(f"HTTP {resp.status}: {raw}")

            except (_aiohttp.ClientError, TimeoutError, asyncio.TimeoutError) as exc:
                last_exc = NetworkError(f"{type(exc).__name__}: {exc}")
                if attempt < self.retry_max_attempts:
                    delay = self.retry_base_delay_sec * (self.backoff_factor ** (attempt - 1))
                    log.warning(
                        "network error attempt=%d/%d delay=%.1fs: %s", attempt, self.retry_max_attempts, delay, exc
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_exc from exc

        raise last_exc or NetworkError("request failed after retries")

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        signed: bool = False,
    ) -> dict[str, object]:
        """Execute an HTTP request against the Binance API.

        Returns the parsed JSON response dict.
        """
        await self._bucket.acquire()
        url, query_params, headers = self._prepare_request(path, params, signed)
        raw = await self._with_retry(method, url, query_params, headers)
        if not isinstance(raw, dict):
            raise NetworkError(f"unexpected response type: {type(raw).__name__}")
        return cast(dict[str, object], raw)

    async def request_list(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        signed: bool = False,
    ) -> list[object]:
        """Execute a request expecting a JSON array response."""
        await self._bucket.acquire()
        url, query_params, headers = self._prepare_request(path, params, signed)
        raw = await self._with_retry(method, url, query_params, headers)
        if not isinstance(raw, list):
            raise NetworkError(f"expected list response, got {type(raw).__name__}")
        return cast(list[object], raw)

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session is not None:
            await self._session.close()
            self._session = None
