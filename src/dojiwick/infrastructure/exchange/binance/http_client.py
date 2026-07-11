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

    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
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

    def _build_signed_url(self, path: str, params: dict[str, str]) -> str:
        """URL with a fresh timestamp and a signature over the exact encoded query.

        Signing must happen per attempt (a slow retry with a stale timestamp
        exceeds recvWindow, -1021) and over the urlencoded string that is
        actually sent — signing raw values while aiohttp percent-encodes them
        yields -1022 for any value needing encoding.
        """
        from urllib.parse import urlencode

        query = dict(params)
        query["timestamp"] = str(self.clock.epoch_ms())
        query["recvWindow"] = str(self.recv_window_ms)
        encoded = urlencode(query)
        return f"{self.base_url}{path}?{encoded}&signature={self.sign(encoded)}"

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
        path: str,
        params: dict[str, str],
        signed: bool,
    ) -> object:
        """Execute HTTP request with per-category retry and exponential backoff."""
        import aiohttp as _aiohttp

        session = await self.ensure_session()
        last_exc: Exception | None = None
        headers = {"X-MBX-APIKEY": self.api_key} if signed else {}

        attempt = 0
        while True:
            attempt += 1
            if signed:
                url = self._build_signed_url(path, params)
                request_params = None
            else:
                url = f"{self.base_url}{path}"
                request_params = params or None
            try:
                resp = await session.request(method, url, params=request_params, headers=headers)
                raw: object = await resp.json()

                if resp.ok:
                    return raw

                result = self._parse_error_body(raw)
                if result is not None:
                    exc, policy, code = result
                    # retry_max_attempts is the TOTAL attempt cap; per-category
                    # policies can only tighten it
                    max_attempts = min(policy.max_retries + 1, self.retry_max_attempts)
                    if policy.category == RetryCategory.NO_RETRY or attempt >= max_attempts:
                        raise exc
                    if policy.category == RetryCategory.IMMEDIATE_RETRY:
                        delay = 0.0
                    else:
                        delay = min(
                            policy.base_delay_ms / 1000.0 * (self.backoff_factor ** (attempt - 1)),
                            policy.max_delay_ms / 1000.0,
                        )
                    # 429/418 carry Retry-After; the ban (418) escalates fast
                    try:
                        retry_after = float(resp.headers.get("Retry-After", ""))
                        delay = max(delay, retry_after)
                    except TypeError, ValueError:
                        pass
                    if resp.status == 418:
                        log.critical("Binance IP ban (418) — backing off %.0fs", max(delay, 60.0))
                        delay = max(delay, 60.0)
                    log.warning("retryable error code=%d attempt=%d/%d delay=%.1fs", code, attempt, max_attempts, delay)
                    last_exc = exc
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue

                raise NetworkError(f"HTTP {resp.status}: {raw}")

            except (_aiohttp.ClientError, TimeoutError, asyncio.TimeoutError) as exc:
                last_exc = NetworkError(f"{type(exc).__name__}: {exc}")
                if attempt < self.retry_max_attempts:
                    delay = self.retry_base_delay_sec * (self.backoff_factor ** (attempt - 1))
                    log.warning(
                        "network error attempt=%d/%d delay=%.1fs: %s",
                        attempt,
                        self.retry_max_attempts,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_exc from exc

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
        raw = await self._with_retry(method, path, dict(params) if params else {}, signed)
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
        raw = await self._with_retry(method, path, dict(params) if params else {}, signed)
        if not isinstance(raw, list):
            raise NetworkError(f"expected list response, got {type(raw).__name__}")
        return cast(list[object], raw)

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session is not None:
            await self._session.close()
            self._session = None
