"""Binance error-code mapping to domain error types and retry/backoff policy.

Error codes reference: https://binance-docs.github.io/apidocs/futures/en/#error-codes

Exchange-specific error details are encapsulated inside this adapter boundary —
the domain layer receives only domain error types (ExchangeError, RateLimitError, etc.).
"""

from dataclasses import dataclass
from enum import StrEnum, unique

from dojiwick.domain.errors import (
    AdapterError,
    AuthenticationError,
    ExchangeError,
    ExchangeTimeoutError,
    InsufficientBalanceError,
    OrderNotFoundError,
    RateLimitError,
)


@unique
class RetryCategory(StrEnum):
    """Retry behavior category per error type."""

    NO_RETRY = "no_retry"
    IMMEDIATE_RETRY = "immediate_retry"
    BACKOFF_RETRY = "backoff_retry"
    RATE_LIMIT_BACKOFF = "rate_limit_backoff"


@dataclass(slots=True, frozen=True)
class RetryPolicy:
    """Retry and backoff parameters for an error category."""

    category: RetryCategory
    max_retries: int = 0
    base_delay_ms: int = 0
    max_delay_ms: int = 0
    jitter: bool = False


# Retry policies per category
RETRY_POLICIES: dict[RetryCategory, RetryPolicy] = {
    RetryCategory.NO_RETRY: RetryPolicy(category=RetryCategory.NO_RETRY),
    RetryCategory.IMMEDIATE_RETRY: RetryPolicy(
        category=RetryCategory.IMMEDIATE_RETRY,
        max_retries=1,
        base_delay_ms=0,
    ),
    RetryCategory.BACKOFF_RETRY: RetryPolicy(
        category=RetryCategory.BACKOFF_RETRY,
        max_retries=3,
        base_delay_ms=500,
        max_delay_ms=5000,
        jitter=True,
    ),
    RetryCategory.RATE_LIMIT_BACKOFF: RetryPolicy(
        category=RetryCategory.RATE_LIMIT_BACKOFF,
        max_retries=5,
        base_delay_ms=1000,
        max_delay_ms=60000,
        jitter=True,
    ),
}


@dataclass(slots=True, frozen=True)
class ErrorMapping:
    """Maps a Binance error code to a domain exception class and retry policy."""

    domain_error: type[AdapterError]
    retry_category: RetryCategory
    message: str


# Binance error code -> domain mapping
# Only the most common/critical codes are mapped; unknown codes fall back to ExchangeError + backoff.
BINANCE_ERROR_MAP: dict[int, ErrorMapping] = {
    # Rate limits
    -1003: ErrorMapping(RateLimitError, RetryCategory.RATE_LIMIT_BACKOFF, "Too many requests"),
    -1015: ErrorMapping(RateLimitError, RetryCategory.RATE_LIMIT_BACKOFF, "Too many orders"),
    # Timeouts / network
    -1007: ErrorMapping(ExchangeTimeoutError, RetryCategory.BACKOFF_RETRY, "Timeout waiting for response"),
    # Authentication
    -2014: ErrorMapping(AuthenticationError, RetryCategory.NO_RETRY, "API key format invalid"),
    -2015: ErrorMapping(AuthenticationError, RetryCategory.NO_RETRY, "Invalid API key/IP/permissions"),
    -1022: ErrorMapping(AuthenticationError, RetryCategory.NO_RETRY, "Invalid signature"),
    # Order errors
    -2013: ErrorMapping(OrderNotFoundError, RetryCategory.NO_RETRY, "Order does not exist"),
    -2022: ErrorMapping(ExchangeError, RetryCategory.NO_RETRY, "Order would immediately trigger"),
    -4003: ErrorMapping(ExchangeError, RetryCategory.NO_RETRY, "Quantity less than zero"),
    -4014: ErrorMapping(ExchangeError, RetryCategory.NO_RETRY, "Price less than zero"),
    # Balance / margin
    -2019: ErrorMapping(InsufficientBalanceError, RetryCategory.NO_RETRY, "Margin is insufficient"),
    -4028: ErrorMapping(InsufficientBalanceError, RetryCategory.NO_RETRY, "Insufficient balance for leverage"),
    # General
    -1000: ErrorMapping(ExchangeError, RetryCategory.BACKOFF_RETRY, "Unknown error"),
    -1001: ErrorMapping(ExchangeError, RetryCategory.BACKOFF_RETRY, "Disconnected"),
    -1002: ErrorMapping(AuthenticationError, RetryCategory.NO_RETRY, "Unauthorized"),
    -1006: ErrorMapping(ExchangeError, RetryCategory.BACKOFF_RETRY, "Unexpected response"),
    -1010: ErrorMapping(ExchangeError, RetryCategory.NO_RETRY, "Message received but not processed"),
    -1021: ErrorMapping(ExchangeError, RetryCategory.IMMEDIATE_RETRY, "Timestamp outside recv window"),
}

# Default for unmapped error codes
DEFAULT_ERROR_MAPPING = ErrorMapping(ExchangeError, RetryCategory.BACKOFF_RETRY, "Unmapped exchange error")


def map_binance_error(error_code: int, raw_message: str = "") -> tuple[AdapterError, RetryPolicy]:
    """Map a Binance error code to a domain exception instance and retry policy.

    Parameters
    ----------
    error_code : int
        The Binance API error code (e.g., -1003).
    raw_message : str
        The raw message from the exchange (for context in the exception).

    Returns
    -------
    tuple[AdapterError, RetryPolicy]
        A domain exception instance and the associated retry policy.
    """
    mapping = BINANCE_ERROR_MAP.get(error_code, DEFAULT_ERROR_MAPPING)
    msg = raw_message or mapping.message
    exception = mapping.domain_error(f"[{error_code}] {msg}")
    policy = RETRY_POLICIES[mapping.retry_category]
    return exception, policy
