"""Custom exception hierarchy for the engine."""


class EngineError(Exception):
    """Base exception for all engine errors."""


class ConfigurationError(EngineError):
    """Raised for invalid configuration values or missing settings."""


class DomainValidationError(EngineError):
    """Raised when domain model invariants are violated."""


class DataQualityError(EngineError):
    """Raised when market data is stale, suspect, or incomplete."""


class ReconciliationError(EngineError):
    """Raised when DB state diverges from exchange state."""


class AdapterError(EngineError):
    """Raised when an adapter encounters an unrecoverable error."""


class NetworkError(AdapterError):
    """Raised for transient connection failures."""


class ExchangeError(AdapterError):
    """Raised for exchange API errors."""


class RateLimitError(ExchangeError):
    """Raised when the exchange returns HTTP 429 (backoff needed)."""


class ExchangeTimeoutError(ExchangeError):
    """Raised when an exchange API call times out (transient)."""


class VetoServiceError(AdapterError):
    """Raised when the AI veto service fails."""


class CircuitBreakerTrippedError(EngineError):
    """Raised when the circuit breaker prevents tick execution."""


class PostExecutionPersistenceError(EngineError):
    """Raised when execution succeeds but outcome persistence fails.

    Real positions may exist with no audit trail. Callers should treat
    this as critical and halt further processing.
    """


class InsufficientBalanceError(ExchangeError):
    """Raised when the exchange rejects for insufficient margin or balance."""


class OrderNotFoundError(ExchangeError):
    """Raised when cancel/modify targets a nonexistent order."""


class AuthenticationError(AdapterError):
    """Raised when API key authentication fails (expiry, revocation)."""


class DataFetchError(AdapterError):
    """Raised when candle or indicator fetching fails (distinct from DataQualityError)."""


class AdaptivePolicyError(EngineError):
    """Raised when the adaptive policy encounters an unrecoverable error."""


class AdaptiveCalibrationError(AdaptivePolicyError):
    """Raised when adaptive calibration metrics indicate a misconfigured policy."""
