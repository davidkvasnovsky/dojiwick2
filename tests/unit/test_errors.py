"""Exception hierarchy tests."""

from dojiwick.domain.errors import (
    AdapterError,
    ConfigurationError,
    DataQualityError,
    DomainValidationError,
    EngineError,
    ExchangeError,
    ExchangeTimeoutError,
    NetworkError,
    PostExecutionPersistenceError,
    RateLimitError,
    ReconciliationError,
    VetoServiceError,
)


def test_all_exceptions_inherit_from_engine_error() -> None:
    exceptions = [
        ConfigurationError,
        DomainValidationError,
        DataQualityError,
        ReconciliationError,
        AdapterError,
        NetworkError,
        ExchangeError,
        RateLimitError,
        ExchangeTimeoutError,
        VetoServiceError,
        PostExecutionPersistenceError,
    ]
    for exc_class in exceptions:
        assert issubclass(exc_class, EngineError), f"{exc_class.__name__} must inherit EngineError"


def test_adapter_subtypes() -> None:
    assert issubclass(NetworkError, AdapterError)
    assert issubclass(ExchangeError, AdapterError)
    assert issubclass(VetoServiceError, AdapterError)


def test_exchange_subtypes() -> None:
    assert issubclass(RateLimitError, ExchangeError)
    assert issubclass(ExchangeTimeoutError, ExchangeError)


def test_isinstance_catches_parent() -> None:
    err = RateLimitError("429 too many requests")
    assert isinstance(err, ExchangeError)
    assert isinstance(err, AdapterError)
    assert isinstance(err, EngineError)
    assert isinstance(err, Exception)


def test_data_quality_error_not_adapter() -> None:
    assert not issubclass(DataQualityError, AdapterError)


def test_reconciliation_error_not_adapter() -> None:
    assert not issubclass(ReconciliationError, AdapterError)
