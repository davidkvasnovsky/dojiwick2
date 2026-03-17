"""Fake exchange data feed for tests."""

from dataclasses import dataclass


@dataclass(slots=True)
class FakeExchangeDataFeed:
    """Minimal fake for ExchangeDataFeed — tracks lifecycle calls."""

    bootstrapped: bool = False
    started: bool = False
    stopped: bool = False
    bootstrap_error: Exception | None = None

    async def bootstrap(self) -> None:
        if self.bootstrap_error is not None:
            err = self.bootstrap_error
            self.bootstrap_error = None
            raise err
        self.bootstrapped = True

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def ensure_fresh(self) -> None:
        pass
