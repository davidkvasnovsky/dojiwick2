"""Unit of work test double."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class FakeUnitOfWork:
    """Tracks transaction commit/rollback for assertions."""

    committed: int = field(default=0, init=False)
    rolled_back: int = field(default=0, init=False)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        try:
            yield
            self.committed += 1
        except Exception:
            self.rolled_back += 1
            raise
