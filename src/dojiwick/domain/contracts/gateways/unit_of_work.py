"""Unit of work protocol for transactional boundaries."""

from contextlib import AbstractAsyncContextManager
from typing import Protocol


class UnitOfWorkPort(Protocol):
    """Coordinates atomic persistence across multiple repositories."""

    def transaction(self) -> AbstractAsyncContextManager[None]: ...
