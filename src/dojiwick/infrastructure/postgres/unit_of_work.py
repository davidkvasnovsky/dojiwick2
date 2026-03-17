"""PostgreSQL unit of work for atomic multi-repository persistence."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from dojiwick.infrastructure.postgres.connection import DbConnection

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PgUnitOfWork:
    """Manages a single transaction across multiple repository calls."""

    connection: DbConnection
    _active: bool = field(default=False, init=False)

    @property
    def active(self) -> bool:
        return self._active

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        self._active = True
        try:
            yield
            await self.connection.commit()
        except Exception:
            await self.connection.rollback()
            raise
        finally:
            self._active = False

    async def rollback_if_active(self) -> bool:
        """Roll back any in-flight transaction during shutdown.

        Returns True if a rollback was performed.
        """
        if not self._active:
            return False
        log.warning("rolling back in-flight transaction during shutdown")
        try:
            await self.connection.rollback()
        except Exception:
            log.exception("rollback during shutdown failed")
        finally:
            self._active = False
        return True
