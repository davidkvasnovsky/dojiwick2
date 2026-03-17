"""PostgreSQL connection helper and shared protocols."""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from dojiwick.config.schema import DatabaseSettings
from dojiwick.domain.errors import ConfigurationError

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from dojiwick.infrastructure.postgres.unit_of_work import PgUnitOfWork

log = logging.getLogger(__name__)


class DbCursor(Protocol):
    """Cursor protocol for queries and batch operations.

    Uses Any for params and rows because SQL queries accept and return
    heterogeneous tuples whose element types vary per query. There is no
    way to express this generically in Python's type system.
    """

    async def execute(self, query: str, params: tuple[Any, ...] | list[Any] | None = None) -> object:
        """Execute a single query."""
        ...

    async def executemany(self, query: str, params: list[tuple[Any, ...]]) -> object:
        """Execute query for many parameter rows."""
        ...

    async def fetchall(self) -> list[tuple[Any, ...]]:
        """Return all rows from the last query."""
        ...

    async def fetchone(self) -> tuple[Any, ...] | None:
        """Return the next row or None."""
        ...

    @property
    def rowcount(self) -> int:
        """Number of rows affected by the last operation."""
        ...

    async def __aenter__(self) -> "DbCursor":
        """Enter async context manager."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> bool | None:
        """Exit async context manager."""
        ...


class DbConnection(Protocol):
    """Minimal connection protocol for repositories."""

    def cursor(self) -> DbCursor:
        """Return db cursor."""
        ...

    async def commit(self) -> None:
        """Commit active transaction."""
        ...

    async def rollback(self) -> None:
        """Roll back active transaction."""
        ...

    async def close(self) -> None:
        """Close the connection."""
        ...


async def connect(settings: DatabaseSettings) -> DbConnection:
    """Create a single async psycopg connection using typed database settings.

    Single-connection is safe only for the current sequential adapter calls
    (one adapter method at a time per tick). If concurrent access is needed
    (e.g. parallel adapter calls or multi-worker), add a ``create_pool()``
    factory backed by ``psycopg_pool.AsyncConnectionPool`` and controlled by
    ``DatabaseSettings.min_connections`` / ``max_connections``.
    """

    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("psycopg is required for postgres adapters") from exc
    conn = await psycopg.AsyncConnection.connect(
        settings.dsn,
        connect_timeout=int(settings.connect_timeout_sec),
        application_name=settings.app_name,
        options=f"-c statement_timeout={settings.statement_timeout_ms}",
    )
    return cast(DbConnection, conn)


async def create_pool(settings: DatabaseSettings) -> "AsyncConnectionPool[Any]":
    """Create an async connection pool using typed database settings."""

    try:
        from psycopg_pool import AsyncConnectionPool as Pool
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("psycopg_pool is required for postgres connection pooling") from exc

    pool: AsyncConnectionPool[Any] = Pool(
        conninfo=settings.dsn,
        min_size=settings.min_connections,
        max_size=settings.max_connections,
        kwargs={
            "connect_timeout": int(settings.connect_timeout_sec),
            "application_name": settings.app_name,
            "options": f"-c statement_timeout={settings.statement_timeout_ms}",
        },
    )
    await pool.open()
    return pool


async def check_db_connectivity(pool: "AsyncConnectionPool[Any]") -> None:
    """Run a simple SELECT 1 health check against the pool.

    Raises ConfigurationError on failure.
    """
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
    except Exception as exc:
        raise ConfigurationError(f"database health check failed: {exc}") from exc
    log.info("database connectivity check passed")


@dataclass(slots=True)
class TransactionAwareConnection:
    """Wraps DbConnection to defer commit/rollback during active transactions."""

    inner: DbConnection
    unit_of_work: PgUnitOfWork | None = None

    def cursor(self) -> DbCursor:
        return self.inner.cursor()

    async def commit(self) -> None:
        if self.unit_of_work is None or not self.unit_of_work.active:
            await self.inner.commit()

    async def rollback(self) -> None:
        if self.unit_of_work is None or not self.unit_of_work.active:
            await self.inner.rollback()

    async def close(self) -> None:
        await self.inner.close()
