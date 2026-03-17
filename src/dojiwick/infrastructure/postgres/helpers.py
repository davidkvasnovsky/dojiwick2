"""Shared PostgreSQL helper functions for repositories."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dojiwick.domain.errors import AdapterError
from dojiwick.infrastructure.postgres.connection import DbConnection


async def pg_execute(conn: DbConnection, query: str, params: tuple[Any, ...], *, error_msg: str) -> None:
    """Write query: execute + commit, rollback on error."""
    try:
        async with conn.cursor() as cursor:
            await cursor.execute(query, params)
        await conn.commit()
    except Exception as exc:
        await conn.rollback()
        raise AdapterError(f"{error_msg}: {exc}") from exc


async def pg_execute_many(conn: DbConnection, query: str, rows: list[tuple[Any, ...]], *, error_msg: str) -> None:
    """Batch write: executemany + commit, rollback on error."""
    try:
        async with conn.cursor() as cursor:
            await cursor.executemany(query, rows)
        await conn.commit()
    except Exception as exc:
        await conn.rollback()
        raise AdapterError(f"{error_msg}: {exc}") from exc


async def pg_fetch_one(
    conn: DbConnection, query: str, params: tuple[Any, ...], *, error_msg: str
) -> tuple[Any, ...] | None:
    """Read one row, rollback on error. No commit (read-only)."""
    try:
        async with conn.cursor() as cursor:
            await cursor.execute(query, params)
            return await cursor.fetchone()
    except Exception as exc:
        await conn.rollback()
        raise AdapterError(f"{error_msg}: {exc}") from exc


async def pg_fetch_all(
    conn: DbConnection, query: str, params: tuple[Any, ...] | None = None, *, error_msg: str
) -> list[tuple[Any, ...]]:
    """Read all rows, rollback on error. No commit (read-only)."""
    try:
        async with conn.cursor() as cursor:
            await cursor.execute(query, params)
            return await cursor.fetchall()
    except Exception as exc:
        await conn.rollback()
        raise AdapterError(f"{error_msg}: {exc}") from exc


def parse_pg_datetime(value: object) -> datetime:
    """Parse datetime from PostgreSQL (str or datetime), ensure UTC."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value)
    else:
        raise AdapterError(f"expected datetime, got {type(value).__name__}")
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def parse_pg_datetime_optional(value: object) -> datetime | None:
    """Parse nullable datetime from PostgreSQL."""
    if value is None:
        return None
    return parse_pg_datetime(value)
