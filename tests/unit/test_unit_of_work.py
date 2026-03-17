"""Unit of work tests."""

from typing import Any

import pytest

from dojiwick.infrastructure.postgres.connection import DbConnection, DbCursor
from fixtures.fakes.unit_of_work import FakeUnitOfWork


pytestmark = pytest.mark.unit


class _StubCursor:
    async def execute(self, query: str, params: tuple[Any, ...] | list[Any] | None = None) -> object:
        _ = (query, params)
        return object()

    async def executemany(self, query: str, params: list[tuple[Any, ...]]) -> object:
        _ = (query, params)
        return object()

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return []

    async def fetchone(self) -> tuple[Any, ...] | None:
        return None

    @property
    def rowcount(self) -> int:
        return 0

    async def __aenter__(self) -> "_StubCursor":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> bool | None:
        _ = (exc_type, exc, tb)
        return None


class _StubConnection:
    def __init__(self, commits: list[str], rollbacks: list[str]) -> None:
        self._commits = commits
        self._rollbacks = rollbacks

    def cursor(self) -> DbCursor:
        return _StubCursor()

    async def commit(self) -> None:
        self._commits.append("commit")

    async def rollback(self) -> None:
        self._rollbacks.append("rollback")

    async def close(self) -> None:
        return


async def test_fake_uow_commits_on_success() -> None:
    uow = FakeUnitOfWork()

    async with uow.transaction():
        pass

    assert uow.committed == 1
    assert uow.rolled_back == 0


async def test_fake_uow_rolls_back_on_error() -> None:
    uow = FakeUnitOfWork()

    with pytest.raises(RuntimeError):
        async with uow.transaction():
            raise RuntimeError("boom")

    assert uow.committed == 0
    assert uow.rolled_back == 1


async def test_transaction_aware_connection_defers_commit_when_active() -> None:
    from dojiwick.infrastructure.postgres.connection import TransactionAwareConnection
    from dojiwick.infrastructure.postgres.unit_of_work import PgUnitOfWork

    commits: list[str] = []
    rollbacks: list[str] = []

    inner: DbConnection = _StubConnection(commits, rollbacks)
    uow = PgUnitOfWork(connection=inner)
    conn = TransactionAwareConnection(inner=inner, unit_of_work=uow)

    # During active transaction, commit/rollback should be no-ops.
    async with uow.transaction():
        await conn.commit()
        await conn.rollback()

    # The UoW itself commits once on success.
    assert commits == ["commit"]
    assert rollbacks == []


async def test_transaction_aware_connection_passes_through_when_inactive() -> None:
    from dojiwick.infrastructure.postgres.connection import TransactionAwareConnection

    commits: list[str] = []
    rollbacks: list[str] = []

    inner: DbConnection = _StubConnection(commits, rollbacks)
    conn = TransactionAwareConnection(inner=inner)

    # Without a UoW, commit/rollback pass through directly.
    await conn.commit()
    await conn.rollback()

    assert commits == ["commit"]
    assert rollbacks == ["rollback"]
