"""Account state port test doubles."""

from dojiwick.domain.models.value_objects.account_state import AccountSnapshot


class FakeAccountState:
    """In-memory fake for AccountStatePort — returns configurable account snapshots."""

    def __init__(self, snapshots: dict[str, AccountSnapshot] | None = None) -> None:
        self._snapshots: dict[str, AccountSnapshot] = snapshots or {}

    def set_snapshot(self, account: str, snapshot: AccountSnapshot) -> None:
        """Test helper: set the snapshot for an account."""
        self._snapshots[account] = snapshot

    async def get_account_snapshot(self, account: str) -> AccountSnapshot:
        if account not in self._snapshots:
            raise KeyError(f"account not found: {account}")
        return self._snapshots[account]
