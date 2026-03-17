"""Account state port — fetching account balances, open positions, margin info."""

from typing import Protocol

from dojiwick.domain.models.value_objects.account_state import AccountSnapshot


class AccountStatePort(Protocol):
    """Fetches account balances, open positions, and margin info."""

    async def get_account_snapshot(self, account: str) -> AccountSnapshot:
        """Return full account snapshot (balances + positions + margin)."""
        ...
