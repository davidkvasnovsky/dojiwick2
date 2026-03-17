"""Pending order provider protocol."""

from typing import Protocol

from dojiwick.domain.enums import PositionSide
from dojiwick.domain.numerics import Quantity


class PendingOrderProviderPort(Protocol):
    """Provides pending (in-flight) order quantities grouped by symbol and position side."""

    async def get_pending_quantities(
        self,
        account: str,
    ) -> dict[tuple[str, PositionSide], Quantity]:
        """Return total pending qty grouped by (symbol, position_side).

        Pending = orders with latest report status in (NEW, PARTIALLY_FILLED).
        Returns remaining unfilled quantity (original_qty - filled_qty).
        """
        ...
