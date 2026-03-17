"""Context provider protocol."""

from datetime import datetime
from typing import Protocol

from dojiwick.domain.models.value_objects.batch_models import BatchDecisionContext


class ContextProviderPort(Protocol):
    """Fetches aligned batch context for multiple pairs.

    The tick always needs market and portfolio data in lockstep (same
    timestamp), so they are bundled in a single port. The adapter
    internally decomposes into separate exchange calls as an
    implementation detail.

    Extension pattern for new data sources:
    1. Add an optional field to ``BatchDecisionContext``
       (e.g. ``funding: BatchFundingSnapshot | None = None``).
    2. Adapter populates it from the relevant exchange call.
    3. Kernels that need the new data receive the full context
       (which they already do).
    """

    async def fetch_context_batch(self, pairs: tuple[str, ...], at: datetime) -> BatchDecisionContext:
        """Return one context batch aligned with input pair order."""
        ...
