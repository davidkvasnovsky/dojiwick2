"""Startup instrument synchronization.

Every order and position write resolves instruments through the DB; on a
fresh database `resolve_id` returns None and persistence fails. Syncing
exchange metadata into the instruments table at startup guarantees every
execution symbol is known — and fails startup hard when a configured symbol
is missing or not trading, instead of discovering it mid-tick.
"""

import logging
from dataclasses import dataclass

from dojiwick.domain.contracts.gateways.exchange_metadata import ExchangeMetadataPort
from dojiwick.domain.contracts.repositories.instrument import InstrumentRepositoryPort
from dojiwick.domain.errors import ConfigurationError
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId

log = logging.getLogger(__name__)

_TRADING_STATUS = "TRADING"


@dataclass(slots=True)
class InstrumentSyncService:
    """Upserts exchange instrument metadata for all execution symbols."""

    exchange_metadata: ExchangeMetadataPort
    instrument_repo: InstrumentRepositoryPort

    async def sync(self, instrument_ids: tuple[InstrumentId, ...]) -> int:
        """Sync metadata for the given instruments; return the count upserted."""
        count = 0
        for iid in instrument_ids:
            info = await self.exchange_metadata.get_instrument(iid)
            if info.status != _TRADING_STATUS:
                raise ConfigurationError(
                    f"instrument {iid.symbol} is not tradeable (status={info.status}) — remove it from the universe"
                )
            await self.instrument_repo.upsert_instrument(info)
            count += 1
        log.info("instrument sync: %d symbols upserted", count)
        return count
