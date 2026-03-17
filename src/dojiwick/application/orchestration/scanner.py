"""Between-tick scanner service for position monitoring."""

import logging
from dataclasses import dataclass

from dojiwick.domain.contracts.gateways.notification import NotificationPort
from dojiwick.domain.contracts.repositories.position_leg import PositionLegRepositoryPort
from dojiwick.domain.models.value_objects.position_leg import PositionLeg

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ScanCycleSummary:
    """Summary counts returned by :meth:`ScannerService.run_scan_cycle`."""

    exits_triggered: int
    liquidation_warnings: int
    stops_updated: int


@dataclass(slots=True)
class ScannerService:
    """Monitors position legs between ticks for exit detection and risk events."""

    position_leg_repository: PositionLegRepositoryPort
    account: str
    notification: NotificationPort | None = None

    async def scan_exits(self, legs: tuple[PositionLeg, ...]) -> int:
        """Check active legs for SL/TP hits (skeleton). Returns count of exits triggered."""
        log.debug("scanning %d active legs for exits", len(legs))
        return 0

    async def scan_liquidation_risk(self, legs: tuple[PositionLeg, ...]) -> int:
        """Check active legs for liquidation proximity (skeleton). Returns warning count."""
        log.debug("scanning %d active legs for liquidation risk", len(legs))
        return 0

    async def update_trailing_stops(self, legs: tuple[PositionLeg, ...]) -> int:
        """Adjust trailing stops for active legs (skeleton). Returns count of stops updated."""
        log.debug("checking trailing stops for %d legs", len(legs))
        return 0

    async def run_scan_cycle(self) -> ScanCycleSummary:
        """Run a full scan cycle: exits, liquidation risk, and trailing stops.

        Returns a :class:`ScanCycleSummary` with counts from each phase.
        """
        legs = await self.position_leg_repository.get_active_legs(self.account)
        exits_triggered = await self.scan_exits(legs)
        liquidation_warnings = await self.scan_liquidation_risk(legs)
        stops_updated = await self.update_trailing_stops(legs)

        return ScanCycleSummary(
            exits_triggered=exits_triggered,
            liquidation_warnings=liquidation_warnings,
            stops_updated=stops_updated,
        )
