"""Interval scheduler aligned to clock boundaries."""

import asyncio
from collections.abc import Awaitable, Callable

from dojiwick.domain.contracts.gateways.clock import ClockPort


class CronRunner:
    """Runs an async callable on a fixed interval, aligned to clock boundaries."""

    def __init__(self, tick: Callable[[], Awaitable[object]], clock: ClockPort) -> None:
        self._tick = tick
        self._clock = clock

    async def run(self, *, interval_sec: int, stop: asyncio.Event) -> None:
        """Execute tick at each interval boundary until stop is set."""
        while not stop.is_set():
            now = self._clock.now_utc()
            epoch = now.timestamp()
            delay = interval_sec - (epoch % interval_sec)
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
                return
            except TimeoutError:
                pass
            await self._tick()
