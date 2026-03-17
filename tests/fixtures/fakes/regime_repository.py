"""Regime repository test doubles."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dojiwick.domain.models.value_objects.batch_models import BatchRegimeProfile


@dataclass(slots=True)
class CapturingRegimeRepo:
    """Stores all insert_batch calls for assertion."""

    batches: list[dict[str, Any]] = field(default_factory=list)

    async def insert_batch(
        self,
        pairs: tuple[str, ...],
        observed_at: datetime,
        regimes: BatchRegimeProfile,
        *,
        target_ids: tuple[str, ...],
        venue: str,
        product: str,
    ) -> None:
        self.batches.append(
            {
                "pairs": pairs,
                "observed_at": observed_at,
                "regimes": regimes,
                "target_ids": target_ids,
                "venue": venue,
                "product": product,
            }
        )


class FailingRegimeRepo:
    """Raises on insert_batch."""

    async def insert_batch(
        self,
        pairs: tuple[str, ...],
        observed_at: datetime,
        regimes: BatchRegimeProfile,
        *,
        target_ids: tuple[str, ...],
        venue: str,
        product: str,
    ) -> None:
        del pairs, observed_at, regimes, target_ids, venue, product
        raise RuntimeError("regime repo failure")


@dataclass(slots=True)
class InMemoryRegimeRepo:
    """Stores regime rows in memory for tests."""

    rows: list[dict[str, object]] = field(default_factory=list)

    async def insert_batch(
        self,
        pairs: tuple[str, ...],
        observed_at: datetime,
        regimes: BatchRegimeProfile,
        *,
        target_ids: tuple[str, ...],
        venue: str,
        product: str,
    ) -> None:
        for index, pair in enumerate(pairs):
            self.rows.append(
                {
                    "pair": pair,
                    "target_id": target_ids[index],
                    "observed_at": observed_at.isoformat(),
                    "coarse_state": int(regimes.coarse_state[index]),
                    "confidence": float(regimes.confidence[index]),
                    "valid": bool(regimes.valid_mask[index]),
                    "venue": venue,
                    "product": product,
                }
            )
