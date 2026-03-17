"""Submission acknowledgement value object.

Represents the exchange's acknowledgement of an order submission,
separate from fill economics.
"""

from dataclasses import dataclass
from datetime import datetime

from dojiwick.domain.enums import SubmissionStatus


@dataclass(slots=True, frozen=True, kw_only=True)
class SubmissionAck:
    """Acknowledgement that an order was received by the exchange."""

    status: SubmissionStatus
    order_id: str = ""
    exchange_timestamp: datetime | None = None
    reason: str = ""
