"""Order state machine — enforced transitions and residual-quantity semantics.

Valid transitions
-----------------
NEW -> PARTIALLY_FILLED -> FILLED
NEW -> CANCELED | EXPIRED | REJECTED
PARTIALLY_FILLED -> FILLED | CANCELED | EXPIRED

Terminal states: FILLED, CANCELED, EXPIRED, REJECTED

Residual-quantity handling
--------------------------
When an order is CANCELED or EXPIRED from PARTIALLY_FILLED:
  - filled_quantity reflects the amount already executed
  - residual_quantity = original_quantity - filled_quantity
  - The residual is considered released (not re-queued)
  - Position accounting uses filled_quantity only
"""

from dojiwick.domain.enums import OrderStatus
from dojiwick.domain.errors import DomainValidationError
from dojiwick.domain.numerics import Quantity

VALID_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.NEW: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.EXPIRED,
        }
    ),
    # Terminal states — no outgoing transitions
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
}

TERMINAL_STATES: frozenset[OrderStatus] = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
        OrderStatus.REJECTED,
    }
)


def validate_transition(current: OrderStatus, target: OrderStatus) -> None:
    """Raise DomainValidationError if the transition is not allowed."""
    allowed = VALID_TRANSITIONS.get(current)
    if allowed is None:
        raise DomainValidationError(f"Unknown order status: {current}")
    if target not in allowed:
        raise DomainValidationError(f"Invalid order transition: {current} -> {target}")


def is_terminal(status: OrderStatus) -> bool:
    """Return True if the status is a terminal (no further transitions)."""
    return status in TERMINAL_STATES


def compute_residual_quantity(original_quantity: Quantity, filled_quantity: Quantity) -> Quantity:
    """Return the unfilled quantity that is released on cancel/expire.

    The residual is not re-queued — position accounting uses filled_quantity only.
    """
    residual = original_quantity - filled_quantity
    if residual < 0:
        raise DomainValidationError(
            f"filled_quantity ({filled_quantity}) exceeds original_quantity ({original_quantity})"
        )
    return residual
