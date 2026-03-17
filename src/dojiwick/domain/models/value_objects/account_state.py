"""Account state value objects for the AccountStatePort boundary."""

from dataclasses import dataclass
from decimal import Decimal

from dojiwick.domain.enums import PositionSide
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.numerics import Money, Price, Quantity


@dataclass(slots=True, frozen=True, kw_only=True)
class AccountBalance:
    """Balance snapshot for a single asset in a margin account."""

    asset: str
    wallet_balance: Money
    available_balance: Money
    cross_unrealized_pnl: Money = Decimal(0)


@dataclass(slots=True, frozen=True, kw_only=True)
class ExchangePositionLeg:
    """A single open position leg as reported by the exchange."""

    instrument_id: InstrumentId
    position_side: PositionSide
    quantity: Quantity
    entry_price: Price
    unrealized_pnl: Money
    leverage: int = 1
    liquidation_price: Price | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class AccountSnapshot:
    """Complete account state snapshot — balances + open legs + margin info."""

    account: str
    balances: tuple[AccountBalance, ...]
    positions: tuple[ExchangePositionLeg, ...]
    total_wallet_balance: Money
    available_balance: Money
    total_unrealized_pnl: Money = Decimal(0)
    total_margin_balance: Money = Decimal(0)
