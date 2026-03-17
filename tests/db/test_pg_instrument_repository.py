"""Integration tests for PgInstrumentRepository."""

from decimal import Decimal
from typing import Any

import pytest

from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.instrument_metadata import InstrumentFilter, InstrumentInfo

pytestmark = pytest.mark.db


@pytest.fixture
def repo(db_connection: Any) -> Any:
    from dojiwick.infrastructure.postgres.repositories.instrument import PgInstrumentRepository

    return PgInstrumentRepository(connection=db_connection)


def _make_info(symbol: str = "BTCUSDC") -> InstrumentInfo:
    return InstrumentInfo(
        instrument_id=InstrumentId(
            venue=BINANCE_VENUE,
            product=BINANCE_USD_C,
            symbol=symbol,
            base_asset="BTC",
            quote_asset="USDC",
            settle_asset="USDC",
        ),
        status="trading",
        filters=InstrumentFilter(
            min_price=Decimal("0.01"),
            max_price=Decimal("100000"),
            tick_size=Decimal("0.01"),
            min_qty=Decimal("0.001"),
            max_qty=Decimal("1000"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("5"),
        ),
        price_precision=2,
        quantity_precision=3,
        base_asset_precision=8,
        quote_asset_precision=8,
        contract_size=Decimal("1"),
        margin_asset="USDC",
    )


async def test_upsert_and_get_by_symbol(repo: Any, clean_tables: None) -> None:
    info = _make_info()
    db_id = await repo.upsert_instrument(info)
    assert db_id > 0

    loaded = await repo.get_by_symbol(BINANCE_VENUE, BINANCE_USD_C, "BTCUSDC")
    assert loaded is not None
    assert loaded.instrument_id.symbol == "BTCUSDC"
    assert loaded.filters.tick_size == Decimal("0.01")
    assert loaded.price_precision == 2


async def test_get_by_symbol_not_found(repo: Any, clean_tables: None) -> None:
    loaded = await repo.get_by_symbol(BINANCE_VENUE, BINANCE_USD_C, "UNKNOWN")
    assert loaded is None


async def test_list_instruments(repo: Any, clean_tables: None) -> None:
    await repo.upsert_instrument(_make_info("BTCUSDC"))
    await repo.upsert_instrument(_make_info("ETHUSDC"))
    instruments = await repo.list_instruments(BINANCE_VENUE, BINANCE_USD_C)
    symbols = {i.instrument_id.symbol for i in instruments}
    assert "BTCUSDC" in symbols
    assert "ETHUSDC" in symbols
