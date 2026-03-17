"""TypedDict definitions for Binance Futures API responses."""

from __future__ import annotations

from typing import TypedDict


# GET /fapi/v2/account
class AssetEntry(TypedDict):
    asset: str
    walletBalance: str
    availableBalance: str
    crossUnPnl: str


class PositionEntry(TypedDict):
    symbol: str
    positionAmt: str
    positionSide: str
    entryPrice: str
    leverage: int | float
    unrealizedProfit: str


class AccountResponse(TypedDict):
    assets: list[AssetEntry]
    positions: list[PositionEntry]
    totalWalletBalance: str
    availableBalance: str
    totalUnrealizedProfit: str
    totalMarginBalance: str


# GET /fapi/v2/ticker/price (array endpoint)
class PriceTickerEntry(TypedDict):
    symbol: str
    price: str


# GET /fapi/v1/openOrders (array endpoint)
class OpenOrderEntry(TypedDict):
    orderId: str
    clientOrderId: str
    symbol: str
    side: str
    positionSide: str
    status: str
    origQty: str
    executedQty: str


# WebSocket ORDER_TRADE_UPDATE inner "o" dict
class OrderUpdatePayload(TypedDict):
    i: int  # orderId
    c: str  # clientOrderId
    s: str  # symbol
    S: str  # side
    o: str  # order type
    X: str  # order status
    x: str  # execution type
    ps: str  # position side
    l: str  # last filled qty  # noqa: E741  # Binance API field name
    L: str  # last filled price
    z: str  # cumulative filled qty
    ap: str  # average price
    n: str  # commission
    N: str  # commission asset
    t: int  # trade id
    T: int  # order trade time ms
    R: bool  # reduce only
    cp: bool  # close position
    rp: str  # realized profit


class OrderTradeUpdateEvent(TypedDict):
    e: str  # event type
    E: int  # event time ms
    T: int  # transaction time ms
    o: OrderUpdatePayload


# GET /fapi/v1/allOrders (array endpoint)
class AllOrderEntry(TypedDict):
    orderId: str | int
    status: str
    updateTime: int | float
    executedQty: str


# GET /fapi/v1/exchangeInfo
class FilterEntry(TypedDict, total=False):
    filterType: str
    minPrice: str
    maxPrice: str
    tickSize: str
    minQty: str
    maxQty: str
    stepSize: str
    notional: str


class SymbolInfo(TypedDict):
    symbol: str
    status: str
    baseAsset: str
    quoteAsset: str
    marginAsset: str
    settlAsset: str
    pricePrecision: int
    quantityPrecision: int
    baseAssetPrecision: int
    quotePrecision: int
    filters: list[FilterEntry]


class ExchangeInfoResponse(TypedDict):
    symbols: list[SymbolInfo]


# Error body
class ErrorBody(TypedDict):
    code: int
    msg: str
