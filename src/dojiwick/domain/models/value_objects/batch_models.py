"""Batch domain models for vectorized live and offline compute."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from dojiwick.domain.indicator_schema import INDICATOR_COUNT
from dojiwick.domain.type_aliases import BoolVector, FloatMatrix, FloatVector, IntVector


def _ensure_float_vector(name: str, values: FloatVector, size: int) -> FloatVector:
    if values.dtype != np.float64:
        raise ValueError(f"{name} must be float64")
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if len(values) != size:
        raise ValueError(f"{name} length mismatch")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def _ensure_int_vector(name: str, values: IntVector, size: int) -> IntVector:
    if values.dtype != np.int64:
        raise ValueError(f"{name} must be int64")
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if len(values) != size:
        raise ValueError(f"{name} length mismatch")
    return values


def _ensure_bool_vector(name: str, values: BoolVector, size: int) -> BoolVector:
    if values.dtype != np.bool_:
        raise ValueError(f"{name} must be bool")
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if len(values) != size:
        raise ValueError(f"{name} length mismatch")
    return values


@dataclass(slots=True, frozen=True, kw_only=True)
class BatchMarketSnapshot:
    """Market batch for one tick across all active pairs."""

    pairs: tuple[str, ...]
    observed_at: datetime
    price: FloatVector
    indicators: FloatMatrix
    asof_timestamp: datetime | None = None
    volume: FloatVector | None = None

    def __post_init__(self) -> None:
        size = len(self.pairs)
        if size < 1:
            raise ValueError("pairs must not be empty")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if self.asof_timestamp is not None and self.asof_timestamp.tzinfo is None:
            raise ValueError("asof_timestamp must be timezone-aware")
        _ensure_float_vector("price", self.price, size)
        if self.indicators.dtype != np.float64:
            raise ValueError("indicators must be float64")
        if self.indicators.ndim != 2:
            raise ValueError("indicators must be 2-dimensional")
        if self.indicators.shape[0] != size:
            raise ValueError("indicators row count mismatch")
        if self.indicators.shape[1] != INDICATOR_COUNT:
            raise ValueError(f"indicators column count {self.indicators.shape[1]} != expected {INDICATOR_COUNT}")
        if not np.all(np.isfinite(self.indicators)):
            raise ValueError("indicators must be finite")
        if self.volume is not None:
            _ensure_float_vector("volume", self.volume, size)


@dataclass(slots=True, frozen=True, kw_only=True)
class BatchPortfolioSnapshot:
    """Portfolio batch aligned with market rows."""

    equity_usd: FloatVector
    day_start_equity_usd: FloatVector
    open_positions_total: IntVector
    has_open_position: BoolVector
    unrealized_pnl_usd: FloatVector

    def __post_init__(self) -> None:
        size = len(self.equity_usd)
        _ensure_float_vector("equity_usd", self.equity_usd, size)
        _ensure_float_vector("day_start_equity_usd", self.day_start_equity_usd, size)
        _ensure_int_vector("open_positions_total", self.open_positions_total, size)
        _ensure_bool_vector("has_open_position", self.has_open_position, size)
        _ensure_float_vector("unrealized_pnl_usd", self.unrealized_pnl_usd, size)


@dataclass(slots=True, frozen=True, kw_only=True)
class BatchDecisionContext:
    """Combined market and portfolio batch for one decision cycle.

    To add new data sources (funding rates, order book depth, etc.),
    add an optional field here, e.g.::

        funding: BatchFundingSnapshot | None = None

    The adapter composes them internally from separate exchange calls.
    Kernels that need the new data receive the full context.
    """

    market: BatchMarketSnapshot
    portfolio: BatchPortfolioSnapshot
    regimes: BatchRegimeProfile | None = None

    def __post_init__(self) -> None:
        if len(self.market.pairs) != len(self.portfolio.equity_usd):
            raise ValueError("market and portfolio batch sizes must match")

    @property
    def size(self) -> int:
        return len(self.market.pairs)


@dataclass(slots=True, frozen=True, kw_only=True)
class BatchRegimeProfile:
    """Vectorized regime output per pair."""

    coarse_state: IntVector
    confidence: FloatVector
    valid_mask: BoolVector

    def __post_init__(self) -> None:
        size = len(self.coarse_state)
        _ensure_int_vector("coarse_state", self.coarse_state, size)
        _ensure_float_vector("confidence", self.confidence, size)
        _ensure_bool_vector("valid_mask", self.valid_mask, size)


@dataclass(slots=True, frozen=True, kw_only=True)
class BatchTradeCandidate:
    """Vectorized candidate proposals from deterministic strategy kernels."""

    action: IntVector
    entry_price: FloatVector
    stop_price: FloatVector
    take_profit_price: FloatVector
    strategy_name: tuple[str, ...]
    strategy_variant: tuple[str, ...]
    reason_codes: tuple[str, ...]
    valid_mask: BoolVector

    def __post_init__(self) -> None:
        size = len(self.action)
        _ensure_int_vector("action", self.action, size)
        if not np.all(np.isin(self.action, np.array([-1, 0, 1], dtype=np.int64))):
            raise ValueError("action values must be in {-1, 0, 1}")
        _ensure_float_vector("entry_price", self.entry_price, size)
        _ensure_float_vector("stop_price", self.stop_price, size)
        _ensure_float_vector("take_profit_price", self.take_profit_price, size)
        _ensure_bool_vector("valid_mask", self.valid_mask, size)
        if len(self.strategy_name) != size:
            raise ValueError("strategy_name length mismatch")
        if len(self.strategy_variant) != size:
            raise ValueError("strategy_variant length mismatch")
        if len(self.reason_codes) != size:
            raise ValueError("reason_codes length mismatch")


@dataclass(slots=True, frozen=True, kw_only=True)
class BatchVetoDecision:
    """AI veto decision per pair."""

    approved_mask: BoolVector
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        size = len(self.approved_mask)
        _ensure_bool_vector("approved_mask", self.approved_mask, size)
        if len(self.reason_codes) != size:
            raise ValueError("reason_codes length mismatch")


@dataclass(slots=True, frozen=True, kw_only=True)
class BatchRiskAssessment:
    """Risk assessment outputs for each pair."""

    allowed_mask: BoolVector
    reason_codes: tuple[str, ...]
    risk_score: FloatVector

    def __post_init__(self) -> None:
        size = len(self.allowed_mask)
        _ensure_bool_vector("allowed_mask", self.allowed_mask, size)
        _ensure_float_vector("risk_score", self.risk_score, size)
        if len(self.reason_codes) != size:
            raise ValueError("reason_codes length mismatch")


@dataclass(slots=True, frozen=True, kw_only=True)
class BatchExecutionIntent:
    """Execution-ready vectors and activation mask."""

    pairs: tuple[str, ...]
    action: IntVector
    quantity: FloatVector
    notional_usd: FloatVector
    entry_price: FloatVector
    stop_price: FloatVector
    take_profit_price: FloatVector
    strategy_name: tuple[str, ...]
    strategy_variant: tuple[str, ...]
    active_mask: BoolVector

    def __post_init__(self) -> None:
        size = len(self.pairs)
        _ensure_float_vector("quantity", self.quantity, size)
        _ensure_float_vector("notional_usd", self.notional_usd, size)
        _ensure_float_vector("entry_price", self.entry_price, size)
        _ensure_float_vector("stop_price", self.stop_price, size)
        _ensure_float_vector("take_profit_price", self.take_profit_price, size)
        _ensure_bool_vector("active_mask", self.active_mask, size)
        if len(self.strategy_name) != size:
            raise ValueError("strategy_name length mismatch")
        if len(self.strategy_variant) != size:
            raise ValueError("strategy_variant length mismatch")


@dataclass(slots=True, frozen=True, kw_only=True)
class BatchSignalFragment:
    """Per-strategy signal fragment for plugin composition."""

    strategy_name: str
    buy_mask: BoolVector
    short_mask: BoolVector

    def __post_init__(self) -> None:
        if not self.strategy_name:
            raise ValueError("strategy_name must not be empty")
        size = len(self.buy_mask)
        _ensure_bool_vector("buy_mask", self.buy_mask, size)
        _ensure_bool_vector("short_mask", self.short_mask, size)


@dataclass(slots=True, frozen=True, kw_only=True)
class RiskRuleDecision:
    """Individual risk rule evaluation result."""

    rule_name: str
    blocked_mask: BoolVector
    reason_code: str
    precedence: int
    risk_score: float

    def __post_init__(self) -> None:
        if not self.rule_name:
            raise ValueError("rule_name must not be empty")
        if not self.reason_code:
            raise ValueError("reason_code must not be empty")
        if self.precedence < 0:
            raise ValueError("precedence must be non-negative")
        if not 0 <= self.risk_score <= 1:
            raise ValueError("risk_score must be in [0, 1]")
        _ensure_bool_vector("blocked_mask", self.blocked_mask, len(self.blocked_mask))
