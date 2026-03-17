"""Output models for execution, persistence, and analytics."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

import numpy as np

from dojiwick.domain.enums import (
    CloseReason,
    DecisionAuthority,
    DecisionStatus,
    ExecutionStatus,
    MarketState,
    TradeAction,
)
from dojiwick.domain.numerics import Confidence, Money, Price, Quantity


type ConfusionMatrix = dict[str, dict[str, int]]


@dataclass(slots=True, frozen=True, kw_only=True)
class ExecutionReceipt:
    """Per-pair execution receipt."""

    status: ExecutionStatus
    reason: str
    fill_price: Price | None = None
    filled_quantity: Quantity = Decimal(0)
    order_id: str = ""
    exchange_timestamp: datetime | None = None
    fees_usd: Money = Decimal(0)
    fee_asset: str = ""
    native_fee_amount: Money = Decimal(0)

    def __post_init__(self) -> None:
        if self.native_fee_amount < 0:
            raise ValueError("native_fee_amount must be non-negative")
        if self.status is ExecutionStatus.FILLED:
            if self.fill_price is None or self.fill_price <= 0:
                raise ValueError("fill_price must be positive when status is FILLED")
            if self.filled_quantity <= 0:
                raise ValueError("filled_quantity must be positive when status is FILLED")
            if self.exchange_timestamp is None:
                raise ValueError("exchange_timestamp is required when status is FILLED")
        if self.status is ExecutionStatus.CANCELLED:
            if self.fill_price is not None:
                raise ValueError("fill_price must be None when status is CANCELLED")
            if self.filled_quantity != 0:
                raise ValueError("filled_quantity must be 0 when status is CANCELLED")


@dataclass(slots=True, frozen=True, kw_only=True)
class DecisionOutcome:
    """Final per-pair decision artifact."""

    pair: str
    target_id: str
    observed_at: datetime
    authority: DecisionAuthority
    status: DecisionStatus
    market_state: MarketState
    action: TradeAction
    strategy_name: str
    strategy_variant: str
    reason_code: str
    confidence: Confidence
    entry_price: Price
    stop_price: Price
    take_profit_price: Price
    quantity: Quantity
    notional_usd: Money
    config_hash: str
    order_id: str = ""
    note: str = ""
    tick_id: str = ""
    confidence_raw: float = 0.0

    def __post_init__(self) -> None:
        if not self.pair:
            raise ValueError("pair must not be empty")
        if not self.target_id:
            raise ValueError("target_id must not be empty")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if not self.config_hash:
            raise ValueError("config_hash must not be empty")


@dataclass(slots=True, frozen=True, kw_only=True)
class TradeDetail:
    """Per-trade detail for debuggability."""

    bar_index: int
    exit_bar_index: int = 0
    hold_bars: int = 1
    close_reason: CloseReason = CloseReason.STOP_LOSS
    pair: str
    strategy_name: str
    action: TradeAction
    entry_price: float
    exit_price: float
    quantity: float
    notional_usd: float
    pnl_usd: float
    regime: MarketState | None = None
    regime_confidence: float = 0.0
    atr_at_entry: float = 0.0
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    strategy_variant: str = ""


def pick_effective_drawdown(max_portfolio_dd: float, max_trade_dd: float) -> float:
    """Prefer portfolio DD when available, fall back to per-trade DD."""
    return max_portfolio_dd if max_portfolio_dd > 0 else max_trade_dd


@dataclass(slots=True, frozen=True, kw_only=True)
class BacktestSummary:
    """Vectorized backtest summary metrics."""

    trades: int
    total_pnl_usd: float
    win_rate: float
    expectancy_usd: float
    sharpe_like: float
    max_drawdown_pct: float
    sortino: float = 0.0
    calmar: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_losses: int = 0
    payoff_ratio: float = 0.0
    avg_notional_usd: float = 0.0
    equity_curve: np.ndarray | None = field(default=None, repr=False)
    drawdown_curve: np.ndarray | None = field(default=None, repr=False)
    config_hash: str = ""
    benchmark_pnl_usd: float = 0.0
    max_portfolio_drawdown_pct: float = 0.0
    portfolio_equity_curve: np.ndarray | None = field(default=None, repr=False)
    portfolio_drawdown_curve: np.ndarray | None = field(default=None, repr=False)
    daily_sharpe: float = 0.0

    @property
    def effective_max_drawdown_pct(self) -> float:
        return pick_effective_drawdown(self.max_portfolio_drawdown_pct, self.max_drawdown_pct)


@dataclass(slots=True, frozen=True, kw_only=True)
class BacktestResult:
    """Full backtest output wrapping summary + trade details."""

    summary: BacktestSummary
    trade_details: tuple[TradeDetail, ...] = ()
    monthly_pnl: dict[str, float] | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class RegimeEvaluationReport:
    """Regime quality report for optimization and diagnostics."""

    total_points: int
    coarse_confusion: ConfusionMatrix
    coarse_macro_f1: float
    flip_rate: float
    mean_run_length: float
    transition_entropy: float
