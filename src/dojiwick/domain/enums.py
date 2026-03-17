"""Core enums for deterministic engine outcomes and actions."""

from enum import IntEnum, StrEnum, unique


class MarketState(IntEnum):
    """Coarse regime labels."""

    TRENDING_UP = 1
    TRENDING_DOWN = 2
    RANGING = 3
    VOLATILE = 4


def safe_market_state(code: int) -> MarketState | None:
    """Convert raw int to MarketState, returning None for unknown codes."""
    try:
        return MarketState(code)
    except ValueError:
        return None


MARKET_STATE_TO_SQL: dict[int, str] = {1: "trending_up", 2: "trending_down", 3: "ranging", 4: "volatile"}
SQL_TO_MARKET_STATE: dict[str, MarketState] = {v: MarketState(k) for k, v in MARKET_STATE_TO_SQL.items()}


class TradeAction(IntEnum):
    """Directional action codes used in vector kernels."""

    HOLD = 0
    BUY = 1
    SHORT = -1


TRADE_ACTION_TO_SQL: dict[int, str] = {0: "hold", 1: "buy", -1: "short"}
SQL_TO_TRADE_ACTION: dict[str, TradeAction] = {v: TradeAction(k) for k, v in TRADE_ACTION_TO_SQL.items()}


class DecisionAuthority(StrEnum):
    """Authority metadata for persisted outcomes."""

    DETERMINISTIC_ONLY = "deterministic_only"
    DETERMINISTIC_PLUS_AI_VETO = "deterministic_plus_ai_veto"
    DETERMINISTIC_PLUS_AI_REGIME = "deterministic_plus_ai_regime"
    DETERMINISTIC_PLUS_AI_REGIME_AND_VETO = "deterministic_plus_ai_regime_and_veto"


class DecisionStatus(StrEnum):
    """Final per-pair decision status."""

    EXECUTED = "executed"
    HOLD = "hold"
    BLOCKED_RISK = "blocked_risk"
    VETOED = "vetoed"
    ERROR = "error"


class ExecutionStatus(StrEnum):
    """Execution adapter receipt status."""

    FILLED = "filled"
    SKIPPED = "skipped"
    REJECTED = "rejected"
    ERROR = "error"
    CANCELLED = "cancelled"


class TickStatus(StrEnum):
    """Tick lifecycle status."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CloseReason(StrEnum):
    """Position close reason — values match SQL close_reason enum."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    PARTIAL_TP = "partial_tp"
    TRAILING_STOP = "trailing_stop"
    END_OF_BACKTEST = "end_of_backtest"
    EMERGENCY = "emergency"
    MANUAL = "manual"
    REPLACED = "replaced"
    LIQUIDATION = "liquidation"
    DOUBLE_FILL = "double_fill"
    TIME_EXIT = "time_exit"


class OrderSide(StrEnum):
    """Order direction."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Order type."""

    LIMIT = "limit"
    MARKET = "market"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"


class OrderStatus(StrEnum):
    """Order lifecycle status."""

    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class AuditSeverity(StrEnum):
    """Audit log severity — values match SQL audit_severity enum."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AdaptiveMode(StrEnum):
    """Adaptive policy operating mode."""

    DISABLED = "disabled"
    CONTINUOUS = "continuous"
    BUCKET_FALLBACK = "bucket_fallback"


class OrderEventType(StrEnum):
    """Order lifecycle event types."""

    PLACED = "placed"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


# Exchange / position / execution enums


@unique
class PositionSide(StrEnum):
    """Position side for hedge / one-way mode."""

    NET = "net"
    LONG = "long"
    SHORT = "short"


@unique
class PositionMode(StrEnum):
    """Position accounting mode."""

    ONE_WAY = "one_way"
    HEDGE = "hedge"


@unique
class PositionEventType(StrEnum):
    """Position lifecycle event types — values match SQL position_event_type enum."""

    OPEN = "open"
    ADD = "add"
    REDUCE = "reduce"
    CLOSE = "close"


@unique
class WorkingType(StrEnum):
    """Price reference for conditional orders."""

    MARK_PRICE = "mark_price"
    CONTRACT_PRICE = "contract_price"


@unique
class OrderLifecycleState(StrEnum):
    """Order lifecycle state machine for execution idempotency."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    FILLED = "filled"
    UNCERTAIN = "uncertain"
    REJECTED = "rejected"


@unique
class ReconciliationHealth(StrEnum):
    """Reconciliation health state machine for risk gating."""

    NORMAL = "normal"
    DEGRADED = "degraded"
    UNCERTAIN = "uncertain"
    HALT = "halt"


class WFMode(StrEnum):
    """Walk-forward validation mode."""

    RATIO = "ratio"
    OOS_SHARPE = "oos_sharpe"
    BOTH = "both"


class ObjectiveMode(StrEnum):
    """Optimization objective evaluation mode."""

    IS_OOS = "is_oos"
    WALK_FORWARD = "walk_forward"


class MissingBarPolicy(StrEnum):
    """Policy for handling stale/missing bar data in tick loop."""

    SKIP = "skip"
    ERROR = "error"
    LAST_KNOWN = "last_known"


class RegimeExitProfile(StrEnum):
    """Regime-adaptive exit profile mode."""

    DEFAULT = "default"
    ADAPTIVE = "adaptive"


class EntryPriceModel(StrEnum):
    """Entry price model for backtest fill simulation."""

    CLOSE = "close"
    NEXT_OPEN = "next_open"
    VWAP_PROXY = "vwap_proxy"
    WORST_CASE = "worst_case"


class SubmissionStatus(StrEnum):
    """Order submission acknowledgement status."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    PENDING = "pending"
    ERROR = "error"


class OrderTimeInForce(StrEnum):
    """Time-in-force qualifiers for orders."""

    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    GTX = "gtx"


class HistoryAlignment(StrEnum):
    """Timeline construction mode for backtest series."""

    INTERSECTION = "intersection"
    ROLLING_JOINED = "rolling_joined"


class BacktestGapPolicy(StrEnum):
    """How to handle mid-series data gaps in backtest replay."""

    FREEZE = "freeze"


class BenchmarkMode(StrEnum):
    """Benchmark construction mode for backtest comparison."""

    STATIC_FULL_WINDOW = "static_full_window"
    ROLLING_JOINED = "rolling_joined"


STATUS_TO_EVENT_TYPE: dict[OrderStatus, OrderEventType] = {
    OrderStatus.NEW: OrderEventType.PLACED,
    OrderStatus.PARTIALLY_FILLED: OrderEventType.PARTIALLY_FILLED,
    OrderStatus.FILLED: OrderEventType.FILLED,
    OrderStatus.CANCELED: OrderEventType.CANCELED,
    OrderStatus.EXPIRED: OrderEventType.EXPIRED,
    OrderStatus.REJECTED: OrderEventType.REJECTED,
}
