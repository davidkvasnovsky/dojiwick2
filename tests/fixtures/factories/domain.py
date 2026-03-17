"""Fluent builders for domain model test data."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Self

import numpy as np

from dojiwick.application.use_cases.run_backtest import BacktestTimeSeries
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchMarketSnapshot,
    BatchPortfolioSnapshot,
)
from dojiwick.domain.models.entities.bot_state import BotState
from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.enums import (
    AuditSeverity,
    OrderEventType,
    PositionSide,
)
from dojiwick.domain.type_aliases import ProductCode, VenueCode
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_USD_C, BINANCE_VENUE
from dojiwick.domain.indicator_schema import INDICATOR_COUNT, INDICATOR_INDEX
from dojiwick.domain.numerics import Confidence, Money, Quantity, to_money, to_price, to_quantity
from dojiwick.domain.type_aliases import CandleInterval
from dojiwick.domain.models.entities.pair_state import PairTradingState
from dojiwick.domain.models.value_objects.adaptive import AdaptiveArmKey, AdaptiveOutcomeEvent
from dojiwick.domain.models.value_objects.ai_evaluation import AIEvaluationResult
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId, PositionLegKey, TargetLegPosition
from dojiwick.domain.models.value_objects.order_event import OrderEvent
from dojiwick.domain.models.value_objects.signal import Signal
from dojiwick.domain.models.value_objects.system_event import SystemEvent
from dojiwick.domain.type_aliases import FloatMatrix


class ContextBuilder:
    """Fluent builder for BatchDecisionContext."""

    def __init__(self, pairs: tuple[str, ...] = ("BTC/USDC", "ETH/USDC")) -> None:
        self._pairs = pairs
        size = len(pairs)
        self._prices = np.full(size, 100.0, dtype=np.float64)
        self._indicators = np.full((size, INDICATOR_COUNT), 50.0, dtype=np.float64)
        self._equity = np.full(size, 1_000.0, dtype=np.float64)
        self._day_start_equity = np.full(size, 1_000.0, dtype=np.float64)
        self._open_positions = np.zeros(size, dtype=np.int64)
        self._has_open = np.zeros(size, dtype=np.bool_)
        self._unrealized = np.zeros(size, dtype=np.float64)
        self._observed_at = datetime.now(UTC)

    def with_prices(self, values: list[float]) -> Self:
        self._prices = np.array(values, dtype=np.float64)
        return self

    def with_indicator(self, name: str, values: list[float]) -> Self:
        self._indicators[:, INDICATOR_INDEX[name]] = np.array(values, dtype=np.float64)
        return self

    def with_rsi(self, values: list[float]) -> Self:
        return self.with_indicator("rsi", values)

    def with_adx(self, values: list[float]) -> Self:
        return self.with_indicator("adx", values)

    def with_atr(self, values: list[float]) -> Self:
        return self.with_indicator("atr", values)

    def with_equity(self, values: list[float]) -> Self:
        self._equity = np.array(values, dtype=np.float64)
        self._day_start_equity = np.array(values, dtype=np.float64)
        return self

    def with_day_start_equity(self, values: list[float]) -> Self:
        self._day_start_equity = np.array(values, dtype=np.float64)
        return self

    def with_open_positions_total(self, values: list[int]) -> Self:
        self._open_positions = np.array(values, dtype=np.int64)
        return self

    def with_indicators(self, indicators: FloatMatrix) -> Self:
        self._indicators = indicators
        return self

    def with_observed_at(self, at: datetime) -> Self:
        self._observed_at = at
        return self

    def trending_up(self) -> Self:
        """Preset: trending up regime (high ADX, fast > slow > base, moderate RSI)."""
        size = len(self._pairs)
        self._indicators[:, INDICATOR_INDEX["rsi"]] = np.full(size, 40.0, dtype=np.float64)
        self._indicators[:, INDICATOR_INDEX["adx"]] = np.full(size, 28.0, dtype=np.float64)
        self._indicators[:, INDICATOR_INDEX["atr"]] = np.full(size, 0.5, dtype=np.float64)
        self._indicators[:, INDICATOR_INDEX["ema_fast"]] = self._prices + 1.0
        self._indicators[:, INDICATOR_INDEX["ema_slow"]] = self._prices + 0.5
        self._indicators[:, INDICATOR_INDEX["ema_base"]] = self._prices - 0.5
        self._indicators[:, INDICATOR_INDEX["bb_upper"]] = self._prices + 4.0
        self._indicators[:, INDICATOR_INDEX["bb_lower"]] = self._prices - 4.0
        return self

    def ranging(self) -> Self:
        """Preset: ranging regime (low ADX, tight EMAs)."""
        size = len(self._pairs)
        self._indicators[:, INDICATOR_INDEX["rsi"]] = np.full(size, 50.0, dtype=np.float64)
        self._indicators[:, INDICATOR_INDEX["adx"]] = np.full(size, 15.0, dtype=np.float64)
        self._indicators[:, INDICATOR_INDEX["atr"]] = np.full(size, 0.3, dtype=np.float64)
        self._indicators[:, INDICATOR_INDEX["ema_fast"]] = self._prices + 0.1
        self._indicators[:, INDICATOR_INDEX["ema_slow"]] = self._prices + 0.05
        self._indicators[:, INDICATOR_INDEX["ema_base"]] = self._prices
        self._indicators[:, INDICATOR_INDEX["bb_upper"]] = self._prices + 4.0
        self._indicators[:, INDICATOR_INDEX["bb_lower"]] = self._prices - 4.0
        return self

    def mean_revert_buy(self) -> Self:
        """Preset: mean-revert buy signal (RANGING + RSI oversold + price at bb_lower)."""
        size = len(self._pairs)
        low_prices = self._prices - 4.0
        self._prices = low_prices
        self._indicators[:, INDICATOR_INDEX["rsi"]] = np.full(size, 30.0, dtype=np.float64)
        self._indicators[:, INDICATOR_INDEX["adx"]] = np.full(size, 15.0, dtype=np.float64)
        self._indicators[:, INDICATOR_INDEX["atr"]] = np.full(size, 0.5, dtype=np.float64)
        self._indicators[:, INDICATOR_INDEX["ema_fast"]] = low_prices + 4.0
        self._indicators[:, INDICATOR_INDEX["ema_slow"]] = low_prices + 3.0
        self._indicators[:, INDICATOR_INDEX["ema_base"]] = low_prices + 2.0
        self._indicators[:, INDICATOR_INDEX["ema_trend"]] = low_prices + 1.0
        self._indicators[:, INDICATOR_INDEX["bb_upper"]] = low_prices + 8.0
        self._indicators[:, INDICATOR_INDEX["bb_lower"]] = low_prices
        return self

    def build(self) -> BatchDecisionContext:
        return BatchDecisionContext(
            market=BatchMarketSnapshot(
                pairs=self._pairs,
                observed_at=self._observed_at,
                price=self._prices,
                indicators=self._indicators,
            ),
            portfolio=BatchPortfolioSnapshot(
                equity_usd=self._equity,
                day_start_equity_usd=self._day_start_equity,
                open_positions_total=self._open_positions,
                has_open_position=self._has_open,
                unrealized_pnl_usd=self._unrealized,
            ),
        )


class TimeSeriesBuilder:
    """Fluent builder for BacktestTimeSeries (multi-bar replay data)."""

    _PRESET_MAP: dict[str, str] = {
        "trending_up": "trending_up",
        "ranging": "ranging",
        "mean_revert_buy": "mean_revert_buy",
    }

    def __init__(
        self,
        n_bars: int = 5,
        pairs: tuple[str, ...] = ("BTC/USDC", "ETH/USDC"),
    ) -> None:
        self._n_bars = n_bars
        self._pairs = pairs
        self._presets: list[str] = ["trending_up"] * n_bars
        self._price_deltas: list[list[float]] | None = None
        self._base_time = datetime.now(UTC)

    def with_regime_sequence(self, presets: list[str]) -> Self:
        """Set per-bar regime presets (e.g. ``["trending_up", "ranging", ...]``)."""
        if len(presets) != self._n_bars:
            raise ValueError(f"expected {self._n_bars} presets, got {len(presets)}")
        for p in presets:
            if p not in self._PRESET_MAP:
                raise ValueError(f"unknown preset: {p!r}")
        self._presets = presets
        return self

    def with_price_deltas(self, deltas: list[list[float]]) -> Self:
        """Set per-bar next_price deltas from current price."""
        if len(deltas) != self._n_bars:
            raise ValueError(f"expected {self._n_bars} delta rows, got {len(deltas)}")
        self._price_deltas = deltas
        return self

    def build(self) -> BacktestTimeSeries:
        contexts: list[BatchDecisionContext] = []
        next_prices: list[np.ndarray] = []

        rng = np.random.default_rng(42)

        for bar_idx in range(self._n_bars):
            builder = ContextBuilder(pairs=self._pairs)
            builder.with_observed_at(self._base_time + timedelta(minutes=bar_idx))
            preset_method = getattr(builder, self._PRESET_MAP[self._presets[bar_idx]])
            preset_method()
            ctx = builder.build()
            contexts.append(ctx)

            if self._price_deltas is not None:
                deltas = np.array(self._price_deltas[bar_idx], dtype=np.float64)
            else:
                deltas = rng.uniform(-0.5, 0.5, size=len(self._pairs))
            next_prices.append(ctx.market.price + deltas)

        return BacktestTimeSeries(
            contexts=tuple(contexts),
            next_prices=tuple(next_prices),
        )


class CandleBuilder:
    """Fluent builder for Candle with presets."""

    def __init__(self) -> None:
        self._pair = "BTC/USDC"
        self._interval: CandleInterval = CandleInterval("1h")
        self._open_time = datetime.now(UTC)
        self._open = Decimal("100")
        self._high = Decimal("105")
        self._low = Decimal("95")
        self._close = Decimal("102")
        self._volume = Decimal("1000")
        self._quote_volume = Decimal("100000")

    def with_pair(self, pair: str) -> Self:
        self._pair = pair
        return self

    def with_interval(self, interval: CandleInterval) -> Self:
        self._interval = interval
        return self

    def with_open_time(self, at: datetime) -> Self:
        self._open_time = at
        return self

    def with_ohlcv(
        self,
        o: Decimal | str | int | float,
        h: Decimal | str | int | float,
        low: Decimal | str | int | float,
        c: Decimal | str | int | float,
        v: Decimal | str | int | float,
    ) -> Self:
        self._open = to_price(o)
        self._high = to_price(h)
        self._low = to_price(low)
        self._close = to_price(c)
        self._volume = to_quantity(v)
        return self

    def bullish(self) -> Self:
        """Preset: bullish candle (close > open, high near close)."""
        self._open = Decimal("100")
        self._high = Decimal("106")
        self._low = Decimal("99")
        self._close = Decimal("105")
        self._volume = Decimal("1500")
        return self

    def bearish(self) -> Self:
        """Preset: bearish candle (close < open, low near close)."""
        self._open = Decimal("105")
        self._high = Decimal("106")
        self._low = Decimal("99")
        self._close = Decimal("100")
        self._volume = Decimal("1500")
        return self

    def doji(self) -> Self:
        """Preset: doji candle (open ≈ close, long wicks)."""
        self._open = Decimal("100")
        self._high = Decimal("105")
        self._low = Decimal("95")
        self._close = Decimal("100.1")
        self._volume = Decimal("500")
        return self

    def build(self) -> Candle:
        return Candle(
            pair=self._pair,
            interval=self._interval,
            open_time=self._open_time,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            quote_volume=self._quote_volume,
        )


class SignalBuilder:
    """Fluent builder for Signal."""

    def __init__(self) -> None:
        self._pair = "BTC/USDC"
        self._signal_type = "breakout"
        self._priority = 0
        self._details: dict[str, object] | None = None
        self._detected_at: datetime | None = datetime.now(UTC)
        self._decision_outcome_id: int | None = None

    def with_pair(self, pair: str) -> Self:
        self._pair = pair
        return self

    def with_type(self, signal_type: str) -> Self:
        self._signal_type = signal_type
        return self

    def with_priority(self, priority: int) -> Self:
        self._priority = priority
        return self

    def with_details(self, details: dict[str, object]) -> Self:
        self._details = details
        return self

    def build(self) -> Signal:
        return Signal(
            pair=self._pair,
            target_id=self._pair,
            signal_type=self._signal_type,
            priority=self._priority,
            details=self._details,
            detected_at=self._detected_at,
            decision_outcome_id=self._decision_outcome_id,
        )


class BotStateBuilder:
    """Fluent builder for BotState with presets."""

    def __init__(self) -> None:
        self._state = BotState()

    def circuit_broken(self) -> Self:
        """Preset: circuit breaker is active."""
        self._state.circuit_breaker_active = True
        self._state.consecutive_errors = 5
        return self

    def at_loss_limit(self, daily_pnl: Decimal | str | int | float = Decimal("-500"), losses: int = 5) -> Self:
        """Preset: at consecutive loss limit."""
        self._state.consecutive_losses = losses
        self._state.daily_pnl_usd = to_money(daily_pnl)
        return self

    def with_consecutive_errors(self, n: int) -> Self:
        self._state.consecutive_errors = n
        return self

    def with_daily_trade_count(self, n: int) -> Self:
        self._state.daily_trade_count = n
        return self

    def build(self) -> BotState:
        return BotState(
            consecutive_errors=self._state.consecutive_errors,
            consecutive_losses=self._state.consecutive_losses,
            daily_trade_count=self._state.daily_trade_count,
            daily_pnl_usd=self._state.daily_pnl_usd,
            circuit_breaker_active=self._state.circuit_breaker_active,
            circuit_breaker_until=self._state.circuit_breaker_until,
            last_tick_at=self._state.last_tick_at,
            last_decay_at=self._state.last_decay_at,
            daily_reset_at=self._state.daily_reset_at,
        )


class PairTradingStateBuilder:
    """Fluent builder for PairTradingState with presets."""

    def __init__(self) -> None:
        self._pair = "BTC/USDC"
        self._venue = "binance"
        self._product = "usd_c"
        self._wins = 0
        self._losses = 0
        self._consecutive_losses = 0
        self._last_trade_at: datetime | None = None
        self._blocked = False

    def with_pair(self, pair: str) -> Self:
        self._pair = pair
        return self

    def on_losing_streak(self, consecutive: int = 5) -> Self:
        """Preset: pair on a losing streak."""
        self._consecutive_losses = consecutive
        self._losses = consecutive
        return self

    def blocked(self) -> Self:
        """Preset: pair is blocked."""
        self._blocked = True
        return self

    def with_record(self, wins: int, losses: int) -> Self:
        self._wins = wins
        self._losses = losses
        return self

    def build(self) -> PairTradingState:
        return PairTradingState(
            pair=self._pair,
            target_id=self._pair,
            venue=self._venue,
            product=self._product,
            wins=self._wins,
            losses=self._losses,
            consecutive_losses=self._consecutive_losses,
            last_trade_at=self._last_trade_at,
            blocked=self._blocked,
        )


class OrderEventBuilder:
    """Fluent builder for OrderEvent with presets."""

    def __init__(self) -> None:
        self._order_id = 1
        self._event_type = OrderEventType.PLACED
        self._occurred_at = datetime.now(UTC)
        self._exchange_order_id = ""
        self._filled_quantity = Decimal(0)
        self._fees_usd = Decimal(0)
        self._detail = ""

    def with_order_id(self, order_id: int) -> Self:
        self._order_id = order_id
        return self

    def with_event_type(self, event_type: OrderEventType) -> Self:
        self._event_type = event_type
        return self

    def with_occurred_at(self, at: datetime) -> Self:
        self._occurred_at = at
        return self

    def filled(
        self,
        quantity: Decimal | str | int | float = Decimal("0.1"),
        fees: Decimal | str | int | float = Decimal("0.05"),
    ) -> Self:
        """Preset: filled order event."""
        self._event_type = OrderEventType.FILLED
        self._filled_quantity = to_quantity(quantity)
        self._fees_usd = to_money(fees)
        return self

    def canceled(self, detail: str = "user_cancel") -> Self:
        """Preset: canceled order event."""
        self._event_type = OrderEventType.CANCELED
        self._detail = detail
        return self

    def build(self) -> OrderEvent:
        return OrderEvent(
            order_id=self._order_id,
            event_type=self._event_type,
            occurred_at=self._occurred_at,
            exchange_order_id=self._exchange_order_id,
            filled_quantity=self._filled_quantity,
            fees_usd=self._fees_usd,
            detail=self._detail,
        )


class SystemEventBuilder:
    """Fluent builder for SystemEvent."""

    def __init__(self) -> None:
        self._component = "tick_service"
        self._severity = AuditSeverity.INFO
        self._message = "tick completed"
        self._correlation_id = ""
        self._context: dict[str, object] | None = None
        self._occurred_at: datetime | None = datetime.now(UTC)

    def with_component(self, component: str) -> Self:
        self._component = component
        return self

    def with_severity(self, severity: AuditSeverity) -> Self:
        self._severity = severity
        return self

    def with_message(self, message: str) -> Self:
        self._message = message
        return self

    def warning(self, message: str = "rate limit approaching") -> Self:
        """Preset: warning event."""
        self._severity = AuditSeverity.WARNING
        self._message = message
        return self

    def critical(self, message: str = "circuit breaker tripped") -> Self:
        """Preset: critical event."""
        self._severity = AuditSeverity.CRITICAL
        self._message = message
        return self

    def build(self) -> SystemEvent:
        return SystemEvent(
            component=self._component,
            severity=self._severity,
            message=self._message,
            correlation_id=self._correlation_id,
            context=self._context,
            occurred_at=self._occurred_at,
        )


class AdaptiveOutcomeBuilder:
    """Fluent builder for AdaptiveOutcomeEvent."""

    def __init__(self) -> None:
        self._position_leg_id = 1
        self._arm = AdaptiveArmKey(regime_idx=0, config_idx=0)
        self._reward = 0.5
        self._observed_at = datetime.now(UTC)

    def with_position_leg_id(self, position_leg_id: int) -> Self:
        self._position_leg_id = position_leg_id
        return self

    def with_arm(self, regime_idx: int = 0, config_idx: int = 0) -> Self:
        self._arm = AdaptiveArmKey(regime_idx=regime_idx, config_idx=config_idx)
        return self

    def with_reward(self, reward: float) -> Self:
        self._reward = reward
        return self

    def with_observed_at(self, at: datetime) -> Self:
        self._observed_at = at
        return self

    def success(self, reward: float = 0.9) -> Self:
        """Preset: successful outcome."""
        self._reward = reward
        return self

    def failure(self, reward: float = 0.1) -> Self:
        """Preset: poor outcome."""
        self._reward = reward
        return self

    def build(self) -> AdaptiveOutcomeEvent:
        return AdaptiveOutcomeEvent(
            position_leg_id=self._position_leg_id,
            arm=self._arm,
            reward=self._reward,
            observed_at=self._observed_at,
        )


class StrategyStateBuilder:
    """Fluent builder for strategy state dicts (as stored in StrategyStateRepositoryPort)."""

    def __init__(self) -> None:
        self._pair = "BTC/USDC"
        self._strategy_name = "trend_follow"
        self._variant = "baseline"
        self._state: dict[str, object] = {}

    def with_pair(self, pair: str) -> Self:
        self._pair = pair
        return self

    def with_strategy(self, name: str, variant: str = "baseline") -> Self:
        self._strategy_name = name
        self._variant = variant
        return self

    def with_state(self, state: dict[str, object]) -> Self:
        self._state = state
        return self

    def build(self) -> dict[str, object]:
        return {
            "pair": self._pair,
            "strategy_name": self._strategy_name,
            "variant": self._variant,
            **self._state,
        }


# Exchange / position / evaluation builders


class InstrumentIdBuilder:
    """Fluent builder for InstrumentId with Binance USD-C BTC/USDC defaults."""

    def __init__(self) -> None:
        self._venue = BINANCE_VENUE
        self._product = BINANCE_USD_C
        self._symbol = "BTCUSDC"
        self._base_asset = "BTC"
        self._quote_asset = "USDC"
        self._settle_asset = "USDC"

    def with_venue(self, venue: VenueCode) -> Self:
        self._venue = venue
        return self

    def with_product(self, product: ProductCode) -> Self:
        self._product = product
        return self

    def with_symbol(self, symbol: str, base: str, quote: str, settle: str | None = None) -> Self:
        self._symbol = symbol
        self._base_asset = base
        self._quote_asset = quote
        self._settle_asset = settle or quote
        return self

    def eth_usdc(self) -> Self:
        """Preset: Binance USD-C ETH/USDC."""
        self._symbol = "ETHUSDC"
        self._base_asset = "ETH"
        self._quote_asset = "USDC"
        self._settle_asset = "USDC"
        return self

    def build(self) -> InstrumentId:
        return InstrumentId(
            venue=self._venue,
            product=self._product,
            symbol=self._symbol,
            base_asset=self._base_asset,
            quote_asset=self._quote_asset,
            settle_asset=self._settle_asset,
        )


class PositionLegKeyBuilder:
    """Fluent builder for PositionLegKey with sensible defaults."""

    def __init__(self) -> None:
        self._account = "default"
        self._instrument_id = InstrumentIdBuilder().build()
        self._position_side = PositionSide.NET

    def with_account(self, account: str) -> Self:
        self._account = account
        return self

    def with_instrument_id(self, instrument_id: InstrumentId) -> Self:
        self._instrument_id = instrument_id
        return self

    def with_position_side(self, side: PositionSide) -> Self:
        self._position_side = side
        return self

    def long(self) -> Self:
        """Preset: LONG side."""
        self._position_side = PositionSide.LONG
        return self

    def short(self) -> Self:
        """Preset: SHORT side."""
        self._position_side = PositionSide.SHORT
        return self

    def build(self) -> PositionLegKey:
        return PositionLegKey(
            account=self._account,
            instrument_id=self._instrument_id,
            position_side=self._position_side,
        )


class TargetLegPositionBuilder:
    """Fluent builder for TargetLegPosition with sensible defaults."""

    def __init__(self) -> None:
        self._account = "default"
        self._instrument_id = InstrumentIdBuilder().build()
        self._position_side = PositionSide.NET
        self._target_notional: Money | None = None
        self._target_qty: Quantity | None = Decimal("0.1")

    def with_account(self, account: str) -> Self:
        self._account = account
        return self

    def with_instrument_id(self, instrument_id: InstrumentId) -> Self:
        self._instrument_id = instrument_id
        return self

    def with_position_side(self, side: PositionSide) -> Self:
        self._position_side = side
        return self

    def with_target_notional(self, notional: Decimal | str | int | float) -> Self:
        self._target_notional = to_money(notional)
        self._target_qty = None
        return self

    def with_target_qty(self, qty: Decimal | str | int | float) -> Self:
        self._target_qty = to_quantity(qty)
        self._target_notional = None
        return self

    def long(self, qty: Decimal | str | int | float = Decimal("0.1")) -> Self:
        """Preset: long target position."""
        self._position_side = PositionSide.LONG
        self._target_qty = to_quantity(qty)
        self._target_notional = None
        return self

    def short(self, qty: Decimal | str | int | float = Decimal("0.1")) -> Self:
        """Preset: short target position."""
        self._position_side = PositionSide.SHORT
        self._target_qty = to_quantity(qty)
        self._target_notional = None
        return self

    def build(self) -> TargetLegPosition:
        return TargetLegPosition(
            account=self._account,
            instrument_id=self._instrument_id,
            position_side=self._position_side,
            target_notional=self._target_notional,
            target_qty=self._target_qty,
        )


class AIEvaluationResultBuilder:
    """Fluent builder for AIEvaluationResult with sensible defaults."""

    def __init__(self) -> None:
        self._approval = True
        self._confidence: Confidence = 0.85
        self._reason_code = "AI_APPROVE"
        self._rationale = "Signal meets evaluation criteria"
        self._policy_flags: dict[str, Any] = {}

    def with_approval(self, approval: bool) -> Self:
        self._approval = approval
        return self

    def with_confidence(self, confidence: Confidence) -> Self:
        self._confidence = confidence
        return self

    def with_reason_code(self, code: str) -> Self:
        self._reason_code = code
        return self

    def with_rationale(self, rationale: str) -> Self:
        self._rationale = rationale
        return self

    def with_policy_flags(self, flags: dict[str, Any]) -> Self:
        self._policy_flags = flags
        return self

    def approved(self, confidence: Confidence = 0.95) -> Self:
        """Preset: high-confidence approval."""
        self._approval = True
        self._confidence = confidence
        self._reason_code = "AI_APPROVE"
        self._rationale = "Strong signal with high conviction"
        return self

    def vetoed(self, reason: str = "Risk too high") -> Self:
        """Preset: vetoed by AI."""
        self._approval = False
        self._confidence = 0.80
        self._reason_code = "AI_VETO"
        self._rationale = reason
        return self

    def low_confidence(self) -> Self:
        """Preset: approved but with low confidence."""
        self._approval = True
        self._confidence = 0.45
        self._reason_code = "AI_APPROVE"
        self._rationale = "Marginal signal, low conviction"
        return self

    def build(self) -> AIEvaluationResult:
        return AIEvaluationResult(
            approval=self._approval,
            confidence=self._confidence,
            reason_code=self._reason_code,
            rationale=self._rationale,
            policy_flags=self._policy_flags,
        )
