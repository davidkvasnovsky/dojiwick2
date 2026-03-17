"""Fluent builders for compute artifact test data."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Self

import numpy as np
import numpy.typing as npt

from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchExecutionIntent,
    BatchMarketSnapshot,
    BatchPortfolioSnapshot,
    BatchRegimeProfile,
    BatchRiskAssessment,
    BatchTradeCandidate,
    BatchVetoDecision,
)
from dojiwick.domain.enums import (
    DecisionAuthority,
    DecisionStatus,
    ExecutionStatus,
    MarketState,
    TradeAction,
)
from dojiwick.domain.indicator_schema import INDICATOR_COUNT, INDICATOR_INDEX
from dojiwick.domain.numerics import to_money, to_price, to_quantity
from dojiwick.application.orchestration.outcome_assembler import OutcomeInputs
from dojiwick.domain.models.value_objects.outcome_models import DecisionOutcome, ExecutionReceipt


def make_indicator_matrix(size: int = 1, **overrides: float) -> npt.NDArray[np.float64]:
    """Build an indicator matrix with sensible defaults and optional overrides."""
    ind = np.full((size, INDICATOR_COUNT), 50.0, dtype=np.float64)
    for name, value in overrides.items():
        ind[:, INDICATOR_INDEX[name]] = value
    return ind


class ExecutionReceiptBuilder:
    """Fluent builder for ExecutionReceipt with preset helpers."""

    def __init__(self) -> None:
        self._status = ExecutionStatus.SKIPPED
        self._reason = "inactive"
        self._fill_price: Decimal | None = None
        self._filled_quantity = Decimal(0)
        self._order_id = ""
        self._exchange_timestamp: datetime | None = None
        self._fees_usd = Decimal(0)
        self._fee_asset = ""
        self._native_fee_amount = Decimal(0)

    def filled(
        self,
        price: Decimal | str | int | float = Decimal("100"),
        quantity: Decimal | str | int | float = Decimal("0.1"),
    ) -> Self:
        """Preset: filled receipt."""
        self._status = ExecutionStatus.FILLED
        self._reason = "filled"
        self._fill_price = to_price(price)
        self._filled_quantity = to_quantity(quantity)
        self._exchange_timestamp = datetime.now(UTC)
        return self

    def skipped(self, reason: str = "inactive") -> Self:
        """Preset: skipped receipt."""
        self._status = ExecutionStatus.SKIPPED
        self._reason = reason
        return self

    def rejected(self, reason: str = "rejected") -> Self:
        """Preset: rejected receipt."""
        self._status = ExecutionStatus.REJECTED
        self._reason = reason
        return self

    def errored(self, reason: str = "gateway_error") -> Self:
        """Preset: error receipt."""
        self._status = ExecutionStatus.ERROR
        self._reason = reason
        return self

    def cancelled(self, reason: str = "cancel_success") -> Self:
        """Preset: cancelled receipt."""
        self._status = ExecutionStatus.CANCELLED
        self._reason = reason
        return self

    def with_order_id(self, oid: str) -> Self:
        self._order_id = oid
        return self

    def with_fees(self, fees: Decimal | str | int | float) -> Self:
        self._fees_usd = to_money(fees)
        return self

    def with_fee_asset(self, asset: str) -> Self:
        self._fee_asset = asset
        return self

    def with_native_fee(self, amount: Decimal | str | int | float) -> Self:
        self._native_fee_amount = to_money(amount)
        return self

    def build(self) -> ExecutionReceipt:
        return ExecutionReceipt(
            status=self._status,
            reason=self._reason,
            fill_price=self._fill_price,
            filled_quantity=self._filled_quantity,
            order_id=self._order_id,
            exchange_timestamp=self._exchange_timestamp,
            fees_usd=self._fees_usd,
            fee_asset=self._fee_asset,
            native_fee_amount=self._native_fee_amount,
        )


class DecisionOutcomeBuilder:
    """Fluent builder for DecisionOutcome with preset helpers."""

    def __init__(self) -> None:
        self._pair = "BTC/USDC"
        self._observed_at = datetime.now(UTC)
        self._authority = DecisionAuthority.DETERMINISTIC_ONLY
        self._status = DecisionStatus.HOLD
        self._market_state = MarketState.RANGING
        self._action = TradeAction.HOLD
        self._strategy_name = "trend_follow"
        self._strategy_variant = "baseline"
        self._reason_code = "strategy_hold"
        self._confidence = 0.8
        self._entry_price = Decimal(0)
        self._stop_price = Decimal(0)
        self._take_profit_price = Decimal(0)
        self._quantity = Decimal(0)
        self._notional_usd = Decimal(0)
        self._config_hash = "test-config-hash"
        self._order_id = ""
        self._note = ""

    def with_pair(self, pair: str) -> Self:
        self._pair = pair
        return self

    def executed(self, order_id: str = "order-1") -> Self:
        """Preset: executed decision."""
        self._status = DecisionStatus.EXECUTED
        self._action = TradeAction.BUY
        self._reason_code = "execution_filled"
        self._entry_price = Decimal("100")
        self._stop_price = Decimal("95")
        self._take_profit_price = Decimal("110")
        self._quantity = Decimal("0.1")
        self._notional_usd = Decimal("10")
        self._order_id = order_id
        return self

    def hold(self) -> Self:
        """Preset: hold decision."""
        self._status = DecisionStatus.HOLD
        self._reason_code = "strategy_hold"
        return self

    def vetoed(self) -> Self:
        """Preset: vetoed decision."""
        self._status = DecisionStatus.VETOED
        self._reason_code = "ai_veto"
        self._action = TradeAction.BUY
        return self

    def blocked_risk(self, reason: str = "risk_daily_loss") -> Self:
        """Preset: blocked by risk."""
        self._status = DecisionStatus.BLOCKED_RISK
        self._reason_code = reason
        return self

    def build(self) -> DecisionOutcome:
        return DecisionOutcome(
            pair=self._pair,
            target_id=self._pair,
            observed_at=self._observed_at,
            authority=self._authority,
            status=self._status,
            market_state=self._market_state,
            action=self._action,
            strategy_name=self._strategy_name,
            strategy_variant=self._strategy_variant,
            reason_code=self._reason_code,
            confidence=self._confidence,
            entry_price=self._entry_price,
            stop_price=self._stop_price,
            take_profit_price=self._take_profit_price,
            quantity=self._quantity,
            notional_usd=self._notional_usd,
            config_hash=self._config_hash,
            order_id=self._order_id,
            note=self._note,
        )


class OutcomeInputsBuilder:
    """Fluent builder for OutcomeInputs."""

    def __init__(self, size: int = 2) -> None:
        self._size = size
        self._pairs = ("BTC/USDC", "ETH/USDC")[:size]
        self._actions: list[int] = [TradeAction.HOLD.value] * size
        self._valid_mask: list[bool] = [False] * size
        self._approved_mask: list[bool] = [True] * size
        self._allowed_mask: list[bool] = [True] * size
        self._active_mask: list[bool] = [False] * size
        self._receipts: tuple[ExecutionReceipt, ...] | None = None
        self._authority = DecisionAuthority.DETERMINISTIC_ONLY

    def with_actions(self, actions: list[int]) -> Self:
        self._actions = actions
        return self

    def with_valid_mask(self, mask: list[bool]) -> Self:
        self._valid_mask = mask
        return self

    def with_approved_mask(self, mask: list[bool]) -> Self:
        self._approved_mask = mask
        return self

    def with_allowed_mask(self, mask: list[bool]) -> Self:
        self._allowed_mask = mask
        return self

    def with_active_mask(self, mask: list[bool]) -> Self:
        self._active_mask = mask
        return self

    def with_receipts(self, receipts: tuple[ExecutionReceipt, ...]) -> Self:
        self._receipts = receipts
        return self

    def build(self) -> OutcomeInputs:
        size = self._size
        now = datetime.now(UTC)
        indicators = make_indicator_matrix(size)

        context = BatchDecisionContext(
            market=BatchMarketSnapshot(
                pairs=self._pairs,
                observed_at=now,
                price=np.full(size, 100.0, dtype=np.float64),
                indicators=indicators,
            ),
            portfolio=BatchPortfolioSnapshot(
                equity_usd=np.full(size, 1000.0, dtype=np.float64),
                day_start_equity_usd=np.full(size, 1000.0, dtype=np.float64),
                open_positions_total=np.zeros(size, dtype=np.int64),
                has_open_position=np.zeros(size, dtype=np.bool_),
                unrealized_pnl_usd=np.zeros(size, dtype=np.float64),
            ),
        )

        regime = BatchRegimeProfile(
            coarse_state=np.full(size, MarketState.RANGING.value, dtype=np.int64),
            confidence=np.full(size, 0.8, dtype=np.float64),
            valid_mask=np.ones(size, dtype=np.bool_),
        )

        candidate = BatchTradeCandidate(
            action=np.array(self._actions, dtype=np.int64),
            entry_price=np.full(size, 100.0, dtype=np.float64),
            stop_price=np.full(size, 99.0, dtype=np.float64),
            take_profit_price=np.full(size, 102.0, dtype=np.float64),
            strategy_name=("trend_follow",) * size,
            strategy_variant=("baseline",) * size,
            reason_codes=("strategy_signal",) * size,
            valid_mask=np.array(self._valid_mask, dtype=np.bool_),
        )

        veto = BatchVetoDecision(
            approved_mask=np.array(self._approved_mask, dtype=np.bool_),
            reason_codes=("veto_approved",) * size,
        )

        risk = BatchRiskAssessment(
            allowed_mask=np.array(self._allowed_mask, dtype=np.bool_),
            reason_codes=("risk_ok",) * size,
            risk_score=np.full(size, 1.0, dtype=np.float64),
        )

        intent = BatchExecutionIntent(
            pairs=self._pairs,
            action=np.array(self._actions, dtype=np.int64),
            quantity=np.full(size, 0.1, dtype=np.float64),
            notional_usd=np.full(size, 10.0, dtype=np.float64),
            entry_price=np.full(size, 100.0, dtype=np.float64),
            stop_price=np.full(size, 99.0, dtype=np.float64),
            take_profit_price=np.full(size, 102.0, dtype=np.float64),
            strategy_name=("trend_follow",) * size,
            strategy_variant=("baseline",) * size,
            active_mask=np.array(self._active_mask, dtype=np.bool_),
        )

        receipts = self._receipts
        if receipts is None:
            receipts = tuple(
                ExecutionReceipt(
                    status=ExecutionStatus.FILLED if self._active_mask[i] else ExecutionStatus.SKIPPED,
                    reason="filled" if self._active_mask[i] else "inactive",
                    fill_price=to_price(100.0) if self._active_mask[i] else None,
                    filled_quantity=to_quantity(0.1) if self._active_mask[i] else Decimal(0),
                    exchange_timestamp=now if self._active_mask[i] else None,
                )
                for i in range(size)
            )

        return OutcomeInputs(
            context=context,
            regime=regime,
            candidate=candidate,
            veto=veto,
            risk=risk,
            intent=intent,
            receipts=receipts,
            authority=self._authority,
            config_hash="test-config-hash",
            target_ids=self._pairs,
        )


class TradeCandidateBuilder:
    """Fluent builder for BatchTradeCandidate."""

    def __init__(self, size: int = 1) -> None:
        self._actions = np.ones(size, dtype=np.int64)
        self._entry_price = np.full(size, 100.0, dtype=np.float64)
        self._stop_price = np.full(size, 95.0, dtype=np.float64)
        self._tp_price = np.full(size, 110.0, dtype=np.float64)
        self._strategy_name = ("trend",) * size
        self._strategy_variant = ("baseline",) * size
        self._reason_codes = ("signal",) * size
        self._valid_mask = np.ones(size, dtype=np.bool_)

    def with_actions(self, actions: list[int]) -> Self:
        self._actions = np.array(actions, dtype=np.int64)
        return self

    def with_valid_mask(self, valid: list[bool]) -> Self:
        self._valid_mask = np.array(valid, dtype=np.bool_)
        return self

    def with_entry_prices(self, prices: list[float]) -> Self:
        self._entry_price = np.array(prices, dtype=np.float64)
        return self

    def with_stop_prices(self, prices: list[float]) -> Self:
        self._stop_price = np.array(prices, dtype=np.float64)
        return self

    def with_tp_prices(self, prices: list[float]) -> Self:
        self._tp_price = np.array(prices, dtype=np.float64)
        return self

    def build(self) -> BatchTradeCandidate:
        return BatchTradeCandidate(
            action=self._actions,
            entry_price=self._entry_price,
            stop_price=self._stop_price,
            take_profit_price=self._tp_price,
            strategy_name=self._strategy_name,
            strategy_variant=self._strategy_variant,
            reason_codes=self._reason_codes,
            valid_mask=self._valid_mask,
        )


class RegimeProfileBuilder:
    """Fluent builder for BatchRegimeProfile."""

    def __init__(self, size: int = 1) -> None:
        self._states = np.ones(size, dtype=np.int64)  # TRENDING_UP=1
        self._confidences = np.full(size, 0.8, dtype=np.float64)
        self._valid_mask = np.ones(size, dtype=np.bool_)

    def with_states(self, states: list[int]) -> Self:
        self._states = np.array(states, dtype=np.int64)
        return self

    def with_confidences(self, confidences: list[float]) -> Self:
        self._confidences = np.array(confidences, dtype=np.float64)
        return self

    def with_valid_mask(self, valid: list[bool]) -> Self:
        self._valid_mask = np.array(valid, dtype=np.bool_)
        return self

    def build(self) -> BatchRegimeProfile:
        return BatchRegimeProfile(
            coarse_state=self._states,
            confidence=self._confidences,
            valid_mask=self._valid_mask,
        )
