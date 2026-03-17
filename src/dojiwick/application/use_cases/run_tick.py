"""Batch-first live tick orchestration service."""

import asyncio
import logging
from dataclasses import dataclass, field, replace as dc_replace
from datetime import datetime
from decimal import Decimal

import numpy as np

from dojiwick.application.models.pipeline_settings import PipelineSettings
from dojiwick.application.orchestration.decision_pipeline import run_decision_pipeline
from dojiwick.application.orchestration.outcome_assembler import OutcomeInputs, build_outcomes
from dojiwick.application.orchestration.regime_hysteresis import RegimeHysteresis
from dojiwick.application.orchestration.target_resolver import ResolvedTargets, resolve_targets
from dojiwick.application.policies.adaptive.service import AdaptiveService
from dojiwick.application.policies.risk.engine import RiskPolicyEngine
from dojiwick.application.registry.strategy_registry import StrategyRegistry
from dojiwick.domain.contracts.gateways.account_state import AccountStatePort
from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.context_provider import ContextProviderPort
from dojiwick.domain.contracts.gateways.execution import ExecutionGatewayPort
from dojiwick.domain.contracts.gateways.execution_planner import ExecutionPlannerPort
from dojiwick.domain.contracts.gateways.metrics import MetricsSinkPort
from dojiwick.domain.contracts.gateways.pending_order_provider import PendingOrderProviderPort
from dojiwick.domain.contracts.gateways.unit_of_work import UnitOfWorkPort
from dojiwick.domain.contracts.policies.regime_classifier import AIRegimeClassifierPort
from dojiwick.domain.contracts.policies.veto import VetoServicePort
from dojiwick.domain.contracts.repositories.bot_state import BotStateRepositoryPort
from dojiwick.domain.contracts.repositories.outcome import OutcomeRepositoryPort
from dojiwick.domain.contracts.repositories.regime import RegimeRepositoryPort
from dojiwick.domain.contracts.repositories.decision_trace import DecisionTraceRepositoryPort
from dojiwick.domain.contracts.repositories.tick import TickRepositoryPort
from dojiwick.application.services.order_ledger import OrderLedgerService
from dojiwick.application.services.position_tracker import PositionTracker
from dojiwick.domain.enums import ExecutionStatus, MissingBarPolicy, ReconciliationHealth, TickStatus
from dojiwick.domain.errors import CircuitBreakerTrippedError, DataQualityError, PostExecutionPersistenceError
from dojiwick.domain.timebase import assert_timebase_valid
from dojiwick.domain.models.entities.bot_state import BotState
from dojiwick.domain.hashing import compute_inputs_hash, compute_intent_hash, compute_ops_hash, compute_tick_id
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchExecutionIntent,
)
from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
from dojiwick.domain.models.value_objects.execution_plan import ExecutionPlan, LegDelta
from dojiwick.domain.models.value_objects.outcome_models import DecisionOutcome, ExecutionReceipt
from dojiwick.domain.models.value_objects.decision_trace import DecisionTrace
from dojiwick.domain.models.value_objects.tick_record import TickRecord

log = logging.getLogger(__name__)

_MAX_ERROR_MESSAGE_LEN = 500


def _skipped_receipts(size: int) -> tuple[ExecutionReceipt, ...]:
    """Create SKIPPED receipts for all batch rows."""
    return tuple(ExecutionReceipt(status=ExecutionStatus.SKIPPED, reason="no_plan_deltas") for _ in range(size))


def align_plan_receipts(
    batch_size: int,
    resolved: ResolvedTargets,
    plan: ExecutionPlan,
    plan_receipts: tuple[ExecutionReceipt, ...],
) -> tuple[ExecutionReceipt, ...]:
    """Map per-delta plan receipts back to per-target batch receipts."""
    result: list[ExecutionReceipt] = [
        ExecutionReceipt(status=ExecutionStatus.SKIPPED, reason="inactive") for _ in range(batch_size)
    ]

    receipts_by_target: dict[int, list[ExecutionReceipt]] = {}
    for delta_index, delta in enumerate(plan.deltas):
        receipt = (
            plan_receipts[delta_index]
            if delta_index < len(plan_receipts)
            else ExecutionReceipt(status=ExecutionStatus.ERROR, reason="missing_plan_receipt")
        )
        receipts_by_target.setdefault(delta.target_index, []).append(receipt)

    for target_idx, batch_idx in enumerate(resolved.batch_indices):
        target_receipts = tuple(receipts_by_target.get(target_idx, ()))
        if not target_receipts:
            result[batch_idx] = ExecutionReceipt(status=ExecutionStatus.SKIPPED, reason="no_target_receipts")
            continue
        result[batch_idx] = _aggregate_target_receipts(target_receipts)

    return tuple(result)


def _aggregate_target_receipts(receipts: tuple[ExecutionReceipt, ...]) -> ExecutionReceipt:
    """Reduce multiple delta receipts to a single target-level receipt."""
    for receipt in receipts:
        if receipt.status is ExecutionStatus.ERROR:
            return receipt

    for receipt in receipts:
        if receipt.status is ExecutionStatus.REJECTED:
            return receipt

    filled = tuple(receipt for receipt in receipts if receipt.status is ExecutionStatus.FILLED)
    if filled:
        # For flip plans (close + open), the last filled receipt reflects final leg intent.
        return filled[-1]

    return ExecutionReceipt(status=ExecutionStatus.SKIPPED, reason="all_plan_deltas_skipped")


@dataclass(slots=True)
class TickService:
    """Orchestrates one batch decision cycle.

    Signal pipeline order: context -> regime -> variant selection ->
    strategy -> AI veto -> risk -> sizing -> target planner -> execution -> persistence.
    """

    settings: PipelineSettings
    strategy_registry: StrategyRegistry
    risk_engine: RiskPolicyEngine
    clock: ClockPort
    context_provider: ContextProviderPort
    execution_gateway: ExecutionGatewayPort
    execution_planner: ExecutionPlannerPort
    account_state_provider: AccountStatePort
    outcome_repository: OutcomeRepositoryPort
    tick_repository: TickRepositoryPort
    regime_repository: RegimeRepositoryPort | None = None
    veto_service: VetoServicePort | None = None
    regime_classifier: AIRegimeClassifierPort | None = None
    adaptive_service: AdaptiveService | None = None
    bot_state_repository: BotStateRepositoryPort | None = None
    metrics: MetricsSinkPort | None = None
    unit_of_work: UnitOfWorkPort | None = None
    decision_trace_repository: DecisionTraceRepositoryPort | None = None
    order_ledger: OrderLedgerService | None = None
    position_tracker: PositionTracker | None = None
    pending_order_provider: PendingOrderProviderPort | None = None
    hysteresis: RegimeHysteresis = field(default_factory=RegimeHysteresis)
    shutdown_event: asyncio.Event | None = None
    config_hash: str = ""
    target_ids: tuple[str, ...] = ()
    instrument_map: dict[str, InstrumentId] | None = None

    def __post_init__(self) -> None:
        if not self.config_hash:
            import hashlib

            self.config_hash = hashlib.sha256(b"unconfigured").hexdigest()
        if not self.target_ids:
            raise ValueError("TickService requires non-empty target_ids")
        if len(self.target_ids) != len(self.settings.trading.active_pairs):
            raise ValueError(
                f"target_ids length ({len(self.target_ids)}) must match "
                f"active_pairs length ({len(self.settings.trading.active_pairs)})"
            )
        if self.instrument_map is None:
            raise ValueError("TickService requires instrument_map")
        for pair in self.settings.trading.active_pairs:
            if pair not in self.instrument_map:
                raise ValueError(f"instrument_map missing pair '{pair}'")

    async def run_tick(
        self,
        pairs: tuple[str, ...] | None = None,
        at: datetime | None = None,
    ) -> tuple[DecisionOutcome, ...]:
        """Run one batch tick and persist outcomes.

        Pipeline: context -> regime -> variants -> strategy -> veto -> risk -> sizing
        -> planner -> execution -> outcomes -> persistence.
        """
        if self.settings.flags.global_halt:
            raise CircuitBreakerTrippedError("global_halt flag is active")

        start_ns = self.clock.monotonic_ns()

        target_pairs = pairs if pairs is not None else self.settings.trading.active_pairs
        observed_at = at if at is not None else self.clock.now_utc()

        state = await self.bot_state_repository.get_state() if self.bot_state_repository else None
        await self._check_circuit_breaker(state, observed_at)

        # 1. Fetch context
        context = await self.context_provider.fetch_context_batch(target_pairs, observed_at)

        # 1b. Timebase validation -- reject stale bar data
        if context.market.asof_timestamp is not None:
            try:
                assert_timebase_valid(
                    observed_at,
                    context.market.asof_timestamp,
                    self.settings.system.tick_interval_sec,
                )
            except DataQualityError:
                policy = self.settings.system.missing_bar_policy
                match policy:
                    case MissingBarPolicy.SKIP:
                        log.warning("stale bar data, skipping tick (policy=%s)", policy)
                        skip_tick_id = compute_tick_id(self.config_hash, observed_at, target_pairs)
                        await self.tick_repository.try_insert(
                            TickRecord(
                                tick_id=skip_tick_id,
                                tick_time=observed_at,
                                config_hash=self.config_hash,
                                inputs_hash=compute_inputs_hash(context),
                                batch_size=context.size,
                                status=TickStatus.SKIPPED,
                            )
                        )
                        return ()
                    case MissingBarPolicy.ERROR:
                        raise
                    case MissingBarPolicy.LAST_KNOWN:
                        log.warning("stale bar data, continuing with last known (policy=%s)", policy)

        # 2. Compute tick identity and inputs hash
        tick_id = compute_tick_id(self.config_hash, observed_at, target_pairs)
        inputs_hash = compute_inputs_hash(context)

        # 3. Dedup -- skip if tick already completed (atomic insert)
        inserted = await self.tick_repository.try_insert(
            TickRecord(
                tick_id=tick_id,
                tick_time=observed_at,
                config_hash=self.config_hash,
                inputs_hash=inputs_hash,
                batch_size=context.size,
            )
        )
        if not inserted:
            log.info("tick dedup skip tick_id=%s", tick_id)
            return ()

        try:
            outcomes = await self._run_pipeline(target_pairs, observed_at, context, tick_id, start_ns, state)
        except Exception as exc:
            duration_ms = (self.clock.monotonic_ns() - start_ns) // 1_000_000
            try:
                await self.tick_repository.update_status(
                    tick_id,
                    TickStatus.FAILED,
                    duration_ms=duration_ms,
                    error_message=str(exc)[:_MAX_ERROR_MESSAGE_LEN],
                )
            except Exception:
                log.exception("failed to mark tick %s as FAILED", tick_id)
            if self.metrics is not None:
                self.metrics.increment("tick_failure_total")
            raise

        return outcomes

    async def _run_pipeline(
        self,
        target_pairs: tuple[str, ...],
        observed_at: datetime,
        context: BatchDecisionContext,
        tick_id: str,
        start_ns: int,
        state: BotState | None,
    ) -> tuple[DecisionOutcome, ...]:
        """Execute the full pipeline with tick tracking."""
        # 4. Adaptive variant selection (influences real variant path)
        adaptive_variant: str | None = None
        if self.adaptive_service is not None:
            selected = await self.adaptive_service.select_variant()
            if isinstance(selected, str) and selected != "baseline":
                adaptive_variant = selected

        # 5-9. Shared decision pipeline (regime -> variants -> strategy -> veto -> risk -> sizing)
        veto = None if self.settings.flags.disable_llm else self.veto_service
        classifier = None if self.settings.flags.disable_llm else self.regime_classifier
        pipeline_start = self.clock.monotonic_ns()
        pipeline = await run_decision_pipeline(
            context=context,
            settings=self.settings,
            strategy_registry=self.strategy_registry,
            risk_engine=self.risk_engine,
            hysteresis=self.hysteresis,
            veto_service=veto,
            regime_classifier=classifier,
            adaptive_variant=adaptive_variant,
        )
        pipeline_duration_us = (self.clock.monotonic_ns() - pipeline_start) // 1_000

        # 10. Filter frozen symbols from execution
        frozen = await self._get_frozen_symbols(state)
        if frozen:
            new_mask = pipeline.intents.active_mask.copy()
            for i, pair in enumerate(pipeline.intents.pairs):
                if pair in frozen:
                    new_mask[i] = False
            pipeline = dc_replace(pipeline, intents=dc_replace(pipeline.intents, active_mask=new_mask))
            log.info("frozen symbols masked: %s", sorted(frozen))

        # 11. Execution via planner (tracked for ops hashing)
        exec_start = self.clock.monotonic_ns()
        receipts, plan, plan_receipts = await self._execute_via_planner_tracked(pipeline.intents, tick_id=tick_id)
        exec_duration_us = (self.clock.monotonic_ns() - exec_start) // 1_000

        # 12. Compute stage hashes (pure -- no I/O dependency on ledger)
        intent_hash = compute_intent_hash(pipeline.intents)
        ops_hash = compute_ops_hash(plan)

        # 13. Outcome assembly (pure -- no I/O dependency on ledger)
        confidence_raw = tuple(pipeline.confidence_raw)
        effective_target_ids = self.target_ids
        outcomes = build_outcomes(
            OutcomeInputs(
                context=context,
                regime=pipeline.regimes,
                candidate=pipeline.candidates,
                veto=pipeline.veto,
                risk=pipeline.risk,
                intent=pipeline.intents,
                receipts=receipts,
                authority=pipeline.authority,
                config_hash=self.config_hash,
                target_ids=effective_target_ids,
                tick_id=tick_id,
                confidence_raw=confidence_raw,
            )
        )

        # 14. Atomic post-execution persistence
        duration_ms = (self.clock.monotonic_ns() - start_ns) // 1_000_000
        persist_start = self.clock.monotonic_ns()

        async def _do_persist() -> None:
            if plan is not None and plan_receipts:
                if self.order_ledger is not None:
                    await self.order_ledger.record_execution(plan, plan_receipts, tick_id=tick_id)
                if self.position_tracker is not None:
                    await self.position_tracker.apply_fills(plan, plan_receipts)
            await self.outcome_repository.append_outcomes(
                outcomes,
                venue=self.settings.exchange.venue,
                product=self.settings.exchange.product,
            )
            if self.regime_repository is not None:
                await self.regime_repository.insert_batch(
                    target_pairs,
                    observed_at,
                    pipeline.regimes,
                    target_ids=effective_target_ids,
                    venue=str(self.settings.exchange.venue),
                    product=str(self.settings.exchange.product),
                )
            await self.tick_repository.update_status(
                tick_id,
                TickStatus.COMPLETED,
                intent_hash=intent_hash,
                ops_hash=ops_hash,
                duration_ms=duration_ms,
            )

        try:
            if self.unit_of_work is not None:
                async with self.unit_of_work.transaction():
                    await _do_persist()
            else:
                await _do_persist()
        except Exception as exc:
            log.critical("post-execution persistence failed: %s", exc)
            raise PostExecutionPersistenceError(str(exc)) from exc
        persist_duration_us = (self.clock.monotonic_ns() - persist_start) // 1_000

        # 15. Decision traces (fire-and-forget -- must not crash tick)
        if self.decision_trace_repository is not None:
            vetoed_count = int(np.sum(~pipeline.veto.approved_mask & pipeline.candidates.valid_mask))
            traces = (
                DecisionTrace(
                    tick_id=tick_id,
                    step_name="pipeline",
                    step_seq=0,
                    artifacts={
                        "batch_size": context.size,
                        "authority": pipeline.authority.value,
                        "vetoed": vetoed_count,
                    },
                    duration_us=pipeline_duration_us,
                ),
                DecisionTrace(
                    tick_id=tick_id,
                    step_name="execution",
                    step_seq=1,
                    artifacts={"deltas": len(plan.deltas) if plan else 0},
                    duration_us=exec_duration_us,
                ),
                DecisionTrace(
                    tick_id=tick_id,
                    step_name="persistence",
                    step_seq=2,
                    artifacts={"outcomes": len(outcomes)},
                    duration_us=persist_duration_us,
                ),
            )
            try:
                await self.decision_trace_repository.insert_batch(traces)
            except Exception:
                log.warning("failed to persist decision traces for tick_id=%s", tick_id, exc_info=True)

        # 16. Adaptive outcome recording
        if self.adaptive_service is not None:
            await self.adaptive_service.record_outcome()

        if self.metrics is not None:
            duration_seconds = (self.clock.monotonic_ns() - start_ns) / 1_000_000_000
            self.metrics.increment("tick_total")
            self.metrics.observe("tick_duration_seconds", duration_seconds)
            self.metrics.gauge("tick_batch_size", float(context.size))
            vetoed_count = int(np.sum(~pipeline.veto.approved_mask & pipeline.candidates.valid_mask))
            if vetoed_count > 0:
                self.metrics.increment("ai_veto_block_total", vetoed_count)
            risk_blocked = int(np.sum(~pipeline.risk.allowed_mask & pipeline.candidates.valid_mask))
            if risk_blocked > 0:
                self.metrics.increment("risk_block_total", risk_blocked)

        await self._update_bot_state(state, observed_at)

        log.info("tick completed tick_id=%s pairs=%d config_hash=%s", tick_id, len(target_pairs), self.config_hash)
        return outcomes

    async def _execute_via_planner_tracked(
        self,
        intents: BatchExecutionIntent,
        *,
        tick_id: str = "",
    ) -> tuple[tuple[ExecutionReceipt, ...], ExecutionPlan | None, tuple[ExecutionReceipt, ...]]:
        """Target-position planner execution path, returning plan for ops hashing."""
        assert self.instrument_map is not None  # validated in __post_init__
        resolved = resolve_targets(
            intents,
            account=self.settings.universe.account,
            position_mode=self.settings.exchange.position_mode,
            venue=self.settings.exchange.venue,
            product=self.settings.exchange.product,
            quote_asset=self.settings.universe.quote_asset,
            settle_asset=self.settings.universe.settle_asset,
            pair_separator=self.settings.universe.pair_separator,
            instrument_map=self.instrument_map,
        )
        if not resolved.targets:
            return _skipped_receipts(len(intents.pairs)), None, ()

        snapshot = await self.account_state_provider.get_account_snapshot(self.settings.universe.account)
        plan = await self.execution_planner.plan(snapshot, resolved.targets)

        if self.pending_order_provider is not None and not plan.is_empty:
            plan = await self._apply_pending_order_guard(plan)

        if plan.is_empty:
            return _skipped_receipts(len(intents.pairs)), plan, ()

        plan_receipts = await self.execution_gateway.execute_plan(plan, tick_id=tick_id)
        return align_plan_receipts(len(intents.pairs), resolved, plan, plan_receipts), plan, plan_receipts

    async def _apply_pending_order_guard(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Deduct pending (in-flight) quantities from plan deltas."""
        assert self.pending_order_provider is not None
        pending = await self.pending_order_provider.get_pending_quantities(plan.account)
        if not pending:
            return plan

        zero = Decimal(0)
        adjusted: list[LegDelta] = []
        for delta in plan.deltas:
            if delta.reduce_only:
                adjusted.append(delta)
                continue

            key = (delta.instrument_id.symbol, delta.position_side)
            pending_qty = pending.get(key, zero)
            if pending_qty <= zero:
                adjusted.append(delta)
                continue

            remaining = delta.quantity - pending_qty
            if remaining <= zero:
                log.info(
                    "pending order guard: skipping delta %s/%s qty=%s (pending=%s)",
                    delta.instrument_id.symbol,
                    delta.position_side,
                    delta.quantity,
                    pending_qty,
                )
                continue

            log.info(
                "pending order guard: reducing delta %s/%s qty=%s->%s (pending=%s)",
                delta.instrument_id.symbol,
                delta.position_side,
                delta.quantity,
                remaining,
                pending_qty,
            )
            adjusted.append(dc_replace(delta, quantity=remaining))

        return dc_replace(plan, deltas=tuple(adjusted))

    async def _check_circuit_breaker(self, state: BotState | None, observed_at: datetime) -> None:
        """Raise if circuit breaker is active and has not expired."""

        if state is None:
            return
        if state.recon_health is ReconciliationHealth.HALT:
            raise CircuitBreakerTrippedError("reconciliation health is HALT")
        if not state.circuit_breaker_active:
            return
        if state.circuit_breaker_until is not None and observed_at >= state.circuit_breaker_until:
            state.circuit_breaker_active = False
            assert self.bot_state_repository is not None
            await self.bot_state_repository.update_state(state)
            log.info("circuit breaker expired, resuming")
            return
        raise CircuitBreakerTrippedError("circuit breaker is active")

    async def _update_bot_state(self, state: BotState | None, observed_at: datetime) -> None:
        """Update last_tick_at after a successful tick."""

        if self.bot_state_repository is None or state is None:
            return
        state.last_tick_at = observed_at
        state.consecutive_errors = 0
        await self.bot_state_repository.update_state(state)

    async def _get_frozen_symbols(self, state: BotState | None) -> tuple[str, ...]:
        """Return symbols frozen by reconciliation health state."""

        if state is None:
            return ()
        if state.recon_health in (
            ReconciliationHealth.DEGRADED,
            ReconciliationHealth.UNCERTAIN,
            ReconciliationHealth.HALT,
        ):
            return state.recon_frozen_symbols
        return ()
