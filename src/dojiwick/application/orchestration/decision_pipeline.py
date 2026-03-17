"""Shared decision pipeline for live tick and backtest paths.

Extracts the deterministic regime -> variant -> strategy -> veto -> risk -> sizing
steps into a reusable function.  Both ``TickService`` and ``BacktestService``
call ``run_decision_pipeline`` so that decision logic is defined once.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TypeVar

import numpy as np

from dojiwick.application.models.pipeline_settings import PipelineSettings
from dojiwick.application.orchestration.regime_hysteresis import RegimeHysteresis
from dojiwick.application.policies.risk.engine import RiskPolicyEngine
from dojiwick.application.registry.strategy_registry import StrategyRegistry
from dojiwick.compute.kernels.regime.classify import classify_regime_batch
from dojiwick.compute.kernels.regime.ensemble import combine_regime_ensemble
from dojiwick.compute.kernels.sizing.fixed_fraction import size_intents
from dojiwick.domain.contracts.policies.regime_classifier import AIRegimeClassifierPort
from dojiwick.domain.contracts.policies.veto import VetoServicePort
from dojiwick.domain.enums import DecisionAuthority, MarketState, TradeAction, safe_market_state
from dojiwick.domain.indicator_schema import INDICATOR_INDEX
from dojiwick.domain.models.value_objects.params import RiskParams, StrategyParams
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchExecutionIntent,
    BatchRegimeProfile,
    BatchRiskAssessment,
    BatchTradeCandidate,
    BatchVetoDecision,
)
from dojiwick.domain.type_aliases import FloatVector
from dojiwick.domain.reason_codes import AI_VETO_ERROR, AI_VETO_NON_CONTRIBUTORY_CODES

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PipelineResult:
    """Immutable bundle returned by ``run_decision_pipeline``."""

    regimes: BatchRegimeProfile
    candidates: BatchTradeCandidate
    veto: BatchVetoDecision
    risk: BatchRiskAssessment
    intents: BatchExecutionIntent
    variants: tuple[str, ...]
    authority: DecisionAuthority
    confidence_raw: FloatVector
    per_pair_params: tuple[StrategyParams, ...] = ()


# Internal helpers


_T = TypeVar("_T")


_regime_state = safe_market_state


def _cached_resolve(
    pairs: tuple[str, ...],
    stable_state: np.ndarray,
    valid_mask: np.ndarray,
    resolve_fn: Callable[[str, MarketState | None], _T],
    cache: dict[tuple[str, int | None], _T] | None = None,
) -> tuple[_T, ...]:
    """Resolve per-pair values with optional caching."""
    resolved: list[_T] = []
    for i, pair in enumerate(pairs):
        regime = _regime_state(int(stable_state[i])) if bool(valid_mask[i]) else None
        regime_key = regime.value if regime is not None else None
        cache_key = (pair, regime_key)
        if cache is not None and cache_key in cache:
            resolved.append(cache[cache_key])
        else:
            value = resolve_fn(pair, regime)
            if cache is not None:
                cache[cache_key] = value
            resolved.append(value)
    return tuple(resolved)


def _resolve_variants(
    pairs: tuple[str, ...],
    stable_state: np.ndarray,
    valid_mask: np.ndarray,
    settings: PipelineSettings,
    adaptive_variant: str | None,
    scope_cache: dict[tuple[str, int | None], StrategyParams] | None = None,
) -> tuple[tuple[str, ...], tuple[StrategyParams, ...]]:
    """Resolve per-pair variants and strategy params using scope overrides.

    Priority: adaptive selection > scope override > default.
    Returns (variants, per_pair_params).
    """
    per_pair_params = _cached_resolve(
        pairs,
        stable_state,
        valid_mask,
        lambda pair, regime: settings.strategy_scope.resolve(pair, regime, settings.strategy),
        cache=scope_cache,
    )
    variants = tuple(adaptive_variant if adaptive_variant is not None else p.default_variant for p in per_pair_params)
    return variants, per_pair_params


def _resolve_risk_params(
    pairs: tuple[str, ...],
    stable_state: np.ndarray,
    valid_mask: np.ndarray,
    settings: PipelineSettings,
    scope_cache: dict[tuple[str, int | None], RiskParams] | None = None,
) -> tuple[RiskParams, ...]:
    """Resolve per-pair risk params using risk scope overrides."""
    return _cached_resolve(
        pairs,
        stable_state,
        valid_mask,
        lambda pair, regime: settings.risk_scope.resolve(pair, regime, settings.risk.params),
        cache=scope_cache,
    )


def _compute_stop_tp_scalar(
    entry: float,
    atr: float,
    direction: int,
    stop_atr_mult: float,
    rr_ratio: float,
    min_stop_pct: float,
) -> tuple[float, float]:
    """Compute ATR-based stop and take-profit for a single pair.

    NOTE: keep formula in sync with the vectorized version in
    ``strategy_registry.py:StrategyRegistry.propose_candidates`` (section 4).
    """
    raw = atr * stop_atr_mult
    min_d = entry * (min_stop_pct / 100.0)
    dist = max(raw, min_d)
    return entry - direction * dist, entry + direction * dist * rr_ratio


def _resolve_exit_overrides(
    pairs: tuple[str, ...],
    stable_state: np.ndarray,
    valid_mask: np.ndarray,
    candidates: BatchTradeCandidate,
    settings: PipelineSettings,
    phase1_params: tuple[StrategyParams, ...],
    context: BatchDecisionContext,
    *,
    has_strategy_rules: bool = True,
    phase2_cache: dict[tuple[str, int | None, str | None], StrategyParams] | None = None,
) -> tuple[BatchTradeCandidate, tuple[StrategyParams, ...]]:
    """Phase 2: re-resolve strategy params with the winning strategy name for exit tuning.

    Strategy-scoped rules only match when strategy name is provided. Phase 1 runs
    without strategy (for signal thresholds); Phase 2 runs with the winning strategy
    name to pick up exit-specific overrides (stop, TP, trailing, breakeven, max_hold_bars).
    """
    if not has_strategy_rules:
        return candidates, phase1_params

    new_stop: np.ndarray | None = None
    new_tp: np.ndarray | None = None
    updated_params = list(phase1_params)

    for i, pair in enumerate(pairs):
        if not bool(valid_mask[i]):
            continue
        act = int(candidates.action[i])
        if act == TradeAction.HOLD.value:
            continue

        strategy_name = candidates.strategy_name[i]
        regime = _regime_state(int(stable_state[i]))
        regime_key = regime.value if regime is not None else None

        cache_key = (pair, regime_key, strategy_name)
        if phase2_cache is not None and cache_key in phase2_cache:
            phase2 = phase2_cache[cache_key]
        else:
            phase2 = settings.strategy_scope.resolve(pair, regime, settings.strategy, strategy=strategy_name)
            if phase2_cache is not None:
                phase2_cache[cache_key] = phase2
        if phase2 is phase1_params[i]:
            continue

        # Recompute stop/TP if exit params changed
        p1 = phase1_params[i]
        if (
            phase2.stop_atr_mult != p1.stop_atr_mult
            or phase2.rr_ratio != p1.rr_ratio
            or phase2.min_stop_distance_pct != p1.min_stop_distance_pct
        ):
            if new_stop is None:
                new_stop = candidates.stop_price.copy()
                new_tp = candidates.take_profit_price.copy()

            entry = float(candidates.entry_price[i])
            atr = float(context.market.indicators[i, INDICATOR_INDEX["atr"]])
            direction = 1 if act == TradeAction.BUY.value else -1

            assert new_tp is not None
            new_stop[i], new_tp[i] = _compute_stop_tp_scalar(
                entry,
                atr,
                direction,
                phase2.stop_atr_mult,
                phase2.rr_ratio,
                phase2.min_stop_distance_pct,
            )

        updated_params[i] = phase2

    if new_stop is not None:
        candidates = replace(candidates, stop_price=new_stop, take_profit_price=new_tp)

    return candidates, tuple(updated_params)


def _apply_veto(candidates: BatchTradeCandidate, veto: BatchVetoDecision) -> BatchTradeCandidate:
    """Return candidates with valid_mask narrowed by veto approval."""
    return BatchTradeCandidate(
        action=candidates.action,
        entry_price=candidates.entry_price,
        stop_price=candidates.stop_price,
        take_profit_price=candidates.take_profit_price,
        strategy_name=candidates.strategy_name,
        strategy_variant=candidates.strategy_variant,
        reason_codes=candidates.reason_codes,
        valid_mask=candidates.valid_mask & veto.approved_mask,
    )


async def _evaluate_veto(
    context: BatchDecisionContext,
    candidates: BatchTradeCandidate,
    settings: PipelineSettings,
    veto_service: VetoServicePort | None,
) -> BatchVetoDecision:
    """Evaluate the AI veto filter, returning all-approved when disabled."""
    size = context.size
    if not (settings.ai.enabled and settings.ai.veto_enabled and veto_service is not None):
        return BatchVetoDecision(
            approved_mask=np.ones(size, dtype=np.bool_),
            reason_codes=("veto_not_enabled",) * size,
        )

    try:
        result = await veto_service.evaluate_batch(context, candidates)
        if settings.flags.ai_veto_shadow_mode:
            log.info("veto shadow mode: logged result, returning all-approved")
            return BatchVetoDecision(
                approved_mask=np.ones(size, dtype=np.bool_),
                reason_codes=result.reason_codes,
            )
        return result
    except (OSError, TimeoutError, ConnectionError):  # fmt: skip
        log.exception("veto evaluation failed (transient)")
        if settings.ai.fail_open_on_error:
            return BatchVetoDecision(
                approved_mask=np.ones(size, dtype=np.bool_),
                reason_codes=(AI_VETO_ERROR,) * size,
            )
        return BatchVetoDecision(
            approved_mask=np.zeros(size, dtype=np.bool_),
            reason_codes=(AI_VETO_ERROR,) * size,
        )
    except Exception:
        log.exception("veto evaluation failed (unexpected)")
        return BatchVetoDecision(
            approved_mask=np.zeros(size, dtype=np.bool_),
            reason_codes=(AI_VETO_ERROR,) * size,
        )


async def _evaluate_ai_regime(
    context: BatchDecisionContext,
    regimes: BatchRegimeProfile,
    settings: PipelineSettings,
    regime_classifier: AIRegimeClassifierPort | None,
) -> tuple[BatchRegimeProfile, bool]:
    """Evaluate the AI regime classifier, returning (ensemble_regimes, ai_regime_active).

    When disabled or on error, returns the deterministic regimes unchanged.
    """
    if not (settings.ai.enabled and settings.ai.regime_enabled and regime_classifier is not None):
        return regimes, False

    try:
        ai_regimes = await regime_classifier.classify_batch(context, regimes)
    except (OSError, TimeoutError, ConnectionError):  # fmt: skip
        log.exception("ai regime classification failed (transient)")
        if settings.ai.regime_fail_open_on_error:
            return regimes, False
        raise
    except Exception:
        log.exception("ai regime classification failed (unexpected)")
        if settings.ai.regime_fail_open_on_error:
            return regimes, False
        raise

    ensemble = combine_regime_ensemble(
        deterministic=regimes,
        ai=ai_regimes,
        boost=settings.ai.regime_agreement_boost,
        penalty=settings.ai.regime_disagreement_penalty,
    )
    if settings.flags.ai_regime_shadow_mode:
        log.info("regime shadow mode: logged result, returning deterministic")
        return regimes, False
    return ensemble, True


def _authority(
    settings: PipelineSettings,
    veto_service: VetoServicePort | None,
    veto: BatchVetoDecision,
    *,
    ai_regime_active: bool = False,
) -> DecisionAuthority:
    """Derive the decision authority from AI feature configuration and outcomes."""
    veto_active = (
        settings.ai.enabled
        and settings.ai.veto_enabled
        and veto_service is not None
        and any(code not in AI_VETO_NON_CONTRIBUTORY_CODES for code in veto.reason_codes)
    )

    if ai_regime_active and veto_active:
        return DecisionAuthority.DETERMINISTIC_PLUS_AI_REGIME_AND_VETO
    if ai_regime_active:
        return DecisionAuthority.DETERMINISTIC_PLUS_AI_REGIME
    if veto_active:
        return DecisionAuthority.DETERMINISTIC_PLUS_AI_VETO
    return DecisionAuthority.DETERMINISTIC_ONLY


# Shared pipeline core


@dataclass(slots=True, frozen=True, kw_only=True)
class _CorePipelineResult:
    """Steps 1-6c output: regime -> hysteresis -> halted -> confidence -> variants -> strategy -> exits."""

    regimes: BatchRegimeProfile
    candidates: BatchTradeCandidate
    confidence_raw: FloatVector
    variants: tuple[str, ...]
    per_pair_params: tuple[StrategyParams, ...]


def _run_core_pipeline(
    *,
    context: BatchDecisionContext,
    settings: PipelineSettings,
    strategy_registry: StrategyRegistry,
    hysteresis: RegimeHysteresis | None = None,
    adaptive_variant: str | None = None,
    strategy_scope_cache: dict[tuple[str, int | None], StrategyParams] | None = None,
    has_strategy_rules: bool | None = None,
    phase2_scope_cache: dict[tuple[str, int | None, str | None], StrategyParams] | None = None,
    hysteresis_bars_override: int | None = None,
    hysteresis_eligibility_mask: np.ndarray | None = None,
) -> _CorePipelineResult:
    """Steps 1-6c: regime -> hysteresis -> halted -> confidence -> variants -> strategy -> exits.

    Pure synchronous function shared by both sync and async pipeline paths.
    """
    pairs = context.market.pairs

    # 1. Classify regime
    regimes = classify_regime_batch(context.market, settings.regime.params)

    # 2. Hysteresis
    if hysteresis is not None:
        effective_bars = (
            hysteresis_bars_override if hysteresis_bars_override is not None else settings.regime.hysteresis_bars
        )
        stable_state = hysteresis.apply(
            pairs, regimes.coarse_state, effective_bars, eligibility_mask=hysteresis_eligibility_mask
        )
        regimes = BatchRegimeProfile(
            coarse_state=stable_state,
            confidence=regimes.confidence,
            valid_mask=regimes.valid_mask,
        )

    # 2b. Halted pairs
    if settings.flags.halted_pairs:
        halted = np.isin(pairs, settings.flags.halted_pairs)
        regimes = replace(regimes, valid_mask=regimes.valid_mask & ~halted)
        log.info("halted_pairs: masked %s", settings.flags.halted_pairs)

    # 2c. Min-confidence gate
    if settings.regime.min_confidence > 0:
        low_conf = regimes.confidence < settings.regime.min_confidence
        regimes = replace(regimes, valid_mask=regimes.valid_mask & ~low_conf)

    # 3. Capture pre-AI confidence (no copy needed — array not mutated downstream)
    confidence_raw = regimes.confidence

    # 5. Resolve per-pair variants and params
    variants, per_pair_params = _resolve_variants(
        pairs,
        regimes.coarse_state,
        regimes.valid_mask,
        settings,
        adaptive_variant,
        scope_cache=strategy_scope_cache,
    )

    # 6. Strategy signal generation
    candidates = strategy_registry.propose_candidates(
        context=context,
        regime=regimes,
        settings=settings.strategy,
        variants=variants,
        per_pair_params=per_pair_params,
    )

    # 6b. Phase 2 exit resolution
    _has_rules = has_strategy_rules if has_strategy_rules is not None else settings.strategy_scope.has_strategy_rules
    candidates, per_pair_params = _resolve_exit_overrides(
        pairs,
        regimes.coarse_state,
        regimes.valid_mask,
        candidates,
        settings,
        per_pair_params,
        context,
        has_strategy_rules=_has_rules,
        phase2_cache=phase2_scope_cache,
    )

    # 6c. Exits-only mode
    if settings.flags.exits_only_mode:
        candidates = replace(candidates, valid_mask=candidates.valid_mask & (candidates.action == TradeAction.HOLD))

    return _CorePipelineResult(
        regimes=regimes,
        candidates=candidates,
        confidence_raw=confidence_raw,
        variants=variants,
        per_pair_params=per_pair_params,
    )


def _run_risk_and_sizing(
    *,
    context: BatchDecisionContext,
    settings: PipelineSettings,
    risk_engine: RiskPolicyEngine,
    regimes: BatchRegimeProfile,
    vetoed: BatchTradeCandidate,
    risk_scope_cache: dict[tuple[str, int | None], RiskParams] | None = None,
) -> tuple[BatchRiskAssessment, BatchExecutionIntent]:
    """Steps 8-10: risk scope -> risk assess -> sizing."""
    risk_params = _resolve_risk_params(
        context.market.pairs,
        regimes.coarse_state,
        regimes.valid_mask,
        settings,
        scope_cache=risk_scope_cache,
    )
    risk = risk_engine.assess_risk(context=context, candidate=vetoed, risk_params=risk_params)
    intents = size_intents(context=context, candidate=vetoed, assessment=risk, risk_params=risk_params)
    return risk, intents


# Public API


def run_decision_pipeline_sync(
    *,
    context: BatchDecisionContext,
    settings: PipelineSettings,
    strategy_registry: StrategyRegistry,
    risk_engine: RiskPolicyEngine,
    hysteresis: RegimeHysteresis | None = None,
    adaptive_variant: str | None = None,
    strategy_scope_cache: dict[tuple[str, int | None], StrategyParams] | None = None,
    risk_scope_cache: dict[tuple[str, int | None], RiskParams] | None = None,
    has_strategy_rules: bool | None = None,
    phase2_scope_cache: dict[tuple[str, int | None, str | None], StrategyParams] | None = None,
    hysteresis_bars_override: int | None = None,
    hysteresis_eligibility_mask: np.ndarray | None = None,
) -> PipelineResult:
    """Synchronous fast-path for backtest mode (no AI services).

    Must only be called when AI veto and regime classifier are disabled.
    """
    size = context.size

    core = _run_core_pipeline(
        context=context,
        settings=settings,
        strategy_registry=strategy_registry,
        hysteresis=hysteresis,
        adaptive_variant=adaptive_variant,
        strategy_scope_cache=strategy_scope_cache,
        has_strategy_rules=has_strategy_rules,
        phase2_scope_cache=phase2_scope_cache,
        hysteresis_bars_override=hysteresis_bars_override,
        hysteresis_eligibility_mask=hysteresis_eligibility_mask,
    )

    # No AI veto in sync path -- skip veto allocation entirely (candidates pass through)
    risk, intents = _run_risk_and_sizing(
        context=context,
        settings=settings,
        risk_engine=risk_engine,
        regimes=core.regimes,
        vetoed=core.candidates,
        risk_scope_cache=risk_scope_cache,
    )

    # Lazy veto construction only for PipelineResult (not on hot path)
    veto = BatchVetoDecision(
        approved_mask=np.ones(size, dtype=np.bool_),
        reason_codes=("veto_not_enabled",) * size,
    )

    return PipelineResult(
        regimes=core.regimes,
        candidates=core.candidates,
        veto=veto,
        risk=risk,
        intents=intents,
        variants=core.variants,
        authority=DecisionAuthority.DETERMINISTIC_ONLY,
        confidence_raw=core.confidence_raw,
        per_pair_params=core.per_pair_params,
    )


async def run_decision_pipeline(
    *,
    context: BatchDecisionContext,
    settings: PipelineSettings,
    strategy_registry: StrategyRegistry,
    risk_engine: RiskPolicyEngine,
    hysteresis: RegimeHysteresis | None = None,
    veto_service: VetoServicePort | None = None,
    regime_classifier: AIRegimeClassifierPort | None = None,
    adaptive_variant: str | None = None,
    strategy_scope_cache: dict[tuple[str, int | None], StrategyParams] | None = None,
    risk_scope_cache: dict[tuple[str, int | None], RiskParams] | None = None,
    has_strategy_rules: bool | None = None,
    phase2_scope_cache: dict[tuple[str, int | None, str | None], StrategyParams] | None = None,
) -> PipelineResult:
    """Run the shared decision pipeline (regime -> variants -> strategy -> veto -> risk -> sizing).

    Parameters
    ----------
    context:
        Aligned market + portfolio batch for one decision cycle.
    settings:
        Full engine settings bundle.
    strategy_registry:
        Registered strategy plugins.
    risk_engine:
        Risk policy engine.
    hysteresis:
        Optional regime hysteresis state machine (live tick and backtest replay).
    veto_service:
        Optional AI veto filter (``None`` -> all-approved).
    regime_classifier:
        Optional AI regime classifier (``None`` -> deterministic-only).
    adaptive_variant:
        Optional adaptive variant override (``None`` -> scope-only resolution).
    """
    core = _run_core_pipeline(
        context=context,
        settings=settings,
        strategy_registry=strategy_registry,
        hysteresis=hysteresis,
        adaptive_variant=adaptive_variant,
        strategy_scope_cache=strategy_scope_cache,
        has_strategy_rules=has_strategy_rules,
        phase2_scope_cache=phase2_scope_cache,
    )

    # 4. AI regime ensemble (confidence adjustment only -- coarse_state never overridden)
    regimes, ai_regime_active = await _evaluate_ai_regime(context, core.regimes, settings, regime_classifier)

    # 7. AI veto filter (enrich context with regime data for confidence gating)
    enriched_context = BatchDecisionContext(
        market=context.market,
        portfolio=context.portfolio,
        regimes=regimes,
    )
    veto = await _evaluate_veto(enriched_context, core.candidates, settings, veto_service)
    vetoed = _apply_veto(core.candidates, veto)

    risk, intents = _run_risk_and_sizing(
        context=context,
        settings=settings,
        risk_engine=risk_engine,
        regimes=regimes,
        vetoed=vetoed,
        risk_scope_cache=risk_scope_cache,
    )

    authority = _authority(settings, veto_service, veto, ai_regime_active=ai_regime_active)

    return PipelineResult(
        regimes=regimes,
        candidates=core.candidates,
        veto=veto,
        risk=risk,
        intents=intents,
        variants=core.variants,
        authority=authority,
        confidence_raw=core.confidence_raw,
        per_pair_params=core.per_pair_params,
    )
