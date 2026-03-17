"""Research gate — anti-overfitting promotion check.

Pure function that evaluates cross-validation, PBO, and walk-forward
results against configurable thresholds.  Produces a pass/fail verdict
with per-criterion rejection reasons.
"""

from dataclasses import dataclass

from dojiwick.application.use_cases.validation.cross_validator import CVResult
from dojiwick.application.use_cases.validation.walk_forward_validator import WalkForwardResult, WindowResult
from dojiwick.domain.enums import WFMode
from dojiwick.domain.models.value_objects.outcome_models import TradeDetail, pick_effective_drawdown


@dataclass(slots=True, frozen=True, kw_only=True)
class ContinuousBacktestResult:
    """Result of a full continuous backtest validation."""

    trades: int
    total_pnl_usd: float
    max_drawdown_pct: float
    max_portfolio_drawdown_pct: float = 0.0
    trade_details: tuple[TradeDetail, ...] = ()
    monthly_pnl: dict[str, float] | None = None

    @property
    def effective_max_drawdown_pct(self) -> float:
        return pick_effective_drawdown(self.max_portfolio_drawdown_pct, self.max_drawdown_pct)


@dataclass(slots=True, frozen=True, kw_only=True)
class ShockTestResult:
    """Profit factor under TP/SL perturbation."""

    profit_factor: float


@dataclass(slots=True, frozen=True, kw_only=True)
class RegimePFResult:
    """Profit factor per regime."""

    regime_pfs: dict[str, float]


@dataclass(slots=True, frozen=True, kw_only=True)
class ConcentrationResult:
    """PnL concentration metrics (0-100 percentage scale)."""

    max_month_pct: float
    max_trade_pct: float


@dataclass(slots=True, frozen=True, kw_only=True)
class PairRobustnessResult:
    """Pair-level profit factor check."""

    pairs_above_threshold: int
    total_pairs: int


@dataclass(slots=True, frozen=True, kw_only=True)
class GateThresholds:
    """All threshold values for gate evaluation."""

    min_cv_sharpe: float
    max_pbo: float
    wf_mode: WFMode
    min_oos_degradation_ratio: float
    min_wf_oos_sharpe: float
    min_continuous_trades: int = 0
    max_continuous_drawdown_pct: float = 100.0
    shock_test_min_pf: float = 0.0
    per_regime_min_pf: float = 0.0
    concentration_max_month_pct: float = 100.0
    concentration_max_trade_pct: float = 100.0
    pair_robustness_min_pairs: int = 0


@dataclass(slots=True, frozen=True, kw_only=True)
class GateCheckResults:
    """Pre-computed check results passed to gate evaluation."""

    continuous: ContinuousBacktestResult | None = None
    shock_test: ShockTestResult | None = None
    regime_pf: RegimePFResult | None = None
    concentration: ConcentrationResult | None = None
    pair_robustness: PairRobustnessResult | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class GateResult:
    """Research gate evaluation result."""

    passed: bool
    cv_sharpe: float
    pbo: float
    oos_degradation_ratio: float
    rejection_reasons: tuple[str, ...]
    aggregate_oos_sharpe: float = 0.0
    wf_windows: tuple[WindowResult, ...] = ()
    checks: GateCheckResults | None = None


def evaluate_research_gate(
    cv_result: CVResult,
    pbo: float,
    walk_forward_result: WalkForwardResult,
    *,
    thresholds: GateThresholds,
    checks: GateCheckResults | None = None,
) -> GateResult:
    """Evaluate strategy against anti-overfitting criteria.

    Checks up to nine criteria:
    1. Mean CV Sharpe >= ``min_cv_sharpe``
    2. PBO <= ``max_pbo``
    3. Walk-forward check (mode-dependent)
    4. Continuous backtest: trades >= ``min_continuous_trades`` (if enabled)
    5. Continuous backtest: max drawdown <= ``max_continuous_drawdown_pct`` (if enabled)
    6. Shock test: PF under TP/SL perturbation >= ``shock_test_min_pf``
    7. Per-regime PF >= ``per_regime_min_pf`` for all regimes with sufficient trades
    8. Concentration: no month > ``concentration_max_month_pct`` of total PnL
    9. Pair robustness: >= ``pair_robustness_min_pairs`` pairs with PF above threshold
    """
    t = thresholds
    reasons: list[str] = []

    if cv_result.mean_sharpe < t.min_cv_sharpe:
        reasons.append(f"CV Sharpe {cv_result.mean_sharpe:.4f} < min {t.min_cv_sharpe}")

    if pbo > t.max_pbo:
        reasons.append(f"PBO {pbo:.4f} > max {t.max_pbo}")

    check_ratio = t.wf_mode in (WFMode.RATIO, WFMode.BOTH)
    check_oos_sharpe = t.wf_mode in (WFMode.OOS_SHARPE, WFMode.BOTH)

    if check_ratio and walk_forward_result.oos_is_ratio < t.min_oos_degradation_ratio:
        reasons.append(f"OOS/IS ratio {walk_forward_result.oos_is_ratio:.4f} < min {t.min_oos_degradation_ratio}")

    if check_oos_sharpe and walk_forward_result.aggregate_oos_sharpe < t.min_wf_oos_sharpe:
        reasons.append(f"WF OOS Sharpe {walk_forward_result.aggregate_oos_sharpe:.4f} < min {t.min_wf_oos_sharpe}")

    if checks is not None:
        _check_continuous(checks.continuous, t, reasons)
        _check_shock_test(checks.shock_test, t, reasons)
        _check_regime_pf(checks.regime_pf, t, reasons)
        _check_concentration(checks.concentration, t, reasons)
        _check_pair_robustness(checks.pair_robustness, t, reasons)

    return GateResult(
        passed=len(reasons) == 0,
        cv_sharpe=cv_result.mean_sharpe,
        pbo=pbo,
        oos_degradation_ratio=walk_forward_result.oos_is_ratio,
        rejection_reasons=tuple(reasons),
        aggregate_oos_sharpe=walk_forward_result.aggregate_oos_sharpe,
        wf_windows=walk_forward_result.windows,
        checks=checks,
    )


def _check_continuous(continuous: ContinuousBacktestResult | None, t: GateThresholds, reasons: list[str]) -> None:
    if continuous is None:
        return
    if t.min_continuous_trades > 0 and continuous.trades < t.min_continuous_trades:
        reasons.append(f"continuous backtest {continuous.trades} trades < min {t.min_continuous_trades}")
    dd_check = continuous.effective_max_drawdown_pct
    if dd_check > t.max_continuous_drawdown_pct:
        reasons.append(f"continuous backtest DD {dd_check:.1f}% > max {t.max_continuous_drawdown_pct:.1f}%")


def _check_shock_test(shock_test: ShockTestResult | None, t: GateThresholds, reasons: list[str]) -> None:
    if shock_test is not None and t.shock_test_min_pf > 0:
        if shock_test.profit_factor < t.shock_test_min_pf:
            reasons.append(f"shock test PF {shock_test.profit_factor:.2f} < min {t.shock_test_min_pf:.2f}")


def _check_regime_pf(regime_pf: RegimePFResult | None, t: GateThresholds, reasons: list[str]) -> None:
    if regime_pf is not None and t.per_regime_min_pf > 0:
        for regime_name, pf_val in sorted(regime_pf.regime_pfs.items()):
            if pf_val < t.per_regime_min_pf:
                reasons.append(f"regime {regime_name} PF {pf_val:.2f} < min {t.per_regime_min_pf:.2f}")


def _check_concentration(concentration: ConcentrationResult | None, t: GateThresholds, reasons: list[str]) -> None:
    if concentration is None:
        return
    if t.concentration_max_month_pct < 100.0 and concentration.max_month_pct > t.concentration_max_month_pct:
        reasons.append(
            f"max month concentration {concentration.max_month_pct:.1f}% > {t.concentration_max_month_pct:.1f}%"
        )
    if t.concentration_max_trade_pct < 100.0 and concentration.max_trade_pct > t.concentration_max_trade_pct:
        reasons.append(
            f"max trade concentration {concentration.max_trade_pct:.1f}% > {t.concentration_max_trade_pct:.1f}%"
        )


def _check_pair_robustness(pair_robustness: PairRobustnessResult | None, t: GateThresholds, reasons: list[str]) -> None:
    if pair_robustness is not None and t.pair_robustness_min_pairs > 0:
        if pair_robustness.pairs_above_threshold < t.pair_robustness_min_pairs:
            reasons.append(
                f"only {pair_robustness.pairs_above_threshold}/{pair_robustness.total_pairs} pairs above PF threshold, need {t.pair_robustness_min_pairs}"
            )
