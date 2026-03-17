"""Integration tests for the research gate anti-overfitting check."""

import numpy as np

from dojiwick.application.use_cases.validation.cross_validator import CVResult
from dojiwick.application.use_cases.validation.research_gate import GateResult, GateThresholds, evaluate_research_gate
from dojiwick.application.use_cases.validation.walk_forward_validator import WalkForwardResult, WindowResult
from dojiwick.domain.enums import WFMode


def _make_cv_result(mean_sharpe: float) -> CVResult:
    return CVResult(
        fold_sharpes=np.array([mean_sharpe] * 5, dtype=np.float64),
        mean_sharpe=mean_sharpe,
        std_sharpe=0.01,
        min_sharpe=mean_sharpe - 0.02,
    )


def _make_wf_result(
    oos_is_ratio: float,
    aggregate_oos_sharpe: float | None = None,
) -> WalkForwardResult:
    if aggregate_oos_sharpe is None:
        aggregate_oos_sharpe = oos_is_ratio
    return WalkForwardResult(
        windows=(
            WindowResult(
                is_sharpe=1.0,
                oos_sharpe=oos_is_ratio,
                is_start=0,
                is_end=199,
                oos_start=200,
                oos_end=249,
                is_trades=40,
                oos_trades=38,
            ),
        ),
        aggregate_oos_sharpe=aggregate_oos_sharpe,
        oos_is_ratio=oos_is_ratio,
        min_oos_sharpe=aggregate_oos_sharpe,
    )


def _gate(
    cv_result: CVResult,
    pbo: float,
    walk_forward_result: WalkForwardResult,
    *,
    min_cv_sharpe: float = 0.3,
    max_pbo: float = 0.5,
    wf_mode: WFMode = WFMode.RATIO,
    min_oos_degradation_ratio: float = 0.5,
    min_wf_oos_sharpe: float = 0.0,
) -> GateResult:
    """Test helper wrapping evaluate_research_gate with sensible defaults."""
    return evaluate_research_gate(
        cv_result,
        pbo,
        walk_forward_result,
        thresholds=GateThresholds(
            min_cv_sharpe=min_cv_sharpe,
            max_pbo=max_pbo,
            wf_mode=wf_mode,
            min_oos_degradation_ratio=min_oos_degradation_ratio,
            min_wf_oos_sharpe=min_wf_oos_sharpe,
        ),
    )


class TestResearchGate:
    def test_gate_rejects_overfit_strategy(self) -> None:
        """Low CV Sharpe + high PBO + low degradation ratio -> rejection."""
        result = _gate(
            cv_result=_make_cv_result(mean_sharpe=0.1),
            pbo=0.8,
            walk_forward_result=_make_wf_result(oos_is_ratio=0.2),
        )
        assert not result.passed
        assert len(result.rejection_reasons) == 3

    def test_gate_accepts_robust_strategy(self) -> None:
        """Good CV Sharpe + low PBO + healthy degradation -> pass."""
        result = _gate(
            cv_result=_make_cv_result(mean_sharpe=0.8),
            pbo=0.1,
            walk_forward_result=_make_wf_result(oos_is_ratio=0.75),
        )
        assert result.passed
        assert len(result.rejection_reasons) == 0

    def test_gate_rejection_reasons_reported(self) -> None:
        """Each failing criterion should produce a distinct reason string."""
        result = _gate(
            cv_result=_make_cv_result(mean_sharpe=0.2),
            pbo=0.7,
            walk_forward_result=_make_wf_result(oos_is_ratio=0.3),
            min_cv_sharpe=0.5,
            max_pbo=0.3,
            min_oos_degradation_ratio=0.6,
        )
        assert not result.passed
        reasons = result.rejection_reasons
        assert any("CV Sharpe" in r for r in reasons)
        assert any("PBO" in r for r in reasons)
        assert any("OOS/IS ratio" in r for r in reasons)

    def test_single_criterion_failure(self) -> None:
        """Only one failing criterion should still reject."""
        result = _gate(
            cv_result=_make_cv_result(mean_sharpe=0.8),
            pbo=0.9,  # only PBO fails
            walk_forward_result=_make_wf_result(oos_is_ratio=0.75),
        )
        assert not result.passed
        assert len(result.rejection_reasons) == 1
        assert "PBO" in result.rejection_reasons[0]

    def test_wf_mode_ratio_is_default(self) -> None:
        """Default wf_mode='ratio' checks only OOS/IS ratio, not OOS Sharpe."""
        result = _gate(
            cv_result=_make_cv_result(mean_sharpe=0.8),
            pbo=0.1,
            walk_forward_result=_make_wf_result(oos_is_ratio=0.2, aggregate_oos_sharpe=0.8),
            min_wf_oos_sharpe=0.3,
        )
        assert not result.passed
        assert any("OOS/IS ratio" in r for r in result.rejection_reasons)
        assert not any("WF OOS Sharpe" in r for r in result.rejection_reasons)

    def test_wf_mode_oos_sharpe_passes_low_ratio(self) -> None:
        """wf_mode='oos_sharpe' passes when low ratio but good absolute OOS Sharpe."""
        result = _gate(
            cv_result=_make_cv_result(mean_sharpe=0.8),
            pbo=0.1,
            walk_forward_result=_make_wf_result(oos_is_ratio=0.03, aggregate_oos_sharpe=0.5),
            wf_mode=WFMode.OOS_SHARPE,
            min_wf_oos_sharpe=0.3,
        )
        assert result.passed
        assert len(result.rejection_reasons) == 0

    def test_wf_mode_oos_sharpe_fails_low_sharpe(self) -> None:
        """wf_mode='oos_sharpe' fails when OOS Sharpe < threshold."""
        result = _gate(
            cv_result=_make_cv_result(mean_sharpe=0.8),
            pbo=0.1,
            walk_forward_result=_make_wf_result(oos_is_ratio=0.03, aggregate_oos_sharpe=0.1),
            wf_mode=WFMode.OOS_SHARPE,
            min_wf_oos_sharpe=0.3,
        )
        assert not result.passed
        assert any("WF OOS Sharpe" in r for r in result.rejection_reasons)
        assert not any("OOS/IS ratio" in r for r in result.rejection_reasons)

    def test_wf_mode_both_requires_both(self) -> None:
        """wf_mode='both' requires both ratio and OOS Sharpe criteria."""
        # Good OOS Sharpe but low ratio → still fails
        result = _gate(
            cv_result=_make_cv_result(mean_sharpe=0.8),
            pbo=0.1,
            walk_forward_result=_make_wf_result(oos_is_ratio=0.2, aggregate_oos_sharpe=0.5),
            wf_mode=WFMode.BOTH,
            min_wf_oos_sharpe=0.3,
        )
        assert not result.passed
        assert any("OOS/IS ratio" in r for r in result.rejection_reasons)
        assert not any("WF OOS Sharpe" in r for r in result.rejection_reasons)

        # Good ratio but low OOS Sharpe → still fails
        result2 = _gate(
            cv_result=_make_cv_result(mean_sharpe=0.8),
            pbo=0.1,
            walk_forward_result=_make_wf_result(oos_is_ratio=0.8, aggregate_oos_sharpe=0.1),
            wf_mode=WFMode.BOTH,
            min_wf_oos_sharpe=0.3,
        )
        assert not result2.passed
        assert any("WF OOS Sharpe" in r for r in result2.rejection_reasons)

        # Both good → passes
        result3 = _gate(
            cv_result=_make_cv_result(mean_sharpe=0.8),
            pbo=0.1,
            walk_forward_result=_make_wf_result(oos_is_ratio=0.8, aggregate_oos_sharpe=0.5),
            wf_mode=WFMode.BOTH,
            min_wf_oos_sharpe=0.3,
        )
        assert result3.passed

    def test_gate_result_includes_oos_sharpe_and_windows(self) -> None:
        """GateResult exposes aggregate_oos_sharpe and wf_windows."""
        wf = _make_wf_result(oos_is_ratio=0.8, aggregate_oos_sharpe=0.5)
        result = _gate(
            cv_result=_make_cv_result(mean_sharpe=0.8),
            pbo=0.1,
            walk_forward_result=wf,
        )
        assert result.aggregate_oos_sharpe == 0.5
        assert len(result.wf_windows) == 1
        assert result.wf_windows[0].is_trades == 40
        assert result.wf_windows[0].oos_trades == 38
