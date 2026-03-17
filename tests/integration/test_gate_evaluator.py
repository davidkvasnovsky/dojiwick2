"""DefaultGateEvaluator integration test — verifies end-to-end wiring."""

from dojiwick.application.use_cases.validation.gate_evaluator import DefaultGateEvaluator
from fixtures.factories.domain import TimeSeriesBuilder
from fixtures.factories.infrastructure import default_research_gate_settings, default_settings


async def test_gate_evaluator_produces_gate_result() -> None:
    """Evaluator wires CV, PBO, and walk-forward, returning a GateResult."""
    series = TimeSeriesBuilder(n_bars=10).build()
    settings = default_settings().model_copy(
        update={
            "research": default_research_gate_settings(
                enabled=True,
                cv_folds=2,
                purge_bars=0,
                embargo_bars=0,
                wf_train_size=4,
                wf_test_size=2,
            ),
        }
    )

    evaluator = DefaultGateEvaluator(
        settings=settings, series=series, target_ids=("btc_usdc", "eth_usdc"), venue="binance", product="usd_c"
    )

    result = await evaluator.evaluate(
        {
            "stop_atr_mult": 1.5,
            "rr_ratio": 2.0,
            "min_stop_distance_pct": 0.3,
            "mean_rsi_oversold": 35.0,
            "mean_rsi_overbought": 70.0,
            "vol_extreme_oversold": 30.0,
            "vol_extreme_overbought": 75.0,
            "trend_pullback_rsi_max": 45.0,
            "trend_breakout_adx_min": 40.0,
            "adx_trend_min": 20.0,
            "atr_high_pct": 0.9,
            "trailing_stop_activation_rr": 1.0,
            "trailing_stop_atr_mult": 1.0,
            "breakeven_after_rr": 1.0,
            "max_hold_bars": 48,
        }
    )

    assert isinstance(result.passed, bool)
    assert isinstance(result.cv_sharpe, float)
    assert 0.0 <= result.pbo <= 1.0
    assert isinstance(result.oos_degradation_ratio, float)
    assert isinstance(result.rejection_reasons, tuple)
