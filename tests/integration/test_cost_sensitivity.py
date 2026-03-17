"""Cost model sensitivity tests — verify fee and slippage parameters affect PnL."""

import numpy as np

from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.use_cases.run_backtest import BacktestService
from fixtures.factories.domain import ContextBuilder
from fixtures.factories.infrastructure import default_backtest_settings, default_settings


async def test_fee_multiplier_changes_pnl() -> None:
    """fee_multiplier=2.0 vs fee_multiplier=3.0 → different total_pnl_usd."""
    context = ContextBuilder().trending_up().build()
    next_prices = context.market.price + np.ones(context.size, dtype=np.float64)
    registry = build_default_strategy_registry()

    s1 = default_settings().model_copy(update={"backtest": default_backtest_settings(fee_multiplier=2.0)})
    svc1 = BacktestService(settings=s1, strategy_registry=registry, risk_engine=build_default_risk_engine(s1.risk))
    r1 = await svc1.run(context, next_prices)

    s2 = default_settings().model_copy(update={"backtest": default_backtest_settings(fee_multiplier=3.0)})
    svc2 = BacktestService(settings=s2, strategy_registry=registry, risk_engine=build_default_risk_engine(s2.risk))
    r2 = await svc2.run(context, next_prices)

    assert r1.total_pnl_usd != r2.total_pnl_usd, "different fee_multiplier should produce different PnL"


async def test_slippage_changes_pnl() -> None:
    """slippage_bps=2.0 vs slippage_bps=5.0 → different total_pnl_usd."""
    context = ContextBuilder().trending_up().build()
    next_prices = context.market.price + np.ones(context.size, dtype=np.float64)
    registry = build_default_strategy_registry()

    s1 = default_settings().model_copy(update={"backtest": default_backtest_settings(slippage_bps=2.0)})
    svc1 = BacktestService(settings=s1, strategy_registry=registry, risk_engine=build_default_risk_engine(s1.risk))
    r1 = await svc1.run(context, next_prices)

    s2 = default_settings().model_copy(update={"backtest": default_backtest_settings(slippage_bps=5.0)})
    svc2 = BacktestService(settings=s2, strategy_registry=registry, risk_engine=build_default_risk_engine(s2.risk))
    r2 = await svc2.run(context, next_prices)

    assert r1.total_pnl_usd != r2.total_pnl_usd, "different slippage_bps should produce different PnL"


async def test_zero_fees_maximizes_pnl() -> None:
    """fee_bps=0, slippage_bps=0 ≥ fee_bps=4, slippage_bps=2."""
    context = ContextBuilder().trending_up().build()
    next_prices = context.market.price + np.ones(context.size, dtype=np.float64)
    registry = build_default_strategy_registry()

    s_zero = default_settings().model_copy(
        update={"backtest": default_backtest_settings(fee_bps=0.0, slippage_bps=0.0)}
    )
    svc_zero = BacktestService(
        settings=s_zero, strategy_registry=registry, risk_engine=build_default_risk_engine(s_zero.risk)
    )
    r_zero = await svc_zero.run(context, next_prices)

    s_fees = default_settings().model_copy(
        update={"backtest": default_backtest_settings(fee_bps=4.0, slippage_bps=2.0)}
    )
    svc_fees = BacktestService(
        settings=s_fees, strategy_registry=registry, risk_engine=build_default_risk_engine(s_fees.risk)
    )
    r_fees = await svc_fees.run(context, next_prices)

    assert r_zero.total_pnl_usd >= r_fees.total_pnl_usd, "zero fees should produce >= PnL vs nonzero fees"
