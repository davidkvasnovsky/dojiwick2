"""End-to-end exit engine tests: bar timing, gap fills, liquidation, equity pool, funding."""

import numpy as np
import pytest
from fixtures.factories.domain import ContextBuilder, TimeSeriesBuilder
from fixtures.factories.infrastructure import default_backtest_settings, default_risk_settings, default_settings

from dojiwick.application.orchestration.decision_pipeline import run_decision_pipeline_sync
from dojiwick.application.policies.risk.defaults import build_default_risk_engine
from dojiwick.application.registry.strategy_registry import build_default_strategy_registry
from dojiwick.application.use_cases.run_backtest import BacktestService, BacktestTimeSeries
from dojiwick.compute.kernels.pnl.pnl import scalar_net_pnl
from dojiwick.config.schema import Settings
from dojiwick.domain.enums import CloseReason
from dojiwick.domain.models.value_objects.batch_models import BatchExecutionIntent

_PAIR = ("BTC/USDC",)


def _service(settings: Settings) -> BacktestService:
    return BacktestService(
        settings=settings,
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
        config_hash="test",
    )


def _entry_intents(settings: Settings) -> BatchExecutionIntent:
    """Resolve the deterministic entry intent the pipeline emits for the preset context."""
    ctx = ContextBuilder(pairs=_PAIR).mean_revert_buy().build()
    pipeline = run_decision_pipeline_sync(
        context=ctx,
        settings=settings,
        strategy_registry=build_default_strategy_registry(),
        risk_engine=build_default_risk_engine(default_risk_settings()),
    )
    return pipeline.intents


def _series_with_ohlc(
    n_bars: int,
    bar1: tuple[float, float, float, float],
    benign: tuple[float, float, float, float],
    funding: list[float] | None = None,
) -> BacktestTimeSeries:
    """Single-pair series: entry context on every bar; bar 1 OHLC crafted, rest benign.

    next_* index t carries bar t+1 data, so bar1's OHLC lives at index 0.
    """
    contexts = tuple(ContextBuilder(pairs=_PAIR).mean_revert_buy().build() for _ in range(n_bars))
    o1, h1, l1, c1 = bar1
    ob, hb, lb, cb = benign
    opens = [np.array([o1 if t == 0 else ob]) for t in range(n_bars)]
    highs = [np.array([h1 if t == 0 else hb]) for t in range(n_bars)]
    lows = [np.array([l1 if t == 0 else lb]) for t in range(n_bars)]
    closes = [np.array([c1 if t == 0 else cb]) for t in range(n_bars)]
    return BacktestTimeSeries(
        contexts=contexts,
        next_prices=tuple(closes),
        active_mask=np.ones((n_bars, 1), dtype=np.bool_),
        next_open=tuple(opens),
        next_high=tuple(highs),
        next_low=tuple(lows),
        next_funding=tuple(np.array([f]) for f in funding) if funding is not None else None,
    )


async def test_first_post_entry_bar_is_stop_checked_and_gaps_fill_at_open() -> None:
    settings = default_settings()
    intents = _entry_intents(settings)
    assert bool(intents.active_mask[0]), "preset must produce an entry"
    entry = float(intents.entry_price[0])
    stop = float(intents.stop_price[0])
    assert 0.0 < stop < entry

    gap_open = stop - 2.0
    series = _series_with_ohlc(
        n_bars=4,
        bar1=(gap_open, gap_open + 1.0, gap_open - 1.0, gap_open + 0.5),
        benign=(entry, entry + 0.1, entry - 0.1, entry),
    )
    result = await _service(settings).run_with_hysteresis(series)

    first = result.trade_details[0]
    assert first.bar_index == 0
    assert first.exit_bar_index == 1, "stop on the first post-entry bar must close the trade on that bar"
    assert first.close_reason is CloseReason.STOP_LOSS
    assert first.exit_price == pytest.approx(gap_open), "gap through the stop must fill at the bar open"  # pyright: ignore[reportUnknownMemberType]
    assert first.hold_bars == 1


async def test_liquidation_outranks_stop_and_caps_loss_at_margin() -> None:
    settings = default_settings().model_copy(
        update={"backtest": default_backtest_settings(leverage=3.0, maintenance_margin_rate=0.05)}
    )
    intents = _entry_intents(settings)
    assert bool(intents.active_mask[0])
    entry = float(intents.entry_price[0])
    liq = entry * (1.0 - (1.0 / 3.0 - 0.05))

    crash_open = liq - 1.0
    series = _series_with_ohlc(
        n_bars=3,
        bar1=(crash_open, crash_open + 0.5, crash_open - 0.5, crash_open),
        benign=(entry, entry + 0.1, entry - 0.1, entry),
    )
    result = await _service(settings).run_with_hysteresis(series)

    first = result.trade_details[0]
    assert first.close_reason is CloseReason.LIQUIDATION
    assert first.exit_price == pytest.approx(crash_open), "gap below liq price fills at the open"  # pyright: ignore[reportUnknownMemberType]
    expected = max(
        scalar_net_pnl(
            is_long=True,
            entry_price=first.entry_price,
            exit_price=crash_open,
            quantity=first.quantity,
            notional=first.notional_usd,
            slippage_bps=settings.backtest.cost_model.slippage_bps,
            fee_bps=settings.backtest.cost_model.fee_bps,
            fee_multiplier=settings.backtest.cost_model.fee_multiplier,
            leverage=3.0,
        ),
        -first.notional_usd,
    )
    assert first.pnl_usd == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]
    assert first.pnl_usd >= -first.notional_usd - 1e-9, "loss can never exceed the tranche margin"


async def test_equity_pool_is_shared_across_pairs() -> None:
    series = TimeSeriesBuilder(n_bars=30).with_regime_sequence(["mean_revert_buy"] * 30).build()
    settings = default_settings()
    result = await _service(settings).run_with_hysteresis(series)

    assert result.summary.trades > 0
    curve = result.summary.portfolio_equity_curve
    assert curve is not None
    total_from_curve = float(curve[-1] - settings.backtest.equity_usd)
    assert total_from_curve == pytest.approx(result.summary.total_pnl_usd, abs=1e-6), (  # pyright: ignore[reportUnknownMemberType]
        "final pool equity must reflect the SUM of all pairs' PnL, not the per-pair mean"
    )


async def test_funding_accrual_charged_on_held_bars() -> None:
    settings = default_settings()
    intents = _entry_intents(settings)
    assert bool(intents.active_mask[0])
    entry = float(intents.entry_price[0])
    stop = float(intents.stop_price[0])
    rate = 0.001

    gap_open = stop - 2.0
    # Funding settles on bar 1 (the bar the trade also exits on)
    base = _series_with_ohlc(
        n_bars=4,
        bar1=(gap_open, gap_open + 1.0, gap_open - 1.0, gap_open + 0.5),
        benign=(entry, entry + 0.1, entry - 0.1, entry),
    )
    funded = _series_with_ohlc(
        n_bars=4,
        bar1=(gap_open, gap_open + 1.0, gap_open - 1.0, gap_open + 0.5),
        benign=(entry, entry + 0.1, entry - 0.1, entry),
        funding=[rate, 0.0, 0.0, 0.0],
    )
    r_base = await _service(settings).run_with_hysteresis(base)
    r_funded = await _service(settings).run_with_hysteresis(funded)

    # Risk scaling can shrink the position below the raw intent — charge on
    # the position's actual notional as reported in the trade detail.
    position_notional = r_base.trade_details[0].notional_usd
    diff = r_base.trade_details[0].pnl_usd - r_funded.trade_details[0].pnl_usd
    assert diff == pytest.approx(rate * position_notional * settings.backtest.leverage, rel=1e-9), (  # pyright: ignore[reportUnknownMemberType]
        "a long position held across a funding settlement must pay rate x leveraged notional"
    )
