"""Shared CLI helpers for backtest-based CLIs."""

import argparse
import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

from dojiwick.application.services.backtest_builder import build_backtest_time_series
from dojiwick.application.use_cases.run_backtest import BacktestService, BacktestTimeSeries, build_backtest_service
from dojiwick.config.composition import build_market_data_fetcher
from dojiwick.config.loader import load_settings
from dojiwick.config.schema import Settings
from dojiwick.config.targets import resolve_symbols, resolve_target_ids
from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.type_aliases import CandleInterval

log = logging.getLogger(__name__)


def setup_env() -> None:
    """Load .env and configure basic logging — shared across CLI entrypoints."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add standard backtest CLI arguments."""
    parser.add_argument("--config", type=Path, required=True, help="Path to config.toml")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")


async def load_settings_and_series(
    args: argparse.Namespace,
) -> tuple[Settings, BacktestTimeSeries, Callable[[], Awaitable[None]]]:
    """Load settings, fetch candles, build time series. Returns (settings, series, cleanup)."""
    settings = load_settings(args.config)

    use_cache = settings.backtest.use_candle_cache
    market_data, cleanup = await build_market_data_fetcher(settings, use_cache=use_cache)

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
    interval = CandleInterval(settings.trading.candle_interval)

    pairs = settings.trading.active_pairs
    symbols = resolve_symbols(settings)

    async def _fetch_pair(pair: str, symbol: str) -> tuple[str, tuple[Candle, ...]]:
        log.info("fetching %s candles for %s (%s to %s)", interval, symbol, args.start, args.end)
        candles = await market_data.fetch_candles_range(symbol, interval, start, end)
        log.info("  -> %d candles", len(candles))
        return pair, candles

    results = await asyncio.gather(*(_fetch_pair(p, s) for p, s in zip(pairs, symbols)))
    candles_by_pair: dict[str, tuple[Candle, ...]] = dict(results)

    for pair, candles in candles_by_pair.items():
        if len(candles) == 0:
            log.error("no candles returned for %s — check date range and pair", pair)
            sys.exit(1)

    warmup = settings.backtest.warmup_bars
    t = settings.trading
    series = build_backtest_time_series(
        candles_by_pair,
        pairs,
        warmup_bars=warmup,
        equity_usd=settings.backtest.equity_usd,
        rsi_period=t.rsi_period,
        ema_fast_period=t.ema_fast_period,
        ema_slow_period=t.ema_slow_period,
        ema_base_period=t.ema_base_period,
        ema_trend_period=t.ema_trend_period,
        atr_period=t.atr_period,
        adx_period=t.adx_period,
        bb_period=t.bb_period,
        bb_std=t.bb_std,
        volume_ema_period=t.volume_ema_period,
    )
    log.info("built time series: %d bars x %d pairs", series.n_bars, series.n_pairs)

    return settings, series, cleanup


def print_gate_result(result: object) -> None:
    """Print research gate result summary. Accepts any object with gate result fields."""
    from dojiwick.application.use_cases.validation.research_gate import GateResult

    r = cast(GateResult, result)
    print(f"  Passed:          {r.passed}")
    print(f"  CV Sharpe:       {r.cv_sharpe:.3f}")
    print(f"  PBO:             {r.pbo:.3f}")
    print(f"  OOS/IS Ratio:    {r.oos_degradation_ratio:.3f}")
    print(f"  Agg OOS Sharpe:  {r.aggregate_oos_sharpe:.3f}")
    if r.rejection_reasons:
        print(f"  Rejections:      {', '.join(r.rejection_reasons)}")


def print_wf_windows(windows: object) -> None:
    """Print walk-forward window table."""
    from dojiwick.application.use_cases.validation.walk_forward_validator import WindowResult

    ws = cast(tuple[WindowResult, ...], windows)
    if not ws:
        return
    print(f"\n  {'Window':<8} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'IS Trades':>10} {'OOS Trades':>11}")
    print(f"  {'-' * 52}")
    for i, w in enumerate(ws):
        print(f"  {i + 1:<8} {w.is_sharpe:>10.4f} {w.oos_sharpe:>11.4f} {w.is_trades:>10} {w.oos_trades:>11}")
    print()


def build_service(settings: Settings) -> BacktestService:
    """Build a BacktestService from settings."""
    return build_backtest_service(
        settings,
        target_ids=resolve_target_ids(settings),
        venue=str(settings.exchange.venue),
        product=str(settings.exchange.product),
    )
