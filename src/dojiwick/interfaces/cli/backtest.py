"""CLI entrypoint for running backtests on historical Binance data.

Usage::

    python -m dojiwick.interfaces.cli.backtest \
        --config config.toml --start 2025-01-01 --end 2025-06-01
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from dojiwick.config.scope import regime_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from dojiwick.domain.models.value_objects.outcome_models import TradeDetail

log = logging.getLogger(__name__)


# Trade analysis helpers


def _group_stats(trades: list[TradeDetail]) -> tuple[int, int, float, float, float]:
    """Return (count, wins, win_rate, total_pnl, avg_pnl) for a group of trades."""
    count = len(trades)
    wins = sum(1 for t in trades if t.pnl_usd > 0)
    wr = wins / count if count else 0.0
    total = sum(t.pnl_usd for t in trades)
    avg = total / count if count else 0.0
    return count, wins, wr, total, avg


def _print_grouped_breakdown(
    title: str,
    trades: tuple[TradeDetail, ...],
    key_func: Callable[[TradeDetail], str],
    *,
    label_width: int = 16,
    show_avg_pnl: bool = False,
    show_total_pnl: bool = True,
    sort_keys: tuple[str, ...] | None = None,
) -> None:
    """Group trades by *key_func*, print count / win-rate / pnl table."""
    groups: dict[str, list[TradeDetail]] = defaultdict(list)
    for td in trades:
        groups[key_func(td)].append(td)

    hdr = f"    {'':<{label_width}} {'Count':>5}  {'Win Rate':>8}"
    if show_avg_pnl:
        hdr += f"  {'Avg PnL':>10}"
    if show_total_pnl:
        hdr += f"  {'Total PnL':>10}"
    print(f"  {title}:")
    print(hdr)

    if sort_keys is not None:
        keys: list[str] = [k for k in sort_keys if k in groups]
    else:
        keys = sorted(groups, key=lambda k: sum(t.pnl_usd for t in groups[k]), reverse=True)

    for key in keys:
        count, _wins, wr, total, avg = _group_stats(groups[key])
        line = f"    {key:<{label_width}} {count:>5}  {wr:>7.1%}"
        if show_avg_pnl:
            line += f"  ${avg:>9,.2f}"
        if show_total_pnl:
            line += f"  ${total:>9,.2f}"
        print(line)
    print()


def _print_cross_tab(
    title: str,
    trades: tuple[TradeDetail, ...],
    row_func: Callable[[TradeDetail], str],
    col_func: Callable[[TradeDetail], str],
) -> None:
    """Print a compact count(WR%) cross-tabulation matrix."""
    cells: dict[str, dict[str, list[TradeDetail]]] = defaultdict(lambda: defaultdict(list))
    for td in trades:
        cells[row_func(td)][col_func(td)].append(td)

    all_cols = sorted({col_func(td) for td in trades})
    all_rows = sorted(cells.keys())
    col_w = max(14, *(len(c) + 2 for c in all_cols))
    row_w = max(18, *(len(r) + 2 for r in all_rows))

    print(f"  {title}:")
    header = f"    {'':<{row_w}}" + "".join(f"{c:>{col_w}}" for c in all_cols)
    print(header)
    for row in all_rows:
        parts = f"    {row:<{row_w}}"
        for col in all_cols:
            g = cells[row][col]
            if not g:
                parts += f"{'—':>{col_w}}"
            else:
                cnt, _wins, wr, _total, _avg = _group_stats(g)
                parts += f"{f'{cnt} ({wr:.0%})':>{col_w}}"
        print(parts)
    print()


def _print_hold_duration_stats(trades: tuple[TradeDetail, ...]) -> None:
    winners = [td.hold_bars for td in trades if td.pnl_usd > 0]
    losers = [td.hold_bars for td in trades if td.pnl_usd <= 0]
    all_holds = [td.hold_bars for td in trades]

    def _avg(xs: list[int]) -> float:
        return statistics.mean(xs) if xs else 0.0

    def _med(xs: list[int]) -> float:
        return statistics.median(xs) if xs else 0.0

    print("  Hold Duration (bars):")
    print(f"    Winners avg: {_avg(winners):>5.1f}  median: {_med(winners):.0f}")
    print(f"    Losers  avg: {_avg(losers):>5.1f}  median: {_med(losers):.0f}")
    print(f"    Overall avg: {_avg(all_holds):>5.1f}")
    print()


def _print_trade_analysis(trades: tuple[TradeDetail, ...]) -> None:
    if not trades:
        return
    from dojiwick.domain.enums import TradeAction

    print(f"{'=' * 50}")
    print("TRADE ANALYSIS")
    print(f"{'=' * 50}")
    print()
    _print_grouped_breakdown(
        "Close Reason Breakdown",
        trades,
        lambda td: td.close_reason,
        label_width=12,
        show_avg_pnl=True,
    )
    _print_grouped_breakdown("Pair Breakdown", trades, lambda td: td.pair)
    _print_grouped_breakdown("Strategy Breakdown", trades, lambda td: td.strategy_name)
    _print_grouped_breakdown(
        "Direction Breakdown",
        trades,
        lambda td: "Long" if td.action == TradeAction.BUY else "Short",
        label_width=12,
        show_avg_pnl=True,
        show_total_pnl=False,
        sort_keys=("Long", "Short"),
    )
    has_regime = any(td.regime is not None for td in trades)
    if has_regime:
        _print_grouped_breakdown(
            "Regime Breakdown",
            trades,
            lambda td: regime_name(td.regime) if td.regime is not None else "unknown",
            show_avg_pnl=True,
        )

    _print_hold_duration_stats(trades)

    # Cross-tabulations
    print(f"{'=' * 50}")
    print("CROSS-TABULATIONS")
    print(f"{'=' * 50}")
    print()
    _print_cross_tab(
        "Strategy x Close Reason",
        trades,
        lambda td: td.strategy_name,
        lambda td: td.close_reason,
    )
    _print_cross_tab(
        "Direction x Strategy",
        trades,
        lambda td: "Long" if td.action == TradeAction.BUY else "Short",
        lambda td: td.strategy_name,
    )
    if has_regime:
        _print_cross_tab(
            "Regime x Close Reason",
            trades,
            lambda td: regime_name(td.regime) if td.regime is not None else "unknown",
            lambda td: td.close_reason,
        )
        _print_cross_tab(
            "Regime x Strategy",
            trades,
            lambda td: regime_name(td.regime) if td.regime is not None else "unknown",
            lambda td: td.strategy_name,
        )
    print(f"{'=' * 50}\n")


# CLI


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a backtest on historical Binance candle data.")
    from dojiwick.interfaces.cli._shared import add_common_args

    add_common_args(parser)
    parser.add_argument("--trades-csv", type=Path, default=None, help="Export trade details to CSV")
    parser.add_argument("--equity-csv", type=Path, default=None, help="Export equity curve to CSV")
    return parser.parse_args()


async def _run() -> None:
    args = _parse_args()

    from dojiwick.interfaces.cli._shared import build_service, load_settings_and_series

    settings, series, cleanup = await load_settings_and_series(args)

    service = build_service(settings)

    try:
        result = await service.run_with_hysteresis(series)
        summary = result.summary

        print(f"\n{'=' * 50}")
        print("BACKTEST RESULTS")
        print(f"{'=' * 50}")
        print(f"  Trades:            {summary.trades}")
        print(f"  Total PnL:         ${summary.total_pnl_usd:,.2f}")
        print(f"  Win Rate:          {summary.win_rate:.1%}")
        print(f"  Expectancy:        ${summary.expectancy_usd:,.2f}")
        print(f"  Sharpe-like:       {summary.sharpe_like:.3f}")
        print(f"  Daily Sharpe:      {summary.daily_sharpe:.3f}")
        print(f"  Sortino:           {summary.sortino:.3f}")
        print(f"  Calmar:            {summary.calmar:.3f}")
        print(f"  Profit Factor:     {summary.profit_factor:.3f}")
        print(f"  Payoff Ratio:      {summary.payoff_ratio:.3f}")
        print(f"  Max Drawdown:      {summary.effective_max_drawdown_pct:.2f}%")
        print(f"  Max Consec Losses: {summary.max_consecutive_losses}")
        print(f"  Benchmark (B&H):   ${summary.benchmark_pnl_usd:,.2f}")
        print(f"  Config Hash:       {summary.config_hash[:16]}...")
        if settings.backtest.leverage > 1.0:
            print(f"  Leverage:          {settings.backtest.leverage:.1f}x")
        print(f"{'=' * 50}\n")

        if result.monthly_pnl:
            print(f"{'=' * 50}")
            print("MONTHLY P&L")
            print(f"{'=' * 50}")
            for month, pnl in sorted(result.monthly_pnl.items()):
                print(f"  {month}:  ${pnl:,.2f}")
            print(f"{'=' * 50}\n")

        if result.trade_details:
            _print_trade_analysis(result.trade_details)

        if args.trades_csv and result.trade_details:
            with open(args.trades_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "bar_index",
                        "exit_bar_index",
                        "hold_bars",
                        "close_reason",
                        "pair",
                        "strategy",
                        "action",
                        "entry_price",
                        "exit_price",
                        "quantity",
                        "notional_usd",
                        "pnl_usd",
                        "regime",
                        "regime_confidence",
                        "atr_at_entry",
                        "stop_price",
                        "take_profit_price",
                        "strategy_variant",
                    ]
                )
                for td in result.trade_details:
                    writer.writerow(
                        [
                            td.bar_index,
                            td.exit_bar_index,
                            td.hold_bars,
                            td.close_reason,
                            td.pair,
                            td.strategy_name,
                            td.action,
                            td.entry_price,
                            td.exit_price,
                            td.quantity,
                            td.notional_usd,
                            td.pnl_usd,
                            regime_name(td.regime) if td.regime is not None else "",
                            td.regime_confidence,
                            td.atr_at_entry,
                            td.stop_price,
                            td.take_profit_price,
                            td.strategy_variant,
                        ]
                    )
            log.info("wrote %d trades to %s", len(result.trade_details), args.trades_csv)

        if args.equity_csv and summary.portfolio_equity_curve is not None:
            with open(args.equity_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "equity", "drawdown_pct"])
                dd = (
                    summary.portfolio_drawdown_curve
                    if summary.portfolio_drawdown_curve is not None
                    else [0.0] * len(summary.portfolio_equity_curve)
                )
                for i, (eq, d) in enumerate(zip(summary.portfolio_equity_curve, dd)):
                    writer.writerow([i, float(eq), float(d)])
            log.info("wrote equity curve to %s", args.equity_csv)

    finally:
        await cleanup()


def main() -> None:
    from dojiwick.interfaces.cli._shared import setup_env

    setup_env()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
