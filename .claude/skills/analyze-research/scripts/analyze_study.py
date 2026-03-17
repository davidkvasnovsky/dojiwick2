#!/usr/bin/env python3
"""Post-optimization deep analysis for Dojiwick trading engine.

Queries Optuna DB and analyzes trade/equity CSVs to produce a 14-section
diagnostic report with data-driven recommendations.

Usage:
    uv run python .claude/skills/analyze-research/scripts/analyze_study.py \
        --study dojiwick_v26 --trades /tmp/trades.csv --equity /tmp/equity.csv

    # Optuna-only (no backtest CSVs):
    uv run python .claude/skills/analyze-research/scripts/analyze_study.py \
        --study dojiwick_v26 --optuna-only
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-optimization analysis")
    p.add_argument("--study", required=True, help="Optuna study name")
    p.add_argument("--trades", default=None, help="Trade CSV path")
    p.add_argument("--equity", default=None, help="Equity CSV path")
    p.add_argument("--db-url", default="postgresql+psycopg://dojiwick:dojiwick@localhost:5432/dojiwick")
    p.add_argument("--optuna-only", action="store_true", help="Skip backtest analysis")
    p.add_argument("--start-equity", type=float, default=10000.0, help="Starting equity (default: 10000.0)")
    p.add_argument("--compare", default=None, help="Previous study name for comparison")
    return p.parse_args()


def _hline(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ── Section 1: Optuna Study Overview ──────────────────────────────────────


def section_optuna_overview(study) -> None:  # noqa: ANN001
    _hline("1. OPTUNA STUDY OVERVIEW")
    trials = study.trials
    completed = [t for t in trials if t.state.name == "COMPLETE"]
    pruned = [t for t in trials if t.state.name == "PRUNED"]
    scores = sorted([t.value for t in completed], reverse=True)

    print(f"Total: {len(trials)} | Completed: {len(completed)} | Pruned: {len(pruned)}")
    print(f"Prune rate: {len(pruned) / len(trials):.0%}")
    print(f"Top 10: {[round(s, 1) for s in scores[:10]]}")
    print(f"Scores > 0: {sum(1 for s in scores if s > 0)} | = -20: {sum(1 for s in scores if s <= -19.9)}")

    # Pruning validation
    pruned_with_iv = sum(1 for t in pruned if t.intermediate_values)
    print(
        f"Pruned with intermediate values: {pruned_with_iv}/{len(pruned)} ({'OK' if pruned_with_iv == len(pruned) else 'WARNING'})"
    )


# ── Section 2: Top Trial Convergence ─────────────────────────────────────


def section_convergence(study) -> None:  # noqa: ANN001
    _hline("2. TOP 20 PARAM CONVERGENCE")
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    top20 = sorted(completed, key=lambda t: t.value, reverse=True)[:20]
    best = study.best_trial

    for key in sorted(best.params):
        vals = [t.params.get(key, 0) for t in top20]
        print(f"  {key:40s} avg={np.mean(vals):>10.4f} [{min(vals):.3f} - {max(vals):.3f}]")

    prods = [t.params.get("risk_per_trade_pct", 0) * t.params.get("leverage", 1) for t in top20]
    print(f"  {'risk x leverage':40s} avg={np.mean(prods):>10.2f}% [{min(prods):.1f}% - {max(prods):.1f}%]")

    print(f"\nBest trial #{best.number} fold scores:")
    for k, v in sorted(best.intermediate_values.items()):
        print(f"  Fold {k}: {v:.2f}")


# ── Section 3-14: Trade/Equity Analysis ──────────────────────────────────


def _load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["pnl_usd"].abs() > 0.01].copy().sort_values("bar_index")


def section_pnl_validation(real: pd.DataFrame, start_equity: float) -> None:
    _hline("3. PnL VALIDATION")
    returns = real["pnl_usd"] / real["notional_usd"]
    print(f"Trades: {len(real)} | PnL: ${real['pnl_usd'].sum():,.2f} | WR: {(real['pnl_usd'] > 0).mean():.1%}")
    print(f"Per-trade return: mean={returns.mean():.3%} median={np.median(returns):.3%} std={returns.std():.3%}")
    print(f"  Min: {returns.min():.3%} | Max: {returns.max():.3%}")
    print("\nPosition sizes:")
    for pct in [25, 50, 75, 90, 95, 99]:
        print(f"  P{pct}: ${np.percentile(real['notional_usd'], pct):>12,.2f}")
    print(f"  Max:  ${real['notional_usd'].max():>12,.2f}")
    for thresh in [500_000, 1_000_000, 2_000_000]:
        n = (real["notional_usd"] > thresh).sum()
        pnl = real[real["notional_usd"] > thresh]["pnl_usd"].sum()
        print(f"  > ${thresh / 1e6:.0f}M: {n} trades, PnL=${pnl:,.2f}")

    # Compounding trace
    equity = start_equity
    print("\nCompounding:")
    for i, (_, row) in enumerate(real.iterrows()):
        equity += row["pnl_usd"]
        if i + 1 in (50, 100, 200, 500, 1000) or i + 1 == len(real):
            print(f"  After {i + 1:>5} trades: ${equity:>14,.2f}")


def section_monthly(real: pd.DataFrame) -> None:
    _hline("4. MONTHLY PERFORMANCE")
    real = real.copy()
    real["month"] = real["bar_index"] // 730
    print(f"{'Month':<10} {'Trades':>6} {'PnL':>14} {'AvgNot':>12} {'WR':>5}")
    for m, g in real.groupby("month"):
        yr = 2022 + int(m) // 12
        mo = int(m) % 12 + 1
        if yr > 2026:
            break
        wr = (g["pnl_usd"] > 0).mean()
        flag = " !" if wr < 0.4 or wr > 0.8 else ""
        print(
            f"{yr}-{mo:02d}    {len(g):>6} ${g['pnl_usd'].sum():>13,.2f} ${g['notional_usd'].mean():>11,.2f} {wr:>4.0%}{flag}"
        )


def section_exit_types(real: pd.DataFrame) -> None:
    _hline("5. EXIT TYPE PERFORMANCE")
    for reason in ["trailing_stop", "take_profit", "partial_tp", "stop_loss", "time_exit"]:
        g = real[real["close_reason"] == reason]
        if len(g) == 0:
            continue
        wr = (g["pnl_usd"] > 0).mean()
        print(
            f"  {reason:15s} n={len(g):>4} WR={wr:.0%} avg=${g['pnl_usd'].mean():>10,.2f} total=${g['pnl_usd'].sum():>12,.2f} hold={g['hold_bars'].mean():.0f}bars"
        )


def section_stop_loss(real: pd.DataFrame) -> None:
    _hline("6. STOP LOSS DEEP DIVE")
    stops = real[real["close_reason"] == "stop_loss"]
    wins = real[real["pnl_usd"] > 0]
    print(f"Stops: {len(stops)}/{len(real)} = {len(stops) / len(real):.0%}")
    print(f"Avg loss: ${stops['pnl_usd'].mean():,.2f} | Median: ${stops['pnl_usd'].median():,.2f}")
    if len(wins) > 0:
        print(f"Risk asymmetry: avg_loss/avg_win = {abs(stops['pnl_usd'].mean() / wins['pnl_usd'].mean()):.2f}x")

    print("\nStop timing (bad entry detection):")
    for bars in [1, 2, 3, 5, 8]:
        quick = stops[stops["hold_bars"] <= bars]
        pct = len(quick) / len(stops) * 100 if len(stops) > 0 else 0
        print(f"  Within {bars} bars: {len(quick):>3} ({pct:.0f}%) loss=${quick['pnl_usd'].sum():>12,.2f}")

    print("\nStops by strategy:")
    for s, g in stops.groupby("strategy"):
        total_s = len(real[real["strategy"] == s])
        print(f"  {s:20s} {len(g):>3}/{total_s} ({len(g) / total_s:.0%})")

    if "regime" in stops.columns:
        print("\nStops by regime:")
        for r, g in stops.groupby("regime"):
            print(f"  {str(r):20s} {len(g):>3} loss=${g['pnl_usd'].sum():>12,.2f}")


def section_strategy(real: pd.DataFrame) -> None:
    _hline("7. STRATEGY BREAKDOWN")
    for s, g in real.groupby("strategy"):
        wr = (g["pnl_usd"] > 0).mean()
        print(f"  {s:20s} n={len(g):>4} WR={wr:.0%} PnL=${g['pnl_usd'].sum():>12,.2f}")

    print("\nDirection:")
    for act in sorted(real["action"].unique()):
        g = real[real["action"] == act]
        d = "Long" if act == 1 else "Short"
        wr = (g["pnl_usd"] > 0).mean()
        print(f"  {d:6s} n={len(g):>4} WR={wr:.1%} PnL=${g['pnl_usd'].sum():>12,.2f}")


def section_regime(real: pd.DataFrame) -> None:
    _hline("8. REGIME PERFORMANCE")
    if "regime" not in real.columns:
        print("  No regime data")
        return
    for r, g in real.groupby("regime"):
        wr = (g["pnl_usd"] > 0).mean()
        print(f"  {str(r):20s} n={len(g):>4} WR={wr:.0%} PnL=${g['pnl_usd'].sum():>12,.2f}")


def section_drawdown(real: pd.DataFrame, equity_df: pd.DataFrame) -> None:
    _hline("9. DRAWDOWN ANALYSIS (from equity CSV — includes unrealized P&L)")
    # Use the backtest's per-bar DD which includes unrealized intra-trade drawdowns
    max_dd = float(equity_df["drawdown_pct"].max())
    max_dd_idx = int(equity_df["drawdown_pct"].idxmax())
    print(f"Max DD: {max_dd:.2f}% (at row {max_dd_idx})")

    # DD events > 10% from equity CSV, correlated with trade data
    dd_vals = equity_df["drawdown_pct"].values
    in_dd = False
    dd_start = 0
    events = 0
    for i in range(len(dd_vals)):
        if dd_vals[i] >= 10 and not in_dd:
            in_dd = True
            dd_start = i
        elif dd_vals[i] < 3 and in_dd:
            in_dd = False
            mx = float(dd_vals[dd_start:i].max())
            # Correlate with trades in this period
            period_trades = real[(real.index >= dd_start) & (real.index < i)]
            stops_in = (period_trades["close_reason"] == "stop_loss").sum() if len(period_trades) > 0 else 0
            wr_in = (period_trades["pnl_usd"] > 0).mean() if len(period_trades) > 0 else 0
            events += 1
            print(
                f"  DD {mx:.1f}%: rows {dd_start}-{i} ({i - dd_start} bars) {len(period_trades)} trades WR={wr_in:.0%} stops={stops_in}"
            )
    if in_dd:
        mx = float(dd_vals[dd_start:].max())
        period_trades = real[real.index >= dd_start]
        stops_in = (period_trades["close_reason"] == "stop_loss").sum() if len(period_trades) > 0 else 0
        wr_in = (period_trades["pnl_usd"] > 0).mean() if len(period_trades) > 0 else 0
        events += 1
        print(
            f"  DD {mx:.1f}%: rows {dd_start}-{len(dd_vals)} {len(period_trades)} trades WR={wr_in:.0%} stops={stops_in} [ongoing]"
        )
    print(f"  Total DD events > 10%: {events}")


def section_streaks(real: pd.DataFrame) -> None:
    _hline("10. CONSECUTIVE LOSS STREAKS")
    streaks: list[int] = []
    cur = 0
    for pnl in real["pnl_usd"]:
        if pnl < 0:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    arr = np.array(streaks) if streaks else np.array([0])
    wr = (real["pnl_usd"] > 0).mean()
    print(f"Max streak: {arr.max()} | Avg: {arr.mean():.1f}")
    for n in [3, 5, 7, 10]:
        cnt = (arr >= n).sum()
        prob = (1 - wr) ** n * 100
        print(f"  Streaks >= {n}: {cnt} (theoretical P={prob:.3f}%)")


def section_pairs(real: pd.DataFrame) -> None:
    _hline("11. PAIR PERFORMANCE")
    for p, g in real.groupby("pair"):
        wr = (g["pnl_usd"] > 0).mean()
        print(f"  {p:12s} n={len(g):>4} WR={wr:.0%} PnL=${g['pnl_usd'].sum():>12,.2f}")

    # Simultaneous stop events
    stops = real[real["close_reason"] == "stop_loss"]
    multi = stops.groupby("bar_index").size()
    double = (multi >= 2).sum()
    print(f"\nSimultaneous stops (same bar, 2+ pairs): {double}")


def section_sizing(real: pd.DataFrame) -> None:
    _hline("12. POSITION SIZING & COMPOUNDING")
    # Growth by half-year
    real = real.copy()
    real["half"] = real["bar_index"] // (730 * 6)
    for h, g in real.groupby("half"):
        yr = 2022 + int(h) // 2
        half = "H1" if int(h) % 2 == 0 else "H2"
        print(f"  {yr}-{half}: {len(g):>4} trades, avg_notional=${g['notional_usd'].mean():>12,.2f}")


def section_feasibility(real: pd.DataFrame) -> None:
    _hline("13. LIVE TRADING FEASIBILITY")
    n_months = max(1, (real["bar_index"].max() - real["bar_index"].min()) / 730)
    print(f"Max position: ${real['notional_usd'].max():,.2f}")
    print(f"Avg position: ${real['notional_usd'].mean():,.2f}")
    print(f"Trades/month: {len(real) / n_months:.1f}")
    print(f"Avg hold: {real['hold_bars'].mean():.1f} bars ({real['hold_bars'].mean() / 24:.1f} days)")
    est_cost = real["notional_usd"].sum() * 0.001  # ~10bps round-trip (fee+slippage)
    gross_wins = real[real["pnl_usd"] > 0]["pnl_usd"].sum()
    gross_losses = abs(real[real["pnl_usd"] < 0]["pnl_usd"].sum())
    print(f"Est round-trip cost: ${est_cost:,.2f} (~10bps per trade)")
    print(f"Gross wins: ${gross_wins:,.2f} | Gross losses: ${gross_losses:,.2f}")
    print(f"Net PnL: ${real['pnl_usd'].sum():,.2f} (costs already deducted by backtest)")


def section_recommendations(real: pd.DataFrame, max_dd: float) -> None:
    _hline("14. RECOMMENDATIONS")
    flags: list[str] = []
    wr = (real["pnl_usd"] > 0).mean()
    pf = (
        abs(real[real["pnl_usd"] > 0]["pnl_usd"].sum() / real[real["pnl_usd"] < 0]["pnl_usd"].sum())
        if (real["pnl_usd"] < 0).any()
        else 99
    )

    stops = real[real["close_reason"] == "stop_loss"]
    one_bar_stops = stops[stops["hold_bars"] <= 1]
    one_bar_pct = len(one_bar_stops) / len(stops) * 100 if len(stops) > 0 else 0

    if max_dd > 20:
        flags.append(f"DD {max_dd:.1f}% > 20% -> Lower max_loss_per_trade_pct or leverage")
    if wr < 0.60:
        flags.append(f"WR {wr:.0%} < 60% -> Increase min_confluence_score or min_confidence")
    if one_bar_pct > 20:
        flags.append(f"1-bar stops {one_bar_pct:.0f}% > 20% -> Widen adaptive_volatile_stop_scale")
    if pf < 1.5:
        flags.append(f"Profit factor {pf:.2f} < 1.5 -> Review stop/TP ratio")

    # Direction balance
    if "action" in real.columns:
        longs = (real["action"] == 1).sum()
        shorts = (real["action"] == -1).sum()
        ratio = max(longs, shorts) / max(min(longs, shorts), 1)
        if ratio > 3:
            flags.append(f"Direction imbalance {longs}L:{shorts}S ({ratio:.1f}:1) -> Check search space")

    # Regime WR
    if "regime" in real.columns:
        for r, g in real.groupby("regime"):
            rwr = (g["pnl_usd"] > 0).mean()
            if rwr < 0.50 and len(g) >= 20:
                flags.append(f"Regime '{r}' WR={rwr:.0%} < 50% ({len(g)} trades) -> Consider regime filter")

    if not flags:
        print("  No issues detected.")
    else:
        for f in flags:
            print(f"  ! {f}")


def section_comparison(current_study, compare_study) -> None:  # noqa: ANN001
    _hline("15. COMPARISON WITH PREVIOUS STUDY")
    cur_completed = [t for t in current_study.trials if t.state.name == "COMPLETE"]
    prev_completed = [t for t in compare_study.trials if t.state.name == "COMPLETE"]

    cur_best = current_study.best_value
    prev_best = compare_study.best_value
    cur_scores = sorted([t.value for t in cur_completed], reverse=True)
    prev_scores = sorted([t.value for t in prev_completed], reverse=True)

    print(f"  {'Metric':<25} {'Previous':>12} {'Current':>12} {'Change':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'Best score':<25} {prev_best:>12.2f} {cur_best:>12.2f} {cur_best - prev_best:>+10.2f}")
    print(
        f"  {'Completed trials':<25} {len(prev_completed):>12} {len(cur_completed):>12} {len(cur_completed) - len(prev_completed):>+10}"
    )
    print(
        f"  {'Median score':<25} {np.median(prev_scores):>12.2f} {np.median(cur_scores):>12.2f} {np.median(cur_scores) - np.median(prev_scores):>+10.2f}"
    )
    print(f"  {'Scores > 0':<25} {sum(1 for s in prev_scores if s > 0):>12} {sum(1 for s in cur_scores if s > 0):>12}")


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=args.study, storage=args.db_url)

    _hline(f"DOJIWICK POST-OPTIMIZATION ANALYSIS: {args.study}")
    section_optuna_overview(study)
    section_convergence(study)

    if args.optuna_only:
        print("\n[Optuna-only mode — skipping backtest analysis]")
        return

    if not args.trades or not args.equity:
        print("\nERROR: Both --trades and --equity are required for backtest analysis.")
        print("Run backtest with: --trades-csv /tmp/trades.csv --equity-csv /tmp/equity.csv")
        return

    real = _load_trades(args.trades)
    equity_df = pd.read_csv(args.equity)

    section_pnl_validation(real, args.start_equity)
    section_monthly(real)
    section_exit_types(real)
    section_stop_loss(real)
    section_strategy(real)
    section_regime(real)

    # Get REAL DD from equity CSV (includes unrealized intra-trade drawdowns)
    max_dd = float(equity_df["drawdown_pct"].max())

    section_drawdown(real, equity_df)
    section_streaks(real)
    section_pairs(real)
    section_sizing(real)
    section_feasibility(real)
    section_recommendations(real, max_dd)

    if args.compare:
        compare_study = optuna.load_study(study_name=args.compare, storage=args.db_url)
        section_comparison(study, compare_study)


if __name__ == "__main__":
    main()
