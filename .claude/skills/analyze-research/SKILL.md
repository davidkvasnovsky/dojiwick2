---
name: analyze-research
description: >
  This skill should be used when the user asks to "analyze research", "analyze optimization",
  "analyze study", "post-optimization analysis", "analyze backtest results", "what went wrong",
  "why is DD high", "why low win rate", "analyze v26", "compare studies", "why less profit with
  more pairs", "check drawdown", "portfolio DD", "daily sharpe", "use pandas", or wants deep
  analysis of Optuna optimization results and backtest performance. Trigger after any optimization
  run completes or after running a backtest with optimized params.
---

# Analyze Research

Deep analysis of Optuna optimization studies and backtest results for the Dojiwick trading engine. Produces a 14-section diagnostic report, followed by targeted pandas analysis and data-driven recommendations.

## Prerequisites

- Completed Optuna study in PostgreSQL (study name from `config.toml`)
- Trade CSV from backtest (`--trades-csv /tmp/trades.csv`)
- Equity CSV from backtest (`--equity-csv /tmp/equity.csv`)

## Workflow

### Step 1: Determine Study Parameters

Read `config.toml` to get:
- `study_name` from `[optimization]` section
- `dsn` from `[database]` section (rewrite `postgres` → `localhost` for host access)
- Optuna storage URL: derive from DSN with `postgresql+psycopg://` scheme

When a study name is provided as argument, use it instead of the config value.

### Step 2: Run Analysis Script

Execute the analysis script with `uv run`:

```bash
uv run python .claude/skills/analyze-research/scripts/analyze_study.py \
  --study <study_name> \
  --trades /tmp/trades.csv \
  --equity /tmp/equity.csv
```

The script connects to the Optuna database and reads trade/equity CSVs. Both `--trades` and `--equity` are required for full analysis. `--optuna-only` skips backtest sections entirely.

### Step 3: Run Targeted Pandas Analysis

The script covers 14 sections but uses per-trade metrics that inflate for multi-pair systems. Run ad-hoc pandas for accurate portfolio-level analysis. See **`references/pandas-snippets.md`** for reusable code.

**Critical analyses to run:**

- **Portfolio DD**: compute actual `avg_equity` peak-to-trough DD — the script's DD is per-trade cumprod which inflates ~5.8× for 6-pair systems
- **Daily Sharpe**: group bar PnL into days, `mean/std * sqrt(365)` — `sharpe_like` overstates by ~40-45% for multi-pair due to correlated same-bar trades
- **Per-regime PF**: `groupby('regime')` to catch strategy drags (e.g., mean_revert PF=0.91 losing $477k)
- **Per-pair PF**: `groupby('pair')` to verify all pairs contribute positively
- **Simultaneous stops**: `groupby('exit_bar_index').size()`, count bars with 2+ stops and total correlated loss
- **Kelly criterion**: `p - (1-p)/b` to check if current risk×leverage is near half-Kelly optimal
- **Shock test**: multiply winning PnL by 0.9, losing by 1.1, check PF > 1.05

**For cross-study comparison** (e.g., v26 vs v34): load both trade CSVs and compare PnL, WR, avg position size, risk×leverage, regime breakdown side-by-side.

### Step 4: Query Optuna DB Directly

For deeper analysis beyond the script, query the Optuna DB:

```python
import optuna
storage = 'postgresql+psycopg://dojiwick:dojiwick@localhost:5432/dojiwick'
study = optuna.load_study(study_name='dojiwick_vXX', storage=storage)
completed = [t for t in study.trials if t.state.name == 'COMPLETE']
top20 = sorted(completed, key=lambda t: t.value or -999, reverse=True)[:20]
```

Check param convergence (tight ranges = optimizer confident), fold score spread (high spread = inconsistent), and whether key params hit search space bounds.

### Step 5: Present Findings

Present the most critical findings first:

1. **Red flags**: portfolio DD > 25%, per-regime PF < 1.0, daily Sharpe < 1.5, PF < 1.2
2. **Key metrics**: trades, PnL, WR, portfolio DD, daily Sharpe, profit factor
3. **Measurement caveats**: note that `sharpe_like` is per-trade annualized (inflated), `max_drawdown_pct` is per-trade cumprod (inflated). Use `daily_sharpe` and `max_portfolio_drawdown_pct` for honest reporting
4. **Detailed breakdowns**: by strategy, regime, pair, exit type
5. **Recommendations**: specific config/code changes with data evidence

### Step 6: Propose Fixes and Iterate

Based on analysis, propose specific changes:
- Config changes (leverage, risk caps, strategy params)
- Search space adjustments (wider/tighter bounds, per-regime params)
- Architecture changes (if structural issues found, e.g., DD metric, fair allocation)
- Strategy enable/disable decisions (if a strategy has PF < 1.0)

After implementing fixes, re-run optimization → backtest → re-analyze. Compare results against previous study to verify improvement. Track version progression with key metric deltas.

## Analysis Sections Overview

| # | Section | Key Metrics |
|---|---------|-------------|
| 1 | Optuna Overview | Trials, scores, pruning rate |
| 2 | Param Convergence | Top 20 params, risk×leverage |
| 3 | PnL Validation | Returns, position sizes, compounding |
| 4 | Monthly Performance | PnL consistency, WR outliers |
| 5 | Exit Types | Which exits make/lose money |
| 6 | Stop Loss Deep Dive | Timing, regime, risk asymmetry |
| 7 | Strategy Breakdown | PnL/WR by strategy, direction |
| 8 | Regime Performance | PnL/WR by regime, cross-tabs |
| 9 | Drawdown Analysis | DD events, causes, budget math |
| 10 | Loss Streaks | Consecutive losses, probability |
| 11 | Pair Performance | Per-pair PnL, correlation |
| 12 | Sizing & Compounding | Notional growth, Kelly criterion |
| 13 | Live Feasibility | Liquidity, slippage, costs |
| 14 | Recommendations | Auto-flagged issues with fixes |

## Recommendation Flags

The script auto-generates flags based on thresholds:

- `DD > 20%` → Check if this is portfolio DD or per-trade DD. Portfolio DD > 25% is concerning; per-trade DD > 50% is normal for multi-pair
- `WR < 60%` → Increase `min_confluence_score` or `min_confidence`
- `1-bar stops > 20%` → Entry timing issue, consider confirmation candle
- `Direction ratio > 3:1` → Check search space for missing directional params
- `Positions > $2M` → Lower `max_notional_usd`
- `Profit factor < 1.5` → Review stop/TP ratio and per-regime PF
- `Regime PF < 1.0` → Strategy in that regime is a net loser, consider disabling

## Additional Resources

### Script
- **`scripts/analyze_study.py`** — Complete pandas analysis (run with `uv run`)

### Reference Files
- **`references/analysis-checklist.md`** — Detailed explanation of each metric, healthy ranges, and common failure patterns
- **`references/pandas-snippets.md`** — Reusable pandas/numpy code for 15 analysis types (portfolio DD, daily Sharpe, cross-study comparison, shock test, Kelly, correlation, etc.)
