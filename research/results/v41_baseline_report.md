# v41 baseline: promoted v38 #7109 params under the post-overhaul engine

2022-01-01 → 2026-03-01, config.toml as promoted (leverage 2.9x), engine at
commit `5a5c3c4`. Artifacts: `v41_baseline_trades.csv`, `v41_baseline_equity.csv`,
`v41_baseline_analysis.txt` (14-section diagnostic).

## Before / after

| Metric | Pre-overhaul engine | Post-overhaul engine |
|---|---|---|
| Daily Sharpe | ~4.06 | **−0.419** |
| Total PnL ($10k start) | strongly positive | **−$8,599** (equity ends $1,401) |
| Max portfolio DD | ~12.4% | **94.29%** |
| Profit factor | > 1.3 | **0.946** |
| Buy & hold benchmark | — | +$1,245 |

Same params, same window, same data. The entire difference is engine honesty:
fees/funding on leveraged notional (was ~2.85× understated), one shared equity
pool (was 6 per-pair wallets), two-sided slippage, gap-through fills,
liquidation modeling, signed historical funding, exit-loop bar alignment.

## Where the money goes

- Estimated round-trip costs ≈ **$7.6k** on $308k gross volume traded — most of
  the net loss. 8,369 trades at 6.2-bar average hold is a cost machine.
- Exits: trailing (+$98.9k, 88% WR) and TP1 (+$45.5k) are profitable;
  **stop-losses (−$149.4k, avg −$98.84) erase them**. Risk asymmetry
  avg_loss/avg_win = 4.18×.
- 67% of stopped trades stop within 8 bars — entries, not exits, are the leak.
- Volatile regime is the only profitable one (+$2.5k); trending regimes lose.
- Sizing collapses with the pool (avg notional $2.8k in 2022-H2 → $230 by
  2026): the strategy digs a hole early and never recovers (94% DD reached and
  still ongoing at window end).

## Verdict

**v38/v39/v40 params are not deployable.** They were optimized into cost and
loss-asymmetry traps the old engine could not see. Next step: v41 study on
Hetzner under the honest engine (`study_name = "dojiwick_v41"` already set) —
expect lower leverage, wider stops or fewer trades, and per-regime throttles to
win. Gate + holdout OOS before any promotion, compared against this baseline.
