# v38 Candidate Evaluation & Promotion Report

**Date:** 2026-07-11
**Decision:** Promoted **v38 trial #7109** (`research/config2.toml`) to production `config.toml`, replacing v34 trial #8385. A fresh 10k-trial **v39 study** was then run on data through 2026-07-10; its best gate-passing candidate (#7269) did not beat #7109 risk-adjusted, so **#7109 remains the champion**.

## Setup

- Optuna store was wiped in the July Docker rebuild — no stored trial scores survived. Ranking is purely empirical re-testing.
- Candle cache truncated and re-warmed from Binance (355k hourly bars, 6 pairs, zero missing bars, listing dates through 2026-07-10) because of the `CachingCandleFetcher` partial-hit bug (see follow-ups).
- **Windows:** full history 2019-09-08 → 2026-07-10; true OOS holdout 2026-03-17 → 2026-07-10 (v38 study trained through 2026-03-17; backtests start 2026-03-09 so the 200-bar indicator warmup ends exactly at the study boundary — every OOS trade is data no trial ever saw).
- Honest metrics per CLAUDE.md: `daily_sharpe` + portfolio `Max Drawdown` (`sharpe_like` overstates ~40-45% on 6 pairs).

## Backtest matrix

### Full window (2019-09-08 → 2026-07-10)

| Config | Trades | Daily Sharpe | Max DD | PF | Win rate | Total PnL | Sortino | Calmar |
|---|---|---|---|---|---|---|---|---|
| prod v34 #8385 | 8,562 | 3.196 | 24.94% | 1.409 | 70.7% | $44.3M | 6.13 | 9.63 |
| c1 #5320 | 13,397 | 4.237 | 12.75% | 1.524 | 80.3% | $120.7M | 8.17 | 18.43 |
| **c2 #7109** | 14,202 | 4.062 | 12.38% | 1.524 | 78.9% | $123.9M | 9.35 | 18.91 |
| c3 #3 | 16,865 | 4.237 | 10.78% | 1.472 | 77.5% | $119.9M | 9.91 | 19.64 |

(Benchmark B&H over the window: $51.8k on $10k equity. PnL figures are compounded backtest sums — treat relatively, not as live projections.)

### OOS holdout (2026-03-17 → 2026-07-10, market B&H −21%)

| Config | Trades | Daily Sharpe | Max DD | PF | Total PnL |
|---|---|---|---|---|---|
| prod v34 #8385 | 508 | −0.865 | 23.51% | 0.898 | −$7,194 |
| c1 #5320 | 508 | −1.178 | 17.31% | 0.877 | −$5,870 |
| **c2 #7109** | 514 | **−0.087** | **16.25%** | **0.971** | **−$1,459** |
| c3 #3 | 688 | −1.269 | 15.11% | 0.891 | −$6,090 |

The holdout was a bear/chop regime (April 2026 was a heavy loss month for every config). #7109 was near-flat while production lost ~5× more; its OOS trades were balanced long/short (281/219 trend_follow), which is what kept it alive in the downtrend.

## Research gate (study window 2019-09-08 → 2026-03-17)

| Criterion | Threshold | c1 #5320 | c2 #7109 |
|---|---|---|---|
| Passed (all 9) | — | **True** | **True** |
| CV Sharpe | ≥ 0.50 | 6.702 | 6.780 |
| PBO | ≤ 0.40 | 0.000 | 0.000 |
| OOS/IS ratio | ≥ 0.50 | 0.990 | 0.995 |
| Agg WF OOS Sharpe | ≥ 0.20 | 6.590 | 6.779 |

20 walk-forward windows each, every window's OOS Sharpe positive (c2 range: 2.6–9.95). No rejection reasons. (c3 not gated — eliminated on OOS rank; prod is the incumbent.)

## Decision

Selection rule (agreed up front): gate pass → rank by OOS daily_sharpe → OOS & full-window DD ≤ 25%, PF > 1 → must beat incumbent.

- #7109 passes gate, ranks first OOS by a wide margin (−0.09 vs −0.87/−1.18/−1.27), has 514 OOS trades (noise guard satisfied), and beats production on the full window too (4.06 vs 3.20 daily Sharpe, half the drawdown).
- #5320 and #3 lose to the incumbent out-of-sample despite great full-window numbers — textbook overfit-looking full-window winners; correctly filtered by the holdout.
- Promoting a config with slightly negative OOS Sharpe is justified: the whole market fell 21% in the holdout; #7109 preserved capital (PF 0.97, −$1.5k) where everything else bled. Sitting on v34 would have been the worst choice on every metric.

## Verification of promotion

- `cp research/config2.toml config.toml`; diff audited — only tuned params changed (base strategy/regime params, 3 risk sizing caps, leverage, 4 auto scope blocks + provenance comment). `[database]/[optimization]/[research]/[ai]/[universe]/[[scope.risk]]` untouched.
- `fingerprint_settings` equality vs `research/config2.toml`: **match** (sha256 `6b9a9793…`).
- Deterministic OOS re-run on promoted config: **identical** output to the c2 run, byte-for-byte.
- `dojiwick explain` resolves `auto_trending_up` to #7109 values (stop 2.492 / RR 2.823 / hold 53).
- `make ci`: **green** (fmt, ruff, basedpyright strict, 753 unit tests).

## v39 study — completed, champion retained

Ran on Hetzner CCX63 (48 workers, ~1.5 h): **10,000 trials** on 2019-09-08 → 2026-07-10 (full window incl. the 2026 bear), streamed into local postgres (`study_name = "dojiwick_v39"`). First launch attempt failed on SSH (ssh-agent offered unrelated keys until the server cut auth — fixed by pinning `-o IdentitiesOnly=yes -i ~/.ssh/id_ed25519`; worth upstreaming into `scripts/hetzner-optimize.sh`). Server auto-deleted after both attempts.

**Candidate chain:** best trial #9671 (score 61.2) and runner-up #9706 (59.7) both **failed the gate** on the same criterion — ranging-regime PF 0.79/0.88 < 0.95 floor. Third-best **#7269** (59.6) **passed all 9 criteria**. Its params were materialized into `research/config_v39_7269.toml` (patch script mirrors `apply_params`; fingerprint-verified identical to the gate-evaluated artifact).

**Head-to-head on the identical full window** (2019-09-08 → 2026-07-10; #7109 gate re-run on this window for fairness):

| Metric | #7109 (production) | #7269 (v39) |
|---|---|---|
| Gate | PASS | PASS |
| CV Sharpe | 6.469 | 6.490 |
| WF OOS Sharpe (21 windows) | 6.811 | 6.731 |
| PBO | 0.000 | 0.000 |
| Backtest daily Sharpe | **4.062** | 3.382 |
| Max Drawdown | **12.38%** | 17.70% |
| Profit Factor | **1.524** | 1.448 |
| Total PnL (compounded) | $123.9M | $181.7M |
| risk_per_trade / leverage | 2.47% / 2.85 | 3.93% / 2.97 |

#7269 makes more raw PnL purely through bigger position sizing, but is worse on every risk-adjusted measure — despite having trained on the full window while #7109 never saw anything past 2026-03-17. Under the agreed risk-adjusted selection rule, **#7109 stays in production**. #7269 is archived as the v39 gate-passing runner-up (`research/config_v39_7269.toml`, `research/params_v39_trial7269.json`) — a candidate if risk appetite ever shifts toward raw PnL.

Config note: `config.toml` keeps `study_name = "dojiwick_v39"` (matches the completed study in the local store; bump to v40 for the next search).

## v40 study + engine-shift halt (2026-07-11, late session)

**v40 ran and validated the methodology fixes:** 10,000 trials on the train window 2019-09-08 → 2026-04-01 (Apr–Jul reserved as untouched holdout), with the new per-regime PF objective penalty and the volatile entry-throttle search dims. **Best trial #6857 passed the research gate on the first attempt** (CV 6.68, PBO 0.00, WF OOS Sharpe 6.74) — versus v39 where the top two trials failed on ranging PF the objective couldn't see. The optimizer's volatile solution: mild trend-entry throttle (`scope_volatile__trend_breakout_adx_min` 26.7, `trend_pullback_adx_min` 19.3) plus much looser vol_revert triggers (`vol_extreme` 44.5/55.3 vs 33.3/62.7) and tight volatile exits.

**Promotion was halted by an engine change, not by the candidate.** During the session the backtest engine gained liquidation modeling, funding accrual, and gap-through fills. Controlled replay (identical #7109 config, identical 2026-03-09→07-10 window, identical cached candles): old engine daily Sharpe −0.09 / DD 16.3%, new engine **−5.82 / DD 67.8%**. At ~2.9× leverage the new engine liquidates positions the old physics let survive. v40 #6857 fares worse still on the tail under new physics (DD 87%).

Consequences:
- Every optimization to date (v38/v39/v40) was scored under the old physics. Under the new, more realistic engine none of those parameter sets is deployable at current leverage.
- `config.toml` stays on v38 #7109 with `study_name = "dojiwick_v40"`; no new promotion.
- v40 artifacts preserved: `research/params_v40_trial{6857,8488,7782,9478}.json`, `research/config_v40_6857.toml` (fingerprint-verified), gate output in `research/results/v40_hetzner.log`.

**Next step (v41), once the engine changes are finalized and committed:** re-run the same disciplined pipeline — train to T−3 months, per-regime PF penalty, volatile throttle dims — under the new physics. Expect materially lower leverage/risk to win (liquidation is now lethal), and treat all pre-v41 numbers as optimistic-physics history.

## Follow-ups

1. ~~Fix `CachingCandleFetcher` partial-hit bug~~ **FIXED (2026-07-11):** cache now returns data only when it covers both range boundaries (`_covers_range` + new `interval_to_seconds` in `domain/timebase.py`); partial overlap triggers a full refetch. Regression tests added; verified on the real path (post-fix OOS re-run: 6/6 cache hits, metrics identical). Interior exchange-maintenance gaps are still tolerated by design.
2. ~~Upstream ssh identity pin~~ **FIXED (2026-07-11):** `scripts/hetzner-optimize.sh` now pins `IdentitiesOnly=yes` with `--identity <path>` (default `~/.ssh/id_ed25519`) for all ssh/scp calls and preflight-checks the file — first v39 launch died on ssh-agent key spray.
3. **`.venv` note:** repo `.venv` is shared between host (macOS arm64) and tooling container (Linux). Container runs currently work; if either side breaks after the other syncs, re-run `uv sync` on that side.
4. **Cadence:** the OOS holdout shrinks to zero the moment v39 trains through 2026-07. Going forward, promote on gate + accumulated fresh OOS; re-evaluate the champion monthly as new data arrives.
5. April-2026-style loss months hit every config; consider regime-aware risk throttle research (drawdown scaling already helps: #7109 OOS DD 16% vs prod 24%).
6. **v40 objective/gate alignment:** v39's top two trials failed the gate on ranging-regime PF — add per-regime PF penalty to the optimizer objective so the search converges on gate-passable optima; consider training only to T-3 months to preserve a rolling true holdout.
