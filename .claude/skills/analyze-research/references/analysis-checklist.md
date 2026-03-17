# Analysis Checklist — Detailed Reference

## Healthy Metric Ranges

| Metric | Healthy | Warning | Critical |
|---|---|---|---|
| Win Rate | > 60% | 50-60% | < 50% |
| Profit Factor | > 1.5 | 1.2-1.5 | < 1.2 |
| Portfolio DD | < 20% | 20-30% | > 30% |
| Per-trade DD | informational only — inflates ~5.8× for multi-pair systems |||
| Daily Sharpe | > 2.0 | 1.0-2.0 | < 1.0 |
| Sharpe-like (per-trade) | informational — overstates ~40-45% for multi-pair |||
| Sortino | > 3.0 | 1.5-3.0 | < 1.5 |
| Payoff Ratio | > 1.0 | 0.7-1.0 | < 0.7 |
| 1-Bar Stop % | < 15% | 15-25% | > 25% |
| Prune Rate | 60-80% | 40-60% | > 90% |
| Trades/Month | > 15 | 8-15 | < 8 |
| Max Consec Losses | < 6 | 6-8 | > 8 |

## Common Failure Patterns

### Pattern: Death Spiral
**Symptom**: Very few trades in continuous backtest despite many in walk-forward windows.
**Cause**: State-dependent filters (ECF, DD halt) block entries permanently.
**Fix**: Replace binary blockers with proportional scalers.

### Pattern: Scope Override Conflict
**Symptom**: Optimizer's max_hold_bars is ignored; trades held much longer.
**Cause**: `[[scope.strategy]]` blocks override optimized params.
**Fix**: Remove static scope overrides; let optimizer control all params.

### Pattern: Risk Asymmetry
**Symptom**: Avg loss > avg win despite positive WR.
**Cause**: Wide stops with short holds; stop is a catastrophe exit.
**Fix**: Add payoff_ratio_weight to objective; tighten stop_atr_mult range.

### Pattern: Direction Imbalance
**Symptom**: 90%+ of trend_follow trades are shorts (or longs).
**Cause**: trend_overbought_rsi_min not in search space.
**Fix**: Add missing params to search space.

### Pattern: Volatile Regime Bleed
**Symptom**: WR < 50% in volatile regime; most stops happen there.
**Cause**: adaptive_volatile_stop_scale < 1.0 TIGHTENS stops in volatile conditions.
**Fix**: Widen search range to (0.8, 1.5).

### Pattern: Capital Starvation (Multi-Pair)
**Symptom**: 6 pairs produce LESS PnL than 2 pairs despite more trades and same WR.
**Cause**: Fixed `max_portfolio_risk_pct` shared by all pairs. First-come-first-served allocation starves later pairs. Position sizes 5× smaller than needed.
**Fix**: Scale portfolio risk by `n_pairs / baseline_pairs`. Use two-pass fair allocation (collect all entries, then scale proportionally). Add risk caps to search space.

### Pattern: DD Metric Inflation
**Symptom**: Research gate rejects with 70%+ DD, but actual portfolio equity only dropped 12%.
**Cause**: `summarize.py` computes DD from per-trade cumprod equity curve. For multi-pair systems, loss clustering inflates DD ~5.8× vs actual portfolio DD.
**Fix**: Use `max_portfolio_drawdown_pct` (tracked from bar-level `avg_equity`) in objective and gate, not `max_drawdown_pct` from summarize.

### Pattern: Mean Revert Drag
**Symptom**: Mean_revert has 64% WR and +$3M trailing stop wins, but still loses -$477k net.
**Cause**: Stop losses average $27k vs wins of $10k. Win/loss ratio = 0.39 (need 0.56 for breakeven at 64% WR). TP target at 3.0× RR is beyond typical ranging oscillation.
**Fix**: Per-regime scope rules with lower RR (1.5), or BB width filter to only enter in tight ranges (P25 percentile = PF 1.42).

### Pattern: Compounding Artifact
**Symptom**: $100M+ PnL that's unrealistic.
**Cause**: No position size cap; equity compounds positions to exchange-impossible sizes.
**Fix**: Set max_notional_usd (e.g., $2M).

## Kelly Criterion in Trading

The Kelly criterion determines optimal bet size to maximize long-run growth:

```
f* = (p * b - q) / b
```

Where: p = win probability, q = 1-p, b = win/loss ratio (payoff ratio)

For crypto trading with 62% WR and 1.0 payoff ratio:
- f* = (0.62 * 1.0 - 0.38) / 1.0 = 0.24 = 24% of equity per trade

In practice, use fractional Kelly (25-50% of optimal) to reduce variance:
- Quarter Kelly: 6% per trade
- Half Kelly: 12% per trade

The `max_loss_per_trade_pct` effectively implements fractional Kelly.

## Drawdown Budget Math

For consecutive losses at X% per stop:

```
DD = 1 - (1 - X/100)^n
```

| Max Loss | 3 stops | 5 stops | 7 stops | 10 stops |
|---|---|---|---|---|
| 1% | 3.0% | 4.9% | 6.8% | 9.6% |
| 2% | 5.9% | 9.6% | 13.2% | 18.3% |
| 3% | 8.7% | 14.1% | 19.2% | 26.3% |
| 4% | 11.5% | 18.5% | 24.9% | 33.5% |
| 5% | 14.3% | 22.6% | 30.2% | 40.1% |

With N correlated pairs, worst case = N x max_loss per bar. For 6 pairs: 6 x 4% = 24% per bar.

## Multi-Pair Correlation Impact

**WARNING**: Correlation depends on measurement method. Per-trade correlation (zero-fill) dramatically understates. Use COMMON exit bars only.

Measured pairwise correlation (v34, common exit bars):
- BTC-SOL: 0.80, BNB-SOL: 0.74, BNB-ETH: 0.70
- ETH-SOL: 0.66, BTC-ETH: 0.64, BNB-XRP: 0.59
- DOGE-SOL: 0.54, ETH-XRP: 0.52, BTC-BNB: 0.46
- Average: 0.52 (on common bars), 0.105 (zero-fill — misleading)

Same-bar entry clustering: 65% of entry bars have 2+ entries. All-win rate is 1.9× expected (80% vs 42% if independent), confirming moderate positive correlation.

Simultaneous stops: ~255 bars with 2+ stops, ~44 bars with 3+ stops. Total multi-stop loss: ~$13.6M (offset by $60M+ in trailing stop wins).

## Slippage Reality Check

| Position Size | Realistic Slippage |
|---|---|
| < $50k | 2 bps (backtest default) |
| $50k - $500k | 5-10 bps |
| $500k - $5M | 20-50 bps |
| > $5M | 50-200 bps |

The backtest uses flat 2bps regardless of size. For realistic PnL estimation, apply variable slippage to large positions.
