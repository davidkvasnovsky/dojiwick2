# Pandas Analysis Snippets

Reusable code for ad-hoc analysis beyond the 14-section script. All snippets assume:
```python
import pandas as pd
import numpy as np
trades = pd.read_csv('/tmp/trades.csv')  # or /tmp/v34_trades.csv
```

## Portfolio DD (actual, not per-trade)

```python
starting_eq = 60000.0  # n_pairs × $10k
eq = starting_eq
peak = eq
max_dd = 0.0
for _, t in trades.iterrows():
    eq += t.pnl_usd
    peak = max(peak, eq)
    dd = (peak - eq) / peak * 100 if peak > 0 else 0
    max_dd = max(max_dd, dd)
print(f'Portfolio DD: {max_dd:.1f}% (vs per-trade DD from equity CSV)')
```

## Daily Sharpe (industry standard)

```python
trades_copy = trades.copy()
trades_copy['day'] = trades_copy['exit_bar_index'] // 24  # 1h candles
day_pnl = trades_copy.groupby('day')['pnl_usd'].sum()
starting_eq = 60000.0
cum_pnl = day_pnl.cumsum()
day_equity = starting_eq + cum_pnl.shift(1, fill_value=0)
day_returns = day_pnl / day_equity
daily_sharpe = day_returns.mean() / day_returns.std() * np.sqrt(365)
print(f'Daily Sharpe: {daily_sharpe:.2f} (vs sharpe_like which overstates ~40-45%)')
```

## Per-Regime Profit Factor

```python
for regime, g in trades.groupby('regime'):
    w = g[g.pnl_usd > 0].pnl_usd.sum()
    l = abs(g[g.pnl_usd < 0].pnl_usd.sum())
    pf = w / l if l > 0 else float('inf')
    print(f'{regime:15s} n={len(g):5d} WR={(g.pnl_usd > 0).mean():.0%} PF={pf:.2f} PnL=${g.pnl_usd.sum():>12,.0f}')
```

## Per-Pair Profit Factor

```python
for pair, g in trades.groupby('pair'):
    w = g[g.pnl_usd > 0].pnl_usd.sum()
    l = abs(g[g.pnl_usd < 0].pnl_usd.sum())
    pf = w / l if l > 0 else float('inf')
    print(f'{pair:12s} n={len(g):4d} WR={(g.pnl_usd > 0).mean():.0%} PF={pf:.2f} PnL=${g.pnl_usd.sum():>12,.0f}')
```

## Simultaneous Stops

```python
stops = trades[trades.close_reason == 'stop_loss']
bar_counts = stops.groupby('exit_bar_index').size()
multi = bar_counts[bar_counts >= 2]
print(f'Bars with 2+ stops: {len(multi)}')
total_multi_loss = sum(stops[stops.exit_bar_index == bar].pnl_usd.sum() for bar in multi.index)
print(f'Total multi-stop loss: ${total_multi_loss:,.0f}')
```

## Kelly Criterion

```python
p = (trades.pnl_usd > 0).mean()
avg_w = trades[trades.pnl_usd > 0].pnl_usd.mean() / trades[trades.pnl_usd > 0].notional_usd.mean()
avg_l = abs(trades[trades.pnl_usd < 0].pnl_usd.mean() / trades[trades.pnl_usd < 0].notional_usd.mean())
b = avg_w / avg_l
kelly = p - (1 - p) / b
print(f'Full Kelly: {kelly*100:.1f}% | Half-Kelly: {kelly*100/2:.1f}%')
```

## Shock Test (TP-10%, SL+10%)

```python
shocked = trades.pnl_usd.copy()
shocked[shocked > 0] *= 0.90   # TP -10%
shocked[shocked < 0] *= 1.10   # SL +10%
w = shocked[shocked > 0].sum()
l = abs(shocked[shocked < 0].sum())
print(f'Shocked PF: {w/l:.3f} (need ≥1.05)')
```

## Cross-Study Comparison

```python
v26 = pd.read_csv('/tmp/v26_4pct_trades.csv')
v34 = pd.read_csv('/tmp/v34_trades.csv')
for name, df in [('v26', v26), ('v34', v34)]:
    print(f'{name}: PnL=${df.pnl_usd.sum():>12,.0f} trades={len(df)} WR={(df.pnl_usd>0).mean():.0%} avg_not=${df.notional_usd.mean():>10,.0f}')
```

## Pairwise Correlation (on common bars only)

```python
# WARNING: zero-fill deflates correlation. Use common bars only.
pairs = sorted(trades.pair.unique())
for i, p1 in enumerate(pairs):
    for p2 in pairs[i+1:]:
        g1 = trades[trades.pair == p1].set_index('exit_bar_index')['pnl_usd']
        g2 = trades[trades.pair == p2].set_index('exit_bar_index')['pnl_usd']
        common = g1.index.intersection(g2.index)
        if len(common) >= 10:
            print(f'{p1} vs {p2}: corr={g1.loc[common].corr(g2.loc[common]):.3f} (n={len(common)})')
```

## Monthly Concentration

```python
trades_copy = trades.copy()
trades_copy['month'] = pd.to_datetime(trades_copy.exit_bar_index, unit='h', origin='2022-01-01').dt.strftime('%Y-%m')
monthly = trades_copy.groupby('month').pnl_usd.sum()
total = trades.pnl_usd.sum()
print(f'Best month: ${monthly.max():,.0f} ({monthly.max()/total*100:.1f}% of total)')
print(f'Positive months: {(monthly > 0).sum()} | Negative: {(monthly < 0).sum()}')
```

## Compounding Trajectory

```python
eq = 60000.0
milestones = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
hit = set()
for i, (_, t) in enumerate(trades.iterrows()):
    eq += t.pnl_usd
    for m in milestones:
        if eq >= m and m not in hit:
            hit.add(m)
            print(f'  ${m:>10,} at trade {i+1:5d} ({i/len(trades)*100:.0f}%)')
```

## Optuna Top-20 Param Convergence

```python
import optuna
storage = 'postgresql+psycopg://dojiwick:dojiwick@localhost:5432/dojiwick'
study = optuna.load_study(study_name='dojiwick_v34', storage=storage)
completed = sorted([t for t in study.trials if t.state.name == 'COMPLETE'], key=lambda t: t.value or -999, reverse=True)
for param in ['leverage', 'risk_per_trade_pct', 'max_portfolio_risk_pct', 'rr_ratio']:
    vals = [t.params.get(param, 0) for t in completed[:20]]
    print(f'{param:30s} avg={sum(vals)/len(vals):.3f} [{min(vals):.3f} - {max(vals):.3f}]')
```
