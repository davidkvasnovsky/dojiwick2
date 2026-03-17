---
name: run-gate
description: Run optimization with research gate validation (CV + PBO + walk-forward) for a given config and date range
---

# Run Gate Validation

Run the full optimization + research gate pipeline. This validates strategy parameters against overfitting using purged K-fold CV, probability of backtest overfitting (PBO/CSCV), and walk-forward IS/OOS degradation ratio.

## Usage

Ask the user for any missing parameters:
- `--config` (required): path to TOML config file
- `--start` (required): start date (YYYY-MM-DD)
- `--end` (required): end date (YYYY-MM-DD)
- `--trials` (optional): number of Optuna trials (default: 50)

## Steps

1. Run optimization with gate validation:
   ```bash
   dojiwick optimize --config <config> --start <start> --end <end> --trials <trials> --gate
   ```

2. If the gate passes, show the summary and recommended parameters.

3. If the gate rejects, explain which validation failed (CV, PBO, or walk-forward) and suggest next steps:
   - Reduce parameter count
   - Expand date range
   - Increase trials
   - Review search space constraints
