# Kickoff: Phase 5 (consolidation + hygiene) and Phase 6 (docs + tests)

Continuation of the comprehensive architecture overhaul. Phases 1–4 are **done
and committed** on `main`. This file is the complete work order for the two
remaining phases — a fresh session can execute it top to bottom. The full
audit-and-fix plan lives at
`~/.claude/plans/review-our-whole-architecture-lucky-deer.md` (reference only;
everything actionable is restated here).

## State of the repo (what already happened)

Commits, oldest first:

- `d870ec7` migration squash to v3 baseline, v38 #7109 promotion, research/ restructure
- `e285b5e` **Phase 1+1b**: leverage-consistent costs (fees/funding on notional×leverage), two-sided slippage, gap-through exit fills via `next_open`, exit-loop bar shift fixed (first post-entry bar is now stop-checked), liquidation modeling in the real path (legacy zero-MMR mode removed; `maintenance_margin_rate > 0` required when leverage > 1), signed historical funding pipeline (`funding_rates` table + Binance fetcher + caching + per-bar `next_funding` accrual), shared equity pool (per-pair sub-account model removed), strategy BUY/SHORT conflict → first-registered wins, 12 stale db tests fixed
- `df9d44a` **Phase 2**: multi-worker `load_study` gets sampler+pruner, per-trial exception catch, heartbeat+RetryFailedTrialCallback, WF negative-IS ratio fix, worst-window gate (`min_wf_worst_window_sharpe`), real shock re-simulation (`perturb_exit_geometry` injected into the gate), embargo applied between CV test folds (`purge_bars` + purged_kfold kernel deleted), risk-scope scaling clamped to search bounds, tuned-Settings revalidation, `objective_trade_freq_cap`, validate-CLI PBO fed trade returns, UTC JSON logs
- `920148e` **Phase 3C**: enricher passes volume (live emitted ZERO entries before), candle_lookback 600 + validator, closed-bars-only indicators, all-zero-row exclusion instead of NaN zero-fill, enrichment failure raises, day_start UTC reset, bar-aligned tick sleeps, live ECF/DD entry scaling in TickService
- `4240ca8` **Phase 3B**: AIServiceError + single retry layer + fail-open honored for all error classes, PgModelCostRepository wired + day-spend restore, veto-rate alert wired, NET-leg PnL sign, overfill loud, unknown-instrument fails persistence, DOJIWICK_LIVE_ACK mainnet interlock, recvWindow bound, gateway qty/price guard, secrets `repr=False`
- `ee49f10` **Phase 4**: deleted adaptive bandit subsystem (+5 tables via migration), metrics plumbing, simulated gateway + flag, scanner, CronRunner, aws script, numpy exchange_filters kernel, WS gap-stub machinery, 19 reason codes, dead flags (`ws_enabled`, `rest_fallback_enabled`, `benchmark_mode`, `inactive_gap_policy`, `max_volume_pct`), dead symbols, 3 unused SQL enums, 8 orphaned fakes; feed now always REST-refreshes prices (WS_ACTIVE price-freeze fixed)
- `ee0d08e` **Phase 3A**: instrument sync at startup, planner Decimal quantization (step/min-qty/min-notional), per-attempt signing + per-category retries + Retry-After/418, MARKET-close param fix, pre-persisted order requests + `position_applied_qty` high-water mark (single fill-application path), `PositionExitState` + `ProtectiveOrderService` (exchange-side STOP/TP/TP1 reconciler with trailing revisions, orphan healing, sibling-cancel on fire), consumer supervisor with reconnect + per-symbol userTrades replay, keepalive escalation, tolerant order-type parsing, terminal-rejection NO_RETRY mapping

Gate status: `make ci` and `make test-db` fully green (run everything inside
Docker — **this host has no uv**: `docker compose run --rm tooling make ci`).
Migrations: 4 files under `db/migrations/`, all applied to local dev+test DBs.

**Backtest numbers changed across the board (costs were understated ~2.85×,
equity was per-pair). v38/v39 studies are NOT comparable. After Phase 6,
re-run `dojiwick optimize --gate --workers 8` as v40 and re-gate before any
promotion.** config.toml `study_name` is already `dojiwick_v40`.

---

## Phase 5 — consolidation + hygiene

Every item verified as a real finding; no speculative work. Items marked
(done) were pulled forward — do not redo them.

### Code consolidation

1. **Shared ATR stop/TP kernel** — the formula exists twice: vectorized in
   `application/registry/strategy_registry.py` (~lines 120–150, see the
   `NOTE: keep formula in sync` comment) and scalar in
   `application/orchestration/decision_pipeline.py::_compute_stop_tp_scalar`.
   Extract one shared implementation (a small compute kernel or a domain
   helper both call); delete the NOTE.
2. **`config/schema.py` `_cached_params` hack** — `RegimeSettings` and
   `RiskSettings` cache their `.params` by mutating `vars(self)` on a frozen
   Pydantic model and override `model_copy` to pop the cache. Replace with a
   plain non-caching property or a cleaner memo that passes basedpyright
   strict. Grep `_cached_params` and `def model_copy` in schema.py.
3. **Fingerprint/loader single source** — `config/loader.py` and
   `config/fingerprint.py` each define `_INFRA_ONLY_FIELDS` and they disagree
   (`reconciliation_interval_ticks`). Share one constant. Also stop embedding
   the `database` section (DSN) in the persisted config snapshot
   (`fingerprint.canonical_json` → `bot_config_snapshots` via runner).
4. **`caching_candle_fetcher.py::_store`** — never cache the still-forming
   candle: drop trailing candles with `open_time + interval_to_seconds(interval)
   > clock-now` before upserting (both the cold-miss path and gap fills).
   The fetcher currently has no clock — inject `ClockPort`. Mirror the same
   guard in `caching_funding_fetcher._store`? No — funding events are settled
   facts, no guard needed there.
5. **Idempotent tick-transaction inserts** — `infrastructure/postgres/repositories/regime.py`
   and `outcome.py` use plain INSERT against UNIQUE constraints
   (`regime_observations (target_id, observed_at)`, `decision_outcomes
   (tick_id, target_id)`); a duplicate key rolls back the whole post-execution
   transaction and halts the engine. Add `ON CONFLICT DO NOTHING`.
6. **`safe_divide` standardization** — division guards exist two ways in
   compute kernels: `math.py::safe_divide` vs inline `np.divide(where=)`
   (min_rr.py, fixed_fraction.py, summarize.py). Standardize on `safe_divide`
   where it doesn't change semantics.
7. **Sharpe ddof consistency** — `summarize.py` and `compute_daily_sharpe`
   use `np.std` (ddof=0) while `cscv.py::_sharpe` uses ddof=1. Pick one
   (ddof=1/sample is conventional) and align all three; update any exact-value
   test expectations (`test_cscv.py` asserts `pbo == 0.0` seeded).

### CLI + config

8. **Gate exit codes** — `interfaces/cli/gate.py` and `optimize.py --gate`
   exit 0 even when `result.passed` is False. Exit non-zero on rejection so
   automation can gate on it.
9. **Unified exit-code contract** — `interfaces/cli/main.py` dispatch:
   `run`/`explain` propagate codes via `SystemExit(int)`, the backtest-family
   commands don't. Make all subcommands return/propagate exit codes one way.
10. **Shared gate-print helper** — gate result printing is duplicated between
    `optimize.py` (~line 310) and `gate.py` (~line 66). Both already import
    `print_gate_result` from `_shared` — verify and collapse any remaining
    duplication (the RESEARCH GATE banner blocks).
11. **One logging bootstrap** — runner uses `config/logging.py` (JSON);
    backtest-family CLIs use `logging.basicConfig` in `_shared.setup_env`.
    Use the JSON bootstrap everywhere.
12. **DSN single source** — `DatabaseSettings.dsn` ignores `DOJIWICK_DB_URL`;
    the env var feeds only Atlas, so app and migrations can silently target
    different databases. Make the loader honor `DOJIWICK_DB_URL` as an
    override of `[database].dsn` and document precedence in config.toml.
13. **Persist `daily_sharpe`** — computed on every run, dropped on write:
    `backtest_runs` has no column and `PgBacktestRunRepository` doesn't write
    it. Add column (new Atlas migration via the dockerized-atlas recipe below)
    + INSERT field. Only the overstated `sharpe_like` is stored today.
14. **`config_explain.py` lossy table** — prints only 4 resolved strategy
    fields while `field_winners` lists all matches including params the
    kernels globally ignore (`min_confluence_score`, trend thresholds — see
    the NOTE in `trend_follow.py`). Print all resolved fields and mark the
    kernel-ignored ones.
15. **`validate.py evaluate(best_params={})`** — the full-gate mode passes an
    empty param set; verify apply_tuned-with-empty-params is intended (it
    re-applies current settings) and add a comment or fix.

### Build/tooling

16. **Ruff real ruleset** — `[tool.ruff]` has no `lint.select`; only default
    E4/E7/E9/F run. Add `select = ["E4","E7","E9","F","I","UP","B","SIM","C4","PIE","RUF"]`
    (tune as needed), then fix the fallout (expect import-sorting churn —
    commit that separately). Keep PEP 758 bare multi-except (ruff prefers it
    on py314 — do NOT parenthesize).
17. **pyproject dep hygiene** — `psycopg-pool` is in core deps (`>=3.3.0`)
    AND the `postgres` extra (`>=3.2`) with different floors, while the
    `psycopg[binary]` driver is extra-only (pool-without-driver install is
    possible). Move the pool into the extra beside the driver; drop the dupe.
18. **Makefile** — split `ENVRUN` so Binance/Anthropic secrets are only
    injected for `run`/`backtest`/`optimize`/`gate`/`validate`, not
    pytest/atlas targets; drop the pointless `&` after `docker compose up -d`
    in `docker-ci` (or use `up --wait`).
19. **CI workflow** — pin `actions/checkout@v6`, `docker/setup-buildx-action@v4`,
    `crazy-max/ghaction-github-runtime@v4` by commit SHA (verify those tags
    even exist — they're ahead of known-good majors). Verify the GHA buildx
    cache actually exports through `docker compose build` (may need
    `COMPOSE_BAKE=true` in the workflow env); if it doesn't, caching is
    silently a no-op.
20. **Runtime image** — Dockerfile builder uses `uv sync --no-dev --all-extras`,
    shipping optuna+sqlalchemy in the live-trading image. Use
    `--extra postgres` for the runtime stage (tooling keeps all extras).
21. **hetzner-optimize.sh** —
    a. `op read` results are written to the remote `.env` unvalidated: a
       failed read inside `{ } 2>/dev/null` yields empty keys and `set -e`
       does not catch command-substitution-in-assignment. Validate
       `[[ -n "$_bk" && -n "$_bs" ]] || die`.
    b. DSN `sed` rewrites host (`@postgres:` → `@localhost:`) but not port
       while the reverse tunnel hardcodes 5432 — rewrite the port too.
    c. SSH is trust-on-first-use every run (`ssh-keygen -R` + `accept-new`)
       over the channel that carries the Binance keys. Pre-generate the host
       key locally, inject via cloud-init `ssh_keys:`, pin with a per-run
       `UserKnownHostsFile`.
    d. `curl -LsSf https://astral.sh/uv/install.sh | sh` runs unpinned as
       root on the secret-bearing box; pin to the Dockerfile's uv version
       (0.11.28) + sha256 check.
22. **`bot_state_history` retention** — schema comment claims "retain last 30
    days via pg_cron" but nothing implements it. Either add a cleanup (a
    DELETE in the runner's periodic recon is fine) or delete the claim.

### Bloat-comment sweep (delete WHAT-narration; keep verified WHY)

Registry, summarize.py, confluence-adjacent files were partially cleaned in
Phases 1–4. Remaining (verify each still exists before editing — line numbers
drift):

- `application/orchestration/decision_pipeline.py` — numbered step banners
  (`# 1. Classify regime`, `# 2. Hysteresis`, `# 2b…`, `# 5…`, `# 6…`, `# 6b…`,
  `# 6c…`). Keep genuine WHY comments (e.g. the "no copy needed" note).
- `application/use_cases/optimization/objective.py` — per-term score labels
  (`# High win-rate bonus`, `# Expectancy component…`, `# Consecutive loss
  penalty`, `# Payoff ratio reward…`, `# Total PnL: log-scaled…`). The
  regime-PF penalty comment and trade-freq cap comment are WHY — keep.
- `application/use_cases/run_tick.py` — the numbered step comments are
  borderline (they map to the documented pipeline order); trim the ones that
  purely restate the next call, keep ordering-constraint ones (e.g. "pure --
  no I/O dependency on ledger", "exchange I/O outside the txn").
- `compute/kernels/strategy/_filters.py` / `min_rr.py:~65` — pure
  restatements; the sign-convention comments at min_rr 45/55 can stay.
- `compute/kernels/strategy/trend_follow.py` — the "(BTC is often volatile
  AND trending)" editorial half; keep the NOTE about scope overrides being
  ignored (that's a real gotcha until someone fixes it).
- `compute/kernels/indicators/compute.py` — `# Wilder smooth…`, `# Smooth DX
  to get ADX` (delete); keep :165/:199 WHY comments.
- `compute/kernels/pnl/entry_price.py:~48`, `partial_fill.py:~53` — optional
  deletes.
- `domain/ai_evaluation.py` module docstring — references a past change and
  its example codes don't match real constants; rewrite to describe what it
  IS.
- `interfaces/cli/optimize.py` — comments say "forked worker"/"before
  forking" but the code uses spawn; module docstring example passes
  `--trials 50` which is not a valid CLI arg.
- `db/schema.sql` — sweep for stale comments after the phase-4/3A table
  changes.

### Deferred decision (flag for the user, don't silently choose)

- **Portfolio-cap pair scaling under the shared pool**:
  `run_backtest._collect_bars` still scales `max_portfolio_risk_pct` by
  `n_pairs / portfolio_risk_baseline_pairs`. That formula was designed for
  per-pair wallets; under the shared pool it inflates total risk as pairs are
  added. Options: drop the scaling (cap = flat fraction of pool) or keep as a
  tunable. Raise with the user before changing — it shifts optimizer
  behavior.

---

## Phase 6 — docs + tests + validation run

### Docs

23. **ARCHITECTURE.md** — rewrite stale sections and add new ones:
    - WS/order-event section: supervisor reconnect + per-symbol userTrades
      replay + keepalive escalation (the old text described gap detection
      that never existed).
    - Cost tracking: PgModelCostRepository is wired; budget survives restarts.
    - Remove: adaptive subsystem, metrics sink, SimulatedExecutionGateway
      "open gap" note (deleted, not pending), scanner, purged k-fold (now
      embargoed contiguous folds — describe honestly).
    - Add: shared equity pool model; signed historical funding pipeline
      (candle-triple mirror, `next_funding` alignment); liquidation modeling;
      gap-through fill semantics; instrument sync; pre-persist + high-water
      mark fill application; PositionExitState + ProtectiveOrderService
      lifecycle (incl. `dw_p{leg}_{rev}` client-id scheme and no-OCO
      sibling-cancel); planner quantization; live ECF/DD scaling; mainnet
      interlock.
    - Param count: search space is ~43 dims.
24. **CLAUDE.md** —
    - Commands: unchanged, but note `DOJIWICK_LIVE_ACK=1` required for
      mainnet `run`.
    - Config Field Sync Points: add `BacktestSettings.funding_mode`
      (enums/schema/port/config/factories/cost_model), `ExchangeSettings` WS +
      protective knobs, `OptimizationSettings.objective_trade_freq_cap`,
      `ResearchGateSettings.min_wf_worst_window_sharpe`; remove references to
      deleted fields (`purge_bars`, `funding_rate_per_bar`, `impact_bps`,
      `max_volume_pct`, `benchmark_mode`, `inactive_gap_policy`, `[adaptive]`,
      `ws_enabled`/`rest_fallback_enabled`, `simulated_execution`).
    - Gotchas: prune stale ones (`--trials` note stays; funding_rate_per_bar,
      adaptive, SQL_TO_TRADE_ACTION mentions go). Add new gotchas: fills
      table is WS-only; `position_applied_qty` high-water mark semantics;
      protective orders use `dw_p` client-id prefix (startup cleanup skips
      them); pending-order guard counts entry kinds only; backtest exit
      checks read `next_*[bar_idx-1]` (the just-closed bar); `funding_mode =
      "historical"` needs DB + exchange reachability at data load; scripts/
      is hetzner-only now.
    - Architecture tree: scanner/simulated/adaptive/metrics/scheduler gone;
      new files (`instrument_sync.py`, `protective_orders.py`,
      `caching_funding_fetcher.py`, `exit_rules.py`,
      `position_exit_state.py`, `funding.py`, `funding_rate.py`).
    - Sharpe guidance: numbers changed; keep the sharpe_like vs daily_sharpe
      note.
25. **README.md** — consistency pass against the above (commands, workflow).

### Tests to add

26. **Fingerprint with non-empty scope rules** — no test proves a scope-rule
    change alters `trading_sha256` (serialization of the stdlib-dataclass
    resolvers through `model_dump` is unguarded). Add: fingerprint changes
    when a rule value changes; stable across rule reordering? (decide and
    encode current behavior).
27. **Stored-tick replay** — recompute `inputs_hash`/`intent_hash`/`ops_hash`
    from a persisted TickRecord's inputs and compare against stored values
    (true replay; current tests only hash twice in-process).
28. **Order lifecycle progression** — one order NEW → PARTIALLY_FILLED →
    FILLED through the consumer: cumulative HWM math applies 0.5 then 0.5
    more, report ends FILLED. (The duplicate-delivery test exists; the
    partial-progression one doesn't.)
29. **Hysteresis edges** — revert-to-original during the pending window
    (A→pendingB→A) and multi-pair independence of pending state.
30. **Planner quantization unit tests** — step-floor, min-qty drop for
    entries, reduce-never-dropped (except sub-step), min-notional drop with
    priced delta, filter cache. (Manual verification was done; encode it.)
31. **DB-marker HWM test** — `advance_applied_qty` FOR UPDATE semantics
    against real Postgres: two sequential advances, second smaller cumulative
    → zero delta.
32. **Pending-guard SQL test** (db marker) — protective/reduce-only rows are
    excluded from pending quantities.
33. **Fix weak tests** — `tests/integration/test_fill_simulation.py`
    `test_next_open_differs_from_close` passes vacuously with zero trades:
    make the fixture guarantee a trade or assert trades > 0. Replace or drop
    the 4 source-text-grep tests in `tests/unit/test_hardening_gaps.py`
    (search "read_text" or the greps at ~lines 81/272/284/437 pre-drift) with
    behavior assertions where cheap.
34. **Drop the unused `slow` marker** from pyproject (declared, used 0
    times) or start using it.

### Validation run (after everything is green)

35. `docker compose run --rm tooling make ci && docker compose run --rm tooling make test-db`
36. Full before/after benchmark: run a continuous backtest 2022→2026 with the
    current promoted params and document the delta vs the pre-overhaul
    numbers (expect PnL/Sharpe down — costs were understated; DD honest now):
    `docker compose run --rm tooling uv run dojiwick backtest --config config.toml --start 2022-01-01 --end 2026-03-01 --trades-csv /tmp/trades.csv --equity-csv /tmp/equity.csv`
    then `/analyze-research` on the outputs. First run will backfill the
    `funding_rates` table (one-time, ~5 requests/pair).
37. Kick off the v40 study on Hetzner (`scripts/hetzner-optimize.sh` — after
    item 21 fixes) and re-gate. **Do not promote v38 params to live: they
    were tuned under the broken cost model.**
38. Live-path smoke before mainnet: run against Binance testnet
    (`testnet = true`) long enough to see: instrument sync succeeds, an entry
    fills, protective STOP+TP appear on the exchange, trailing revision
    replaces the stop, and killing the WS mid-run reconnects + replays.
    Mainnet needs `DOJIWICK_LIVE_ACK=1`.

### Practical notes for the executor

- **Docker-only host**: never run `uv`/`pytest` on the host. Everything goes
  through `docker compose run --rm tooling …`.
- **Atlas migrations from the host** (no atlas binary, no .env): spin a
  throwaway dev DB and run the pinned atlas image:
  ```bash
  docker run -d --rm --name atlas-dev -e POSTGRES_PASSWORD=dev -p 5499:5432 postgres:18.4-alpine
  sleep 4
  docker run --rm -v "$PWD/db:/db" arigaio/atlas:1.2.3-community migrate diff <name> \
    --dir file:///db/migrations --to file:///db/schema.sql \
    --dev-url "postgres://postgres:dev@host.docker.internal:5499/postgres?sslmode=disable" \
    --format '{{ sql . "  " }}'
  docker stop atlas-dev
  ```
  Then `docker compose run --rm tooling sh -c "make db-apply && make db-apply-test"`.
- **Never edit applied migration files** — always a new `migrate diff`.
- Tests import fakes from `tests/fixtures/fakes/`; keep one fake per port.
  `FakeExchangeMetadata`/`FakeExecutionPlanner` exist but are currently
  unused — either use them for the quantization tests (item 30) or fold their
  logic into `_StaticMetadata`-style locals as `test_protective_orders.py`
  does, and delete the orphans.
- Commit per logical group with the same message style as the existing
  commits; run the full gate before each commit.
