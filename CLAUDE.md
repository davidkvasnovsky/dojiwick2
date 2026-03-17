# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Dojiwick is a batch-first deterministic trading engine for crypto futures. Python 3.14, numpy-first computation, PostgreSQL persistence, Binance-only runtime. Pydantic v2 for all config/param models.

## Commands

```bash
# Setup
uv sync                # install deps + dojiwick in editable mode
make onboard           # first-time setup (applies DB migrations)

# Quality gate (run before committing — same as CI)
make ci                # fmt-check + lint + typecheck + unit tests

# Individual steps
make fmt               # auto-format
make lint              # ruff check
make typecheck         # basedpyright (strict)
make test              # pytest -q -m "not db"
make test-db           # pytest DB tests (applies migrations first)

# CLI (console script via [project.scripts])
dojiwick run --config config.toml
dojiwick backtest --config config.toml --start 2025-01-01 --end 2025-06-01
dojiwick backtest --config config.toml --start 2025-01-01 --end 2025-06-01 --trades-csv /tmp/trades.csv --equity-csv /tmp/equity.csv
dojiwick optimize --config config.toml --start 2025-01-01 --end 2025-06-01
dojiwick optimize --config config.toml --start 2025-01-01 --end 2025-06-01 --gate
dojiwick optimize --config config.toml --start 2025-01-01 --end 2025-06-01 --gate --workers 8
dojiwick gate --config config.toml --start 2025-01-01 --end 2025-06-01 --params-file params.json
dojiwick validate --config config.toml --start 2025-01-01 --end 2025-06-01 --mode walk-forward
dojiwick explain --config config.toml --pair BTCUSDT --regime trending --format json

# Makefile shortcuts (pass args via ARGS=)
make run ARGS="--config config.toml"
make optimize ARGS="--config config.toml --start 2025-01-01 --end 2025-06-01"

# Run a single test
uv run pytest tests/unit/test_hashing.py -q
uv run pytest tests/unit/test_hashing.py::test_tick_id_deterministic -q
uv run ruff check --fix .   # auto-fix lint issues

# Note: on host (outside Docker), prefix with `uv run`:
# uv run dojiwick optimize --config config.toml --start 2022-01-01 --end 2026-03-01 --gate --workers 8
```

## Database

Atlas manages schema and migrations. Canonical schema: `db/schema.sql`. Migrations: `db/migrations/`.

```bash
make db-apply           # apply migrations to local DB
make db-apply-test      # apply migrations to test DB
make db-diff name=foo   # generate new migration from schema.sql changes
make db-status          # check migration status
make db-lint            # lint migrations (Atlas)
make db-hash            # rehash migrations directory
make ci-db              # CI target for DB tests (applies migrations first)
```

Env vars: `DOJIWICK_DB_URL`, `DOJIWICK_TEST_DB_URL`, `ANTHROPIC_API_KEY`, `BINANCE_API_KEY`, `BINANCE_API_SECRET` (see `.env.example`).

## Optimization

Optuna-based hyperparameter search with anti-overfitting validation. Add `--gate` to any optimize command for CV + PBO + walk-forward validation. Post-optimization analysis: run `/analyze-research` or `uv run python .claude/skills/analyze-research/scripts/analyze_study.py --study <name> --trades /tmp/trades.csv` for a 14-section diagnostic report.

Pipeline: `OptunaRunner` (TPE sampler) → `WalkForwardObjective` (5-fold walk-forward) → composite score with configurable weights plus progressive drawdown cliff, trade frequency bonus, and trade density penalty. Search space: ~41 params (base + conditional + per-regime) in `application/use_cases/optimization/search_space.py`. Per-regime scope params (`scope_ranging__*`, `scope_trending__*`, `scope_volatile__*`) let the optimizer tune exit params independently per market regime. `SearchSpace` takes `enabled_strategies` to skip params for disabled strategies.

**Drawdown metric**: The objective uses `portfolio_max_drawdown_pct` (actual bar-level portfolio DD from `avg_equity`), NOT the per-trade normalized DD from `summarize()`. The per-trade DD inflates ~5.8× for multi-pair systems due to loss clustering. `BacktestSummary.effective_max_drawdown_pct` property selects the correct metric.

**Research gate** evaluates 9 criteria: (1) CV Sharpe, (2) PBO via CSCV, (3) walk-forward IS/OOS ratio, (4) continuous backtest min trades, (5) continuous max DD, (6) shock test PF (TP-10%/SL+10%), (7) per-regime min PF, (8) concentration (max month/trade %), (9) pair robustness (min N pairs above PF threshold). All configurable in `[research]` section. Rejection reasons in `GateResult.rejection_reasons`.

## Architecture

> See [ARCHITECTURE.md](ARCHITECTURE.md) for the canonical architecture document.

Hexagonal (ports & adapters) with strict boundary enforcement (verified by `test_architecture_rules.py`):

```
domain/          Pure business logic, zero external deps. Contracts (ports) live here.
  contracts/     Protocol classes: repositories/, gateways/, policies/
  models/        entities/ (mutable) and value_objects/ (frozen dataclasses/Pydantic)
  enums.py       All domain enums (TickStatus, MarketState, TradeAction, etc.)
  errors.py      Exception hierarchy rooted at EngineError
  type_aliases.py  Numpy type aliases: FloatVector, FloatMatrix, BoolVector, IntVector
  hashing.py     Determinism spine: tick_id, inputs_hash, intent_hash, ops_hash
  numerics.py    Semantic numeric types (Price, Money, Quantity) and converters
  symbols.py     Symbol and pair normalization helpers
  timebase.py    Bar-close alignment and anti-leakage validation
  order_state_machine.py  Order lifecycle state transitions and residual-quantity semantics
  reason_codes.py  Stable reason codes for risk, strategy, and pipeline outcomes
  indicator_schema.py  Canonical indicator ordering for batch tensors
  reconciliation_health.py  Reconciliation health state machine

application/     Use cases and orchestration. Depends on domain only.
  use_cases/     run_tick.py (TickService), run_backtest.py, run_reconciliation.py
    optimization/  objective.py, search_space.py, runner.py, pruning.py (Optuna integration)
    validation/    research_gate.py, cross_validator.py, walk_forward_validator.py, gate_evaluator.py
  models/        PipelineSettings (protocol), StartupResult, and application-layer value objects
  orchestration/ decision_pipeline.py, execution_planner.py, outcome_assembler.py,
                 regime_hysteresis.py, scanner.py, target_resolver.py
  services/      OrderLedgerService, PositionTracker, StartupOrchestrator,
                 BacktestBuilder, build_backtest_time_series(), CachingCandleFetcher,
                 OrderEventConsumer, StartupOrderCleanupService
  policies/      Risk engine, adaptive calibration
  registry/      Strategy plugin registry

compute/         Stateless numpy kernels. Pure functions, batch-shaped arrays.
  kernels/       strategy/ regime/ risk/ sizing/ indicators/ math.py metrics/ pnl/ validation/

infrastructure/  Adapter implementations.
  postgres/      PgUnitOfWork, typed repositories, connection pooling
  exchange/      Shared: cache.py, cached_context_provider.py, feed.py, indicator_enricher.py, reconciliation.py
    binance/     HTTP client, market data, execution, order events, readiness guard
    simulated/   SimulatedExecutionGateway scaffold (not wired — see ARCHITECTURE.md)
  ai/            LLM veto service, regime classifier, confidence gate, cost tracker, prompts/
  observability/ Metrics, alerts, logging
  system/        SystemClock (only place allowed to call datetime.now/time.time)

config/          Pydantic v2 settings, TOML loader, adapter composition
  schema.py      Frozen Pydantic models (Settings, SystemSettings, TradingSettings, etc.)
  loader.py      tomllib → model_validate pipeline
  composition.py Venue-dispatched adapter builder (_VENUE_BUILDERS registry)
  scope.py       StrategyScopeResolver + StrategyOverrideValues (per-pair strategy param overrides)
  risk_scope.py  RiskScopeResolver + RiskOverrideValues (per-pair risk param overrides)
  param_tuning.py Pydantic-dependent config ops (model_copy etc.), injected as callables into application
  targets.py     Target identity model, resolve_instrument_map() for display_pair → InstrumentId
  fingerprint.py fingerprint_settings() — stable config hash for tick lifecycle and backtest summary
  logging.py     Minimal logging bootstrap with JSON output

scripts/         Operational scripts (Hetzner deployment)

interfaces/      CLI entrypoints
  cli/main.py      Dispatcher — `dojiwick <command>` console script entrypoint
  cli/runner.py    Long-running tick loop with signal handling
  cli/backtest.py  Backtest on historical data
  cli/optimize.py  Optuna hyperparameter search
  cli/gate.py      Research gate evaluation
  cli/validate.py  Walk-forward / cross-validation
  cli/config_explain.py  Debug scope/risk resolution for a pair/regime
  cli/_shared.py   Common args and settings loader
```

**Dependency flow**: domain ← application ← infrastructure/compute/config ← interfaces. Domain and application must never import from infrastructure (enforced by architecture tests). Domain/application must not reference "binance" — exchange-specific code lives in infrastructure only.

**Enforced rules** (`test_architecture_rules.py`, 10 tests): no `datetime.now`/`time.time` outside `clock.py`, no `Settings()` zero-arg in `src/`, no infrastructure imports in domain/application, no config imports in application (uses `PipelineSettings` Protocol), no `binance` refs in domain/application, no production banners in `src/`, `StrategyOverrideValues` fields must match `StrategyParams`, `RiskOverrideValues` fields must match `RiskParams`.

## Key Patterns

- **Batch-first**: All computation operates on `(N,)` numpy arrays where N = number of active pairs. Kernels receive/return `FloatVector`/`BoolVector`/`FloatMatrix`.
- **Ports & adapters**: Domain defines `Protocol` classes in `contracts/`. Infrastructure provides implementations. Tests use fakes from `tests/fixtures/fakes/`.
- **Decision pipeline**: `async run_decision_pipeline()` in `orchestration/decision_pipeline.py` — single code path shared by live tick and backtest. Steps: regime classify → hysteresis → variant resolve → strategy select → AI ensemble → veto → risk assess → size intents.
- **Backtest vs live parity**: Backtest uses `run_decision_pipeline_sync` (no AI veto/regime). Live uses `run_decision_pipeline` (async, with AI). Strategy signal generation is identical in both paths. The sync path exists for optimizer performance — AI services are skipped, not mocked.
- **Pydantic v2 config**: Domain params (`RegimeParams`, `StrategyParams`, `RiskParams`) are frozen `BaseModel` in `params.py`. Config settings in `schema.py` also frozen. Use `model_copy(update={...})` not `dataclasses.replace()`. Loader uses `model_validate()`.
- **Determinism spine**: Every tick gets a `tick_id` computed from `(config_hash, timestamp, pairs)`. `TickRecord` tracks STARTED→COMPLETED/FAILED lifecycle with input/intent/ops hashes for replay verification.
- **ClockPort**: All time access goes through `ClockPort` protocol. Only `SystemClock` in `infrastructure/system/clock.py` may call `datetime.now` or `time.time`.
- **Async boundary**: Config loading, compute kernels, and CLI arg parsing are sync. Everything touching database, exchange, or market data is async (`async with`, `await`). Entry points use `asyncio.run()`.
- **Frozen dataclasses for value objects**: `@dataclass(slots=True, frozen=True, kw_only=True)` with `__post_init__` validation. Pydantic `BaseModel` is used only for config params/settings, never for domain value objects.
- **Naming**: Postgres repository implementations use `Pg` prefix (`PgOutcomeRepository`, `PgTickRepository`). Enum-to-SQL mapping tables (`TRADE_ACTION_TO_SQL`, `SQL_TO_TRADE_ACTION`) for bidirectional conversion.
- **Strategy plugin system**: Strategies implement `StrategyPlugin` protocol, registered in `StrategyRegistry` via `build_default_strategy_registry()`. Registration order = priority (first registered wins). Signal masks OR-merged across plugins.
- **PipelineSettings protocol**: Application accesses config through `PipelineSettings` (structural Protocol in `application/models/pipeline_settings.py`), not direct config imports. Config-layer functions needing Pydantic methods (e.g., `model_copy`) live in `config/param_tuning.py` and are injected as callables.
- **Shared optimization utilities in `search_space.py`**: `extract_regularization_baseline()`, `NON_STRATEGY_PARAMS`, `_STRATEGY_SIGNAL_PARAMS` (strategy→signal-param mapping). Canonical location for cross-layer optimization logic that must work from both application (`objective.py`) and config (`param_tuning.py`) layers via the `PipelineSettings` Protocol.
- **Gate evaluation API**: `evaluate_research_gate()` takes `GateThresholds` + `GateCheckResults` frozen dataclasses, NOT individual threshold kwargs. CLI commands should use `DefaultGateEvaluator` (not hand-roll gate logic) — it wires all 9 criteria with correct PBO computation.

## Testing

- **Builders pattern**: `tests/fixtures/factories/` — `domain.py` (`ContextBuilder`), `infrastructure.py` (`SettingsBuilder`), `compute.py` (`ExecutionReceiptBuilder`), `integration.py`. Fluent API for test data.
- **Fakes**: `tests/fixtures/fakes/` — one fake per port (e.g., `FakeClockPort`, `FakeExecutionGateway`). Used in unit and integration tests.
- **Markers**: `unit`, `integration`, `e2e`, `db`, `slow`. Default run excludes `db`.
- **pytest-asyncio**: `asyncio_mode = "auto"` — async test functions auto-detected.
- `conftest.py` at test root exposes builder fixtures (`context_builder`, `settings_builder`, etc.).
- `ruff` line-length 120, `basedpyright` strict mode.
- **Property-based tests**: Hypothesis strategies in `tests/fixtures/strategies.py` (`st_batch_decision_context()`). Used for determinism, shape, and bounds verification.

## Config Field Sync Points

Adding a new field to `RiskParams` requires updating 6 files in lockstep (architecture tests enforce):
1. `domain/models/value_objects/params.py` — field + validator
2. `config/schema.py` — field + forward in `.params` property
3. `config/risk_scope.py` — `RiskOverrideValues` (must match `RiskParams` exactly)
4. `application/models/pipeline_settings.py` — protocol property
5. `tests/fixtures/factories/infrastructure.py` — both `default_risk_params()` AND `default_risk_settings()`
6. `config.toml` — no defaults allowed, must be explicit

Same pattern for `StrategyParams` → `StrategyOverrideValues` (scope.py), but only 4 files: (1) `params.py`, (2) `scope.py` StrategyOverrideValues, (3) `tests/fixtures/factories/infrastructure.py`, (4) `config.toml`. No `schema.py` or `pipeline_settings.py` change needed since `Settings.strategy` IS `StrategyParams` directly.
Same for `OptimizationSettings` → `OptimizationSettingsPort`, `ResearchGateSettings` → `ResearchSettingsPort`.

## Backtest Risk Protection Stack (run_backtest.py)

Applied in order per new position entry:
1. **ECF scaling** — proportional size reduction when equity < SMA (never blocks)
2. **DD scaling** — sqrt-curve reduction with floor, uses `min(dd_scale, ecf_scale)` not product (never blocks)
3. **max_loss_per_trade_pct** — caps leveraged risk per individual trade
4. **max_portfolio_risk_pct** — caps total leveraged risk, scaled by `n_pairs / portfolio_risk_baseline_pairs` for multi-pair fairness
5. **max_notional_usd** — absolute position size cap (in sizing kernel)

Portfolio risk allocation uses **two-pass fair allocation**: Pass 1 collects all candidate entries (DD/ECF + per-trade cap applied). Pass 2 applies portfolio cap proportionally across ALL pending entries (not first-come-first-served). This eliminates pair ordering bias.

`leverage` is NOT passed to `size_intents()` (defaults to 1.0) — it only amplifies PnL in the cost model. Per-trade and portfolio risk caps must account for leverage via `cost.leverage`.

## Scope System (Per-Pair/Per-Regime Overrides)

`[[scope.strategy]]` and `[[scope.risk]]` in config.toml override params per pair/regime/strategy combo. Selectors: `pair`, `regime`, `strategy` (combinable). Priority + specificity resolve conflicts. Both strategy and risk scope rules are proportionally scaled during optimization via `_scale_scope_rules` / `_scale_risk_scope_rules`.

**Auto regime scope rules**: The optimizer generates per-regime scope rules (`auto_ranging`, `auto_trending_up`, `auto_trending_down`, `auto_volatile`) at priority 10 from `scope_*__*` search space params. Stale auto rules in config.toml are filtered out before merging fresh ones (prevents duplicate ID errors). User-defined rules at priority 100 override auto rules.

`active_pairs` and `primary_pair` are auto-derived from `[[universe.targets]]` — never set them manually in `[trading]`.

## Gotchas

- **Python 3.14 except syntax (PEP 758)**: `except A, B, C:` without parentheses is valid and is ruff's preferred format. Do not "fix" this to `except (A, B, C):` — the formatter will revert it.
- **`--trials` is NOT a CLI arg**: Trial count comes from `config.toml` `[optimization].trials`. The CLI only accepts `--config`, `--start`, `--end`, `--gate`, `--workers`.
- **config.toml is SSOT**: No behavioral tuning parameters may be hardcoded in source code. All must come from config.toml. Mathematical constants (BPS conversion 10_000, division guards 1e-9) are exempt.
- **Optuna DB connection**: Config DSN uses `postgres` hostname (Docker). For host-side CLI/scripts, the optimize CLI auto-rewrites to `localhost`. For manual queries: `postgresql+psycopg://dojiwick:dojiwick@localhost:5432/dojiwick`.
- **`extra="forbid"` on all settings**: Every Pydantic settings model uses `ConfigDict(frozen=True, extra="forbid")`. Unknown TOML keys cause validation errors — don't invent new config fields without adding them to `schema.py` first.
- **`PostExecutionPersistenceError` requires halt**: Orders may exist on exchange with no audit trail. Callers must stop processing — never swallow this exception.
- **TOML scope parsing**: `[[scope.strategy]]` and `[[scope.risk]]` sections are parsed and removed from the raw dict before `Settings.model_validate()` runs. They're injected as `strategy_scope` / `risk_scope` resolver objects separately (see `loader.py`).
- **`Settings()` zero-arg construction banned in `src/`**: Architecture test enforces this. Always load via `load_settings(path)`. Tests may use `Settings()` or `SettingsBuilder`.
- **`Edit` tool `replace_all`**: Avoid `replace_all=true` when the search string appears on both sides of assignments (e.g., replacing `settings.risk.foo` also replaces `_foo = settings.risk.foo` LHS).
- **Sharpe overstatement for multi-pair**: `sharpe_like` (per-trade annualized) overstates by ~40-45% for 6-pair systems due to correlated same-bar trades. `daily_sharpe` (daily portfolio returns × √365) is the industry-standard metric. Both are on `BacktestSummary`. Use `sharpe_like` for optimizer ranking (relative), `daily_sharpe` for reporting (absolute).
- **Equity CSV exports portfolio equity**: The `--equity-csv` output uses actual per-bar portfolio equity (from `avg_equity`), not the per-trade cumulative product. DD in the CSV matches `max_portfolio_drawdown_pct`.
- **Auto scope rule duplicate IDs**: When config.toml contains `auto_*` scope rules AND the optimizer generates fresh ones, `apply_params` filters stale rules by ID before merging. If you see `duplicate scope.strategy.id` errors, the filter in `apply_params` is not catching them.
- **Concentration `_pct` fields use 0-100 scale**: `concentration_max_month_pct = 40.0` means 40%, consistent with `max_continuous_drawdown_pct = 30.0`. NOT 0-1 fractions.
- **`_collect_bars` `collect_equity` flag**: When `False`, skips portfolio equity array allocation and per-bar writes. `run_with_hysteresis_summary_only` (optimizer path) passes `False`; `run_with_hysteresis` (full backtest) passes `True`.
