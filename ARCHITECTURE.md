# Architecture

Dojiwick is a batch-first deterministic trading engine built with hexagonal (ports & adapters) architecture. This document describes the layer boundaries, dependency rules, and core design patterns.

## Layer boundaries

```
domain/          Pure business logic, zero external deps.
application/     Use cases and orchestration. Depends on domain only.
compute/         Stateless numpy kernels. Pure functions, batch-shaped arrays.
config/          Pydantic v2 settings, TOML loader, adapter composition.
infrastructure/  Adapter implementations (Postgres, Binance, AI).
interfaces/      CLI entrypoints and scheduler.
```

**Dependency flow**: `domain ← application ← infrastructure/compute/config ← interfaces`

### Enforced rules

These rules are verified by `test_architecture_rules.py` (10 tests):

| Rule | Scope | Purpose |
|------|-------|---------|
| No `datetime.now` outside `clock.py` | All `src/` | Single time source via `ClockPort` |
| No `time.time()` / `time.monotonic()` outside `clock.py` | All `src/` | Consistent monotonic time |
| No `Settings()` zero-arg in `src/` | All `src/` | Always load via `load_settings()` |
| No infrastructure imports in domain | `domain/` | Domain purity |
| No infrastructure imports in application | `application/` | Hexagonal boundary |
| No config imports in application | `application/` | Protocol-based decoupling via `PipelineSettings` |
| No `binance` references in domain/application | `domain/`, `application/` | Venue-agnostic core |
| No production banners | All `src/` | Prevent stale status markers |
| `StrategyOverrideValues` fields match `StrategyParams` | `scope.py` | Config sync |
| `RiskOverrideValues` fields match `RiskParams` | `risk_scope.py` | Config sync |

## Ports & adapters

Domain defines `Protocol` classes in `contracts/`. Infrastructure provides implementations. Tests use fakes from `tests/fixtures/fakes/`.

```
domain/contracts/
  gateways/ (18)   ClockPort, ExecutionGatewayPort, ExecutionPlannerPort,
                   ContextProviderPort, AccountStatePort, MarketDataFeedPort,
                   MarketDataProviderPort, HistoricalCandleSourcePort,
                   OrderEventStreamPort, OpenOrderPort, PendingOrderProviderPort,
                   ReconciliationPort, ExchangeMetadataPort, UnitOfWorkPort,
                   LLMClientPort, MetricsSinkPort, AuditLogPort, NotificationPort

  repositories/ (28) TickRepositoryPort, OutcomeRepositoryPort, RegimeRepositoryPort,
                   DecisionTraceRepositoryPort, SignalRepositoryPort, PairStateRepositoryPort,
                   StrategyStateRepositoryPort, CandleRepositoryPort, BacktestRunRepositoryPort,
                   BotStateRepositoryPort, BotConfigSnapshotRepositoryPort,
                   InstrumentRepositoryPort, OrderRequestRepositoryPort,
                   OrderReportRepositoryPort, OrderEventRepositoryPort, FillRepositoryPort,
                   PositionLegRepositoryPort, PositionEventRepositoryPort,
                   PerformanceRepositoryPort, SystemEventRepositoryPort,
                   ModelCostRepositoryPort, ReconciliationRunRepositoryPort,
                   StreamCursorRepositoryPort, AdaptiveCalibrationRepositoryPort,
                   AdaptiveConfigRepositoryPort, AdaptiveOutcomeRepositoryPort,
                   AdaptivePosteriorRepositoryPort, AdaptiveSelectionRepositoryPort

  policies/ (5)    VetoServicePort, AIRegimeClassifierPort,
                   AdaptiveSelectionPolicyPort, AdaptiveRewardPolicyPort,
                   AdaptiveCalibrationPolicyPort

infrastructure/
  postgres/          PgUnitOfWork, 25+ typed repositories, audit log gateway
  exchange/binance/  HTTP client, market data, execution, order events, account state,
                     exchange metadata, readiness guard
  exchange/ (shared) ExchangeCache, CachedContextProvider, ExchangeDataFeed,
                     IndicatorEnricher, ExchangeReconciliation
  exchange/simulated/ SimulatedExecutionGateway scaffold (not wired — see Open Gaps)
  ai/                Anthropic client, veto service, regime classifier,
                     confidence gate, cost tracker, factory
  observability/     Alert evaluator, log notification, null metrics
  system/            SystemClock (only place allowed datetime.now / time.time)
```

### PipelineSettings protocol

The application layer accesses configuration through `PipelineSettings` (defined in `application/models/pipeline_settings.py`), a structural Protocol that `Settings` satisfies via duck typing. This prevents application from importing config types directly.

Config-layer functions that need Pydantic methods (e.g., `model_copy` in `apply_params`) live in `config/param_tuning.py` and are injected as callables into application-layer classes.

## Batch-first computation

All computation operates on `(N,)` numpy arrays where N = number of active pairs. Kernels in `compute/` receive and return `FloatVector`/`BoolVector`/`FloatMatrix` (numpy type aliases defined in `domain/type_aliases.py`).

### Compute kernel structure

```
compute/kernels/
  strategy/     trend_follow, mean_revert, vol_revert, confluence, _filters,
                plugin (StrategyPlugin protocol + StrategyPluginAdapter)
  regime/       classify, ensemble, evaluate
  risk/         rule (RiskRule protocol), rules/ (daily_loss, max_positions, min_rr, zero_stop)
  sizing/       fixed_fraction (vectorized position sizing),
                exchange_filters (price/qty/notional constraints)
  indicators/   compute (13 indicators: RSI, ADX, ATR, EMA×4, BB×3,
                MACD histogram+signal, volume EMA ratio)
  metrics/      summarize (Sharpe, Sortino, Calmar, profit_factor, win_rate,
                expectancy, payoff_ratio, max_consecutive_losses, etc.)
  pnl/          pnl (apply_slippage, gross_pnl, net_pnl, scalar_net_pnl),
                entry_price (4 models: close, next_open, vwap_proxy, worst_case),
                partial_fill, portfolio_evolution, liquidation
  validation/   purged_kfold (embargo-aware CV), cscv (PBO estimation),
                walk_forward (rolling/expanding windows)
  math.py       clamp01, safe_divide
```

## Decision pipeline

`run_decision_pipeline()` in `application/orchestration/decision_pipeline.py` is the single code path shared by live tick and backtest:

```
regime classify → hysteresis → halted pairs → min-confidence gate → variant resolve
  → strategy select → exit resolve → AI regime ensemble → AI veto → risk assess → size intents
```

Two entry points:
- `run_decision_pipeline` (async) — live tick path, supports AI veto + regime classifier
- `run_decision_pipeline_sync` — backtest fast-path, no AI services

Both share `_run_core_pipeline` (sync, steps 1-6c): regime classify → hysteresis → halted pairs → min-confidence gate → capture pre-AI confidence → variant resolve → strategy signal generation → phase 2 exit resolution → exits-only mode. The async path then adds AI regime ensemble → AI veto before risk + sizing.

Returns `PipelineResult` containing: regimes, candidates, veto, risk, intents, variants, authority, confidence_raw, per_pair_params.

## Strategy plugin system

Strategies implement `StrategyPlugin` protocol (`compute/kernels/strategy/plugin.py`):

```
StrategyPlugin.signal(states, indicators, prices, settings, ...) → BatchSignalFragment
```

`StrategyPluginAdapter` wraps raw `SignalFunction` (returning `(buy_mask, short_mask)` tuple) into a `StrategyPlugin`.

`StrategyRegistry` (`application/registry/strategy_registry.py`) manages plugin composition:
- Registration order = priority (first registered wins)
- Built-in plugins: `trend_follow`, `mean_revert`, `vol_revert`
- Signal composition: OR-merge buy/short masks across plugins, per-strategy deconfliction (short suppressed where buy also true), first plugin whose signal is true wins the name
- Confluence filter: RSI/volume/ADX weighting (`compute/kernels/strategy/confluence.py`)
- ATR-based stop/TP: `entry ± direction * max(atr * stop_atr_mult, entry * min_stop_distance_pct / 100)`

## Scope resolution

Two-phase per-pair/regime parameter overrides:

- `StrategyScopeResolver` (`config/scope.py`) — `[[scope.strategy]]` overrides strategy params per (pair, regime, strategy)
- `RiskScopeResolver` (`config/risk_scope.py`) — `[[scope.risk]]` overrides risk params per (pair, regime)
- Selectors: `pair`, `regime`, `strategy` — combinable. Priority + specificity resolve conflicts.
- Both strategy and risk scope rules are proportionally scaled during optimization (via `_scale_scope_rules` / `_scale_risk_scope_rules` in `param_tuning.py`).
- Pipeline caches resolution per `(pair, regime_key)` to avoid repeated lookups.
- Phase 2 exit resolution: after strategy signals are generated, scope rules are re-resolved with the winning strategy name for exit-specific overrides (stop, TP, trailing, breakeven, max_hold_bars).

## Risk policy engine

`RiskPolicyEngine` (`application/policies/risk/engine.py`): registry of pluggable `RiskRule` objects.

- `RiskRule` protocol: `evaluate(context, candidate, risk_params) → RiskRuleDecision` with blocked_mask, precedence, risk_score, reason_code
- Precedence-based merging: lower number = higher priority. Hard-coded gate: no-candidate rows at precedence 0.
- Default rules (`defaults.py`): `daily_loss`, `max_positions`, `min_rr`, `zero_stop`
- Returns `BatchRiskAssessment` with `blocked_mask`, `reason_codes`, `risk_scores`

## Config loading and fingerprinting

1. `tomllib.load()` reads TOML
2. `_validate_sections()` checks for unknown/missing sections
3. Reject authored `trading.active_pairs` / `trading.primary_pair` — these must not appear in TOML
4. Derive `active_pairs` and `primary_pair` from `[[universe.targets]].display_pair`
5. Inject derived pairs into raw dict
6. Parse scope rules (`[[scope.strategy]]`, `[[scope.risk]]`) against target-derived pairs
7. `Settings.model_validate(raw)` — validates target/trading alignment
8. `_enforce_explicit_config()` rejects missing behavior-bearing fields

Note: `fingerprint_settings()` (SHA-256 config hash) is called by CLI entrypoints after `load_settings()` returns — not within the loader itself.

## Target identity model

`[[universe.targets]]` is the sole identity source — required, non-empty. Each `TargetConfig` has four required, non-empty fields: `target_id`, `display_pair`, `execution_instrument`, `market_data_instrument`.

`trading.active_pairs` and `primary_pair` are runtime-derived from target `display_pair` values, never authored in TOML. `Settings._validate()` enforces `active_pairs == tuple(t.display_pair for t in targets)` and `primary_pair == first target` at construction time.

`resolve_instrument_map()` (in `config/targets.py`) builds a `display_pair → InstrumentId` mapping at the CLI edge. `InstrumentId` is a frozen dataclass with six validated fields: `venue`, `product`, `symbol`, `base_asset`, `quote_asset`, `settle_asset`. `resolve_targets()` (in `application/orchestration/target_resolver.py`) requires `instrument_map` as a keyword argument — there is no `pair_to_instrument_id()` fallback.

## Persistence identity

Every tick gets a `tick_id` computed from `(config_hash, timestamp, pairs)`. `TickRecord` tracks STARTED→COMPLETED/FAILED lifecycle with `inputs_hash`, `intent_hash`, and `ops_hash` for replay verification.

### Provenance enforcement

All repository write methods require non-empty `venue` and `product` parameters — no defaults on ports or implementations. Pg implementations raise `AdapterError` on empty provenance. Regime `insert_batch()` additionally validates `len(target_ids) == len(pairs)`.

Domain models validate non-empty identity fields in `__post_init__`:
- `BacktestRunRecord`: `config_hash`, `venue`, `product`, `target_ids`, plus `len(target_ids) == len(pairs)`
- `PairTradingState`: `pair`, `target_id`, `venue`, `product`
- `Signal`: `pair`, `target_id`, `signal_type`
- `InstrumentId`: all six fields (`venue`, `product`, `symbol`, `base_asset`, `quote_asset`, `settle_asset`)

No read-path fallbacks — malformed persisted data raises errors. DDL: all provenance columns are `NOT NULL` without `DEFAULT`.

`TickService.__post_init__` validates non-empty `target_ids`, length match with `active_pairs`, and complete `instrument_map` coverage for all active pairs.

### Execution lifecycle

- `SubmissionAck` — exchange acknowledgement of order submission (accepted/rejected/cancelled)
- `FillReport` — economics of a single fill (price, quantity, fees)
- `ExecutionReceipt` — transitional wrapper combining both (used by `execute_plan`)

Fee tracking includes both `fees_usd` (approximate USD) and `fee_asset`/`native_fee_amount` (native exchange fee).

## Order state machine

`domain/order_state_machine.py` defines valid order lifecycle transitions:

- NEW → {PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED, REJECTED}
- PARTIALLY_FILLED → {FILLED, CANCELED, EXPIRED}
- Terminal states (FILLED, CANCELED, EXPIRED, REJECTED): no outgoing transitions.

`compute_residual_quantity(original, filled)` returns the unfilled quantity on cancel/expire — not re-queued.

## Order & position lifecycle

Three-table order persistence: `order_requests` → `order_reports` → `fills`, plus immutable `order_events` audit trail.

- `OrderRequest` (`order_request.py`) — submitted intent: side, type, qty, price, position_side, reduce_only, close_position, time_in_force
- `OrderReport` (`order_request.py`) — exchange-acknowledged state: status, filled_qty, avg_price, cumulative_quote_qty
- `Fill` (`order_request.py`) — individual fill: price, qty, commission, commission_asset, realized_pnl_exchange
- `FillReport` (`fill_report.py`) — fill economics for execution receipt: fill_price, filled_quantity, fees_usd, fee_asset, native_fee_amount
- `OrderEvent` (`order_event.py`) — immutable lifecycle event log

Position tracking via `PositionLeg` (hedge-native: account, instrument_id, position_side, quantity, entry_price, leverage, liquidation_price) + `PositionEventRecord` (open/add/reduce/close with quantity, price, realized_pnl).

`OrderEventConsumer` (`application/services/order_event_consumer.py`) continuously processes WS ORDER_TRADE_UPDATE events, persisting reports/fills/events and updating position legs. `StreamCursorRecord` tracks stream position for crash recovery replay.

## Reconciliation health state machine

`domain/reconciliation_health.py` defines state transitions for DB-to-exchange drift detection:

- **NORMAL** + clean → NORMAL | + orphaned_exchange → UNCERTAIN | + qty_mismatch → DEGRADED
- **DEGRADED** + clean → NORMAL | + timeout → UNCERTAIN | else → DEGRADED
- **UNCERTAIN** + clean → NORMAL | + timeout → HALT | else → UNCERTAIN
- **HALT** → HALT (manual recovery only)

`HealthState` tracks: `health`, `health_since`, `frozen_symbols` (symbols that diverged). Persisted in the `bot_state` table.

`ReconciliationService` (`application/use_cases/run_reconciliation.py`): startup gate blocks the first tick until reconciliation completes. Periodic checks run between ticks.

## Cost model

`CostModel` frozen dataclass (`domain/models/value_objects/cost_model.py`):

- `fee_bps` — fee basis points
- `fee_multiplier` — applied to fee_bps for round-trip cost
- `slippage_bps` — execution slippage
- `impact_bps` — market impact (optional)
- `funding_rate_per_bar` — perpetual funding cost
- `leverage` — position leverage (scales PnL)
- `maintenance_margin_rate` — liquidation margin (0 disables liquidation modeling)

Consumed by PnL kernels: `apply_slippage()`, `gross_pnl()`, `net_pnl()`, `scalar_net_pnl()`.

## Backtest risk protection stack

Applied in order per new position entry in `run_backtest.py`:

1. **ECF scaling** — proportional size reduction when equity < SMA (never blocks)
2. **DD scaling** — sqrt-curve reduction with floor (never blocks)
3. **max_loss_per_trade_pct** — caps leveraged risk per individual trade
4. **max_portfolio_risk_pct** — caps total leveraged risk across ALL open positions
5. **max_notional_usd** — absolute position size cap (in sizing kernel)

`leverage` amplifies PnL in the cost model. The decision pipeline does not pass leverage to `size_intents()` (uses default 1.0), though the parameter exists for direct callers.

The backtest tracks two distinct drawdown measures: `max_drawdown_pct` (worst per-trade peak-to-trough from metrics) and `max_portfolio_drawdown_pct` (portfolio equity curve peak-to-trough, always tracked). `effective_max_drawdown_pct` prefers portfolio DD when available. The optimization objective and research gate use the effective measure for drawdown penalty and validation checks.

## Optimization pipeline

Optuna-based hyperparameter search in `application/use_cases/optimization/`:

- `OptunaRunner` (`runner.py`): TPE sampler, trial count from `config.toml`
- `HysteresisObjective` (`objective.py`): composite `_base_score` with ~10 components: graduated min-trades penalty, drawdown-based score, progressive drawdown cliff, logarithmic trade frequency bonus, trade density penalty, high win-rate bonus, normalized expectancy, consecutive loss penalty, payoff ratio reward, and total PnL (log return). `WalkForwardObjective` adds consistency penalty (std across folds) + regularization.
- Search space (`search_space.py`): ~21 base + 15 per-regime scope + 4 risk sizing + 5 adaptive exit + 3 conditional = ~44 maximum hyperparameters across regime, strategy, risk, and exit params
- Pruning (`pruning.py`): early stopping for unpromising trials
- CLI: `dojiwick optimize --config ... --start ... --end ... [--gate] [--workers N]`

## Research gate

9-criterion anti-overfitting validation in `application/use_cases/validation/`:

1. **CV Sharpe ≥ threshold** — purged K-fold (`cross_validator.py` + `compute/kernels/validation/purged_kfold.py`)
2. **PBO ≤ threshold** — CSCV method (`compute/kernels/validation/cscv.py`)
3. **Walk-forward IS/OOS ratio** — (`walk_forward_validator.py` + `compute/kernels/validation/walk_forward.py`)
4. **Continuous backtest min trades**
5. **Continuous backtest max drawdown**
6. **Shock test PF** — profit factor under TP-10%/SL+10% perturbation
7. **Per-regime min PF** — minimum profit factor per market regime
8. **Concentration** — max month % and max trade % thresholds
9. **Pair robustness** — min N pairs above PF threshold

`gate_evaluator.py` synthesizes criteria into `GateResult` with `rejection_reasons` list. Config: `ResearchGateSettings` with thresholds for each criterion. CLI should use `DefaultGateEvaluator` (not hand-roll gate logic).

## Adaptive policy

Thompson sampling arm selection (`application/policies/adaptive/service.py`):

- `AdaptiveMode`: DISABLED (default), CONTINUOUS, BUCKET_FALLBACK
- Per-regime/config arm selection using Beta-distribution posteriors
- Records which config was selected per position leg (`AdaptiveSelectionEvent`) and observed reward post-close (`AdaptiveOutcomeEvent`)
- Calibration diagnostics and periodic decay
- DB tables: `adaptive_configs`, `adaptive_posteriors`, `adaptive_selections`, `adaptive_outcomes`, `adaptive_calibration_metrics`

## Exchange data feed

`ExchangeDataFeed` (`infrastructure/exchange/feed.py`): WS-first with REST fallback.

- Feed status: DISCONNECTED → BOOTSTRAPPING → WS_ACTIVE (or REST_FALLBACK)
- `ExchangeCache` (`cache.py`): async-locked atomic snapshots (prevents torn reads)
- `CachedContextProvider` (`cached_context_provider.py`): reads cache, enriches with indicators
- `IndicatorEnricher` (`indicator_enricher.py`): fetches candles, computes 13-indicator matrix

## AI ensemble subsystem

`build_ai_services()` factory (`infrastructure/ai/factory.py`) → `AIServices` bundle:

- `LLMVetoService` — evaluates batch candidates, returns approval mask + reason codes
- `LLMRegimeClassifier` — overrides deterministic regime confidence when AI disagrees
- `compute_llm_review_mask()` (`confidence_gate.py`) — pre-filter: determines which pairs need LLM veto review based on regime confidence thresholds
- `CostTracker` — daily budget enforcement, per-model token costs, flush to DB (`model_costs` table)
- `NullRegimeClassifier` / `NullVetoService` — no-op implementations for AI-disabled mode
- `ModelCostRecord` — tracks input/output tokens and cost per tick

## Adapter composition

`config/composition.py` — venue-dispatched adapter builder:

- `ComposedAdapters` frozen dataclass bundles: context_provider, execution_gateway, execution_planner, exchange_metadata, account_state, open_order_port, feed, order_stream
- `_VENUE_BUILDERS` registry: `VenueCode → AdapterBuilder` (Binance-only currently)
- `build_market_data_fetcher()` — backtest CLI helper: Binance provider optionally wrapped with Postgres `CachingCandleFetcher`
- Adding a new exchange: implement adapters → readiness guard → builder function → register in `_VENUE_BUILDERS`

## Domain entities

- `BotState` (`domain/models/entities/bot_state.py`) — mutable circuit breaker and operational state: consecutive_errors, consecutive_losses, daily_trade_count, daily_pnl_usd, circuit_breaker_active, circuit_breaker_until, last_tick_at, last_decay_at, daily_reset_at, recon_health, recon_health_since, recon_frozen_symbols
- `PairTradingState` (`domain/models/entities/pair_state.py`) — mutable per-pair performance tracker: wins, losses, consecutive_losses, last_trade_at, blocked flag, win_rate property
- Both validate non-negative counts in `__post_init__`; `PairTradingState` validates non-empty identity fields (pair, target_id, venue, product)

## Open gaps

### Execution boundary parity

`SimulatedExecutionGateway` exists as a scaffold. `simulated_execution: bool` exists in `BacktestSettings`. Neither is wired. The vectorized PnL path (`scalar_net_pnl`) is the only execution model for backtests.

**Status**: Explicitly tracked gap. If `simulated_execution = true` in config, the engine raises `ConfigurationError` at startup. The shallow fix (routing exit PnL through a gateway) does not establish planner/gateway/fill lifecycle parity. Proper fix requires:

- An execution-model spec defining what "parity" means (order submission → ack → fill → fee → PnL)
- `SimulatedExecutionGateway` handling all lifecycle states (partial fill, rejection, timeout, cancel)
- Validation that simulated and vectorized paths produce equivalent results on the same inputs

Do NOT wire the gateway until this spec is written and approved.

## Testing patterns

- **Builders**: `tests/fixtures/factories/` — `ContextBuilder`, `SettingsBuilder`, etc.
- **Fakes**: `tests/fixtures/fakes/` — one fake per port
- **Markers**: `unit`, `integration`, `e2e`, `db`, `slow`
- **Property-based**: Hypothesis strategies in `tests/fixtures/strategies.py`
