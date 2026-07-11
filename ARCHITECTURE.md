# Architecture

Dojiwick is a batch-first deterministic trading engine built with hexagonal (ports & adapters) architecture. This document describes the layer boundaries, dependency rules, and core design patterns.

## Layer boundaries

```
domain/          Pure business logic, zero external deps.
application/     Use cases and orchestration. Depends on domain only.
compute/         Stateless numpy kernels. Pure functions, batch-shaped arrays.
config/          Pydantic v2 settings, TOML loader, adapter composition.
infrastructure/  Adapter implementations (Postgres, Binance, AI).
interfaces/      CLI entrypoints.
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
                   HistoricalFundingSourcePort, OrderEventStreamPort, OpenOrderPort,
                   PendingOrderProviderPort, ReconciliationPort, ExchangeMetadataPort,
                   UnitOfWorkPort, LLMClientPort, AuditLogPort, NotificationPort

  repositories/ (25) TickRepositoryPort, OutcomeRepositoryPort, RegimeRepositoryPort,
                   DecisionTraceRepositoryPort, SignalRepositoryPort, PairStateRepositoryPort,
                   StrategyStateRepositoryPort, CandleRepositoryPort, FundingRateRepositoryPort,
                   BacktestRunRepositoryPort, BotStateRepositoryPort,
                   BotConfigSnapshotRepositoryPort, InstrumentRepositoryPort,
                   OrderRequestRepositoryPort, OrderReportRepositoryPort,
                   OrderEventRepositoryPort, FillRepositoryPort, PositionLegRepositoryPort,
                   PositionEventRepositoryPort, PositionExitStateRepositoryPort,
                   PerformanceRepositoryPort, SystemEventRepositoryPort,
                   ModelCostRepositoryPort, ReconciliationRunRepositoryPort,
                   StreamCursorRepositoryPort

  policies/ (2)    VetoServicePort, AIRegimeClassifierPort

infrastructure/
  postgres/          PgUnitOfWork, typed repositories, audit log gateway
  exchange/binance/  HTTP client, market data, execution, order event stream,
                     account state, exchange metadata, funding rates, readiness guard
  exchange/ (shared) ExchangeCache, CachedContextProvider, ExchangeDataFeed,
                     IndicatorEnricher, ExchangeReconciliation
  ai/                Anthropic client, veto service, regime classifier,
                     confidence gate, cost tracker, factory
  observability/     Alert evaluator, log notification
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
                stop_tp (shared ATR stop/TP placement, batch + scalar),
                plugin (StrategyPlugin protocol + StrategyPluginAdapter)
  regime/       classify, ensemble, evaluate
  risk/         rule (RiskRule protocol), rules/ (daily_loss, max_positions, min_rr, zero_stop)
  sizing/       fixed_fraction (vectorized position sizing)
  indicators/   compute (13 indicators: RSI, ADX, ATR, EMA×4, BB×3,
                MACD histogram+signal, volume EMA ratio)
  metrics/      summarize (Sharpe, Sortino, Calmar, profit_factor, win_rate,
                expectancy, payoff_ratio, max_consecutive_losses, daily Sharpe;
                all std uses ddof=1)
  pnl/          pnl (scalar_net_pnl — the single PnL implementation),
                entry_price (4 models: close, next_open, vwap_proxy, worst_case),
                partial_fill, liquidation (liquidation_price)
  validation/   cscv (PBO estimation), walk_forward (rolling/expanding windows)
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

Both share `_run_core_pipeline` (sync deterministic stages): regime classify → hysteresis → halted pairs → min-confidence gate → variant resolve → strategy signal generation → phase 2 exit resolution → exits-only mode. The async path then adds AI regime ensemble → AI veto before risk + sizing.

Returns `PipelineResult` containing: regimes, candidates, veto, risk, intents, variants, authority, confidence_raw, per_pair_params.

## Strategy plugin system

Strategies implement `StrategyPlugin` protocol (`compute/kernels/strategy/plugin.py`):

```
StrategyPlugin.signal(states, indicators, prices, settings, ...) → BatchSignalFragment
```

`StrategyPluginAdapter` wraps raw `SignalFunction` (returning `(buy_mask, short_mask)` tuple) into a `StrategyPlugin`.

`StrategyRegistry` (`application/registry/strategy_registry.py`) manages plugin composition:
- Registration order = priority: on a cross-strategy BUY/SHORT conflict the first-registered plugin claims both the action and the attribution
- Built-in plugins: `trend_follow`, `mean_revert`, `vol_revert`
- Within one plugin, short yields to buy on the same row
- Confluence filter: RSI/MACD/volume/EMA/ADX scoring (`compute/kernels/strategy/confluence.py`) — gated on global `StrategyParams` (per-pair scope overrides of `confluence_filter_enabled`/`min_confluence_score` have no effect)
- ATR stop/TP: one shared kernel (`compute/kernels/strategy/stop_tp.py`) used by both the batch registry path and the scalar phase-2 exit resolution: `distance = max(atr * stop_atr_mult, entry * min_stop_distance_pct / 100)`

## Scope resolution

Two-phase per-pair/regime parameter overrides:

- `StrategyScopeResolver` (`config/scope.py`) — `[[scope.strategy]]` overrides strategy params per (pair, regime, strategy)
- `RiskScopeResolver` (`config/risk_scope.py`) — `[[scope.risk]]` overrides risk params per (pair, regime)
- Selectors: `pair`, `regime`, `strategy` — combinable. Priority + specificity resolve conflicts.
- Both strategy and risk scope rules are proportionally scaled during optimization (via `_scale_scope_rules` / `_scale_risk_scope_rules` in `param_tuning.py`); scaled risk values are clamped to search-space bounds.
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
9. `DOJIWICK_DB_URL` (the same env var Atlas reads) overrides `[database].dsn`, so the app and migrations always target one database

`fingerprint_settings()` (SHA-256 config hash) is called by CLI entrypoints after `load_settings()` returns. The `database` section (credentials-bearing DSN) never reaches any fingerprint payload — the canonical JSON is persisted to `bot_config_snapshots`. `INFRA_ONLY_FIELDS` (single source in `config/schema.py`) lists the fields excluded from both the loader's explicit-config enforcement and the trading hash; scope rules flow into `trading_sha256` and rule order is hash-significant.

## Target identity model

`[[universe.targets]]` is the sole identity source — required, non-empty. Each `TargetConfig` has four required, non-empty fields: `target_id`, `display_pair`, `execution_instrument`, `market_data_instrument`.

`trading.active_pairs` and `primary_pair` are runtime-derived from target `display_pair` values, never authored in TOML. `Settings._validate()` enforces `active_pairs == tuple(t.display_pair for t in targets)` and `primary_pair == first target` at construction time.

`resolve_instrument_map()` (in `config/targets.py`) builds a `display_pair → InstrumentId` mapping at the CLI edge. `InstrumentId` is a frozen dataclass with six validated fields: `venue`, `product`, `symbol`, `base_asset`, `quote_asset`, `settle_asset`. `resolve_targets()` (in `application/orchestration/target_resolver.py`) requires `instrument_map` as a keyword argument.

## Persistence identity

Every tick gets a `tick_id` computed from `(config_hash, timestamp, pairs)`. `TickRecord` tracks STARTED→COMPLETED/FAILED lifecycle with `inputs_hash`, `intent_hash`, and `ops_hash` for replay verification.

### Provenance enforcement

All repository write methods require non-empty `venue` and `product` parameters — no defaults on ports or implementations. Pg implementations raise `AdapterError` on empty provenance. Regime `insert_batch()` additionally validates `len(target_ids) == len(pairs)`.

Domain models validate non-empty identity fields in `__post_init__`:
- `BacktestRunRecord`: `config_hash`, `venue`, `product`, `target_ids`, plus `len(target_ids) == len(pairs)`
- `PairTradingState`: `pair`, `target_id`, `venue`, `product`
- `Signal`: `pair`, `target_id`, `signal_type`
- `InstrumentId`: all six fields

No read-path fallbacks — malformed persisted data raises errors. DDL: all provenance columns are `NOT NULL` without `DEFAULT`. `regime_observations` and `decision_outcomes` inserts are idempotent (`ON CONFLICT DO NOTHING`) so a duplicate key cannot halt the tick transaction.

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

## Order & position lifecycle

Three-table order persistence: `order_requests` → `order_reports` → `fills`, plus immutable `order_events` audit trail.

- `OrderRequest` — submitted intent: side, type, qty, price, position_side, reduce_only, close_position, time_in_force, `order_kind` (entry/exit/protective_stop/protective_tp/protective_tp1), `position_leg_id`
- `OrderReport` — exchange-acknowledged state: status, filled_qty, avg_price, cumulative_quote_qty
- `Fill` — individual fill: price, qty, commission, commission_asset, realized_pnl_exchange (written only from WS events / userTrades replay, deduplicated by trade id)
- `OrderEvent` — immutable lifecycle event log

### Pre-persist + fill high-water mark

Order requests are persisted **before** submission (`OrderLedgerService.record_requests`) so the WS consumer can resolve `client_order_id` the moment the first event arrives; reports and events land in the post-execution transaction (`record_results`).

Fill application has one path: `PositionTracker.apply_order_fill(order_request_id, cumulative_filled_qty, ...)`, backed by `order_requests.position_applied_qty` and the `advance_applied_qty` port method (SELECT ... FOR UPDATE). It returns the unapplied delta — zero when the cumulative quantity was already applied — so REST receipts and WS events can both funnel through it and duplicates, races, and replays are no-ops. Position tracking is hedge-native via `PositionLeg` + `PositionEventRecord`; NET legs derive PnL sign from the signed quantity.

### Instrument sync + quantization

`InstrumentSyncService` runs at startup before anything else: exchange metadata is upserted into the `instruments` table for every execution symbol and the process hard-fails if a symbol is missing or not TRADING (every order/position write resolves instruments through the DB).

`DefaultExecutionPlanner` quantizes with exchange filters on the Decimal side: quantities floored to `step_size`, entry deltas below `min_qty`/`min_notional` dropped as SKIPPED, reduce deltas quantized defensively, prices rounded to `tick_size` (protective stops round away from the entry). Helpers live in `domain/numerics.py` (`quantize_qty_to_step`, `round_price_to_tick`, `meets_min_notional`).

## Protective orders (exchange-side exits)

Live positions are protected by resting exchange orders, not in-process logic — a crashed bot keeps its stops.

- `PositionExitState` (entity + `position_exit_state` table) holds the exit plan per leg: stop, TP, TP1 price/fraction, trailing activation/distance, breakeven, max_hold_bars, and a `revision` counter.
- `ProtectiveOrderService` (`application/services/protective_orders.py`) is a desired-state reconciler. `register_entry` derives exit levels at entry fill (mirroring the backtest `_open_position`); `sync` diffs desired specs against resting `dw_p`-prefixed orders (deterministic client ids via `compute_protective_client_order_id(leg, kind, revision)`) — placing missing ones and cancelling unknown/orphaned ones with tolerant cancels; `update_trailing` advances trailing stops per tick (last-price approximation) and executes time exits as reduce-only MARKET orders; `release_for_symbols` clears protection before reduce/flip entries; `on_leg_closed` cancels the surviving sibling.
- STOP_MARKET covers the full leg quantity; TAKE_PROFIT_MARKET covers the remainder after TP1; a trailing amendment bumps `revision`, which changes the client id and lets sync replace the resting stop.
- Binance USDT-M has no native OCO: when one protective order fires, the WS consumer cancels the sibling (`_handle_protective_fill`); the next tick's `sync` is the idempotent net.
- Sync runs at tick end (outside the persistence transaction — a crash between them is healed by startup sync), at startup after reconciliation, and on the periodic reconciliation cadence. The startup order cleanup keeps `dw_p` orders for live legs and cancels the rest. The pending-order guard counts only entry-kind, non-reduce-only requests, so resting protective orders never suppress new entries.

## WS order-event consumer

`OrderEventConsumer.run()` is a supervisor: a dropped socket reconnects with exponential backoff forever; only cancellation and fatal errors (`AdapterError` — persistence down, `AuthenticationError`) propagate to the runner's watchdog. One malformed event is logged and skipped.

Recovery is REST-based: on startup (and after downtime) `replay_trades(symbol, start_ms)` pulls `userTrades` per execution symbol, groups them by order, and feeds them through the same `process_update` → high-water-mark path as live events, so missed fills are applied — not just audited. `StreamCursorRecord` tracks the replay cursor (flushed in batches and at replay end). Listen-key keepalive escalates after N consecutive failures: critical alert, force-close, fresh listen key.

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
- `fee_multiplier` — applied to fee_bps for round-trip cost (2.0 = entry + exit leg)
- `slippage_bps` — execution slippage, adverse on **both** entry and exit fills
- `leverage` — position leverage
- `maintenance_margin_rate` — liquidation margin; must be > 0 whenever leverage > 1

`notional` is margin-based (quantity × entry price); the exchange position is `notional × leverage`, so **fees and funding are charged on the leveraged value** just as gross PnL scales with the leveraged quantity. `scalar_net_pnl()` (`compute/kernels/pnl/pnl.py`) is the single PnL implementation; accrued signed funding enters as `funding_usd`.

## Backtest engine

`run_backtest.py` simulates on a **shared equity pool**: one wallet scalar backs all pairs (as on the exchange); per-row snapshot arrays broadcast the pool so kernel shapes are unchanged. PnL from every close (including TP1 partials) sums into the pool; sizing, ECF, DD scaling, the daily-loss gate, the equity curve, and daily Sharpe all read it.

### Exit engine

`next_*` arrays are forward-shifted (`next_high[t]` = high of bar t+1), so at loop index `bar_idx` the just-closed bar lives at `next_*[bar_idx - 1]`. A position entered at the close of bar t is first exit-checked against bar t+1 — no bar is skipped and nothing looks ahead.

Per just-closed bar, in order:

1. **Funding accrual** — `next_funding[bar_idx - 1]` (summed signed rate settling during that bar) × notional × leverage, sign by direction (longs pay positive rates). Settled into PnL at TP1 (then reset) and at final exit.
2. **Liquidation** — outranks every managed exit (the exchange force-closes before any stop could fill). Long liquidates when `low ≤ liquidation_price`; fill at the liquidation price capped by the bar open; loss capped at the tranche margin.
3. **Trailing stop update**, then **TP1** (partial take-profit: fraction of entry quantity at `max(tp1, open)`, stop moved toward entry by `partial_tp_stop_ratio`).
4. **Stop / TP / time exit** with gap-through fills: long stop fills at `min(stop, open)`, short at `max(stop, open)`; TP mirrored (favorable gaps fill better); time exit at the open. Stop wins over TP on the same bar.
5. **END_OF_BACKTEST** — at a pair's last active bar an open position closes at the just-closed bar's close (the following bar is never exit-checked and never accrues funding).

### Historical funding pipeline

`funding_mode = "historical"` loads settled funding events through the same triple as candles: `BinanceFundingRateProvider` → `CachingFundingRateFetcher` → `PgFundingRateRepository` (`funding_rates` table, immutable upserts). `backtest_builder._bin_funding` bins events into per-bar summed rates (searchsorted + add.at) aligned exactly like `next_prices`; head/tail coverage gaps are hard errors, interior gaps warn. Funding is backtest-only — live funding is settled by the exchange and absorbed by reconciliation.

### Risk protection stack

Applied in order per new position entry:

1. **ECF scaling** — proportional size reduction when pool equity < SMA (never blocks)
2. **DD scaling** — sqrt-curve reduction with floor; combined as `min(dd_scale, ecf_scale)` (never blocks)
3. **max_loss_per_trade_pct** — caps leveraged risk per individual trade
4. **max_portfolio_risk_pct** — flat fraction of the shared pool capping total leveraged risk (no pair-count scaling)
5. **max_notional_usd** — absolute position size cap (in sizing kernel)

Portfolio risk allocation uses two-pass fair allocation: pass 1 collects all candidate entries (DD/ECF + per-trade cap applied); pass 2 applies the portfolio cap proportionally across ALL pending entries — no pair-ordering bias.

The decision pipeline does not pass leverage to `size_intents()` (defaults to 1.0); per-trade and portfolio caps account for leverage via `cost.leverage`.

Two drawdown measures: `max_drawdown_pct` (per-trade, inflates for multi-pair systems) and `max_portfolio_drawdown_pct` (bar-level pool equity, always tracked). `effective_max_drawdown_pct` prefers the portfolio measure; the objective and gate use it.

## Optimization pipeline

Optuna-based hyperparameter search in `application/use_cases/optimization/`:

- `OptunaRunner` (`runner.py`): TPE sampler (multivariate + constant-liar for parallel workers), trial count from `config.toml`. `load_study` passes the same sampler/pruner as `create_study`, so `--workers N` keeps the configured distribution. Per-trial exceptions mark the trial FAILED and continue (`catch=(Exception,)`); trial timeouts cancel the objective coroutine; RDBStorage heartbeats + `RetryFailedTrialCallback` reclaim trials from killed workers.
- `HysteresisObjective` / `WalkForwardObjective` (`objective.py`): composite score — graduated min-trades penalty, drawdown score with progressive cliff, capped log trade-frequency bonus, trade density penalty, win-rate bonus, normalized expectancy, consecutive-loss penalty, payoff-ratio reward, log-scaled total PnL, per-regime PF penalty. Walk-forward adds a consistency penalty (std across folds) + regularization toward the baseline.
- Search space (`search_space.py`): 31 base + 17 per-regime scope = 48 dims, plus conditional partial-TP (+2) and confluence (+1) — 51 with the live config. Per-regime scope params (`scope_ranging__*`, `scope_trending__*`, `scope_volatile__*`) tune exits independently per market regime.
- Pruning (`pruning.py`): percentile-based early stopping.
- CLI: `dojiwick optimize --config ... --start ... --end ... [--gate] [--workers N]`

## Research gate

9-criterion anti-overfitting validation in `application/use_cases/validation/`:

1. **CV Sharpe ≥ threshold** — embargoed contiguous K-fold (`cross_validator.py`: `np.array_split` folds, embargo trims fold starts)
2. **PBO ≤ threshold** — CSCV on trade returns (`compute/kernels/validation/cscv.py`)
3. **Walk-forward IS/OOS ratio** — non-positive IS Sharpe yields ratio 0.0; `min_wf_worst_window_sharpe` gates the worst window
4. **Continuous backtest min trades**
5. **Continuous backtest max drawdown** (portfolio measure)
6. **Shock test PF** — real re-simulations with TP −10% / SL +10 % via `perturb_exit_geometry` (not PnL scaling)
7. **Per-regime min PF**
8. **Concentration** — max month % and max trade % of total PnL
9. **Pair robustness** — min N pairs above PF threshold

`gate_evaluator.py` synthesizes criteria into `GateResult` with `rejection_reasons`. CLI (`gate`, `optimize --gate`, `validate --mode full-gate`) exits 2 on rejection. Always use `DefaultGateEvaluator` — it wires all 9 criteria with correct PBO computation.

## Exchange data feed

`ExchangeDataFeed` (`infrastructure/exchange/feed.py`): prices are always REST-refreshed in `ensure_fresh` — the WS user-data stream carries order events, not market data. Feed status: DISCONNECTED → BOOTSTRAPPING → WS_ACTIVE ↔ REST_FALLBACK.

- `ExchangeCache` (`cache.py`): async-locked atomic snapshots (prevents torn reads)
- `CachedContextProvider` (`cached_context_provider.py`): reads cache, enriches with indicators, resets `day_start_equity` at the UTC day boundary
- `IndicatorEnricher` (`indicator_enricher.py`): fetches `candle_lookback` (600) candles, drops the still-forming bar, passes volume through, computes the 13-indicator matrix; rows with insufficient or non-finite data are excluded via the validity mask instead of silently zero-filled; enrichment failure raises rather than producing a silent no-trade tick

Live ticks are aligned to bar close (plus a small settle delay), and live entries get the same ECF/DD sizing reductions the backtest applies (`TickService._update_entry_risk_scale`).

## AI ensemble subsystem

`build_ai_services()` factory (`infrastructure/ai/factory.py`) → `AIServices` bundle:

- `LLMVetoService` — evaluates batch candidates, returns approval mask + reason codes. All error classes (typed `AIServiceError` from the client, OS/timeouts, unexpected) honor `fail_open_on_error`.
- `AnthropicClient` — wraps SDK errors into typed domain errors; one retry layer with backoff+jitter (SDK retries disabled)
- `LLMRegimeClassifier` — adjusts deterministic regime confidence when AI disagrees (never overrides the coarse state)
- `compute_llm_review_mask()` (`confidence_gate.py`) — pre-filter: trending pairs above the confidence threshold skip review; VOLATILE/RANGING always review
- `CostTracker` — daily budget enforcement persisted through `PgModelCostRepository`; today's spend is restored at startup, so the budget survives restarts
- `NullRegimeClassifier` / `NullVetoService` — no-op implementations for AI-disabled mode

## Adapter composition

`config/composition.py` — venue-dispatched adapter builder:

- `ComposedAdapters` frozen dataclass bundles: context_provider, execution_gateway, execution_planner, exchange_metadata, account_state, open_order_port, feed, order_stream
- `_VENUE_BUILDERS` registry: `VenueCode → AdapterBuilder` (Binance-only currently)
- `build_market_data_fetcher()` — backtest CLI helper returning a `MarketDataFetchers` bundle (candles + funding when `funding_mode = "historical"`), optionally wrapped with Postgres caching; cache setup falls back to direct exchange fetch only on connection errors
- Adding a new exchange: implement adapters → readiness guard → builder function → register in `_VENUE_BUILDERS`

## Live safety interlock

`dojiwick run` against mainnet (`testnet = false`) additionally requires `DOJIWICK_LIVE_ACK=1` in the environment. API credentials carry `repr=False` everywhere they are stored; the gateway rejects non-positive quantities/prices before they reach the exchange.

## Domain entities

- `BotState` (`domain/models/entities/bot_state.py`) — mutable circuit breaker and operational state: consecutive_errors, consecutive_losses, daily_trade_count, daily_pnl_usd, circuit_breaker_active, circuit_breaker_until, last_tick_at, last_decay_at, daily_reset_at, recon_health, recon_health_since, recon_frozen_symbols
- `PairTradingState` (`domain/models/entities/pair_state.py`) — mutable per-pair performance tracker: wins, losses, consecutive_losses, last_trade_at, blocked flag, win_rate property
- `PositionExitState` (`domain/models/entities/position_exit_state.py`) — mutable per-leg exit plan with revision counter (see Protective orders)
- `domain/exit_rules.py` — pure trailing-stop and time-exit rules shared conceptually with the backtest exit engine (Hypothesis parity-tested)

## Testing patterns

- **Builders**: `tests/fixtures/factories/` — `ContextBuilder`, `SettingsBuilder`, etc.
- **Fakes**: `tests/fixtures/fakes/` — one fake per port
- **Markers**: `unit`, `integration`, `e2e`, `db`
- **Property-based**: Hypothesis strategies in `tests/fixtures/strategies.py`
