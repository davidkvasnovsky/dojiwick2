-- Baseline migration: full schema replacement.
-- This is pre-production so we replace the baseline rather than migrate incrementally.
-- Canonical schema lives at db/schema.sql — this file mirrors it.

-- Dojiwick schema — canonical definition of all tables.
-- Apply with: psql -f db/schema.sql
--
-- Type conventions:
--   NUMERIC           — monetary values (prices, PnL, quantities, fees)
--   DOUBLE PRECISION  — statistical metrics (confidence, reward)
--   TIMESTAMPTZ       — all timestamps (timezone-aware)

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ============================================================
-- ENUM TYPES
-- ============================================================

-- Decision & execution
CREATE TYPE decision_status     AS ENUM ('executed', 'hold', 'blocked_risk', 'vetoed', 'error');
CREATE TYPE decision_authority  AS ENUM ('deterministic_only', 'deterministic_plus_ai_veto', 'deterministic_plus_ai_regime', 'deterministic_plus_ai_regime_and_veto');
CREATE TYPE execution_status    AS ENUM ('filled', 'skipped', 'rejected', 'error', 'cancelled');

-- Trade & market
CREATE TYPE trade_action        AS ENUM ('hold', 'buy', 'short');
CREATE TYPE market_state        AS ENUM ('trending_up', 'trending_down', 'ranging', 'volatile');
CREATE TYPE close_reason        AS ENUM ('stop_loss', 'take_profit', 'trailing_stop', 'emergency', 'manual', 'replaced', 'liquidation', 'double_fill');

-- Order lifecycle
CREATE TYPE order_side          AS ENUM ('buy', 'sell');
CREATE TYPE order_type          AS ENUM ('limit', 'market', 'stop_market', 'stop_limit', 'take_profit_market');
CREATE TYPE order_status        AS ENUM ('new', 'partially_filled', 'filled', 'canceled', 'expired', 'rejected');
CREATE TYPE order_event_type    AS ENUM ('placed', 'partially_filled', 'filled', 'canceled', 'expired', 'rejected');
CREATE TYPE order_time_in_force AS ENUM ('gtc', 'ioc', 'fok', 'gtx');

-- Audit & observability
CREATE TYPE audit_severity          AS ENUM ('info', 'warning', 'critical');
CREATE TYPE system_event_severity   AS ENUM ('info', 'warning', 'critical');

-- Tick lifecycle
CREATE TYPE tick_status            AS ENUM ('started', 'completed', 'failed', 'skipped');

-- Adaptive policy
CREATE TYPE adaptive_outcome_result AS ENUM ('win', 'loss', 'breakeven');

-- Position / execution
CREATE TYPE position_side       AS ENUM ('net', 'long', 'short');
CREATE TYPE position_mode       AS ENUM ('one_way', 'hedge');
CREATE TYPE working_type        AS ENUM ('mark_price', 'contract_price');

-- Reconciliation health
CREATE TYPE reconciliation_health AS ENUM ('normal', 'degraded', 'uncertain', 'halt');

-- Position lifecycle
CREATE TYPE position_event_type AS ENUM ('open', 'add', 'reduce', 'close');

-- ============================================================
-- EXCHANGE METADATA
-- ============================================================

-- Instruments (exchange instrument definitions)
CREATE TABLE instruments (
    id              BIGSERIAL PRIMARY KEY,
    venue           TEXT             NOT NULL CHECK (venue = lower(venue)),
    product         TEXT             NOT NULL CHECK (product = lower(product)),
    symbol          TEXT             NOT NULL,
    base_asset      TEXT             NOT NULL,
    quote_asset     TEXT             NOT NULL,
    settle_asset    TEXT             NOT NULL,
    status          TEXT             NOT NULL DEFAULT 'trading',
    price_precision     INTEGER NOT NULL,
    quantity_precision  INTEGER NOT NULL,
    base_asset_precision  INTEGER NOT NULL,
    quote_asset_precision INTEGER NOT NULL,
    contract_size   NUMERIC(18,8)    NOT NULL DEFAULT 1,
    margin_asset    TEXT             NOT NULL DEFAULT '',
    metadata        JSONB,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT instruments_unique_symbol UNIQUE (venue, product, symbol)
);

CREATE INDEX idx_instruments_venue_product
    ON instruments (venue, product);

-- Instrument filters (price/quantity/notional filter rules per instrument)
CREATE TABLE instrument_filters (
    id              BIGSERIAL PRIMARY KEY,
    instrument_id   BIGINT           NOT NULL,
    min_price       NUMERIC(20,8)    NOT NULL,
    max_price       NUMERIC(20,8),
    tick_size       NUMERIC(20,8)    NOT NULL,
    min_qty         NUMERIC(18,8)    NOT NULL,
    max_qty         NUMERIC(18,8),
    step_size       NUMERIC(18,8)    NOT NULL,
    min_notional    NUMERIC(18,8)    NOT NULL,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT fk_instrument_filters_instrument
        FOREIGN KEY (instrument_id) REFERENCES instruments(id),
    CONSTRAINT instrument_filters_min_price_check  CHECK (min_price >= 0),
    CONSTRAINT instrument_filters_tick_size_check  CHECK (tick_size > 0),
    CONSTRAINT instrument_filters_min_qty_check    CHECK (min_qty > 0),
    CONSTRAINT instrument_filters_step_size_check  CHECK (step_size > 0),
    CONSTRAINT instrument_filters_min_notional_check CHECK (min_notional >= 0)
);

CREATE UNIQUE INDEX idx_instrument_filters_unique_instrument
    ON instrument_filters (instrument_id);

-- ============================================================
-- DECISION & REGIME TRACKING
-- ============================================================

-- Decision outcomes (denormalized — no JSONB blob)
CREATE TABLE decision_outcomes (
    id                  BIGSERIAL PRIMARY KEY,
    target_id           TEXT             NOT NULL,
    venue               TEXT             NOT NULL,
    product             TEXT             NOT NULL,
    pair                TEXT             NOT NULL,
    observed_at         TIMESTAMPTZ      NOT NULL,
    status              decision_status  NOT NULL,
    authority           decision_authority NOT NULL,
    reason_code         TEXT             NOT NULL,
    action              trade_action     NOT NULL,
    strategy_name       TEXT             NOT NULL,
    strategy_variant    TEXT             NOT NULL,
    confidence          DOUBLE PRECISION NOT NULL,
    entry_price         NUMERIC(20,8)    NOT NULL DEFAULT 0,
    stop_price          NUMERIC(20,8)    NOT NULL DEFAULT 0,
    take_profit_price   NUMERIC(20,8)    NOT NULL DEFAULT 0,
    quantity            NUMERIC(18,8)    NOT NULL DEFAULT 0,
    notional_usd        NUMERIC(18,8)    NOT NULL DEFAULT 0,
    config_hash         TEXT             NOT NULL,
    order_id            TEXT             NOT NULL DEFAULT '',
    note                TEXT             NOT NULL DEFAULT '',
    market_state        market_state     NOT NULL,
    tick_id             TEXT             NOT NULL DEFAULT '',
    confidence_raw      DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT decision_outcomes_confidence_check CHECK (confidence BETWEEN 0 AND 1),
    CONSTRAINT decision_outcomes_confidence_raw_check CHECK (confidence_raw BETWEEN 0 AND 1)
);

CREATE INDEX idx_decision_outcomes_pair_ts
    ON decision_outcomes (pair, observed_at DESC);

CREATE INDEX idx_decision_outcomes_status
    ON decision_outcomes (status) WHERE status IN ('error', 'blocked_risk');

CREATE INDEX idx_decision_outcomes_config_hash
    ON decision_outcomes (config_hash);

CREATE UNIQUE INDEX idx_decision_outcomes_unique_tick_target
    ON decision_outcomes (tick_id, target_id) WHERE tick_id != '';

CREATE INDEX idx_decision_outcomes_target_id
    ON decision_outcomes (target_id) WHERE target_id != '';

-- Regime observations
CREATE TABLE regime_observations (
    id              BIGSERIAL PRIMARY KEY,
    target_id       TEXT             NOT NULL,
    venue           TEXT             NOT NULL,
    product         TEXT             NOT NULL,
    pair            TEXT             NOT NULL,
    observed_at     TIMESTAMPTZ      NOT NULL,
    coarse_state    INTEGER          NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    valid           BOOLEAN          NOT NULL,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT regime_observations_coarse_state_check CHECK (coarse_state BETWEEN 1 AND 4),
    CONSTRAINT regime_observations_confidence_check   CHECK (confidence BETWEEN 0 AND 1)
);

CREATE INDEX idx_regime_observations_pair_ts
    ON regime_observations (pair, observed_at DESC);

CREATE UNIQUE INDEX idx_regime_observations_unique_target_ts
    ON regime_observations (target_id, observed_at);

-- Ticks (deterministic tick lifecycle)
CREATE TABLE ticks (
    tick_id         TEXT PRIMARY KEY,
    tick_time       TIMESTAMPTZ      NOT NULL,
    config_hash     TEXT             NOT NULL,
    schema_ver      INTEGER          NOT NULL DEFAULT 1,
    inputs_hash     TEXT             NOT NULL,
    intent_hash     TEXT             NOT NULL DEFAULT '',
    ops_hash        TEXT             NOT NULL DEFAULT '',
    authority       decision_authority NOT NULL DEFAULT 'deterministic_only',
    status          tick_status      NOT NULL DEFAULT 'started',
    batch_size      INTEGER          NOT NULL DEFAULT 0,
    duration_ms     INTEGER,
    error_message   TEXT,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX idx_ticks_tick_time
    ON ticks (tick_time DESC);

CREATE INDEX idx_ticks_status
    ON ticks (status) WHERE status IN ('started', 'failed');

-- Decision traces (per-step audit trail — schema only, no Python code in Phase 1)
CREATE TABLE decision_traces (
    id              BIGSERIAL PRIMARY KEY,
    tick_id         TEXT             NOT NULL,
    step_name       TEXT             NOT NULL,
    step_seq        INTEGER          NOT NULL,
    artifacts       JSONB,
    step_hash       TEXT             NOT NULL DEFAULT '',
    duration_us     INTEGER,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT fk_decision_traces_tick
        FOREIGN KEY (tick_id) REFERENCES ticks(tick_id)
);

CREATE INDEX idx_decision_traces_tick_id
    ON decision_traces (tick_id);

-- ============================================================
-- POSITION LEGS (hedge-native position state)
-- ============================================================

CREATE TABLE position_legs (
    id                  BIGSERIAL PRIMARY KEY,
    account             TEXT             NOT NULL,
    instrument_id       BIGINT           NOT NULL,
    position_side       position_side    NOT NULL,
    quantity            NUMERIC(18,8)    NOT NULL DEFAULT 0,
    entry_price         NUMERIC(20,8)    NOT NULL DEFAULT 0,
    unrealized_pnl      NUMERIC(18,8)    NOT NULL DEFAULT 0,
    leverage            INTEGER          NOT NULL DEFAULT 1,
    liquidation_price   NUMERIC(20,8),
    opened_at           TIMESTAMPTZ      NOT NULL DEFAULT now(),
    closed_at           TIMESTAMPTZ,
    created_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT fk_position_legs_instrument
        FOREIGN KEY (instrument_id) REFERENCES instruments(id),
    CONSTRAINT position_legs_quantity_check  CHECK (quantity >= 0),
    CONSTRAINT position_legs_leverage_check  CHECK (leverage >= 1)
);

-- Partial unique index: only one active leg per (account, instrument_id, position_side)
CREATE UNIQUE INDEX idx_position_legs_active
    ON position_legs (account, instrument_id, position_side) WHERE closed_at IS NULL;

CREATE INDEX idx_position_legs_instrument_id
    ON position_legs (instrument_id);

CREATE INDEX idx_position_legs_account
    ON position_legs (account) WHERE closed_at IS NULL;

-- Position events (immutable lifecycle history per leg)
CREATE TABLE position_events (
    id                  BIGSERIAL PRIMARY KEY,
    position_leg_id     BIGINT               NOT NULL,
    event_type          position_event_type   NOT NULL,
    quantity            NUMERIC(18,8)         NOT NULL,
    price               NUMERIC(20,8)         NOT NULL,
    realized_pnl        NUMERIC(18,8),
    occurred_at         TIMESTAMPTZ           NOT NULL DEFAULT now(),
    created_at          TIMESTAMPTZ           NOT NULL DEFAULT now(),
    CONSTRAINT fk_position_events_leg
        FOREIGN KEY (position_leg_id) REFERENCES position_legs(id),
    CONSTRAINT position_events_quantity_check CHECK (quantity > 0),
    CONSTRAINT position_events_price_check    CHECK (price > 0)
);

CREATE INDEX idx_position_events_leg_id
    ON position_events (position_leg_id);

CREATE INDEX idx_position_events_occurred
    ON position_events (occurred_at DESC);

-- ============================================================
-- ORDER LIFECYCLE (normalized: requests → reports → fills)
-- ============================================================

-- Order requests (submitted order intents)
CREATE TABLE order_requests (
    id                  BIGSERIAL PRIMARY KEY,
    venue               TEXT             NOT NULL CHECK (venue = lower(venue)),
    product             TEXT             NOT NULL CHECK (product = lower(product)),
    client_order_id     TEXT             NOT NULL,
    instrument_id       BIGINT           NOT NULL,
    account             TEXT             NOT NULL,
    side                order_side       NOT NULL,
    order_type          order_type       NOT NULL,
    quantity            NUMERIC(18,8)    NOT NULL,
    price               NUMERIC(20,8),
    position_side       position_side    NOT NULL DEFAULT 'net',
    reduce_only         BOOLEAN          NOT NULL DEFAULT false,
    close_position      BOOLEAN          NOT NULL DEFAULT false,
    time_in_force       order_time_in_force NOT NULL DEFAULT 'gtc',
    working_type        working_type     NOT NULL DEFAULT 'contract_price',
    price_protect       BOOLEAN          NOT NULL DEFAULT false,
    recv_window_ms      INTEGER,
    tick_id             TEXT             REFERENCES ticks(tick_id),
    created_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT fk_order_requests_instrument
        FOREIGN KEY (instrument_id) REFERENCES instruments(id),
    CONSTRAINT order_requests_quantity_check CHECK (quantity > 0),
    CONSTRAINT order_requests_unique_client_order_scoped UNIQUE (venue, product, client_order_id)
);

CREATE INDEX idx_order_requests_instrument_id
    ON order_requests (instrument_id);

CREATE INDEX idx_order_requests_account
    ON order_requests (account, created_at DESC);

CREATE INDEX idx_order_requests_venue_product
    ON order_requests (venue, product, created_at DESC);

-- Order reports (exchange-acknowledged order state)
CREATE TABLE order_reports (
    id                      BIGSERIAL PRIMARY KEY,
    order_request_id        BIGINT           NOT NULL,
    exchange_order_id       TEXT             NOT NULL,
    status                  order_status     NOT NULL,
    filled_qty              NUMERIC(18,8)    NOT NULL DEFAULT 0,
    avg_price               NUMERIC(20,8),
    cumulative_quote_qty    NUMERIC(18,8),
    reported_at             TIMESTAMPTZ      NOT NULL DEFAULT now(),
    created_at              TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT fk_order_reports_request
        FOREIGN KEY (order_request_id) REFERENCES order_requests(id),
    CONSTRAINT order_reports_filled_qty_check CHECK (filled_qty >= 0),
    CONSTRAINT order_reports_unique_exchange_order UNIQUE (exchange_order_id)
);

CREATE INDEX idx_order_reports_request_id
    ON order_reports (order_request_id);

CREATE INDEX idx_order_reports_non_terminal
    ON order_reports (status)
    WHERE status IN ('new', 'partially_filled');

-- Fills (individual fill events)
CREATE TABLE fills (
    id                  BIGSERIAL PRIMARY KEY,
    order_request_id    BIGINT           NOT NULL,
    fill_id             TEXT             NOT NULL DEFAULT '',
    price               NUMERIC(20,8)    NOT NULL,
    quantity            NUMERIC(18,8)    NOT NULL,
    commission          NUMERIC(18,8)    NOT NULL DEFAULT 0,
    commission_asset    TEXT             NOT NULL DEFAULT '',
    realized_pnl_exchange NUMERIC(18,8)  DEFAULT NULL,
    filled_at           TIMESTAMPTZ      NOT NULL DEFAULT now(),
    created_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT fk_fills_order_request
        FOREIGN KEY (order_request_id) REFERENCES order_requests(id),
    CONSTRAINT fills_price_check      CHECK (price > 0),
    CONSTRAINT fills_quantity_check    CHECK (quantity > 0),
    CONSTRAINT fills_commission_check  CHECK (commission >= 0)
);

CREATE INDEX idx_fills_order_request_id
    ON fills (order_request_id);

CREATE INDEX idx_fills_filled_at
    ON fills (filled_at DESC);

CREATE UNIQUE INDEX idx_fills_unique_fill_id
    ON fills (order_request_id, fill_id) WHERE fill_id != '';

-- Order events (lifecycle audit trail per order — aligned to domain contract)
CREATE TABLE order_events (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id            BIGINT           NOT NULL,
    event_type          order_event_type NOT NULL,
    exchange_order_id   TEXT             NOT NULL DEFAULT '',
    filled_quantity     NUMERIC(18,8)    NOT NULL DEFAULT 0,
    fees_usd            NUMERIC(18,8)    NOT NULL DEFAULT 0,
    fee_asset           TEXT             NOT NULL DEFAULT '',
    native_fee_amount   NUMERIC(18,8)   NOT NULL DEFAULT 0,
    realized_pnl_exchange NUMERIC(18,8)  DEFAULT NULL,
    detail              TEXT             NOT NULL DEFAULT '',
    occurred_at         TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT fk_order_events_order_request
        FOREIGN KEY (order_id) REFERENCES order_requests(id),
    CONSTRAINT order_events_filled_quantity_check CHECK (filled_quantity >= 0),
    CONSTRAINT order_events_fees_usd_check        CHECK (fees_usd >= 0),
    CONSTRAINT order_events_native_fee_amount_check CHECK (native_fee_amount >= 0)
);

CREATE INDEX idx_order_events_order_id
    ON order_events (order_id);

CREATE INDEX idx_order_events_occurred
    ON order_events (occurred_at DESC);

-- ============================================================
-- MARKET DATA
-- ============================================================

-- Candles (OHLCV data cache)
CREATE TABLE candles (
    id              BIGSERIAL PRIMARY KEY,
    venue           TEXT             NOT NULL,
    product         TEXT             NOT NULL,
    pair            TEXT             NOT NULL,
    timeframe       TEXT             NOT NULL,
    open_time       TIMESTAMPTZ      NOT NULL,
    open            NUMERIC(20,8)    NOT NULL,
    high            NUMERIC(20,8)    NOT NULL,
    low             NUMERIC(20,8)    NOT NULL,
    close           NUMERIC(20,8)    NOT NULL,
    volume          NUMERIC(24,8)    NOT NULL,
    quote_volume    NUMERIC(24,8)    NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT candles_unique       UNIQUE (venue, product, pair, timeframe, open_time),
    CONSTRAINT candles_open_check   CHECK (open > 0),
    CONSTRAINT candles_high_check   CHECK (high > 0),
    CONSTRAINT candles_low_check    CHECK (low > 0),
    CONSTRAINT candles_close_check  CHECK (close > 0),
    CONSTRAINT candles_volume_check CHECK (volume >= 0),
    CONSTRAINT candles_high_low_check CHECK (high >= low)
);

CREATE INDEX idx_candles_pair_timeframe_time
    ON candles (pair, timeframe, open_time DESC);

-- Signals (detected market events)
CREATE TABLE signals (
    id                  BIGSERIAL PRIMARY KEY,
    target_id           TEXT             NOT NULL,
    venue               TEXT             NOT NULL,
    product             TEXT             NOT NULL,
    pair                TEXT             NOT NULL,
    signal_type         TEXT             NOT NULL,
    priority            INTEGER          NOT NULL DEFAULT 0,
    details             JSONB,
    detected_at         TIMESTAMPTZ      NOT NULL DEFAULT now(),
    decision_outcome_id BIGINT,
    CONSTRAINT fk_signals_decision_outcome
        FOREIGN KEY (decision_outcome_id) REFERENCES decision_outcomes(id)
);

CREATE INDEX idx_signals_pair_detected
    ON signals (pair, detected_at DESC);

CREATE INDEX idx_signals_decision_outcome_id
    ON signals (decision_outcome_id) WHERE decision_outcome_id IS NOT NULL;

-- ============================================================
-- OPERATIONAL STATE
-- ============================================================

-- Bot state (singleton row for circuit breaker)
CREATE TABLE bot_state (
    id                      BIGSERIAL PRIMARY KEY,
    consecutive_errors      INTEGER          NOT NULL DEFAULT 0,
    consecutive_losses      INTEGER          NOT NULL DEFAULT 0,
    daily_trade_count       INTEGER          NOT NULL DEFAULT 0,
    daily_pnl_usd           NUMERIC(18,8)    NOT NULL DEFAULT 0,
    circuit_breaker_active  BOOLEAN          NOT NULL DEFAULT false,
    circuit_breaker_until   TIMESTAMPTZ,
    last_tick_at            TIMESTAMPTZ,
    last_decay_at           TIMESTAMPTZ,
    daily_reset_at          TIMESTAMPTZ,
    recon_health            reconciliation_health NOT NULL DEFAULT 'normal',
    recon_health_since      TIMESTAMPTZ,
    recon_frozen_symbols    TEXT[]               NOT NULL DEFAULT '{}',
    updated_at              TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT bot_state_singleton                 CHECK (id = 1),
    CONSTRAINT bot_state_consecutive_errors_check  CHECK (consecutive_errors >= 0),
    CONSTRAINT bot_state_consecutive_losses_check  CHECK (consecutive_losses >= 0),
    CONSTRAINT bot_state_daily_trade_count_check   CHECK (daily_trade_count >= 0)
);

-- Bot state history (append-only audit trail, retain last 30 days via pg_cron)
CREATE TABLE bot_state_history (
    id                      BIGSERIAL PRIMARY KEY,
    consecutive_errors      INTEGER NOT NULL,
    consecutive_losses      INTEGER NOT NULL,
    daily_trade_count       INTEGER NOT NULL,
    daily_pnl_usd           NUMERIC(18,8) NOT NULL,
    circuit_breaker_active  BOOLEAN NOT NULL,
    circuit_breaker_until   TIMESTAMPTZ,
    last_tick_at            TIMESTAMPTZ,
    recon_health            reconciliation_health NOT NULL,
    recon_frozen_symbols    TEXT[] NOT NULL DEFAULT '{}',
    captured_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_bot_state_history_captured
    ON bot_state_history (captured_at DESC);

-- Performance snapshots (periodic equity/PnL)
CREATE TABLE performance_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    observed_at     TIMESTAMPTZ      NOT NULL,
    equity_usd      NUMERIC(18,8)    NOT NULL,
    unrealized_pnl_usd NUMERIC(18,8) NOT NULL DEFAULT 0,
    realized_pnl_usd   NUMERIC(18,8) NOT NULL DEFAULT 0,
    open_positions  INTEGER          NOT NULL DEFAULT 0,
    drawdown_pct    NUMERIC(10,8)    NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT performance_snapshots_equity_check    CHECK (equity_usd > 0),
    CONSTRAINT performance_snapshots_drawdown_check  CHECK (drawdown_pct >= 0)
);

CREATE INDEX idx_performance_snapshots_observed
    ON performance_snapshots (observed_at DESC);

-- Audit log (immutable event log)
CREATE TABLE audit_log (
    id              BIGSERIAL PRIMARY KEY,
    severity        audit_severity   NOT NULL DEFAULT 'info',
    event_type      TEXT             NOT NULL,
    message         TEXT             NOT NULL,
    context         JSONB,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX idx_audit_log_created
    ON audit_log (created_at DESC);

CREATE INDEX idx_audit_log_severity
    ON audit_log (severity) WHERE severity != 'info';

-- Pair trading state (per-pair performance tracking for risk gates)
CREATE TABLE pair_trading_state (
    pair                TEXT PRIMARY KEY,
    target_id           TEXT             NOT NULL,
    venue               TEXT             NOT NULL,
    product             TEXT             NOT NULL,
    wins                INTEGER          NOT NULL DEFAULT 0,
    losses              INTEGER          NOT NULL DEFAULT 0,
    consecutive_losses  INTEGER          NOT NULL DEFAULT 0,
    last_trade_at       TIMESTAMPTZ,
    blocked             BOOLEAN          NOT NULL DEFAULT false,
    created_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT pair_trading_state_wins_check              CHECK (wins >= 0),
    CONSTRAINT pair_trading_state_losses_check             CHECK (losses >= 0),
    CONSTRAINT pair_trading_state_consecutive_losses_check CHECK (consecutive_losses >= 0)
);

-- Strategy state (per-pair/strategy/variant tracking)
CREATE TABLE strategy_state (
    target_id           TEXT             NOT NULL,
    venue               TEXT             NOT NULL,
    product             TEXT             NOT NULL,
    pair                TEXT             NOT NULL,
    active_strategy     TEXT             NOT NULL,
    variant             TEXT             NOT NULL,
    state_json          JSONB,
    updated_at          TIMESTAMPTZ      NOT NULL DEFAULT now(),
    PRIMARY KEY (pair, active_strategy, variant)
);

-- ============================================================
-- SYNC & RECOVERY
-- ============================================================

-- Stream cursors (last-processed event position per stream)
CREATE TABLE stream_cursors (
    stream_name     TEXT PRIMARY KEY,
    last_event_id   TEXT             NOT NULL DEFAULT '',
    last_event_time TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ      NOT NULL DEFAULT now()
);

-- Reconciliation runs (reconciliation results and divergences)
CREATE TABLE reconciliation_runs (
    id                      BIGSERIAL PRIMARY KEY,
    run_type                TEXT             NOT NULL,
    status                  TEXT             NOT NULL DEFAULT 'completed',
    orphaned_db_count       INTEGER          NOT NULL DEFAULT 0,
    orphaned_exchange_count INTEGER          NOT NULL DEFAULT 0,
    mismatch_count          INTEGER          NOT NULL DEFAULT 0,
    resolved_count          INTEGER          NOT NULL DEFAULT 0,
    divergences             JSONB,
    started_at              TIMESTAMPTZ      NOT NULL DEFAULT now(),
    completed_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT reconciliation_runs_run_type_check CHECK (run_type IN ('startup', 'periodic'))
);

CREATE INDEX idx_reconciliation_runs_created
    ON reconciliation_runs (created_at DESC);

-- ============================================================
-- AUDIT & OBSERVABILITY
-- ============================================================

-- System event log (structured system-level audit events — aligned to domain)
CREATE TABLE system_event_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    component       TEXT                    NOT NULL,
    severity        system_event_severity   NOT NULL,
    message         TEXT                    NOT NULL,
    correlation_id  TEXT                    NOT NULL DEFAULT '',
    context         JSONB,
    occurred_at     TIMESTAMPTZ             NOT NULL DEFAULT now()
);

CREATE INDEX idx_system_event_log_occurred
    ON system_event_log (occurred_at DESC);

CREATE INDEX idx_system_event_log_severity
    ON system_event_log (severity) WHERE severity != 'info';

-- Bot config snapshots (configuration audit and rollback)
CREATE TABLE bot_config_snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_hash     TEXT             NOT NULL,
    config_json     JSONB            NOT NULL,
    snapshot_at     TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX idx_bot_config_snapshots_snapshot
    ON bot_config_snapshots (snapshot_at DESC);

-- ============================================================
-- ADAPTIVE POLICY (Thompson Sampling)
-- ============================================================

-- Adaptive configs
CREATE TABLE adaptive_configs (
    config_idx      INTEGER PRIMARY KEY,
    params_json     JSONB            NOT NULL,
    source          TEXT             NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now()
);

-- Adaptive posteriors (Beta distribution state)
CREATE TABLE adaptive_posteriors (
    regime_idx      INTEGER          NOT NULL,
    config_idx      INTEGER          NOT NULL,
    alpha           DOUBLE PRECISION NOT NULL DEFAULT 1,
    beta            DOUBLE PRECISION NOT NULL DEFAULT 1,
    n_updates       INTEGER          NOT NULL DEFAULT 0,
    last_decay_at   TIMESTAMPTZ,
    PRIMARY KEY (regime_idx, config_idx),
    CONSTRAINT adaptive_posteriors_alpha_check CHECK (alpha > 0),
    CONSTRAINT adaptive_posteriors_beta_check CHECK (beta > 0),
    CONSTRAINT fk_adaptive_posteriors_config
        FOREIGN KEY (config_idx) REFERENCES adaptive_configs(config_idx)
);

-- Adaptive selections (which config was used per position leg)
CREATE TABLE adaptive_selections (
    position_leg_id BIGINT PRIMARY KEY,
    regime_idx      INTEGER          NOT NULL,
    config_idx      INTEGER          NOT NULL,
    selected_at     TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT fk_adaptive_selections_position_leg
        FOREIGN KEY (position_leg_id) REFERENCES position_legs(id),
    CONSTRAINT fk_adaptive_selections_config
        FOREIGN KEY (config_idx) REFERENCES adaptive_configs(config_idx)
);

CREATE INDEX idx_adaptive_selections_config_idx
    ON adaptive_selections (config_idx);

-- Adaptive outcomes (observed reward after position closes)
CREATE TABLE adaptive_outcomes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    selection_id    BIGINT                  NOT NULL,
    result          adaptive_outcome_result NOT NULL,
    reward          DOUBLE PRECISION        NOT NULL,
    recorded_at     TIMESTAMPTZ             NOT NULL DEFAULT now(),
    CONSTRAINT fk_adaptive_outcomes_selection
        FOREIGN KEY (selection_id) REFERENCES adaptive_selections(position_leg_id),
    CONSTRAINT adaptive_outcomes_reward_check CHECK (reward BETWEEN 0 AND 1)
);

CREATE INDEX idx_adaptive_outcomes_selection
    ON adaptive_outcomes (selection_id);

-- Adaptive calibration metrics (diagnostic metrics per regime)
CREATE TABLE adaptive_calibration_metrics (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    regime_idx      INTEGER          NOT NULL,
    metric_name     TEXT             NOT NULL,
    metric_value    DOUBLE PRECISION NOT NULL,
    computed_at     TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX idx_adaptive_calibration_regime
    ON adaptive_calibration_metrics (regime_idx, computed_at DESC);

-- ============================================================
-- MODEL COSTS (LLM token usage and cost tracking)
-- ============================================================

CREATE TABLE model_costs (
    id              BIGSERIAL PRIMARY KEY,
    tick_id         TEXT             NOT NULL,
    model           TEXT             NOT NULL,
    input_tokens    INTEGER          NOT NULL,
    output_tokens   INTEGER          NOT NULL,
    cost_usd        NUMERIC(12,8)    NOT NULL,
    purpose         TEXT             NOT NULL,
    created_at      TIMESTAMPTZ      NOT NULL DEFAULT now(),
    CONSTRAINT fk_model_costs_tick
        FOREIGN KEY (tick_id) REFERENCES ticks(tick_id),
    CONSTRAINT model_costs_input_tokens_check  CHECK (input_tokens >= 0),
    CONSTRAINT model_costs_output_tokens_check CHECK (output_tokens >= 0),
    CONSTRAINT model_costs_cost_usd_check      CHECK (cost_usd >= 0)
);

CREATE INDEX idx_model_costs_tick_id
    ON model_costs (tick_id);

CREATE INDEX idx_model_costs_created
    ON model_costs (created_at DESC);

CREATE INDEX idx_model_costs_model
    ON model_costs (model);

-- ============================================================
-- BACKTEST RUNS (backtest result persistence)
-- ============================================================

CREATE TABLE backtest_runs (
    id                  BIGSERIAL PRIMARY KEY,
    target_ids          TEXT[]           NOT NULL,
    venue               TEXT             NOT NULL,
    product             TEXT             NOT NULL,
    config_hash         TEXT             NOT NULL,
    start_date          TIMESTAMPTZ      NOT NULL,
    end_date            TIMESTAMPTZ      NOT NULL,
    timeframe           TEXT             NOT NULL,
    pairs               TEXT[]           NOT NULL,
    trades              INTEGER          NOT NULL,
    total_pnl_usd       NUMERIC(18,8)    NOT NULL,
    win_rate            DOUBLE PRECISION NOT NULL,
    expectancy_usd      NUMERIC(18,8)    NOT NULL,
    sharpe_like         DOUBLE PRECISION NOT NULL,
    max_drawdown_pct    DOUBLE PRECISION NOT NULL,
    sortino             DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    calmar              DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    profit_factor       DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    max_consecutive_losses INTEGER       NOT NULL DEFAULT 0,
    payoff_ratio        DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    source              TEXT             NOT NULL DEFAULT 'backtest',
    params_json         JSONB            NOT NULL DEFAULT '{}',
    created_at          TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX idx_backtest_runs_config_hash
    ON backtest_runs (config_hash);

CREATE INDEX idx_backtest_runs_created_at
    ON backtest_runs (created_at DESC);
