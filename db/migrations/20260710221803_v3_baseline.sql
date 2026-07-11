-- Create enum type "reconciliation_health"
CREATE TYPE "public"."reconciliation_health" AS ENUM ('normal', 'degraded', 'uncertain', 'halt');
-- Create enum type "position_event_type"
CREATE TYPE "public"."position_event_type" AS ENUM ('open', 'add', 'reduce', 'close');
-- Create enum type "execution_status"
CREATE TYPE "public"."execution_status" AS ENUM ('filled', 'skipped', 'rejected', 'error', 'cancelled');
-- Create enum type "trade_action"
CREATE TYPE "public"."trade_action" AS ENUM ('hold', 'buy', 'short');
-- Create enum type "market_state"
CREATE TYPE "public"."market_state" AS ENUM ('trending_up', 'trending_down', 'ranging', 'volatile');
-- Create "adaptive_calibration_metrics" table
CREATE TABLE "public"."adaptive_calibration_metrics" (
  "id" uuid NOT NULL DEFAULT gen_random_uuid(),
  "regime_idx" integer NOT NULL,
  "metric_name" text NOT NULL,
  "metric_value" double precision NOT NULL,
  "computed_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id")
);
-- Create index "idx_adaptive_calibration_regime" to table: "adaptive_calibration_metrics"
CREATE INDEX "idx_adaptive_calibration_regime" ON "public"."adaptive_calibration_metrics" ("regime_idx", "computed_at" DESC);
-- Create enum type "order_side"
CREATE TYPE "public"."order_side" AS ENUM ('buy', 'sell');
-- Create enum type "order_type"
CREATE TYPE "public"."order_type" AS ENUM ('limit', 'market', 'stop_market', 'stop_limit', 'take_profit_market');
-- Create enum type "order_status"
CREATE TYPE "public"."order_status" AS ENUM ('new', 'partially_filled', 'filled', 'canceled', 'expired', 'rejected');
-- Create enum type "order_event_type"
CREATE TYPE "public"."order_event_type" AS ENUM ('placed', 'partially_filled', 'filled', 'canceled', 'expired', 'rejected');
-- Create enum type "order_time_in_force"
CREATE TYPE "public"."order_time_in_force" AS ENUM ('gtc', 'ioc', 'fok', 'gtx');
-- Create enum type "audit_severity"
CREATE TYPE "public"."audit_severity" AS ENUM ('info', 'warning', 'critical');
-- Create enum type "system_event_severity"
CREATE TYPE "public"."system_event_severity" AS ENUM ('info', 'warning', 'critical');
-- Create enum type "tick_status"
CREATE TYPE "public"."tick_status" AS ENUM ('started', 'completed', 'failed', 'skipped');
-- Create enum type "adaptive_outcome_result"
CREATE TYPE "public"."adaptive_outcome_result" AS ENUM ('win', 'loss', 'breakeven');
-- Create enum type "position_side"
CREATE TYPE "public"."position_side" AS ENUM ('net', 'long', 'short');
-- Create enum type "position_mode"
CREATE TYPE "public"."position_mode" AS ENUM ('one_way', 'hedge');
-- Create enum type "working_type"
CREATE TYPE "public"."working_type" AS ENUM ('mark_price', 'contract_price');
-- Create enum type "decision_authority"
CREATE TYPE "public"."decision_authority" AS ENUM ('deterministic_only', 'deterministic_plus_ai_veto', 'deterministic_plus_ai_regime', 'deterministic_plus_ai_regime_and_veto');
-- Create "pair_trading_state" table
CREATE TABLE "public"."pair_trading_state" (
  "pair" text NOT NULL,
  "target_id" text NOT NULL,
  "venue" text NOT NULL,
  "product" text NOT NULL,
  "wins" integer NOT NULL DEFAULT 0,
  "losses" integer NOT NULL DEFAULT 0,
  "consecutive_losses" integer NOT NULL DEFAULT 0,
  "last_trade_at" timestamptz NULL,
  "blocked" boolean NOT NULL DEFAULT false,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("pair"),
  CONSTRAINT "pair_trading_state_consecutive_losses_check" CHECK (consecutive_losses >= 0),
  CONSTRAINT "pair_trading_state_losses_check" CHECK (losses >= 0),
  CONSTRAINT "pair_trading_state_wins_check" CHECK (wins >= 0)
);
-- Create enum type "close_reason"
CREATE TYPE "public"."close_reason" AS ENUM ('stop_loss', 'take_profit', 'trailing_stop', 'emergency', 'manual', 'replaced', 'liquidation', 'double_fill');
-- Create "adaptive_configs" table
CREATE TABLE "public"."adaptive_configs" (
  "config_idx" integer NOT NULL,
  "params_json" jsonb NOT NULL,
  "source" text NOT NULL DEFAULT '',
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("config_idx")
);
-- Create "system_event_log" table
CREATE TABLE "public"."system_event_log" (
  "id" uuid NOT NULL DEFAULT gen_random_uuid(),
  "component" text NOT NULL,
  "severity" "public"."system_event_severity" NOT NULL,
  "message" text NOT NULL,
  "correlation_id" text NOT NULL DEFAULT '',
  "context" jsonb NULL,
  "occurred_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id")
);
-- Create index "idx_system_event_log_occurred" to table: "system_event_log"
CREATE INDEX "idx_system_event_log_occurred" ON "public"."system_event_log" ("occurred_at" DESC);
-- Create index "idx_system_event_log_severity" to table: "system_event_log"
CREATE INDEX "idx_system_event_log_severity" ON "public"."system_event_log" ("severity") WHERE (severity <> 'info'::public.system_event_severity);
-- Create "stream_cursors" table
CREATE TABLE "public"."stream_cursors" (
  "stream_name" text NOT NULL,
  "last_event_id" text NOT NULL DEFAULT '',
  "last_event_time" timestamptz NULL,
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("stream_name")
);
-- Create enum type "decision_status"
CREATE TYPE "public"."decision_status" AS ENUM ('executed', 'hold', 'blocked_risk', 'vetoed', 'error');
-- Create "audit_log" table
CREATE TABLE "public"."audit_log" (
  "id" bigserial NOT NULL,
  "severity" "public"."audit_severity" NOT NULL DEFAULT 'info',
  "event_type" text NOT NULL,
  "message" text NOT NULL,
  "context" jsonb NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id")
);
-- Create index "idx_audit_log_created" to table: "audit_log"
CREATE INDEX "idx_audit_log_created" ON "public"."audit_log" ("created_at" DESC);
-- Create index "idx_audit_log_severity" to table: "audit_log"
CREATE INDEX "idx_audit_log_severity" ON "public"."audit_log" ("severity") WHERE (severity <> 'info'::public.audit_severity);
-- Create "backtest_runs" table
CREATE TABLE "public"."backtest_runs" (
  "id" bigserial NOT NULL,
  "target_ids" text[] NOT NULL,
  "venue" text NOT NULL,
  "product" text NOT NULL,
  "config_hash" text NOT NULL,
  "start_date" timestamptz NOT NULL,
  "end_date" timestamptz NOT NULL,
  "timeframe" text NOT NULL,
  "pairs" text[] NOT NULL,
  "trades" integer NOT NULL,
  "total_pnl_usd" numeric(18,8) NOT NULL,
  "win_rate" double precision NOT NULL,
  "expectancy_usd" numeric(18,8) NOT NULL,
  "sharpe_like" double precision NOT NULL,
  "max_drawdown_pct" double precision NOT NULL,
  "sortino" double precision NOT NULL DEFAULT 0.0,
  "calmar" double precision NOT NULL DEFAULT 0.0,
  "profit_factor" double precision NOT NULL DEFAULT 0.0,
  "max_consecutive_losses" integer NOT NULL DEFAULT 0,
  "payoff_ratio" double precision NOT NULL DEFAULT 0.0,
  "source" text NOT NULL DEFAULT 'backtest',
  "params_json" jsonb NOT NULL DEFAULT '{}',
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id")
);
-- Create index "idx_backtest_runs_config_hash" to table: "backtest_runs"
CREATE INDEX "idx_backtest_runs_config_hash" ON "public"."backtest_runs" ("config_hash");
-- Create index "idx_backtest_runs_created_at" to table: "backtest_runs"
CREATE INDEX "idx_backtest_runs_created_at" ON "public"."backtest_runs" ("created_at" DESC);
-- Create "bot_config_snapshots" table
CREATE TABLE "public"."bot_config_snapshots" (
  "id" uuid NOT NULL DEFAULT gen_random_uuid(),
  "config_hash" text NOT NULL,
  "config_json" jsonb NOT NULL,
  "snapshot_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id")
);
-- Create index "idx_bot_config_snapshots_snapshot" to table: "bot_config_snapshots"
CREATE INDEX "idx_bot_config_snapshots_snapshot" ON "public"."bot_config_snapshots" ("snapshot_at" DESC);
-- Create "bot_state" table
CREATE TABLE "public"."bot_state" (
  "id" bigserial NOT NULL,
  "consecutive_errors" integer NOT NULL DEFAULT 0,
  "consecutive_losses" integer NOT NULL DEFAULT 0,
  "daily_trade_count" integer NOT NULL DEFAULT 0,
  "daily_pnl_usd" numeric(18,8) NOT NULL DEFAULT 0,
  "circuit_breaker_active" boolean NOT NULL DEFAULT false,
  "circuit_breaker_until" timestamptz NULL,
  "last_tick_at" timestamptz NULL,
  "last_decay_at" timestamptz NULL,
  "daily_reset_at" timestamptz NULL,
  "recon_health" "public"."reconciliation_health" NOT NULL DEFAULT 'normal',
  "recon_health_since" timestamptz NULL,
  "recon_frozen_symbols" text[] NOT NULL DEFAULT '{}',
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "bot_state_consecutive_errors_check" CHECK (consecutive_errors >= 0),
  CONSTRAINT "bot_state_consecutive_losses_check" CHECK (consecutive_losses >= 0),
  CONSTRAINT "bot_state_daily_trade_count_check" CHECK (daily_trade_count >= 0),
  CONSTRAINT "bot_state_singleton" CHECK (id = 1)
);
-- Create "bot_state_history" table
CREATE TABLE "public"."bot_state_history" (
  "id" bigserial NOT NULL,
  "consecutive_errors" integer NOT NULL,
  "consecutive_losses" integer NOT NULL,
  "daily_trade_count" integer NOT NULL,
  "daily_pnl_usd" numeric(18,8) NOT NULL,
  "circuit_breaker_active" boolean NOT NULL,
  "circuit_breaker_until" timestamptz NULL,
  "last_tick_at" timestamptz NULL,
  "recon_health" "public"."reconciliation_health" NOT NULL,
  "recon_frozen_symbols" text[] NOT NULL DEFAULT '{}',
  "captured_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id")
);
-- Create index "idx_bot_state_history_captured" to table: "bot_state_history"
CREATE INDEX "idx_bot_state_history_captured" ON "public"."bot_state_history" ("captured_at" DESC);
-- Create "candles" table
CREATE TABLE "public"."candles" (
  "id" bigserial NOT NULL,
  "venue" text NOT NULL,
  "product" text NOT NULL,
  "pair" text NOT NULL,
  "timeframe" text NOT NULL,
  "open_time" timestamptz NOT NULL,
  "open" numeric(20,8) NOT NULL,
  "high" numeric(20,8) NOT NULL,
  "low" numeric(20,8) NOT NULL,
  "close" numeric(20,8) NOT NULL,
  "volume" numeric(24,8) NOT NULL,
  "quote_volume" numeric(24,8) NOT NULL DEFAULT 0,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "candles_unique" UNIQUE ("venue", "product", "pair", "timeframe", "open_time"),
  CONSTRAINT "candles_close_check" CHECK (close > (0)::numeric),
  CONSTRAINT "candles_high_check" CHECK (high > (0)::numeric),
  CONSTRAINT "candles_high_low_check" CHECK (high >= low),
  CONSTRAINT "candles_low_check" CHECK (low > (0)::numeric),
  CONSTRAINT "candles_open_check" CHECK (open > (0)::numeric),
  CONSTRAINT "candles_volume_check" CHECK (volume >= (0)::numeric)
);
-- Create index "idx_candles_pair_timeframe_time" to table: "candles"
CREATE INDEX "idx_candles_pair_timeframe_time" ON "public"."candles" ("pair", "timeframe", "open_time" DESC);
-- Create "strategy_state" table
CREATE TABLE "public"."strategy_state" (
  "target_id" text NOT NULL,
  "venue" text NOT NULL,
  "product" text NOT NULL,
  "pair" text NOT NULL,
  "active_strategy" text NOT NULL,
  "variant" text NOT NULL,
  "state_json" jsonb NULL,
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("pair", "active_strategy", "variant")
);
-- Create "regime_observations" table
CREATE TABLE "public"."regime_observations" (
  "id" bigserial NOT NULL,
  "target_id" text NOT NULL,
  "venue" text NOT NULL,
  "product" text NOT NULL,
  "pair" text NOT NULL,
  "observed_at" timestamptz NOT NULL,
  "coarse_state" integer NOT NULL,
  "confidence" double precision NOT NULL,
  "valid" boolean NOT NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "regime_observations_coarse_state_check" CHECK ((coarse_state >= 1) AND (coarse_state <= 4)),
  CONSTRAINT "regime_observations_confidence_check" CHECK ((confidence >= (0)::double precision) AND (confidence <= (1)::double precision))
);
-- Create index "idx_regime_observations_pair_ts" to table: "regime_observations"
CREATE INDEX "idx_regime_observations_pair_ts" ON "public"."regime_observations" ("pair", "observed_at" DESC);
-- Create index "idx_regime_observations_unique_target_ts" to table: "regime_observations"
CREATE UNIQUE INDEX "idx_regime_observations_unique_target_ts" ON "public"."regime_observations" ("target_id", "observed_at");
-- Create "reconciliation_runs" table
CREATE TABLE "public"."reconciliation_runs" (
  "id" bigserial NOT NULL,
  "run_type" text NOT NULL,
  "status" text NOT NULL DEFAULT 'completed',
  "orphaned_db_count" integer NOT NULL DEFAULT 0,
  "orphaned_exchange_count" integer NOT NULL DEFAULT 0,
  "mismatch_count" integer NOT NULL DEFAULT 0,
  "resolved_count" integer NOT NULL DEFAULT 0,
  "divergences" jsonb NULL,
  "started_at" timestamptz NOT NULL DEFAULT now(),
  "completed_at" timestamptz NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "reconciliation_runs_run_type_check" CHECK (run_type = ANY (ARRAY['startup'::text, 'periodic'::text]))
);
-- Create index "idx_reconciliation_runs_created" to table: "reconciliation_runs"
CREATE INDEX "idx_reconciliation_runs_created" ON "public"."reconciliation_runs" ("created_at" DESC);
-- Create "performance_snapshots" table
CREATE TABLE "public"."performance_snapshots" (
  "id" bigserial NOT NULL,
  "observed_at" timestamptz NOT NULL,
  "equity_usd" numeric(18,8) NOT NULL,
  "unrealized_pnl_usd" numeric(18,8) NOT NULL DEFAULT 0,
  "realized_pnl_usd" numeric(18,8) NOT NULL DEFAULT 0,
  "open_positions" integer NOT NULL DEFAULT 0,
  "drawdown_pct" numeric(10,8) NOT NULL DEFAULT 0,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "performance_snapshots_drawdown_check" CHECK (drawdown_pct >= (0)::numeric),
  CONSTRAINT "performance_snapshots_equity_check" CHECK (equity_usd > (0)::numeric)
);
-- Create index "idx_performance_snapshots_observed" to table: "performance_snapshots"
CREATE INDEX "idx_performance_snapshots_observed" ON "public"."performance_snapshots" ("observed_at" DESC);
-- Create "instruments" table
CREATE TABLE "public"."instruments" (
  "id" bigserial NOT NULL,
  "venue" text NOT NULL,
  "product" text NOT NULL,
  "symbol" text NOT NULL,
  "base_asset" text NOT NULL,
  "quote_asset" text NOT NULL,
  "settle_asset" text NOT NULL,
  "status" text NOT NULL DEFAULT 'trading',
  "price_precision" integer NOT NULL,
  "quantity_precision" integer NOT NULL,
  "base_asset_precision" integer NOT NULL,
  "quote_asset_precision" integer NOT NULL,
  "contract_size" numeric(18,8) NOT NULL DEFAULT 1,
  "margin_asset" text NOT NULL DEFAULT '',
  "metadata" jsonb NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "instruments_unique_symbol" UNIQUE ("venue", "product", "symbol"),
  CONSTRAINT "instruments_product_check" CHECK (product = lower(product)),
  CONSTRAINT "instruments_venue_check" CHECK (venue = lower(venue))
);
-- Create index "idx_instruments_venue_product" to table: "instruments"
CREATE INDEX "idx_instruments_venue_product" ON "public"."instruments" ("venue", "product");
-- Create "position_legs" table
CREATE TABLE "public"."position_legs" (
  "id" bigserial NOT NULL,
  "account" text NOT NULL,
  "instrument_id" bigint NOT NULL,
  "position_side" "public"."position_side" NOT NULL,
  "quantity" numeric(18,8) NOT NULL DEFAULT 0,
  "entry_price" numeric(20,8) NOT NULL DEFAULT 0,
  "unrealized_pnl" numeric(18,8) NOT NULL DEFAULT 0,
  "leverage" integer NOT NULL DEFAULT 1,
  "liquidation_price" numeric(20,8) NULL,
  "opened_at" timestamptz NOT NULL DEFAULT now(),
  "closed_at" timestamptz NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "fk_position_legs_instrument" FOREIGN KEY ("instrument_id") REFERENCES "public"."instruments" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT "position_legs_leverage_check" CHECK (leverage >= 1),
  CONSTRAINT "position_legs_quantity_check" CHECK (quantity >= (0)::numeric)
);
-- Create index "idx_position_legs_account" to table: "position_legs"
CREATE INDEX "idx_position_legs_account" ON "public"."position_legs" ("account") WHERE (closed_at IS NULL);
-- Create index "idx_position_legs_active" to table: "position_legs"
CREATE UNIQUE INDEX "idx_position_legs_active" ON "public"."position_legs" ("account", "instrument_id", "position_side") WHERE (closed_at IS NULL);
-- Create index "idx_position_legs_instrument_id" to table: "position_legs"
CREATE INDEX "idx_position_legs_instrument_id" ON "public"."position_legs" ("instrument_id");
-- Create "adaptive_selections" table
CREATE TABLE "public"."adaptive_selections" (
  "position_leg_id" bigint NOT NULL,
  "regime_idx" integer NOT NULL,
  "config_idx" integer NOT NULL,
  "selected_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("position_leg_id"),
  CONSTRAINT "fk_adaptive_selections_config" FOREIGN KEY ("config_idx") REFERENCES "public"."adaptive_configs" ("config_idx") ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT "fk_adaptive_selections_position_leg" FOREIGN KEY ("position_leg_id") REFERENCES "public"."position_legs" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION
);
-- Create index "idx_adaptive_selections_config_idx" to table: "adaptive_selections"
CREATE INDEX "idx_adaptive_selections_config_idx" ON "public"."adaptive_selections" ("config_idx");
-- Create "adaptive_outcomes" table
CREATE TABLE "public"."adaptive_outcomes" (
  "id" uuid NOT NULL DEFAULT gen_random_uuid(),
  "selection_id" bigint NOT NULL,
  "result" "public"."adaptive_outcome_result" NOT NULL,
  "reward" double precision NOT NULL,
  "recorded_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "fk_adaptive_outcomes_selection" FOREIGN KEY ("selection_id") REFERENCES "public"."adaptive_selections" ("position_leg_id") ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT "adaptive_outcomes_reward_check" CHECK ((reward >= (0)::double precision) AND (reward <= (1)::double precision))
);
-- Create index "idx_adaptive_outcomes_selection" to table: "adaptive_outcomes"
CREATE INDEX "idx_adaptive_outcomes_selection" ON "public"."adaptive_outcomes" ("selection_id");
-- Create "adaptive_posteriors" table
CREATE TABLE "public"."adaptive_posteriors" (
  "regime_idx" integer NOT NULL,
  "config_idx" integer NOT NULL,
  "alpha" double precision NOT NULL DEFAULT 1,
  "beta" double precision NOT NULL DEFAULT 1,
  "n_updates" integer NOT NULL DEFAULT 0,
  "last_decay_at" timestamptz NULL,
  PRIMARY KEY ("regime_idx", "config_idx"),
  CONSTRAINT "fk_adaptive_posteriors_config" FOREIGN KEY ("config_idx") REFERENCES "public"."adaptive_configs" ("config_idx") ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT "adaptive_posteriors_alpha_check" CHECK (alpha > (0)::double precision),
  CONSTRAINT "adaptive_posteriors_beta_check" CHECK (beta > (0)::double precision)
);
-- Create "ticks" table
CREATE TABLE "public"."ticks" (
  "tick_id" text NOT NULL,
  "tick_time" timestamptz NOT NULL,
  "config_hash" text NOT NULL,
  "schema_ver" integer NOT NULL DEFAULT 1,
  "inputs_hash" text NOT NULL,
  "intent_hash" text NOT NULL DEFAULT '',
  "ops_hash" text NOT NULL DEFAULT '',
  "authority" "public"."decision_authority" NOT NULL DEFAULT 'deterministic_only',
  "status" "public"."tick_status" NOT NULL DEFAULT 'started',
  "batch_size" integer NOT NULL DEFAULT 0,
  "duration_ms" integer NULL,
  "error_message" text NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("tick_id")
);
-- Create index "idx_ticks_status" to table: "ticks"
CREATE INDEX "idx_ticks_status" ON "public"."ticks" ("status") WHERE (status = ANY (ARRAY['started'::public.tick_status, 'failed'::public.tick_status]));
-- Create index "idx_ticks_tick_time" to table: "ticks"
CREATE INDEX "idx_ticks_tick_time" ON "public"."ticks" ("tick_time" DESC);
-- Create "decision_traces" table
CREATE TABLE "public"."decision_traces" (
  "id" bigserial NOT NULL,
  "tick_id" text NOT NULL,
  "step_name" text NOT NULL,
  "step_seq" integer NOT NULL,
  "artifacts" jsonb NULL,
  "step_hash" text NOT NULL DEFAULT '',
  "duration_us" integer NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "fk_decision_traces_tick" FOREIGN KEY ("tick_id") REFERENCES "public"."ticks" ("tick_id") ON UPDATE NO ACTION ON DELETE NO ACTION
);
-- Create index "idx_decision_traces_tick_id" to table: "decision_traces"
CREATE INDEX "idx_decision_traces_tick_id" ON "public"."decision_traces" ("tick_id");
-- Create "order_requests" table
CREATE TABLE "public"."order_requests" (
  "id" bigserial NOT NULL,
  "venue" text NOT NULL,
  "product" text NOT NULL,
  "client_order_id" text NOT NULL,
  "instrument_id" bigint NOT NULL,
  "account" text NOT NULL,
  "side" "public"."order_side" NOT NULL,
  "order_type" "public"."order_type" NOT NULL,
  "quantity" numeric(18,8) NOT NULL,
  "price" numeric(20,8) NULL,
  "position_side" "public"."position_side" NOT NULL DEFAULT 'net',
  "reduce_only" boolean NOT NULL DEFAULT false,
  "close_position" boolean NOT NULL DEFAULT false,
  "time_in_force" "public"."order_time_in_force" NOT NULL DEFAULT 'gtc',
  "working_type" "public"."working_type" NOT NULL DEFAULT 'contract_price',
  "price_protect" boolean NOT NULL DEFAULT false,
  "recv_window_ms" integer NULL,
  "tick_id" text NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "order_requests_unique_client_order_scoped" UNIQUE ("venue", "product", "client_order_id"),
  CONSTRAINT "fk_order_requests_instrument" FOREIGN KEY ("instrument_id") REFERENCES "public"."instruments" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT "order_requests_tick_id_fkey" FOREIGN KEY ("tick_id") REFERENCES "public"."ticks" ("tick_id") ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT "order_requests_product_check" CHECK (product = lower(product)),
  CONSTRAINT "order_requests_quantity_check" CHECK (quantity > (0)::numeric),
  CONSTRAINT "order_requests_venue_check" CHECK (venue = lower(venue))
);
-- Create index "idx_order_requests_account" to table: "order_requests"
CREATE INDEX "idx_order_requests_account" ON "public"."order_requests" ("account", "created_at" DESC);
-- Create index "idx_order_requests_instrument_id" to table: "order_requests"
CREATE INDEX "idx_order_requests_instrument_id" ON "public"."order_requests" ("instrument_id");
-- Create index "idx_order_requests_venue_product" to table: "order_requests"
CREATE INDEX "idx_order_requests_venue_product" ON "public"."order_requests" ("venue", "product", "created_at" DESC);
-- Create "fills" table
CREATE TABLE "public"."fills" (
  "id" bigserial NOT NULL,
  "order_request_id" bigint NOT NULL,
  "fill_id" text NOT NULL DEFAULT '',
  "price" numeric(20,8) NOT NULL,
  "quantity" numeric(18,8) NOT NULL,
  "commission" numeric(18,8) NOT NULL DEFAULT 0,
  "commission_asset" text NOT NULL DEFAULT '',
  "realized_pnl_exchange" numeric(18,8) NULL DEFAULT NULL::numeric,
  "filled_at" timestamptz NOT NULL DEFAULT now(),
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "fk_fills_order_request" FOREIGN KEY ("order_request_id") REFERENCES "public"."order_requests" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT "fills_commission_check" CHECK (commission >= (0)::numeric),
  CONSTRAINT "fills_price_check" CHECK (price > (0)::numeric),
  CONSTRAINT "fills_quantity_check" CHECK (quantity > (0)::numeric)
);
-- Create index "idx_fills_filled_at" to table: "fills"
CREATE INDEX "idx_fills_filled_at" ON "public"."fills" ("filled_at" DESC);
-- Create index "idx_fills_order_request_id" to table: "fills"
CREATE INDEX "idx_fills_order_request_id" ON "public"."fills" ("order_request_id");
-- Create index "idx_fills_unique_fill_id" to table: "fills"
CREATE UNIQUE INDEX "idx_fills_unique_fill_id" ON "public"."fills" ("order_request_id", "fill_id") WHERE (fill_id <> ''::text);
-- Create "instrument_filters" table
CREATE TABLE "public"."instrument_filters" (
  "id" bigserial NOT NULL,
  "instrument_id" bigint NOT NULL,
  "min_price" numeric(20,8) NOT NULL,
  "max_price" numeric(20,8) NULL,
  "tick_size" numeric(20,8) NOT NULL,
  "min_qty" numeric(18,8) NOT NULL,
  "max_qty" numeric(18,8) NULL,
  "step_size" numeric(18,8) NOT NULL,
  "min_notional" numeric(18,8) NOT NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "fk_instrument_filters_instrument" FOREIGN KEY ("instrument_id") REFERENCES "public"."instruments" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT "instrument_filters_min_notional_check" CHECK (min_notional >= (0)::numeric),
  CONSTRAINT "instrument_filters_min_price_check" CHECK (min_price >= (0)::numeric),
  CONSTRAINT "instrument_filters_min_qty_check" CHECK (min_qty > (0)::numeric),
  CONSTRAINT "instrument_filters_step_size_check" CHECK (step_size > (0)::numeric),
  CONSTRAINT "instrument_filters_tick_size_check" CHECK (tick_size > (0)::numeric)
);
-- Create index "idx_instrument_filters_unique_instrument" to table: "instrument_filters"
CREATE UNIQUE INDEX "idx_instrument_filters_unique_instrument" ON "public"."instrument_filters" ("instrument_id");
-- Create "model_costs" table
CREATE TABLE "public"."model_costs" (
  "id" bigserial NOT NULL,
  "tick_id" text NOT NULL,
  "model" text NOT NULL,
  "input_tokens" integer NOT NULL,
  "output_tokens" integer NOT NULL,
  "cost_usd" numeric(12,8) NOT NULL,
  "purpose" text NOT NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "fk_model_costs_tick" FOREIGN KEY ("tick_id") REFERENCES "public"."ticks" ("tick_id") ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT "model_costs_cost_usd_check" CHECK (cost_usd >= (0)::numeric),
  CONSTRAINT "model_costs_input_tokens_check" CHECK (input_tokens >= 0),
  CONSTRAINT "model_costs_output_tokens_check" CHECK (output_tokens >= 0)
);
-- Create index "idx_model_costs_created" to table: "model_costs"
CREATE INDEX "idx_model_costs_created" ON "public"."model_costs" ("created_at" DESC);
-- Create index "idx_model_costs_model" to table: "model_costs"
CREATE INDEX "idx_model_costs_model" ON "public"."model_costs" ("model");
-- Create index "idx_model_costs_tick_id" to table: "model_costs"
CREATE INDEX "idx_model_costs_tick_id" ON "public"."model_costs" ("tick_id");
-- Create "order_events" table
CREATE TABLE "public"."order_events" (
  "id" uuid NOT NULL DEFAULT gen_random_uuid(),
  "order_id" bigint NOT NULL,
  "event_type" "public"."order_event_type" NOT NULL,
  "exchange_order_id" text NOT NULL DEFAULT '',
  "filled_quantity" numeric(18,8) NOT NULL DEFAULT 0,
  "fees_usd" numeric(18,8) NOT NULL DEFAULT 0,
  "fee_asset" text NOT NULL DEFAULT '',
  "native_fee_amount" numeric(18,8) NOT NULL DEFAULT 0,
  "realized_pnl_exchange" numeric(18,8) NULL DEFAULT NULL::numeric,
  "detail" text NOT NULL DEFAULT '',
  "occurred_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "fk_order_events_order_request" FOREIGN KEY ("order_id") REFERENCES "public"."order_requests" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT "order_events_fees_usd_check" CHECK (fees_usd >= (0)::numeric),
  CONSTRAINT "order_events_filled_quantity_check" CHECK (filled_quantity >= (0)::numeric),
  CONSTRAINT "order_events_native_fee_amount_check" CHECK (native_fee_amount >= (0)::numeric)
);
-- Create index "idx_order_events_occurred" to table: "order_events"
CREATE INDEX "idx_order_events_occurred" ON "public"."order_events" ("occurred_at" DESC);
-- Create index "idx_order_events_order_id" to table: "order_events"
CREATE INDEX "idx_order_events_order_id" ON "public"."order_events" ("order_id");
-- Create "order_reports" table
CREATE TABLE "public"."order_reports" (
  "id" bigserial NOT NULL,
  "order_request_id" bigint NOT NULL,
  "exchange_order_id" text NOT NULL,
  "status" "public"."order_status" NOT NULL,
  "filled_qty" numeric(18,8) NOT NULL DEFAULT 0,
  "avg_price" numeric(20,8) NULL,
  "cumulative_quote_qty" numeric(18,8) NULL,
  "reported_at" timestamptz NOT NULL DEFAULT now(),
  "created_at" timestamptz NOT NULL DEFAULT now(),
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "order_reports_unique_exchange_order" UNIQUE ("exchange_order_id"),
  CONSTRAINT "fk_order_reports_request" FOREIGN KEY ("order_request_id") REFERENCES "public"."order_requests" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT "order_reports_filled_qty_check" CHECK (filled_qty >= (0)::numeric)
);
-- Create index "idx_order_reports_non_terminal" to table: "order_reports"
CREATE INDEX "idx_order_reports_non_terminal" ON "public"."order_reports" ("status") WHERE (status = ANY (ARRAY['new'::public.order_status, 'partially_filled'::public.order_status]));
-- Create index "idx_order_reports_request_id" to table: "order_reports"
CREATE INDEX "idx_order_reports_request_id" ON "public"."order_reports" ("order_request_id");
-- Create "position_events" table
CREATE TABLE "public"."position_events" (
  "id" bigserial NOT NULL,
  "position_leg_id" bigint NOT NULL,
  "event_type" "public"."position_event_type" NOT NULL,
  "quantity" numeric(18,8) NOT NULL,
  "price" numeric(20,8) NOT NULL,
  "realized_pnl" numeric(18,8) NULL,
  "occurred_at" timestamptz NOT NULL DEFAULT now(),
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "fk_position_events_leg" FOREIGN KEY ("position_leg_id") REFERENCES "public"."position_legs" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT "position_events_price_check" CHECK (price > (0)::numeric),
  CONSTRAINT "position_events_quantity_check" CHECK (quantity > (0)::numeric)
);
-- Create index "idx_position_events_leg_id" to table: "position_events"
CREATE INDEX "idx_position_events_leg_id" ON "public"."position_events" ("position_leg_id");
-- Create index "idx_position_events_occurred" to table: "position_events"
CREATE INDEX "idx_position_events_occurred" ON "public"."position_events" ("occurred_at" DESC);
-- Create "decision_outcomes" table
CREATE TABLE "public"."decision_outcomes" (
  "id" bigserial NOT NULL,
  "target_id" text NOT NULL,
  "venue" text NOT NULL,
  "product" text NOT NULL,
  "pair" text NOT NULL,
  "observed_at" timestamptz NOT NULL,
  "status" "public"."decision_status" NOT NULL,
  "authority" "public"."decision_authority" NOT NULL,
  "reason_code" text NOT NULL,
  "action" "public"."trade_action" NOT NULL,
  "strategy_name" text NOT NULL,
  "strategy_variant" text NOT NULL,
  "confidence" double precision NOT NULL,
  "entry_price" numeric(20,8) NOT NULL DEFAULT 0,
  "stop_price" numeric(20,8) NOT NULL DEFAULT 0,
  "take_profit_price" numeric(20,8) NOT NULL DEFAULT 0,
  "quantity" numeric(18,8) NOT NULL DEFAULT 0,
  "notional_usd" numeric(18,8) NOT NULL DEFAULT 0,
  "config_hash" text NOT NULL,
  "order_id" text NOT NULL DEFAULT '',
  "note" text NOT NULL DEFAULT '',
  "market_state" "public"."market_state" NOT NULL,
  "tick_id" text NOT NULL DEFAULT '',
  "confidence_raw" double precision NOT NULL DEFAULT 0,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "decision_outcomes_confidence_check" CHECK ((confidence >= (0)::double precision) AND (confidence <= (1)::double precision)),
  CONSTRAINT "decision_outcomes_confidence_raw_check" CHECK ((confidence_raw >= (0)::double precision) AND (confidence_raw <= (1)::double precision))
);
-- Create index "idx_decision_outcomes_config_hash" to table: "decision_outcomes"
CREATE INDEX "idx_decision_outcomes_config_hash" ON "public"."decision_outcomes" ("config_hash");
-- Create index "idx_decision_outcomes_pair_ts" to table: "decision_outcomes"
CREATE INDEX "idx_decision_outcomes_pair_ts" ON "public"."decision_outcomes" ("pair", "observed_at" DESC);
-- Create index "idx_decision_outcomes_status" to table: "decision_outcomes"
CREATE INDEX "idx_decision_outcomes_status" ON "public"."decision_outcomes" ("status") WHERE (status = ANY (ARRAY['error'::public.decision_status, 'blocked_risk'::public.decision_status]));
-- Create index "idx_decision_outcomes_target_id" to table: "decision_outcomes"
CREATE INDEX "idx_decision_outcomes_target_id" ON "public"."decision_outcomes" ("target_id") WHERE (target_id <> ''::text);
-- Create index "idx_decision_outcomes_unique_tick_target" to table: "decision_outcomes"
CREATE UNIQUE INDEX "idx_decision_outcomes_unique_tick_target" ON "public"."decision_outcomes" ("tick_id", "target_id") WHERE (tick_id <> ''::text);
-- Create "signals" table
CREATE TABLE "public"."signals" (
  "id" bigserial NOT NULL,
  "target_id" text NOT NULL,
  "venue" text NOT NULL,
  "product" text NOT NULL,
  "pair" text NOT NULL,
  "signal_type" text NOT NULL,
  "priority" integer NOT NULL DEFAULT 0,
  "details" jsonb NULL,
  "detected_at" timestamptz NOT NULL DEFAULT now(),
  "decision_outcome_id" bigint NULL,
  PRIMARY KEY ("id"),
  CONSTRAINT "fk_signals_decision_outcome" FOREIGN KEY ("decision_outcome_id") REFERENCES "public"."decision_outcomes" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION
);
-- Create index "idx_signals_decision_outcome_id" to table: "signals"
CREATE INDEX "idx_signals_decision_outcome_id" ON "public"."signals" ("decision_outcome_id") WHERE (decision_outcome_id IS NOT NULL);
-- Create index "idx_signals_pair_detected" to table: "signals"
CREATE INDEX "idx_signals_pair_detected" ON "public"."signals" ("pair", "detected_at" DESC);
