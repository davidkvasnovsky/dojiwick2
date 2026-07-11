-- Modify "backtest_runs" table
ALTER TABLE "public"."backtest_runs" ADD COLUMN "daily_sharpe" double precision NOT NULL DEFAULT 0.0;
