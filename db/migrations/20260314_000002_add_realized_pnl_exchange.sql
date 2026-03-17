-- Add exchange-reported realized PnL to fills and order_events for reconciliation.
ALTER TABLE fills ADD COLUMN realized_pnl_exchange NUMERIC(18,8) DEFAULT NULL;
ALTER TABLE order_events ADD COLUMN realized_pnl_exchange NUMERIC(18,8) DEFAULT NULL;
