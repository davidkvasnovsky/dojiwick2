-- Add native fee tracking columns to order_events.
-- fee_asset: the asset in which the exchange charged fees (e.g. BNB, USDT)
-- native_fee_amount: the fee amount in the native fee asset

ALTER TABLE order_events ADD COLUMN fee_asset TEXT NOT NULL DEFAULT '';
ALTER TABLE order_events ADD COLUMN native_fee_amount NUMERIC(18,8) NOT NULL DEFAULT 0;
ALTER TABLE order_events ADD CONSTRAINT order_events_native_fee_amount_check CHECK (native_fee_amount >= 0);
