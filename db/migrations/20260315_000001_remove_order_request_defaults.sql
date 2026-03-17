-- Remove Binance-specific defaults from order_requests venue/product columns.
-- Values must now be supplied explicitly by the repository.

ALTER TABLE order_requests ALTER COLUMN venue DROP DEFAULT;
ALTER TABLE order_requests ALTER COLUMN product DROP DEFAULT;
