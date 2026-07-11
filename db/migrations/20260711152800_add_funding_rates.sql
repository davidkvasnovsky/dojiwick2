-- Create "funding_rates" table
CREATE TABLE "public"."funding_rates" (
  "id" bigserial NOT NULL,
  "venue" text NOT NULL,
  "product" text NOT NULL,
  "symbol" text NOT NULL,
  "funding_time" timestamptz NOT NULL,
  "funding_rate" numeric(12,10) NOT NULL,
  "created_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("id"),
  CONSTRAINT "funding_rates_unique" UNIQUE ("venue", "product", "symbol", "funding_time"),
  CONSTRAINT "funding_rates_rate_check" CHECK ((funding_rate > ('-1'::integer)::numeric) AND (funding_rate < (1)::numeric))
);
-- Create index "idx_funding_rates_symbol_time" to table: "funding_rates"
CREATE INDEX "idx_funding_rates_symbol_time" ON "public"."funding_rates" ("symbol", "funding_time" DESC);
