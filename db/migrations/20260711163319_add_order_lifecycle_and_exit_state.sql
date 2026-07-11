-- Create enum type "order_kind"
CREATE TYPE "public"."order_kind" AS ENUM ('entry', 'exit', 'protective_stop', 'protective_tp', 'protective_tp1');
-- Modify "order_requests" table
ALTER TABLE "public"."order_requests" ADD COLUMN "order_kind" "public"."order_kind" NOT NULL DEFAULT 'entry', ADD COLUMN "position_leg_id" bigint NULL, ADD COLUMN "position_applied_qty" numeric(18,8) NOT NULL DEFAULT 0, ADD
CONSTRAINT "order_requests_position_leg_id_fkey" FOREIGN KEY ("position_leg_id") REFERENCES "public"."position_legs" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION;
-- Create index "idx_order_requests_protective" to table: "order_requests"
CREATE INDEX "idx_order_requests_protective" ON "public"."order_requests" ("position_leg_id") WHERE (order_kind = ANY (ARRAY['protective_stop'::public.order_kind, 'protective_tp'::public.order_kind, 'protective_tp1'::public.order_kind]));
-- Create "position_exit_state" table
CREATE TABLE "public"."position_exit_state" (
  "position_leg_id" bigint NOT NULL,
  "is_long" boolean NOT NULL,
  "entry_price" numeric(20,8) NOT NULL,
  "stop_price" numeric(20,8) NOT NULL,
  "original_stop" numeric(20,8) NOT NULL,
  "take_profit_price" numeric(20,8) NOT NULL,
  "trailing_activation_price" numeric(20,8) NOT NULL DEFAULT 0,
  "trailing_distance" numeric(20,8) NOT NULL DEFAULT 0,
  "breakeven_price" numeric(20,8) NOT NULL DEFAULT 0,
  "extreme_price" numeric(20,8) NOT NULL DEFAULT 0,
  "max_hold_bars" integer NOT NULL DEFAULT 0,
  "bars_held" integer NOT NULL DEFAULT 0,
  "tp1_price" numeric(20,8) NOT NULL DEFAULT 0,
  "tp1_fraction" numeric(6,4) NOT NULL DEFAULT 0,
  "tp1_filled" boolean NOT NULL DEFAULT false,
  "revision" integer NOT NULL DEFAULT 0,
  "updated_at" timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY ("position_leg_id"),
  CONSTRAINT "position_exit_state_position_leg_id_fkey" FOREIGN KEY ("position_leg_id") REFERENCES "public"."position_legs" ("id") ON UPDATE NO ACTION ON DELETE NO ACTION
);
