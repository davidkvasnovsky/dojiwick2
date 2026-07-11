# Remaining operations (post-overhaul)

The comprehensive architecture overhaul (phases 1–6) is complete and committed
on `main`. What remains is operational — cloud spend and live-exchange access,
run when ready:

## 1. v41 optimization study (Hetzner)

```bash
./scripts/hetzner-optimize.sh --config config.toml --start 2019-09-08 --end <train_end> --gate
```

- `config.toml` `study_name` is already `dojiwick_v41`.
- **Never promote v38/v39/v40 params** — all were tuned under the pre-overhaul
  cost physics (fees/funding understated ~2.85× at leverage, per-pair wallets,
  no liquidation/funding modeling). Their numbers are optimistic history.
  Expect lower leverage to win under honest physics.
- Evaluation protocol (holdout OOS + gate on the candidate's own config) is in
  the memory note `v38-promotion-2026-07` and `research/results/`.
- Baseline for comparison: `research/results/v41_baseline_*.csv` (current
  promoted params under the new engine, 2022→2026).

## 2. Binance testnet smoke (before any mainnet run)

Run `make run` with `testnet = true` long enough to observe:
- instrument sync succeeds on a fresh DB
- an entry fills and protective STOP + TP appear on the exchange
- a trailing revision replaces the resting stop
- killing the WS mid-run reconnects and replays missed fills

Mainnet additionally requires `DOJIWICK_LIVE_ACK=1`.

## Practical notes

- **Docker-only host**: never run `uv`/`pytest` on this Mac —
  `docker compose run --rm tooling …`.
- **Atlas migrations from the host** (no atlas binary): throwaway dev DB +
  pinned atlas image:
  ```bash
  docker run -d --rm --name atlas-dev -e POSTGRES_PASSWORD=dev -p 5499:5432 postgres:18.4-alpine
  sleep 4
  docker run --rm -v "$PWD/db:/db" arigaio/atlas:1.2.3-community migrate diff <name> \
    --dir file:///db/migrations --to file:///db/schema.sql \
    --dev-url "postgres://postgres:dev@host.docker.internal:5499/postgres?sslmode=disable" \
    --format '{{ sql . "  " }}'
  docker stop atlas-dev
  ```
  Then `docker compose run --rm tooling sh -c "make db-apply && make db-apply-test"`.
- **Never edit applied migration files** — always a new `migrate diff`.
