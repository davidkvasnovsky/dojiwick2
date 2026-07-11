# Dojiwick

Batch-first deterministic trading engine for crypto futures.

## Quickstart

Prerequisites: Docker (Compose v2+), [uv](https://docs.astral.sh/uv/), [Atlas](https://atlasgo.io/), [1Password CLI](https://developer.1password.com/docs/cli/) (`op`).

```bash
cp .env.example .env      # fill in op:// secret references
uv sync                   # host venv for local development
make onboard              # start postgres + apply migrations
```

## Working on the host

```bash
make ci                                                        # fmt-check + lint + typecheck + unit tests
make test-db                                                   # DB tests (applies migrations first)
make backtest ARGS="--config config.toml --start 2025-01-01 --end 2025-06-01"
make optimize ARGS="--config config.toml --start 2025-01-01 --end 2025-06-01 --gate"
make run ARGS="--config config.toml"                           # live loop, secrets via op run
```

Secrets (`op run`) are injected only for `run`/`backtest`/`optimize`/`gate`/`validate`; pytest and Atlas targets receive just the DB URLs. Mainnet (`testnet = false`) additionally requires `DOJIWICK_LIVE_ACK=1`. CLI exit codes: 0 success, 1 error, 2 research-gate rejection.

## Working in containers

`docker compose up` starts only postgres; the app and tooling services sit behind profiles.

```bash
make sh                                              # shell in the tooling container (uv, make, atlas)
make docker-ci                                       # full quality gate in the container, same as CI
op run --env-file=.env -- docker compose up -d app   # live trading loop (profile: live)
docker compose run --rm app backtest --config config.toml --start 2025-01-01 --end 2025-06-01
```

The runtime image bakes the package with the `postgres`/`ai`/`exchange` extras into `/opt/venv` (optimization deps stay out of the live image); `config.toml` is bind-mounted, secrets come from the environment. CI runs the same `make ci` / `make ci-db` inside the tooling image.
