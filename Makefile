.PHONY: onboard up down sh docker-ci fmt fmt-check lint typecheck test test-db ci ci-db db-status db-diff db-lint db-apply db-apply-test db-hash run backtest optimize gate validate

# Secret resolution (op run materializes Binance/Anthropic keys from op://
# refs) is limited to commands that talk to the exchange or AI. On the host
# op loads .env; inside the tooling container op is absent and compose
# provides the env.
SECRETS_RUN := $(if $(shell command -v op 2>/dev/null),op run --env-file=.env --,)
# pytest/atlas need only the plain DB URLs from .env -- never the secrets
DB_ENV := $(if $(wildcard .env),env $(shell grep -Ehs '^DOJIWICK_(TEST_)?DB_URL=' .env),)

onboard: up db-apply

up:
	docker compose up --wait postgres

down:
	docker compose down

sh:
	docker compose run --rm tooling bash

docker-ci:
	docker compose up -d --wait postgres
	docker compose build tooling
	docker compose run --rm tooling make ci
	docker compose run --rm tooling make ci-db
	docker compose build app

fmt:
	uv run ruff format .

fmt-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run basedpyright

test:
	uv run pytest -q -m "not db"

test-db: db-apply-test
	$(DB_ENV) uv run pytest -q tests/db

ci: fmt-check lint typecheck test

ci-db: test-db

db-status:
	$(DB_ENV) atlas migrate status --config file://db/atlas.hcl --env local

# db-diff and db-lint need the docker:// dev database — run on the host, not inside tooling
db-diff:
	@test -n "$(name)" || (echo "usage: make db-diff name=<migration_name>" && exit 1)
	$(DB_ENV) atlas migrate diff $(name) --config file://db/atlas.hcl --env local

db-lint:
	$(DB_ENV) atlas migrate lint --config file://db/atlas.hcl --env local

db-apply:
	$(DB_ENV) atlas migrate apply --config file://db/atlas.hcl --env local

db-apply-test:
	$(DB_ENV) atlas migrate apply --config file://db/atlas.hcl --env test

db-hash:
	atlas migrate hash --dir file://db/migrations

run:
	$(SECRETS_RUN) uv run dojiwick run $(ARGS)

backtest:
	$(SECRETS_RUN) uv run dojiwick backtest $(ARGS)

optimize:
	$(SECRETS_RUN) uv run dojiwick optimize $(ARGS)

gate:
	$(SECRETS_RUN) uv run dojiwick gate $(ARGS)

validate:
	$(SECRETS_RUN) uv run dojiwick validate $(ARGS)
