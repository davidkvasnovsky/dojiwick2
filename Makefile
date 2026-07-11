.PHONY: onboard up down sh docker-ci fmt fmt-check lint typecheck test test-db ci ci-db db-status db-diff db-lint db-apply db-apply-test db-hash run backtest optimize

# op loads .env on the host; inside the tooling container op is absent and compose provides the env
ENVRUN := $(if $(shell command -v op 2>/dev/null),op run --env-file=.env --,)

onboard: up db-apply

up:
	docker compose up --wait postgres

down:
	docker compose down

sh:
	docker compose run --rm tooling bash

docker-ci:
	docker compose up -d postgres &
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
	$(ENVRUN) uv run pytest -q tests/db

ci: fmt-check lint typecheck test

ci-db: test-db

db-status:
	$(ENVRUN) atlas migrate status --config file://db/atlas.hcl --env local

# db-diff and db-lint need the docker:// dev database — run on the host, not inside tooling
db-diff:
	@test -n "$(name)" || (echo "usage: make db-diff name=<migration_name>" && exit 1)
	$(ENVRUN) atlas migrate diff $(name) --config file://db/atlas.hcl --env local

db-lint:
	$(ENVRUN) atlas migrate lint --config file://db/atlas.hcl --env local

db-apply:
	$(ENVRUN) atlas migrate apply --config file://db/atlas.hcl --env local

db-apply-test:
	$(ENVRUN) atlas migrate apply --config file://db/atlas.hcl --env test

db-hash:
	atlas migrate hash --dir file://db/migrations

run:
	$(ENVRUN) uv run dojiwick run $(ARGS)

backtest:
	$(ENVRUN) uv run dojiwick backtest $(ARGS)

optimize:
	$(ENVRUN) uv run dojiwick optimize $(ARGS)
