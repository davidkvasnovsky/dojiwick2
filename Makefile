.PHONY: onboard fmt fmt-check lint typecheck test test-db ci ci-db db-status db-diff db-lint db-apply db-apply-test db-hash run backtest optimize

onboard: db-apply

fmt:
	uv run ruff format .

fmt-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	basedpyright

test:
	uv run pytest -q -m "not db"

test-db: db-apply-test
	uv run pytest -q tests/db

ci: fmt-check lint typecheck test

ci-db: test-db

db-status:
	atlas migrate status --config file://db/atlas.hcl --env local

db-diff:
	@test -n "$(name)" || (echo "usage: make db-diff name=<migration_name>" && exit 1)
	atlas migrate diff $(name) --config file://db/atlas.hcl --env local

db-lint:
	atlas migrate lint --config file://db/atlas.hcl --env local

db-apply:
	atlas migrate apply --config file://db/atlas.hcl --env local

db-apply-test:
	atlas migrate apply --config file://db/atlas.hcl --env test

db-hash:
	atlas migrate hash --dir file://db/migrations

run:
	op run --env-file=.env -- uv run dojiwick run $(ARGS)

backtest:
	op run --env-file=.env -- uv run dojiwick backtest $(ARGS)

optimize:
	op run --env-file=.env -- uv run dojiwick optimize $(ARGS)
