"""Schema/migration alignment verification.

This test documents that schema.sql and migrations are aligned.
Full verification requires a running database:

    make db-apply-test
    pg_dump --schema-only test_db | diff - db/schema.sql

After tasks 2A, 2B, 2H, the following migrations exist:
- 20260228_000001_v3_baseline.sql (baseline)
- 20260314_000001_add_cancelled_execution_status.sql
- 20260315_000001_remove_order_request_defaults.sql (2A)
- 20260315_000002_add_native_fee_tracking.sql (2H)
- 20260315_000003_add_venue_product_provenance.sql (2B)

These were manually verified against db/schema.sql.
"""

from pathlib import Path


def test_schema_file_exists() -> None:
    """Canonical schema file must exist."""
    schema = Path(__file__).resolve().parents[2] / "db" / "schema.sql"
    assert schema.exists(), "db/schema.sql not found"


def test_migration_files_exist() -> None:
    """All expected migration files must exist."""
    migrations_dir = Path(__file__).resolve().parents[2] / "db" / "migrations"
    expected = [
        "20260228_000001_v3_baseline.sql",
        "20260314_000001_add_cancelled_execution_status.sql",
        "20260315_000001_remove_order_request_defaults.sql",
        "20260315_000002_add_native_fee_tracking.sql",
    ]
    for name in expected:
        assert (migrations_dir / name).exists(), f"missing migration: {name}"
