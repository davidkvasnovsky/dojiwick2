"""Schema/migration alignment verification.

This test documents that schema.sql and migrations are aligned.
Full verification requires a running database:

    make db-apply-test
    pg_dump --schema-only test_db | diff - db/schema.sql

The migration directory was squashed on 2026-07-11 to a single baseline
regenerated from db/schema.sql (the canonical schema). Historical increments
had duplicate Atlas 1.x versions and collided with the regenerated baseline
on fresh databases.
"""

from pathlib import Path


def test_schema_file_exists() -> None:
    """Canonical schema file must exist."""
    schema = Path(__file__).resolve().parents[2] / "db" / "schema.sql"
    assert schema.exists(), "db/schema.sql not found"


def test_migration_files_exist() -> None:
    """Migrations dir must hold the squashed baseline plus the Atlas hash file."""
    migrations_dir = Path(__file__).resolve().parents[2] / "db" / "migrations"
    assert any(migrations_dir.glob("*_v3_baseline.sql")), "baseline migration missing"
    assert (migrations_dir / "atlas.sum").exists(), "atlas.sum missing"
