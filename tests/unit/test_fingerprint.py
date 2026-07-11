"""Tests for settings fingerprinting."""

from fixtures.factories.infrastructure import default_settings, default_system_settings

from dojiwick.config.fingerprint import fingerprint_settings
from dojiwick.config.schema import (
    DatabaseSettings,
)


def test_database_section_never_fingerprinted() -> None:
    """The DSN carries credentials: it must not reach either hash or the persisted snapshot."""
    base = default_settings()
    modified = base.model_copy(
        update={"database": DatabaseSettings(dsn="postgresql://other:s3cret@localhost:5432/other")}
    )

    base_fp = fingerprint_settings(base)
    mod_fp = fingerprint_settings(modified)

    assert base_fp.trading_sha256 == mod_fp.trading_sha256
    assert base_fp.sha256 == mod_fp.sha256
    assert "s3cret" not in mod_fp.canonical_json
    assert '"dsn"' not in mod_fp.canonical_json


def test_trading_hash_includes_strategy() -> None:
    """Changing strategy settings changes trading hash."""
    from fixtures.factories.infrastructure import default_strategy_params

    base = default_settings()
    modified = base.model_copy(update={"strategy": default_strategy_params(rr_ratio=5.0)})

    base_fp = fingerprint_settings(base)
    mod_fp = fingerprint_settings(modified)

    assert base_fp.trading_sha256 != mod_fp.trading_sha256


def test_infra_only_log_level_excluded() -> None:
    """Changing log_level doesn't change trading hash."""
    base = default_settings()
    modified = base.model_copy(update={"system": default_system_settings(log_level="DEBUG")})

    base_fp = fingerprint_settings(base)
    mod_fp = fingerprint_settings(modified)

    assert base_fp.trading_sha256 == mod_fp.trading_sha256
    assert base_fp.sha256 != mod_fp.sha256
