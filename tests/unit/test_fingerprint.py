"""Tests for settings fingerprinting."""

from fixtures.factories.infrastructure import default_settings, default_system_settings

from dojiwick.config.fingerprint import fingerprint_settings
from dojiwick.config.schema import (
    DatabaseSettings,
)
from dojiwick.config.scope import (
    ScopeSelector,
    StrategyOverrideValues,
    StrategyScopeResolver,
    StrategyScopeRule,
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


def _resolver_with_rr(rr: float, rule_id: str = "r1") -> StrategyScopeResolver:
    return StrategyScopeResolver(
        rules=(
            StrategyScopeRule(
                id=rule_id,
                priority=10,
                selector=ScopeSelector(pair=None, regime=None, strategy=None),
                values=StrategyOverrideValues(rr_ratio=rr),
            ),
        )
    )


def test_scope_rule_value_changes_trading_hash() -> None:
    """A scope-rule override change must alter trading_sha256 (rules ARE trading behavior)."""
    base = default_settings().model_copy(update={"strategy_scope": _resolver_with_rr(2.0)})
    modified = base.model_copy(update={"strategy_scope": _resolver_with_rr(3.0)})

    assert fingerprint_settings(base).trading_sha256 != fingerprint_settings(modified).trading_sha256


def test_scope_rule_order_is_hash_significant() -> None:
    """Rule order is part of the fingerprint: reordering equal rules changes the hash."""
    r1 = StrategyScopeRule(
        id="a",
        priority=10,
        selector=ScopeSelector(pair=None, regime=None, strategy=None),
        values=StrategyOverrideValues(rr_ratio=2.0),
    )
    r2 = StrategyScopeRule(
        id="b",
        priority=20,
        selector=ScopeSelector(pair=None, regime=None, strategy=None),
        values=StrategyOverrideValues(rr_ratio=3.0),
    )
    fwd = default_settings().model_copy(update={"strategy_scope": StrategyScopeResolver(rules=(r1, r2))})
    rev = default_settings().model_copy(update={"strategy_scope": StrategyScopeResolver(rules=(r2, r1))})

    assert fingerprint_settings(fwd).trading_sha256 != fingerprint_settings(rev).trading_sha256
