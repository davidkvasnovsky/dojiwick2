"""Configuration loader and strategy scope resolver tests."""

from pathlib import Path

import pytest

from dojiwick.config.composition import build_adapters
from dojiwick.config.loader import load_settings
from fixtures.factories.infrastructure import default_settings
from dojiwick.config.scope import (
    ScopeSelector,
    StrategyOverrideValues,
    StrategyScopeResolver,
    StrategyScopeRule,
    parse_regime,
)
from dojiwick.domain.enums import AdaptiveMode, MarketState, PositionMode
from dojiwick.infrastructure.exchange.binance.constants import BINANCE_VENUE
from dojiwick.domain.errors import ConfigurationError
from dojiwick.domain.models.value_objects.params import StrategyParams
from fixtures.factories.infrastructure import (
    default_adaptive_settings,
    default_exchange_settings,
    default_strategy_params,
    default_universe_settings,
)

_BASE_CONFIG_PATH = Path("config.toml")


def _write_base_config(tmp_path: Path, *, content: str | None = None) -> Path:
    config = tmp_path / "config.toml"
    body = _BASE_CONFIG_PATH.read_text(encoding="utf-8") if content is None else content
    config.write_text(body, encoding="utf-8")
    return config


def test_load_settings_requires_mandatory_sections(tmp_path: Path) -> None:
    """Removing a mandatory section (universe) from TOML raises ConfigurationError."""
    body = _BASE_CONFIG_PATH.read_text(encoding="utf-8")
    # Remove [universe] section entirely (replace header and following lines until next section)
    lines = body.splitlines(keepends=True)
    filtered: list[str] = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped == "[universe]" or stripped.startswith("[[universe."):
            skip = True
            continue
        if skip and stripped.startswith("["):
            skip = False
        if not skip:
            filtered.append(line)
    config = _write_base_config(tmp_path, content="".join(filtered))

    with pytest.raises(ConfigurationError, match="missing required config sections"):
        load_settings(config)


def test_load_settings_fills_defaults_for_missing_keys(tmp_path: Path) -> None:
    """Pydantic fills defaults when optional keys are absent from TOML."""
    body = _BASE_CONFIG_PATH.read_text(encoding="utf-8").replace('log_level = "INFO"\n', "")
    config = _write_base_config(tmp_path, content=body)
    settings = load_settings(config)
    assert settings.system.log_level == "INFO"  # default value


def test_load_settings_rejects_unknown_section(tmp_path: Path) -> None:
    bad = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + "\n[bogus]\nfoo=1\n"
    config = _write_base_config(tmp_path, content=bad)

    with pytest.raises(ConfigurationError, match="unknown config sections"):
        load_settings(config)


def test_load_settings_ignores_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DOJIWICK_DB_DSN", "postgresql://env:env@localhost:5432/env")
    monkeypatch.setenv("DOJIWICK_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DOJIWICK_TICK_INTERVAL", "1")

    settings = load_settings(_write_base_config(tmp_path))

    assert settings.database.dsn == "postgresql://dojiwick:dojiwick@postgres:5432/dojiwick"
    assert settings.system.log_level == "INFO"
    assert settings.system.tick_interval_sec == 90


def test_load_settings_parses_all_sections(tmp_path: Path) -> None:
    settings = load_settings(_write_base_config(tmp_path))

    assert settings.trading.primary_pair == "BTC/USDT"
    assert settings.system.max_ticks == 0
    assert settings.exchange.venue == BINANCE_VENUE
    assert settings.exchange.position_mode == PositionMode.ONE_WAY
    assert settings.universe.quote_asset == "USDT"
    assert settings.adaptive.mode == AdaptiveMode.DISABLED


def test_scope_risk_is_parsed(tmp_path: Path) -> None:
    content = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.risk]]\nid='btc_risk'\npriority=10\npair='BTC/USDT'\nmax_daily_loss_pct=3.0\n"
    )
    config = _write_base_config(tmp_path, content=content)
    settings = load_settings(config)
    rule_ids = {r.id for r in settings.risk_scope.rules}
    assert "btc_risk" in rule_ids


def test_scope_risk_unknown_key_rejected(tmp_path: Path) -> None:
    bad = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.risk]]\nid='r1'\npriority=10\nmax_daily_loss_pct=3.0\nsurprise=1\n"
    )
    config = _write_base_config(tmp_path, content=bad)

    with pytest.raises(ConfigurationError, match="unknown keys in scope.risk"):
        load_settings(config)


def test_scope_risk_pair_must_be_active_pair(tmp_path: Path) -> None:
    bad = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.risk]]\nid='r1'\npriority=10\npair='SOL/USDC'\nmax_daily_loss_pct=3.0\n"
    )
    config = _write_base_config(tmp_path, content=bad)

    with pytest.raises(ConfigurationError, match="must be one of active_pairs"):
        load_settings(config)


def test_scope_risk_regime_must_be_valid(tmp_path: Path) -> None:
    bad = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.risk]]\nid='r1'\npriority=10\npair='BTC/USDT'\nregime='moon'\nmax_daily_loss_pct=3.0\n"
    )
    config = _write_base_config(tmp_path, content=bad)

    with pytest.raises(ConfigurationError, match="regime must be one of"):
        load_settings(config)


def test_scope_risk_duplicate_selector_priority_rejected(tmp_path: Path) -> None:
    bad = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.risk]]\n"
        "id='r1'\n"
        "priority=20\n"
        "pair='BTC/USDT'\n"
        "max_daily_loss_pct=3.0\n"
        "\n[[scope.risk]]\n"
        "id='r2'\n"
        "priority=20\n"
        "pair='BTC/USDT'\n"
        "risk_per_trade_pct=0.5\n"
    )
    config = _write_base_config(tmp_path, content=bad)

    with pytest.raises(ConfigurationError, match=r"duplicate scope.risk selector\+priority"):
        load_settings(config)


def test_scope_risk_priority_resolution(tmp_path: Path) -> None:
    content = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.risk]]\n"
        "id='global_loss'\n"
        "priority=10\n"
        "max_daily_loss_pct=4.0\n"
        "\n[[scope.risk]]\n"
        "id='btc_loss'\n"
        "priority=20\n"
        "pair='BTC/USDT'\n"
        "max_daily_loss_pct=2.0\n"
    )
    settings = load_settings(_write_base_config(tmp_path, content=content))
    resolved = settings.risk_scope.resolve("BTC/USDT", None, settings.risk.params)
    assert resolved.max_daily_loss_pct == 2.0


def test_scope_strategy_unknown_key_rejected(tmp_path: Path) -> None:
    bad = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.strategy]]\nid='s1'\npriority=10\nrr_ratio=2.5\nsurprise=1\n"
    )
    config = _write_base_config(tmp_path, content=bad)

    with pytest.raises(ConfigurationError, match="unknown keys in scope.strategy"):
        load_settings(config)


def test_scope_strategy_pair_must_be_active_pair(tmp_path: Path) -> None:
    bad = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.strategy]]\nid='s1'\npriority=10\npair='SOL/USDC'\nrr_ratio=2.5\n"
    )
    config = _write_base_config(tmp_path, content=bad)

    with pytest.raises(ConfigurationError, match="must be one of active_pairs"):
        load_settings(config)


def test_scope_strategy_regime_must_be_valid(tmp_path: Path) -> None:
    bad = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.strategy]]\nid='s1'\npriority=10\npair='BTC/USDT'\nregime='moon'\nrr_ratio=2.5\n"
    )
    config = _write_base_config(tmp_path, content=bad)

    with pytest.raises(ConfigurationError, match="regime must be one of"):
        load_settings(config)


def test_scope_strategy_duplicate_selector_priority_is_rejected(tmp_path: Path) -> None:
    bad = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.strategy]]\n"
        "id='s1'\n"
        "priority=20\n"
        "pair='BTC/USDT'\n"
        "rr_ratio=2.5\n"
        "\n[[scope.strategy]]\n"
        "id='s2'\n"
        "priority=20\n"
        "pair='BTC/USDT'\n"
        "stop_atr_mult=1.8\n"
    )
    config = _write_base_config(tmp_path, content=bad)

    with pytest.raises(ConfigurationError, match=r"duplicate scope.strategy selector\+priority"):
        load_settings(config)


def test_scope_strategy_priority_resolution_and_specificity(tmp_path: Path) -> None:
    content = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.strategy]]\n"
        "id='default_rr'\n"
        "priority=10\n"
        "rr_ratio=2.1\n"
        "\n[[scope.strategy]]\n"
        "id='btc_rr'\n"
        "priority=20\n"
        "pair='BTC/USDT'\n"
        "rr_ratio=2.4\n"
        "\n[[scope.strategy]]\n"
        "id='btc_down_variant'\n"
        "priority=20\n"
        "pair='BTC/USDT'\n"
        "regime='trending_down'\n"
        "default_variant='defensive'\n"
    )
    settings = load_settings(_write_base_config(tmp_path, content=content))

    down = settings.strategy_scope.resolve("BTC/USDT", MarketState.TRENDING_DOWN, settings.strategy)
    up = settings.strategy_scope.resolve("BTC/USDT", MarketState.TRENDING_UP, settings.strategy)

    assert down.rr_ratio == 2.4
    assert down.default_variant == "defensive"
    assert up.rr_ratio == 2.4
    assert up.default_variant == settings.strategy.default_variant


def test_scope_explain_contains_winners() -> None:
    resolver = StrategyScopeResolver(
        rules=(
            StrategyScopeRule(
                id="base",
                priority=10,
                selector=ScopeSelector(),
                values=StrategyOverrideValues(rr_ratio=2.1),
            ),
            StrategyScopeRule(
                id="btc_pair",
                priority=20,
                selector=ScopeSelector(pair="BTC/USDC"),
                values=StrategyOverrideValues(rr_ratio=2.5),
            ),
            StrategyScopeRule(
                id="btc_down",
                priority=20,
                selector=ScopeSelector(pair="BTC/USDC", regime=MarketState.TRENDING_DOWN),
                values=StrategyOverrideValues(default_variant="defensive"),
            ),
        )
    )

    trace = resolver.explain("BTC/USDC", MarketState.TRENDING_DOWN, default_strategy_params())

    assert trace.matched_rules[0].rule_id == "btc_down"
    assert {winner.field_name for winner in trace.field_winners} == {"rr_ratio", "default_variant"}
    assert trace.resolved.rr_ratio == 2.5
    assert trace.resolved.default_variant == "defensive"


def test_parse_regime_supports_sql_literals() -> None:
    assert parse_regime("trending_up") is MarketState.TRENDING_UP


def test_exchange_settings_validates_recv_window() -> None:
    with pytest.raises(ValueError, match="exchange.recv_window_ms must be >= 1000"):
        default_exchange_settings(recv_window_ms=500)


def test_exchange_settings_validates_rate_limit() -> None:
    with pytest.raises(ValueError, match="exchange.rate_limit_per_sec must be >= 1"):
        default_exchange_settings(rate_limit_per_sec=0)


def test_exchange_settings_validates_backoff_factor() -> None:
    with pytest.raises(ValueError, match="exchange.backoff_factor must be >= 1.0"):
        default_exchange_settings(backoff_factor=0.5)


def test_universe_settings_validates_quote_asset() -> None:
    with pytest.raises(ValueError, match="universe.quote_asset must not be empty"):
        default_universe_settings(quote_asset="")


def test_universe_settings_validates_settle_asset() -> None:
    with pytest.raises(ValueError, match="universe.settle_asset must not be empty"):
        default_universe_settings(settle_asset="")


def test_adaptive_settings_validates_exploration_rate() -> None:
    with pytest.raises(ValueError, match="adaptive.exploration_rate must be in"):
        default_adaptive_settings(exploration_rate=1.5)


def test_adaptive_settings_validates_min_samples() -> None:
    with pytest.raises(ValueError, match="adaptive.min_samples must be >= 1"):
        default_adaptive_settings(min_samples=0)


def test_build_adapters_raises_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_adapters raises ConfigurationError when API credentials are missing."""
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    settings = default_settings()
    with pytest.raises(ConfigurationError, match="Binance API key not set"):
        from dojiwick.infrastructure.system.clock import SystemClock

        build_adapters(settings, clock=SystemClock())


def test_build_adapters_raises_for_unsupported_venue(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_adapters raises ConfigurationError for venues not in the registry."""
    from dojiwick.config import composition

    monkeypatch.setattr(composition, "_VENUE_BUILDERS", {})
    settings = default_settings()
    with pytest.raises(ConfigurationError, match="unsupported exchange venue"):
        from dojiwick.infrastructure.system.clock import SystemClock

        build_adapters(settings, clock=SystemClock())


def test_config_is_parseable() -> None:
    settings = load_settings(Path("config.toml"))
    assert settings.exchange.venue == BINANCE_VENUE
    assert settings.exchange.position_mode == PositionMode.ONE_WAY
    assert settings.universe.quote_asset == "USDT"
    assert settings.adaptive.mode == AdaptiveMode.DISABLED


def test_bool_coerced_to_int_field(tmp_path: Path) -> None:
    """Pydantic coerces bool to int (True→1) for int fields."""
    body = _BASE_CONFIG_PATH.read_text(encoding="utf-8").replace("tick_interval_sec = 90", "tick_interval_sec = true")
    config = _write_base_config(tmp_path, content=body)
    settings = load_settings(config)
    assert settings.system.tick_interval_sec == 1


def test_bool_for_float_field_fails_validation(tmp_path: Path) -> None:
    """Bool coerced to float (False→0.0) fails domain validation (must be > 0)."""
    bad = _BASE_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "connect_timeout_sec = 5.0", "connect_timeout_sec = false"
    )
    config = _write_base_config(tmp_path, content=bad)

    with pytest.raises(ConfigurationError, match="must be > 0"):
        load_settings(config)


def test_coerces_int_to_float(tmp_path: Path) -> None:
    """Int values in TOML are accepted for float fields (e.g. connect_timeout_sec = 5)."""
    body = _BASE_CONFIG_PATH.read_text(encoding="utf-8").replace("connect_timeout_sec = 5.0", "connect_timeout_sec = 5")
    config = _write_base_config(tmp_path, content=body)
    settings = load_settings(config)
    assert settings.database.connect_timeout_sec == 5.0


# --- Strategy scope: strategy dimension ---


def test_scope_strategy_with_strategy_field_parsed(tmp_path: Path) -> None:
    content = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.strategy]]\n"
        "id='custom_strat_exits'\n"
        "priority=500\n"
        "strategy='custom_strat'\n"
        "stop_atr_mult=1.5\n"
        "rr_ratio=1.1\n"
        "max_hold_bars=12\n"
    )
    settings = load_settings(_write_base_config(tmp_path, content=content))
    rule_ids = {r.id for r in settings.strategy_scope.rules}
    assert "custom_strat_exits" in rule_ids
    rule = next(r for r in settings.strategy_scope.rules if r.id == "custom_strat_exits")
    assert rule.selector.strategy == "custom_strat"
    assert rule.values.stop_atr_mult == 1.5
    assert rule.values.max_hold_bars == 12


def test_scope_strategy_phase1_excludes_strategy_rules(tmp_path: Path) -> None:
    """Phase 1 (strategy=None) should NOT match strategy-scoped rules."""
    content = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.strategy]]\n"
        "id='custom_strat_exits'\n"
        "priority=500\n"
        "strategy='custom_strat'\n"
        "stop_atr_mult=1.5\n"
        "rr_ratio=1.1\n"
    )
    settings = load_settings(_write_base_config(tmp_path, content=content))
    # Phase 1: no strategy → default params unchanged by strategy rule
    resolved = settings.strategy_scope.resolve("BTC/USDT", None, settings.strategy)
    assert resolved.stop_atr_mult == settings.strategy.stop_atr_mult


def test_scope_strategy_phase2_includes_strategy_rules(tmp_path: Path) -> None:
    """Phase 2 (strategy='custom_strat') should match strategy-scoped rules."""
    content = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.strategy]]\n"
        "id='custom_strat_exits'\n"
        "priority=500\n"
        "strategy='custom_strat'\n"
        "stop_atr_mult=1.5\n"
        "rr_ratio=1.1\n"
    )
    settings = load_settings(_write_base_config(tmp_path, content=content))
    # Phase 2: with strategy name → strategy rule matches
    resolved = settings.strategy_scope.resolve("BTC/USDT", None, settings.strategy, strategy="custom_strat")
    assert resolved.stop_atr_mult == 1.5
    assert resolved.rr_ratio == 1.1


def test_scope_strategy_wrong_strategy_no_match(tmp_path: Path) -> None:
    """Phase 2 with wrong strategy name should not match."""
    content = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.strategy]]\nid='custom_strat_exits'\npriority=500\nstrategy='custom_strat'\nstop_atr_mult=1.5\n"
    )
    settings = load_settings(_write_base_config(tmp_path, content=content))
    resolved = settings.strategy_scope.resolve("BTC/USDT", None, settings.strategy, strategy="unknown_strategy")
    assert resolved.stop_atr_mult == settings.strategy.stop_atr_mult


def test_scope_strategy_field_empty_string_rejected(tmp_path: Path) -> None:
    bad = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.strategy]]\nid='s1'\npriority=10\nstrategy=''\nrr_ratio=2.5\n"
    )
    config = _write_base_config(tmp_path, content=bad)
    with pytest.raises(ConfigurationError, match="strategy must be a non-empty string"):
        load_settings(config)


def test_scope_strategy_new_override_fields_parsed(tmp_path: Path) -> None:
    content = _BASE_CONFIG_PATH.read_text(encoding="utf-8") + (
        "\n[[scope.strategy]]\nid='test_overrides'\npriority=50\ntrend_pullback_adx_min=25.0\nmax_hold_bars=15\n"
    )
    settings = load_settings(_write_base_config(tmp_path, content=content))
    rule = next(r for r in settings.strategy_scope.rules if r.id == "test_overrides")
    assert rule.values.trend_pullback_adx_min == 25.0
    assert rule.values.max_hold_bars == 15


def test_scope_specificity_with_strategy() -> None:
    """Verify specificity ranks: pair+regime+strategy=4, strategy=1, none=0."""
    assert ScopeSelector().specificity == 0
    assert ScopeSelector(strategy="mean_revert").specificity == 1
    assert ScopeSelector(regime=MarketState.RANGING).specificity == 1
    assert ScopeSelector(pair="BTC/USDC").specificity == 2
    assert ScopeSelector(pair="BTC/USDC", strategy="mean_revert").specificity == 3
    assert ScopeSelector(regime=MarketState.RANGING, strategy="mean_revert").specificity == 2
    assert ScopeSelector(pair="BTC/USDC", regime=MarketState.RANGING).specificity == 3
    assert ScopeSelector(pair="BTC/USDC", regime=MarketState.RANGING, strategy="mean_revert").specificity == 4


def test_scope_canonical_key_includes_strategy() -> None:
    s = ScopeSelector(pair="BTC/USDC", strategy="mean_revert")
    assert "strategy=mean_revert" in s.canonical_key
    s2 = ScopeSelector()
    assert "strategy=*" in s2.canonical_key


def test_scope_strategy_layered_resolution() -> None:
    """pair+regime+strategy rule wins over strategy-only rule."""
    resolver = StrategyScopeResolver(
        rules=(
            StrategyScopeRule(
                id="mr_global",
                priority=100,
                selector=ScopeSelector(strategy="mean_revert"),
                values=StrategyOverrideValues(stop_atr_mult=1.5),
            ),
            StrategyScopeRule(
                id="btc_mr_ranging",
                priority=200,
                selector=ScopeSelector(pair="BTC/USDC", regime=MarketState.RANGING, strategy="mean_revert"),
                values=StrategyOverrideValues(stop_atr_mult=1.2),
            ),
        )
    )
    resolved = resolver.resolve("BTC/USDC", MarketState.RANGING, default_strategy_params(), strategy="mean_revert")
    assert resolved.stop_atr_mult == 1.2


def test_strategy_override_fields_match_strategy_params() -> None:
    """StrategyOverrideValues must mirror StrategyParams overridable fields."""
    from dataclasses import fields as dc_fields

    from dojiwick.config.scope import StrategyOverrideValues

    override_names = {f.name for f in dc_fields(StrategyOverrideValues)}
    param_names = set(StrategyParams.model_fields.keys())
    assert override_names == param_names, (
        f"mismatch: override-only={override_names - param_names}, param-only={param_names - override_names}"
    )


def test_risk_override_fields_match_risk_params() -> None:
    """RiskOverrideValues must mirror RiskParams overridable fields."""
    from dataclasses import fields as dc_fields

    from dojiwick.config.risk_scope import RiskOverrideValues
    from dojiwick.domain.models.value_objects.params import RiskParams

    override_names = {f.name for f in dc_fields(RiskOverrideValues)}
    param_names = set(RiskParams.model_fields.keys())
    assert override_names == param_names, (
        f"mismatch: override-only={override_names - param_names}, param-only={param_names - override_names}"
    )


def test_config_is_parseable_with_strategy_scopes() -> None:
    """Production config.toml should parse cleanly (scope rules are optional)."""
    settings = load_settings(Path("config.toml"))
    assert settings.strategy_scope is not None
