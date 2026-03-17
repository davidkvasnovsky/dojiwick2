"""Tests for FeatureFlags dataclass and Settings wiring."""

import pytest
from pydantic import ValidationError

from dojiwick.config.schema import FeatureFlags
from fixtures.factories.infrastructure import SettingsBuilder, default_settings


class TestFeatureFlags:
    def test_all_flags_default_to_safe(self) -> None:
        flags = FeatureFlags()
        assert flags.ai_veto_shadow_mode is False
        assert flags.ai_regime_shadow_mode is False
        assert flags.exits_only_mode is False
        assert flags.global_halt is False
        assert flags.disable_llm is False
        assert flags.halted_pairs == ()

    def test_settings_includes_flags(self) -> None:
        settings = default_settings()
        assert settings.flags == FeatureFlags()

    def test_settings_builder_with_global_halt(self) -> None:
        settings = SettingsBuilder().with_global_halt().build()
        assert settings.flags.global_halt is True

    def test_settings_builder_with_exits_only(self) -> None:
        settings = SettingsBuilder().with_exits_only().build()
        assert settings.flags.exits_only_mode is True

    def test_settings_builder_with_halted_pairs(self) -> None:
        settings = SettingsBuilder().with_halted_pairs(("BTC/USDC",)).build()
        assert settings.flags.halted_pairs == ("BTC/USDC",)

    def test_feature_flags_is_frozen(self) -> None:
        flags = FeatureFlags()
        with pytest.raises(ValidationError):
            flags.global_halt = True  # type: ignore[misc]
