"""Tests for AI service factory."""

import os
from unittest.mock import patch

import pytest

from dojiwick.domain.errors import ConfigurationError
from dojiwick.infrastructure.ai.factory import build_ai_services
from fixtures.factories.infrastructure import default_ai_settings
from fixtures.fakes.clock import FixedClock


class TestAIFactory:
    def test_disabled_returns_none(self) -> None:
        settings = default_ai_settings(enabled=False)
        clock = FixedClock()
        result = build_ai_services(settings, clock)
        assert result.veto_service is None
        assert result.regime_classifier is None

    def test_no_api_key_returns_none(self) -> None:
        settings = default_ai_settings(
            enabled=True,
            veto_enabled=True,
            veto_model="claude-sonnet-4-6",
            regime_model="claude-sonnet-4-6",
        )
        clock = FixedClock()
        with patch.dict(os.environ, {}, clear=True):
            result = build_ai_services(settings, clock)
        assert result.veto_service is None
        assert result.regime_classifier is None

    def test_api_key_present_builds_veto(self) -> None:
        settings = default_ai_settings(
            enabled=True,
            veto_enabled=True,
            veto_model="claude-sonnet-4-6",
            regime_model="claude-sonnet-4-6",
            regime_enabled=False,
        )
        clock = FixedClock()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
            result = build_ai_services(settings, clock)
        assert result.veto_service is not None
        assert result.regime_classifier is None

    def test_api_key_present_builds_both(self) -> None:
        settings = default_ai_settings(
            enabled=True,
            veto_enabled=True,
            regime_enabled=True,
            veto_model="claude-sonnet-4-6",
            regime_model="claude-sonnet-4-6",
        )
        clock = FixedClock()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
            result = build_ai_services(settings, clock)
        assert result.veto_service is not None
        assert result.regime_classifier is not None

    def test_empty_model_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="ai.veto_model must be set"):
            default_ai_settings(
                enabled=True,
                veto_enabled=True,
                veto_model="",
            )

    def test_disabled_veto_allows_empty_model(self) -> None:
        settings = default_ai_settings(
            enabled=True,
            veto_enabled=False,
            regime_enabled=False,
        )
        clock = FixedClock()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
            result = build_ai_services(settings, clock)
        assert result.veto_service is None
        assert result.regime_classifier is None
