"""Tests for explicit USDT/USDC instrument mapping."""

import pytest

from dojiwick.config.schema import TargetConfig
from fixtures.factories.infrastructure import default_universe_settings


def test_target_requires_target_id() -> None:
    """TargetConfig requires non-empty target_id."""
    with pytest.raises(ValueError, match="target_id must be non-empty"):
        TargetConfig(
            target_id="", display_pair="BTC/USDC", execution_instrument="BTCUSDC", market_data_instrument="BTCUSDC"
        )


def test_target_requires_execution_instrument() -> None:
    """TargetConfig requires non-empty execution_instrument."""
    with pytest.raises(ValueError, match="execution_instrument must be non-empty"):
        TargetConfig(
            target_id="btc_usdc", display_pair="BTC/USDC", execution_instrument="", market_data_instrument="BTCUSDC"
        )


def test_target_requires_market_data_instrument() -> None:
    """TargetConfig requires non-empty market_data_instrument."""
    with pytest.raises(ValueError, match="market_data_instrument must be non-empty"):
        TargetConfig(
            target_id="btc_usdc", display_pair="BTC/USDC", execution_instrument="BTCUSDC", market_data_instrument=""
        )


def test_target_config_explicit_instruments() -> None:
    """TargetConfig supports explicit market_data -> execution mapping."""
    t = TargetConfig(
        target_id="btc_usdc",
        display_pair="BTC/USDC",
        market_data_instrument="BTCUSDT",
        execution_instrument="BTCUSDC",
    )
    assert t.market_data_instrument == "BTCUSDT"
    assert t.execution_instrument == "BTCUSDC"


def test_universe_settings_with_targets() -> None:
    """UniverseSettings accepts targets tuple."""
    u = default_universe_settings(
        targets=(
            TargetConfig(
                target_id="btc_usdc",
                display_pair="BTC/USDC",
                market_data_instrument="BTCUSDT",
                execution_instrument="BTCUSDC",
            ),
        ),
    )
    assert len(u.targets) == 1
    assert u.targets[0].display_pair == "BTC/USDC"


def test_universe_settings_duplicate_target_id_rejected() -> None:
    """Duplicate target_id across targets is rejected."""
    with pytest.raises(ValueError, match="target_id must be unique"):
        default_universe_settings(
            targets=(
                TargetConfig(
                    target_id="btc",
                    display_pair="BTC/USDC",
                    execution_instrument="BTCUSDC",
                    market_data_instrument="BTCUSDC",
                ),
                TargetConfig(
                    target_id="btc",
                    display_pair="ETH/USDC",
                    execution_instrument="ETHUSDC",
                    market_data_instrument="ETHUSDC",
                ),
            ),
        )


def test_universe_settings_duplicate_display_pair_rejected() -> None:
    """Duplicate display_pair across targets is rejected."""
    with pytest.raises(ValueError, match="display_pair must be unique"):
        default_universe_settings(
            targets=(
                TargetConfig(
                    target_id="btc1",
                    display_pair="BTC/USDC",
                    execution_instrument="BTCUSDC",
                    market_data_instrument="BTCUSDC",
                ),
                TargetConfig(
                    target_id="btc2",
                    display_pair="BTC/USDC",
                    execution_instrument="BTCUSDT",
                    market_data_instrument="BTCUSDT",
                ),
            ),
        )
