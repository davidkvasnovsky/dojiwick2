"""Tests for validate CLI argument parsing and mode dispatch."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from dojiwick.interfaces.cli.validate import _parse_args  # pyright: ignore[reportPrivateUsage]


def test_default_mode_is_full_gate() -> None:
    """Default mode is full-gate."""
    with patch("sys.argv", ["validate", "--config", "c.toml", "--start", "2025-01-01", "--end", "2025-06-01"]):
        args = _parse_args()
        assert args.mode == "full-gate"


def test_walk_forward_mode() -> None:
    with patch(
        "sys.argv",
        ["validate", "--config", "c.toml", "--start", "2025-01-01", "--end", "2025-06-01", "--mode", "walk-forward"],
    ):
        args = _parse_args()
        assert args.mode == "walk-forward"


def test_cross_validate_mode() -> None:
    with patch(
        "sys.argv",
        [
            "validate",
            "--config",
            "c.toml",
            "--start",
            "2025-01-01",
            "--end",
            "2025-06-01",
            "--mode",
            "cross-validate",
        ],
    ):
        args = _parse_args()
        assert args.mode == "cross-validate"


def test_validate_cli_no_methodology_args() -> None:
    """Methodology args (--folds, --purge-bars, --embargo-bars, --train-size, --test-size) are removed."""
    with patch(
        "sys.argv",
        ["validate", "--config", "c.toml", "--start", "2025-01-01", "--end", "2025-06-01"],
    ):
        args = _parse_args()
        assert not hasattr(args, "folds")
        assert not hasattr(args, "purge_bars")
        assert not hasattr(args, "embargo_bars")
        assert not hasattr(args, "train_size")
        assert not hasattr(args, "test_size")
        assert not hasattr(args, "expanding")


def test_no_cache_cli_removed() -> None:
    """--no-cache flag is removed from CLI — candle caching is now config-driven."""
    with patch(
        "sys.argv",
        ["validate", "--config", "c.toml", "--start", "2025-01-01", "--end", "2025-06-01"],
    ):
        args = _parse_args()
        assert not hasattr(args, "no_cache")


@pytest.mark.asyncio
async def test_mode_dispatch_walk_forward() -> None:
    """walk-forward mode calls _run_walk_forward."""
    with (
        patch(
            "sys.argv",
            ["v", "--config", "c.toml", "--start", "2025-01-01", "--end", "2025-06-01", "--mode", "walk-forward"],
        ),
        patch("dojiwick.interfaces.cli.validate._run_walk_forward", new_callable=AsyncMock) as mock_wf,
        patch("dojiwick.interfaces.cli.validate._run_cross_validate", new_callable=AsyncMock) as mock_cv,
        patch("dojiwick.interfaces.cli.validate._run_full_gate", new_callable=AsyncMock) as mock_fg,
    ):
        from dojiwick.interfaces.cli.validate import _run  # pyright: ignore[reportPrivateUsage]

        await _run()
        mock_wf.assert_awaited_once()
        mock_cv.assert_not_awaited()
        mock_fg.assert_not_awaited()


@pytest.mark.asyncio
async def test_mode_dispatch_full_gate() -> None:
    """full-gate mode calls _run_full_gate."""
    with (
        patch("sys.argv", ["v", "--config", "c.toml", "--start", "2025-01-01", "--end", "2025-06-01"]),
        patch("dojiwick.interfaces.cli.validate._run_walk_forward", new_callable=AsyncMock) as mock_wf,
        patch("dojiwick.interfaces.cli.validate._run_cross_validate", new_callable=AsyncMock) as mock_cv,
        patch("dojiwick.interfaces.cli.validate._run_full_gate", new_callable=AsyncMock) as mock_fg,
    ):
        from dojiwick.interfaces.cli.validate import _run  # pyright: ignore[reportPrivateUsage]

        await _run()
        mock_wf.assert_not_awaited()
        mock_cv.assert_not_awaited()
        mock_fg.assert_awaited_once()
