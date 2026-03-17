"""Config explain CLI tests."""

from pathlib import Path
import json

import pytest

from dojiwick.interfaces.cli.config_explain import main


def test_config_explain_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.toml"
    config.write_text(Path("config.toml").read_text(encoding="utf-8"), encoding="utf-8")

    exit_code = main(["--config", str(config), "--pair", "BTC/USDC", "--regime", "trending_down", "--format", "json"])

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert isinstance(payload["config_hash"], str)
    assert payload["trace"]["pair"] == "BTC/USDC"
    assert payload["trace"]["regime"] == "trending_down"
