"""Tests for hardening gap remediation (Gaps 1-6)."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fixtures.factories.infrastructure import (
    default_settings,
    default_universe_settings,
)
from pydantic import ValidationError

from dojiwick.config.loader import load_settings
from dojiwick.config.schema import (
    TargetConfig,
)
from dojiwick.domain.errors import ConfigurationError

_BASE = Path("config.toml")


def _write_config(tmp_path: Path, *, content: str | None = None) -> Path:
    config = tmp_path / "config.toml"
    body = _BASE.read_text(encoding="utf-8") if content is None else content
    config.write_text(body, encoding="utf-8")
    return config


# --- Gap 1: Config SSOT ---


def test_settings_fails_on_missing_behavior_field(tmp_path: Path) -> None:
    """Missing behavior-bearing field (backtest.fee_bps) raises ConfigurationError."""
    body = _BASE.read_text(encoding="utf-8").replace("fee_bps = 4.0\n", "")
    config = _write_config(tmp_path, content=body)
    with pytest.raises(ConfigurationError, match=r"backtest.fee_bps"):
        load_settings(config)


def test_ai_settings_require_explicit_config(tmp_path: Path) -> None:
    """Missing ai.enabled raises ConfigurationError."""
    body = _BASE.read_text(encoding="utf-8").replace("enabled = true\nveto_enabled", "veto_enabled")
    config = _write_config(tmp_path, content=body)
    with pytest.raises(ConfigurationError, match=r"ai.enabled"):
        load_settings(config)


def test_infra_fields_accept_defaults(tmp_path: Path) -> None:
    """Missing system.log_level (infra-only) does not fail."""
    body = _BASE.read_text(encoding="utf-8").replace('log_level = "INFO"\n', "")
    config = _write_config(tmp_path, content=body)
    settings = load_settings(config)
    assert settings.system.log_level == "INFO"  # uses code default


def test_missing_required_section_fails(tmp_path: Path) -> None:
    """Missing [system] section raises ConfigurationError."""
    body = _BASE.read_text(encoding="utf-8")
    lines = body.splitlines(keepends=True)
    filtered: list[str] = []
    skip = False
    for line in lines:
        if line.strip() == "[system]":
            skip = True
            continue
        if skip and line.strip().startswith("["):
            skip = False
        if not skip:
            filtered.append(line)
    config = _write_config(tmp_path, content="".join(filtered))
    with pytest.raises(ConfigurationError, match=r"missing required config sections.*system"):
        load_settings(config)


# --- Gap 2: Instrument mapping ---


def test_target_requires_execution_instrument() -> None:
    """Empty execution_instrument raises ValidationError."""
    with pytest.raises(ValidationError, match="execution_instrument must be non-empty"):
        TargetConfig(
            target_id="btc", display_pair="BTC/USDC", execution_instrument="", market_data_instrument="BTCUSDC"
        )


def test_active_pairs_derived_from_targets(tmp_path: Path) -> None:
    """When targets are configured, active_pairs is derived from display_pairs."""
    config = _write_config(tmp_path)
    settings = load_settings(config)
    # config.toml has targets derived from [[universe.targets]]
    assert "BTC/USDT" in settings.trading.active_pairs
    assert len(settings.trading.active_pairs) >= 2
    assert settings.trading.primary_pair == "BTC/USDT"


# --- Gap 3: Candle cache scoped ---


def test_candle_cache_scoped_by_venue_product() -> None:
    """CachingCandleFetcher requires venue and product fields."""
    from datetime import UTC, datetime

    from fixtures.fakes.candle_repository import InMemoryCandleRepo
    from fixtures.fakes.clock import FixedClock

    from dojiwick.application.services.caching_candle_fetcher import CachingCandleFetcher

    class FakeFetcher:
        async def fetch_candles_range(self, symbol: str, interval: str, start: object, end: object) -> tuple[()]:
            return ()

    cacher = CachingCandleFetcher(
        candle_repo=InMemoryCandleRepo(),
        fetcher=FakeFetcher(),
        venue="binance",
        product="usd_c",
        clock=FixedClock(datetime(2026, 1, 1, tzinfo=UTC)),
    )
    assert cacher.venue == "binance"
    assert cacher.product == "usd_c"


# --- Gap 2b: resolve_symbols ---


def test_resolve_symbols_uses_targets() -> None:
    """When targets configured, resolve_symbols returns market_data_instrument."""
    from dojiwick.config.targets import resolve_symbols

    settings = default_settings(
        universe=default_universe_settings(
            targets=(
                TargetConfig(
                    target_id="btc_usdc",
                    display_pair="BTC/USDC",
                    execution_instrument="BTCUSDC",
                    market_data_instrument="BTCUSDT",
                ),
            ),
        ),
        trading=default_settings().trading.model_copy(update={"active_pairs": ("BTC/USDC",)}),
    )
    symbols = resolve_symbols(settings)
    assert symbols == ("BTCUSDT",)


# --- Gap 3b: Domain models carry target_id ---


def test_params_fail_without_defaults() -> None:
    """RegimeParams() with no args raises ValidationError — no code defaults allowed."""
    from pydantic import ValidationError as PydanticValidationError

    from dojiwick.domain.models.value_objects.params import RegimeParams, RiskParams, StrategyParams

    with pytest.raises(PydanticValidationError):
        RegimeParams()  # pyright: ignore[reportCallIssue]
    with pytest.raises(PydanticValidationError):
        StrategyParams()  # pyright: ignore[reportCallIssue]
    with pytest.raises(PydanticValidationError):
        RiskParams()  # pyright: ignore[reportCallIssue]


def test_outcome_carries_target_id() -> None:
    """OutcomeInputs with target_ids populates DecisionOutcome.target_id."""
    from fixtures.factories.domain import ContextBuilder

    from dojiwick.application.orchestration.outcome_assembler import OutcomeInputs, build_outcomes
    from dojiwick.domain.enums import DecisionAuthority, ExecutionStatus
    from dojiwick.domain.models.value_objects.batch_models import (
        BatchExecutionIntent,
        BatchRegimeProfile,
        BatchRiskAssessment,
        BatchTradeCandidate,
        BatchVetoDecision,
    )
    from dojiwick.domain.models.value_objects.outcome_models import ExecutionReceipt

    ctx = ContextBuilder(pairs=("BTC/USDC",)).build()
    n = 1
    regimes = BatchRegimeProfile(
        coarse_state=np.array([1], dtype=np.int64),
        confidence=np.array([0.6], dtype=np.float64),
        valid_mask=np.ones(n, dtype=np.bool_),
    )
    candidate = BatchTradeCandidate(
        action=np.array([0], dtype=np.int64),
        entry_price=np.zeros(n, dtype=np.float64),
        stop_price=np.zeros(n, dtype=np.float64),
        take_profit_price=np.zeros(n, dtype=np.float64),
        strategy_name=("none",),
        strategy_variant=("baseline",),
        valid_mask=np.zeros(n, dtype=np.bool_),
        reason_codes=("hold",),
    )
    veto = BatchVetoDecision(approved_mask=np.ones(n, dtype=np.bool_), reason_codes=("",))
    risk = BatchRiskAssessment(
        allowed_mask=np.ones(n, dtype=np.bool_), reason_codes=("",), risk_score=np.zeros(n, dtype=np.float64)
    )
    intent = BatchExecutionIntent(
        pairs=("BTC/USDC",),
        action=np.array([0], dtype=np.int64),
        entry_price=np.zeros(n),
        stop_price=np.zeros(n),
        take_profit_price=np.zeros(n),
        quantity=np.zeros(n),
        notional_usd=np.zeros(n),
        active_mask=np.zeros(n, dtype=np.bool_),
        strategy_name=("none",),
        strategy_variant=("baseline",),
    )
    receipts = (ExecutionReceipt(status=ExecutionStatus.SKIPPED, reason="test"),)

    inputs = OutcomeInputs(
        context=ctx,
        regime=regimes,
        candidate=candidate,
        veto=veto,
        risk=risk,
        intent=intent,
        receipts=receipts,
        authority=DecisionAuthority.DETERMINISTIC_ONLY,
        config_hash="testhash",
        target_ids=("btc_usdc",),
    )
    outcomes = build_outcomes(inputs)
    assert outcomes[0].target_id == "btc_usdc"


def test_backtest_run_record_has_provenance() -> None:
    """BacktestRunRecord requires target_ids, venue, product."""
    from datetime import UTC, datetime

    from dojiwick.domain.models.value_objects.backtest_run import BacktestRunRecord
    from dojiwick.domain.models.value_objects.outcome_models import BacktestSummary

    summary = BacktestSummary(
        trades=10,
        total_pnl_usd=100.0,
        win_rate=0.5,
        expectancy_usd=10.0,
        sharpe_like=1.0,
        max_drawdown_pct=5.0,
    )
    record = BacktestRunRecord(
        config_hash="abc",
        start_date=datetime(2025, 1, 1, tzinfo=UTC),
        end_date=datetime(2025, 6, 1, tzinfo=UTC),
        interval="1h",
        pairs=("BTC/USDC",),
        target_ids=("btc_usdc",),
        venue="binance",
        product="usd_c",
        summary=summary,
    )
    assert record.target_ids == ("btc_usdc",)
    assert record.venue == "binance"


# --- Gap 4: Postgres repos write provenance ---


def test_regime_repo_writes_target_id() -> None:
    """PgRegimeRepository INSERT SQL includes target_id, venue, product columns."""
    from dojiwick.infrastructure.postgres.repositories.regime import _INSERT_SQL  # pyright: ignore[reportPrivateUsage]

    assert "target_id" in _INSERT_SQL
    assert "venue" in _INSERT_SQL
    assert "product" in _INSERT_SQL


# --- Gap 4b: Binance adapter no suffix heuristic ---


async def test_account_state_resolves_assets_from_exchange_info() -> None:
    """Base/quote come from exchange info, not any symbol-suffix heuristic."""
    from dojiwick.infrastructure.exchange.binance.account_state import BinanceAccountStateProvider

    class StubClient:
        async def request(self, method: str, path: str, **kwargs: object) -> object:
            assert path == "/fapi/v1/exchangeInfo"
            # A symbol whose quote a USDT-suffix heuristic would get wrong
            return {"symbols": [{"symbol": "WEIRDPERP", "baseAsset": "WEIRD", "quoteAsset": "USDC"}]}

    provider = BinanceAccountStateProvider(client=StubClient())  # pyright: ignore[reportArgumentType]
    base, quote = await provider._resolve_assets("WEIRDPERP")  # pyright: ignore[reportPrivateUsage]
    assert (base, quote) == ("WEIRD", "USDC")


# --- Gap 5: CLI overrides removed ---


def test_validate_cli_no_methodology_args() -> None:
    """validate CLI no longer exposes --folds, --purge-bars, --embargo-bars."""
    from dojiwick.interfaces.cli.validate import _parse_args  # pyright: ignore[reportPrivateUsage]

    with patch(
        "sys.argv",
        ["validate", "--config", "c.toml", "--start", "2025-01-01", "--end", "2025-06-01"],
    ):
        args = _parse_args()
        assert not hasattr(args, "folds")
        assert not hasattr(args, "purge_bars")
        assert not hasattr(args, "embargo_bars")


def test_optimize_cli_no_behavior_args() -> None:
    """optimize CLI no longer exposes --trials, --objective-mode, --train-fraction."""
    from dojiwick.interfaces.cli.optimize import _parse_args  # pyright: ignore[reportPrivateUsage]

    with patch(
        "sys.argv",
        ["optimize", "--config", "c.toml", "--start", "2025-01-01", "--end", "2025-06-01"],
    ):
        args = _parse_args()
        assert not hasattr(args, "trials")
        assert not hasattr(args, "objective_mode")
        assert not hasattr(args, "train_fraction")


def test_no_cache_cli_removed() -> None:
    """--no-cache flag is removed from shared CLI args."""
    from dojiwick.interfaces.cli.validate import _parse_args  # pyright: ignore[reportPrivateUsage]

    with patch(
        "sys.argv",
        ["validate", "--config", "c.toml", "--start", "2025-01-01", "--end", "2025-06-01"],
    ):
        args = _parse_args()
        assert not hasattr(args, "no_cache")


# --- Gap 6: Simulated execution ---


def test_resolve_targets_uses_instrument_map() -> None:
    """When instrument_map is provided, resolve_targets uses mapped InstrumentId."""
    from dojiwick.application.orchestration.target_resolver import resolve_targets
    from dojiwick.domain.enums import TradeAction
    from dojiwick.domain.models.value_objects.batch_models import BatchExecutionIntent
    from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
    from dojiwick.domain.type_aliases import ProductCode, VenueCode

    venue = VenueCode("binance")
    product = ProductCode("usd_m")

    instrument_map = {
        "BTC/USDT": InstrumentId(
            venue=venue,
            product=product,
            symbol="BTCUSDC",
            base_asset="BTC",
            quote_asset="USDC",
            settle_asset="USDC",
        ),
    }
    intents = BatchExecutionIntent(
        pairs=("BTC/USDT",),
        action=np.array([TradeAction.BUY.value], dtype=np.int64),
        entry_price=np.array([50000.0]),
        stop_price=np.array([49000.0]),
        take_profit_price=np.array([52000.0]),
        quantity=np.array([0.1]),
        notional_usd=np.array([5000.0]),
        active_mask=np.array([True], dtype=np.bool_),
        strategy_name=("trend_follow",),
        strategy_variant=("baseline",),
    )

    resolved = resolve_targets(
        intents,
        account="default",
        venue=venue,
        product=product,
        quote_asset="USDT",
        instrument_map=instrument_map,
    )

    assert len(resolved.targets) == 1
    # InstrumentId comes from the map, not pair_to_instrument_id
    assert resolved.targets[0].instrument_id.symbol == "BTCUSDC"
    assert resolved.targets[0].instrument_id.quote_asset == "USDC"


def test_resolve_targets_missing_pair_in_map_raises() -> None:
    """When instrument_map is provided but pair is missing, raises ValueError."""
    from dojiwick.application.orchestration.target_resolver import resolve_targets
    from dojiwick.domain.enums import TradeAction
    from dojiwick.domain.models.value_objects.batch_models import BatchExecutionIntent
    from dojiwick.domain.models.value_objects.exchange_types import InstrumentId
    from dojiwick.domain.type_aliases import ProductCode, VenueCode

    venue = VenueCode("binance")
    product = ProductCode("usd_m")
    instrument_map: dict[str, InstrumentId] = {}  # empty map
    intents = BatchExecutionIntent(
        pairs=("BTC/USDT",),
        action=np.array([TradeAction.BUY.value], dtype=np.int64),
        entry_price=np.array([50000.0]),
        stop_price=np.array([49000.0]),
        take_profit_price=np.array([52000.0]),
        quantity=np.array([0.1]),
        notional_usd=np.array([5000.0]),
        active_mask=np.array([True], dtype=np.bool_),
        strategy_name=("trend_follow",),
        strategy_variant=("baseline",),
    )

    with pytest.raises(ValueError, match="not found in instrument_map"):
        resolve_targets(
            intents,
            account="default",
            venue=venue,
            product=product,
            quote_asset="USDT",
            instrument_map=instrument_map,
        )


# --- Phase 2: Account-state unknown symbol ---


async def test_account_state_unknown_symbol_raises() -> None:
    """_resolve_assets raises ExchangeError for symbols not in exchange info."""
    from dojiwick.domain.errors import ExchangeError
    from dojiwick.infrastructure.exchange.binance.account_state import BinanceAccountStateProvider

    class StubClient:
        async def request(self, method: str, path: str, **kwargs: object) -> object:
            return {"symbols": []}

    provider = BinanceAccountStateProvider(client=StubClient())  # pyright: ignore[reportArgumentType]
    with pytest.raises(ExchangeError, match="not found in exchange info"):
        await provider._resolve_assets("NOPE")  # pyright: ignore[reportPrivateUsage]
