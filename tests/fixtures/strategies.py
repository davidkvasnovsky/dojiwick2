"""Hypothesis composite strategies for generating valid domain objects."""

# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportUnknownParameterType=false, reportUnknownArgumentType=false

from datetime import UTC, datetime

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from dojiwick.domain.indicator_schema import INDICATOR_COUNT
from dojiwick.domain.models.value_objects.batch_models import (
    BatchDecisionContext,
    BatchMarketSnapshot,
    BatchPortfolioSnapshot,
)

VALID_PAIRS = ("BTC/USDC", "ETH/USDC", "SOL/USDC", "DOGE/USDC")


@st.composite
def st_batch_decision_context(
    draw: st.DrawFn,
    min_size: int = 1,
    max_size: int = 4,
) -> BatchDecisionContext:
    """Generate a valid BatchDecisionContext with random pairs, prices, indicators, and portfolio."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    pairs = tuple(draw(st.sampled_from(VALID_PAIRS)) for _ in range(size))

    prices = draw(
        arrays(
            dtype=np.float64,
            shape=size,
            elements=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        )
    )
    indicators = draw(
        arrays(
            dtype=np.float64,
            shape=(size, INDICATOR_COUNT),
            elements=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        )
    )
    equity = draw(
        arrays(
            dtype=np.float64,
            shape=size,
            elements=st.floats(min_value=100.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        )
    )
    day_start_equity = draw(
        arrays(
            dtype=np.float64,
            shape=size,
            elements=st.floats(min_value=100.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        )
    )
    unrealized = draw(
        arrays(
            dtype=np.float64,
            shape=size,
            elements=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
        )
    )

    return BatchDecisionContext(
        market=BatchMarketSnapshot(
            pairs=pairs,
            observed_at=datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC),
            price=prices,
            indicators=indicators,
        ),
        portfolio=BatchPortfolioSnapshot(
            equity_usd=equity,
            day_start_equity_usd=day_start_equity,
            open_positions_total=np.zeros(size, dtype=np.int64),
            has_open_position=np.zeros(size, dtype=np.bool_),
            unrealized_pnl_usd=unrealized,
        ),
    )
