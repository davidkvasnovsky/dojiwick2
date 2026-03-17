"""Adaptive policy value objects for Thompson-sampling arm selection."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, frozen=True, kw_only=True)
class AdaptiveArmKey:
    """Unique identifier for a regime-config arm in the adaptive policy."""

    regime_idx: int
    config_idx: int


@dataclass(slots=True, frozen=True, kw_only=True)
class AdaptivePosterior:
    """Beta-distribution posterior for a single arm."""

    arm: AdaptiveArmKey
    alpha: float = 1.0
    beta: float = 1.0
    n_updates: int = 0
    last_decay_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.n_updates < 0:
            raise ValueError("n_updates must be non-negative")
        if self.last_decay_at is not None and self.last_decay_at.tzinfo is None:
            raise ValueError("last_decay_at must be timezone-aware if set")


@dataclass(slots=True, frozen=True, kw_only=True)
class AdaptiveSelectionEvent:
    """Records which arm was selected for a given position leg."""

    position_leg_id: int
    arm: AdaptiveArmKey
    selected_at: datetime

    def __post_init__(self) -> None:
        if self.selected_at.tzinfo is None:
            raise ValueError("selected_at must be timezone-aware")


@dataclass(slots=True, frozen=True, kw_only=True)
class AdaptiveOutcomeEvent:
    """Records the observed reward after a position leg closes."""

    position_leg_id: int
    arm: AdaptiveArmKey
    reward: float
    observed_at: datetime

    def __post_init__(self) -> None:
        if not 0 <= self.reward <= 1:
            raise ValueError("reward must be in [0, 1]")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")


@dataclass(slots=True, frozen=True, kw_only=True)
class AdaptiveCalibrationMetrics:
    """Calibration diagnostics for a single arm."""

    arm: AdaptiveArmKey
    expected_reward: float
    selection_count: int
    empirical_reward: float
    posterior_variance: float
    calibration_gap: float

    def __post_init__(self) -> None:
        if self.selection_count < 0:
            raise ValueError("selection_count must be non-negative")
