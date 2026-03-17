"""Application-layer adaptive policy orchestration."""

import logging
from dataclasses import dataclass

from dojiwick.domain.enums import AdaptiveMode
from dojiwick.domain.contracts.policies.adaptive_reward import AdaptiveRewardPolicyPort
from dojiwick.domain.contracts.policies.adaptive_selection import AdaptiveSelectionPolicyPort
from dojiwick.domain.models.value_objects.position_leg import PositionEventRecord, PositionLeg
from dojiwick.domain.models.value_objects.adaptive import (
    AdaptiveArmKey,
    AdaptiveOutcomeEvent,
    AdaptivePosterior,
    AdaptiveSelectionEvent,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class AdaptiveService:
    """Orchestrates adaptive arm selection and outcome recording.

    Routes to the underlying policy ports when adaptive mode is enabled;
    returns safe defaults (``"baseline"``) when disabled.
    """

    mode: AdaptiveMode
    selection_policy: AdaptiveSelectionPolicyPort | None = None
    reward_policy: AdaptiveRewardPolicyPort | None = None

    async def select_variant(
        self,
        regime_idx: int = 0,
        posteriors: tuple[AdaptivePosterior, ...] = (),
    ) -> str | AdaptiveArmKey:
        """Select an adaptive variant.

        Returns ``"baseline"`` when adaptive mode is disabled.
        When enabled, delegates to the selection policy and returns the
        chosen :class:`AdaptiveArmKey`.
        """
        if self.mode == AdaptiveMode.DISABLED:
            return "baseline"

        if self.selection_policy is None:
            return "baseline"

        try:
            return await self.selection_policy.select(regime_idx, posteriors)
        except Exception:
            if self.mode == AdaptiveMode.BUCKET_FALLBACK:
                log.warning("adaptive selection failed; falling back to baseline")
                return "baseline"
            raise

    async def record_outcome(
        self,
        leg: PositionLeg | None = None,
        close_event: PositionEventRecord | None = None,
        selection: AdaptiveSelectionEvent | None = None,
    ) -> AdaptiveOutcomeEvent | None:
        """Record an adaptive outcome.

        No-op when adaptive mode is disabled or when required arguments
        are not provided. When enabled and all arguments are supplied,
        delegates to :class:`AdaptiveRewardPolicyPort`.
        """
        if self.mode == AdaptiveMode.DISABLED:
            return None

        if self.reward_policy is None:
            return None

        if leg is None or close_event is None or selection is None:
            return None

        return await self.reward_policy.compute_reward(leg, close_event, selection)
