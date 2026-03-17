"""Pre-filter to skip LLM veto for high-confidence regime signals."""

import numpy as np

from dojiwick.domain.enums import MarketState
from dojiwick.domain.models.value_objects.batch_models import BatchRegimeProfile, BatchTradeCandidate
from dojiwick.domain.type_aliases import BoolVector


def compute_llm_review_mask(
    candidates: BatchTradeCandidate,
    regimes: BatchRegimeProfile,
    confidence_threshold: float,
) -> BoolVector:
    """Return mask of pairs that need LLM veto review.

    Rules:
    - ``valid_mask=False`` on candidate -> skip (no candidate to review)
    - Regime confidence >= threshold AND trending (up/down) -> skip LLM
    - VOLATILE regime -> always review
    - RANGING with confidence < threshold -> always review
    """
    size = len(candidates.valid_mask)
    needs_review = np.zeros(size, dtype=np.bool_)

    for i in range(size):
        if not candidates.valid_mask[i]:
            continue

        state = int(regimes.coarse_state[i])
        conf = float(regimes.confidence[i])

        if state == MarketState.VOLATILE:
            needs_review[i] = True
        elif state in (MarketState.TRENDING_UP, MarketState.TRENDING_DOWN) and conf >= confidence_threshold:
            needs_review[i] = False
        else:
            needs_review[i] = True

    return needs_review
