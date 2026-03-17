"""Stable reason codes used across risk, strategy, and pipeline outcomes."""

STRATEGY_HOLD = "strategy_hold"
STRATEGY_SIGNAL = "strategy_signal"
REGIME_INVALID = "regime_invalid"
AI_VETO = "ai_veto"
AI_VETO_ERROR = "ai_veto_error"

# AI veto — specific (persisted in DecisionOutcome.reason_code)
AI_VETO_APPROVED = "ai_veto_approved"
AI_VETO_CONFLICTING_REGIME = "ai_veto_conflicting_regime"
AI_VETO_EXTREME_VOLATILITY = "ai_veto_extreme_volatility"
AI_VETO_CORRELATION_RISK = "ai_veto_correlation_risk"
AI_VETO_STALE_SIGNAL = "ai_veto_stale_signal"

# AI veto — fallback (non-contributory for authority)
AI_VETO_CONFIDENCE_SKIP = "ai_veto_confidence_skip"
AI_VETO_BUDGET_EXCEEDED = "ai_veto_budget_exceeded"
AI_VETO_PARSE_ERROR = "ai_veto_parse_error"

AI_VETO_BATCH_TIMEOUT = "ai_veto_batch_timeout"

AI_VETO_NON_CONTRIBUTORY_CODES: frozenset[str] = frozenset(
    {
        AI_VETO_ERROR,
        AI_VETO_PARSE_ERROR,
        AI_VETO_BUDGET_EXCEEDED,
        AI_VETO_CONFIDENCE_SKIP,
        AI_VETO_BATCH_TIMEOUT,
        "no_candidate",
        "veto_not_enabled",
        "veto_approved",
    }
)
AI_REGIME_AGREE = "ai_regime_agree"
AI_REGIME_DISAGREE = "ai_regime_disagree"
AI_REGIME_ERROR = "ai_regime_error"
RISK_OK = "risk_ok"
RISK_DAILY_LOSS = "risk_daily_loss"
RISK_MAX_POSITIONS = "risk_max_positions"
RISK_MIN_RR = "risk_min_rr"
RISK_ZERO_STOP_DISTANCE = "risk_zero_stop_distance"
RISK_NO_CANDIDATE = "risk_no_candidate"
RISK_COOLDOWN = "risk_cooldown"
RISK_PAIR_BLOCKED = "risk_pair_blocked"
RISK_SECTOR_LIMIT = "risk_sector_limit"
EXECUTION_FILLED = "execution_filled"
EXECUTION_SKIPPED = "execution_skipped"
EXECUTION_REJECTED = "execution_rejected"
EXECUTION_ERROR = "execution_error"

# Position lifecycle — values match CloseReason enum members
CLOSE_STOP_LOSS = "stop_loss"
CLOSE_TAKE_PROFIT = "take_profit"
CLOSE_TRAILING_STOP = "trailing_stop"
CLOSE_EMERGENCY = "emergency"
CLOSE_MANUAL = "manual"
CLOSE_DOUBLE_FILL = "double_fill"

# Sizing / exchange
RISK_EXCHANGE_FILTER = "risk_exchange_filter"
RISK_OVERSIZED = "risk_oversized"
RISK_INVALID_SIZING = "risk_invalid_sizing"

# Data quality
DATA_STALE = "data_stale"
DATA_INCOMPLETE = "data_incomplete"
DATA_FALLBACK = "data_fallback"
