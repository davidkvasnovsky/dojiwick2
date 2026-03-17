"""Decision trace value object for per-step audit trail."""

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True, kw_only=True)
class DecisionTrace:
    """Immutable record of a single pipeline step for audit."""

    tick_id: str
    step_name: str
    step_seq: int
    artifacts: dict[str, object] = field(default_factory=dict)
    step_hash: str = ""
    duration_us: int | None = None
