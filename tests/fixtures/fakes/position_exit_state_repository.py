"""In-memory position exit-state repository for tests."""

from dataclasses import dataclass, field

from dojiwick.domain.models.entities.position_exit_state import PositionExitState


@dataclass(slots=True)
class FakePositionExitStateRepository:
    """Dict-backed exit-state store keyed by position leg id."""

    states: dict[int, PositionExitState] = field(default_factory=dict)
    open_leg_ids: set[int] = field(default_factory=set)

    async def upsert(self, state: PositionExitState) -> None:
        self.states[state.position_leg_id] = state
        self.open_leg_ids.add(state.position_leg_id)

    async def get(self, position_leg_id: int) -> PositionExitState | None:
        return self.states.get(position_leg_id)

    async def list_active(self, account: str) -> tuple[PositionExitState, ...]:
        return tuple(s for leg_id, s in sorted(self.states.items()) if leg_id in self.open_leg_ids)

    async def delete(self, position_leg_id: int) -> None:
        self.states.pop(position_leg_id, None)
        self.open_leg_ids.discard(position_leg_id)
