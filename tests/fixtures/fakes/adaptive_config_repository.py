"""Fake adaptive config repository for tests."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class FakeAdaptiveConfigRepository:
    """In-memory adaptive config repository for test assertions."""

    configs: dict[int, dict[str, object]] = field(default_factory=dict)

    async def get_config(self, config_idx: int) -> dict[str, object] | None:
        return self.configs.get(config_idx)

    async def get_all_configs(self) -> tuple[tuple[int, dict[str, object]], ...]:
        return tuple((idx, cfg) for idx, cfg in sorted(self.configs.items()))
