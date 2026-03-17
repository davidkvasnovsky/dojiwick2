"""Pair-aware regime hysteresis state machine."""

from dataclasses import dataclass, field

from dojiwick.domain.type_aliases import IntVector


@dataclass(slots=True)
class RegimeHysteresis:
    """Applies pair-local hysteresis to reduce regime flip noise."""

    stable_by_pair: dict[str, int] = field(default_factory=dict)
    pending_by_pair: dict[str, tuple[int, int]] = field(default_factory=dict)

    def apply(self, pairs: tuple[str, ...], raw_state: IntVector, bars: int) -> IntVector:
        """Return stable regime state per pair."""

        if bars < 1:
            raise ValueError(f"hysteresis_bars must be >= 1, got {bars}")

        stable = raw_state.copy()
        if bars == 1:
            for index, pair in enumerate(pairs):
                self.stable_by_pair[pair] = int(raw_state[index])
                self.pending_by_pair.pop(pair, None)
            self._cleanup_stale(pairs)
            return stable

        for index, pair in enumerate(pairs):
            incoming = int(raw_state[index])
            current = self.stable_by_pair.get(pair)

            if current is None:
                self.stable_by_pair[pair] = incoming
                self.pending_by_pair.pop(pair, None)
                stable[index] = incoming
                continue

            if incoming == current:
                self.pending_by_pair.pop(pair, None)
                stable[index] = current
                continue

            pending = self.pending_by_pair.get(pair)
            if pending is not None and pending[0] == incoming:
                count = pending[1] + 1
            else:
                count = 1
            self.pending_by_pair[pair] = (incoming, count)

            if count >= bars:
                self.stable_by_pair[pair] = incoming
                self.pending_by_pair.pop(pair, None)
                stable[index] = incoming
            else:
                stable[index] = current

        self._cleanup_stale(pairs)
        return stable

    def _cleanup_stale(self, pairs: tuple[str, ...]) -> None:
        """Remove state for pairs no longer in the active set."""
        active_set = set(pairs)
        for pair in list(self.stable_by_pair):
            if pair not in active_set:
                del self.stable_by_pair[pair]
                self.pending_by_pair.pop(pair, None)
