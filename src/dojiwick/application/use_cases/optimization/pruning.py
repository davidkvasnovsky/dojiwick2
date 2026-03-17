"""Pruning types for optimization trial control flow."""

from typing import Protocol


class PrunedError(Exception):
    """Raised during optimization when an in-progress trial is pruned.

    This is a control-flow exception caught by the optimizer to mark the
    trial as pruned — not a failure.
    """


class PruningCallback(Protocol):
    """Callback for trial pruning during bar collection."""

    def report(self, value: float, step: int) -> None:
        """Report an intermediate objective value."""
        ...

    def should_prune(self) -> bool:
        """Return True if the trial should be pruned."""
        ...
