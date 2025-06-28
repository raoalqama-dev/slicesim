"""Running one scenario under several schedulers.

Each run keeps the scenario's seed, and every random stream is named rather
than drawn in order, so all schedulers see the same UE positions, channels
and packet arrivals: the differences between rows come from the schedulers.
"""

from __future__ import annotations

from collections.abc import Sequence

from slicesim.config import Scenario
from slicesim.engine import simulate
from slicesim.metrics import RunSummary

SCHEDULERS = ("rr", "maxci", "pf")


def with_scheduler(scenario: Scenario, name: str) -> Scenario:
    """The scenario under scheduler ``name``, keeping its settings if it already uses it."""
    if name not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {name!r}; choose from {', '.join(SCHEDULERS)}")
    if scenario.scheduler.model == name:
        return scenario
    return scenario.replace(scheduler={"model": name})


def compare_schedulers(
    scenario: Scenario, names: Sequence[str]
) -> list[tuple[Scenario, RunSummary]]:
    """Run the scenario once per scheduler, in the order given."""
    runs = []
    for name in names:
        variant = with_scheduler(scenario, name)
        runs.append((variant, simulate(variant).summary()))
    return runs
