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
    """The scenario under scheduler ``name``, keeping its settings if it already uses it.

    In a sliced cell every slice gets the scheduler.
    """
    if name not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {name!r}; choose from {', '.join(SCHEDULERS)}")
    if scenario.slices:
        if all(s.scheduler.model == name for s in scenario.slices):
            return scenario
        slices = [
            {
                **s.model_dump(exclude_unset=True),
                "scheduler": (
                    s.scheduler.model_dump() if s.scheduler.model == name else {"model": name}
                ),
            }
            for s in scenario.slices
        ]
        return scenario.replace(slices=slices)
    if scenario.scheduler.model == name:
        return scenario
    return scenario.replace(scheduler={"model": name})


def with_policy(scenario: Scenario, policy: str) -> Scenario:
    """The scenario under another inter-slice policy, its other slicing settings kept."""
    slicing = {**scenario.slicing_config.model_dump(), "policy": policy}
    return scenario.replace(slicing=slicing)


def compare_schedulers(
    scenario: Scenario, names: Sequence[str]
) -> list[tuple[Scenario, RunSummary]]:
    """Run the scenario once per scheduler, in the order given."""
    runs = []
    for name in names:
        variant = with_scheduler(scenario, name)
        runs.append((variant, simulate(variant).summary()))
    return runs
