"""Network slicing: slices, inter-slice policies and SLA accounting."""

from __future__ import annotations

from slicesim.config import Scenario
from slicesim.slicing.policies import (
    DelayAware,
    Proportional,
    SliceDemand,
    SlicingPolicy,
    StaticQuotas,
    WeightedShares,
    WholeCell,
)
from slicesim.slicing.slices import Slice, build_slices, ue_slices

POLICIES = ("static", "delay_aware", "proportional")

__all__ = [
    "POLICIES",
    "DelayAware",
    "Proportional",
    "Slice",
    "SliceDemand",
    "SlicingPolicy",
    "StaticQuotas",
    "WeightedShares",
    "WholeCell",
    "build_policy",
    "build_slices",
    "ue_slices",
]


def build_policy(scenario: Scenario, slices: list[Slice], n_prb: int) -> SlicingPolicy:
    """The inter-slice policy a scenario asks for; WholeCell when it is not sliced."""
    if not scenario.slices:
        return WholeCell()
    policy = scenario.slicing_config.policy
    if policy == "static":
        return StaticQuotas([s.quota or 0.0 for s in slices], n_prb)
    if policy == "delay_aware":
        return DelayAware([s.delay_budget_s for s in slices])
    return Proportional()
