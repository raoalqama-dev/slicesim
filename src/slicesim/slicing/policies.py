"""Inter-slice policies: how many PRBs each slice gets for new data each slot.

HARQ retransmissions are served before any policy runs, each charged to the
slice of its UE; the policy splits the PRBs that are left. Within a slice,
the slice's own scheduler then shares its PRBs among its UEs. The
reinforcement-learning controller replaces exactly this step.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SliceDemand:
    """What a policy knows at the start of a slot, one entry per slice."""

    slot: int
    n_prb: int
    """PRBs on the carrier."""
    free: int
    """PRBs left for new data after retransmissions."""
    retx_prbs: IntArray
    """PRBs each slice's retransmissions took this slot."""
    need: IntArray
    """PRBs each slice's UEs could fill with new data."""


class SlicingPolicy(ABC):
    """Splits the PRBs left after retransmissions between slices."""

    name: ClassVar[str]

    @abstractmethod
    def split(self, demand: SliceDemand) -> IntArray:
        """PRBs for each slice's new data; they add up to at most ``demand.free``."""


def largest_remainder(shares: npt.NDArray[np.float64], total: int) -> IntArray:
    """Integer parts of ``shares`` (which add up to at most ``total``), rounding so
    the parts add up to ``floor(sum(shares))`` and the largest fractions round up."""
    floors = np.floor(shares).astype(np.int64)
    left = min(total, math.floor(float(shares.sum()) + 1e-9)) - int(floors.sum())
    if left > 0:
        order = np.argsort(-(shares - floors), kind="stable")
        floors[order[:left]] += 1
    return floors


class WholeCell(SlicingPolicy):
    """An unsliced cell: its one implicit slice gets everything."""

    name = "whole_cell"

    def split(self, demand: SliceDemand) -> IntArray:
        return np.array([demand.free], dtype=np.int64)


class StaticQuotas(SlicingPolicy):
    """A hard partition: each slice owns a fixed share of the carrier.

    Slice s owns round(quota_s * N_PRB) PRBs, its retransmissions included.
    What a slice does not use stays idle rather than going to another slice:
    that isolation is what static slicing buys, and its cost.
    """

    name = "static"

    def __init__(self, quotas: list[float], n_prb: int) -> None:
        if any(q < 0 for q in quotas) or sum(quotas) > 1 + 1e-9:
            raise ValueError("quotas must be non-negative and add up to at most 1")
        self.owned = largest_remainder(np.asarray(quotas) * n_prb, n_prb)

    def split(self, demand: SliceDemand) -> IntArray:
        budget: IntArray = np.maximum(self.owned - demand.retx_prbs, 0)
        if budget.sum() > demand.free:
            raise RuntimeError("static quotas exceed the PRBs left after retransmissions")
        return budget


class DelayAware(SlicingPolicy):
    """Serve slices in order of their delay budget, tightest first, each up to its need.

    URLLC (a few ms) goes before mMTC (around a second), which goes before eMBB
    (no delay budget). Work-conserving: whatever one slice cannot use goes to
    the next.
    """

    name = "delay_aware"

    def __init__(self, delay_budgets_s: list[float]) -> None:
        self.order = np.argsort(np.asarray(delay_budgets_s), kind="stable")

    def split(self, demand: SliceDemand) -> IntArray:
        budget = np.zeros_like(demand.need)
        left = demand.free
        for s in self.order:
            give = min(int(demand.need[s]), left)
            budget[s] = give
            left -= give
        return budget


class Proportional(SlicingPolicy):
    """Split the PRBs in proportion to each slice's demand.

    When the slices together need less than is free, each gets its need;
    otherwise slice s gets free * need_s / sum(need), rounded by largest
    remainder. Work-conserving and blind to delay: a burst of eMBB data takes
    a burst of PRBs.
    """

    name = "proportional"

    def split(self, demand: SliceDemand) -> IntArray:
        need = demand.need
        total = int(need.sum())
        if total <= demand.free:
            return need.copy()
        return largest_remainder(demand.free * need / total, demand.free)
