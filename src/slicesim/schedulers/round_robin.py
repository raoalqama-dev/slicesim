"""Round robin: the PRBs shared equally among the UEs with data waiting."""

from __future__ import annotations

import numpy as np

from slicesim.schedulers.base import IntArray, Scheduler, SlotDemand


def share_prbs(need: IntArray, n_prb: int, rotation: int) -> IntArray:
    """Split ``n_prb`` resource blocks equally among the UEs that need any.

    No UE gets more than it needs: what a UE cannot use is shared again among
    the rest (water-filling). When fewer blocks remain than UEs want them,
    one block each goes to UEs in turn, starting at a point that moves with
    ``rotation``, so no UE is favoured over time.
    """
    alloc = np.zeros_like(need)
    remaining = n_prb
    wanting = np.flatnonzero(need > 0)
    while remaining > 0 and wanting.size:
        share = remaining // wanting.size
        if share == 0:
            turn = (rotation + np.arange(remaining)) % wanting.size
            alloc[wanting[turn]] += 1
            break
        give = np.minimum(share, need[wanting] - alloc[wanting])
        alloc[wanting] += give
        remaining -= int(give.sum())
        wanting = wanting[alloc[wanting] < need[wanting]]
    return alloc


class RoundRobin(Scheduler):
    """Equal resources for every UE with data, whatever its channel.

    Round robin in the frequency domain: each slot the PRBs are split equally
    among the UEs with data, rather than whole slots going to one UE in turn.
    Over time both give every busy UE the same share of resources; this way
    small packets wait less.
    """

    name = "rr"

    def allocate(self, demand: SlotDemand) -> IntArray:
        return share_prbs(demand.need, demand.n_prb, demand.slot)
