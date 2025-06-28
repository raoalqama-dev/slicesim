"""Maximum carrier-to-interference: the best channel first."""

from __future__ import annotations

from slicesim.schedulers.base import IntArray, Scheduler, SlotDemand, fill_in_order, rank


class MaxCI(Scheduler):
    """Serve the UEs with the highest rate per PRB first, each as much as it needs.

    This maximizes the bits sent per slot, and starves UEs whose channel is
    always worse than someone else's.
    """

    name = "maxci"

    def allocate(self, demand: SlotDemand) -> IntArray:
        busy = demand.need > 0
        metric = demand.bits_per_prb.astype(float)
        metric[~busy] = -1.0
        return fill_in_order(rank(metric, demand.slot), demand.need, demand.n_prb)
