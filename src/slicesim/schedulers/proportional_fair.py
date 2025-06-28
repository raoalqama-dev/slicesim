"""Proportional fair: the best channel relative to what a UE has been getting."""

from __future__ import annotations

import numpy as np

from slicesim.schedulers.base import IntArray, Scheduler, SlotDemand, fill_in_order, rank


class ProportionalFair(Scheduler):
    """Serve UEs in order of instantaneous rate over average throughput.

    The average is an exponentially weighted moving average of the bits each
    UE received per slot, with a time constant of ``time_constant_slots``:

        R <- (1 - 1/T) R + (1/T) r_received

    A UE is served when its channel is good relative to its own average, so
    every UE gets its turn (fairness) and tends to get it at its fading peaks
    (multi-user diversity). A short time constant behaves like round robin in
    time, a very long one maximizes the sum of log throughputs.
    """

    name = "pf"

    def __init__(self, n_ues: int, time_constant_slots: float) -> None:
        super().__init__(n_ues)
        if time_constant_slots < 1:
            raise ValueError("time_constant_slots must be at least 1")
        self.time_constant_slots = time_constant_slots
        # One bit per slot: a UE that has received nothing yet ranks first.
        self.average_bits = np.ones(n_ues)

    def allocate(self, demand: SlotDemand) -> IntArray:
        metric = demand.bits_per_prb / self.average_bits
        metric[demand.need <= 0] = -1.0
        return fill_in_order(rank(metric, demand.slot), demand.need, demand.n_prb)

    def delivered(self, bits: IntArray) -> None:
        weight = 1.0 / self.time_constant_slots
        self.average_bits = (1 - weight) * self.average_bits + weight * bits
        # Keep the floor of one bit so the metric stays finite.
        np.maximum(self.average_bits, 1.0, out=self.average_bits)
