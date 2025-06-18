"""The slot loop and its random number streams.

Each slot the scheduler sees the queues as they stand at the start of the
slot, the cell sends what it allocated, and the packets that arrive during
the slot join the queues, to be served from the next slot on. A packet's
delay runs from its arrival to the end of the slot that carries its last
byte.
"""

from __future__ import annotations

import zlib
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from slicesim.config import Scenario
from slicesim.metrics import MetricsCollector, SimulationResult
from slicesim.traffic import Arrivals, TrafficSource, build_source
from slicesim.ue import UEPopulation


class RngStreams:
    """Independent random generators derived from one scenario seed.

    A stream is named by what it drives (``"traffic"``, later ``"fading"`` and
    so on) plus optional keys, and is derived from the seed and that name
    alone, never from the order streams are created in. Changing one model's
    use of randomness, or adding a stream, leaves every other stream's
    sequence unchanged.
    """

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.seed = seed

    def generator(self, name: str, *keys: int | str) -> np.random.Generator:
        spawn_key = tuple(
            zlib.crc32(k.encode("utf-8")) if isinstance(k, str) else k for k in (name, *keys)
        )
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=spawn_key))


def share_prbs(need: npt.NDArray[np.int64], n_prb: int, rotation: int) -> npt.NDArray[np.int64]:
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


class Simulation:
    """A scenario being simulated, one slot at a time."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.carrier = scenario.carrier.build()
        self.n_slots = scenario.n_slots
        self.slot = 0
        streams = RngStreams(scenario.seed)
        self.sources: list[TrafficSource] = [
            build_source(group.traffic, group.count, streams.generator("traffic", group.name))
            for group in scenario.ues
        ]
        self.ues = UEPopulation.from_groups(
            [group.count for group in scenario.ues], [s.full_buffer for s in self.sources]
        )
        counts = np.array([group.count for group in scenario.ues])
        self._group_start = (np.cumsum(counts) - counts).tolist()
        self.bits_per_prb = np.full(self.ues.size, scenario.link.bits_per_prb, dtype=np.int64)
        self.metrics = MetricsCollector(self.ues.size, self.n_slots)

    @property
    def finished(self) -> bool:
        return self.slot >= self.n_slots

    @property
    def now(self) -> float:
        """Simulated time at the start of the next slot, in seconds."""
        return self.slot * self.carrier.slot_duration_s

    def step(self, slots: int = 1) -> None:
        """Advance up to ``slots`` slots, stopping at the end of the run."""
        if slots < 0:
            raise ValueError("slots must be non-negative")
        for _ in range(min(slots, self.n_slots - self.slot)):
            self._run_slot()

    def run(self) -> SimulationResult:
        """Simulate the remaining slots and return the result."""
        self.step(self.n_slots - self.slot)
        return self.result()

    def result(self) -> SimulationResult:
        """What the run has recorded so far."""
        delay_ue, delay_s = self.metrics.delays()
        return SimulationResult(
            scenario=self.scenario,
            carrier=self.carrier,
            n_slots=self.slot,
            ue_group=self.ues.group,
            full_buffer=self.ues.full_buffer,
            offered_bytes=self.metrics.offered_bytes.copy(),
            delivered_bytes=self.metrics.delivered_bytes.copy(),
            packets_arrived=self.metrics.packets_arrived.copy(),
            packets_delivered=self.metrics.packets_delivered.copy(),
            backlog_bytes=self.ues.backlog_bytes.copy(),
            delay_ue=delay_ue,
            delay_s=delay_s,
            prb_used=self.metrics.prb_used[: self.slot].copy(),
        )

    def _run_slot(self) -> None:
        slot_s = self.carrier.slot_duration_s
        t0 = self.slot * slot_s
        t1 = (self.slot + 1) * slot_s
        n_prb = self.carrier.n_prb

        # Schedule on the queues as they stand at the start of the slot.
        need = -(-self.ues.backlog_bytes * 8 // self.bits_per_prb)
        need[self.ues.full_buffer] = n_prb
        prbs = share_prbs(np.minimum(need, n_prb), n_prb, self.slot)
        capacity = prbs * self.bits_per_prb // 8

        delivered = np.zeros(self.ues.size, dtype=np.int64)
        delay_ue: list[int] = []
        delays: list[float] = []
        for i in np.flatnonzero(prbs).tolist():
            before = len(delays)
            delivered[i] = self.ues.serve(i, int(capacity[i]), t1, delays)
            delay_ue.extend([i] * (len(delays) - before))
        self.metrics.record_slot(self.slot, prbs, delivered)
        self.metrics.record_delays(
            np.asarray(delay_ue, dtype=np.intp), np.asarray(delays, dtype=np.float64)
        )

        # Packets arriving during the slot wait for a later one.
        for source, start in zip(self.sources, self._group_start, strict=True):
            arrivals = source.arrivals(t0, t1)
            if len(arrivals):
                self._enqueue(arrivals, start)
        self.slot += 1

    def _enqueue(self, arrivals: Arrivals, start: int) -> None:
        ue = arrivals.ue + start
        self.metrics.record_arrivals(ue, arrivals.size)
        firsts = np.flatnonzero(np.diff(ue, prepend=-1))
        bounds = [*firsts.tolist(), len(ue)]
        times = arrivals.time.tolist()
        sizes = arrivals.size.tolist()
        for a, b in pairwise(bounds):
            self.ues.enqueue(int(ue[a]), times[a:b], sizes[a:b])


def simulate(scenario: Scenario) -> SimulationResult:
    """Run a scenario from start to finish."""
    return Simulation(scenario).run()
