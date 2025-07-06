"""The slot loop and its random number streams.

Each slot:

1. every UE's SINR is read from the channel, and the scheduler sizes its
   transport blocks from the CQI each UE reported ``cqi_delay_slots`` ago;
2. HARQ retransmissions that are due take their PRBs first;
3. the scheduler shares the remaining PRBs among the UEs with data waiting,
   as the queues stand at the start of the slot;
4. each block decodes or fails with the BLER at the SINR it actually meets;
5. packets of decoded blocks are released to the UE in order;
6. the packets that arrive during the slot join the queues, to be served
   from the next slot on.

A packet's delay runs from its arrival to the end of the slot in which it is
released. With the fixed link every block decodes and steps 1, 2 and 4 drop
out.
"""

from __future__ import annotations

import math
import zlib
from collections import deque
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from slicesim.channel.cell import GroupStreams, RadioChannel
from slicesim.channel.link import bits_per_prb, bler, cqi_from_sinr
from slicesim.config import AdaptiveLink, Scenario
from slicesim.harq import BlockState, DeliveryQueue, Harq, TransportBlock
from slicesim.metrics import MetricsCollector, SimulationResult
from slicesim.schedulers import SlotDemand
from slicesim.slicing import SliceDemand, SlicingPolicy, build_policy, build_slices, ue_slices
from slicesim.traffic import Arrivals, TrafficSource, build_source
from slicesim.ue import UEPopulation


class RngStreams:
    """Independent random generators derived from one scenario seed.

    A stream is named by what it drives (``"traffic"``, ``"fading"`` and so
    on) plus optional keys, and is derived from the seed and that name
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
        self.slices = build_slices(scenario, self.ues.group, self.carrier.slot_duration_s)
        self.ue_slice = ue_slices(self.slices, self.ues.size)
        self.policy: SlicingPolicy = build_policy(scenario, self.slices, self.carrier.n_prb)
        """The inter-slice policy; a controller may replace it between slots."""
        window_s = scenario.slicing_config.window_ms * 1e-3
        self.window_slots = max(1, round(window_s / self.carrier.slot_duration_s))
        self.metrics = MetricsCollector(
            self.ues.size, self.n_slots, n_slices=len(self.slices), window_slots=self.window_slots
        )
        self._cqi_backoff_db = np.zeros(self.ues.size)
        self.delay_budget_s = np.full(self.ues.size, math.inf)
        """Each UE's packet delay budget, from its slice; infinite for eMBB."""
        for s in self.slices:
            self._cqi_backoff_db[s.members] = s.cqi_backoff_db
            self.delay_budget_s[s.members] = s.delay_budget_s
        self.reported_cqi: npt.NDArray[np.intp] | None = None
        """The CQI the scheduler used in the last slot (before the first, the CQI of
        each UE's mean SINR); None on the fixed link."""
        self.delivery = [None if b is None else DeliveryQueue() for b in self.ues.buffers]

        link = scenario.link
        self.channel: RadioChannel | None = None
        self.harq: Harq | None = None
        self._fixed_bits = np.zeros(self.ues.size, dtype=np.int64)
        if isinstance(link, AdaptiveLink):
            self.channel = RadioChannel(
                self.carrier,
                scenario.cell,
                scenario.channel,
                [
                    (
                        group.count,
                        GroupStreams(
                            drop=streams.generator("ue_drop", group.name),
                            los=streams.generator("los", group.name),
                            shadowing=streams.generator("shadowing", group.name),
                            fading=streams.generator("fading", group.name),
                        ),
                    )
                    for group in scenario.ues
                ],
            )
            self.harq = Harq(link.harq_rtt_slots, link.max_transmissions)
            self.reported_cqi = cqi_from_sinr(self.channel.mean_sinr_db - self._cqi_backoff_db)
        else:
            self._fixed_bits[:] = link.bits_per_prb
        cqi_delay = link.cqi_delay_slots if isinstance(link, AdaptiveLink) else 0
        self._sinr_history: deque[npt.NDArray[np.float64]] = deque(maxlen=cqi_delay + 1)
        self._decoding = streams.generator("harq")

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

    def in_flight_bytes(self) -> npt.NDArray[np.int64]:
        """Bytes in blocks waiting for a retransmission, per UE."""
        waiting = np.zeros(self.ues.size, dtype=np.int64)
        if self.harq is not None:
            for block in self.harq.waiting():
                waiting[block.ue] += block.nbytes
        return waiting

    def overdue_packets(self) -> npt.NDArray[np.int64]:
        """Per UE, packets queued now that have waited longer than the slice's delay budget."""
        overdue = np.zeros(self.ues.size, dtype=np.int64)
        for s in self.slices:
            if math.isinf(s.delay_budget_s):
                continue
            cutoff = self.now - s.delay_budget_s
            for ue in s.members.tolist():
                buffer = self.ues.buffers[ue]
                if buffer is not None:
                    overdue[ue] = buffer.waiting_since(cutoff)
        return overdue

    def result(self) -> SimulationResult:
        """What the run has recorded so far."""
        m = self.metrics
        delay_ue, delay_s = m.delays()
        channel = self.channel
        return SimulationResult(
            scenario=self.scenario,
            carrier=self.carrier,
            n_slots=self.slot,
            ue_group=self.ues.group,
            full_buffer=self.ues.full_buffer,
            offered_bytes=m.offered_bytes.copy(),
            delivered_bytes=m.delivered_bytes.copy(),
            packets_arrived=m.packets_arrived.copy(),
            packets_delivered=m.packets_delivered.copy(),
            packets_lost=m.packets_lost.copy(),
            packets_late=m.packets_late.copy(),
            backlog_bytes=self.ues.backlog_bytes.copy(),
            in_flight_bytes=self.in_flight_bytes(),
            dropped_bytes=m.dropped_bytes.copy(),
            blocks_sent=m.blocks_sent.copy(),
            blocks_failed=m.blocks_failed.copy(),
            retransmissions=m.retransmissions.copy(),
            blocks_dropped=m.blocks_dropped.copy(),
            delay_ue=delay_ue,
            delay_s=delay_s,
            prb_used=m.prb_used[: self.slot].copy(),
            distance_m=None if channel is None else channel.distance_m,
            los=None if channel is None else channel.los.copy(),
            mean_sinr_db=None if channel is None else channel.mean_sinr_db.copy(),
            mean_cqi=None if channel is None else m.cqi_sum / max(self.slot, 1),
            mean_buffer_bytes=m.backlog_sum / max(self.slot, 1),
            ue_slice=self.ue_slice,
            slice_prbs=m.slice_prbs[: self.slot].copy(),
            window_slots=self.window_slots,
            window_delivered=m.window_delivered.copy(),
            window_offered=m.window_offered.copy(),
            window_backlog=m.window_backlog.copy(),
            overdue_packets=self.overdue_packets(),
        )

    def _link_state(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64] | None, npt.NDArray[np.intp] | None]:
        """Bits per PRB for each UE this slot, with the SINR and reported CQI behind them."""
        if self.channel is None:
            return self._fixed_bits, None, None
        link = self.scenario.link
        assert isinstance(link, AdaptiveLink)
        sinr = self.channel.sinr_db(self.slot)
        # The scheduler only knows the CQI the UE reported cqi_delay_slots ago, and
        # chooses it with the slice's back-off margin.
        self._sinr_history.append(sinr)
        cqi = cqi_from_sinr(self._sinr_history[0] - self._cqi_backoff_db)
        self.metrics.cqi_sum += cqi
        self.reported_cqi = cqi
        return bits_per_prb(cqi, link.overhead), sinr, cqi

    def _retransmit(
        self,
        harq: Harq,
        sinr: npt.NDArray[np.float64],
        delivered: npt.NDArray[np.int64],
        resolved: set[int],
        retx_prbs: npt.NDArray[np.int64],
    ) -> None:
        """Send the retransmissions due this slot, adding their PRBs to ``retx_prbs``
        under each UE's slice.

        They always fit: the blocks due now are exactly those that failed one
        round trip ago, and those were all sent within that slot's PRBs.
        """
        m = self.metrics
        sending = harq.due(self.slot)
        used = sum(block.prbs for block in sending)
        if used > self.carrier.n_prb:
            raise RuntimeError(f"retransmissions need {used} PRBs, more than the carrier has")
        for block in sending:
            retx_prbs[self.ue_slice[block.ue]] += block.prbs
        draws = self._decoding.random(len(sending)).tolist()
        for block, draw in zip(sending, draws, strict=True):
            harq.transmit(block, self.slot, float(sinr[block.ue]), draw)
            m.retransmissions[block.ue] += 1
            if block.state is BlockState.PENDING:
                continue
            if block.state is BlockState.DECODED:
                delivered[block.ue] += block.nbytes
            else:
                m.blocks_dropped[block.ue] += 1
                m.dropped_bytes[block.ue] += block.nbytes
            resolved.add(block.ue)

    def _allocate(
        self,
        free: int,
        retx_prbs: npt.NDArray[np.int64],
        bits: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """New-data PRBs per UE: the policy splits ``free`` between slices, then each
        slice's scheduler shares its part among its UEs."""
        # A UE out of coverage (CQI 0) can carry nothing.
        reachable = bits > 0
        need = np.zeros(self.ues.size, dtype=np.int64)
        need[reachable] = -(-self.ues.backlog_bytes[reachable] * 8 // bits[reachable])
        need[self.ues.full_buffer & reachable] = free
        need = np.minimum(need, free)
        slice_need = np.array(
            [min(int(need[s.members].sum()), free) for s in self.slices], dtype=np.int64
        )
        budgets = self.policy.split(
            SliceDemand(
                slot=self.slot,
                n_prb=self.carrier.n_prb,
                free=free,
                retx_prbs=retx_prbs,
                need=slice_need,
            )
        )
        if np.any(budgets < 0) or int(budgets.sum()) > free:
            raise RuntimeError(
                f"policy {self.policy.name} split {budgets.tolist()} of {free} free PRBs"
            )
        prbs = np.zeros(self.ues.size, dtype=np.int64)
        for s, budget in zip(self.slices, budgets.tolist(), strict=True):
            if budget > 0:
                members = s.members
                prbs[members] = s.scheduler.allocate(
                    SlotDemand(
                        slot=self.slot,
                        n_prb=budget,
                        need=np.minimum(need[members], budget),
                        bits_per_prb=bits[members],
                    )
                )
        return prbs

    def _run_slot(self) -> None:
        slot = self.slot
        slot_s = self.carrier.slot_duration_s
        t0 = slot * slot_s
        t1 = (slot + 1) * slot_s
        m = self.metrics
        bits, sinr, cqi = self._link_state()

        delivered = np.zeros(self.ues.size, dtype=np.int64)
        delay_ue: list[int] = []
        delays: list[float] = []
        resolved: set[int] = set()

        retx_prbs = np.zeros(len(self.slices), dtype=np.int64)
        if self.harq is not None and sinr is not None and len(self.harq):
            self._retransmit(self.harq, sinr, delivered, resolved, retx_prbs)
        free = self.carrier.n_prb - int(retx_prbs.sum())

        m.backlog_sum += self.ues.backlog_bytes
        if slot % self.window_slots == 0:
            m.start_window(slot, self.ues.backlog_bytes)
        prbs = self._allocate(free, retx_prbs, bits)
        scheduled = np.flatnonzero(prbs)
        if sinr is None or cqi is None:
            decoded = np.ones(scheduled.size, dtype=np.bool_)
        else:
            failure = bler(sinr[scheduled], cqi[scheduled])
            decoded = self._decoding.random(scheduled.size) >= failure
        m.blocks_sent[scheduled] += 1

        for i, ok in zip(scheduled.tolist(), decoded.tolist(), strict=True):
            capacity = int(prbs[i] * bits[i] // 8)
            buffer = self.ues.buffers[i]
            arrivals: list[float] = []
            first_seq = 0
            if buffer is None:
                sent = capacity
            else:
                first_seq = buffer.head_seq
                sent = self.ues.serve(i, capacity, arrivals)
            queue = self.delivery[i]
            if ok:
                delivered[i] += sent
                if queue is None:
                    continue
                if not len(queue):
                    before = len(delays)
                    m.packets_lost[i] += queue.deliver(first_seq, arrivals, t1, delays)
                    delay_ue.extend([i] * (len(delays) - before))
                    continue
            block = TransportBlock(
                ue=i,
                prbs=int(prbs[i]),
                cqi=0 if cqi is None else int(cqi[i]),
                nbytes=sent,
                first_seq=first_seq,
                arrivals=arrivals,
                partial_tail=buffer is not None and buffer.head_started,
                sinr_db=[] if sinr is None else [float(sinr[i])],
            )
            if ok:
                # Decoded, but an earlier block for this UE is still pending.
                block.state = BlockState.DECODED
            else:
                assert self.harq is not None
                m.blocks_failed[i] += 1
                self.harq.failed(block, slot)
                if block.state is BlockState.DROPPED:
                    m.blocks_dropped[i] += 1
                    m.dropped_bytes[i] += sent
                    resolved.add(i)
            if queue is not None:
                queue.add(block)

        for i in sorted(resolved):
            queue = self.delivery[i]
            if queue is not None:
                before = len(delays)
                m.packets_lost[i] += queue.release(t1, delays)
                delay_ue.extend([i] * (len(delays) - before))

        for s in self.slices:
            s.scheduler.delivered(delivered[s.members] * 8)
            m.slice_prbs[slot, s.index] = retx_prbs[s.index] + prbs[s.members].sum()
        m.record_slot(slot, int(retx_prbs.sum() + prbs.sum()), delivered)
        delay_ues = np.asarray(delay_ue, dtype=np.intp)
        delay_values = np.asarray(delays, dtype=np.float64)
        m.record_delays(delay_ues, delay_values)
        late = delay_ues[delay_values > self.delay_budget_s[delay_ues]]
        if late.size:
            m.packets_late += np.bincount(late, minlength=self.ues.size)

        # Packets arriving during the slot wait for a later one.
        for source, start in zip(self.sources, self._group_start, strict=True):
            incoming = source.arrivals(t0, t1)
            if len(incoming):
                self._enqueue(incoming, start)
        self.slot += 1

    def _enqueue(self, arrivals: Arrivals, start: int) -> None:
        ue = arrivals.ue + start
        self.metrics.record_arrivals(ue, arrivals.size, self.slot)
        firsts = np.flatnonzero(np.diff(ue, prepend=-1))
        bounds = [*firsts.tolist(), len(ue)]
        times = arrivals.time.tolist()
        sizes = arrivals.size.tolist()
        for a, b in pairwise(bounds):
            self.ues.enqueue(int(ue[a]), times[a:b], sizes[a:b])


def simulate(scenario: Scenario) -> SimulationResult:
    """Run a scenario from start to finish."""
    return Simulation(scenario).run()
