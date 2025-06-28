"""Collecting what a run delivered, and summarizing it per UE group."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from slicesim.carrier import Carrier
from slicesim.channel.link import bits_per_prb
from slicesim.config import AdaptiveLink, Scenario

DELAY_PERCENTILES = (50.0, 95.0, 99.0)

IntArray = npt.NDArray[np.int64]


class MetricsCollector:
    """Per-UE counters and per-packet delays, filled in slot by slot."""

    def __init__(self, n_ues: int, n_slots: int) -> None:
        self.n_ues = n_ues
        self.offered_bytes = np.zeros(n_ues, dtype=np.int64)
        self.packets_arrived = np.zeros(n_ues, dtype=np.int64)
        self.delivered_bytes = np.zeros(n_ues, dtype=np.int64)
        """Bytes in transport blocks that decoded."""
        self.packets_delivered = np.zeros(n_ues, dtype=np.int64)
        self.packets_lost = np.zeros(n_ues, dtype=np.int64)
        self.blocks_sent = np.zeros(n_ues, dtype=np.int64)
        """First transmissions of transport blocks."""
        self.blocks_failed = np.zeros(n_ues, dtype=np.int64)
        """First transmissions that did not decode."""
        self.retransmissions = np.zeros(n_ues, dtype=np.int64)
        self.blocks_dropped = np.zeros(n_ues, dtype=np.int64)
        self.dropped_bytes = np.zeros(n_ues, dtype=np.int64)
        self.cqi_sum = np.zeros(n_ues, dtype=np.int64)
        """Sum of each UE's reported CQI over the slots so far."""
        self.backlog_sum = np.zeros(n_ues, dtype=np.int64)
        """Sum of each UE's queued bytes at the start of every slot so far."""
        self.prb_used = np.zeros(n_slots, dtype=np.int32)
        self._delay_ue: list[npt.NDArray[np.intp]] = []
        self._delay_s: list[npt.NDArray[np.float64]] = []

    def record_arrivals(self, ue: npt.NDArray[np.intp], size: IntArray) -> None:
        """Count packets arriving at the gNB, one entry per packet."""
        self.offered_bytes += np.bincount(ue, weights=size, minlength=self.n_ues).astype(np.int64)
        self.packets_arrived += np.bincount(ue, minlength=self.n_ues)

    def record_slot(self, slot: int, prbs_used: int, delivered: IntArray) -> None:
        """Record how many PRBs a slot used and the bytes it delivered to each UE."""
        self.prb_used[slot] = prbs_used
        self.delivered_bytes += delivered

    def record_delays(self, ue: npt.NDArray[np.intp], delay_s: npt.NDArray[np.float64]) -> None:
        """Record delivered packets, each with the delay from arrival to delivery."""
        if ue.size:
            self._delay_ue.append(ue)
            self._delay_s.append(delay_s)
            self.packets_delivered += np.bincount(ue, minlength=self.n_ues)

    def delays(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
        """Every delivered packet's UE and delay, in delivery order."""
        if not self._delay_ue:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float64)
        return np.concatenate(self._delay_ue), np.concatenate(self._delay_s)


def jain_index(values: npt.ArrayLike) -> float | None:
    """Jain's fairness index (sum x)^2 / (n sum x^2): 1 when all are equal, 1/n when
    one value takes everything. None when every value is zero."""
    x = np.asarray(values, dtype=np.float64)
    squares = float((x**2).sum())
    if x.size == 0 or squares == 0:
        return None
    return float(x.sum() ** 2 / (x.size * squares))


@dataclass(frozen=True)
class DelayStats:
    """Packet delay statistics in milliseconds."""

    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    @classmethod
    def of(cls, delay_s: npt.NDArray[np.float64]) -> DelayStats | None:
        """Statistics of a set of delays in seconds, or None if there are none."""
        if delay_s.size == 0:
            return None
        p50, p95, p99 = np.percentile(delay_s, DELAY_PERCENTILES) * 1e3
        return cls(
            mean_ms=float(delay_s.mean() * 1e3),
            p50_ms=float(p50),
            p95_ms=float(p95),
            p99_ms=float(p99),
        )


@dataclass(frozen=True)
class GroupSummary:
    """What one UE group was offered and received over a run."""

    name: str
    traffic: str
    ues: int
    offered_mbps: float | None
    """Offered load measured from arrivals; None for full-buffer traffic."""
    delivered_mbps: float
    per_ue_mbps: float
    jain: float | None
    """Jain's index over the group's per-UE throughputs."""
    mean_sinr_db: float | None
    """Mean over the group's UEs of their SINR before fading; None for the fixed link."""
    mean_cqi: float | None
    packets_arrived: int
    packets_delivered: int
    packets_lost: int
    delay: DelayStats | None
    backlog_bytes: int
    """Bytes still queued at the end of the run."""
    mean_buffer_bytes: float
    """Time-average queued bytes per UE, measured at the start of each slot."""


@dataclass(frozen=True)
class CellSummary:
    """Cell-wide totals for a run."""

    duration_s: float
    n_slots: int
    peak_mbps: float
    """What the cell delivers with every PRB in use at the highest rate."""
    delivered_mbps: float
    prb_utilization: float
    bler: float | None
    """Share of first transmissions that failed; None if nothing was sent."""
    residual_bler: float | None
    """Share of transport blocks dropped after every HARQ transmission failed."""
    retransmissions: int
    jain: float | None
    """Jain's index over every UE's throughput; meaningful when UEs have alike traffic."""
    edge_mbps: float
    """5th percentile of per-UE throughput."""
    median_ue_mbps: float
    delay: DelayStats | None
    """Delay over every delivered packet."""
    mean_buffer_bytes: float
    """Time-average queued bytes per UE."""


@dataclass(frozen=True)
class RunSummary:
    scenario: str
    seed: int
    carrier: str
    cell: CellSummary
    groups: list[GroupSummary]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class SimulationResult:
    """Everything a finished run recorded, per UE and per packet."""

    scenario: Scenario
    carrier: Carrier
    n_slots: int
    ue_group: npt.NDArray[np.intp]
    full_buffer: npt.NDArray[np.bool_]
    offered_bytes: IntArray
    delivered_bytes: IntArray
    packets_arrived: IntArray
    packets_delivered: IntArray
    packets_lost: IntArray
    backlog_bytes: IntArray
    in_flight_bytes: IntArray
    """Bytes in blocks waiting for a retransmission when the run ended."""
    dropped_bytes: IntArray
    blocks_sent: IntArray
    blocks_failed: IntArray
    retransmissions: IntArray
    blocks_dropped: IntArray
    delay_ue: npt.NDArray[np.intp]
    delay_s: npt.NDArray[np.float64]
    prb_used: npt.NDArray[np.int32]
    distance_m: npt.NDArray[np.float64] | None
    """Ground distance of each UE from the gNB; None for the fixed link."""
    los: npt.NDArray[np.bool_] | None
    mean_sinr_db: npt.NDArray[np.float64] | None
    """Each UE's SINR before fading."""
    mean_cqi: npt.NDArray[np.float64] | None
    mean_buffer_bytes: npt.NDArray[np.float64]
    """Each UE's time-average queued bytes."""

    @property
    def duration_s(self) -> float:
        return self.n_slots * self.carrier.slot_duration_s

    def throughput_bps(self) -> npt.NDArray[np.float64]:
        """Each UE's mean delivered throughput over the run."""
        return self.delivered_bytes * 8 / self.duration_s

    def peak_bps(self) -> float:
        """The cell's rate with every PRB carrying the most it can."""
        link = self.scenario.link
        per_prb = (
            int(bits_per_prb(15, link.overhead))
            if isinstance(link, AdaptiveLink)
            else link.bits_per_prb
        )
        return self.carrier.n_prb * per_prb / self.carrier.slot_duration_s

    def summary(self) -> RunSummary:
        duration = self.duration_s
        groups = []
        for g, group in enumerate(self.scenario.ues):
            members = self.ue_group == g
            delivered_bits = float(self.delivered_bytes[members].sum()) * 8
            full = bool(self.full_buffer[members].all())
            groups.append(
                GroupSummary(
                    name=group.name,
                    traffic=group.traffic.model,
                    ues=int(members.sum()),
                    offered_mbps=(
                        None
                        if full
                        else float(self.offered_bytes[members].sum()) * 8 / duration / 1e6
                    ),
                    delivered_mbps=delivered_bits / duration / 1e6,
                    per_ue_mbps=delivered_bits / duration / 1e6 / int(members.sum()),
                    jain=jain_index(self.delivered_bytes[members]),
                    mean_sinr_db=(
                        None
                        if self.mean_sinr_db is None
                        else float(self.mean_sinr_db[members].mean())
                    ),
                    mean_cqi=None
                    if self.mean_cqi is None
                    else float(self.mean_cqi[members].mean()),
                    packets_arrived=int(self.packets_arrived[members].sum()),
                    packets_delivered=int(self.packets_delivered[members].sum()),
                    packets_lost=int(self.packets_lost[members].sum()),
                    delay=DelayStats.of(self.delay_s[members[self.delay_ue]]),
                    backlog_bytes=int(self.backlog_bytes[members].sum()),
                    mean_buffer_bytes=float(self.mean_buffer_bytes[members].mean()),
                )
            )
        sent = int(self.blocks_sent.sum())
        throughput = self.throughput_bps()
        cell = CellSummary(
            duration_s=duration,
            n_slots=self.n_slots,
            peak_mbps=self.peak_bps() / 1e6,
            delivered_mbps=float(self.delivered_bytes.sum()) * 8 / duration / 1e6,
            prb_utilization=float(self.prb_used.sum()) / (self.carrier.n_prb * self.n_slots),
            bler=_ratio(int(self.blocks_failed.sum()), sent),
            residual_bler=_ratio(int(self.blocks_dropped.sum()), sent),
            retransmissions=int(self.retransmissions.sum()),
            jain=jain_index(self.delivered_bytes),
            edge_mbps=float(np.percentile(throughput, 5)) / 1e6,
            median_ue_mbps=float(np.median(throughput)) / 1e6,
            delay=DelayStats.of(self.delay_s),
            mean_buffer_bytes=float(self.mean_buffer_bytes.mean()),
        )
        return RunSummary(
            scenario=self.scenario.name,
            seed=self.scenario.seed,
            carrier=str(self.carrier),
            cell=cell,
            groups=groups,
        )
