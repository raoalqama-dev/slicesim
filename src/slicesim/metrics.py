"""Collecting what a run delivered, and summarizing it per UE group."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from slicesim.carrier import Carrier
from slicesim.config import Scenario

DELAY_PERCENTILES = (50.0, 95.0, 99.0)


class MetricsCollector:
    """Per-UE counters and per-packet delays, filled in slot by slot."""

    def __init__(self, n_ues: int, n_slots: int) -> None:
        self.n_ues = n_ues
        self.offered_bytes = np.zeros(n_ues, dtype=np.int64)
        self.packets_arrived = np.zeros(n_ues, dtype=np.int64)
        self.delivered_bytes = np.zeros(n_ues, dtype=np.int64)
        self.packets_delivered = np.zeros(n_ues, dtype=np.int64)
        self.prb_used = np.zeros(n_slots, dtype=np.int32)
        self._delay_ue: list[npt.NDArray[np.intp]] = []
        self._delay_s: list[npt.NDArray[np.float64]] = []

    def record_arrivals(self, ue: npt.NDArray[np.intp], size: npt.NDArray[np.int64]) -> None:
        """Count packets arriving at the gNB, one entry per packet."""
        self.offered_bytes += np.bincount(ue, weights=size, minlength=self.n_ues).astype(np.int64)
        self.packets_arrived += np.bincount(ue, minlength=self.n_ues)

    def record_slot(
        self, slot: int, prbs: npt.NDArray[np.int64], delivered: npt.NDArray[np.int64]
    ) -> None:
        """Record one slot's PRB allocation and the bytes it delivered to each UE."""
        self.prb_used[slot] = int(prbs.sum())
        self.delivered_bytes += delivered

    def record_delays(self, ue: npt.NDArray[np.intp], delay_s: npt.NDArray[np.float64]) -> None:
        """Record completed packets, each with the delay from arrival to delivery."""
        if ue.size:
            self._delay_ue.append(ue)
            self._delay_s.append(delay_s)
            self.packets_delivered += np.bincount(ue, minlength=self.n_ues)

    def delays(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
        """Every delivered packet's UE and delay, in delivery order."""
        if not self._delay_ue:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float64)
        return np.concatenate(self._delay_ue), np.concatenate(self._delay_s)


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
    packets_arrived: int
    packets_delivered: int
    delay: DelayStats | None
    backlog_bytes: int
    """Bytes still queued at the end of the run."""


@dataclass(frozen=True)
class CellSummary:
    """Cell-wide totals for a run."""

    duration_s: float
    n_slots: int
    capacity_mbps: float
    """What the cell delivers with every PRB in use."""
    delivered_mbps: float
    prb_utilization: float


@dataclass(frozen=True)
class RunSummary:
    scenario: str
    seed: int
    carrier: str
    cell: CellSummary
    groups: list[GroupSummary]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulationResult:
    """Everything a finished run recorded, per UE and per packet."""

    scenario: Scenario
    carrier: Carrier
    n_slots: int
    ue_group: npt.NDArray[np.intp]
    full_buffer: npt.NDArray[np.bool_]
    offered_bytes: npt.NDArray[np.int64]
    delivered_bytes: npt.NDArray[np.int64]
    packets_arrived: npt.NDArray[np.int64]
    packets_delivered: npt.NDArray[np.int64]
    backlog_bytes: npt.NDArray[np.int64]
    delay_ue: npt.NDArray[np.intp]
    delay_s: npt.NDArray[np.float64]
    prb_used: npt.NDArray[np.int32]

    @property
    def duration_s(self) -> float:
        return self.n_slots * self.carrier.slot_duration_s

    def throughput_bps(self) -> npt.NDArray[np.float64]:
        """Each UE's mean delivered throughput over the run."""
        return self.delivered_bytes * 8 / self.duration_s

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
                    packets_arrived=int(self.packets_arrived[members].sum()),
                    packets_delivered=int(self.packets_delivered[members].sum()),
                    delay=DelayStats.of(self.delay_s[members[self.delay_ue]]),
                    backlog_bytes=int(self.backlog_bytes[members].sum()),
                )
            )
        slot_s = self.carrier.slot_duration_s
        capacity = self.carrier.n_prb * self.scenario.link.bits_per_prb / slot_s
        cell = CellSummary(
            duration_s=duration,
            n_slots=self.n_slots,
            capacity_mbps=capacity / 1e6,
            delivered_mbps=float(self.delivered_bytes.sum()) * 8 / duration / 1e6,
            prb_utilization=float(self.prb_used.sum()) / (self.carrier.n_prb * self.n_slots),
        )
        return RunSummary(
            scenario=self.scenario.name,
            seed=self.scenario.seed,
            carrier=str(self.carrier),
            cell=cell,
            groups=groups,
        )
