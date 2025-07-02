"""Measuring each slice against its service level agreement.

URLLC and mMTC SLAs are about packets: a packet violates the SLA if it is
delivered after the delay budget, lost to HARQ, or still queued past the
budget when the run ends. The violation ratio is violations over packets
whose fate is known, and the SLA is met when it is at most 1 - reliability.

eMBB SLAs are about throughput, checked per UE over windows of ``window_ms``.
A UE cannot be short of a rate it did not ask for, so in each window it is
owed the minimum rate or what it offered (its queue at the start plus what
arrived), whichever is less; a full-buffer UE is always owed the minimum. The
violation ratio is the share of UE-windows that fell short, and the
satisfaction is the mean of delivered over owed, capped at 1. The SLA is met
when at most 1 - availability of UE-windows fell short. Windows the run ended
part way through are left out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from slicesim.config import EmbbSla, MmtcSla, SlaConfig, UrllcSla
from slicesim.metrics import DelayStats, SimulationResult


@dataclass(frozen=True)
class SliceSummary:
    """How one slice did against its SLA."""

    name: str
    sla: str
    """The SLA type: embb, urllc or mmtc."""
    target: str
    """The SLA in words."""
    ues: int
    prb_share: float
    """Mean share of the carrier's PRBs the slice used, retransmissions included."""
    delivered_mbps: float
    packets_delivered: int
    packets_lost: int
    delay: DelayStats | None
    violation_ratio: float | None
    """None when nothing could be measured, such as no packets at all."""
    met: bool | None
    satisfaction: float | None
    """eMBB: mean throughput satisfaction. URLLC and mMTC: the share of packets on time."""
    served_density_per_km2: float | None
    """mMTC only: devices meeting the reliability target, per km^2 of cell area."""


def describe(sla: SlaConfig, window_ms: float) -> str:
    """The SLA in words, as the report prints it."""
    if isinstance(sla, EmbbSla):
        return (
            f"{sla.min_throughput_mbps:g} Mbit/s per UE in {sla.availability:.4g} of "
            f"{window_ms:g} ms windows"
        )
    return f"{sla.reliability:.4g} of packets within {sla.delay_budget_ms:g} ms"


def packet_violations(
    delay_s: npt.NDArray[np.float64], lost: int, overdue: int, budget_s: float
) -> tuple[int, int]:
    """Violations and packets whose fate is known, for a packet-delay SLA."""
    late = int(np.count_nonzero(delay_s > budget_s))
    return late + lost + overdue, int(delay_s.size) + lost + overdue


def embb_windows(
    delivered: npt.NDArray[np.int64],
    offered: npt.NDArray[np.int64],
    backlog: npt.NDArray[np.int64],
    full_buffer: npt.NDArray[np.bool_],
    min_bits: float,
) -> tuple[float | None, float | None]:
    """Violation ratio and satisfaction over UE-windows, arrays shaped (windows, UEs)."""
    owed = np.minimum((offered + backlog) * 8.0, min_bits)
    owed[:, full_buffer] = min_bits
    counted = owed > 0
    if not counted.any():
        return None, None
    got = delivered * 8.0
    short = (got < owed - 1e-9) & counted
    ratio = np.where(counted, np.minimum(got / np.where(counted, owed, 1.0), 1.0), 0.0)
    return float(short.sum() / counted.sum()), float(ratio.sum() / counted.sum())


def summarize_slices(result: SimulationResult) -> list[SliceSummary]:
    """One summary per configured slice; empty for an unsliced cell."""
    scenario = result.scenario
    if not scenario.slices:
        return []
    window_ms = scenario.slicing_config.window_ms
    window_slots = result.window_slots
    complete = result.n_slots // window_slots
    window_s = window_slots * result.carrier.slot_duration_s
    cell = scenario.cell
    area_km2 = math.pi * (cell.radius_m**2 - cell.min_distance_m**2) / 1e6
    duration = result.duration_s
    summaries = []
    for index, config in enumerate(scenario.slices):
        members = result.ue_slice == index
        packet_mask = members[result.delay_ue]
        delays = result.delay_s[packet_mask]
        lost = int(result.packets_lost[members].sum())
        violation: float | None = None
        satisfaction: float | None = None
        density: float | None = None
        sla = config.sla
        if isinstance(sla, UrllcSla | MmtcSla):
            budget = sla.delay_budget_ms * 1e-3
            bad, known = packet_violations(
                delays, lost, int(result.overdue_packets[members].sum()), budget
            )
            if known:
                violation = bad / known
                satisfaction = 1 - violation
            if isinstance(sla, MmtcSla) and area_km2 > 0:
                served = 0
                for ue in np.flatnonzero(members).tolist():
                    ue_bad, ue_known = packet_violations(
                        result.delay_s[result.delay_ue == ue],
                        int(result.packets_lost[ue]),
                        int(result.overdue_packets[ue]),
                        budget,
                    )
                    served += ue_known == 0 or 1 - ue_bad / ue_known >= sla.reliability
                density = served / area_km2
            met = None if violation is None else violation <= 1 - sla.reliability + 1e-12
        else:
            violation, satisfaction = embb_windows(
                result.window_delivered[:complete, members],
                result.window_offered[:complete, members],
                result.window_backlog[:complete, members],
                result.full_buffer[members],
                sla.min_throughput_mbps * 1e6 * window_s,
            )
            met = None if violation is None else violation <= 1 - sla.availability + 1e-12
        summaries.append(
            SliceSummary(
                name=config.name,
                sla=sla.type,
                target=describe(sla, window_ms),
                ues=int(members.sum()),
                prb_share=float(result.slice_prbs[:, index].sum())
                / (result.carrier.n_prb * max(result.n_slots, 1)),
                delivered_mbps=float(result.delivered_bytes[members].sum()) * 8 / duration / 1e6,
                packets_delivered=int(delays.size),
                packets_lost=lost,
                delay=DelayStats.of(delays),
                violation_ratio=violation,
                met=met,
                satisfaction=satisfaction,
                served_density_per_km2=density,
            )
        )
    return summaries
