"""Slices at run time: which UEs each serves and the scheduler that shares its PRBs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from slicesim.config import MmtcSla, Scenario, SlaConfig, UrllcSla
from slicesim.schedulers import Scheduler, build_scheduler


@dataclass
class Slice:
    """One slice: its UEs, its SLA and its intra-slice scheduler."""

    index: int
    name: str
    sla: SlaConfig | None
    """None for the single implicit slice of an unsliced cell."""
    members: npt.NDArray[np.intp]
    """UE numbers in the slice, ascending."""
    scheduler: Scheduler
    quota: float | None
    cqi_backoff_db: float = 0.0
    """The slice's UEs are given the CQI for their SINR less this margin."""

    @property
    def delay_budget_s(self) -> float:
        """The slice's packet delay budget; infinite for eMBB and the implicit slice."""
        if isinstance(self.sla, UrllcSla | MmtcSla):
            return self.sla.delay_budget_ms * 1e-3
        return math.inf


def build_slices(
    scenario: Scenario, ue_group: npt.NDArray[np.intp], slot_duration_s: float
) -> list[Slice]:
    """The scenario's slices, or one implicit slice holding every UE."""
    if not scenario.slices:
        members = np.arange(ue_group.size, dtype=np.intp)
        return [
            Slice(
                index=0,
                name="cell",
                sla=None,
                members=members,
                scheduler=build_scheduler(scenario.scheduler, members.size, slot_duration_s),
                quota=None,
            )
        ]
    index_of = {config.name: i for i, config in enumerate(scenario.slices)}
    group_slice = [index_of[g.slice] for g in scenario.ues if g.slice is not None]
    ue_slice = np.asarray(group_slice, dtype=np.intp)[ue_group]
    slices = []
    for index, config in enumerate(scenario.slices):
        members = np.flatnonzero(ue_slice == index).astype(np.intp)
        slices.append(
            Slice(
                index=index,
                name=config.name,
                sla=config.sla,
                members=members,
                scheduler=build_scheduler(config.scheduler, members.size, slot_duration_s),
                quota=config.quota,
                cqi_backoff_db=config.cqi_backoff_db,
            )
        )
    return slices


def ue_slices(slices: list[Slice], n_ues: int) -> npt.NDArray[np.intp]:
    """The slice index of every UE."""
    owner = np.full(n_ues, -1, dtype=np.intp)
    for s in slices:
        owner[s.members] = s.index
    return owner
