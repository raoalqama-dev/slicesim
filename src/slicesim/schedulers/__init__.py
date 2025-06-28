"""MAC schedulers: round robin, max C/I and proportional fair."""

from __future__ import annotations

from slicesim.config import (
    MaxCIScheduler,
    ProportionalFairScheduler,
    RoundRobinScheduler,
    SchedulerConfig,
)
from slicesim.schedulers.base import Scheduler, SlotDemand
from slicesim.schedulers.max_ci import MaxCI
from slicesim.schedulers.proportional_fair import ProportionalFair
from slicesim.schedulers.round_robin import RoundRobin

__all__ = [
    "MaxCI",
    "ProportionalFair",
    "RoundRobin",
    "Scheduler",
    "SlotDemand",
    "build_scheduler",
]


def build_scheduler(config: SchedulerConfig, n_ues: int, slot_duration_s: float) -> Scheduler:
    """Create the scheduler a scenario's ``scheduler`` section describes."""
    match config:
        case RoundRobinScheduler():
            return RoundRobin(n_ues)
        case MaxCIScheduler():
            return MaxCI(n_ues)
        case ProportionalFairScheduler():
            slots = max(1.0, config.time_constant_ms * 1e-3 / slot_duration_s)
            return ProportionalFair(n_ues, time_constant_slots=slots)
