"""The MAC scheduler interface.

Every slot the engine tells the scheduler how many PRBs are left for new data
(after HARQ retransmissions), how many each UE could use, and how many bits a
PRB carries for each UE at its reported CQI. The scheduler returns how many
PRBs each UE gets. The channel is flat across the carrier, so which PRBs a UE
gets does not matter, only how many.

After the slot, the scheduler is told how many bits each UE received, which
throughput-tracking schedulers use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SlotDemand:
    """What the scheduler knows at the start of a slot."""

    slot: int
    n_prb: int
    """PRBs available for new data."""
    need: IntArray
    """PRBs each UE could fill; 0 for UEs with nothing to send or out of coverage."""
    bits_per_prb: IntArray
    """Bits one PRB carries for each UE at its reported CQI."""


class Scheduler(ABC):
    """Allocates PRBs among UEs, one slot at a time."""

    name: ClassVar[str]

    def __init__(self, n_ues: int) -> None:
        self.n_ues = n_ues

    @abstractmethod
    def allocate(self, demand: SlotDemand) -> IntArray:
        """PRBs for each UE: at most its need, and at most ``n_prb`` in total."""

    def delivered(self, bits: IntArray) -> None:  # noqa: B027
        """Bits each UE received in the slot just scheduled; ignored by default."""


def fill_in_order(order: npt.NDArray[np.intp], need: IntArray, n_prb: int) -> IntArray:
    """Give UEs PRBs in ``order``, each as many as it needs, until none are left."""
    alloc = np.zeros_like(need)
    granted = np.minimum(need[order], np.maximum(n_prb - (np.cumsum(need[order]) - need[order]), 0))
    alloc[order] = granted
    return alloc


def rank(metric: npt.NDArray[np.float64], slot: int) -> npt.NDArray[np.intp]:
    """UE indices from the highest ``metric`` down.

    Ties go to UEs in turn, starting at a position that moves with ``slot``,
    so equal UEs share the top spot over time instead of the lowest index
    always winning.
    """
    n = metric.size
    turn = (np.arange(n) - slot) % max(n, 1)
    order: npt.NDArray[np.intp] = np.lexsort((turn, -metric))
    return order
