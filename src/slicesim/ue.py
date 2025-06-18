"""UE state: who each UE is and the downlink packets waiting for it at the gNB."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class PacketBuffer:
    """First-in, first-out queue of the packets waiting for one UE.

    Packets are served in arrival order and may be split across slots: a
    packet counts as delivered, and its delay is recorded, in the slot that
    carries its last byte.
    """

    __slots__ = ("_arrivals", "_head_sent", "_sizes", "backlog_bytes")

    def __init__(self) -> None:
        self._arrivals: deque[float] = deque()
        self._sizes: deque[int] = deque()
        self._head_sent = 0
        self.backlog_bytes = 0
        """Bytes still to send, the partly sent head packet included."""

    def __len__(self) -> int:
        return len(self._sizes)

    def push(self, arrival_times: Iterable[float], sizes: Iterable[int]) -> None:
        """Queue packets, which must not arrive earlier than those already queued."""
        for arrival, size in zip(arrival_times, sizes, strict=True):
            if size <= 0:
                raise ValueError(f"packet size must be positive, got {size}")
            if self._arrivals and arrival < self._arrivals[-1]:
                raise ValueError("packets must be pushed in arrival order")
            self._arrivals.append(arrival)
            self._sizes.append(size)
            self.backlog_bytes += size

    def serve(self, nbytes: int, now: float, delays: list[float]) -> int:
        """Send up to ``nbytes`` from the head of the queue, delivering at ``now``.

        Appends the delay of every packet completed to ``delays`` and returns
        the bytes actually sent, which is less than ``nbytes`` only when the
        queue runs dry.
        """
        sent = 0
        while nbytes > 0 and self._sizes:
            left = self._sizes[0] - self._head_sent
            if nbytes < left:
                self._head_sent += nbytes
                sent += nbytes
                break
            nbytes -= left
            sent += left
            self._sizes.popleft()
            delays.append(now - self._arrivals.popleft())
            self._head_sent = 0
        self.backlog_bytes -= sent
        return sent


@dataclass
class UEPopulation:
    """Every UE in the cell, as parallel arrays indexed by UE number.

    UEs are numbered group by group in scenario order. Full-buffer UEs always
    have data to send, so they have no packet buffer.
    """

    group: npt.NDArray[np.intp]
    full_buffer: npt.NDArray[np.bool_]
    buffers: list[PacketBuffer | None]
    backlog_bytes: npt.NDArray[np.int64]
    """Bytes waiting for each UE; 0 for full-buffer UEs, which never run dry."""

    @classmethod
    def from_groups(cls, counts: list[int], full_buffer: list[bool]) -> UEPopulation:
        """Create ``counts[g]`` UEs for each group ``g``."""
        if len(counts) != len(full_buffer):
            raise ValueError("counts and full_buffer must have one entry per group")
        group = np.repeat(np.arange(len(counts), dtype=np.intp), counts)
        full = np.repeat(np.asarray(full_buffer, dtype=np.bool_), counts)
        buffers = [None if f else PacketBuffer() for f in full.tolist()]
        backlog = np.zeros(group.size, dtype=np.int64)
        return cls(group=group, full_buffer=full, buffers=buffers, backlog_bytes=backlog)

    @property
    def size(self) -> int:
        return int(self.group.size)

    def enqueue(self, ue: int, arrival_times: list[float], sizes: list[int]) -> None:
        """Queue packets for one UE, in arrival order."""
        buffer = self.buffers[ue]
        if buffer is None:
            raise ValueError(f"UE {ue} is full-buffer and takes no packets")
        before = buffer.backlog_bytes
        buffer.push(arrival_times, sizes)
        self.backlog_bytes[ue] += buffer.backlog_bytes - before

    def serve(self, ue: int, nbytes: int, now: float, delays: list[float]) -> int:
        """Send up to ``nbytes`` to a UE at ``now``; see PacketBuffer.serve."""
        buffer = self.buffers[ue]
        if buffer is None:
            return nbytes
        sent = buffer.serve(nbytes, now, delays)
        self.backlog_bytes[ue] -= sent
        return sent
