"""UE state: who each UE is and the downlink packets waiting for it at the gNB."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class PacketBuffer:
    """First-in, first-out queue of the packets waiting for one UE.

    Packets are served in arrival order and may be split across transport
    blocks. Each packet has a sequence number, its position in the UE's
    stream counting from 0, so the blocks carrying it can be traced.
    """

    __slots__ = ("_arrivals", "_head_sent", "_sizes", "backlog_bytes", "head_seq")

    def __init__(self) -> None:
        self._arrivals: deque[float] = deque()
        self._sizes: deque[int] = deque()
        self._head_sent = 0
        self.backlog_bytes = 0
        """Bytes still to send, the partly sent head packet included."""
        self.head_seq = 0
        """Sequence number of the packet at the head of the queue."""

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

    def waiting_since(self, time: float) -> int:
        """How many queued packets arrived before ``time``."""
        count = 0
        for arrival in self._arrivals:
            if arrival >= time:
                break
            count += 1
        return count

    @property
    def head_started(self) -> bool:
        """Whether part of the head packet has already been sent."""
        return self._head_sent > 0

    def serve(self, nbytes: int, completed: list[float]) -> int:
        """Send up to ``nbytes`` from the head of the queue.

        Appends the arrival time of every packet whose last byte was sent to
        ``completed`` and returns the bytes actually sent, which is less than
        ``nbytes`` only when the queue runs dry.
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
            completed.append(self._arrivals.popleft())
            self._head_sent = 0
            self.head_seq += 1
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

    def serve(self, ue: int, nbytes: int, completed: list[float]) -> int:
        """Send up to ``nbytes`` to a UE; see PacketBuffer.serve."""
        buffer = self.buffers[ue]
        if buffer is None:
            return nbytes
        sent = buffer.serve(nbytes, completed)
        self.backlog_bytes[ue] -= sent
        return sent
