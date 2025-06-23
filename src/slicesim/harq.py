"""HARQ retransmission and in-order delivery of packets.

A transport block that fails to decode is sent again ``rtt_slots`` later, on
the same number of PRBs at the same CQI, and the UE chase-combines every copy
it has received: the effective SINR is the linear sum of the copies' SINRs.
After ``max_transmissions`` failures the block is dropped.

Packets reach the UE in order, as PDCP reordering would deliver them: a packet
is delivered once the block carrying its last byte and every earlier block for
that UE have been resolved, and it is lost if any of its bytes were in a
dropped block. A block that waits for an earlier one therefore adds that wait
to the delay of its packets, which is how retransmissions show up in delay.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from slicesim.channel.link import bler, combine_db


class BlockState(Enum):
    PENDING = "pending"
    DECODED = "decoded"
    DROPPED = "dropped"


@dataclass(slots=True)
class TransportBlock:
    """One transport block for one UE, and what became of it."""

    ue: int
    prbs: int
    cqi: int
    nbytes: int
    first_seq: int
    """Sequence number of the first packet with bytes in this block."""
    arrivals: list[float]
    """Arrival times of the packets whose last byte is in this block."""
    partial_tail: bool
    """Whether the block ends part way through a packet."""
    sinr_db: list[float] = field(default_factory=list)
    """The SINR of each transmission so far."""
    due_slot: int = 0
    state: BlockState = BlockState.PENDING

    @property
    def transmissions(self) -> int:
        return len(self.sinr_db)

    @property
    def last_seq(self) -> int:
        """Sequence number of the last packet with bytes in this block."""
        return self.first_seq + len(self.arrivals) - (0 if self.partial_tail else 1)


class DeliveryQueue:
    """One UE's blocks awaiting in-order release, and the packets known to be lost."""

    __slots__ = ("_blocks", "_lost")

    def __init__(self) -> None:
        self._blocks: deque[TransportBlock] = deque()
        self._lost: set[int] = set()

    def __len__(self) -> int:
        return len(self._blocks)

    def add(self, block: TransportBlock) -> None:
        """Queue a block behind those sent before it."""
        self._blocks.append(block)

    def deliver(
        self, first_seq: int, arrivals: list[float], now: float, delays: list[float]
    ) -> int:
        """Release a block decoded at once with nothing queued before it.

        The common case, handled without creating a TransportBlock. Returns
        how many of its packets were lost to an earlier dropped block.
        """
        if self._blocks:
            raise ValueError("earlier blocks are still waiting")
        return self._settle(first_seq, arrivals, False, False, now, delays)

    def release(self, now: float, delays: list[float]) -> int:
        """Release every resolved block at the head of the queue.

        Appends the delay of each delivered packet to ``delays`` and returns
        how many packets were lost.
        """
        lost = 0
        while self._blocks and self._blocks[0].state is not BlockState.PENDING:
            block = self._blocks.popleft()
            lost += self._settle(
                block.first_seq,
                block.arrivals,
                block.partial_tail,
                block.state is BlockState.DROPPED,
                now,
                delays,
            )
        return lost

    def _settle(
        self,
        first_seq: int,
        arrivals: list[float],
        partial_tail: bool,
        dropped: bool,
        now: float,
        delays: list[float],
    ) -> int:
        lost = 0
        if self._lost or dropped:
            for k, arrival in enumerate(arrivals):
                seq = first_seq + k
                if dropped or seq in self._lost:
                    self._lost.discard(seq)
                    lost += 1
                else:
                    delays.append(now - arrival)
        else:
            delays.extend(now - arrival for arrival in arrivals)
        if dropped and partial_tail:
            # The packet this block ends in completes in a later block.
            self._lost.add(first_seq + len(arrivals))
        return lost


class Harq:
    """The cell's blocks waiting to be retransmitted, oldest first."""

    def __init__(self, rtt_slots: int, max_transmissions: int) -> None:
        if rtt_slots < 1 or max_transmissions < 1:
            raise ValueError("rtt_slots and max_transmissions must be at least 1")
        self.rtt_slots = rtt_slots
        self.max_transmissions = max_transmissions
        self._waiting: deque[TransportBlock] = deque()

    def __len__(self) -> int:
        return len(self._waiting)

    def waiting(self) -> list[TransportBlock]:
        return list(self._waiting)

    def due(self, slot: int) -> list[TransportBlock]:
        """Take the blocks due for retransmission by ``slot``, oldest first."""
        blocks = []
        while self._waiting and self._waiting[0].due_slot <= slot:
            blocks.append(self._waiting.popleft())
        return blocks

    def transmit(self, block: TransportBlock, slot: int, sinr_db: float, draw: float) -> None:
        """Send ``block`` at ``slot`` and resolve it.

        ``draw`` is a uniform random number in [0, 1): the block decodes when
        it is at least the BLER at the combined SINR of every copy so far.
        """
        block.sinr_db.append(sinr_db)
        effective = sinr_db if block.transmissions == 1 else combine_db(block.sinr_db)
        if draw >= float(bler(effective, block.cqi)):
            block.state = BlockState.DECODED
        else:
            self.failed(block, slot)

    def failed(self, block: TransportBlock, slot: int) -> None:
        """Handle a block whose latest transmission, at ``slot``, did not decode."""
        if block.transmissions >= self.max_transmissions:
            block.state = BlockState.DROPPED
        else:
            block.due_slot = slot + self.rtt_slots
            self._waiting.append(block)
