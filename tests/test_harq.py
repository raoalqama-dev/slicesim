import pytest

from slicesim.channel.link import CQI_SINR_THRESHOLDS_DB
from slicesim.harq import BlockState, DeliveryQueue, Harq, TransportBlock


def block(
    first_seq: int = 0,
    arrivals: list[float] | None = None,
    partial_tail: bool = False,
    state: BlockState = BlockState.PENDING,
    cqi: int = 9,
) -> TransportBlock:
    b = TransportBlock(
        ue=0,
        prbs=10,
        cqi=cqi,
        nbytes=100,
        first_seq=first_seq,
        arrivals=arrivals if arrivals is not None else [],
        partial_tail=partial_tail,
    )
    b.state = state
    return b


def state(b: TransportBlock) -> BlockState:
    """Read a block's state without the narrowing from earlier assertions."""
    return b.state


class TestDeliveryQueue:
    def test_direct_delivery(self) -> None:
        queue = DeliveryQueue()
        delays: list[float] = []
        assert queue.deliver(0, [0.1, 0.2], 1.0, delays) == 0
        assert delays == pytest.approx([0.9, 0.8])

    def test_direct_delivery_refuses_to_overtake(self) -> None:
        queue = DeliveryQueue()
        queue.add(block())
        with pytest.raises(ValueError, match="still waiting"):
            queue.deliver(1, [0.0], 1.0, [])

    def test_a_later_block_waits_for_an_earlier_one(self) -> None:
        queue = DeliveryQueue()
        first = block(0, [0.0])
        queue.add(first)
        queue.add(block(1, [0.1], state=BlockState.DECODED))
        delays: list[float] = []
        assert queue.release(1.0, delays) == 0
        assert delays == []
        assert len(queue) == 2
        first.state = BlockState.DECODED
        assert queue.release(2.0, delays) == 0
        # Both packets are released when the earlier block decodes.
        assert delays == pytest.approx([2.0, 1.9])
        assert len(queue) == 0

    def test_a_dropped_block_loses_its_packets(self) -> None:
        queue = DeliveryQueue()
        queue.add(block(0, [0.0, 0.1], state=BlockState.DROPPED))
        queue.add(block(2, [0.2], state=BlockState.DECODED))
        delays: list[float] = []
        assert queue.release(1.0, delays) == 2
        assert delays == pytest.approx([0.8])

    def test_a_packet_split_across_a_dropped_block_is_lost(self) -> None:
        queue = DeliveryQueue()
        # Packet 0 completes in the first block, packet 1 starts there and ends later.
        queue.add(block(0, [0.0], partial_tail=True, state=BlockState.DROPPED))
        queue.add(block(1, [0.1, 0.2], state=BlockState.DECODED))
        delays: list[float] = []
        assert queue.release(1.0, delays) == 2
        assert delays == pytest.approx([0.8])

    def test_a_split_packet_completing_later_is_lost_on_the_direct_path(self) -> None:
        queue = DeliveryQueue()
        queue.add(block(4, [], partial_tail=True, state=BlockState.DROPPED))
        assert queue.release(1.0, []) == 0
        delays: list[float] = []
        assert queue.deliver(4, [0.5, 0.6], 2.0, delays) == 1
        assert delays == pytest.approx([1.4])
        # The loss is accounted once.
        assert queue.deliver(6, [0.7], 3.0, delays) == 0

    def test_last_seq(self) -> None:
        assert block(3, [0.0, 0.0], partial_tail=False).last_seq == 4
        assert block(3, [0.0, 0.0], partial_tail=True).last_seq == 5
        assert block(3, [], partial_tail=True).last_seq == 3


class TestHarq:
    def test_a_failed_block_is_due_after_the_round_trip(self) -> None:
        harq = Harq(rtt_slots=8, max_transmissions=4)
        b = block()
        b.sinr_db.append(10.0)
        harq.failed(b, slot=3)
        assert b.state is BlockState.PENDING
        assert len(harq) == 1
        assert harq.due(10) == []
        assert harq.due(11) == [b]
        assert len(harq) == 0

    def test_blocks_come_due_oldest_first(self) -> None:
        harq = Harq(rtt_slots=1, max_transmissions=4)
        blocks = [block(k) for k in range(3)]
        for k, b in enumerate(blocks):
            b.sinr_db.append(0.0)
            harq.failed(b, slot=k)
        assert harq.waiting() == blocks
        assert harq.due(2) == blocks[:2]
        assert harq.waiting() == blocks[2:]

    def test_retransmissions_combine_until_they_decode(self) -> None:
        harq = Harq(rtt_slots=8, max_transmissions=4)
        threshold = float(CQI_SINR_THRESHOLDS_DB[8])  # CQI 9
        b = block(cqi=9)
        b.sinr_db.append(threshold - 3.0)
        harq.failed(b, slot=0)
        # A second copy at the same SINR doubles it: back to the threshold, 10% BLER.
        assert harq.due(8) == [b]
        harq.transmit(b, slot=8, sinr_db=threshold - 3.0, draw=0.05)
        assert b.state is BlockState.PENDING
        assert b.due_slot == 16
        assert harq.due(16) == [b]
        harq.transmit(b, slot=16, sinr_db=threshold - 3.0, draw=0.05)
        # Three copies are 1.8 dB above the threshold: BLER about 0.3%.
        assert state(b) is BlockState.DECODED
        assert b.transmissions == 3
        assert len(harq) == 0

    def test_a_block_is_dropped_after_the_last_transmission(self) -> None:
        harq = Harq(rtt_slots=2, max_transmissions=2)
        b = block()
        b.sinr_db.append(-20.0)
        harq.failed(b, 0)
        assert harq.due(2) == [b]
        harq.transmit(b, 2, -20.0, draw=0.5)
        assert b.state is BlockState.DROPPED
        assert len(harq) == 0

    def test_a_single_transmission_drops_at_once(self) -> None:
        harq = Harq(rtt_slots=2, max_transmissions=1)
        b = block()
        b.sinr_db.append(-20.0)
        harq.failed(b, 0)
        assert b.state is BlockState.DROPPED

    def test_invalid_settings(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            Harq(0, 4)
