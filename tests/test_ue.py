import numpy as np
import pytest

from slicesim.ue import PacketBuffer, UEPopulation


class TestPacketBuffer:
    def test_serves_packets_in_arrival_order(self) -> None:
        buffer = PacketBuffer()
        buffer.push([0.1, 0.2, 0.3], [100, 200, 300])
        assert len(buffer) == 3
        assert buffer.backlog_bytes == 600
        completed: list[float] = []
        assert buffer.serve(300, completed) == 300
        assert completed == [0.1, 0.2]
        assert buffer.backlog_bytes == 300
        assert len(buffer) == 1
        assert buffer.head_seq == 2
        assert not buffer.head_started

    def test_a_packet_split_across_blocks_completes_with_its_last_byte(self) -> None:
        buffer = PacketBuffer()
        buffer.push([0.0], [250])
        completed: list[float] = []
        assert buffer.serve(100, completed) == 100
        assert buffer.head_started
        assert buffer.serve(100, completed) == 100
        assert completed == []
        assert buffer.head_seq == 0
        assert buffer.backlog_bytes == 50
        assert buffer.serve(100, completed) == 50
        assert completed == [0.0]
        assert buffer.head_seq == 1
        assert not buffer.head_started
        assert buffer.backlog_bytes == 0
        assert len(buffer) == 0

    def test_serving_more_than_the_backlog_sends_what_there_is(self) -> None:
        buffer = PacketBuffer()
        buffer.push([0.5, 0.5], [10, 20])
        completed: list[float] = []
        assert buffer.serve(1000, completed) == 30
        assert completed == [0.5, 0.5]
        assert buffer.serve(1000, completed) == 0

    def test_rejects_out_of_order_or_empty_packets(self) -> None:
        buffer = PacketBuffer()
        buffer.push([1.0], [10])
        with pytest.raises(ValueError, match="arrival order"):
            buffer.push([0.5], [10])
        with pytest.raises(ValueError, match="positive"):
            buffer.push([2.0], [0])
        with pytest.raises(ValueError, match="zip"):
            buffer.push([2.0, 3.0], [10])


class TestUEPopulation:
    def test_numbers_ues_group_by_group(self) -> None:
        ues = UEPopulation.from_groups([2, 3, 1], [False, False, True])
        assert ues.size == 6
        assert ues.group.tolist() == [0, 0, 1, 1, 1, 2]
        assert ues.full_buffer.tolist() == [False] * 5 + [True]
        assert [b is None for b in ues.buffers] == [False] * 5 + [True]
        assert ues.backlog_bytes.tolist() == [0] * 6

    def test_backlog_follows_enqueue_and_serve(self) -> None:
        ues = UEPopulation.from_groups([2], [False])
        ues.enqueue(1, [0.0, 0.1], [100, 50])
        assert ues.backlog_bytes.tolist() == [0, 150]
        completed: list[float] = []
        assert ues.serve(1, 120, completed) == 120
        assert ues.backlog_bytes.tolist() == [0, 30]
        assert completed == [0.0]
        buffer = ues.buffers[1]
        assert buffer is not None
        assert buffer.backlog_bytes == 30

    def test_full_buffer_ues_take_whatever_they_are_given(self) -> None:
        ues = UEPopulation.from_groups([1], [True])
        completed: list[float] = []
        assert ues.serve(0, 4096, completed) == 4096
        assert completed == []
        assert ues.backlog_bytes[0] == 0
        with pytest.raises(ValueError, match="full-buffer"):
            ues.enqueue(0, [0.0], [10])

    def test_rejects_mismatched_group_lists(self) -> None:
        with pytest.raises(ValueError, match="one entry per group"):
            UEPopulation.from_groups([1, 2], [False])

    def test_arrays_have_the_documented_types(self) -> None:
        ues = UEPopulation.from_groups([1, 1], [False, True])
        assert ues.group.dtype == np.intp
        assert ues.backlog_bytes.dtype == np.int64
