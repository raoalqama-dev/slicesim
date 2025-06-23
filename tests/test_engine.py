from collections.abc import Callable
from dataclasses import fields
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from factories import poisson_group, scenario
from slicesim.config import Scenario
from slicesim.engine import RngStreams, Simulation, share_prbs, simulate
from slicesim.harq import TransportBlock
from slicesim.metrics import SimulationResult
from slicesim.traffic import Periodic


def assert_identical(a: SimulationResult, b: SimulationResult) -> None:
    for field in fields(SimulationResult):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, np.ndarray):
            assert np.array_equal(x, y), field.name
        else:
            assert x == y, field.name


class TestRngStreams:
    def test_a_stream_depends_only_on_the_seed_and_its_name(self) -> None:
        first = RngStreams(5)
        first.generator("fading")
        a = first.generator("traffic", "web").random(4)
        b = RngStreams(5).generator("traffic", "web").random(4)
        assert np.array_equal(a, b)

    def test_streams_are_distinct(self) -> None:
        streams = RngStreams(5)
        draws = [
            streams.generator("traffic", "web").random(4),
            streams.generator("traffic", "video").random(4),
            streams.generator("traffic", 1).random(4),
            streams.generator("fading").random(4),
            RngStreams(6).generator("fading").random(4),
        ]
        for i, x in enumerate(draws):
            for y in draws[i + 1 :]:
                assert not np.array_equal(x, y)

    def test_negative_seeds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            RngStreams(-1)


class TestSharePrbs:
    def test_equal_needs_share_equally(self) -> None:
        alloc = share_prbs(np.array([100, 100, 100, 100]), 52, 0)
        assert alloc.tolist() == [13, 13, 13, 13]

    def test_what_a_ue_cannot_use_goes_to_the_others(self) -> None:
        alloc = share_prbs(np.array([2, 100, 5, 100]), 51, 0)
        assert alloc.tolist() == [2, 22, 5, 22]

    def test_nobody_gets_more_than_it_needs(self) -> None:
        need = np.array([3, 0, 1, 4])
        alloc = share_prbs(need, 51, 0)
        assert alloc.tolist() == need.tolist()

    def test_remainder_blocks_rotate_between_ues(self) -> None:
        need = np.array([50, 50, 50])
        total = np.sum([share_prbs(need, 10, slot) for slot in range(3)], axis=0)
        assert total.tolist() == [10, 10, 10]
        assert share_prbs(need, 10, 0).sum() == 10

    def test_more_ues_than_blocks(self) -> None:
        need = np.ones(7, dtype=np.int64)
        assert share_prbs(need, 3, 0).tolist() == [1, 1, 1, 0, 0, 0, 0]
        assert share_prbs(need, 3, 5).tolist() == [1, 0, 0, 0, 0, 1, 1]

    def test_no_demand_allocates_nothing(self) -> None:
        assert share_prbs(np.zeros(3, dtype=np.int64), 51, 0).tolist() == [0, 0, 0]

    @pytest.mark.parametrize("seed", range(20))
    def test_allocation_invariants(self, seed: int) -> None:
        gen = np.random.default_rng(seed)
        need = gen.integers(0, 30, size=gen.integers(1, 40))
        n_prb = int(gen.integers(1, 120))
        alloc = share_prbs(need, n_prb, int(gen.integers(0, 1000)))
        assert np.all(alloc <= need)
        assert alloc.sum() == min(n_prb, need.sum())
        # Max-min fair: a UE that got less than it needs got at least as much as anyone.
        short = alloc < need
        if short.any():
            assert np.min(alloc[short]) >= np.max(alloc) - 1


class TestDeterminism:
    def test_the_same_seed_gives_identical_results(self) -> None:
        assert_identical(simulate(scenario()), simulate(scenario()))

    def test_a_different_seed_gives_different_results(self) -> None:
        a, b = simulate(scenario(seed=1)), simulate(scenario(seed=2))
        assert not np.array_equal(a.offered_bytes, b.offered_bytes)

    def test_changing_one_groups_traffic_leaves_the_others_arrivals_alone(self) -> None:
        base = scenario()
        data = base.model_dump()
        data["ues"][2]["traffic"] = {"model": "poisson", "rate_pps": 900, "packet_bytes": 1200}
        changed = base.model_validate(data)
        data["ues"].insert(0, poisson_group("extra", count=2))
        extended = base.model_validate(data)
        a, b, c = simulate(base), simulate(changed), simulate(extended)
        web = a.ue_group == 0
        assert np.array_equal(a.offered_bytes[web], b.offered_bytes[web])
        assert np.array_equal(a.packets_arrived[web], b.packets_arrived[web])
        # Inserting a group renumbers the UEs but keeps each group's stream.
        assert np.array_equal(a.offered_bytes[web], c.offered_bytes[c.ue_group == 1])
        assert not np.array_equal(
            a.offered_bytes[a.ue_group == 2], b.offered_bytes[b.ue_group == 2]
        )

    def test_stepping_in_chunks_matches_one_run(self) -> None:
        sim = Simulation(scenario())
        while not sim.finished:
            sim.step(37)
        assert_identical(sim.result(), simulate(scenario()))


class TestSlotLoop:
    def test_bytes_are_conserved(self) -> None:
        r = simulate(scenario(duration_s=1.0))
        packets = ~r.full_buffer
        assert np.array_equal(
            r.delivered_bytes[packets] + r.backlog_bytes[packets], r.offered_bytes[packets]
        )
        assert r.packets_delivered.sum() == r.delay_s.size
        assert np.array_equal(
            np.bincount(r.delay_ue, minlength=r.ue_group.size), r.packets_delivered
        )

    def test_packets_wait_for_the_slot_after_they_arrive(self) -> None:
        r = simulate(scenario(duration_s=1.0))
        assert np.min(r.delay_s) > r.carrier.slot_duration_s

    def test_a_lightly_loaded_cell_delivers_in_the_next_slot(self) -> None:
        s = scenario(ues=[poisson_group(count=5, rate_pps=50)], duration_s=2.0)
        r = simulate(s)
        slot = r.carrier.slot_duration_s
        assert r.delay_s.size > 400
        assert np.all((r.delay_s > slot) & (r.delay_s <= 2 * slot + 1e-12))

    def test_a_packet_larger_than_a_slot_takes_several_slots(self) -> None:
        # 106 PRBs of 8 bits carry 106 bytes per 1 ms slot; 300 bytes need 3 slots.
        s = scenario(
            carrier={"bandwidth_mhz": 20, "scs_khz": 15},
            link={"model": "fixed", "bits_per_prb": 8},
            duration_s=0.1,
            ues=[
                {
                    "name": "u",
                    "count": 1,
                    "traffic": {"model": "periodic", "period_ms": 10, "packet_bytes": 300},
                }
            ],
        )
        sim = Simulation(s)
        r = sim.run()
        source = sim.sources[0]
        assert isinstance(source, Periodic)
        phase = float(source.phase[0])
        arrivals = phase + 0.01 * np.arange(r.delay_s.size)
        arrival_slot = np.floor(arrivals / 1e-3)
        expected = (arrival_slot + 4) * 1e-3 - arrivals
        assert np.allclose(r.delay_s, expected)

    def test_full_buffer_ues_fill_the_cell(self) -> None:
        s = scenario(ues=[{"name": "bulk", "count": 2, "traffic": {"model": "full_buffer"}}])
        r = simulate(s)
        assert np.all(r.prb_used == 51)
        # 51 PRBs alternate 26/25 and 25/26 between the two UEs, 500 bits each.
        per_slot = (26 * 500 // 8) + (25 * 500 // 8)
        assert r.delivered_bytes.sum() == per_slot * r.n_slots
        assert abs(int(r.delivered_bytes[0]) - int(r.delivered_bytes[1])) <= 26 * 500 // 8

    def test_an_overloaded_cell_builds_a_queue(self) -> None:
        s = scenario(ues=[poisson_group(count=4, rate_pps=6000)], duration_s=1.0)
        r = simulate(s)
        capacity_bytes = 51 * 500 // 8 * r.n_slots
        assert r.offered_bytes.sum() > 1.5 * capacity_bytes
        assert r.delivered_bytes.sum() >= 0.99 * capacity_bytes
        assert r.backlog_bytes.sum() > 0
        assert np.all(r.prb_used[100:] == 51)

    def test_step_stops_at_the_end_of_the_run(self) -> None:
        sim = Simulation(scenario(duration_s=0.01))
        assert sim.n_slots == 20
        assert sim.now == 0.0
        sim.step(5)
        assert sim.now == pytest.approx(0.0025)
        assert sim.result().n_slots == 5
        assert sim.result().prb_used.size == 5
        sim.step(100)
        assert sim.finished
        assert sim.slot == 20
        sim.step()
        assert sim.slot == 20
        with pytest.raises(ValueError, match="non-negative"):
            sim.step(-1)

    def test_the_spec_scale_runs(self) -> None:
        groups = [poisson_group(f"g{i}", count=10, rate_pps=100) for i in range(10)]
        r = simulate(scenario(ues=groups, duration_s=1.0))
        assert r.ue_group.size == 100
        assert r.n_slots == 2000


def radio(**changes: Any) -> Scenario:
    """A scenario over the radio channel, by default without shadowing or fading."""
    defaults: dict[str, Any] = {
        "link": {"model": "adaptive", "cqi_delay_slots": 0},
        "cell": {"radius_m": 100},
        "channel": {"shadowing": False, "fading": "none"},
    }
    return scenario(**{**defaults, **changes})


def one_ue(period_ms: float, duration_s: float = 0.05) -> Scenario:
    return radio(
        duration_s=duration_s,
        ues=[
            {
                "name": "u",
                "count": 1,
                "traffic": {"model": "periodic", "period_ms": period_ms, "packet_bytes": 100},
            }
        ],
    )


def always(value: float) -> Callable[..., npt.NDArray[np.float64]]:
    def fake_bler(sinr_db: npt.ArrayLike, cqi: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.full(np.shape(sinr_db), value)

    return fake_bler


class TestRadioLink:
    def test_runs_are_reproducible(self) -> None:
        s = radio(channel={"fading": "rayleigh", "coherence_ms": 2}, duration_s=0.3)
        assert_identical(simulate(s), simulate(s))

    def test_bytes_are_conserved_through_harq(self) -> None:
        s = radio(
            link={"model": "adaptive", "cqi_delay_slots": 6, "max_transmissions": 2},
            channel={"fading": "rayleigh", "coherence_ms": 1},
            duration_s=1.0,
        )
        r = simulate(s)
        assert r.blocks_failed.sum() > 0
        assert r.blocks_dropped.sum() > 0
        assert r.packets_lost.sum() > 0
        p = ~r.full_buffer
        assert np.array_equal(
            r.delivered_bytes[p] + r.backlog_bytes[p] + r.in_flight_bytes[p] + r.dropped_bytes[p],
            r.offered_bytes[p],
        )
        assert np.all(r.packets_delivered + r.packets_lost <= r.packets_arrived)
        assert np.all(r.retransmissions <= r.blocks_failed * (2 - 1))

    def test_a_retransmission_adds_one_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("slicesim.engine.bler", always(1.0))
        monkeypatch.setattr("slicesim.harq.bler", always(0.0))
        sim = Simulation(one_ue(period_ms=10))
        r = sim.run()
        source = sim.sources[0]
        assert isinstance(source, Periodic)
        slot = 0.5e-3
        arrivals = source.phase[0] + 0.01 * np.arange(r.delay_s.size)
        arrival_slot = np.floor(arrivals / slot)
        # Sent in the next slot, failed, resent 8 slots later and decoded.
        assert np.allclose(r.delay_s, (arrival_slot + 10) * slot - arrivals)
        assert r.blocks_failed[0] == r.blocks_sent[0]
        # One retransmission per delivered packet; the last may still be pending.
        assert r.retransmissions[0] == r.packets_delivered[0]
        assert r.packets_lost[0] == 0

    def test_later_blocks_wait_for_an_earlier_retransmission(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        failed: list[bool] = []

        def fail_first_block(sinr_db: npt.ArrayLike, cqi: npt.ArrayLike) -> npt.NDArray[np.float64]:
            shape = np.shape(sinr_db)
            if not failed and np.size(sinr_db):
                failed.append(True)
                return np.ones(shape)
            return np.zeros(shape)

        monkeypatch.setattr("slicesim.engine.bler", fail_first_block)
        monkeypatch.setattr("slicesim.harq.bler", always(0.0))
        # A packet every slot: packet k is sent in slot k + 1.
        sim = Simulation(one_ue(period_ms=0.5, duration_s=0.01))
        r = sim.run()
        source = sim.sources[0]
        assert isinstance(source, Periodic)
        arrivals = source.phase[0] + 0.5e-3 * np.arange(9)
        # The first block is resent in slot 9; the eight packets behind it wait for it.
        assert np.allclose(r.delay_s[:9], 10 * 0.5e-3 - arrivals)
        later = r.delay_s[9:]
        assert np.all(later < 2 * 0.5e-3 + 1e-12)
        assert r.retransmissions[0] == 1

    def test_four_failures_lose_the_packet(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("slicesim.engine.bler", always(1.0))
        monkeypatch.setattr("slicesim.harq.bler", always(1.0))
        r = simulate(one_ue(period_ms=10, duration_s=0.1))
        assert r.packets_delivered[0] == 0
        assert r.packets_lost[0] == r.blocks_dropped[0] >= 6
        assert r.delivered_bytes[0] == 0
        assert r.retransmissions[0] >= 3 * r.blocks_dropped[0]

    def test_a_single_transmission_drops_at_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("slicesim.engine.bler", always(1.0))
        s = one_ue(period_ms=10, duration_s=0.1)
        data = s.model_dump()
        data["link"]["max_transmissions"] = 1
        r = simulate(s.model_validate(data))
        assert r.retransmissions[0] == 0
        assert r.blocks_dropped[0] == r.blocks_sent[0] == r.packets_lost[0] == 10
        assert r.packets_delivered[0] == 0

    def test_retransmissions_that_cannot_fit_are_an_error(self) -> None:
        sim = Simulation(one_ue(period_ms=10))
        assert sim.harq is not None
        b = TransportBlock(
            ue=0, prbs=52, cqi=9, nbytes=10, first_seq=0, arrivals=[], partial_tail=False
        )
        b.sinr_db.append(0.0)
        sim.harq.failed(b, slot=-8)
        with pytest.raises(RuntimeError, match="need 52 PRBs"):
            sim.step()

    def test_the_scheduler_uses_the_cqi_reported_earlier(self) -> None:
        def alternating(slot: int) -> npt.NDArray[np.float64]:
            # CQI 0 on even slots, CQI 15 on odd ones.
            return np.full(1, -20.0 if slot % 2 == 0 else 30.0)

        totals = []
        for delay in (0, 1):
            s = radio(
                link={"model": "adaptive", "cqi_delay_slots": delay},
                duration_s=0.005,
                ues=[{"name": "bulk", "count": 1, "traffic": {"model": "full_buffer"}}],
            )
            sim = Simulation(s)
            assert sim.channel is not None
            sim.channel.sinr_db = alternating  # type: ignore[method-assign]
            sim.run()
            totals.append(int(sim.metrics.cqi_sum[0]))
        # Ten slots: 0,15,0,15,... reported at once, and 0,0,15,0,15,... a slot late.
        assert totals == [75, 60]

    def test_ues_out_of_coverage_get_nothing(self) -> None:
        s = radio(
            cell={"radius_m": 5000, "min_distance_m": 4900, "tx_power_dbm": 0},
            ues=[{"name": "far", "count": 3, "traffic": {"model": "full_buffer"}}],
        )
        r = simulate(s)
        assert r.mean_cqi is not None
        assert np.all(r.mean_cqi == 0)
        assert r.delivered_bytes.sum() == 0
        assert r.prb_used.sum() == 0

    def test_retransmissions_go_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("slicesim.engine.bler", always(1.0))
        monkeypatch.setattr("slicesim.harq.bler", always(0.0))
        s = radio(
            duration_s=0.05,
            ues=[{"name": "bulk", "count": 2, "traffic": {"model": "full_buffer"}}],
        )
        r = simulate(s)
        # Every new block fails. Eight slots later its retransmission fills the cell and
        # decodes, leaving nothing for new data, so new data and retransmissions
        # alternate in runs of eight slots: 52 slots of new data, 48 of retransmissions.
        assert np.all(r.prb_used == 51)
        assert r.blocks_sent.sum() == 2 * 52
        assert r.retransmissions.sum() == 2 * 48
        assert r.blocks_dropped.sum() == 0
