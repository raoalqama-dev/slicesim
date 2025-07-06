from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from factories import scenario_data
from slicesim.config import EmbbSla, MmtcSla, Scenario, UrllcSla, load_scenario, parse_scenario
from slicesim.engine import Simulation, simulate
from slicesim.slicing import (
    DelayAware,
    Proportional,
    SliceDemand,
    SlicingPolicy,
    StaticQuotas,
    WholeCell,
    build_policy,
)
from slicesim.slicing.sla import describe, embb_windows, packet_violations

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"


def sliced(policy: str = "delay_aware", **changes: Any) -> Scenario:
    """Three small slices over the fixed link."""
    data = scenario_data(
        duration_s=0.5,
        slicing={"policy": policy, "window_ms": 50},
        slices=[
            {"name": "embb", "sla": {"type": "embb", "min_throughput_mbps": 1}, "quota": 0.6},
            {"name": "urllc", "sla": {"type": "urllc"}, "quota": 0.3},
            {"name": "mmtc", "sla": {"type": "mmtc"}, "quota": 0.1},
        ],
        ues=[
            {
                "name": "video",
                "count": 3,
                "slice": "embb",
                "traffic": {"model": "full_buffer"},
            },
            {
                "name": "ctrl",
                "count": 4,
                "slice": "urllc",
                "traffic": {"model": "periodic", "period_ms": 2, "packet_bytes": 150},
            },
            {
                "name": "sensors",
                "count": 20,
                "slice": "mmtc",
                "traffic": {"model": "poisson", "rate_pps": 20, "packet_bytes": 60},
            },
        ],
    )
    data.update(changes)
    return parse_scenario(data)


class TestSla:
    def test_packet_violations_count_late_lost_and_overdue(self) -> None:
        delays = np.array([0.001, 0.004, 0.006, 0.010])
        assert packet_violations(delays, lost=1, overdue=2, budget_s=0.005) == (5, 7)
        assert packet_violations(np.empty(0), 0, 0, 0.005) == (0, 0)

    def test_embb_windows_owe_the_minimum_or_the_demand(self) -> None:
        # Two windows, three UEs; the minimum is 1,000 bits a window.
        delivered = np.array([[125, 10, 0], [50, 10, 0]])
        offered = np.array([[200, 10, 0], [200, 10, 0]])
        backlog = np.zeros((2, 3), dtype=np.int64)
        full = np.array([False, False, False])
        violation, satisfaction = embb_windows(delivered, offered, backlog, full, 1000.0)
        # UE 0 got all it was owed, then 400 of 1,000 bits; UE 1 was owed only what it
        # offered and got it; UE 2 asked for nothing and does not count.
        assert violation == pytest.approx(1 / 4)
        assert satisfaction == pytest.approx((1 + 0.4 + 1 + 1) / 4)

    def test_embb_full_buffer_ues_are_always_owed_the_minimum(self) -> None:
        delivered = np.array([[100]])
        zeros = np.zeros((1, 1), dtype=np.int64)
        violation, satisfaction = embb_windows(delivered, zeros, zeros, np.array([True]), 1000.0)
        assert violation == 1.0
        assert satisfaction == pytest.approx(0.8)

    def test_embb_without_demand_is_unmeasured(self) -> None:
        zeros = np.zeros((2, 2), dtype=np.int64)
        assert embb_windows(zeros, zeros, zeros, np.array([False, False]), 1.0) == (None, None)

    def test_describe(self) -> None:
        assert describe(EmbbSla(type="embb", min_throughput_mbps=2), 100) == (
            "2 Mbit/s per UE in 0.95 of 100 ms windows"
        )
        assert describe(UrllcSla(type="urllc"), 100) == "0.999 of packets within 5 ms"
        assert describe(MmtcSla(type="mmtc"), 100) == "0.99 of packets within 1000 ms"


class TestSlicedEngine:
    def test_build_policy(self) -> None:
        for name, kind in (
            ("static", StaticQuotas),
            ("delay_aware", DelayAware),
            ("proportional", Proportional),
        ):
            s = sliced(name)
            sim = Simulation(s)
            assert isinstance(sim.policy, kind)
            assert isinstance(build_policy(s, sim.slices, 51), kind)
        assert isinstance(Simulation(parse_scenario(scenario_data())).policy, WholeCell)

    def test_every_ue_belongs_to_its_groups_slice(self) -> None:
        sim = Simulation(sliced())
        assert [s.name for s in sim.slices] == ["embb", "urllc", "mmtc"]
        assert sim.ue_slice.tolist() == [0] * 3 + [1] * 4 + [2] * 20
        assert [s.members.size for s in sim.slices] == [3, 4, 20]
        assert [s.scheduler.n_ues for s in sim.slices] == [3, 4, 20]

    def test_static_slices_never_exceed_their_quota(self) -> None:
        r = simulate(sliced("static"))
        assert np.all(r.slice_prbs <= np.array([31, 15, 5]))
        # The full-buffer eMBB slice could use the whole carrier but keeps to its share.
        assert np.all(r.slice_prbs[:, 0] == 31)
        assert np.max(r.prb_used) <= 51

    def test_work_conserving_policies_give_embb_the_rest(self) -> None:
        r = simulate(sliced("delay_aware"))
        assert np.all(r.prb_used == 51)
        assert np.all(r.slice_prbs.sum(axis=1) == r.prb_used)

    def test_delay_aware_serves_urllc_in_the_next_slot(self) -> None:
        def urllc_delays(policy: str) -> npt.NDArray[np.float64]:
            r = simulate(sliced(policy))
            delays: npt.NDArray[np.float64] = r.delay_s[r.ue_slice[r.delay_ue] == 1]
            return delays

        # Every packet goes out in the slot after it arrives, so within two slots.
        assert np.max(urllc_delays("delay_aware")) <= 1e-3
        # Proportional splits with a full-buffer slice: some URLLC packets wait a slot.
        waited = np.mean(urllc_delays("proportional") > 1e-3)
        assert 0.1 < waited < 0.5

    def test_slice_summaries(self) -> None:
        s = sliced("delay_aware")
        r = simulate(s)
        embb, urllc, mmtc = r.summary().slices
        assert (embb.sla, urllc.sla, mmtc.sla) == ("embb", "urllc", "mmtc")
        assert (embb.ues, urllc.ues, mmtc.ues) == (3, 4, 20)
        assert sum(x.prb_share for x in (embb, urllc, mmtc)) == pytest.approx(1.0)
        assert urllc.packets_delivered == int(r.packets_delivered[r.ue_slice == 1].sum())
        assert embb.delay is None
        assert embb.satisfaction is not None
        assert mmtc.served_density_per_km2 is not None
        area = np.pi * (300**2 - 35**2) / 1e6
        assert mmtc.served_density_per_km2 == pytest.approx(20 / area)
        assert embb.served_density_per_km2 is None
        assert embb.target == "1 Mbit/s per UE in 0.95 of 50 ms windows"

    def test_windows_split_the_run(self) -> None:
        r = simulate(sliced())
        # 50 ms windows of 0.5 ms slots: 10 windows in 0.5 s.
        assert r.window_slots == 100
        assert r.window_delivered.shape == (10, 27)
        assert np.array_equal(r.window_delivered.sum(axis=0), r.delivered_bytes)
        assert np.array_equal(r.window_offered.sum(axis=0), r.offered_bytes)

    def test_overdue_packets_count_as_violations(self) -> None:
        # Nothing can be sent: the URLLC slice's packets pile up past their budget.
        s = sliced("static", link={"model": "fixed", "bits_per_prb": 1})
        sim = Simulation(s)
        r = sim.run()
        assert r.overdue_packets[r.ue_slice == 1].sum() > 0
        urllc = r.summary().slices[1]
        assert urllc.violation_ratio == pytest.approx(1.0)
        assert urllc.met is False

    def test_the_policy_can_be_replaced_between_slots(self) -> None:
        class GiveUrllcEverything(SlicingPolicy):
            name = "urllc_only"

            def split(self, demand: SliceDemand) -> np.ndarray:
                return np.array([0, demand.free, 0], dtype=np.int64)

        sim = Simulation(sliced())
        sim.step(10)
        sim.policy = GiveUrllcEverything()
        sim.step(10)
        r = sim.result()
        assert np.all(r.slice_prbs[10:20, 0] == 0)
        assert np.all(r.slice_prbs[10:20, 2] == 0)

    def test_a_policy_that_overspends_is_an_error(self) -> None:
        class Greedy(SlicingPolicy):
            name = "greedy"

            def split(self, demand: SliceDemand) -> np.ndarray:
                return np.full(3, demand.free, dtype=np.int64)

        sim = Simulation(sliced())
        sim.policy = Greedy()
        with pytest.raises(RuntimeError, match="policy greedy split"):
            sim.step()

    def test_cqi_back_off_lowers_the_slices_cqi(self) -> None:
        def mean_cqi(backoff: float) -> float:
            s = sliced(link={"model": "adaptive"}, channel={"fading": "none"})
            data = s.model_dump(exclude_unset=True)
            data["slices"][1]["cqi_backoff_db"] = backoff
            r = simulate(parse_scenario(data))
            assert r.mean_cqi is not None
            return float(r.mean_cqi[r.ue_slice == 1].mean())

        assert mean_cqi(6.0) < mean_cqi(0.0) - 1

    def test_unsliced_cells_report_no_slices(self) -> None:
        assert simulate(parse_scenario(scenario_data(duration_s=0.05))).summary().slices == []


def test_static_quotas_violate_urllc_while_delay_aware_does_not() -> None:
    """The regression pinned by the spec, on the medium-load example scenario."""
    medium = load_scenario(SCENARIOS / "medium.yaml")
    results = {}
    for policy in ("static", "delay_aware"):
        data = medium.model_dump(exclude_unset=True)
        data["slicing"]["policy"] = policy
        results[policy] = simulate(parse_scenario(data)).summary().slices[1]
    static, delay_aware = results["static"], results["delay_aware"]
    assert static.name == delay_aware.name == "urllc"
    assert static.violation_ratio is not None
    assert delay_aware.violation_ratio is not None
    assert static.met is False
    assert static.violation_ratio > 0.002
    assert delay_aware.met is True
    assert delay_aware.violation_ratio <= 0.001


def test_late_packets_are_counted_against_each_ues_budget() -> None:
    s = sliced("proportional")
    sim = Simulation(s)
    r = sim.run()
    budget = sim.delay_budget_s
    assert budget[r.ue_slice == 0].tolist() == [float("inf")] * 3
    assert np.all(budget[r.ue_slice == 1] == 0.005)
    late = np.bincount(r.delay_ue[r.delay_s > budget[r.delay_ue]], minlength=r.ue_slice.size)
    assert np.array_equal(r.packets_late, late)
