from typing import Any

import numpy as np
import pytest

from slicesim.config import MaxCIScheduler, ProportionalFairScheduler, RoundRobinScheduler
from slicesim.schedulers import (
    MaxCI,
    ProportionalFair,
    RoundRobin,
    Scheduler,
    SlotDemand,
    build_scheduler,
)
from slicesim.schedulers.base import fill_in_order, rank
from slicesim.schedulers.round_robin import share_prbs


def demand(need: list[int], bits: list[int], n_prb: int = 51, slot: int = 0) -> SlotDemand:
    return SlotDemand(
        slot=slot,
        n_prb=n_prb,
        need=np.array(need, dtype=np.int64),
        bits_per_prb=np.array(bits, dtype=np.int64),
    )


class TestSharePrbs:
    def test_equal_needs_share_equally(self) -> None:
        alloc = share_prbs(np.array([100, 100, 100, 100]), 52, 0)
        assert alloc.tolist() == [13, 13, 13, 13]

    def test_what_a_ue_cannot_use_goes_to_the_others(self) -> None:
        alloc = share_prbs(np.array([2, 100, 5, 100]), 51, 0)
        assert alloc.tolist() == [2, 22, 5, 22]

    def test_nobody_gets_more_than_it_needs(self) -> None:
        need = np.array([3, 0, 1, 4])
        assert share_prbs(need, 51, 0).tolist() == need.tolist()

    def test_remainder_blocks_rotate_between_ues(self) -> None:
        need = np.array([50, 50, 50])
        total = np.sum([share_prbs(need, 10, slot) for slot in range(3)], axis=0)
        assert total.tolist() == [10, 10, 10]

    def test_more_ues_than_blocks(self) -> None:
        need = np.ones(7, dtype=np.int64)
        assert share_prbs(need, 3, 0).tolist() == [1, 1, 1, 0, 0, 0, 0]
        assert share_prbs(need, 3, 5).tolist() == [1, 0, 0, 0, 0, 1, 1]

    def test_no_demand_allocates_nothing(self) -> None:
        assert share_prbs(np.zeros(3, dtype=np.int64), 51, 0).tolist() == [0, 0, 0]

    @pytest.mark.parametrize("seed", range(20))
    def test_max_min_fair(self, seed: int) -> None:
        gen = np.random.default_rng(seed)
        need = gen.integers(0, 30, size=gen.integers(1, 40))
        alloc = share_prbs(need, int(gen.integers(1, 120)), int(gen.integers(0, 1000)))
        # A UE that got less than it needs got at least as much as anyone, give or take one.
        short = alloc < need
        if short.any():
            assert np.min(alloc[short]) >= np.max(alloc) - 1


class TestHelpers:
    def test_fill_in_order(self) -> None:
        need = np.array([10, 30, 20, 5])
        order = np.array([1, 2, 0, 3])
        assert fill_in_order(order, need, 45).tolist() == [0, 30, 15, 0]
        assert fill_in_order(order, need, 100).tolist() == [10, 30, 20, 5]
        assert fill_in_order(order, need, 0).tolist() == [0, 0, 0, 0]

    def test_rank_breaks_ties_in_turn(self) -> None:
        metric = np.array([1.0, 5.0, 5.0, 5.0])
        firsts = [int(rank(metric, slot)[0]) for slot in range(6)]
        assert sorted(set(firsts)) == [1, 2, 3]
        assert rank(metric, 0).tolist()[-1] == 0


class TestRoundRobin:
    def test_shares_equally_whatever_the_channel(self) -> None:
        alloc = RoundRobin(3).allocate(demand([100, 100, 100], [800, 100, 20]))
        assert alloc.tolist() == [17, 17, 17]


class TestMaxCI:
    def test_serves_the_best_channel_first(self) -> None:
        alloc = MaxCI(4).allocate(demand([30, 30, 30, 30], [100, 800, 400, 200]))
        assert alloc.tolist() == [0, 30, 21, 0]

    def test_skips_ues_with_nothing_to_send(self) -> None:
        alloc = MaxCI(3).allocate(demand([0, 10, 60], [802, 300, 100]))
        assert alloc.tolist() == [0, 10, 41]

    def test_equal_channels_take_turns(self) -> None:
        sched = MaxCI(3)
        winners = {
            int(np.argmax(sched.allocate(demand([60] * 3, [500] * 3, slot=t)))) for t in range(3)
        }
        assert winners == {0, 1, 2}


class TestProportionalFair:
    def test_ranks_by_rate_over_average(self) -> None:
        sched = ProportionalFair(3, time_constant_slots=10)
        sched.average_bits = np.array([1000.0, 100.0, 400.0])
        # Metrics 0.8, 2.0 and 1.0: the poor channel with the poor history goes first.
        alloc = sched.allocate(demand([30, 30, 30], [800, 200, 400]))
        assert alloc.tolist() == [0, 30, 21]

    def test_average_is_an_exponential_moving_average(self) -> None:
        sched = ProportionalFair(2, time_constant_slots=4)
        sched.average_bits = np.array([100.0, 100.0])
        sched.delivered(np.array([500, 0]))
        assert sched.average_bits.tolist() == [200.0, 75.0]

    def test_average_never_falls_below_one_bit(self) -> None:
        sched = ProportionalFair(1, time_constant_slots=2)
        for _ in range(20):
            sched.delivered(np.array([0]))
        assert sched.average_bits.tolist() == [1.0]

    def test_a_poor_channel_still_gets_served(self) -> None:
        sched = ProportionalFair(2, time_constant_slots=20)
        served = np.zeros(2)
        for t in range(200):
            alloc = sched.allocate(demand([51, 51], [800, 80], slot=t))
            bits = alloc * np.array([800, 80])
            sched.delivered(bits)
            served += alloc
        # Over time both UEs get about half the PRBs.
        assert served[1] / served.sum() == pytest.approx(0.5, abs=0.05)

    def test_rejects_a_time_constant_below_one_slot(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ProportionalFair(1, time_constant_slots=0.5)


@pytest.mark.parametrize("make", [RoundRobin, MaxCI, lambda n: ProportionalFair(n, 50)])
@pytest.mark.parametrize("seed", range(10))
def test_every_scheduler_respects_needs_and_uses_every_prb_it_can(make: Any, seed: int) -> None:
    gen = np.random.default_rng(seed)
    n = int(gen.integers(1, 30))
    sched: Scheduler = make(n)
    for t in range(20):
        need = gen.integers(0, 40, n)
        bits = gen.integers(1, 803, n)
        n_prb = int(gen.integers(0, 120))
        alloc = sched.allocate(
            SlotDemand(slot=t, n_prb=n_prb, need=need.astype(np.int64), bits_per_prb=bits)
        )
        assert np.all(alloc >= 0)
        assert np.all(alloc <= need)
        assert alloc.sum() == min(n_prb, need.sum())
        sched.delivered(alloc * bits)


def test_build_scheduler() -> None:
    assert isinstance(build_scheduler(RoundRobinScheduler(), 3, 5e-4), RoundRobin)
    assert isinstance(build_scheduler(MaxCIScheduler(), 3, 5e-4), MaxCI)
    pf = build_scheduler(ProportionalFairScheduler(time_constant_ms=100), 3, 5e-4)
    assert isinstance(pf, ProportionalFair)
    assert pf.time_constant_slots == pytest.approx(200)
    short = build_scheduler(ProportionalFairScheduler(time_constant_ms=0.1), 3, 1e-3)
    assert isinstance(short, ProportionalFair)
    assert short.time_constant_slots == 1.0
