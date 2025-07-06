import numpy as np
import pytest

from slicesim.slicing import (
    DelayAware,
    Proportional,
    SliceDemand,
    StaticQuotas,
    WeightedShares,
    WholeCell,
)
from slicesim.slicing.policies import largest_remainder


def demand(need: list[int], free: int = 51, retx: list[int] | None = None) -> SliceDemand:
    return SliceDemand(
        slot=0,
        n_prb=51,
        free=free,
        retx_prbs=np.array(retx or [0] * len(need), dtype=np.int64),
        need=np.array(need, dtype=np.int64),
    )


class TestPolicies:
    def test_largest_remainder(self) -> None:
        assert largest_remainder(np.array([40.8, 7.65, 2.55]), 51).tolist() == [41, 8, 2]
        assert largest_remainder(np.array([0.5, 0.5]), 1).tolist() == [1, 0]
        assert largest_remainder(np.array([2.0, 3.0]), 10).tolist() == [2, 3]

    def test_static_quotas_are_a_hard_partition(self) -> None:
        policy = StaticQuotas([0.8, 0.15, 0.05], n_prb=51)
        assert policy.owned.tolist() == [41, 8, 2]
        # A slice with nothing to send leaves its PRBs idle rather than lending them.
        assert policy.split(demand([100, 0, 0])).tolist() == [41, 8, 2]
        # Retransmissions come out of the slice's own share.
        assert policy.split(demand([100, 10, 1], free=46, retx=[0, 5, 0])).tolist() == [41, 3, 2]

    def test_static_quotas_must_not_exceed_the_carrier(self) -> None:
        with pytest.raises(ValueError, match="at most 1"):
            StaticQuotas([0.7, 0.4], n_prb=51)
        policy = StaticQuotas([0.5, 0.5], n_prb=10)
        with pytest.raises(RuntimeError, match="exceed"):
            policy.split(demand([5, 5], free=4, retx=[6, 0]))

    def test_delay_aware_serves_the_tightest_budget_first(self) -> None:
        policy = DelayAware([float("inf"), 0.005, 1.0])
        assert policy.split(demand([60, 30, 10])).tolist() == [11, 30, 10]
        assert policy.split(demand([60, 60, 10])).tolist() == [0, 51, 0]
        assert policy.split(demand([5, 2, 1])).tolist() == [5, 2, 1]

    def test_proportional_to_demand(self) -> None:
        policy = Proportional()
        assert policy.split(demand([10, 5, 1])).tolist() == [10, 5, 1]
        split = policy.split(demand([300, 60, 0]))
        assert split.tolist() == [43, 8, 0]
        assert split.sum() == 51

    def test_whole_cell(self) -> None:
        assert WholeCell().split(demand([7], free=40)).tolist() == [40]

    @pytest.mark.parametrize("seed", range(10))
    def test_work_conserving_policies_never_overspend(self, seed: int) -> None:
        gen = np.random.default_rng(seed)
        for policy in (DelayAware([0.005, 1.0, float("inf")]), Proportional()):
            for _ in range(50):
                need = gen.integers(0, 60, 3)
                free = int(gen.integers(0, 52))
                split = policy.split(demand(need.tolist(), free=free))
                assert np.all(split >= 0)
                assert np.all(split <= need)
                assert split.sum() == min(free, need.sum())


class TestWeightedShares:
    def test_splits_by_weight(self) -> None:
        policy = WeightedShares([0.6, 0.3, 0.1])
        assert policy.split(demand([100, 100, 100])).tolist() == [31, 15, 5]

    def test_passes_on_what_a_slice_cannot_use(self) -> None:
        policy = WeightedShares([0.6, 0.3, 0.1])
        # eMBB needs only 5: the other 46 are split 3:1 between the rest.
        assert policy.split(demand([5, 100, 100])).tolist() == [5, 35, 11]
        assert policy.split(demand([0, 0, 3])).tolist() == [0, 0, 3]
        assert policy.split(demand([1, 1, 1])).tolist() == [1, 1, 1]

    def test_zero_weights_still_get_spare_prbs(self) -> None:
        policy = WeightedShares([1.0, 0.0, 0.0])
        split = policy.split(demand([0, 20, 10]))
        assert split.sum() == 30
        assert split.tolist() == [0, 20, 10]

    def test_weights_can_change_between_slots(self) -> None:
        policy = WeightedShares([1.0, 1.0])
        assert policy.split(demand([60, 60])).sum() == 51
        policy.weights = np.array([3.0, 1.0])
        assert policy.weights.tolist() == [3.0, 1.0]
        assert policy.split(demand([60, 60])).tolist() == [38, 13]

    def test_rejects_bad_weights(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            WeightedShares([1.0, -0.5])
        with pytest.raises(ValueError, match="finite"):
            WeightedShares([np.inf, 1.0])
        with pytest.raises(ValueError, match="expected 3 weights, got 2"):
            WeightedShares([1.0, 1.0]).split(demand([1, 1, 1]))

    @pytest.mark.parametrize("seed", range(10))
    def test_never_overspends_and_wastes_nothing(self, seed: int) -> None:
        gen = np.random.default_rng(seed)
        for _ in range(50):
            policy = WeightedShares(gen.dirichlet(np.ones(3)))
            need = gen.integers(0, 60, 3)
            free = int(gen.integers(0, 52))
            split = policy.split(demand(need.tolist(), free=free))
            assert np.all(split >= 0)
            assert np.all(split <= need)
            assert split.sum() == min(free, need.sum())
