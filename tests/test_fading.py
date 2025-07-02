import numpy as np
import pytest

from slicesim.channel.fading import BlockFading, NoFading


def run(fading: BlockFading | NoFading, slots: int) -> np.ndarray:
    return np.array([fading.gain_db(t) for t in range(slots)])


def test_gain_is_exponential_with_unit_mean() -> None:
    gains = 10 ** (run(BlockFading(200, 1, np.random.default_rng(0)), 500) / 10)
    assert gains.mean() == pytest.approx(1.0, abs=0.01)
    # Exponential: the variance equals the squared mean, and P(g < 0.1) = 1 - e^-0.1.
    assert gains.var() == pytest.approx(1.0, abs=0.03)
    assert (gains < 0.1).mean() == pytest.approx(1 - np.exp(-0.1), abs=0.005)


def test_gain_holds_for_the_coherence_time() -> None:
    fading = BlockFading(50, 8, np.random.default_rng(1))
    offset = fading.offset.copy()
    gains = run(fading, 64)
    changes = np.diff(gains, axis=0) != 0
    for ue in range(50):
        changed_at = np.flatnonzero(changes[:, ue]) + 1
        # Each UE redraws exactly at its own block boundaries.
        assert np.all((changed_at + offset[ue]) % 8 == 0)
        assert len(changed_at) >= 7


def test_ues_do_not_all_change_together() -> None:
    fading = BlockFading(100, 20, np.random.default_rng(2))
    assert len(np.unique(fading.offset)) > 10


def test_blocks_are_independent() -> None:
    gains = run(BlockFading(1, 1, np.random.default_rng(3)), 20_000)[:, 0]
    assert abs(np.corrcoef(gains[:-1], gains[1:])[0, 1]) < 0.03


def test_slots_must_be_consecutive() -> None:
    fading = BlockFading(1, 4, np.random.default_rng(4))
    fading.gain_db(0)
    with pytest.raises(ValueError, match="expected slot 1, got 2"):
        fading.gain_db(2)
    with pytest.raises(ValueError, match="at least 1"):
        BlockFading(1, 0, np.random.default_rng(4))


def test_no_fading_is_flat() -> None:
    gains = run(NoFading(3), 5)
    assert gains.shape == (5, 3)
    assert np.all(gains == 0.0)


def test_diversity_keeps_the_mean_and_shrinks_deep_fades() -> None:
    def gains(diversity: int) -> np.ndarray:
        fading = BlockFading(400, 1, np.random.default_rng(5), diversity)
        return 10 ** (run(fading, 250) / 10)

    one, two, four = gains(1), gains(2), gains(4)
    for g in (one, two, four):
        assert g.mean() == pytest.approx(1.0, abs=0.01)
    # Below -10 dB: 9.5% of the time with one branch, 1.8% with two, 0.08% with four.
    assert (one < 0.1).mean() == pytest.approx(0.0952, abs=0.003)
    assert (two < 0.1).mean() == pytest.approx(0.0175, abs=0.002)
    assert (four < 0.1).mean() == pytest.approx(0.00078, abs=0.0003)
    # Gamma(N, 1/N): the variance falls as 1/N.
    assert two.var() == pytest.approx(0.5, abs=0.02)
    assert four.var() == pytest.approx(0.25, abs=0.01)


def test_one_branch_is_plain_rayleigh() -> None:
    a = run(BlockFading(10, 3, np.random.default_rng(6)), 20)
    b = run(BlockFading(10, 3, np.random.default_rng(6), diversity=1), 20)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="diversity"):
        BlockFading(1, 1, np.random.default_rng(6), diversity=0)
