import math

import numpy as np
import pytest

from slicesim.channel.pathloss import (
    UMA_BS_HEIGHT_M,
    breakpoint_distance,
    distance_3d,
    uma_los,
    uma_los_probability,
    uma_nlos,
)

# The spec's reference figures are at a 3D distance of 100 m.
D2D_FOR_100M_3D = math.sqrt(100**2 - (UMA_BS_HEIGHT_M - 1.5) ** 2)


def test_uma_los_at_100_m_and_3_5_ghz() -> None:
    # 28.0 + 22 log10(100) + 20 log10(3.5) = 82.88 dB
    assert float(uma_los(D2D_FOR_100M_3D, 3.5)) == pytest.approx(82.9, abs=0.05)


def test_uma_nlos_at_100_m_and_3_5_ghz() -> None:
    # 13.54 + 39.08 log10(100) + 20 log10(3.5) = 102.58 dB
    assert float(uma_nlos(D2D_FOR_100M_3D, 3.5)) == pytest.approx(102.6, abs=0.05)


def test_distance_3d_includes_the_antenna_heights() -> None:
    assert float(distance_3d(D2D_FOR_100M_3D, 25.0, 1.5)) == pytest.approx(100.0)
    assert distance_3d([0.0], 25.0, 1.5).tolist() == [23.5]


def test_breakpoint_distance() -> None:
    # 4 * (25 - 1) * (1.5 - 1) * 3.5e9 / 3e8 = 560 m
    assert breakpoint_distance(3.5, 25.0, 1.5) == pytest.approx(560.0)
    assert breakpoint_distance(28.0, 25.0, 1.5) == pytest.approx(4480.0)


def test_los_path_loss_is_continuous_at_the_breakpoint() -> None:
    before, after = uma_los([559.999, 560.001], 3.5)
    assert after == pytest.approx(before, abs=0.01)
    # Beyond the breakpoint the slope is 40 dB per decade instead of 22.
    near = np.diff(uma_los([100.0, 200.0], 3.5))[0]
    far = np.diff(uma_los([1000.0, 2000.0], 3.5))[0]
    assert near == pytest.approx(22 * math.log10(2), abs=0.3)
    assert far == pytest.approx(40 * math.log10(2), abs=0.1)


def test_path_loss_grows_with_distance_and_frequency() -> None:
    d = np.linspace(35, 3000, 200)
    for model in (uma_los, uma_nlos):
        assert np.all(np.diff(model(d, 3.5)) > 0)
        assert np.all(model(d, 28.0) > model(d, 3.5))


def test_nlos_is_never_better_than_los() -> None:
    d = np.linspace(10, 5000, 500)
    assert np.all(uma_nlos(d, 3.5) >= uma_los(d, 3.5))


def test_taller_ues_lose_less_in_nlos() -> None:
    assert float(uma_nlos(200.0, 3.5, h_ut=11.5)) < float(uma_nlos(200.0, 3.5, h_ut=1.5))


def test_los_probability() -> None:
    p = uma_los_probability([5.0, 18.0, 50.0, 100.0, 300.0, 1000.0])
    assert p[:2].tolist() == [1.0, 1.0]
    # 18/d + exp(-d/63)(1 - 18/d) at 100 m.
    assert p[3] == pytest.approx(0.18 + math.exp(-100 / 63) * 0.82)
    assert np.all(np.diff(p) <= 0)
    assert 0 < p[-1] < 0.05


def test_los_probability_for_tall_ues_includes_the_height_term() -> None:
    low = float(uma_los_probability(100.0, h_ut=13.0))
    high = float(uma_los_probability(100.0, h_ut=23.0))
    c_prime = 1.0
    expected = low * (1 + c_prime * 1.25 * 1.0 * math.exp(-100 / 150))
    assert high == pytest.approx(expected)


@pytest.mark.parametrize(("fc", "h_ut"), [(0.4, 1.5), (120.0, 1.5), (3.5, 1.0), (3.5, 14.0)])
def test_out_of_range_inputs_are_rejected(fc: float, h_ut: float) -> None:
    with pytest.raises(ValueError, match=r"UMa is defined|UE height"):
        uma_los(100.0, fc, h_ut=h_ut)
