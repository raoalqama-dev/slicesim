import math
from itertools import pairwise

import numpy as np
import pytest

from factories import poisson_group, scenario
from slicesim.carrier import Carrier
from slicesim.channel.cell import GroupStreams, RadioChannel, drop_ues, noise_per_prb_dbm
from slicesim.channel.pathloss import uma_los_probability
from slicesim.config import CellConfig, ChannelConfig
from slicesim.engine import RngStreams, simulate

CARRIER = Carrier.create(20, 30)


def streams(seed: int = 0, name: str = "g") -> GroupStreams:
    s = RngStreams(seed)
    return GroupStreams(
        drop=s.generator("ue_drop", name),
        los=s.generator("los", name),
        shadowing=s.generator("shadowing", name),
        fading=s.generator("fading", name),
    )


def channel(
    count: int,
    cell: CellConfig | None = None,
    channel: ChannelConfig | None = None,
    seed: int = 0,
) -> RadioChannel:
    return RadioChannel(
        CARRIER, cell or CellConfig(), channel or ChannelConfig(), [(count, streams(seed))]
    )


def test_ues_are_uniform_over_the_annulus() -> None:
    x, y = drop_ues(100_000, 300.0, 35.0, np.random.default_rng(0))
    r = np.hypot(x, y)
    assert r.min() >= 35.0
    assert r.max() <= 300.0
    # Uniform in area: the share inside radius a is (a^2 - 35^2) / (300^2 - 35^2).
    inner = (150**2 - 35**2) / (300**2 - 35**2)
    assert (r < 150).mean() == pytest.approx(inner, abs=0.005)
    angle = np.arctan2(y, x)
    assert (angle > 0).mean() == pytest.approx(0.5, abs=0.005)


def test_noise_per_prb() -> None:
    # -174 dBm/Hz over 360 kHz, plus a 7 dB noise figure.
    assert noise_per_prb_dbm(CARRIER, 7.0) == pytest.approx(-174 + 10 * math.log10(360e3) + 7)


def test_link_budget_without_shadowing_or_fading() -> None:
    cell = CellConfig(
        tx_power_dbm=46, antenna_gain_db=5, noise_figure_db=7, interference_margin_db=3
    )
    ch = channel(20, cell, ChannelConfig(shadowing=False, fading="none"))
    expected = (
        46 - 10 * math.log10(51) + 5 - ch.pathloss_db - (-174 + 10 * math.log10(360e3) + 7 + 3)
    )
    assert np.allclose(ch.mean_sinr_db, expected)
    assert np.allclose(ch.sinr_db(0), expected)
    assert np.all(ch.shadowing_db == 0)


def test_los_share_follows_the_los_probability() -> None:
    ch = channel(20_000)
    expected = uma_los_probability(ch.distance_m).mean()
    assert ch.los.mean() == pytest.approx(expected, abs=0.01)
    near, far = ch.distance_m < 80, ch.distance_m > 250
    assert ch.los[near].mean() > ch.los[far].mean()


def test_shadowing_has_the_uma_standard_deviations() -> None:
    ch = channel(40_000)
    assert ch.shadowing_db[ch.los].std() == pytest.approx(4.0, rel=0.03)
    assert ch.shadowing_db[~ch.los].std() == pytest.approx(6.0, rel=0.03)
    assert abs(ch.shadowing_db.mean()) < 0.1


def test_fading_varies_the_sinr_around_its_mean() -> None:
    ch = channel(10, channel=ChannelConfig(fading="rayleigh", coherence_ms=1.0))
    assert ch.coherence_slots == 2
    sinr = np.array([ch.sinr_db(t) for t in range(400)])
    gain = 10 ** ((sinr - ch.mean_sinr_db) / 10)
    assert gain.mean() == pytest.approx(1.0, abs=0.15)
    assert len(np.unique(sinr[:, 0])) > 100


def test_coherence_is_at_least_one_slot() -> None:
    ch = channel(1, channel=ChannelConfig(coherence_ms=0.01))
    assert ch.coherence_slots == 1


def test_3d_distance() -> None:
    ch = channel(5)
    assert np.allclose(ch.distance_3d_m, np.hypot(ch.distance_m, 23.5))


def test_each_group_keeps_its_own_channel() -> None:
    a = RadioChannel(CARRIER, CellConfig(), ChannelConfig(), [(5, streams(1, "a"))])
    ab = RadioChannel(
        CARRIER, CellConfig(), ChannelConfig(), [(3, streams(1, "b")), (5, streams(1, "a"))]
    )
    assert np.array_equal(a.mean_sinr_db, ab.mean_sinr_db[3:])
    assert np.array_equal(a.sinr_db(0), ab.sinr_db(0)[3:])


def test_capacity_falls_with_distance_on_average() -> None:
    s = scenario(
        duration_s=0.5,
        cell={"radius_m": 1000},
        link={"model": "adaptive"},
        ues=[{"name": "bulk", "count": 300, "traffic": {"model": "full_buffer"}}],
    )
    r = simulate(s)
    assert r.distance_m is not None
    assert r.mean_cqi is not None
    edges = [35, 200, 400, 600, 800, 1000]
    bins = np.digitize(r.distance_m, edges[1:-1])
    mean_cqi = [r.mean_cqi[bins == b].mean() for b in range(len(edges) - 1)]
    assert all(a > b for a, b in pairwise(mean_cqi)), mean_cqi


def test_changing_traffic_leaves_the_channel_alone() -> None:
    base = scenario(link={"model": "adaptive"}, duration_s=0.05)
    data = base.model_dump()
    data["ues"][0]["traffic"] = poisson_group(rate_pps=900)["traffic"]
    changed = base.model_validate(data)
    a, b = simulate(base), simulate(changed)
    assert a.mean_sinr_db is not None
    assert b.mean_sinr_db is not None
    assert np.array_equal(a.mean_sinr_db, b.mean_sinr_db)
    assert a.distance_m is not None
    assert b.distance_m is not None
    assert np.array_equal(a.distance_m, b.distance_m)
