import math

import numpy as np
import pytest

from slicesim.channel.link import (
    BLER_TARGET,
    CQI_SINR_THRESHOLDS_DB,
    CQI_TABLE,
    bits_per_prb,
    bler,
    combine_db,
    cqi_from_sinr,
    efficiency,
)


def test_cqi_table_matches_ts_38_214() -> None:
    assert len(CQI_TABLE) == 15
    assert CQI_TABLE[0] == ("QPSK", 78, 0.1523)
    assert CQI_TABLE[6] == ("16QAM", 378, 1.4766)
    assert CQI_TABLE[9] == ("64QAM", 466, 2.7305)
    assert CQI_TABLE[14] == ("64QAM", 948, 5.5547)
    bits = {"QPSK": 2, "16QAM": 4, "64QAM": 6}
    for entry in CQI_TABLE:
        # Efficiency is bits per symbol times the code rate.
        expected = bits[entry.modulation] * entry.code_rate_x1024 / 1024
        assert entry.efficiency == pytest.approx(expected, abs=5e-4)


def test_cqi_rises_with_sinr() -> None:
    sinr = np.linspace(-15, 35, 1001)
    cqi = cqi_from_sinr(sinr)
    assert np.all(np.diff(cqi) >= 0)
    assert cqi[0] == 0
    assert cqi[-1] == 15
    assert set(cqi.tolist()) == set(range(16))


def test_cqi_thresholds() -> None:
    for c, threshold in enumerate(CQI_SINR_THRESHOLDS_DB, start=1):
        assert cqi_from_sinr(threshold) == c
        assert cqi_from_sinr(threshold - 1e-6) == c - 1
    assert np.all(np.diff(CQI_SINR_THRESHOLDS_DB) > 0)


def test_bits_per_prb() -> None:
    # 12 subcarriers x 14 symbols x 5.5547 x (1 - 0.14) = 802.5 bits at CQI 15.
    assert bits_per_prb(15).tolist() == 802
    assert bits_per_prb(1).tolist() == math.floor(168 * 0.1523 * 0.86)
    assert bits_per_prb(0).tolist() == 0
    assert bits_per_prb(15, overhead=0.0).tolist() == math.floor(168 * 5.5547)
    assert np.all(np.diff(bits_per_prb(np.arange(16))) > 0)
    assert efficiency([0, 15]).tolist() == [0.0, 5.5547]
    with pytest.raises(ValueError, match="overhead"):
        bits_per_prb(5, overhead=1.0)


def test_bler_is_the_target_at_the_threshold() -> None:
    for c, threshold in enumerate(CQI_SINR_THRESHOLDS_DB, start=1):
        assert float(bler(threshold, c)) == pytest.approx(BLER_TARGET)


def test_bler_falls_with_margin() -> None:
    margins = np.linspace(-10, 10, 201)
    curve = bler(10.3 + margins, 9)
    assert np.all(np.diff(curve) < 0)
    assert curve[0] > 0.999
    assert curve[-1] < 1e-6
    # About 3.4 dB from 90% to 1%.
    low = float(np.interp(0.9, curve[::-1], margins[::-1]))
    high = float(np.interp(0.01, curve[::-1], margins[::-1]))
    assert high - low == pytest.approx(3.4, abs=0.1)


def test_cqi_zero_always_fails() -> None:
    assert float(bler(40.0, 0)) == pytest.approx(1.0)


def test_chase_combining_adds_linear_sinr() -> None:
    assert combine_db([10.0, 10.0]) == pytest.approx(10 + 10 * math.log10(2))
    assert combine_db([0.0, 0.0, 0.0, 0.0]) == pytest.approx(10 * math.log10(4))
    assert combine_db([5.0]) == pytest.approx(5.0)
