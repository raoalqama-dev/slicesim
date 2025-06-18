import pytest

from slicesim import __version__
from slicesim.carrier import NRB_TABLE, Carrier, supported_bandwidths


@pytest.mark.parametrize(
    ("bandwidth", "scs", "n_prb"),
    [
        (5, 15, 25),
        (20, 15, 106),
        (50, 15, 270),
        (20, 30, 51),
        (40, 30, 106),
        (100, 30, 273),
        (10, 60, 11),
        (100, 60, 135),
    ],
)
def test_prb_count_follows_ts_38_101_1(bandwidth: int, scs: int, n_prb: int) -> None:
    assert Carrier.create(bandwidth, scs).n_prb == n_prb


@pytest.mark.parametrize(
    ("scs", "numerology", "slot_ms", "slots_per_subframe"),
    [(15, 0, 1.0, 1), (30, 1, 0.5, 2), (60, 2, 0.25, 4)],
)
def test_numerology_sets_slot_duration(
    scs: int, numerology: int, slot_ms: float, slots_per_subframe: int
) -> None:
    carrier = Carrier.create(20, scs)
    assert carrier.numerology == numerology
    assert carrier.slot_duration_s == pytest.approx(slot_ms * 1e-3)
    assert carrier.slots_per_subframe == slots_per_subframe
    assert carrier.prb_bandwidth_hz == pytest.approx(12 * scs * 1e3)
    assert carrier.resource_elements_per_prb == 168


def test_occupied_bandwidth_fits_the_channel() -> None:
    for scs, row in NRB_TABLE.items():
        for bandwidth, n_prb in row.items():
            occupied_mhz = n_prb * 12 * scs / 1e3
            assert occupied_mhz < bandwidth, (bandwidth, scs)
            # Guard bands never take more than a quarter of the channel.
            assert occupied_mhz > 0.75 * bandwidth, (bandwidth, scs)


def test_undefined_bandwidth_is_rejected_with_the_valid_choices() -> None:
    with pytest.raises(ValueError, match=r"100 MHz channel is not defined at 15 kHz") as info:
        Carrier.create(100, 15)
    assert "5, 10, 15, 20, 25, 30, 35, 40, 45, 50 MHz" in str(info.value)
    with pytest.raises(ValueError, match="not defined at 60 kHz"):
        Carrier.create(5, 60)


def test_unsupported_subcarrier_spacing_is_rejected() -> None:
    with pytest.raises(
        ValueError, match=r"120 kHz is not supported in FR1; choose one of 15, 30, 60"
    ):
        Carrier.create(100, 120)
    with pytest.raises(ValueError, match="not supported"):
        supported_bandwidths(45)


def test_supported_bandwidths() -> None:
    assert supported_bandwidths(30)[-1] == 100
    assert 5 not in supported_bandwidths(60)


def test_description() -> None:
    assert str(Carrier.create(20, 30)) == "20 MHz at 30 kHz (numerology 1): 51 PRBs, 0.5 ms slots"


def test_version_is_set() -> None:
    assert __version__.startswith("0.")
