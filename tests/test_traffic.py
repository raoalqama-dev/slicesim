import numpy as np
import pytest

from slicesim.config import FullBufferTraffic, OnOffTraffic, PeriodicTraffic, PoissonTraffic
from slicesim.traffic import (
    Arrivals,
    FullBuffer,
    OnOff,
    Periodic,
    Poisson,
    TrafficSource,
    build_source,
)


def collect(source: TrafficSource, step_s: float, steps: int) -> Arrivals:
    """Query a source over consecutive intervals and check each batch."""
    parts = []
    for k in range(steps):
        t0, t1 = k * step_s, (k + 1) * step_s
        a = source.arrivals(t0, t1)
        assert np.all((a.time >= t0) & (a.time < t1))
        assert np.all(np.diff(a.ue) >= 0), "arrivals must be grouped by UE"
        same_ue = np.diff(a.ue) == 0
        assert np.all(np.diff(a.time)[same_ue] >= 0), "arrivals must be in time order per UE"
        parts.append(a)
    return Arrivals(
        ue=np.concatenate([p.ue for p in parts]),
        time=np.concatenate([p.time for p in parts]),
        size=np.concatenate([p.size for p in parts]),
    )


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


class TestPoisson:
    def test_long_run_rate_is_within_tolerance(self) -> None:
        rate, n_ues, seconds = 1000.0, 10, 10.0
        a = collect(Poisson(n_ues, rng(), rate, 100), 1e-3, int(seconds * 1000))
        per_ue = np.bincount(a.ue, minlength=n_ues)
        expected = rate * seconds
        # 10,000 expected arrivals per UE: one standard deviation is 1%.
        assert np.all(np.abs(per_ue - expected) < 0.04 * expected)
        assert abs(len(a) - expected * n_ues) < 0.01 * expected * n_ues

    def test_arrivals_have_poisson_statistics(self) -> None:
        rate = 500.0
        a = collect(Poisson(1, rng(1), rate, 100), 0.01, 4000)
        gaps = np.diff(a.time)
        # Exponential gaps: mean 1/rate and a coefficient of variation of 1.
        assert gaps.mean() == pytest.approx(1 / rate, rel=0.02)
        assert gaps.std() / gaps.mean() == pytest.approx(1.0, abs=0.03)
        # Counts per 10 ms window: variance equal to the mean.
        counts = np.bincount((a.time / 0.01).astype(int))
        assert counts.var() / counts.mean() == pytest.approx(1.0, abs=0.08)

    def test_packets_have_the_configured_size(self) -> None:
        a = collect(Poisson(3, rng(), 2000.0, 321), 1e-3, 100)
        assert len(a) > 0
        assert np.all(a.size == 321)
        assert Poisson(3, rng(), 2000.0, 321).mean_rate_bps == 2000 * 321 * 8

    def test_same_seed_gives_the_same_arrivals(self) -> None:
        first = collect(Poisson(5, rng(9), 300.0, 100), 5e-4, 2000)
        second = collect(Poisson(5, rng(9), 300.0, 100), 5e-4, 2000)
        assert np.array_equal(first.time, second.time)
        assert np.array_equal(first.ue, second.ue)

    def test_invalid_parameters_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Poisson(1, rng(), 0.0, 100)
        with pytest.raises(ValueError, match="at least one UE"):
            Poisson(0, rng(), 1.0, 100)


class TestPeriodic:
    def test_one_packet_per_period_at_each_ues_phase(self) -> None:
        period, n_ues, steps = 0.005, 6, 2000
        source = Periodic(n_ues, rng(), period, 64)
        assert np.all((source.phase >= 0) & (source.phase < period))
        a = collect(source, 1e-3, steps)
        for ue in range(n_ues):
            times = a.time[a.ue == ue]
            k = np.arange(times.size)
            assert np.array_equal(times, source.phase[ue] + k * period)
            assert times.size == int((steps * 1e-3 - source.phase[ue]) // period) + 1

    def test_arrival_times_do_not_drift_over_long_runs(self) -> None:
        source = Periodic(1, rng(4), 0.001, 32)
        a = collect(source, 1.0, 100)
        assert len(a) == 100_000
        assert a.time[-1] == source.phase[0] + (len(a) - 1) * 0.001

    def test_arrivals_do_not_depend_on_slot_length(self) -> None:
        coarse = collect(Periodic(4, rng(2), 0.0013, 32), 1e-3, 1000)
        fine = collect(Periodic(4, rng(2), 0.0013, 32), 2.5e-4, 4000)
        order_c = np.lexsort((coarse.time, coarse.ue))
        order_f = np.lexsort((fine.time, fine.ue))
        assert np.array_equal(coarse.time[order_c], fine.time[order_f])

    def test_periods_shorter_than_a_slot_give_several_packets(self) -> None:
        source = Periodic(2, rng(), 0.0002, 10)
        a = collect(source, 1e-3, 10)
        assert np.bincount(a.ue).tolist() == [50, 50]
        assert source.mean_rate_bps == pytest.approx(10 * 8 / 0.0002)

    def test_invalid_parameters_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Periodic(1, rng(), 0.001, 0)


class TestOnOff:
    def make(self, n_ues: int = 100, seed: int = 0) -> OnOff:
        return OnOff(
            n_ues, rng(seed), mean_on_s=0.05, mean_off_s=0.15, frame_rate_fps=100, frame_bytes=1000
        )

    def test_long_run_rate_matches_the_on_probability(self) -> None:
        source = self.make()
        assert source.on_probability == pytest.approx(0.25)
        seconds = 100.0
        a = collect(source, 0.01, int(seconds / 0.01))
        measured_bps = a.size.sum() * 8 / seconds / source.n_ues
        assert source.mean_rate_bps == pytest.approx(1000 * 8 * 100 * 0.25)
        assert measured_bps == pytest.approx(source.mean_rate_bps, rel=0.03)

    def test_frames_are_evenly_spaced_until_the_on_period_ends(self) -> None:
        source = OnOff(1, rng(), mean_on_s=1.0, mean_off_s=1e6, frame_rate_fps=100, frame_bytes=1)
        source.on[:] = True
        source.state_end[:] = 0.1
        source.next_frame[:] = 0.002
        a = collect(source, 1e-3, 300)
        assert np.allclose(a.time, 0.002 + 0.01 * np.arange(10))
        assert not source.on[0]
        assert source.state_end[0] > 0.1

    def test_the_first_frame_comes_within_one_interval_of_the_start(self) -> None:
        for seed in range(20):
            source = OnOff(
                1, rng(seed), mean_on_s=1e6, mean_off_s=1.0, frame_rate_fps=50, frame_bytes=1
            )
            source.on[:] = False
            source.state_end[:] = 0.0105
            a = collect(source, 1e-3, 80)
            assert 0.0105 <= a.time[0] < 0.0305
            assert np.allclose(np.diff(a.time), 0.02)
            assert source.on[0]

    def test_short_periods_can_switch_several_times_in_one_interval(self) -> None:
        source = OnOff(
            50, rng(8), mean_on_s=0.002, mean_off_s=0.002, frame_rate_fps=2000, frame_bytes=10
        )
        a = collect(source, 0.05, 400)
        # Half the time on at 2,000 frames per second, over 20 seconds.
        assert len(a) / 50 == pytest.approx(20_000, rel=0.05)

    def test_initial_state_is_stationary(self) -> None:
        source = self.make(n_ues=20_000, seed=5)
        assert source.on.mean() == pytest.approx(0.25, abs=0.01)
        assert np.all(source.state_end > 0)

    def test_frames_have_the_configured_size(self) -> None:
        a = collect(self.make(n_ues=10), 1e-3, 1000)
        assert len(a) > 0
        assert np.all(a.size == 1000)

    def test_quiet_intervals_return_no_arrivals(self) -> None:
        source = OnOff(1, rng(), mean_on_s=0.001, mean_off_s=1000, frame_rate_fps=1, frame_bytes=1)
        source.on[:] = False
        source.state_end[:] = 1000.0
        assert len(source.arrivals(0.0, 0.001)) == 0

    def test_invalid_parameters_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            OnOff(1, rng(), 0.0, 1.0, 30, 100)


def test_full_buffer_never_queues_packets() -> None:
    source = FullBuffer(3, rng())
    assert source.full_buffer
    assert len(source.arrivals(0.0, 1.0)) == 0
    assert source.mean_rate_bps == float("inf")


def test_empty_arrivals_cannot_be_modified() -> None:
    empty = Arrivals.empty()
    with pytest.raises(ValueError, match="read-only"):
        empty.time[...] = 1.0


def test_build_source_converts_milliseconds_to_seconds() -> None:
    periodic = build_source(
        PeriodicTraffic(model="periodic", period_ms=5, packet_bytes=40), 2, rng()
    )
    assert isinstance(periodic, Periodic)
    assert periodic.period_s == pytest.approx(0.005)
    on_off = build_source(
        OnOffTraffic(
            model="on_off", mean_on_ms=100, mean_off_ms=300, frame_rate_fps=25, frame_bytes=900
        ),
        2,
        rng(),
    )
    assert isinstance(on_off, OnOff)
    assert (on_off.mean_on_s, on_off.mean_off_s) == pytest.approx((0.1, 0.3))
    assert on_off.frame_interval_s == pytest.approx(0.04)
    poisson = build_source(PoissonTraffic(model="poisson", rate_pps=5, packet_bytes=9), 2, rng())
    assert isinstance(poisson, Poisson)
    assert isinstance(build_source(FullBufferTraffic(model="full_buffer"), 2, rng()), FullBuffer)
