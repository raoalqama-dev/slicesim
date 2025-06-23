import json

import numpy as np
import pytest

from factories import scenario
from slicesim.engine import simulate
from slicesim.metrics import DelayStats, MetricsCollector


def test_delay_stats_in_milliseconds() -> None:
    stats = DelayStats.of(np.arange(1, 101, dtype=np.float64) * 1e-3)
    assert stats is not None
    assert stats.mean_ms == pytest.approx(50.5)
    assert stats.p50_ms == pytest.approx(50.5)
    assert stats.p95_ms == pytest.approx(95.05)
    assert stats.p99_ms == pytest.approx(99.01)
    assert DelayStats.of(np.empty(0)) is None


def test_collector_counts_arrivals_deliveries_and_prbs() -> None:
    m = MetricsCollector(n_ues=3, n_slots=2)
    m.record_arrivals(np.array([0, 2, 2]), np.array([10, 20, 30]))
    assert m.offered_bytes.tolist() == [10, 0, 50]
    assert m.packets_arrived.tolist() == [1, 0, 2]
    m.record_slot(1, 10, np.array([10, 0, 25]))
    assert m.prb_used.tolist() == [0, 10]
    assert m.delivered_bytes.tolist() == [10, 0, 25]
    m.record_delays(np.empty(0, dtype=np.intp), np.empty(0))
    ue, delay = m.delays()
    assert ue.size == delay.size == 0
    m.record_delays(np.array([0, 2]), np.array([0.001, 0.002]))
    m.record_delays(np.array([2]), np.array([0.003]))
    ue, delay = m.delays()
    assert ue.tolist() == [0, 2, 2]
    assert delay.tolist() == [0.001, 0.002, 0.003]
    assert m.packets_delivered.tolist() == [1, 0, 2]


def test_summary_per_group() -> None:
    r = simulate(scenario(duration_s=1.0))
    summary = r.summary()
    assert [g.name for g in summary.groups] == ["web", "ctrl", "video", "bulk"]
    web, ctrl, _, bulk = summary.groups
    assert web.ues == 4
    assert web.traffic == "poisson"
    # 4 UEs at 200 packets/s of 500 bytes: 3.2 Mbit/s offered.
    assert web.offered_mbps == pytest.approx(3.2, rel=0.1)
    assert web.delivered_mbps == pytest.approx(web.offered_mbps, rel=0.02)
    assert web.per_ue_mbps == pytest.approx(web.delivered_mbps / 4)
    assert web.packets_delivered <= web.packets_arrived
    assert web.delay is not None
    # 3 UEs, a packet of 80 bytes every 5 ms each, for 1 s.
    assert ctrl.packets_arrived == 600
    assert bulk.offered_mbps is None
    assert bulk.delay is None
    assert bulk.delivered_mbps > 10
    # The fixed link has no channel and never fails.
    assert (web.mean_sinr_db, web.mean_cqi, web.packets_lost) == (None, None, 0)

    cell = summary.cell
    assert cell.n_slots == 2000
    assert cell.duration_s == pytest.approx(1.0)
    assert cell.peak_mbps == pytest.approx(51 * 500 / 0.5e-3 / 1e6)
    assert (cell.bler, cell.residual_bler, cell.retransmissions) == (0.0, 0.0, 0)
    assert cell.delivered_mbps == pytest.approx(sum(g.delivered_mbps for g in summary.groups))
    assert cell.prb_utilization == pytest.approx(1.0)
    assert summary.carrier.startswith("20 MHz at 30 kHz")
    assert r.throughput_bps().sum() / 1e6 == pytest.approx(cell.delivered_mbps)


def test_summary_serializes_to_json() -> None:
    summary = simulate(scenario(duration_s=0.05)).summary()
    data = json.loads(json.dumps(summary.to_dict()))
    assert data["scenario"] == "test"
    assert data["seed"] == 7
    assert data["groups"][3]["offered_mbps"] is None
    assert set(data["cell"]) == {
        "duration_s",
        "n_slots",
        "peak_mbps",
        "delivered_mbps",
        "prb_utilization",
        "bler",
        "residual_bler",
        "retransmissions",
    }


def test_summary_of_a_run_over_the_radio_channel() -> None:
    r = simulate(scenario(duration_s=1.0, link={"model": "adaptive"}))
    summary = r.summary()
    cell = summary.cell
    # 51 PRBs at CQI 15: floor(168 * 5.5547 * 0.86) = 802 bits each per 0.5 ms slot.
    assert cell.peak_mbps == pytest.approx(51 * 802 / 0.5e-3 / 1e6)
    assert cell.bler is not None
    assert 0 < cell.bler < 0.15
    assert cell.retransmissions > 0
    for g in summary.groups:
        assert g.mean_sinr_db is not None
        assert g.mean_cqi is not None
        assert 0 < g.mean_cqi <= 15
    assert r.distance_m is not None
    assert r.mean_cqi is not None
    assert r.mean_cqi.shape == r.ue_group.shape


def test_ratios_of_an_idle_cell_are_undefined() -> None:
    s = scenario(
        duration_s=0.001,
        link={"model": "adaptive"},
        ues=[
            {
                "name": "rare",
                "count": 1,
                "traffic": {"model": "periodic", "period_ms": 1000, "packet_bytes": 10},
            }
        ],
    )
    cell = simulate(s).summary().cell
    assert (cell.bler, cell.residual_bler) == (None, None)
