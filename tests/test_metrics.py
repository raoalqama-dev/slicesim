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
    m.record_slot(1, np.array([4, 0, 6]), np.array([10, 0, 25]))
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

    cell = summary.cell
    assert cell.n_slots == 2000
    assert cell.duration_s == pytest.approx(1.0)
    assert cell.capacity_mbps == pytest.approx(51 * 500 / 0.5e-3 / 1e6)
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
        "capacity_mbps",
        "delivered_mbps",
        "prb_utilization",
    }
