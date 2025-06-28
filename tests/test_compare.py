from typing import Any

import numpy as np
import pytest

from factories import scenario
from slicesim.compare import compare_schedulers, with_scheduler
from slicesim.config import (
    ProportionalFairScheduler,
    RoundRobinScheduler,
    Scenario,
    parse_scenario,
)
from slicesim.engine import simulate
from slicesim.metrics import SimulationResult


def full_buffer(cell: dict[str, Any], seed: int, duration_s: float = 4.0) -> Scenario:
    return parse_scenario(
        {
            "name": "full-buffer",
            "seed": seed,
            "duration_s": duration_s,
            "carrier": {"bandwidth_mhz": 20, "scs_khz": 30},
            "cell": cell,
            "channel": {"shadowing": False, "fading": "rayleigh"},
            "ues": [{"name": "bulk", "count": 10, "traffic": {"model": "full_buffer"}}],
        }
    )


def throughput_and_fairness(s: Scenario) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for variant, summary in compare_schedulers(s, ["rr", "maxci", "pf"]):
        jain = summary.cell.jain
        assert jain is not None
        out[variant.scheduler.model] = (summary.cell.delivered_mbps, jain)
    return out


# Every UE 15 m from the gNB, where TR 38.901 gives line of sight with certainty, with
# power and interference set for a mid-range SINR: equal channels, independent fading.
EQUAL_CHANNELS = {
    "radius_m": 15,
    "min_distance_m": 15,
    "tx_power_dbm": 10,
    "interference_margin_db": 20,
}


def test_with_equal_channels() -> None:
    result = throughput_and_fairness(full_buffer(EQUAL_CHANNELS, seed=1))
    (rr, rr_jain), (pf, pf_jain), (maxci, maxci_jain) = (
        result["rr"],
        result["pf"],
        result["maxci"],
    )
    assert rr_jain > 0.99
    assert rr < pf < maxci
    # PF is as fair as equal resources here (both within 0.002 of perfect, the order
    # between them depending on the seed); max C/I is clearly less fair.
    assert pf_jain > 0.99
    assert maxci_jain < min(rr_jain, pf_jain) - 0.02


def test_in_a_realistic_cell() -> None:
    result = throughput_and_fairness(full_buffer({"radius_m": 500}, seed=3))
    (rr, rr_jain), (pf, pf_jain), (maxci, maxci_jain) = (
        result["rr"],
        result["pf"],
        result["maxci"],
    )
    assert rr < pf < maxci
    # Max C/I starves the far UEs. PF is fairer than equal resources here: scheduling
    # at fading peaks gains most where the rate grows fastest with SINR, at the edge.
    assert maxci_jain < 0.3
    assert maxci_jain < rr_jain < pf_jain


def test_compare_gives_every_scheduler_the_same_ues_and_traffic() -> None:
    s = scenario(duration_s=0.3, link={"model": "adaptive"})
    results: list[SimulationResult] = [
        simulate(with_scheduler(s, name)) for name in ("rr", "maxci", "pf")
    ]
    for r in results[1:]:
        assert np.array_equal(r.offered_bytes, results[0].offered_bytes)
        assert r.mean_sinr_db is not None
        assert results[0].mean_sinr_db is not None
        assert np.array_equal(r.mean_sinr_db, results[0].mean_sinr_db)


def test_with_scheduler_keeps_the_scenarios_own_settings() -> None:
    s = scenario(scheduler={"model": "pf", "time_constant_ms": 250})
    assert with_scheduler(s, "pf") is s
    assert with_scheduler(s, "rr").scheduler == RoundRobinScheduler()
    assert with_scheduler(scenario(), "pf").scheduler == ProportionalFairScheduler()
    with pytest.raises(ValueError, match="unknown scheduler 'edf'"):
        with_scheduler(s, "edf")
