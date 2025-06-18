"""Small scenarios for tests, built from a base that each test adjusts."""

from __future__ import annotations

from typing import Any

from slicesim.config import Scenario, parse_scenario


def poisson_group(name: str = "web", count: int = 4, rate_pps: float = 200.0) -> dict[str, Any]:
    return {
        "name": name,
        "count": count,
        "traffic": {"model": "poisson", "rate_pps": rate_pps, "packet_bytes": 500},
    }


def scenario_data(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "test",
        "seed": 7,
        "duration_s": 0.5,
        "carrier": {"bandwidth_mhz": 20, "scs_khz": 30},
        "link": {"model": "fixed", "bits_per_prb": 500},
        "ues": [
            poisson_group(),
            {
                "name": "ctrl",
                "count": 3,
                "traffic": {"model": "periodic", "period_ms": 5, "packet_bytes": 80},
            },
            {
                "name": "video",
                "count": 2,
                "traffic": {
                    "model": "on_off",
                    "mean_on_ms": 200,
                    "mean_off_ms": 100,
                    "frame_rate_fps": 30,
                    "frame_bytes": 6000,
                },
            },
            {"name": "bulk", "count": 1, "traffic": {"model": "full_buffer"}},
        ],
    }
    data.update(changes)
    return data


def scenario(**changes: Any) -> Scenario:
    return parse_scenario(scenario_data(**changes))
