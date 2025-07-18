"""An interactive slicesim dashboard: build a sliced scenario and run it.

Run with ``streamlit run dashboard/app.py`` after ``pip install -e ".[dashboard]"``.
The sidebar builds a three-slice scenario like ``scenarios/medium.yaml``; the
page shows its YAML, and after a run the cell's totals, each slice against its
SLA, PRB shares and throughput over time, and the URLLC delay distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import streamlit as st
import yaml

from slicesim.carrier import NRB_TABLE
from slicesim.config import Scenario, ScenarioError, parse_scenario
from slicesim.engine import simulate
from slicesim.metrics import SimulationResult

POLICIES = {
    "delay_aware": "delay-aware",
    "proportional": "proportional to demand",
    "static": "static quotas",
}


@dataclass(frozen=True)
class Form:
    """Everything the sidebar sets."""

    bandwidth_mhz: int = 20
    scs_khz: int = 30
    radius_m: float = 300.0
    diversity: int = 2
    video_ues: int = 16
    urllc_ues: int = 10
    urllc_period_ms: float = 5.0
    urllc_packet_bytes: int = 200
    sensors: int = 100
    policy: str = "delay_aware"
    quotas: tuple[float, float, float] = (0.80, 0.15, 0.05)
    duration_s: float = 2.0
    seed: int = 1


def scenario_data(form: Form) -> dict[str, Any]:
    """The scenario the form describes, as it would appear in a YAML file."""
    embb_quota, urllc_quota, mmtc_quota = form.quotas
    return {
        "name": "dashboard",
        "description": "built in the dashboard",
        "seed": form.seed,
        "duration_s": form.duration_s,
        "carrier": {"bandwidth_mhz": form.bandwidth_mhz, "scs_khz": form.scs_khz},
        "cell": {"radius_m": form.radius_m},
        "channel": {"coherence_ms": 40, "diversity": form.diversity},
        "slicing": {"policy": form.policy, "window_ms": 100},
        "slices": [
            {
                "name": "embb",
                "sla": {"type": "embb", "min_throughput_mbps": 2, "availability": 0.95},
                "scheduler": {"model": "pf", "time_constant_ms": 100},
                "quota": embb_quota,
            },
            {
                "name": "urllc",
                "sla": {"type": "urllc", "delay_budget_ms": 5, "reliability": 0.999},
                "scheduler": {"model": "rr"},
                "quota": urllc_quota,
            },
            {
                "name": "mmtc",
                "sla": {"type": "mmtc", "delay_budget_ms": 1000, "reliability": 0.99},
                "scheduler": {"model": "rr"},
                "quota": mmtc_quota,
            },
        ],
        "ues": [
            {
                "name": "video",
                "slice": "embb",
                "count": form.video_ues,
                "traffic": {
                    "model": "on_off",
                    "mean_on_ms": 2000,
                    "mean_off_ms": 1000,
                    "frame_rate_fps": 30,
                    "frame_bytes": 12500,
                },
            },
            {
                "name": "control",
                "slice": "urllc",
                "count": form.urllc_ues,
                "traffic": {
                    "model": "periodic",
                    "period_ms": form.urllc_period_ms,
                    "packet_bytes": form.urllc_packet_bytes,
                },
            },
            {
                "name": "sensors",
                "slice": "mmtc",
                "count": form.sensors,
                "traffic": {"model": "poisson", "rate_pps": 1, "packet_bytes": 100},
            },
        ],
    }


Series = dict[str, list[float]]


def slice_series(result: SimulationResult) -> tuple[list[float], Series, Series]:
    """Per SLA window: its start in seconds, each slice's share of the PRBs, and each
    slice's throughput in Mbit/s."""
    window = result.window_slots
    windows = result.n_slots // window
    slot_s = result.carrier.slot_duration_s
    names = [s.name for s in result.scenario.slices]
    times = [w * window * slot_s for w in range(windows)]
    shares: Series = {}
    throughput: Series = {}
    for index, name in enumerate(names):
        prbs = result.slice_prbs[: windows * window, index].reshape(windows, window)
        shares[name] = (prbs.sum(axis=1) / (result.carrier.n_prb * window)).tolist()
        delivered = result.window_delivered[:windows, result.ue_slice == index].sum(axis=1)
        throughput[name] = (delivered * 8 / (window * slot_s) / 1e6).tolist()
    return times, shares, throughput


def delay_cdf(result: SimulationResult, slice_name: str) -> tuple[list[float], list[float]]:
    """The empirical CDF of one slice's packet delay: delays in ms and shares."""
    index = [s.name for s in result.scenario.slices].index(slice_name)
    delays = np.sort(result.delay_s[result.ue_slice[result.delay_ue] == index]) * 1e3
    if delays.size == 0:
        return [], []
    quantiles = np.linspace(0, 1, min(delays.size, 400))
    return np.quantile(delays, quantiles).tolist(), quantiles.tolist()


def sidebar() -> Form:
    st.sidebar.header("Cell")
    scs = st.sidebar.selectbox("subcarrier spacing (kHz)", sorted(NRB_TABLE), index=1)
    bandwidths = sorted(NRB_TABLE[scs])
    bandwidth = st.sidebar.selectbox(
        "bandwidth (MHz)", bandwidths, index=bandwidths.index(20) if 20 in bandwidths else 0
    )
    radius = st.sidebar.slider("cell radius (m)", 100, 1000, 300, step=50)
    diversity = st.sidebar.selectbox("receive antennas", [1, 2, 4], index=1)

    st.sidebar.header("Traffic")
    video = st.sidebar.slider("eMBB video UEs", 0, 40, 16)
    urllc = st.sidebar.slider("URLLC control UEs", 1, 40, 10)
    period = st.sidebar.slider("URLLC packet period (ms)", 1, 50, 5)
    size = st.sidebar.slider("URLLC packet size (bytes)", 32, 1500, 200, step=8)
    sensors = st.sidebar.slider("mMTC sensors", 1, 500, 100)

    st.sidebar.header("Slicing")
    policy = st.sidebar.selectbox("policy", list(POLICIES), format_func=lambda key: POLICIES[key])
    quotas = (0.80, 0.15, 0.05)
    if policy == "static":
        embb = st.sidebar.slider("eMBB quota", 0.0, 1.0, 0.80, step=0.05)
        urllc_quota = st.sidebar.slider("URLLC quota", 0.0, 1.0, 0.15, step=0.05)
        mmtc = st.sidebar.slider("mMTC quota", 0.0, 1.0, 0.05, step=0.05)
        quotas = (embb, urllc_quota, mmtc)

    st.sidebar.header("Run")
    duration = st.sidebar.slider("duration (s)", 0.5, 10.0, 2.0, step=0.5)
    seed = int(st.sidebar.number_input("seed", min_value=0, value=1, step=1))
    return Form(
        bandwidth_mhz=int(bandwidth),
        scs_khz=int(scs),
        radius_m=float(radius),
        diversity=int(diversity),
        video_ues=int(video),
        urllc_ues=int(urllc),
        urllc_period_ms=float(period),
        urllc_packet_bytes=int(size),
        sensors=int(sensors),
        policy=str(policy),
        quotas=quotas,
        duration_s=float(duration),
        seed=seed,
    )


def show_result(result: SimulationResult) -> None:
    summary = result.summary()
    cell = summary.cell
    columns = st.columns(4)
    columns[0].metric("delivered", f"{cell.delivered_mbps:.1f} Mbit/s")
    columns[1].metric("PRBs used", f"{cell.prb_utilization:.0%}")
    columns[2].metric("BLER", "-" if cell.bler is None else f"{cell.bler:.1%}")
    columns[3].metric(
        "SLAs met", f"{sum(bool(s.met) for s in summary.slices)} of {len(summary.slices)}"
    )

    st.subheader("Slices")
    st.dataframe(
        [
            {
                "slice": s.name,
                "UEs": s.ues,
                "PRB share": f"{s.prb_share:.1%}",
                "Mbit/s": round(s.delivered_mbps, 2),
                "p99 delay (ms)": None if s.delay is None else round(s.delay.p99_ms, 2),
                "violations": None if s.violation_ratio is None else f"{s.violation_ratio:.3%}",
                "SLA met": "-" if s.met is None else ("yes" if s.met else "no"),
                "target": s.target,
            }
            for s in summary.slices
        ],
        hide_index=True,
    )

    times, shares, throughput = slice_series(result)
    left, right = st.columns(2)
    left.markdown("**Share of PRBs per 100 ms window**")
    left.line_chart({"time (s)": times, **shares}, x="time (s)")
    right.markdown("**Throughput per slice (Mbit/s)**")
    right.line_chart({"time (s)": times, **throughput}, x="time (s)")

    delays, quantiles = delay_cdf(result, "urllc")
    if delays:
        st.markdown("**URLLC packet delay: share delivered within (5 ms budget)**")
        st.line_chart({"delay (ms)": delays, "share": quantiles}, x="delay (ms)", y="share")


def main() -> None:
    st.set_page_config(page_title="slicesim", layout="wide")
    st.title("slicesim")
    st.caption(
        "A 5G NR cell shared by eMBB, URLLC and mMTC slices. Build a scenario in the "
        "sidebar and run it."
    )
    form = sidebar()
    data = scenario_data(form)
    with st.expander("Scenario YAML"):
        text = yaml.safe_dump(data, sort_keys=False)
        st.code(text, language="yaml")
        st.download_button("Download scenario.yaml", text, file_name="scenario.yaml")
    if st.button("Run", type="primary"):
        try:
            scenario: Scenario = parse_scenario(data, "dashboard")
        except ScenarioError as exc:
            st.error("\n".join(exc.problems))
            return
        with st.spinner(f"Simulating {scenario.n_slots:,} slots"):
            result = simulate(scenario)
        show_result(result)


if __name__ == "__main__":
    main()
