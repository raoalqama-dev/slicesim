"""Figures from an experiment's CSV files (the plots extra: matplotlib).

- ``sla_vs_load.png``: URLLC packets late or lost, and eMBB UE-windows short of
  the minimum rate, against load, per policy, with each SLA's limit;
- ``delay_cdf.png``: the distribution of URLLC packet delay at one load, per
  policy, with the tail on a log scale;
- ``throughput_per_slice.png``: each slice's throughput per policy at each
  load, on a log scale.

Points are means over seeds; error bars span the lowest and highest seed.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

POLICY_LABELS = {
    "static": "static quotas",
    "proportional": "proportional",
    "delay_aware": "delay-aware",
    "equal_weights": "equal weights",
    "ppo": "PPO agent",
}
POLICY_COLORS = {
    "static": "#d62728",
    "proportional": "#ff7f0e",
    "delay_aware": "#2ca02c",
    "equal_weights": "#7f7f7f",
    "ppo": "#1f77b4",
}
LOAD_ORDER = ("low", "medium", "high")


def _read(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ordered(values: Sequence[str], order: Sequence[str]) -> list[str]:
    seen = list(dict.fromkeys(values))
    return sorted(seen, key=lambda v: (order.index(v) if v in order else len(order), seen.index(v)))


def _label(policy: str) -> str:
    return POLICY_LABELS.get(policy, policy)


def _color(policy: str) -> str | None:
    return POLICY_COLORS.get(policy)


def _sla_panel(
    ax: Axes,
    rows: list[dict[str, str]],
    slice_name: str,
    levels: list[str],
    policies: list[str],
    limit: float,
    title: str,
) -> None:
    x = np.arange(len(levels))
    for k, policy in enumerate(policies):
        means, lows, highs = [], [], []
        for level in levels:
            values = [
                float(r["violation_ratio"]) * 100
                for r in rows
                if r["slice"] == slice_name
                and r["policy"] == policy
                and r["level"] == level
                and r["violation_ratio"] != ""
            ]
            values = values or [float("nan")]
            means.append(float(np.mean(values)))
            lows.append(float(np.min(values)))
            highs.append(float(np.max(values)))
        offset = (k - (len(policies) - 1) / 2) * 0.06
        mean = np.array(means)
        ax.errorbar(
            x + offset,
            mean,
            # Plain lists: matplotlib's check of array error bars trips a numpy
            # deprecation when there is a single load level.
            yerr=[(mean - np.array(lows)).tolist(), (np.array(highs) - mean).tolist()],
            marker="o",
            capsize=3,
            label=_label(policy),
            color=_color(policy),
        )
    ax.axhline(limit, color="black", linestyle="--", linewidth=1, label="SLA limit")
    ax.set_xticks(x, levels)
    ax.set_xlabel("load")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)


def plot_sla_vs_load(runs: list[dict[str, str]], out: Path) -> Path:
    levels = _ordered([r["level"] for r in runs], LOAD_ORDER)
    policies = _ordered([r["policy"] for r in runs], list(POLICY_LABELS))
    fig = Figure(figsize=(11, 4.2))
    urllc, embb = fig.subplots(1, 2)
    _sla_panel(urllc, runs, "urllc", levels, policies, 0.1, "URLLC: packets late or lost")
    urllc.set_ylabel("% of packets")
    _sla_panel(embb, runs, "embb", levels, policies, 5.0, "eMBB: UE-windows short of 2 Mbit/s")
    embb.set_ylabel("% of UE-windows")
    urllc.legend(fontsize=8)
    fig.tight_layout()
    path = out / "sla_vs_load.png"
    fig.savefig(path, dpi=120)
    return path


def plot_delay_cdf(
    delays: list[dict[str, str]],
    out: Path,
    level: str = "medium",
    slice_name: str = "urllc",
    budget_ms: float = 5.0,
) -> Path:
    curves: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in delays:
        if r["level"] == level and r["slice"] == slice_name:
            curves[r["policy"]][float(r["quantile"])].append(float(r["delay_ms"]))
    if not curves:
        raise ValueError(f"no {slice_name} delays at {level} load")
    fig = Figure(figsize=(11, 4.2))
    cdf, tail = fig.subplots(1, 2)
    for policy in _ordered(list(curves), list(POLICY_LABELS)):
        quantiles = np.array(sorted(curves[policy]))
        # The quantile curve averaged over seeds.
        delay = np.array([np.mean(curves[policy][q]) for q in quantiles])
        kwargs = {"label": _label(policy), "color": _color(policy)}
        cdf.plot(delay, quantiles, **kwargs)
        keep = quantiles < 1
        tail.plot(delay[keep], 1 - quantiles[keep], **kwargs)
    for ax in (cdf, tail):
        ax.axvline(budget_ms, color="black", linestyle="--", linewidth=1, label="delay budget")
        ax.set_xlabel("packet delay (ms)")
        ax.grid(alpha=0.3)
    cdf.set_ylabel("share of packets delivered within")
    cdf.set_title(f"{slice_name.upper()} delay at {level} load")
    tail.set_yscale("log")
    tail.set_ylabel("share of packets slower than")
    tail.set_title("the tail")
    cdf.legend(fontsize=8)
    fig.tight_layout()
    path = out / "delay_cdf.png"
    fig.savefig(path, dpi=120)
    return path


def plot_throughput(runs: list[dict[str, str]], out: Path) -> Path:
    levels = _ordered([r["level"] for r in runs], LOAD_ORDER)
    policies = _ordered([r["policy"] for r in runs], list(POLICY_LABELS))
    slices = list(dict.fromkeys(r["slice"] for r in runs))
    fig = Figure(figsize=(4 * len(levels), 4.2))
    axes = fig.subplots(1, len(levels), squeeze=False)
    width = 0.8 / len(policies)
    for ax, level in zip(axes[0], levels, strict=True):
        x = np.arange(len(slices))
        for k, policy in enumerate(policies):
            heights = [
                float(
                    np.mean(
                        [
                            float(r["delivered_mbps"])
                            for r in runs
                            if r["level"] == level and r["policy"] == policy and r["slice"] == s
                        ]
                        or [0.0]
                    )
                )
                for s in slices
            ]
            ax.bar(
                x + (k - (len(policies) - 1) / 2) * width,
                heights,
                width,
                label=_label(policy),
                color=_color(policy),
            )
        ax.set_xticks(x, slices)
        ax.set_title(f"{level} load")
        # A log scale, so mMTC's tenths of a Mbit/s show next to eMBB's tens.
        ax.set_yscale("log")
        ax.set_ylabel("throughput (Mbit/s)")
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    path = out / "throughput_per_slice.png"
    fig.savefig(path, dpi=120)
    return path


def make_figures(results_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Draw every figure from ``runs.csv`` and ``delays.csv`` in ``results_dir``."""
    results = Path(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = _read(results / "runs.csv")
    delays = _read(results / "delays.csv")
    if not runs:
        raise ValueError(f"{results / 'runs.csv'} has no runs")
    levels = {r["level"] for r in delays}
    level = "medium" if "medium" in levels else _ordered(list(levels), LOAD_ORDER)[0]
    return [
        plot_sla_vs_load(runs, out),
        plot_delay_cdf(delays, out, level=level),
        plot_throughput(runs, out),
    ]
