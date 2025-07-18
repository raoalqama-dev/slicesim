"""Experiments: every load level under every policy with several seeds.

``run_experiment`` runs each combination once and records, per run and per
slice, throughput, PRB share, the SLA violation ratio and whether the SLA was
met, delay percentiles, and the delay quantiles a CDF is drawn from. All
policies run a level with the same seeds, so they see the same UEs, channels
and traffic.

The heuristic policies need only the core package. ``equal_weights`` (the
agent's action space with every logit at zero) and ``ppo`` (a trained agent)
need the rl extra, imported only when one of them is asked for.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from slicesim.compare import with_policy
from slicesim.config import Scenario
from slicesim.engine import simulate
from slicesim.metrics import SimulationResult

HELD_OUT_SEEDS = 1_000_000
"""The first scenario seed experiments use by default: the agent trains on seeds
below this, so results above it are on traffic and channels it never saw."""

HEURISTICS = ("static", "proportional", "delay_aware")
LEARNED = ("equal_weights", "ppo")
POLICIES = (*HEURISTICS, *LEARNED)

DELAY_QUANTILES = tuple(
    round(float(q), 3)
    for q in np.concatenate(
        [np.linspace(0.0, 0.9, 19), np.linspace(0.91, 0.99, 9), [0.995, 0.999, 1.0]]
    )
)
"""Quantiles recorded per slice and run: 5% steps through the body of the delay
distribution, 1% steps and then finer through its tail."""


@dataclass(frozen=True)
class SliceRow:
    """One slice in one run."""

    level: str
    policy: str
    seed: int
    slice: str
    sla: str
    ues: int
    delivered_mbps: float
    prb_share: float
    violation_ratio: float | None
    met: bool | None
    satisfaction: float | None
    packets: int
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None


@dataclass(frozen=True)
class DelayRow:
    """One quantile of one slice's packet delay in one run."""

    level: str
    policy: str
    seed: int
    slice: str
    quantile: float
    delay_ms: float


@dataclass(frozen=True)
class RunResult:
    slices: list[SliceRow]
    delays: list[DelayRow]


def run_policy(scenario: Scenario, policy: str, checkpoint: str | None = None) -> SimulationResult:
    """Run ``scenario`` under ``policy``; ``ppo`` needs ``checkpoint``."""
    if policy in HEURISTICS:
        return simulate(with_policy(scenario, policy))
    if policy not in LEARNED:
        raise ValueError(f"unknown policy {policy!r}; choose from {', '.join(POLICIES)}")
    if policy == "ppo" and checkpoint is None:
        raise ValueError("the ppo policy needs a checkpoint")
    # The learned policies need the rl extra.
    from slicesim.rl.evaluate import run_agent_result
    from slicesim.rl.ppo import PPO

    agent = None
    decision_slots = 10
    if checkpoint is not None and policy == "ppo":
        agent, metadata = PPO.load(checkpoint)
        decision_slots = int(metadata.get("decision_slots", decision_slots))
    return run_agent_result(agent, scenario, decision_slots)


def summarize_run(level: str, policy: str, seed: int, result: SimulationResult) -> RunResult:
    """The rows one run contributes to an experiment's CSV files."""
    slices: list[SliceRow] = []
    delays: list[DelayRow] = []
    for summary in result.summary().slices:
        index = [s.name for s in result.scenario.slices].index(summary.name)
        packet_delays = result.delay_s[result.ue_slice[result.delay_ue] == index] * 1e3
        delay = summary.delay
        slices.append(
            SliceRow(
                level=level,
                policy=policy,
                seed=seed,
                slice=summary.name,
                sla=summary.sla,
                ues=summary.ues,
                delivered_mbps=summary.delivered_mbps,
                prb_share=summary.prb_share,
                violation_ratio=summary.violation_ratio,
                met=summary.met,
                satisfaction=summary.satisfaction,
                packets=summary.packets_delivered,
                p50_ms=None if delay is None else delay.p50_ms,
                p95_ms=None if delay is None else delay.p95_ms,
                p99_ms=None if delay is None else delay.p99_ms,
            )
        )
        if packet_delays.size:
            values = np.quantile(packet_delays, DELAY_QUANTILES)
            # Delays to 0.1 microsecond: finer than any slot, and a smaller file.
            delays.extend(
                DelayRow(level, policy, seed, summary.name, q, round(float(v), 4))
                for q, v in zip(DELAY_QUANTILES, values, strict=True)
            )
    return RunResult(slices=slices, delays=delays)


@dataclass(frozen=True)
class _Job:
    scenario: Scenario
    level: str
    policy: str
    checkpoint: str | None


def _run_job(job: _Job) -> RunResult:
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.set_num_threads(1)
    result = run_policy(job.scenario, job.policy, job.checkpoint)
    return summarize_run(job.level, job.policy, job.scenario.seed, result)


def run_experiment(
    levels: Sequence[Scenario],
    policies: Sequence[str] = HEURISTICS,
    seeds: Sequence[int] = (HELD_OUT_SEEDS, HELD_OUT_SEEDS + 1, HELD_OUT_SEEDS + 2),
    duration_s: float | None = None,
    checkpoint: str | Path | None = None,
    workers: int = 1,
) -> RunResult:
    """Run every level under every policy with every seed.

    ``duration_s`` overrides each level's duration. Runs are ordered by level,
    then seed, then policy.
    """
    unknown = [p for p in policies if p not in POLICIES]
    if unknown:
        raise ValueError(f"unknown policy {unknown[0]!r}; choose from {', '.join(POLICIES)}")
    if "ppo" in policies and checkpoint is None:
        raise ValueError("the ppo policy needs a checkpoint")
    if not seeds:
        raise ValueError("give at least one seed")
    unsliced = [level.name for level in levels if not level.slices]
    if unsliced:
        raise ValueError(f"experiments need sliced scenarios; not sliced: {', '.join(unsliced)}")
    jobs = []
    for level in levels:
        for seed in seeds:
            changes: dict[str, object] = {"seed": seed}
            if duration_s is not None:
                changes["duration_s"] = duration_s
            scenario = level.replace(**changes)
            for policy in policies:
                jobs.append(
                    _Job(
                        scenario,
                        level.name,
                        policy,
                        None if checkpoint is None else str(checkpoint),
                    )
                )
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_run_job, jobs))
    else:
        results = [_run_job(job) for job in jobs]
    return RunResult(
        slices=[row for r in results for row in r.slices],
        delays=[row for r in results for row in r.delays],
    )


def write_results(result: RunResult, out_dir: str | Path) -> tuple[Path, Path]:
    """Write ``runs.csv`` (a row per slice and run) and ``delays.csv`` (delay quantiles)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows, kind in (
        ("runs.csv", result.slices, SliceRow),
        ("delays.csv", result.delays, DelayRow),
    ):
        path = out / name
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(kind)])
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        paths.append(path)
    return paths[0], paths[1]
