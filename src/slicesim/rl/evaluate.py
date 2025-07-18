"""Comparing a trained agent with the heuristic policies on held-out seeds.

Every policy runs every load level with the same held-out scenario seeds, so
each sees the same UEs, channels and traffic, and none of them were seen in
training. The agent acts deterministically (the mean of its policy). Next to
the static, proportional and delay-aware policies, the comparison includes
equal weights: the agent's own action space with every logit at zero, which
is what an untrained agent starts near.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from slicesim.compare import with_policy
from slicesim.config import Scenario
from slicesim.engine import simulate
from slicesim.metrics import RunSummary, SimulationResult
from slicesim.rl.env import SliceEnv
from slicesim.rl.ppo import PPO
from slicesim.rl.train import TRAINING_SEEDS

HELD_OUT_SEEDS = TRAINING_SEEDS[1]
"""Evaluation uses scenario seeds from here up, which training never draws."""

HEURISTICS = ("static", "proportional", "delay_aware")
AGENT = "ppo"
EQUAL = "equal_weights"


@dataclass(frozen=True)
class EvaluationRow:
    """One policy on one load level with one seed."""

    level: str
    policy: str
    seed: int
    cell_mbps: float
    embb_short: float
    """Share of eMBB UE-windows short of the minimum rate."""
    embb_met: bool
    urllc_violations: float
    """Share of URLLC packets late, lost or overdue."""
    urllc_met: bool
    mmtc_violations: float
    mmtc_met: bool


def _row(level: str, policy: str, seed: int, summary: RunSummary) -> EvaluationRow:
    by_type = {s.sla: s for s in summary.slices}
    embb, urllc, mmtc = by_type["embb"], by_type["urllc"], by_type["mmtc"]
    return EvaluationRow(
        level=level,
        policy=policy,
        seed=seed,
        cell_mbps=summary.cell.delivered_mbps,
        embb_short=embb.violation_ratio or 0.0,
        embb_met=bool(embb.met),
        urllc_violations=urllc.violation_ratio or 0.0,
        urllc_met=bool(urllc.met),
        mmtc_violations=mmtc.violation_ratio or 0.0,
        mmtc_met=bool(mmtc.met),
    )


def run_agent(agent: PPO | None, scenario: Scenario, decision_slots: int) -> RunSummary:
    """Run ``scenario`` with the agent choosing the split, or equal weights if None."""
    return run_agent_result(agent, scenario, decision_slots).summary()


def run_agent_result(
    agent: PPO | None, scenario: Scenario, decision_slots: int
) -> SimulationResult:
    """Like ``run_agent``, returning the full result rather than its summary."""
    env = SliceEnv(
        [scenario],
        decision_slots=decision_slots,
        episode_s=scenario.duration_s,
        scenario_seeds=(scenario.seed, scenario.seed + 1),
    )
    obs, _ = env.reset(seed=0, options={"level": 0, "scenario_seed": scenario.seed})
    zeros = np.zeros(len(env.slice_names), dtype=np.float32)
    done = False
    while not done:
        action = zeros if agent is None else agent.act(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    assert env.sim is not None
    return env.sim.result()


@dataclass(frozen=True)
class _Job:
    checkpoint: str | None
    scenario: Scenario
    policy: str
    decision_slots: int


def _run_job(job: _Job) -> EvaluationRow:
    torch.set_num_threads(1)
    scenario = job.scenario
    if job.policy in HEURISTICS:
        summary = simulate(with_policy(scenario, job.policy)).summary()
    else:
        agent = None
        if job.policy == AGENT:
            assert job.checkpoint is not None
            agent, _ = PPO.load(job.checkpoint)
        summary = run_agent(agent, scenario, job.decision_slots)
    return _row(scenario.name, job.policy, scenario.seed, summary)


def evaluate(
    checkpoint: str | Path,
    levels: Sequence[Scenario],
    seeds: int = 3,
    duration_s: float = 10.0,
    workers: int = 1,
) -> list[EvaluationRow]:
    """Run the agent, equal weights and the heuristics on every level and seed."""
    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    _, metadata = PPO.load(checkpoint)
    expected = metadata.get("slices")
    decision_slots = int(metadata.get("decision_slots", 10))
    jobs = []
    for level in levels:
        names = [s.name for s in level.slices]
        if expected is not None and names != expected:
            raise ValueError(
                f"scenario {level.name} has slices {', '.join(names)}, but the agent was "
                f"trained on {', '.join(expected)}"
            )
        for k in range(seeds):
            scenario = level.replace(seed=HELD_OUT_SEEDS + k, duration_s=duration_s)
            for policy in (AGENT, EQUAL, *HEURISTICS):
                jobs.append(_Job(str(checkpoint), scenario, policy, decision_slots))
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            return list(pool.map(_run_job, jobs))
    return [_run_job(job) for job in jobs]


@dataclass(frozen=True)
class PolicyTotals:
    """One policy on one level, over every seed."""

    level: str
    policy: str
    seeds: int
    cell_mbps: float
    embb_short: float
    embb_met: int
    urllc_violations: float
    urllc_met: int
    mmtc_violations: float
    mmtc_met: int


def aggregate(rows: Sequence[EvaluationRow]) -> list[PolicyTotals]:
    """Means of the rates and counts of SLAs met, per level and policy, in run order."""
    keys: list[tuple[str, str]] = []
    for row in rows:
        if (row.level, row.policy) not in keys:
            keys.append((row.level, row.policy))
    totals = []
    for level, policy in keys:
        group = [r for r in rows if r.level == level and r.policy == policy]
        totals.append(
            PolicyTotals(
                level=level,
                policy=policy,
                seeds=len(group),
                cell_mbps=float(np.mean([r.cell_mbps for r in group])),
                embb_short=float(np.mean([r.embb_short for r in group])),
                embb_met=sum(r.embb_met for r in group),
                urllc_violations=float(np.mean([r.urllc_violations for r in group])),
                urllc_met=sum(r.urllc_met for r in group),
                mmtc_violations=float(np.mean([r.mmtc_violations for r in group])),
                mmtc_met=sum(r.mmtc_met for r in group),
            )
        )
    return totals


def write_csv(rows: Sequence[EvaluationRow], path: str | Path) -> Path:
    """Every run as one CSV row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(EvaluationRow)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path
