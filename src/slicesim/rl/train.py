"""Training the slicing agent: seeded, logged to CSV, checkpointed."""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch

from slicesim.config import Scenario
from slicesim.rl.env import SliceEnv
from slicesim.rl.ppo import PPO, PPOConfig, UpdateStats

TRAINING_SEEDS = (0, 1_000_000)
"""Scenario seeds training draws from; evaluation uses seeds above this range."""

LOG_FIELDS = [f.name for f in fields(UpdateStats)]


@dataclass(frozen=True)
class TrainingRun:
    """Where a training run wrote its log and checkpoints, and what it logged."""

    log: Path
    checkpoint: Path
    """The final checkpoint."""
    checkpoints: list[Path]
    """Intermediate checkpoints, in order."""
    history: list[UpdateStats]


def train(
    levels: Sequence[Scenario],
    out_dir: str | Path,
    config: PPOConfig | None = None,
    decision_slots: int = 10,
    episode_s: float = 1.0,
    checkpoint_every: int = 5,
    on_update: Callable[[UpdateStats], None] | None = None,
) -> TrainingRun:
    """Train a PPO agent on the load ``levels`` and write its log and checkpoints.

    ``out_dir`` receives ``log.csv`` (one row per update), a checkpoint every
    ``checkpoint_every`` updates (``checkpoint-0005.pt`` and so on) and the
    final ``agent.pt``. Torch runs on one thread, which is faster for networks
    this small and makes the run reproducible from ``config.seed``.
    """
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")
    config = config or PPOConfig()
    torch.set_num_threads(1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = SliceEnv(
        levels,
        decision_slots=decision_slots,
        episode_s=episode_s,
        scenario_seeds=TRAINING_SEEDS,
    )
    agent = PPO.for_env(env, config)
    metadata = {
        "slices": env.slice_names,
        "levels": [scenario.name for scenario in levels],
        "decision_slots": decision_slots,
        "episode_s": episode_s,
    }
    log_path = out / "log.csv"
    checkpoints: list[Path] = []
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        writer.writeheader()

        def record(stats: UpdateStats) -> None:
            writer.writerow(asdict(stats))
            log_file.flush()
            if stats.update % checkpoint_every == 0:
                path = out / f"checkpoint-{stats.update:04d}.pt"
                checkpoints.append(agent.save(path, metadata))
            if on_update is not None:
                on_update(stats)

        history = agent.learn(env, on_update=record)
    final = agent.save(out / "agent.pt", metadata)
    return TrainingRun(log=log_path, checkpoint=final, checkpoints=checkpoints, history=history)
