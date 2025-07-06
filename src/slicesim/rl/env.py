"""A Gymnasium environment in which an agent splits a sliced cell's PRBs.

Each step runs ``decision_slots`` slots of the simulator under one split. The
agent chooses the split with one logit per slice, in [-1, 1] as Gymnasium
recommends for continuous actions. The logits are scaled by ``LOGIT_SCALE``
and a softmax turns them into weights, so one slice can weigh up to e^10
times another; the cell's PRBs are shared by those weights, a slice that
cannot use its share passing the rest on (``WeightedShares``). Every action
is therefore a valid split. HARQ retransmissions and the slices' own
schedulers work as in any other run; only the inter-slice policy is replaced.

Observation, five features per slice, in slice order:

1. queue backlog: queued bytes B over B + C, where C is what the carrier
   carries in one step at the highest CQI;
2. head-of-line delay: the oldest queued packet's wait d over d + D, where D
   is the slice's delay budget (the SLA window for eMBB);
3. recent throughput: bytes delivered in the last step over C;
4. mean CQI: the slice's mean reported CQI over 15 (1 on the fixed link);
5. SLA slack, from -1 to 1: for URLLC and mMTC 1 - 2 min(d / D, 1), so 1 with
   an empty queue and -1 once the oldest packet is past its budget; for eMBB
   2 s - 1, where s is the slice's mean share of its owed rate so far in the
   current SLA window.

Reward per step: for each slice, a weight times its satisfaction in the step,
minus a penalty times its new violations.

- URLLC and mMTC satisfaction is the share of packets delivered in the step
  that were on time. A violation is counted when a packet passes its
  deadline, whether it is then delivered late, still queued or lost, so
  starving a slice is penalized as it happens, and over an episode the
  penalties count exactly the violations the SLA accounting reports.
- eMBB satisfaction is the mean share of its owed rate each UE received in
  the step, owed as in the SLA (the minimum rate or what the UE asked for).
  Violations are the UE-windows short of the minimum, counted when each SLA
  window closes.

Each reset picks one of the load levels at random and a fresh scenario seed,
both from the environment's seeded generator, so ``reset(seed=s)`` followed by
the same actions always replays the same episode.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from slicesim.channel.link import bits_per_prb
from slicesim.config import AdaptiveLink, EmbbSla, Scenario
from slicesim.engine import Simulation
from slicesim.slicing import WeightedShares
from slicesim.slicing.sla import embb_window_counts

FEATURES = ("backlog", "head_of_line_delay", "throughput", "cqi", "sla_slack")
"""The per-slice observation features, in order."""

LOGIT_SCALE = 5.0
"""Actions in [-1, 1] become logits in [-5, 5] before the softmax."""

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights and penalties, keyed by SLA type."""

    satisfaction: Mapping[str, float] = field(
        default_factory=lambda: {"embb": 1.0, "urllc": 1.0, "mmtc": 0.5}
    )
    """Weight of each slice's satisfaction in the step."""
    penalty: Mapping[str, float] = field(
        default_factory=lambda: {"embb": 0.2, "urllc": 1.0, "mmtc": 0.1}
    )
    """Penalty per violation: per late or lost packet, or per short eMBB UE-window."""


def softmax(logits: npt.ArrayLike) -> FloatArray:
    """The softmax of ``logits``, computed stably."""
    x = np.asarray(logits, dtype=np.float64)
    z = np.exp(x - np.max(x))
    result: FloatArray = z / z.sum()
    return result


class SliceEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """Choose each slice's share of the cell, one decision interval at a time."""

    # Gymnasium declares metadata as an instance attribute, so it cannot be a ClassVar.
    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        levels: Sequence[Scenario],
        decision_slots: int = 10,
        episode_s: float = 1.0,
        reward: RewardConfig | None = None,
    ) -> None:
        """``levels`` are the scenarios an episode is drawn from, typically one per
        load level; they must have the same slices in the same order."""
        super().__init__()
        if not levels:
            raise ValueError("give at least one scenario")
        first = levels[0]
        if not first.slices:
            raise ValueError(f"scenario {first.name} is not sliced")
        signature = [(s.name, s.sla.type) for s in first.slices]
        for scenario in levels[1:]:
            if [(s.name, s.sla.type) for s in scenario.slices] != signature:
                raise ValueError(
                    f"scenario {scenario.name} has different slices from {first.name}; "
                    "every level needs the same slices in the same order"
                )
        if decision_slots < 1:
            raise ValueError("decision_slots must be at least 1")
        if not episode_s > 0:
            raise ValueError("episode_s must be positive")
        self.levels = list(levels)
        self.decision_slots = decision_slots
        self.episode_s = episode_s
        self.reward_config = reward or RewardConfig()
        self.slice_names = [name for name, _ in signature]
        self.sla_types = [kind for _, kind in signature]
        for table in (self.reward_config.satisfaction, self.reward_config.penalty):
            missing = sorted(set(self.sla_types) - set(table))
            if missing:
                raise ValueError(f"reward weights missing for {', '.join(missing)}")

        n = len(signature)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        low = np.tile(np.array([0, 0, 0, 0, -1], dtype=np.float32), n)
        high = np.ones(5 * n, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.sim: Simulation | None = None
        self.level = 0
        self._policy = WeightedShares(np.full(n, 1.0 / n))

    # -- episode control -------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        if options and "level" in options:
            self.level = int(options["level"])
            if not 0 <= self.level < len(self.levels):
                raise ValueError(f"level must be 0 to {len(self.levels) - 1}")
        else:
            self.level = int(rng.integers(len(self.levels)))
        scenario_seed = int(rng.integers(2**31))
        scenario = self.levels[self.level].replace(seed=scenario_seed, duration_s=self.episode_s)
        self.sim = Simulation(scenario)
        self.sim.policy = self._policy
        self._policy.weights = np.full(len(self.slice_names), 1.0 / len(self.slice_names))
        self._setup_scales(scenario)
        self._previous = self._counters()
        self._closed_windows = 0
        self._last_delivered = np.zeros(len(self.slice_names))
        return self._observation(), {"level": scenario.name, "seed": scenario_seed}

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        sim = self._require_sim()
        clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if clipped.shape != (len(self.slice_names),):
            raise ValueError(f"expected {len(self.slice_names)} logits, got {clipped.shape}")
        weights = softmax(LOGIT_SCALE * clipped)
        self._policy.weights = weights

        start_backlog = sim.ues.backlog_bytes.copy()
        before = self._previous
        sim.step(self.decision_slots)
        after = self._counters()
        self._previous = after

        satisfaction, violations = self._step_outcome(before, after, start_backlog)
        rewards = self.reward_config
        reward = sum(
            rewards.satisfaction[kind] * float(satisfaction[i])
            - rewards.penalty[kind] * float(violations[i])
            for i, kind in enumerate(self.sla_types)
        )
        delivered = after["delivered"] - before["delivered"]
        self._last_delivered = np.array(
            [delivered[s.members].sum() for s in sim.slices], dtype=np.float64
        )
        info = {
            "level": sim.scenario.name,
            "slot": sim.slot,
            "weights": weights,
            "satisfaction": satisfaction,
            "violations": violations,
        }
        return self._observation(), float(reward), False, sim.finished, info

    # -- internals ---------------------------------------------------------

    def _require_sim(self) -> Simulation:
        if self.sim is None:
            raise RuntimeError("call reset() before step()")
        return self.sim

    def _setup_scales(self, scenario: Scenario) -> None:
        sim = self._require_sim()
        link = scenario.link
        peak_bits = (
            int(bits_per_prb(15, link.overhead))
            if isinstance(link, AdaptiveLink)
            else link.bits_per_prb
        )
        self._step_capacity = sim.carrier.n_prb * peak_bits / 8 * self.decision_slots
        window_s = scenario.slicing_config.window_ms * 1e-3
        self._delay_scale = np.array(
            [s.delay_budget_s if math.isfinite(s.delay_budget_s) else window_s for s in sim.slices]
        )
        step_s = self.decision_slots * sim.carrier.slot_duration_s
        self._min_rate_bits = np.array(
            [
                s.sla.min_throughput_mbps * 1e6 if isinstance(s.sla, EmbbSla) else 0.0
                for s in sim.slices
            ]
        )
        self._step_s = step_s
        self._window_s = window_s

    def _counters(self) -> dict[str, npt.NDArray[np.int64]]:
        sim = self._require_sim()
        m = sim.metrics
        return {
            "delivered": m.delivered_bytes.copy(),
            "offered": m.offered_bytes.copy(),
            "packets": m.packets_delivered.copy(),
            "late": m.packets_late.copy(),
            "lost": m.packets_lost.copy(),
            "overdue": sim.overdue_packets(),
        }

    def _step_outcome(
        self,
        before: dict[str, npt.NDArray[np.int64]],
        after: dict[str, npt.NDArray[np.int64]],
        start_backlog: npt.NDArray[np.int64],
    ) -> tuple[FloatArray, FloatArray]:
        sim = self._require_sim()
        n = len(sim.slices)
        satisfaction = np.ones(n)
        violations = np.zeros(n)
        delta = {key: after[key] - before[key] for key in after}
        for i, s in enumerate(sim.slices):
            members = s.members
            if isinstance(s.sla, EmbbSla):
                owed = np.minimum(
                    (start_backlog[members] + delta["offered"][members]) * 8.0,
                    self._min_rate_bits[i] * self._step_s,
                )
                owed[sim.ues.full_buffer[members]] = self._min_rate_bits[i] * self._step_s
                counted = owed > 0
                if counted.any():
                    got = delta["delivered"][members][counted] * 8.0
                    satisfaction[i] = float(np.minimum(got / owed[counted], 1.0).mean())
            else:
                # A packet's violation is counted once, when it passes its deadline:
                # late deliveries and losses add, and so does growth in the queue of
                # packets already past it.
                violations[i] = float(
                    delta["late"][members].sum()
                    + delta["lost"][members].sum()
                    + delta["overdue"][members].sum()
                )
                delivered = int(delta["packets"][members].sum())
                if delivered:
                    late = int(delta["late"][members].sum())
                    satisfaction[i] = 1.0 - late / delivered
        violations += self._closed_window_violations()
        return satisfaction, violations

    def _closed_window_violations(self) -> FloatArray:
        """Short eMBB UE-windows in the SLA windows that closed during the step."""
        sim = self._require_sim()
        m = sim.metrics
        closed = sim.slot // sim.window_slots
        short = np.zeros(len(sim.slices))
        if closed > self._closed_windows:
            window = slice(self._closed_windows, closed)
            for i, s in enumerate(sim.slices):
                if not isinstance(s.sla, EmbbSla):
                    continue
                members = s.members
                short[i] = embb_window_counts(
                    m.window_delivered[window][:, members],
                    m.window_offered[window][:, members],
                    m.window_backlog[window][:, members],
                    sim.ues.full_buffer[members],
                    self._min_rate_bits[i] * self._window_s,
                ).short
            self._closed_windows = closed
        return short

    def _observation(self) -> npt.NDArray[np.float32]:
        sim = self._require_sim()
        now = sim.now
        backlog = sim.ues.backlog_bytes
        hol = sim.ues.head_of_line_delay(now)
        cqi = sim.reported_cqi
        m = sim.metrics
        features = []
        window_start = (sim.slot // sim.window_slots) * sim.window_slots
        elapsed = (sim.slot - window_start) * sim.carrier.slot_duration_s
        current = min(sim.slot // sim.window_slots, m.window_delivered.shape[0] - 1)
        for i, s in enumerate(sim.slices):
            members = s.members
            queued = float(backlog[members].sum())
            oldest = float(np.max(hol[members])) if members.size else 0.0
            scale = float(self._delay_scale[i])
            throughput = min(self._last_delivered[i] / self._step_capacity, 1.0)
            mean_cqi = 1.0 if cqi is None else float(cqi[members].mean()) / 15
            if isinstance(s.sla, EmbbSla):
                slack = 1.0
                if elapsed > 0:
                    owed = np.minimum(
                        (m.window_backlog[current, members] + m.window_offered[current, members])
                        * 8.0,
                        self._min_rate_bits[i] * elapsed,
                    )
                    owed[sim.ues.full_buffer[members]] = self._min_rate_bits[i] * elapsed
                    counted = owed > 0
                    if counted.any():
                        got = m.window_delivered[current, members][counted] * 8.0
                        slack = 2 * float(np.minimum(got / owed[counted], 1.0).mean()) - 1
            else:
                slack = 1 - 2 * min(oldest / scale, 1.0)
            features += [
                queued / (queued + self._step_capacity),
                oldest / (oldest + scale),
                throughput,
                mean_cqi,
                slack,
            ]
        return np.asarray(features, dtype=np.float32)


ENV_ID = "slicesim/Slicing-v0"
"""The environment's Gymnasium id: ``gymnasium.make(ENV_ID, levels=[...])``."""

if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="slicesim.rl.env:SliceEnv")
