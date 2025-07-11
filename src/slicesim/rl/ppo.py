"""Proximal policy optimization, written from scratch in PyTorch.

The agent is an actor-critic with separate networks. The actor outputs the
mean of a diagonal Gaussian over the action (one logit per slice), with a
learned, state-independent log standard deviation; the critic estimates the
state value. Training follows Schulman et al., "Proximal Policy Optimization
Algorithms" (2017):

- rollouts of ``rollout_steps`` environment steps;
- advantages by generalized advantage estimation (Schulman et al., 2016),
  gamma 0.99 and lambda 0.95 by default, bootstrapping from the critic where
  an episode was cut off by its time limit rather than ended;
- the clipped surrogate objective, epsilon 0.2;
- a clipped value loss, so the critic cannot move further than epsilon from
  its old estimate either;
- an entropy bonus;
- observations normalized by running mean and variance;
- advantages normalized per minibatch, and gradients clipped by global norm.

Sampled actions are clipped to the action space when they are sent to the
environment; the log-probabilities are those of the unclipped samples.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from torch import nn

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters, with the defaults used for the slicing agent."""

    total_steps: int = 60_000
    rollout_steps: int = 2_048
    epochs: int = 10
    minibatch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (64, 64)
    initial_log_std: float = -0.5
    seed: int = 0

    def __post_init__(self) -> None:
        positive = ("total_steps", "rollout_steps", "epochs", "minibatch_size")
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.minibatch_size > self.rollout_steps:
            raise ValueError("minibatch_size cannot exceed rollout_steps")
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
        if self.clip_range <= 0 or self.value_clip_range <= 0:
            raise ValueError("clip ranges must be positive")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("learning_rate and max_grad_norm must be positive")


class RunningNorm:
    """Running mean and variance of observations (Chan et al.'s parallel update)."""

    def __init__(self, size: int, clip: float = 10.0) -> None:
        self.mean = np.zeros(size)
        self.var = np.ones(size)
        self.count = 1e-4
        self.clip = clip

    def update(self, batch: FloatArray) -> None:
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, self.mean.size)
        n = batch.shape[0]
        if n == 0:
            return
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        m2 = self.var * self.count + batch_var * n + delta**2 * self.count * n / total
        self.var = m2 / total
        self.count = total

    def __call__(self, obs: npt.ArrayLike) -> FloatArray:
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        result: FloatArray = np.clip(x, -self.clip, self.clip)
        return result

    def state(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> RunningNorm:
        norm = cls(len(state["mean"]))
        norm.mean = np.asarray(state["mean"], dtype=np.float64)
        norm.var = np.asarray(state["var"], dtype=np.float64)
        norm.count = float(state["count"])
        return norm


def _mlp(sizes: list[int], out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(pairwise(sizes)):
        linear = nn.Linear(a, b)
        last = i == len(sizes) - 2
        nn.init.orthogonal_(linear.weight, gain=out_gain if last else math.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if not last:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


@dataclass(frozen=True)
class DiagonalGaussian:
    """A Gaussian with independent components; log-probabilities and entropies
    are summed over the action's components."""

    mean: torch.Tensor
    log_std: torch.Tensor

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(self.mean.shape, generator=generator)
        return self.mean + self.log_std.exp() * noise

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / self.log_std.exp()
        return (-0.5 * z**2 - self.log_std - _LOG_SQRT_2PI).sum(-1)

    def entropy(self) -> torch.Tensor:
        return (self.log_std + 0.5 + _LOG_SQRT_2PI).sum(-1)


class ActorCritic(nn.Module):
    """A Gaussian policy over the actions and a state-value estimate."""

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        initial_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.actor = _mlp([obs_size, *hidden_sizes, action_size], out_gain=0.01)
        self.critic = _mlp([obs_size, *hidden_sizes, 1], out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((action_size,), initial_log_std))

    def policy(self, obs: torch.Tensor) -> DiagonalGaussian:
        mean: torch.Tensor = self.actor(obs)
        return DiagonalGaussian(mean, self.log_std.expand_as(mean))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.critic(obs).squeeze(-1)
        return result


def compute_gae(
    rewards: FloatArray,
    values: FloatArray,
    next_values: FloatArray,
    terminated: npt.NDArray[np.bool_],
    episode_end: npt.NDArray[np.bool_],
    gamma: float,
    lam: float,
) -> tuple[FloatArray, FloatArray]:
    """Advantages and returns by generalized advantage estimation.

    ``next_values[t]`` is the critic's value of the state after step t. Where
    an episode terminated it is not used (the value after a terminal state is
    0); where it was truncated it bootstraps the cut-off return. The
    advantage recursion never crosses ``episode_end``.
    """
    n = rewards.size
    advantages = np.zeros(n)
    running = 0.0
    for t in range(n - 1, -1, -1):
        bootstrap = 0.0 if terminated[t] else next_values[t]
        delta = rewards[t] + gamma * bootstrap - values[t]
        running = delta + (0.0 if episode_end[t] else gamma * lam * running)
        advantages[t] = running
    return advantages, advantages + values


@dataclass
class UpdateStats:
    """What one rollout and update did, as logged."""

    update: int
    env_steps: int
    episodes: int
    mean_return: float
    """Mean return of the episodes that finished in this rollout; NaN if none."""
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    action_std: float
    seconds: float


@dataclass
class PPO:
    """A PPO agent for a Box-in, Box-out environment."""

    obs_size: int
    action_low: FloatArray
    action_high: FloatArray
    config: PPOConfig = field(default_factory=PPOConfig)
    model: ActorCritic = field(init=False)
    norm: RunningNorm = field(init=False)

    def __post_init__(self) -> None:
        torch.manual_seed(self.config.seed)
        self._torch_rng = torch.Generator().manual_seed(self.config.seed)
        self.model = ActorCritic(
            self.obs_size,
            self.action_low.size,
            self.config.hidden_sizes,
            self.config.initial_log_std,
        )
        self.norm = RunningNorm(self.obs_size)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate, eps=1e-5
        )

    @classmethod
    def for_env(cls, env: gym.Env[Any, Any], config: PPOConfig | None = None) -> PPO:
        obs_space, action_space = env.observation_space, env.action_space
        if not isinstance(obs_space, gym.spaces.Box) or not isinstance(
            action_space, gym.spaces.Box
        ):
            raise TypeError("PPO here needs Box observation and action spaces")
        return cls(
            obs_size=int(np.prod(obs_space.shape)),
            action_low=action_space.low.astype(np.float64),
            action_high=action_space.high.astype(np.float64),
            config=config or PPOConfig(),
        )

    # -- acting ------------------------------------------------------------

    def act(self, obs: npt.ArrayLike, deterministic: bool = True) -> npt.NDArray[np.float32]:
        """The action for ``obs``: the policy mean, or a sample from the policy."""
        with torch.no_grad():
            x = torch.as_tensor(self.norm(obs), dtype=torch.float32)
            dist = self.model.policy(x)
            action = dist.mean if deterministic else dist.sample(self._torch_rng)
        result: npt.NDArray[np.float32] = np.clip(
            action.numpy(), self.action_low, self.action_high
        ).astype(np.float32)
        return result

    # -- training ----------------------------------------------------------

    def learn(
        self,
        env: gym.Env[Any, Any],
        on_update: Callable[[UpdateStats], None] | None = None,
    ) -> list[UpdateStats]:
        """Train for ``config.total_steps`` environment steps."""
        cfg = self.config
        generator = np.random.default_rng(cfg.seed)
        obs, _ = env.reset(seed=int(generator.integers(2**31)))
        episode_return = 0.0
        history: list[UpdateStats] = []
        steps = 0
        update = 0
        while steps < cfg.total_steps:
            started = time.perf_counter()
            n = min(cfg.rollout_steps, cfg.total_steps - steps)
            # The normalized observation the policy acted on, as it was then.
            seen = np.zeros((n, self.obs_size))
            actions = np.zeros((n, self.action_low.size))
            log_probs = np.zeros(n)
            rewards = np.zeros(n)
            values = np.zeros(n)
            next_values = np.zeros(n)
            terminated = np.zeros(n, dtype=np.bool_)
            episode_end = np.zeros(n, dtype=np.bool_)
            finished: list[float] = []
            for t in range(n):
                self.norm.update(np.asarray(obs, dtype=np.float64)[None, :])
                seen[t] = self.norm(obs)
                with torch.inference_mode():
                    x = torch.as_tensor(seen[t], dtype=torch.float32)
                    dist = self.model.policy(x)
                    sample = dist.sample(self._torch_rng)
                    log_probs[t] = dist.log_prob(sample).item()
                    values[t] = self.model.value(x).item()
                actions[t] = sample.numpy()
                clipped = np.clip(actions[t], self.action_low, self.action_high)
                obs, reward, term, trunc, _ = env.step(clipped.astype(np.float32))
                rewards[t] = float(reward)
                episode_return += float(reward)
                terminated[t] = term
                episode_end[t] = term or trunc
                # Within an episode the next state's value is values[t + 1], filled in
                # below; only a cut-off episode, or the rollout's last step, needs the
                # critic's value of the state it stopped in.
                if (trunc and not term) or (t == n - 1 and not (term or trunc)):
                    next_values[t] = self._value(obs)
                if term or trunc:
                    finished.append(episode_return)
                    episode_return = 0.0
                    obs, _ = env.reset(seed=int(generator.integers(2**31)))
            within = ~episode_end[:-1]
            next_values[:-1][within] = values[1:][within]
            steps += n
            update += 1
            advantages, returns = compute_gae(
                rewards, values, next_values, terminated, episode_end, cfg.gamma, cfg.gae_lambda
            )
            losses = self._update(
                torch.as_tensor(seen, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                torch.as_tensor(log_probs, dtype=torch.float32),
                torch.as_tensor(advantages, dtype=torch.float32),
                torch.as_tensor(returns, dtype=torch.float32),
                torch.as_tensor(values, dtype=torch.float32),
                generator,
            )
            stats = UpdateStats(
                update=update,
                env_steps=steps,
                episodes=len(finished),
                mean_return=float(np.mean(finished)) if finished else math.nan,
                action_std=self.model.log_std.detach().exp().mean().item(),
                seconds=time.perf_counter() - started,
                **losses,
            )
            history.append(stats)
            if on_update is not None:
                on_update(stats)
        return history

    def _value(self, obs: npt.ArrayLike) -> float:
        with torch.inference_mode():
            return self.model.value(torch.as_tensor(self.norm(obs), dtype=torch.float32)).item()

    def _update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        generator: np.random.Generator,
    ) -> dict[str, float]:
        cfg = self.config
        n = obs.shape[0]
        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        clipped = 0.0
        batches = 0
        for _ in range(cfg.epochs):
            order = generator.permutation(n)
            for start in range(0, n, cfg.minibatch_size):
                idx = torch.as_tensor(order[start : start + cfg.minibatch_size])
                dist = self.model.policy(obs[idx])
                log_prob = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()
                ratio = (log_prob - old_log_probs[idx]).exp()
                adv = advantages[idx]
                if adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                surrogate = torch.min(
                    ratio * adv,
                    ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * adv,
                )
                policy_loss = -surrogate.mean()

                value = self.model.value(obs[idx])
                old = old_values[idx]
                value_clipped = old + (value - old).clamp(
                    -cfg.value_clip_range, cfg.value_clip_range
                )
                value_loss = (
                    0.5
                    * torch.max(
                        (value - returns[idx]) ** 2, (value_clipped - returns[idx]) ** 2
                    ).mean()
                )

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                self.optimizer.zero_grad()
                torch.autograd.backward(loss)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    log_ratio = log_prob - old_log_probs[idx]
                    totals["approx_kl"] += ((log_ratio.exp() - 1) - log_ratio).mean().item()
                    clipped += ((ratio - 1).abs() > cfg.clip_range).float().mean().item()
                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy.item()
                batches += 1
        result = {key: value / batches for key, value in totals.items()}
        result["clip_fraction"] = clipped / batches
        return result

    # -- checkpoints -------------------------------------------------------

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Write the model, the observation statistics and the config to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = asdict(self.config)
        torch.save(
            {
                "format": 1,
                "obs_size": self.obs_size,
                "action_low": self.action_low.tolist(),
                "action_high": self.action_high.tolist(),
                "config": config,
                "model": self.model.state_dict(),
                "norm": self.norm.state(),
                "metadata": metadata or {},
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> tuple[PPO, dict[str, Any]]:
        """Read a checkpoint written by ``save``; returns the agent and its metadata."""
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
        if data.get("format") != 1:
            raise ValueError(f"{path} is not a slicesim PPO checkpoint")
        config = dict(data["config"])
        config["hidden_sizes"] = tuple(config["hidden_sizes"])
        agent = cls(
            obs_size=int(data["obs_size"]),
            action_low=np.asarray(data["action_low"], dtype=np.float64),
            action_high=np.asarray(data["action_high"], dtype=np.float64),
            config=PPOConfig(**config),
        )
        agent.model.load_state_dict(data["model"])
        agent.norm = RunningNorm.from_state(data["norm"])
        return agent, dict(data["metadata"])
