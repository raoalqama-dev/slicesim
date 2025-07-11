import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from gymnasium import Env, spaces  # noqa: E402

from slicesim.rl.ppo import (  # noqa: E402
    PPO,
    DiagonalGaussian,
    PPOConfig,
    RunningNorm,
    compute_gae,
)


class Target(Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """A toy task: the observation is a point, and the reward is minus the squared
    distance between the action and 0.8 times that point. Doing nothing scores
    about -0.43 per step; a perfect policy scores 0."""

    def __init__(self) -> None:
        self.observation_space: spaces.Space[Any] = spaces.Box(-1, 1, (2,), np.float32)
        self.action_space: spaces.Space[Any] = spaces.Box(-1, 1, (2,), np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self.t = 0
        self.x = self.np_random.uniform(-1, 1, 2).astype(np.float32)
        return self.x, {}

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        reward = -float(np.sum((np.asarray(action) - 0.8 * self.x) ** 2))
        self.t += 1
        self.x = self.np_random.uniform(-1, 1, 2).astype(np.float32)
        return self.x, reward, False, self.t >= 8, {}


TOY_CONFIG = PPOConfig(
    total_steps=8_192, rollout_steps=512, minibatch_size=128, epochs=5, learning_rate=1e-3
)


def score(agent: PPO, steps: int = 400) -> float:
    env = Target()
    obs, _ = env.reset(seed=123)
    total = 0.0
    for _ in range(steps):
        obs, reward, _, truncated, _ = env.step(agent.act(obs))
        total += reward
        if truncated:
            obs, _ = env.reset()
    return total / steps


@pytest.fixture(scope="module")
def toy_agent() -> PPO:
    torch.set_num_threads(1)
    agent = PPO.for_env(Target(), TOY_CONFIG)
    agent.learn(Target())
    return agent


def test_ppo_solves_a_toy_task(toy_agent: PPO) -> None:
    untrained = PPO.for_env(Target(), TOY_CONFIG)
    assert score(untrained) < -0.3
    assert score(toy_agent) > -0.1


def test_training_is_reproducible_from_the_seed() -> None:
    torch.set_num_threads(1)
    config = PPOConfig(total_steps=512, rollout_steps=256, minibatch_size=64, epochs=2, seed=3)
    first = PPO.for_env(Target(), config).learn(Target())
    second = PPO.for_env(Target(), config).learn(Target())
    assert [s.policy_loss for s in first] == [s.policy_loss for s in second]
    assert [s.mean_return for s in first] == [s.mean_return for s in second]
    assert [s.env_steps for s in first] == [256, 512]
    assert all(s.episodes == 32 for s in first)


def test_update_statistics_are_sane(toy_agent: PPO) -> None:
    config = PPOConfig(total_steps=1024, rollout_steps=512, minibatch_size=128, epochs=3)
    history = PPO.for_env(Target(), config).learn(Target())
    for stats in history:
        assert stats.value_loss > 0
        assert 0 <= stats.clip_fraction <= 1
        assert stats.approx_kl >= 0
        assert stats.action_std == pytest.approx(math.exp(-0.5), rel=0.2)
        assert stats.seconds > 0


class TestGae:
    def test_one_episode_matches_the_recursion(self) -> None:
        rewards = np.array([1.0, 0.0, 2.0])
        values = np.array([0.5, 0.4, 0.3])
        next_values = np.array([0.4, 0.3, 0.0])
        terminated = np.array([False, False, True])
        end = terminated.copy()
        adv, ret = compute_gae(rewards, values, next_values, terminated, end, 0.9, 0.8)
        d2 = 2.0 - 0.3
        d1 = 0.0 + 0.9 * 0.3 - 0.4
        d0 = 1.0 + 0.9 * 0.4 - 0.5
        a2 = d2
        a1 = d1 + 0.9 * 0.8 * a2
        a0 = d0 + 0.9 * 0.8 * a1
        assert adv == pytest.approx([a0, a1, a2])
        assert ret == pytest.approx(adv + values)

    def test_truncation_bootstraps_and_episodes_do_not_mix(self) -> None:
        rewards = np.array([1.0, 1.0, 1.0])
        values = np.zeros(3)
        next_values = np.array([0.0, 5.0, 7.0])
        terminated = np.array([False, False, False])
        end = np.array([False, True, False])
        adv, _ = compute_gae(rewards, values, next_values, terminated, end, 1.0, 1.0)
        # Step 1 was cut off: it bootstraps from the state it stopped in (5), and
        # step 2's advantage does not leak back across the boundary.
        assert adv[1] == pytest.approx(6.0)
        assert adv[0] == pytest.approx(1.0 + 0.0 + adv[1])
        assert adv[2] == pytest.approx(8.0)

    def test_termination_does_not_bootstrap(self) -> None:
        adv, _ = compute_gae(
            np.array([1.0]),
            np.array([0.0]),
            np.array([100.0]),
            np.array([True]),
            np.array([True]),
            0.99,
            0.95,
        )
        assert adv.tolist() == [1.0]


class TestRunningNorm:
    def test_matches_the_mean_and_variance_of_everything_seen(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(3.0, 2.0, (1000, 4))
        norm = RunningNorm(4)
        for chunk in np.array_split(data, 7):
            norm.update(chunk)
        assert norm.mean == pytest.approx(data.mean(axis=0), abs=1e-3)
        assert norm.var == pytest.approx(data.var(axis=0), rel=1e-3)
        z = norm(data)
        assert z.mean(axis=0) == pytest.approx(np.zeros(4), abs=1e-2)
        assert z.std(axis=0) == pytest.approx(np.ones(4), rel=1e-2)

    def test_clips_and_round_trips(self) -> None:
        norm = RunningNorm(1, clip=5.0)
        norm.update(np.array([[0.0], [2.0]]))
        assert norm(np.array([1000.0])).tolist() == [5.0]
        norm.update(np.empty((0, 1)))
        copy = RunningNorm.from_state(norm.state())
        assert copy.mean.tolist() == norm.mean.tolist()
        assert copy.var.tolist() == norm.var.tolist()
        assert copy.count == norm.count


def test_diagonal_gaussian() -> None:
    mean = torch.tensor([[0.5, -1.0]])
    log_std = torch.tensor([[0.0, math.log(2.0)]])
    dist = DiagonalGaussian(mean, log_std)
    x = torch.tensor([[1.5, 1.0]])
    expected = sum(
        -0.5 * ((xi - m) / s) ** 2 - math.log(s) - 0.5 * math.log(2 * math.pi)
        for xi, m, s in ((1.5, 0.5, 1.0), (1.0, -1.0, 2.0))
    )
    assert dist.log_prob(x).item() == pytest.approx(expected)
    entropy = sum(0.5 * math.log(2 * math.pi * math.e * s**2) for s in (1.0, 2.0))
    assert dist.entropy().item() == pytest.approx(entropy)
    samples = DiagonalGaussian(torch.zeros(20_000, 1), torch.zeros(20_000, 1)).sample(
        torch.Generator().manual_seed(0)
    )
    assert samples.mean().item() == pytest.approx(0.0, abs=0.03)
    assert samples.std().item() == pytest.approx(1.0, abs=0.03)


def test_checkpoints_round_trip(toy_agent: PPO, tmp_path: Path) -> None:
    path = toy_agent.save(tmp_path / "nested" / "agent.pt", {"note": "toy"})
    assert path.stat().st_size < 100_000
    loaded, metadata = PPO.load(path)
    assert metadata == {"note": "toy"}
    assert loaded.config == toy_agent.config
    obs = np.array([0.3, -0.7], dtype=np.float32)
    assert np.array_equal(loaded.act(obs), toy_agent.act(obs))
    other = tmp_path / "other.pt"
    torch.save({"format": 99}, other)
    with pytest.raises(ValueError, match="not a slicesim PPO checkpoint"):
        PPO.load(other)


def test_actions_stay_in_the_action_space(toy_agent: PPO) -> None:
    obs = np.array([1.0, 1.0], dtype=np.float32)
    for deterministic in (True, False):
        for _ in range(50):
            action = toy_agent.act(obs, deterministic=deterministic)
            assert action.dtype == np.float32
            assert np.all((action >= -1) & (action <= 1))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"total_steps": 0}, "total_steps"),
        ({"minibatch_size": 4096}, "minibatch_size cannot exceed"),
        ({"gamma": 0.0}, "gamma"),
        ({"clip_range": 0.0}, "clip ranges"),
        ({"learning_rate": 0.0}, "learning_rate"),
    ],
)
def test_config_validation(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PPOConfig(**changes)


def test_needs_box_spaces() -> None:
    env = Target()
    env.action_space = spaces.Discrete(3)
    with pytest.raises(TypeError, match="Box"):
        PPO.for_env(env)
