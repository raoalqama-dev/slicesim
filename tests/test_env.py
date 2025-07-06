from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium import spaces  # noqa: E402
from gymnasium.utils.env_checker import check_env  # noqa: E402

from factories import scenario_data  # noqa: E402
from slicesim.config import Scenario, load_scenario, parse_scenario  # noqa: E402
from slicesim.rl.env import (  # noqa: E402
    ENV_ID,
    FEATURES,
    LOGIT_SCALE,
    RewardConfig,
    SliceEnv,
    softmax,
)

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"
LEVELS = [load_scenario(SCENARIOS / f"{name}.yaml") for name in ("low", "medium", "high")]


def run_episode(
    env: SliceEnv, seed: int, actions: npt.NDArray[np.float32] | None = None
) -> tuple[list[npt.NDArray[np.float32]], list[float], list[dict[str, Any]]]:
    obs, _ = env.reset(seed=seed)
    observations, rewards, infos = [obs], [], []
    rng = np.random.default_rng(0)
    done = False
    step = 0
    while not done:
        action = (
            actions[step]
            if actions is not None
            else rng.uniform(-1, 1, size=len(env.slice_names)).astype(np.float32)
        )
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        rewards.append(float(reward))
        infos.append(info)
        done = terminated or truncated
        step += 1
    return observations, rewards, infos


def test_passes_the_gymnasium_environment_checker() -> None:
    env = gym.make(ENV_ID, levels=LEVELS, episode_s=0.05)
    assert env.spec is not None
    assert env.spec.id == "slicesim/Slicing-v0"
    check_env(env.unwrapped)


def test_spaces() -> None:
    env = SliceEnv(LEVELS)
    assert env.action_space.shape == (3,)
    assert env.observation_space.shape == (15,)
    assert len(FEATURES) == 5
    assert env.slice_names == ["embb", "urllc", "mmtc"]
    assert env.sla_types == ["embb", "urllc", "mmtc"]
    space = env.observation_space
    assert isinstance(space, spaces.Box)
    low = space.low.reshape(3, 5)
    assert np.all(low[:, :4] == 0)
    assert np.all(low[:, 4] == -1)


def test_episodes_replay_exactly_from_a_seed() -> None:
    env = SliceEnv(LEVELS, episode_s=0.1)
    obs_a, rewards_a, _ = run_episode(env, seed=11)
    obs_b, rewards_b, _ = run_episode(SliceEnv(LEVELS, episode_s=0.1), seed=11)
    assert rewards_a == rewards_b
    assert all(np.array_equal(a, b) for a, b in zip(obs_a, obs_b, strict=True))
    _, rewards_c, _ = run_episode(env, seed=12)
    assert rewards_c != rewards_a


def test_each_reset_draws_a_load_level() -> None:
    env = SliceEnv(LEVELS, episode_s=0.01)
    names = [env.reset(seed=seed)[1]["level"] for seed in range(30)]
    assert set(names) == {"low", "medium", "high"}
    _, info = env.reset(seed=1, options={"level": 2})
    assert info["level"] == "high"
    with pytest.raises(ValueError, match="level must be 0 to 2"):
        env.reset(options={"level": 3})


def test_an_episode_lasts_its_duration() -> None:
    env = SliceEnv(LEVELS, decision_slots=10, episode_s=0.1)
    observations, rewards, infos = run_episode(env, seed=3)
    # 0.1 s of 0.5 ms slots is 200 slots: 20 decisions of 10 slots.
    assert len(rewards) == 20
    assert infos[-1]["slot"] == 200
    for obs in observations:
        assert env.observation_space.contains(obs)


def test_the_first_observation_is_an_idle_cell() -> None:
    env = SliceEnv(LEVELS)
    obs, _ = env.reset(seed=0)
    per_slice = obs.reshape(3, 5)
    assert np.all(per_slice[:, :3] == 0)  # nothing queued, waiting or delivered yet
    assert np.all(per_slice[:, 4] == 1)  # all slack
    assert np.all((per_slice[:, 3] > 0) & (per_slice[:, 3] <= 1))


def test_actions_set_the_slice_weights() -> None:
    env = SliceEnv(LEVELS, episode_s=0.05)
    env.reset(seed=4, options={"level": 2})
    _, _, _, _, info = env.step(np.array([1.0, -1.0, -1.0], dtype=np.float32))
    expected = softmax(LOGIT_SCALE * np.array([1.0, -1.0, -1.0]))
    assert np.allclose(info["weights"], expected)
    assert info["weights"][0] > 0.99
    # Out-of-range actions are clipped to [-1, 1].
    _, _, _, _, info = env.step(np.array([7.0, -7.0, 0.0], dtype=np.float32))
    assert np.allclose(info["weights"], softmax(LOGIT_SCALE * np.array([1.0, -1.0, 0.0])))


def test_starving_a_slice_shows_in_its_observation_and_reward() -> None:
    env = SliceEnv(LEVELS, episode_s=0.1)
    env.reset(seed=2, options={"level": 1})
    starve_urllc = np.array([1.0, -1.0, 1.0], dtype=np.float32)
    serve_urllc = np.array([-1.0, 1.0, -1.0], dtype=np.float32)
    totals = {}
    for name, action in (("starve", starve_urllc), ("serve", serve_urllc)):
        env.reset(seed=2, options={"level": 1})
        total = 0.0
        violations = np.zeros(3)
        for _ in range(20):
            obs, reward, _, _, info = env.step(action)
            total += float(reward)
            violations += info["violations"]
        totals[name] = (total, violations, obs.reshape(3, 5))
    # Weight e^-10 relative to eMBB still leaves URLLC some PRBs, but too few: its
    # packets go past their deadline, the slack goes negative and the reward falls.
    assert totals["starve"][1][1] > 0
    assert totals["serve"][1][1] == 0
    assert totals["starve"][0] < totals["serve"][0]
    assert totals["starve"][2][1, 4] < totals["serve"][2][1, 4]


def test_episode_violations_match_the_sla_accounting() -> None:
    env = SliceEnv(LEVELS, episode_s=0.5)
    actions = np.tile(np.array([1.0, -1.0, 0.0], dtype=np.float32), (100, 1))
    _, _, infos = run_episode(env, seed=7, actions=actions)
    counted = np.sum([info["violations"] for info in infos], axis=0)
    assert env.sim is not None
    result = env.sim.result()
    urllc = result.ue_slice == 1
    expected_urllc = (
        result.packets_late[urllc].sum()
        + result.packets_lost[urllc].sum()
        + result.overdue_packets[urllc].sum()
    )
    assert counted[1] == expected_urllc > 0
    summary = result.summary().slices
    embb = summary[0]
    assert embb.violation_ratio is not None
    windows = result.n_slots // result.window_slots
    owed_ue_windows = windows * int((result.ue_slice == 0).sum())
    assert counted[0] <= owed_ue_windows
    assert counted[0] == pytest.approx(embb.violation_ratio * _counted_embb(result), abs=1e-9)


def _counted_embb(result: Any) -> int:
    from slicesim.slicing.sla import embb_window_counts

    complete = result.n_slots // result.window_slots
    members = result.ue_slice == 0
    return embb_window_counts(
        result.window_delivered[:complete, members],
        result.window_offered[:complete, members],
        result.window_backlog[:complete, members],
        result.full_buffer[members],
        2e6 * 0.1,
    ).counted


def test_reward_weights_change_the_reward() -> None:
    action = np.zeros(3, dtype=np.float32)

    def first_reward(reward: RewardConfig | None) -> float:
        env = SliceEnv(LEVELS, episode_s=0.05, reward=reward)
        env.reset(seed=1)
        return float(env.step(action)[1])

    default = first_reward(None)
    doubled = first_reward(
        RewardConfig(
            satisfaction={"embb": 2.0, "urllc": 2.0, "mmtc": 1.0},
            penalty={"embb": 0.4, "urllc": 2.0, "mmtc": 0.2},
        )
    )
    assert doubled == pytest.approx(2 * default)


def test_the_fixed_link_reports_full_cqi() -> None:
    data = LEVELS[0].model_dump(exclude_unset=True)
    data["link"] = {"model": "fixed", "bits_per_prb": 400}
    env = SliceEnv([parse_scenario(data)], episode_s=0.02)
    obs, _ = env.reset(seed=0)
    assert np.all(obs.reshape(3, 5)[:, 3] == 1.0)
    obs, *_ = env.step(np.zeros(3, dtype=np.float32))
    assert env.observation_space.contains(obs)


def test_softmax() -> None:
    assert np.allclose(softmax([0.0, 0.0]), [0.5, 0.5])
    assert np.allclose(softmax([1000.0, 0.0]), [1.0, 0.0])
    assert softmax([3.0, 1.0, -2.0]).sum() == pytest.approx(1.0)


def unsliced() -> Scenario:
    return parse_scenario(scenario_data())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"levels": []}, "at least one scenario"),
        ({"levels": [unsliced()]}, "is not sliced"),
        ({"decision_slots": 0}, "decision_slots"),
        ({"episode_s": 0.0}, "episode_s"),
        (
            {"reward": RewardConfig(satisfaction={"embb": 1.0}, penalty={})},
            "reward weights missing for mmtc, urllc",
        ),
    ],
)
def test_invalid_settings(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SliceEnv(**{"levels": LEVELS, **kwargs})


def test_levels_need_the_same_slices() -> None:
    data = LEVELS[0].model_dump(exclude_unset=True)
    data["name"] = "renamed"
    data["slices"][2]["name"] = "iot"
    for group in data["ues"]:
        if group["slice"] == "mmtc":
            group["slice"] = "iot"
    with pytest.raises(ValueError, match="renamed has different slices from low"):
        SliceEnv([LEVELS[0], parse_scenario(data)])


def test_step_needs_a_reset_and_the_right_action() -> None:
    env = SliceEnv(LEVELS)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(3, dtype=np.float32))
    env.reset(seed=0)
    with pytest.raises(ValueError, match="expected 3 logits"):
        env.step(np.zeros(2, dtype=np.float32))
