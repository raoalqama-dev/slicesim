import csv
import io
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from slicesim.cli import main  # noqa: E402
from slicesim.config import load_scenario, parse_scenario  # noqa: E402
from slicesim.rl.env import SliceEnv  # noqa: E402
from slicesim.rl.evaluate import (  # noqa: E402
    HELD_OUT_SEEDS,
    aggregate,
    evaluate,
    run_agent,
    write_csv,
)
from slicesim.rl.ppo import PPO, PPOConfig  # noqa: E402
from slicesim.rl.train import TRAINING_SEEDS, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOW = load_scenario(ROOT / "scenarios" / "low.yaml")


SMALL = PPOConfig(total_steps=120, rollout_steps=40, minibatch_size=20, epochs=2)


def test_train_writes_a_log_and_checkpoints(tmp_path: Path) -> None:
    seen: list[int] = []
    run = train(
        [LOW],
        tmp_path / "run",
        SMALL,
        episode_s=0.1,
        checkpoint_every=2,
        on_update=lambda stats: seen.append(stats.update),
    )
    assert seen == [1, 2, 3]
    assert [p.name for p in run.checkpoints] == ["checkpoint-0002.pt"]
    assert run.checkpoint.name == "agent.pt"
    with run.log.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["env_steps"]) for r in rows] == [40, 80, 120]
    assert {"policy_loss", "value_loss", "entropy", "approx_kl", "mean_return"} <= set(rows[0])
    _, metadata = PPO.load(run.checkpoint)
    assert metadata == {
        "slices": ["embb", "urllc", "mmtc"],
        "levels": ["low"],
        "decision_slots": 10,
        "episode_s": 0.1,
    }
    with pytest.raises(ValueError, match="checkpoint_every"):
        train([LOW], tmp_path, SMALL, checkpoint_every=0)


def test_training_and_evaluation_seeds_do_not_overlap() -> None:
    assert TRAINING_SEEDS[1] <= HELD_OUT_SEEDS
    env = SliceEnv([LOW], episode_s=0.01, scenario_seeds=TRAINING_SEEDS)
    seeds = [env.reset(seed=s)[1]["seed"] for s in range(50)]
    assert all(TRAINING_SEEDS[0] <= seed < TRAINING_SEEDS[1] for seed in seeds)


@pytest.fixture(scope="module")
def small_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return train([LOW], tmp_path_factory.mktemp("small"), SMALL, episode_s=0.1).checkpoint


def test_evaluate_runs_every_policy_on_held_out_seeds(small_checkpoint: Path) -> None:
    rows = evaluate(small_checkpoint, [LOW], seeds=2, duration_s=0.2)
    assert [(r.policy, r.seed) for r in rows] == [
        (policy, HELD_OUT_SEEDS + k)
        for k in range(2)
        for policy in ("ppo", "equal_weights", "static", "proportional", "delay_aware")
    ]
    totals = aggregate(rows)
    assert [t.policy for t in totals] == [
        "ppo",
        "equal_weights",
        "static",
        "proportional",
        "delay_aware",
    ]
    assert all(t.seeds == 2 and t.level == "low" for t in totals)
    for t in totals:
        assert 0 <= t.urllc_met <= 2
        assert t.cell_mbps > 0


def test_equal_weights_runs_the_scenario_seed_it_is_given() -> None:
    scenario = LOW.replace(seed=HELD_OUT_SEEDS, duration_s=0.1)
    first = run_agent(None, scenario, decision_slots=10)
    second = run_agent(None, scenario, decision_slots=10)
    assert first == second
    other = run_agent(None, scenario.replace(seed=HELD_OUT_SEEDS + 1), decision_slots=10)
    assert other != first


def test_evaluate_in_parallel_matches_serial(small_checkpoint: Path) -> None:
    serial = evaluate(small_checkpoint, [LOW], seeds=1, duration_s=0.1, workers=1)
    parallel = evaluate(small_checkpoint, [LOW], seeds=1, duration_s=0.1, workers=2)
    assert serial == parallel


def test_evaluate_checks_the_slices(small_checkpoint: Path) -> None:
    data = LOW.model_dump(exclude_unset=True)
    data["slices"][2]["name"] = "iot"
    for group in data["ues"]:
        if group["slice"] == "mmtc":
            group["slice"] = "iot"
    with pytest.raises(ValueError, match="trained on embb, urllc, mmtc"):
        evaluate(small_checkpoint, [parse_scenario(data)], seeds=1, duration_s=0.1)
    with pytest.raises(ValueError, match="seeds"):
        evaluate(small_checkpoint, [LOW], seeds=0)


def test_write_csv(small_checkpoint: Path, tmp_path: Path) -> None:
    rows = evaluate(small_checkpoint, [LOW], seeds=1, duration_s=0.1)
    path = write_csv(rows, tmp_path / "out" / "eval.csv")
    with path.open(encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == 5
    assert written[0]["policy"] == "ppo"


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_cli_train_and_evaluate(tmp_path: Path) -> None:
    low = str(ROOT / "scenarios" / "low.yaml")
    code, out, err = run_cli(
        "train", low, "--out", str(tmp_path), "--steps", "60", "--episode", "0.05"
    )
    assert (code, err) == (0, "")
    assert out.splitlines()[0].startswith("update    1  steps      60")
    assert out.splitlines()[-1].startswith("wrote ")
    agent = tmp_path / "agent.pt"
    assert agent.exists()

    csv_path = tmp_path / "eval.csv"
    code, out, err = run_cli(
        "evaluate", str(agent), low, "--seeds", "1", "--duration", "0.1", "--csv", str(csv_path)
    )
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0].split()[:3] == ["level", "policy", "cell"]
    assert [re.split(r"\s{2,}", line)[1] for line in lines[2:7]] == [
        "PPO agent",
        "equal weights",
        "static quotas",
        "proportional",
        "delay-aware",
    ]
    assert csv_path.exists()

    code, out, _ = run_cli(
        "evaluate", str(agent), low, "--seeds", "1", "--duration", "0.1", "--json"
    )
    assert code == 0
    assert "urllc_violations" in out


def test_cli_reports_a_missing_checkpoint(tmp_path: Path) -> None:
    low = str(ROOT / "scenarios" / "low.yaml")
    code, _, err = run_cli("evaluate", str(tmp_path / "none.pt"), low)
    assert code == 2
    assert err.startswith("slicesim: ")


def test_cli_explains_a_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "slicesim.rl.train", None)
    code, _, err = run_cli("train", str(ROOT / "scenarios" / "low.yaml"))
    assert code == 2
    assert "needs the rl extra" in err
    assert 'pip install "slicesim[rl]"' in err


def test_cli_explains_a_missing_extra_for_evaluate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "slicesim.rl.evaluate", None)
    code, _, err = run_cli("evaluate", "agent.pt", str(ROOT / "scenarios" / "low.yaml"))
    assert code == 2
    assert "evaluate needs the rl extra" in err


def test_cli_reports_training_errors(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    code, _, err = run_cli(
        "train", str(ROOT / "scenarios" / "low.yaml"), "--out", str(blocker / "run")
    )
    assert code == 2
    assert err.startswith("slicesim: ")


@pytest.mark.parametrize(
    ("value", "message"), [("0", "must be at least 1"), ("x", "not an integer")]
)
def test_cli_rejects_bad_counts(
    value: str, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        main(["train", "x.yaml", "--steps", value])
    assert info.value.code == 2
    assert message in capsys.readouterr().err
