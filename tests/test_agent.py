from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
gym = pytest.importorskip("gymnasium")

from slicesim.compare import with_policy  # noqa: E402
from slicesim.config import load_scenario  # noqa: E402
from slicesim.engine import simulate  # noqa: E402
from slicesim.rl.evaluate import HELD_OUT_SEEDS, run_agent  # noqa: E402
from slicesim.rl.ppo import PPO  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

MODEL = ROOT / "models" / "ppo-slicing.pt"


def test_the_committed_agent_beats_static_quotas_on_urllc() -> None:
    """The spec's done-when, on the medium scenario with a held-out seed."""
    assert MODEL.stat().st_size < 1_000_000
    agent, metadata = PPO.load(MODEL)
    assert metadata["slices"] == ["embb", "urllc", "mmtc"]
    medium = load_scenario(ROOT / "scenarios" / "medium.yaml").replace(seed=HELD_OUT_SEEDS)
    learned = run_agent(agent, medium, int(metadata["decision_slots"])).slices[1]
    static = simulate(with_policy(medium, "static")).summary().slices[1]
    assert learned.name == static.name == "urllc"
    assert learned.violation_ratio is not None
    assert static.violation_ratio is not None
    assert learned.violation_ratio < static.violation_ratio / 2
