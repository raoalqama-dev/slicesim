import csv
import io
import sys
from pathlib import Path

import numpy as np
import pytest

from factories import scenario_data
from slicesim.cli import main
from slicesim.config import load_scenario, parse_scenario
from slicesim.experiments.runner import (
    DELAY_QUANTILES,
    HELD_OUT_SEEDS,
    HEURISTICS,
    RunResult,
    run_experiment,
    run_policy,
    summarize_run,
    write_results,
)

ROOT = Path(__file__).resolve().parent.parent
LEVELS = [load_scenario(ROOT / "scenarios" / f"{name}.yaml") for name in ("low", "medium")]
MODEL = ROOT / "models" / "ppo-slicing.pt"


@pytest.fixture(scope="module")
def small() -> RunResult:
    return run_experiment(LEVELS, HEURISTICS, seeds=(1, 2), duration_s=0.2)


def test_every_level_policy_seed_and_slice_gets_a_row(small: RunResult) -> None:
    keys = [(r.level, r.seed, r.policy, r.slice) for r in small.slices]
    assert keys == [
        (level, seed, policy, slice_name)
        for level in ("low", "medium")
        for seed in (1, 2)
        for policy in HEURISTICS
        for slice_name in ("embb", "urllc", "mmtc")
    ]
    for row in small.slices:
        assert 0 <= row.prb_share <= 1
        assert row.delivered_mbps >= 0
        assert row.sla in ("embb", "urllc", "mmtc")


def test_policies_see_the_same_traffic(small: RunResult) -> None:
    by_policy = {
        r.policy: (r.ues, r.packets) for r in small.slices if r.level == "low" and r.seed == 1
    }
    assert len({ues for ues, _ in by_policy.values()}) == 1


def test_delay_quantiles_trace_each_slices_distribution(small: RunResult) -> None:
    rows = [
        d
        for d in small.delays
        if (d.level, d.seed, d.policy, d.slice) == ("low", 1, "static", "urllc")
    ]
    assert [d.quantile for d in rows] == list(DELAY_QUANTILES)
    delays = [d.delay_ms for d in rows]
    assert delays == sorted(delays)
    assert DELAY_QUANTILES[0] == 0.0
    assert DELAY_QUANTILES[-1] == 1.0


def test_the_run_rows_match_the_runs_own_summary() -> None:
    scenario = LEVELS[0].replace(seed=4, duration_s=0.2)
    result = run_policy(scenario, "delay_aware")
    rows = summarize_run("low", "delay_aware", 4, result)
    summary = result.summary().slices
    for row, expected in zip(rows.slices, summary, strict=True):
        assert row.slice == expected.name
        assert row.violation_ratio == expected.violation_ratio
        assert row.met == expected.met
        assert row.packets == expected.packets_delivered
    urllc = [d for d in rows.delays if d.slice == "urllc"]
    packet_delays = result.delay_s[result.ue_slice[result.delay_ue] == 1] * 1e3
    # Delays are recorded to 0.1 microsecond.
    assert urllc[-1].delay_ms == pytest.approx(float(np.max(packet_delays)), abs=5e-5)


def test_parallel_runs_match_serial_ones() -> None:
    serial = run_experiment(LEVELS[:1], ("static",), seeds=(3,), duration_s=0.1)
    parallel = run_experiment(
        LEVELS[:1], ("static", "proportional"), seeds=(3,), duration_s=0.1, workers=2
    )
    assert parallel.slices[:3] == serial.slices
    assert parallel.delays[: len(serial.delays)] == serial.delays


def test_the_default_seeds_are_held_out() -> None:
    result = run_experiment(LEVELS[:1], ("static",), duration_s=0.02)
    assert sorted({r.seed for r in result.slices}) == [
        HELD_OUT_SEEDS,
        HELD_OUT_SEEDS + 1,
        HELD_OUT_SEEDS + 2,
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"policies": ("random",)}, "unknown policy 'random'"),
        ({"policies": ("ppo",)}, "the ppo policy needs a checkpoint"),
        ({"seeds": ()}, "at least one seed"),
        ({"levels": [parse_scenario(scenario_data())]}, "not sliced: test"),
    ],
)
def test_invalid_experiments(kwargs: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {"levels": LEVELS, **kwargs}
    with pytest.raises(ValueError, match=message):
        run_experiment(**arguments)  # type: ignore[arg-type]


def test_run_policy_rejects_unknown_policies() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        run_policy(LEVELS[0], "random")
    with pytest.raises(ValueError, match="needs a checkpoint"):
        run_policy(LEVELS[0].replace(duration_s=0.01), "ppo")


def test_write_results(small: RunResult, tmp_path: Path) -> None:
    runs, delays = write_results(small, tmp_path / "out")
    with runs.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(small.slices)
    assert {"level", "policy", "seed", "slice", "violation_ratio", "p99_ms"} <= set(rows[0])
    with delays.open(encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == len(small.delays)


def test_learned_policies_run_when_the_rl_extra_is_installed() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("gymnasium")
    result = run_experiment(
        LEVELS[:1], ("equal_weights", "ppo"), seeds=(5,), duration_s=0.1, checkpoint=MODEL
    )
    assert {r.policy for r in result.slices} == {"equal_weights", "ppo"}


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_cli_experiment(tmp_path: Path) -> None:
    low = str(ROOT / "scenarios" / "low.yaml")
    code, out, err = run_cli(
        "experiment", low, "--policies", "static,delay_aware", "--seeds", "2",
        "--duration", "0.05", "--out", str(tmp_path),
    )  # fmt: skip
    assert (code, err) == (0, "")
    assert out.endswith("(12 slice rows)\n")
    with (tmp_path / "runs.csv").open(encoding="utf-8") as f:
        seeds = sorted({int(r["seed"]) for r in csv.DictReader(f)})
    assert seeds == [HELD_OUT_SEEDS, HELD_OUT_SEEDS + 1]
    code, _, _ = run_cli(
        "experiment", low, "--policies", "static", "--first-seed", "7", "--seeds", "1",
        "--duration", "0.05", "--out", str(tmp_path),
    )  # fmt: skip
    assert code == 0
    with (tmp_path / "runs.csv").open(encoding="utf-8") as f:
        assert {int(r["seed"]) for r in csv.DictReader(f)} == {7}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("static,random", "unknown policy 'random'"),
        ("static,static", "each policy may be named once"),
        (",", "name at least one policy"),
    ],
)
def test_cli_experiment_rejects_bad_policy_lists(
    value: str, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(["experiment", "x.yaml", "--policies", value])
    assert message in capsys.readouterr().err


def test_cli_experiment_reports_errors(tmp_path: Path) -> None:
    low = str(ROOT / "scenarios" / "low.yaml")
    code, _, err = run_cli("experiment", low, "--policies", "ppo", "--out", str(tmp_path))
    assert code == 2
    assert err == "slicesim: the ppo policy needs a checkpoint\n"


def test_cli_experiment_explains_a_missing_rl_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "slicesim.rl.evaluate", None)
    low = str(ROOT / "scenarios" / "low.yaml")
    code, _, err = run_cli(
        "experiment",
        low,
        "--policies",
        "equal_weights",
        "--duration",
        "0.01",
        "--out",
        str(tmp_path),
    )
    assert code == 2
    assert "experiment needs the rl extra (equal_weights:" in err


class TestPlots:
    @pytest.fixture(autouse=True)
    def _needs_matplotlib(self) -> None:
        pytest.importorskip("matplotlib")

    def test_make_figures(self, small: RunResult, tmp_path: Path) -> None:
        from slicesim.experiments.plots import make_figures

        write_results(small, tmp_path / "results")
        paths = make_figures(tmp_path / "results", tmp_path / "figures")
        assert [p.name for p in paths] == [
            "sla_vs_load.png",
            "delay_cdf.png",
            "throughput_per_slice.png",
        ]
        for path in paths:
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_delay_cdf_uses_the_first_level_without_a_medium_one(self, tmp_path: Path) -> None:
        from slicesim.experiments.plots import make_figures

        result = run_experiment(LEVELS[:1], ("static",), seeds=(1,), duration_s=0.1)
        write_results(result, tmp_path)
        assert len(make_figures(tmp_path, tmp_path / "figures")) == 3

    def test_errors(self, tmp_path: Path) -> None:
        from slicesim.experiments.plots import make_figures, plot_delay_cdf

        with pytest.raises(ValueError, match="no urllc delays at medium load"):
            plot_delay_cdf([], tmp_path)
        write_results(RunResult(slices=[], delays=[]), tmp_path)
        with pytest.raises(ValueError, match="has no runs"):
            make_figures(tmp_path, tmp_path / "figures")

    def test_cli_plot(self, small: RunResult, tmp_path: Path) -> None:
        write_results(small, tmp_path / "results")
        code, out, err = run_cli("plot", str(tmp_path / "results"), "--out", str(tmp_path / "f"))
        assert (code, err) == (0, "")
        assert len(out.splitlines()) == 3
        code, _, err = run_cli("plot", str(tmp_path / "missing"))
        assert code == 2
        assert err.startswith("slicesim: ")


def test_cli_plot_explains_a_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "slicesim.experiments.plots", None)
    code, _, err = run_cli("plot", "results")
    assert code == 2
    assert "needs the plots extra" in err
    assert 'pip install "slicesim[plots]"' in err
