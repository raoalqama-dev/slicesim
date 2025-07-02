import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from factories import scenario, scenario_data
from slicesim import __version__
from slicesim.cli import main
from slicesim.engine import simulate
from slicesim.report import format_table, render_comparison, render_summary

BASIC = Path(__file__).resolve().parent.parent / "scenarios" / "basic.yaml"


def run(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def small(tmp_path: Path) -> Path:
    path = tmp_path / "small.yaml"
    path.write_text(
        yaml.safe_dump(scenario_data(duration_s=0.2, description="a small cell")), encoding="utf-8"
    )
    return path


def test_run_prints_a_summary_table(small: Path) -> None:
    code, out, err = run("run", str(small))
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0] == "test: a small cell"
    assert "  seed      7" in lines
    assert "  duration  0.2 s (400 slots)" in lines
    assert "  carrier   20 MHz at 30 kHz (numerology 1): 51 PRBs, 0.5 ms slots" in lines
    assert "  link      fixed, 500 bits per PRB, no decoding errors" in lines
    assert any(line.startswith("  result    ") and "of 51.0 peak" in line for line in lines)
    header = next(line for line in lines if line.startswith("group"))
    assert header.split() == [
        "group",
        "UEs",
        "traffic",
        "offered",
        "delivered",
        "per",
        "UE",
        "Jain",
        "SINR",
        "CQI",
        "packets",
        "lost",
        "delay",
        "p50",
        "p95",
        "p99",
        "buffer",
    ]
    table = lines[lines.index(header) + 2 : lines.index(header) + 6]
    rows = {line.split()[0]: line.split() for line in table}
    assert set(rows) == {"web", "ctrl", "video", "bulk"}
    assert rows["web"][1:3] == ["4", "poisson"]
    # The fixed link has no SINR or CQI; full-buffer UEs have no packets.
    assert rows["web"][7:9] == ["-", "-"]
    assert 0 < float(rows["web"][6]) <= 1  # Jain
    assert "  scheduler round robin" in lines
    assert rows["bulk"][3] == "-"
    assert rows["bulk"][-1] == "-"
    assert lines[-1].startswith("Throughput in Mbit/s;")


def test_run_over_the_radio_channel(tmp_path: Path) -> None:
    path = tmp_path / "radio.yaml"
    data = scenario_data(duration_s=0.2, link={"model": "adaptive"})
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    code, out, _ = run("run", str(path))
    assert code == 0
    lines = out.splitlines()
    assert (
        "  cell      UMa at 3.5 GHz, 300 m radius, 46 dBm, shadowing, Rayleigh fading "
        "(40 ms coherence)"
    ) in lines
    assert (
        "  link      CQI-adaptive at 10% BLER, 14% overhead, CQI 2 slots old, HARQ every 8 "
        "slots up to 4 transmissions"
    ) in lines
    result = next(line for line in lines if line.startswith("  result"))
    assert "BLER" in result
    assert "of blocks dropped" in result
    web = next(line for line in lines if line.startswith("web")).split()
    assert float(web[7]) > 0  # mean SINR
    assert 0 < float(web[8]) <= 15  # mean CQI

    data["channel"] = {"shadowing": False, "fading": "none"}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    _, out, _ = run("run", str(path))
    assert "no shadowing, no fading" in out


def test_run_is_reproducible(small: Path) -> None:
    assert run("run", str(small)) == run("run", str(small))


def test_json_output_and_overrides(small: Path) -> None:
    code, out, _ = run("run", str(small), "--json")
    assert code == 0
    base = json.loads(out)
    assert base["seed"] == 7
    assert base["cell"]["n_slots"] == 400

    _, out, _ = run("run", str(small), "--json", "--seed", "8", "--duration", "0.1")
    changed = json.loads(out)
    assert changed["seed"] == 8
    assert changed["cell"]["n_slots"] == 200
    assert changed["groups"][0]["packets_arrived"] != base["groups"][0]["packets_arrived"]


def test_invalid_scenario_exits_with_usage_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(scenario_data(seed=-1, colour="red")), encoding="utf-8")
    code, out, err = run("run", str(path))
    assert code == 2
    assert out == ""
    assert err == (
        f"slicesim: {path}: invalid scenario\n"
        "  seed: Input should be greater than or equal to 0 (got -1)\n"
        "  colour: unknown field\n"
    )


def test_an_override_that_breaks_the_scenario_names_the_file(small: Path) -> None:
    code, _, err = run("run", str(small), "--duration", "0.0001")
    assert code == 2
    assert err.startswith(f"slicesim: {small}: invalid scenario\n")
    assert "duration_s must cover at least one slot" in err


def test_missing_file(tmp_path: Path) -> None:
    code, _, err = run("run", str(tmp_path / "nope.yaml"))
    assert code == 2
    assert "cannot read file" in err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "x.yaml", "--seed", "-4"], "must be non-negative"),
        (["run", "x.yaml", "--seed", "abc"], "not an integer: 'abc'"),
        (["run", "x.yaml", "--duration", "0"], "must be positive"),
        (["run", "x.yaml", "--duration", "soon"], "not a number: 'soon'"),
        ([], "the following arguments are required: COMMAND"),
        (["fly"], "invalid choice: 'fly'"),
    ],
)
def test_bad_arguments_exit_with_usage_errors(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    assert message in capsys.readouterr().err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out == f"slicesim {__version__}\n"


def test_the_example_scenario_runs_as_a_module() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "slicesim", "run", str(BASIC), "--duration", "0.05"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("basic: four traffic types sharing a 20 MHz cell\n")


def test_format_table_aligns_columns() -> None:
    table = format_table(["name", "n"], [["a", "1"], ["long name", "100"]])
    assert table.splitlines() == [
        "name         n",
        "---------  ---",
        "a            1",
        "long name  100",
    ]
    assert format_table(["n", "name"], [["1", "ab"]], align="rl").splitlines() == [
        "n  name",
        "-  ----",
        "1  ab",
    ]
    with pytest.raises(ValueError, match="one cell per header"):
        format_table(["a", "b"], [["only one"]])
    with pytest.raises(ValueError, match="one 'l' or 'r' per column"):
        format_table(["a", "b"], [], align="lc")


def test_render_summary_without_a_description_or_deliveries() -> None:
    s = scenario(
        duration_s=0.001,
        ues=[
            {
                "name": "rare",
                "count": 1,
                "traffic": {"model": "periodic", "period_ms": 1000, "packet_bytes": 10},
            }
        ],
    )
    text = render_summary(simulate(s).summary(), s)
    assert text.splitlines()[0] == "test"
    row = next(line for line in text.splitlines() if line.startswith("rare")).split()
    # Nothing was delivered yet, so there are no delay statistics.
    assert row[11:15] == ["-", "-", "-", "-"]


FAIRNESS = Path(__file__).resolve().parent.parent / "scenarios" / "fairness.yaml"


def test_compare_prints_one_row_per_scheduler() -> None:
    code, out, err = run("compare", str(FAIRNESS), "--duration", "0.2")
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert lines[0] == "fairness: twenty full-buffer UEs across a 500 m cell"
    header = next(line for line in lines if line.startswith("scheduler"))
    assert header.split() == [
        "scheduler",
        "delivered",
        "Jain",
        "edge",
        "UE",
        "median",
        "UE",
        "delay",
        "p50",
        "p95",
        "p99",
        "buffer",
        "BLER",
    ]
    start = lines.index(header) + 2
    names = [line.split("  ")[0].strip() for line in lines[start : start + 3]]
    assert names == ["round robin", "max C/I", "proportional fair (100 ms average)"]
    assert lines[-1].startswith("Throughput in Mbit/s;")


def test_compare_runs_the_schedulers_named(small: Path) -> None:
    code, out, _ = run("compare", str(small), "--schedulers", "pf, maxci", "--json")
    assert code == 0
    data = json.loads(out)
    assert [entry["scheduler"]["model"] for entry in data] == ["pf", "maxci"]
    assert data[0]["scheduler"]["time_constant_ms"] == 100
    assert data[0]["seed"] == data[1]["seed"] == 7
    assert data[0]["cell"]["jain"] is not None
    # The fixed link never fails, so the BLER column is empty.
    _, out, _ = run("compare", str(small), "--schedulers", "rr")
    row = next(line for line in out.splitlines() if line.startswith("round robin"))
    assert row.split()[-1] == "-"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("rr,edf", "unknown scheduler 'edf'; choose from rr, maxci, pf"),
        ("rr,rr", "each scheduler may be named once"),
        (",", "name at least one scheduler"),
    ],
)
def test_compare_rejects_bad_scheduler_lists(
    value: str, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as info:
        main(["compare", "x.yaml", "--schedulers", value])
    assert info.value.code == 2
    assert message in capsys.readouterr().err


def test_compare_reports_an_invalid_scenario(tmp_path: Path) -> None:
    code, _, err = run("compare", str(tmp_path / "missing.yaml"))
    assert code == 2
    assert "cannot read file" in err


def test_render_comparison_needs_runs() -> None:
    with pytest.raises(ValueError, match="nothing to compare"):
        render_comparison([])


MEDIUM = Path(__file__).resolve().parent.parent / "scenarios" / "medium.yaml"


@pytest.fixture
def sliced_file(tmp_path: Path) -> Path:
    data = yaml.safe_load(MEDIUM.read_text(encoding="utf-8"))
    data["duration_s"] = 0.3
    path = tmp_path / "sliced.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_run_prints_a_slice_table(sliced_file: Path) -> None:
    code, out, err = run("run", str(sliced_file))
    assert (code, err) == (0, "")
    lines = out.splitlines()
    assert "  slicing   delay-aware policy" in lines
    assert (
        "  scheduler embb proportional fair (100 ms average); urllc round robin; mmtc round robin"
    ) in lines
    header = next(line for line in lines if line.startswith("slice "))
    assert header.split()[:4] == ["slice", "UEs", "PRBs", "delivered"]
    start = lines.index(header) + 2
    rows = [line.split() for line in lines[start : start + 3]]
    assert [row[0] for row in rows] == ["embb", "urllc", "mmtc"]
    assert any(line.endswith("urllc: 0.999 of packets within 5 ms") for line in lines)


def test_run_can_override_the_policy(sliced_file: Path) -> None:
    code, out, _ = run("run", str(sliced_file), "--policy", "static")
    assert code == 0
    assert "  slicing   static quotas policy (embb 80%, urllc 15%, mmtc 5%)" in out
    _, out, _ = run("run", str(sliced_file), "--policy", "proportional", "--json")
    data = json.loads(out)
    assert [s["name"] for s in data["slices"]] == ["embb", "urllc", "mmtc"]
    assert {"violation_ratio", "met", "satisfaction", "prb_share"} <= set(data["slices"][0])


def test_a_policy_needs_slices(small: Path) -> None:
    code, _, err = run("run", str(small), "--policy", "static")
    assert code == 2
    assert err == f"slicesim: {small}: invalid scenario\n  slicing needs slices\n"


def test_compare_on_a_sliced_cell_sets_every_slices_scheduler(sliced_file: Path) -> None:
    code, out, _ = run("compare", str(sliced_file), "--schedulers", "maxci,pf")
    assert code == 0
    assert "max C/I" in out
    assert "embb proportional fair (100 ms average); urllc proportional fair" in out
