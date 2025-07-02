from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from factories import poisson_group, scenario, scenario_data
from slicesim.config import (
    AdaptiveLink,
    CellConfig,
    ChannelConfig,
    EmbbSla,
    FixedLink,
    OnOffTraffic,
    PoissonTraffic,
    RoundRobinScheduler,
    ScenarioError,
    UrllcSla,
    load_scenario,
    parse_scenario,
)

SCENARIOS = Path(__file__).resolve().parent.parent / "scenarios"


def problems(data: Any) -> list[str]:
    with pytest.raises(ScenarioError) as info:
        parse_scenario(data, "s.yaml")
    assert info.value.source == "s.yaml"
    return info.value.problems


@pytest.mark.parametrize("path", sorted(SCENARIOS.glob("*.yaml")), ids=lambda p: p.name)
def test_example_scenarios_are_valid(path: Path) -> None:
    loaded = load_scenario(path)
    assert loaded.n_slots > 0


def test_a_scenario_parses_into_typed_models() -> None:
    s = scenario()
    assert s.carrier.build().n_prb == 51
    assert isinstance(s.ues[0].traffic, PoissonTraffic)
    assert isinstance(s.ues[2].traffic, OnOffTraffic)
    assert s.link == FixedLink(bits_per_prb=500)


def test_the_radio_channel_is_the_default() -> None:
    data = scenario_data()
    del data["link"]
    s = parse_scenario(data)
    assert s.link == AdaptiveLink(
        model="adaptive", overhead=0.14, cqi_delay_slots=2, harq_rtt_slots=8, max_transmissions=4
    )
    assert s.cell == CellConfig(
        radius_m=300,
        min_distance_m=35,
        ue_height_m=1.5,
        frequency_ghz=3.5,
        tx_power_dbm=46,
        antenna_gain_db=0,
        noise_figure_db=7,
        interference_margin_db=3,
    )
    assert s.channel == ChannelConfig(shadowing=True, fading="rayleigh", coherence_ms=40)
    assert s.scheduler == RoundRobinScheduler()


def test_a_ring_of_ues_is_allowed() -> None:
    s = scenario(cell={"radius_m": 15, "min_distance_m": 15})
    assert s.cell.radius_m == s.cell.min_distance_m == 15


def test_the_fixed_link_takes_a_rate() -> None:
    s = scenario(link={"model": "fixed", "bits_per_prb": 640})
    assert s.link == FixedLink(bits_per_prb=640)


@pytest.mark.parametrize(
    ("duration_s", "scs", "slots"),
    [(10, 30, 20_000), (0.3, 15, 300), (0.0015, 15, 1), (1, 60, 4000)],
)
def test_duration_becomes_whole_slots(duration_s: float, scs: int, slots: int) -> None:
    s = scenario(duration_s=duration_s, carrier={"bandwidth_mhz": 20, "scs_khz": scs})
    assert s.n_slots == slots


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        (
            {"carrier": {"bandwidth_mhz": 100, "scs_khz": 15}},
            "carrier: a 100 MHz channel is not defined at 15 kHz subcarrier spacing "
            "(TS 38.101-1 Table 5.3.2-1); choose one of 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 MHz",
        ),
        (
            {"carrier": {"bandwidth_mhz": 20, "scs_khz": 120}},
            "carrier: subcarrier spacing 120 kHz is not supported in FR1; choose one of 15, 30, "
            "60 kHz",
        ),
        (
            {"ues": [{"name": "a", "count": 1, "traffic": {"model": "gaussian"}}]},
            "ues[0].traffic: unknown 'model' 'gaussian'; expected one of 'full_buffer', "
            "'poisson', 'periodic', 'on_off'",
        ),
        (
            {"ues": [{"name": "a", "count": 1, "traffic": {"rate_pps": 3}}]},
            "ues[0].traffic: 'model' is required",
        ),
        (
            {
                "ues": [
                    {
                        "name": "a",
                        "count": 1,
                        "traffic": {
                            "model": "periodic",
                            "period_ms": 1,
                            "packet_bytes": 1,
                            "jitter_ms": 2,
                        },
                    }
                ]
            },
            "ues[0].traffic.jitter_ms: unknown field",
        ),
        (
            {"ues": [poisson_group("a"), poisson_group("b"), poisson_group("a")]},
            "UE group names must be unique; repeated: a",
        ),
        ({"duration_s": 0.0001}, "duration_s must cover at least one slot (0.5 ms at 30 kHz)"),
        ({"duration_s": -1}, "duration_s: Input should be greater than 0 (got -1)"),
        ({"ues": []}, "ues: List should have at least 1 item after validation, not 0"),
        (
            {"link": {"model": "turbo"}},
            "link: unknown 'model' 'turbo'; expected one of 'adaptive', 'fixed'",
        ),
        ({"link": {"bits_per_prb": 600}}, "link: 'model' is required"),
        (
            {"link": {"model": "fixed", "bits_per_prb": 0}},
            "link.bits_per_prb: Input should be greater than 0 (got 0)",
        ),
        (
            {"link": {"model": "adaptive", "max_transmissions": 0}},
            "link.max_transmissions: Input should be greater than or equal to 1 (got 0)",
        ),
        (
            {"link": {"model": "adaptive", "bits_per_prb": 600}},
            "link.bits_per_prb: unknown field",
        ),
        (
            {"cell": {"radius_m": 30}},
            "cell: radius_m must be at least min_distance_m",
        ),
        (
            {"cell": {"ue_height_m": 20}},
            "cell.ue_height_m: Input should be less than or equal to 13 (got 20)",
        ),
        (
            {"scheduler": {"model": "edf"}},
            "scheduler: unknown 'model' 'edf'; expected one of 'rr', 'maxci', 'pf'",
        ),
        (
            {"scheduler": {"model": "pf", "time_constant_ms": 0}},
            "scheduler.time_constant_ms: Input should be greater than 0 (got 0)",
        ),
        (
            {"scheduler": {"model": "rr", "time_constant_ms": 50}},
            "scheduler.time_constant_ms: unknown field",
        ),
        (
            {"channel": {"fading": "jakes"}},
            "channel.fading: Input should be 'rayleigh' or 'none' (got 'jakes')",
        ),
        ({"seed": -3}, "seed: Input should be greater than or equal to 0 (got -3)"),
        ({"colour": "red"}, "colour: unknown field"),
        (
            {"ues": [{"name": "bad name", "count": 1, "traffic": {"model": "full_buffer"}}]},
            "ues[0].name: String should match pattern '^[A-Za-z0-9_.-]+$' (got 'bad name')",
        ),
    ],
)
def test_invalid_scenarios_fail_with_clear_messages(changes: dict[str, Any], expected: str) -> None:
    assert problems(scenario_data(**changes)) == [expected]


def test_every_problem_is_reported_at_once() -> None:
    data = scenario_data(seed="abc")
    del data["duration_s"]
    data["ues"][0]["count"] = 0
    data["ues"][0]["traffic"]["rate_pps"] = -1
    assert problems(data) == [
        "seed: Input should be a valid integer, unable to parse string as an integer (got 'abc')",
        "duration_s: Field required",
        "ues[0].count: Input should be greater than 0 (got 0)",
        "ues[0].traffic.rate_pps: Input should be greater than 0 (got -1)",
    ]


@pytest.mark.parametrize("data", [None, [1, 2], "text"])
def test_the_top_level_must_be_a_mapping(data: Any) -> None:
    assert problems(data) == ["the top level must be a mapping of scenario fields"]


def test_error_message_names_the_file_and_lists_problems() -> None:
    err = ScenarioError("s.yaml", ["first", "second"])
    assert str(err) == "s.yaml: invalid scenario\n  first\n  second"


def test_load_reports_yaml_syntax_errors_with_a_position(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("name: x\nseed: [1, 2\n", encoding="utf-8")
    with pytest.raises(ScenarioError) as info:
        load_scenario(path)
    assert info.value.source == str(path)
    [problem] = info.value.problems
    assert problem.startswith("line 3, column 1: invalid YAML:")


def test_load_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ScenarioError, match="cannot read file"):
        load_scenario(tmp_path / "absent.yaml")


def test_load_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ScenarioError, match="top level must be a mapping"):
        load_scenario(path)


def test_replace_validates_the_result() -> None:
    s = scenario()
    assert s.replace(seed=99).seed == 99
    assert s.replace(seed=99).ues == s.ues
    with pytest.raises(ScenarioError, match="duration_s"):
        s.replace(duration_s=0)


def test_scenarios_are_immutable() -> None:
    s = scenario()
    with pytest.raises(ValidationError, match="frozen"):
        s.seed = 3  # type: ignore[misc]


SLICES = [
    {"name": "embb", "sla": {"type": "embb", "min_throughput_mbps": 2}, "quota": 0.7},
    {"name": "urllc", "sla": {"type": "urllc"}, "quota": 0.3},
]


def sliced_data(**changes: Any) -> dict[str, Any]:
    ues = [
        {"name": "video", "count": 2, "slice": "embb", "traffic": {"model": "full_buffer"}},
        {
            "name": "ctrl",
            "count": 2,
            "slice": "urllc",
            "traffic": {"model": "periodic", "period_ms": 5, "packet_bytes": 50},
        },
    ]
    return scenario_data(**{"slices": SLICES, "ues": ues, **changes})


def test_a_sliced_scenario() -> None:
    s = parse_scenario(sliced_data())
    assert [x.name for x in s.slices] == ["embb", "urllc"]
    assert s.slicing is None
    assert s.slicing_config.policy == "delay_aware"
    assert s.slicing_config.window_ms == 100
    urllc = s.slices[1].sla
    assert isinstance(urllc, UrllcSla)
    assert (urllc.delay_budget_ms, urllc.reliability) == (5.0, 0.999)
    embb = s.slices[0].sla
    assert isinstance(embb, EmbbSla)
    assert embb.availability == 0.95
    assert s.slices[0].scheduler == RoundRobinScheduler()
    assert s.slices[0].cqi_backoff_db == 0.0


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"slicing": {"policy": "static"}, "slices": []}, "slicing needs slices"),
        (
            {"slices": [], "ues": [{**poisson_group(), "slice": "embb"}]},
            "UE groups name slices but none are defined: web",
        ),
        (
            {"ues": [poisson_group()]},
            "UE group web must name its slice",
        ),
        (
            {"ues": [{**poisson_group(), "slice": "gold"}]},
            "UE group web names slice 'gold', which is not defined; slices are embb, urllc",
        ),
        (
            {"ues": [{**poisson_group(), "slice": "embb"}]},
            "slices without UEs: urllc",
        ),
        (
            {"slices": [SLICES[0], {**SLICES[1], "name": "embb"}]},
            "slice names must be unique; repeated: embb",
        ),
        (
            {"scheduler": {"model": "pf"}},
            "a sliced cell sets a scheduler in each slice, not at the top level",
        ),
        (
            {"slicing": {"policy": "static"}, "slices": [SLICES[0], {**SLICES[1], "quota": None}]},
            "the static policy needs a quota for every slice; missing: urllc",
        ),
        (
            {"slicing": {"policy": "static"}, "slices": [SLICES[0], {**SLICES[1], "quota": 0.5}]},
            "slice quotas add up to 1.2, more than 1",
        ),
        (
            {"slicing": {"policy": "fair"}},
            "slicing.policy: Input should be 'static', 'delay_aware' or 'proportional' "
            "(got 'fair')",
        ),
        (
            {"slices": [SLICES[0], {**SLICES[1], "sla": {"type": "gold"}}]},
            "slices[1].sla: unknown 'type' 'gold'; expected one of 'embb', 'urllc', 'mmtc'",
        ),
        (
            {"slices": [SLICES[0], {**SLICES[1], "sla": {"type": "urllc", "reliability": 1}}]},
            "slices[1].sla.reliability: Input should be less than 1 (got 1)",
        ),
        (
            {"slices": [{**SLICES[0], "sla": {"type": "embb"}}, SLICES[1]]},
            "slices[0].sla.min_throughput_mbps: Field required",
        ),
    ],
)
def test_invalid_slicing_fails_with_clear_messages(changes: dict[str, Any], expected: str) -> None:
    assert problems(sliced_data(**changes)) == [expected]


def test_replace_keeps_a_sliced_scenario_valid() -> None:
    s = parse_scenario(sliced_data())
    assert s.replace(seed=3).slices == s.slices
