import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from slicesim.config import load_scenario, parse_scenario
from slicesim.engine import simulate

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


def load_app() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dashboard_app", APP)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up in sys.modules while they are defined.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_default_form_is_the_medium_scenario() -> None:
    app = load_app()
    built = parse_scenario(app.scenario_data(app.Form()))
    medium = load_scenario(APP.parent.parent / "scenarios" / "medium.yaml")
    assert built.slices == medium.slices
    assert [(g.name, g.count, g.traffic) for g in built.ues] == [
        (g.name, g.count, g.traffic) for g in medium.ues
    ]
    assert built.channel == medium.channel
    assert built.slicing == medium.slicing


def test_series_and_delay_cdf() -> None:
    app = load_app()
    form = app.Form(duration_s=0.5, video_ues=4, sensors=20)
    result = simulate(parse_scenario(app.scenario_data(form)))
    times, shares, throughput = app.slice_series(result)
    # 0.5 s of 100 ms windows.
    assert times == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert set(shares) == set(throughput) == {"embb", "urllc", "mmtc"}
    for name in shares:
        assert len(shares[name]) == len(throughput[name]) == 5
        assert all(0 <= share <= 1 for share in shares[name])
    total = sum(sum(values) for values in throughput.values()) * 0.1
    assert total == pytest.approx(result.delivered_bytes.sum() * 8 / 1e6)
    delays, quantiles = app.delay_cdf(result, "urllc")
    assert quantiles[0] == 0.0
    assert quantiles[-1] == 1.0
    assert delays == sorted(delays)


def test_delay_cdf_of_a_silent_slice_is_empty() -> None:
    app = load_app()
    form = app.Form(duration_s=0.001, video_ues=1, urllc_ues=1, sensors=1)
    result = simulate(parse_scenario(app.scenario_data(form)))
    assert app.delay_cdf(result, "mmtc") == ([], [])


def test_the_app_runs_a_scenario() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    assert at.title[0].value == "slicesim"
    duration = next(s for s in at.sidebar.slider if s.label == "duration (s)")
    duration.set_value(0.5)
    video = next(s for s in at.sidebar.slider if s.label == "eMBB video UEs")
    video.set_value(4)
    at.button[0].click().run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["delivered", "PRBs used", "BLER", "SLAs met"]
    assert len(at.dataframe) == 1
    assert at.subheader[0].value == "Slices"


def test_the_app_shows_quota_sliders_for_static_slicing() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    labels = [s.label for s in at.sidebar.slider]
    assert "URLLC quota" not in labels
    policy = next(s for s in at.sidebar.selectbox if s.label == "policy")
    policy.set_value("static").run()
    labels = [s.label for s in at.sidebar.slider]
    assert {"eMBB quota", "URLLC quota", "mMTC quota"} <= set(labels)


def test_the_app_reports_an_invalid_scenario() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    policy = next(s for s in at.sidebar.selectbox if s.label == "policy")
    policy.set_value("static").run()
    for label in ("eMBB quota", "URLLC quota"):
        next(s for s in at.sidebar.slider if s.label == label).set_value(0.8)
    at.button[0].click().run()
    assert not at.exception
    assert "slice quotas add up to 1.65, more than 1" in at.error[0].value
