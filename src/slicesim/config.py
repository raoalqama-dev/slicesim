"""Scenario files: pydantic models and YAML loading.

A scenario describes one simulation completely: the carrier, how much each
resource block carries, the UEs and their traffic, how long to run and the
seed. Unknown keys are rejected, so a misspelt option fails loudly instead of
being ignored.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import ErrorDetails

from slicesim.carrier import Carrier


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CarrierConfig(_Model):
    """Channel bandwidth and subcarrier spacing, checked against TS 38.101-1."""

    bandwidth_mhz: int = Field(gt=0)
    scs_khz: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_defined(self) -> CarrierConfig:
        Carrier.create(self.bandwidth_mhz, self.scs_khz)
        return self

    def build(self) -> Carrier:
        return Carrier.create(self.bandwidth_mhz, self.scs_khz)


class FixedLink(_Model):
    """Every resource block carries the same number of bits, whoever it serves."""

    model: Literal["fixed"] = "fixed"
    bits_per_prb: int = Field(default=500, gt=0)


class FullBufferTraffic(_Model):
    """The UE always has more data waiting than the cell can send it."""

    model: Literal["full_buffer"]


class PoissonTraffic(_Model):
    """Packets of a fixed size arriving as a Poisson process."""

    model: Literal["poisson"]
    rate_pps: float = Field(gt=0, description="mean packets per second per UE")
    packet_bytes: int = Field(gt=0)


class PeriodicTraffic(_Model):
    """Small packets at a fixed period, URLLC style, each UE at a random phase."""

    model: Literal["periodic"]
    period_ms: float = Field(gt=0)
    packet_bytes: int = Field(gt=0)


class OnOffTraffic(_Model):
    """Video-like traffic: frames at a fixed rate during exponential on periods."""

    model: Literal["on_off"]
    mean_on_ms: float = Field(gt=0)
    mean_off_ms: float = Field(gt=0)
    frame_rate_fps: float = Field(gt=0)
    frame_bytes: int = Field(gt=0)


TrafficConfig = Annotated[
    FullBufferTraffic | PoissonTraffic | PeriodicTraffic | OnOffTraffic,
    Field(discriminator="model"),
]


class UEGroup(_Model):
    """A number of UEs sharing one traffic model."""

    name: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    count: int = Field(gt=0, le=100_000)
    traffic: TrafficConfig


class Scenario(_Model):
    """One complete, seeded simulation."""

    name: str = Field(min_length=1)
    description: str = ""
    seed: int = Field(ge=0, lt=2**63)
    duration_s: float = Field(gt=0, le=3600)
    carrier: CarrierConfig
    link: FixedLink = FixedLink()
    ues: list[UEGroup] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Scenario:
        names = [g.name for g in self.ues]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"UE group names must be unique; repeated: {', '.join(duplicates)}")
        slot_s = self.carrier.build().slot_duration_s
        if self.duration_s < slot_s:
            raise ValueError(
                f"duration_s must cover at least one slot ({slot_s * 1e3:g} ms at "
                f"{self.carrier.scs_khz} kHz)"
            )
        return self

    @property
    def n_slots(self) -> int:
        """Slots simulated: the duration rounded down to whole slots."""
        slot_s = self.carrier.build().slot_duration_s
        # The epsilon keeps durations such as 0.3 s from losing a slot to rounding.
        return math.floor(self.duration_s / slot_s + 1e-9)

    def replace(self, **changes: Any) -> Scenario:
        """A copy with some fields changed, validated like a freshly loaded file.

        Raises ScenarioError if the result is invalid.
        """
        return parse_scenario({**self.model_dump(), **changes})


class ScenarioError(Exception):
    """A scenario file that cannot be read or does not validate."""

    def __init__(self, source: str, problems: list[str]) -> None:
        self.source = source
        self.problems = problems
        detail = "\n".join(f"  {p}" for p in problems)
        super().__init__(f"{source}: invalid scenario\n{detail}")


_TRAFFIC_MODELS = frozenset(
    get_args(m.model_fields["model"].annotation)[0]
    for m in (FullBufferTraffic, PoissonTraffic, PeriodicTraffic, OnOffTraffic)
)


def _location(loc: tuple[int | str, ...]) -> str:
    """Render a pydantic error location as a path like ``ues[0].traffic.rate_pps``.

    A discriminated union adds the chosen tag to the location; it is dropped,
    since the file never mentions it as a key.
    """
    parts: list[str] = []
    for i, item in enumerate(loc):
        if isinstance(item, int):
            parts.append(f"[{item}]")
        elif i > 0 and loc[i - 1] == "traffic" and item in _TRAFFIC_MODELS:
            continue
        else:
            parts.append(f".{item}" if parts else item)
    return "".join(parts)


def _describe(error: ErrorDetails) -> str:
    """One line per problem, worded in terms of the file rather than the models."""
    kind = error["type"]
    ctx = error.get("ctx", {})
    where = _location(tuple(error["loc"]))
    if kind == "extra_forbidden":
        message = "unknown field"
    elif kind == "union_tag_not_found":
        message = f"{ctx['discriminator']} is required"
    elif kind == "union_tag_invalid":
        message = (
            f"unknown {ctx['discriminator']} {ctx['tag']!r}; expected one of {ctx['expected_tags']}"
        )
    else:
        message = str(error["msg"]).removeprefix("Value error, ")
        given = error.get("input")
        if kind != "missing" and not isinstance(given, dict | list):
            message += f" (got {given!r})"
    return f"{where}: {message}" if where else message


def parse_scenario(data: Any, source: str = "<scenario>") -> Scenario:
    """Validate already-parsed YAML, reporting every problem at once."""
    if not isinstance(data, dict):
        raise ScenarioError(source, ["the top level must be a mapping of scenario fields"])
    try:
        return Scenario.model_validate(data)
    except ValidationError as exc:
        raise ScenarioError(source, [_describe(e) for e in exc.errors()]) from None


def load_scenario(path: str | Path) -> Scenario:
    """Read and validate a YAML scenario file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError(str(path), [f"cannot read file: {exc.strerror or exc}"]) from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        where = ""
        mark = getattr(exc, "problem_mark", None)
        if mark is not None:
            where = f"line {mark.line + 1}, column {mark.column + 1}: "
        problem = getattr(exc, "problem", None) or str(exc)
        raise ScenarioError(str(path), [f"{where}invalid YAML: {problem}"]) from None
    return parse_scenario(data, str(path))
