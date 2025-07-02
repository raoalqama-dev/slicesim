"""The ``slicesim`` command."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import TextIO

from slicesim import __version__
from slicesim.compare import SCHEDULERS, compare_schedulers, with_policy
from slicesim.config import Scenario, ScenarioError, load_scenario
from slicesim.engine import simulate
from slicesim.report import render_comparison, render_summary
from slicesim.slicing import POLICIES

EXIT_OK = 0
EXIT_USAGE = 2
"""Bad arguments or an invalid scenario, the same code argparse uses."""


def _seed(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _duration(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not value > 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _schedulers(text: str) -> list[str]:
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in SCHEDULERS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown scheduler {unknown[0]!r}; choose from {', '.join(SCHEDULERS)}"
        )
    if not names:
        raise argparse.ArgumentTypeError("name at least one scheduler")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("each scheduler may be named once")
    return names


def _add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("scenario", help="scenario YAML file")
    parser.add_argument("--seed", type=_seed, help="override the scenario's seed")
    parser.add_argument(
        "--duration", type=_duration, metavar="SECONDS", help="override the scenario's duration"
    )
    parser.add_argument("--json", action="store_true", help="print the results as JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slicesim",
        description="Discrete-time 5G RAN slicing simulator.",
    )
    parser.add_argument("--version", action="version", version=f"slicesim {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = commands.add_parser(
        "run",
        help="simulate a scenario and print a summary",
        description="Simulate a scenario and print a summary per UE group.",
    )
    _add_scenario_arguments(run)
    run.add_argument(
        "--policy",
        choices=POLICIES,
        help="override a sliced scenario's inter-slice policy",
    )

    compare = commands.add_parser(
        "compare",
        help="run a scenario under several schedulers",
        description=(
            "Run a scenario once per scheduler, with the same UEs, channels and traffic, "
            "and print one row per scheduler."
        ),
    )
    _add_scenario_arguments(compare)
    compare.add_argument(
        "--schedulers",
        type=_schedulers,
        default=list(SCHEDULERS),
        metavar="NAMES",
        help=f"comma-separated schedulers to run (default: {','.join(SCHEDULERS)})",
    )
    return parser


def _load(args: argparse.Namespace) -> Scenario:
    scenario = load_scenario(args.scenario)
    changes: dict[str, object] = {}
    if args.seed is not None:
        changes["seed"] = args.seed
    if args.duration is not None:
        changes["duration_s"] = args.duration
    if changes:
        try:
            scenario = scenario.replace(**changes)
        except ScenarioError as exc:
            raise ScenarioError(args.scenario, exc.problems) from None
    return scenario


def _run(args: argparse.Namespace, out: TextIO) -> int:
    scenario = _load(args)
    if args.policy is not None:
        try:
            scenario = with_policy(scenario, args.policy)
        except ScenarioError as exc:
            raise ScenarioError(args.scenario, exc.problems) from None
    summary = simulate(scenario).summary()
    if args.json:
        json.dump(summary.to_dict(), out, indent=2)
        out.write("\n")
    else:
        out.write(render_summary(summary, scenario) + "\n")
    return EXIT_OK


def _compare(args: argparse.Namespace, out: TextIO) -> int:
    runs = compare_schedulers(_load(args), args.schedulers)
    if args.json:
        data = [
            {"scheduler": scenario.scheduler.model_dump(), **summary.to_dict()}
            for scenario, summary in runs
        ]
        json.dump(data, out, indent=2)
        out.write("\n")
    else:
        out.write(render_comparison(runs) + "\n")
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None, out: TextIO | None = None, err: TextIO | None = None
) -> int:
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    args = build_parser().parse_args(argv)
    command = _compare if args.command == "compare" else _run
    try:
        return command(args, out)
    except ScenarioError as exc:
        err.write(f"slicesim: {exc}\n")
        return EXIT_USAGE
