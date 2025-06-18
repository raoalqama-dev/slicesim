"""The ``slicesim`` command."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import TextIO

from slicesim import __version__
from slicesim.config import ScenarioError, load_scenario
from slicesim.engine import simulate
from slicesim.report import render_summary

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
    run.add_argument("scenario", help="scenario YAML file")
    run.add_argument("--seed", type=_seed, help="override the scenario's seed")
    run.add_argument(
        "--duration", type=_duration, metavar="SECONDS", help="override the scenario's duration"
    )
    run.add_argument("--json", action="store_true", help="print the summary as JSON")
    return parser


def _run(args: argparse.Namespace, out: TextIO) -> int:
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
    summary = simulate(scenario).summary()
    if args.json:
        json.dump(summary.to_dict(), out, indent=2)
        out.write("\n")
    else:
        out.write(render_summary(summary, scenario) + "\n")
    return EXIT_OK


def main(
    argv: Sequence[str] | None = None, out: TextIO | None = None, err: TextIO | None = None
) -> int:
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    args = build_parser().parse_args(argv)
    try:
        return _run(args, out)
    except ScenarioError as exc:
        err.write(f"slicesim: {exc}\n")
        return EXIT_USAGE
