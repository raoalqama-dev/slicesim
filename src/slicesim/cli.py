"""The ``slicesim`` command."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from typing import TextIO

from slicesim import __version__
from slicesim.compare import SCHEDULERS, compare_schedulers, with_policy
from slicesim.config import Scenario, ScenarioError, load_scenario
from slicesim.engine import simulate
from slicesim.report import render_comparison, render_evaluation, render_summary
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


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
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

    train = commands.add_parser(
        "train",
        help="train the PPO slicing agent (needs the rl extra)",
        description=(
            "Train the PPO slicing agent on one or more sliced scenarios, one per load "
            "level. Writes log.csv, periodic checkpoints and agent.pt to --out."
        ),
    )
    train.add_argument("levels", nargs="+", metavar="SCENARIO", help="sliced scenario files")
    train.add_argument("--out", default="runs/ppo", help="output directory (default: runs/ppo)")
    train.add_argument("--steps", type=_positive_int, default=60_000, help="environment steps")
    train.add_argument("--seed", type=_seed, default=0, help="training seed (default: 0)")
    train.add_argument(
        "--episode", type=_duration, default=1.0, metavar="SECONDS", help="episode length"
    )
    train.add_argument(
        "--decision-slots", type=_positive_int, default=10, help="slots per decision"
    )
    train.add_argument("--checkpoint-every", type=_positive_int, default=5, metavar="UPDATES")

    evaluate = commands.add_parser(
        "evaluate",
        help="compare a trained agent with the heuristic policies (needs the rl extra)",
        description=(
            "Run a trained agent, equal weights and the static, proportional and "
            "delay-aware policies on every scenario with the same held-out seeds."
        ),
    )
    evaluate.add_argument("checkpoint", help="agent checkpoint written by slicesim train")
    evaluate.add_argument("levels", nargs="+", metavar="SCENARIO", help="sliced scenario files")
    evaluate.add_argument("--seeds", type=_positive_int, default=3, help="held-out seeds")
    evaluate.add_argument(
        "--duration", type=_duration, default=10.0, metavar="SECONDS", help="run length"
    )
    evaluate.add_argument("--workers", type=_positive_int, default=1, help="parallel runs")
    evaluate.add_argument("--csv", metavar="PATH", help="also write every run to a CSV file")
    evaluate.add_argument("--json", action="store_true", help="print the results as JSON")
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


class MissingExtraError(Exception):
    """A command needs the rl extra, which is not installed."""


class CommandError(Exception):
    """A command could not run, for a reason worth one line: a missing file, say."""


def _load_levels(paths: Sequence[str]) -> list[Scenario]:
    return [load_scenario(path) for path in paths]


def _train(args: argparse.Namespace, out: TextIO) -> int:
    try:
        from slicesim.rl.ppo import PPOConfig, UpdateStats
        from slicesim.rl.train import train
    except ImportError as exc:
        raise MissingExtraError(str(exc)) from None
    levels = _load_levels(args.levels)

    def progress(stats: UpdateStats) -> None:
        mean = "-" if stats.mean_return != stats.mean_return else f"{stats.mean_return:9.1f}"
        out.write(
            f"update {stats.update:4d}  steps {stats.env_steps:7,d}  episodes {stats.episodes:3d}"
            f"  return {mean}  std {stats.action_std:.3f}  kl {stats.approx_kl:.4f}"
            f"  {stats.seconds:5.1f}s\n"
        )
        out.flush()

    try:
        run = train(
            levels,
            args.out,
            PPOConfig(total_steps=args.steps, seed=args.seed),
            decision_slots=args.decision_slots,
            episode_s=args.episode,
            checkpoint_every=args.checkpoint_every,
            on_update=progress,
        )
    except (OSError, ValueError) as exc:
        raise CommandError(str(exc)) from None
    out.write(f"wrote {run.log} and {run.checkpoint}\n")
    return EXIT_OK


def _evaluate(args: argparse.Namespace, out: TextIO) -> int:
    try:
        from slicesim.rl.evaluate import aggregate, evaluate, write_csv
    except ImportError as exc:
        raise MissingExtraError(str(exc)) from None
    levels = _load_levels(args.levels)
    try:
        rows = evaluate(
            args.checkpoint,
            levels,
            seeds=args.seeds,
            duration_s=args.duration,
            workers=args.workers,
        )
    except (OSError, ValueError) as exc:
        raise CommandError(str(exc)) from None
    totals = aggregate(rows)
    if args.csv:
        write_csv(rows, args.csv)
    if args.json:
        json.dump([asdict(t) for t in totals], out, indent=2)
        out.write("\n")
    else:
        out.write(render_evaluation(totals) + "\n")
    return EXIT_OK


COMMANDS = {"run": _run, "compare": _compare, "train": _train, "evaluate": _evaluate}


def main(
    argv: Sequence[str] | None = None, out: TextIO | None = None, err: TextIO | None = None
) -> int:
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args, out)
    except ScenarioError as exc:
        err.write(f"slicesim: {exc}\n")
        return EXIT_USAGE
    except MissingExtraError as exc:
        err.write(
            f"slicesim: {args.command} needs the rl extra ({exc}); "
            'install it with: pip install "slicesim[rl]"\n'
        )
        return EXIT_USAGE
    except CommandError as exc:
        err.write(f"slicesim: {exc}\n")
        return EXIT_USAGE
