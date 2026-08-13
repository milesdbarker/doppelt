"""Doppelt CLI — play, replay, and inspect action logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from doppelt.cli.commands import run_decode, run_play, run_random, run_replay, run_simulate
from doppelt.sim.policy import POLICY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doppelt",
        description="Doppelt So Clever solo engine — play, simulate, and replay games.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="interactive solo game")
    play.add_argument("--seed", type=int, default=1, help="game seed (default: 1)")
    play.add_argument(
        "--save",
        type=Path,
        metavar="LOG",
        help="write binary action log when the game finishes",
    )

    random_cmd = subparsers.add_parser("random", help="play a random legal solo game")
    random_cmd.add_argument("--seed", type=int, default=1, help="game seed (default: 1)")
    random_cmd.add_argument(
        "--max-actions",
        type=int,
        default=10_000,
        help="safety cap on actions (default: 10000)",
    )
    random_cmd.add_argument(
        "--save",
        type=Path,
        metavar="LOG",
        help="write binary action log",
    )

    replay = subparsers.add_parser("replay", help="replay a binary action log")
    replay.add_argument("log", type=Path, help="path to .bin log file")
    replay.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="print each action while replaying",
    )

    decode = subparsers.add_parser("decode", help="pretty-print a binary log header and actions")
    decode.add_argument("log", type=Path, help="path to .bin log file")

    simulate = subparsers.add_parser(
        "simulate",
        help="batch-simulate games with a baseline bot and report games/s",
    )
    simulate.add_argument("--games", type=int, default=1000, help="number of games (default: 1000)")
    simulate.add_argument("--seed", type=int, default=0, help="first game seed (default: 0)")
    simulate.add_argument(
        "--policy",
        choices=list(POLICY_NAMES),
        default="random_legal",
        help="bot policy (default: random_legal)",
    )
    simulate.add_argument(
        "--workers",
        type=int,
        default=0,
        help="process count (default: 0 = CPU count)",
    )
    simulate.add_argument(
        "--max-actions",
        type=int,
        default=5_000,
        help="safety cap on actions per game (default: 5000)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "play":
        return run_play(seed=args.seed, save_path=args.save)
    if args.command == "random":
        return run_random(seed=args.seed, max_actions=args.max_actions, save_path=args.save)
    if args.command == "replay":
        return run_replay(args.log, verbose=args.verbose)
    if args.command == "decode":
        return run_decode(args.log)
    if args.command == "simulate":
        return run_simulate(
            games=args.games,
            seed=args.seed,
            workers=args.workers,
            max_actions=args.max_actions,
            policy=args.policy,
        )

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
