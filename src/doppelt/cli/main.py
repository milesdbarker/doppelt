"""Doppelt CLI — play, replay, and inspect action logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from doppelt.cli.commands import (
    run_dataset_analyze,
    run_dataset_generate,
    run_decode,
    run_eval,
    run_play,
    run_random,
    run_replay,
    run_simulate,
    run_train_bc,
    run_train_expert,
    run_train_selfplay,
    run_visualize,
)
from doppelt.ml.eval_suite import DEFAULT_EVAL_SUITE, SUITE_NAMES
from doppelt.sim.dataset import DATASET_POLICIES, DEFAULT_MIX, DEFAULT_SHARD_SIZE
from doppelt.sim.policy import CLI_POLICY_CHOICES, POLICY_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doppelt",
        description="Doppelt So Clever solo engine — play, simulate, replay, and analyze datasets.",
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

    random_cmd = subparsers.add_parser(
        "random",
        help="play one solo game with a bot policy and optionally save a log",
    )
    random_cmd.add_argument("--seed", type=int, default=1, help="game seed (default: 1)")
    random_cmd.add_argument(
        "--policy",
        choices=list(CLI_POLICY_CHOICES) + ["neural"],
        default="random",
        help="bot policy (default: random). Also accepts random_legal, greedy_immediate, mcts_lite",
    )
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
    random_cmd.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="neural policy checkpoint (required with --policy neural)",
    )
    random_cmd.add_argument(
        "--mcts-sims",
        type=int,
        default=0,
        help="PUCT simulations per neural decision (0 = greedy, no search)",
    )
    random_cmd.add_argument(
        "--mcts-plies",
        type=int,
        default=2,
        help="PUCT decision plies when --mcts-sims > 0 (1 = root + chance leaf; 2 = + one reply)",
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

    visualize = subparsers.add_parser(
        "visualize",
        help="overlay a replay on the score sheet and step through decisions",
    )
    visualize.add_argument("log", type=Path, help="path to .bin log file")
    visualize.add_argument(
        "--out",
        type=Path,
        metavar="PNG",
        help="write the selected frame as a PNG",
    )
    visualize.add_argument(
        "--step",
        type=int,
        default=None,
        help="frame index to show/save (0 = before any action; default = final)",
    )
    visualize.add_argument(
        "--no-window",
        action="store_true",
        help="do not open the interactive viewer",
    )

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

    dataset = subparsers.add_parser(
        "dataset",
        help="generate or inspect solo training datasets",
    )
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    generate = dataset_sub.add_parser(
        "generate",
        help="simulate mixed bots into DPLD shards (no mcts_lite)",
    )
    generate.add_argument(
        "--out",
        type=Path,
        default=Path("data/datasets/solo_v1"),
        help="output directory (default: data/datasets/solo_v1)",
    )
    generate.add_argument(
        "--games",
        type=int,
        default=1_000_000,
        help="total games across the mix (default: 1000000)",
    )
    generate.add_argument(
        "--mix",
        default=DEFAULT_MIX,
        help=f"policy:weight list (default: {DEFAULT_MIX}; allowed: {', '.join(DATASET_POLICIES)})",
    )
    generate.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help=f"games per shard file (default: {DEFAULT_SHARD_SIZE})",
    )
    generate.add_argument("--seed", type=int, default=0, help="first seed (default: 0)")
    generate.add_argument(
        "--workers",
        type=int,
        default=0,
        help="process count (default: 0 = CPU count)",
    )
    generate.add_argument(
        "--max-actions",
        type=int,
        default=5_000,
        help="safety cap on actions per game (default: 5000)",
    )
    generate.add_argument(
        "--no-resume",
        action="store_true",
        help="regenerate shards even if they already exist",
    )
    analyze = dataset_sub.add_parser(
        "analyze",
        help="score distributions, fox impact, and strategy flags from DPLD shards",
    )
    analyze.add_argument(
        "--in",
        dest="in_dir",
        type=Path,
        default=Path("data/datasets/solo_v1"),
        help="dataset directory (default: data/datasets/solo_v1)",
    )
    analyze.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="optional JSON report path",
    )

    train = subparsers.add_parser("train", help="train a neural policy")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    train_bc = train_sub.add_parser(
        "bc",
        help="behavioral cloning from DPLD shards or live bot games",
    )
    train_bc.add_argument(
        "--in",
        dest="in_dir",
        type=Path,
        default=None,
        help="DPLD dataset directory (optional if --live-games is set)",
    )
    train_bc.add_argument(
        "--live-games",
        type=int,
        default=0,
        help="play this many bot games instead of (or in addition to) shards",
    )
    train_bc.add_argument(
        "--policy",
        default="heuristic",
        help="dataset filter / live bot name (default: heuristic)",
    )
    train_bc.add_argument(
        "--max-games",
        type=int,
        default=2_000,
        help="max shard games to replay (default: 2000)",
    )
    train_bc.add_argument("--seed", type=int, default=0, help="live-game seed start / torch seed")
    train_bc.add_argument("--epochs", type=int, default=8, help="training epochs (default: 8)")
    train_bc.add_argument("--batch-size", type=int, default=64, help="minibatch size")
    train_bc.add_argument("--hidden", type=int, default=256, help="trunk hidden size (default: 256)")
    train_bc.add_argument(
        "--arch",
        dest="architecture",
        choices=["pvn_v1", "mlp"],
        default="pvn_v1",
        help="model architecture (default: pvn_v1; mlp is the original flat trunk)",
    )
    train_bc.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    train_bc.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="fraction of seeds held out by seed %% 10 (default: 0.1)",
    )
    train_bc.add_argument(
        "--out",
        type=Path,
        default=Path("data/models/bc_v1.pt"),
        help="checkpoint path (default: data/models/bc_v1.pt)",
    )

    train_sp = train_sub.add_parser(
        "selfplay",
        help="actor-critic self-play from a BC checkpoint (solo vs dice RNG)",
    )
    train_sp.add_argument(
        "--init",
        type=Path,
        default=None,
        help="starting checkpoint (BC); random pvn_v1 if omitted",
    )
    train_sp.add_argument(
        "--out",
        type=Path,
        default=Path("data/models/selfplay_v1.pt"),
        help="best checkpoint path (default: data/models/selfplay_v1.pt)",
    )
    train_sp.add_argument("--iters", type=int, default=20, help="gradient steps (default: 20)")
    train_sp.add_argument(
        "--games",
        type=int,
        default=16,
        help="sampled games per iteration (default: 16)",
    )
    train_sp.add_argument("--seed", type=int, default=0, help="first self-play seed")
    train_sp.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    train_sp.add_argument("--entropy", type=float, default=0.01, help="entropy bonus")
    train_sp.add_argument("--value-coef", type=float, default=0.5, help="value loss weight")
    train_sp.add_argument(
        "--kl",
        type=float,
        default=0.0,
        help="KL penalty toward --init (default: 0)",
    )
    train_sp.add_argument(
        "--eval-games",
        type=int,
        default=16,
        help="greedy eval games on the dev seed range (not the frozen report suite)",
    )
    train_sp.add_argument(
        "--eval-seed",
        type=int,
        default=20_000,
        help="dev eval seed start (suite 'dev' is 20000; do not use report seeds here)",
    )
    train_sp.add_argument("--eval-every", type=int, default=5, help="eval every N iters")
    train_sp.add_argument("--hidden", type=int, default=256, help="used only if --init is omitted")
    train_sp.add_argument(
        "--arch",
        dest="architecture",
        choices=["pvn_v1", "mlp"],
        default="pvn_v1",
        help="used only if --init is omitted",
    )

    train_ex = train_sub.add_parser(
        "expert",
        help="expert iteration: PUCT search games as teacher, distill onto the net (3.6.1)",
    )
    train_ex.add_argument(
        "--init",
        type=Path,
        required=True,
        help="starting checkpoint (required; search labels come from this net)",
    )
    train_ex.add_argument(
        "--out",
        type=Path,
        default=Path("data/models/expert_v1.pt"),
        help="best greedy checkpoint (default: data/models/expert_v1.pt)",
    )
    train_ex.add_argument("--rounds", type=int, default=3, help="generate+distill cycles (default: 3)")
    train_ex.add_argument(
        "--games",
        type=int,
        default=32,
        help="search games per round (default: 32)",
    )
    train_ex.add_argument(
        "--mcts-sims",
        type=int,
        default=32,
        help="PUCT simulations per decision when labeling (default: 32)",
    )
    train_ex.add_argument(
        "--mcts-plies",
        type=int,
        default=2,
        help="PUCT decision plies for search labels (default: 2 = root + one reply)",
    )
    train_ex.add_argument(
        "--visit-sample",
        action="store_true",
        help="sample from visit counts instead of visit-greedy labels",
    )
    train_ex.add_argument("--seed", type=int, default=0, help="first search-game seed")
    train_ex.add_argument("--epochs", type=int, default=4, help="BC epochs per round (default: 4)")
    train_ex.add_argument("--batch-size", type=int, default=64, help="minibatch size")
    train_ex.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate for distill")
    train_ex.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="fraction of seeds held out by seed %% 10 (default: 0.1)",
    )
    train_ex.add_argument(
        "--eval-games",
        type=int,
        default=16,
        help="greedy eval games on the dev seed range (not the frozen report suite)",
    )
    train_ex.add_argument(
        "--eval-seed",
        type=int,
        default=20_000,
        help="dev eval seed start (do not use report/holdout)",
    )

    evaluate = subparsers.add_parser(
        "eval",
        help="score a neural checkpoint on a frozen seed suite (report = 300-claim)",
    )
    evaluate.add_argument("checkpoint", type=Path, help="path to .pt checkpoint")
    evaluate.add_argument(
        "--suite",
        choices=[*SUITE_NAMES, "custom"],
        default=DEFAULT_EVAL_SUITE,
        help="frozen suite: report/holdout are not for retune; custom needs --games and --seed "
        f"(default: {DEFAULT_EVAL_SUITE})",
    )
    evaluate.add_argument(
        "--games",
        type=int,
        default=None,
        help="override suite length (marks the run as truncated / not a 300 claim)",
    )
    evaluate.add_argument(
        "--seed",
        type=int,
        default=None,
        help="only for --suite custom (first seed); named suites ignore this",
    )
    evaluate.add_argument(
        "--baselines",
        default="random_legal",
        help="comma-separated bot names (default: random_legal)",
    )
    evaluate.add_argument(
        "--sample",
        action="store_true",
        help="sample from the policy instead of greedy argmax",
    )
    evaluate.add_argument(
        "--mcts-sims",
        type=int,
        default=0,
        help="PUCT simulations per neural decision (0 = greedy, no search)",
    )
    evaluate.add_argument(
        "--mcts-plies",
        type=int,
        default=2,
        help="PUCT decision plies when --mcts-sims > 0 (default: 2)",
    )
    evaluate.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="optional JSON report path",
    )
    evaluate.add_argument(
        "--no-failures",
        action="store_true",
        help="skip failure-mining flags (zero-color, pink-6, plus-one, foxes)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "play":
        return run_play(seed=args.seed, save_path=args.save)
    if args.command == "random":
        return run_random(
            seed=args.seed,
            max_actions=args.max_actions,
            save_path=args.save,
            policy=args.policy,
            checkpoint=args.checkpoint,
            mcts_sims=args.mcts_sims,
            mcts_plies=args.mcts_plies,
        )
    if args.command == "replay":
        return run_replay(args.log, verbose=args.verbose)
    if args.command == "decode":
        return run_decode(args.log)
    if args.command == "visualize":
        return run_visualize(
            args.log,
            out_path=args.out,
            window=not args.no_window,
            step=args.step,
        )
    if args.command == "simulate":
        return run_simulate(
            games=args.games,
            seed=args.seed,
            workers=args.workers,
            max_actions=args.max_actions,
            policy=args.policy,
        )
    if args.command == "dataset":
        if args.dataset_command == "generate":
            return run_dataset_generate(
                out_dir=args.out,
                games=args.games,
                mix=args.mix,
                shard_size=args.shard_size,
                seed=args.seed,
                workers=args.workers,
                max_actions=args.max_actions,
                resume=not args.no_resume,
            )
        if args.dataset_command == "analyze":
            return run_dataset_analyze(in_dir=args.in_dir, json_path=args.json_path)
        parser.error(f"unknown dataset command {args.dataset_command!r}")
    if args.command == "train":
        if args.train_command == "bc":
            return run_train_bc(
                in_dir=args.in_dir,
                live_games=args.live_games,
                policy=args.policy,
                max_games=args.max_games,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden=args.hidden,
                architecture=args.architecture,
                lr=args.lr,
                val_frac=args.val_frac,
                out_path=args.out,
            )
        if args.train_command == "selfplay":
            return run_train_selfplay(
                init_checkpoint=args.init,
                out_path=args.out,
                iterations=args.iters,
                games_per_iter=args.games,
                seed=args.seed,
                lr=args.lr,
                entropy_coef=args.entropy,
                value_coef=args.value_coef,
                kl_coef=args.kl,
                eval_games=args.eval_games,
                eval_seed=args.eval_seed,
                eval_every=args.eval_every,
                hidden=args.hidden,
                architecture=args.architecture,
            )
        if args.train_command == "expert":
            return run_train_expert(
                init_checkpoint=args.init,
                out_path=args.out,
                rounds=args.rounds,
                games_per_round=args.games,
                mcts_sims=args.mcts_sims,
                mcts_plies=args.mcts_plies,
                visit_sample=args.visit_sample,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                val_frac=args.val_frac,
                eval_games=args.eval_games,
                eval_seed=args.eval_seed,
            )
        parser.error(f"unknown train command {args.train_command!r}")
    if args.command == "eval":
        return run_eval(
            checkpoint=args.checkpoint,
            suite=args.suite,
            games=args.games,
            seed=args.seed,
            baselines=args.baselines,
            sample=args.sample,
            mcts_sims=args.mcts_sims,
            mcts_plies=args.mcts_plies,
            json_path=args.json_path,
            collect_failures=not args.no_failures,
        )

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
