"""Chain expert iteration overnight: each stage starts from the previous checkpoint.

One stage is 50 expert rounds: each round plays 50 search games with 50 PUCT sims,
distills, and keeps the best checkpoint by greedy eval on the full dev suite (64 games).
After each successful stage, ``latest.pt`` is replaced. Default is 20 stages.

    python scripts/overnight_expert.py --init data/models/selfplay_v1.pt

Resume:

    python scripts/overnight_expert.py --init data/models/overnight_expert/latest.pt --out-dir data/models/overnight_expert --seed 1000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _stage_seed(seed_start: int, stage: int, rounds: int, games: int) -> int:
    """New search-game seeds every stage so chunks do not repeat the same games."""
    return seed_start + stage * rounds * games


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overnight chained expert iteration (search teacher)")
    parser.add_argument("--init", type=Path, required=True, help="starting checkpoint")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/models/overnight_expert"),
        help="directory for stage_01.pt … and latest.pt (default: data/models/overnight_expert)",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=20,
        help="how many expert stages to chain (default: 20)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="expert distill cycles per stage (default: 50)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="search games per expert round (default: 50)",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=50,
        help="PUCT simulations per decision (default: 50)",
    )
    parser.add_argument(
        "--mcts-plies",
        type=int,
        default=2,
        help="PUCT decision plies (default: 2 = root + chance + one reply)",
    )
    parser.add_argument("--epochs", type=int, default=4, help="BC epochs per expert round")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate for distill")
    parser.add_argument(
        "--eval-games",
        type=int,
        default=64,
        help="greedy eval games on the dev seed range (default: 64 = full dev suite)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=20_000,
        help="dev-suite seed start (default: 20000). Do not point this at report/holdout.",
    )
    parser.add_argument("--seed", type=int, default=0, help="search-game seed for stage 1")
    parser.add_argument(
        "--visit-sample",
        action="store_true",
        help="sample from visit counts instead of visit-greedy labels",
    )
    return parser.parse_args(argv)


def run_stage(
    *,
    init: Path,
    out: Path,
    rounds: int,
    games: int,
    mcts_sims: int,
    mcts_plies: int,
    seed: int,
    epochs: int,
    lr: float,
    eval_games: int,
    eval_seed: int,
    visit_sample: bool,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "doppelt",
        "train",
        "expert",
        "--init",
        str(init),
        "--out",
        str(out),
        "--rounds",
        str(rounds),
        "--games",
        str(games),
        "--mcts-sims",
        str(mcts_sims),
        "--mcts-plies",
        str(mcts_plies),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--eval-games",
        str(eval_games),
        "--eval-seed",
        str(eval_seed),
    ]
    if visit_sample:
        cmd.append("--visit-sample")
    print(" ".join(cmd), flush=True)
    started = time.perf_counter()
    completed = subprocess.run(cmd)
    elapsed = time.perf_counter() - started
    print(f"stage exit={completed.returncode} elapsed_sec={elapsed:.0f}", flush=True)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stages < 1 or args.rounds < 1 or args.games < 1 or args.mcts_sims < 1 or args.mcts_plies < 1:
        print("stages, rounds, games, mcts-sims, and mcts-plies must be positive")
        return 2
    init = args.init
    if not init.is_file():
        print(f"missing --init {init}")
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    latest = args.out_dir / "latest.pt"

    print(
        f"overnight expert: {args.stages} stages x {args.rounds} round(s) x "
        f"{args.games} games x {args.mcts_sims} mcts-sims x {args.mcts_plies} plies "
        f"(eval {args.eval_games} games @ seed {args.eval_seed})",
        flush=True,
    )
    print(f"init={init}  out={args.out_dir}  latest={latest}", flush=True)

    current_init = init
    for stage in range(args.stages):
        out = args.out_dir / f"stage_{stage + 1:02d}.pt"
        seed = _stage_seed(args.seed, stage, args.rounds, args.games)
        print(
            f"\n=== stage {stage + 1}/{args.stages}  init={current_init}  "
            f"out={out}  seed={seed} ===",
            flush=True,
        )
        code = run_stage(
            init=current_init,
            out=out,
            rounds=args.rounds,
            games=args.games,
            mcts_sims=args.mcts_sims,
            mcts_plies=args.mcts_plies,
            seed=seed,
            epochs=args.epochs,
            lr=args.lr,
            eval_games=args.eval_games,
            eval_seed=args.eval_seed,
            visit_sample=args.visit_sample,
        )
        if code != 0:
            print(f"stopped at stage {stage + 1}; last good init was {current_init}")
            return code
        if not out.is_file():
            print(f"stage {stage + 1} did not write {out}")
            return 1
        latest.write_bytes(out.read_bytes())
        current_init = latest
        print(f"wrote {latest}", flush=True)
    print(f"\ndone. latest={latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
