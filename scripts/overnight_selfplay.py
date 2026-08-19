"""Chain self-play overnight: each stage starts from the previous checkpoint.

Example (10 stages x 1000 iters x 100 games — a full night on a fast CPU):

    python scripts/overnight_selfplay.py --init data/models/bc_v1.pt --out-dir data/models/overnight

Does not use MCTS. Training stays one forward per decision. Eval is greedy and
infrequent so it does not dominate wall time.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _stage_seed(seed_start: int, stage: int, iters: int, games: int) -> int:
    """New dice/self-play seeds every stage so chunks do not repeat the same games."""
    return seed_start + stage * iters * games


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overnight chained self-play")
    parser.add_argument("--init", type=Path, required=True, help="starting checkpoint (usually BC)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/models/overnight"),
        help="directory for stage_01.pt, stage_02.pt, ...",
    )
    parser.add_argument("--stages", type=int, default=10, help="how many chained runs (default: 10)")
    parser.add_argument("--iters", type=int, default=1000, help="gradient steps per stage (default: 1000)")
    parser.add_argument("--games", type=int, default=100, help="games per gradient step (default: 100)")
    parser.add_argument("--kl", type=float, default=0.01, help="KL toward that stage's --init")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=50, help="greedy eval period (default: 50)")
    parser.add_argument("--eval-games", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=20_000, help="fixed eval seeds across stages")
    parser.add_argument("--seed", type=int, default=0, help="self-play seed for stage 1")
    return parser.parse_args(argv)


def run_stage(
    *,
    init: Path,
    out: Path,
    iters: int,
    games: int,
    seed: int,
    kl: float,
    lr: float,
    entropy: float,
    eval_every: int,
    eval_games: int,
    eval_seed: int,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "doppelt",
        "train",
        "selfplay",
        "--init",
        str(init),
        "--out",
        str(out),
        "--iters",
        str(iters),
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--kl",
        str(kl),
        "--lr",
        str(lr),
        "--entropy",
        str(entropy),
        "--eval-every",
        str(eval_every),
        "--eval-games",
        str(eval_games),
        "--eval-seed",
        str(eval_seed),
    ]
    print(" ".join(cmd), flush=True)
    started = time.perf_counter()
    completed = subprocess.run(cmd)
    elapsed = time.perf_counter() - started
    print(f"stage exit={completed.returncode} elapsed_sec={elapsed:.0f}", flush=True)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stages < 1 or args.iters < 1 or args.games < 1:
        print("stages, iters, and games must be positive")
        return 2
    init = args.init
    if not init.is_file():
        print(f"missing --init {init}")
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "overnight.log"
    latest = args.out_dir / "latest.pt"

    print(f"overnight: {args.stages} stages x {args.iters} iters x {args.games} games", flush=True)
    print(f"init={init}  out={args.out_dir}  log={log_path}", flush=True)

    current_init = init
    for stage in range(args.stages):
        out = args.out_dir / f"stage_{stage + 1:02d}.pt"
        seed = _stage_seed(args.seed, stage, args.iters, args.games)
        print(f"\n=== stage {stage + 1}/{args.stages}  init={current_init}  out={out}  seed={seed} ===", flush=True)
        code = run_stage(
            init=current_init,
            out=out,
            iters=args.iters,
            games=args.games,
            seed=seed,
            kl=args.kl,
            lr=args.lr,
            entropy=args.entropy,
            eval_every=args.eval_every,
            eval_games=args.eval_games,
            eval_seed=args.eval_seed,
        )
        if code != 0:
            print(f"stopped at stage {stage + 1}; last good init was {current_init}")
            return code
        if not out.is_file():
            print(f"stage {stage + 1} did not write {out}")
            return 1
        latest.write_bytes(out.read_bytes())
        current_init = out
    print(f"\ndone. latest={latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
