"""Evaluate a neural checkpoint against baseline bots on a fixed seed range."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from doppelt.core.scoring import total_score
from doppelt.engine.game import is_terminal
from doppelt.ml.policy import NeuralPolicy
from doppelt.sim.batch import play_game_with_policy
from doppelt.sim.policy import make_policy


@dataclass(frozen=True)
class EvalRow:
    name: str
    games: int
    mean_score: float
    median_score: float
    unfinished: int


def _summarize(name: str, scores: Sequence[float], unfinished: int) -> EvalRow:
    ordered = sorted(scores)
    n = len(ordered)
    if n == 0:
        median = 0.0
        mean = 0.0
    elif n % 2:
        median = float(ordered[n // 2])
        mean = sum(ordered) / n
    else:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
        mean = sum(ordered) / n
    return EvalRow(
        name=name,
        games=n,
        mean_score=mean,
        median_score=median,
        unfinished=unfinished,
    )


def eval_policy_scores(
    policy,
    *,
    games: int,
    seed_start: int,
    max_actions: int = 5_000,
) -> EvalRow:
    scores: list[float] = []
    unfinished = 0
    for index in range(games):
        state = play_game_with_policy(seed_start + index, policy, max_actions=max_actions)
        done, _ = is_terminal(state)
        if not done:
            unfinished += 1
        scores.append(float(total_score(state.sheet)))
    return _summarize(policy.name, scores, unfinished)


def eval_checkpoint(
    checkpoint: Path,
    *,
    games: int = 64,
    seed_start: int = 10_000,
    baselines: Sequence[str] = ("random_legal",),
    max_actions: int = 5_000,
    sample: bool = False,
    mcts_sims: int = 0,
) -> list[EvalRow]:
    """Run the neural policy and named baselines on the same seeds."""
    rows = [
        eval_policy_scores(
            NeuralPolicy(checkpoint, seed=seed_start, sample=sample, mcts_sims=mcts_sims),
            games=games,
            seed_start=seed_start,
            max_actions=max_actions,
        )
    ]
    for name in baselines:
        rows.append(
            eval_policy_scores(
                make_policy(name, seed_start),
                games=games,
                seed_start=seed_start,
                max_actions=max_actions,
            )
        )
    return rows


def format_eval_rows(rows: Sequence[EvalRow]) -> str:
    lines = ["Eval (same seeds):"]
    for row in rows:
        lines.append(
            f"  {row.name:16s}  games={row.games:4d}  mean={row.mean_score:6.1f}  "
            f"median={row.median_score:6.1f}  unfinished={row.unfinished}"
        )
    return "\n".join(lines)
