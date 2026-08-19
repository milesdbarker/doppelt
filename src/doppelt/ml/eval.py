"""Evaluate a neural checkpoint against baseline bots on a frozen seed suite."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from doppelt.core.scoring import total_score
from doppelt.engine.game import is_terminal
from doppelt.ml.eval_suite import (
    SCORE_GOAL,
    EvalSuite,
    custom_eval_suite,
    resolve_eval_suite,
    suite_to_json,
)
from doppelt.ml.policy import NeuralPolicy
from doppelt.sim.batch import play_game_with_policy
from doppelt.sim.failures import (
    FailureSummary,
    flags_from_state,
    format_failure_summary,
    summarize_failures,
)
from doppelt.sim.policy import make_policy


@dataclass(frozen=True)
class EvalRow:
    name: str
    games: int
    mean_score: float
    median_score: float
    std_score: float
    p10: float
    minimum: float
    maximum: float
    share_ge_goal: float
    unfinished: int


@dataclass(frozen=True)
class EvalReport:
    suite: EvalSuite
    score_goal: int
    rows: tuple[EvalRow, ...]
    failures: dict[str, FailureSummary]


def _percentile(ordered: Sequence[float], p: float) -> float:
    if not ordered:
        return 0.0
    if p <= 0:
        return float(ordered[0])
    if p >= 100:
        return float(ordered[-1])
    idx = (len(ordered) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summarize(name: str, scores: Sequence[float], unfinished: int, *, goal: int) -> EvalRow:
    ordered = sorted(scores)
    n = len(ordered)
    if n == 0:
        return EvalRow(name, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, unfinished)
    mean = sum(ordered) / n
    var = sum((value - mean) ** 2 for value in ordered) / n
    if n % 2:
        median = float(ordered[n // 2])
    else:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    ge_goal = sum(1 for value in ordered if value >= goal) / n
    return EvalRow(
        name=name,
        games=n,
        mean_score=mean,
        median_score=median,
        std_score=var**0.5,
        p10=_percentile(ordered, 10),
        minimum=ordered[0],
        maximum=ordered[-1],
        share_ge_goal=ge_goal,
        unfinished=unfinished,
    )


def eval_policy_scores(
    policy,
    *,
    games: int,
    seed_start: int,
    max_actions: int = 5_000,
    goal: int = SCORE_GOAL,
    collect_failures: bool = True,
) -> tuple[EvalRow, FailureSummary | None]:
    scores: list[float] = []
    unfinished = 0
    flags = []
    for index in range(games):
        seed = seed_start + index
        state = play_game_with_policy(seed, policy, max_actions=max_actions)
        done, _ = is_terminal(state)
        if not done:
            unfinished += 1
        scores.append(float(total_score(state.sheet)))
        if collect_failures:
            flags.append(flags_from_state(state, seed=seed))
    row = _summarize(policy.name, scores, unfinished, goal=goal)
    summary = summarize_failures(flags, goal=goal) if collect_failures else None
    return row, summary


def eval_checkpoint(
    checkpoint: Path,
    *,
    games: int = 64,
    seed_start: int = 10_000,
    baselines: Sequence[str] = ("random_legal",),
    max_actions: int = 5_000,
    sample: bool = False,
    mcts_sims: int = 0,
    mcts_plies: int = 2,
    suite: EvalSuite | str | None = None,
    collect_failures: bool = True,
) -> EvalReport:
    """Run the neural policy and named baselines on the same seeds."""
    if isinstance(suite, str):
        resolved = resolve_eval_suite(suite, games=games, seed=seed_start)
    elif suite is None:
        resolved = custom_eval_suite(seed_start=seed_start, games=games)
    else:
        resolved = suite
    rows: list[EvalRow] = []
    failures: dict[str, FailureSummary] = {}
    neural = NeuralPolicy(
        checkpoint, seed=resolved.seed_start, sample=sample, mcts_sims=mcts_sims, mcts_plies=mcts_plies
    )
    policies: list = [neural]
    for name in baselines:
        policies.append(make_policy(name, resolved.seed_start))
    for policy in policies:
        row, summary = eval_policy_scores(
            policy,
            games=resolved.games,
            seed_start=resolved.seed_start,
            max_actions=max_actions,
            collect_failures=collect_failures,
        )
        rows.append(row)
        if summary is not None:
            failures[row.name] = summary
    return EvalReport(
        suite=resolved,
        score_goal=SCORE_GOAL,
        rows=tuple(rows),
        failures=failures,
    )


def format_eval_report(report: EvalReport) -> str:
    suite = report.suite
    retune = "ok to retune" if suite.retune else "DO NOT retune / select checkpoints on this suite"
    lines = [
        f"Eval suite={suite.name}  seeds={suite.seed_start}..{suite.seed_start + suite.games - 1}  "
        f"games={suite.games}  goal={report.score_goal}",
        f"  {suite.purpose}",
        f"  {retune}",
        "",
    ]
    for row in report.rows:
        lines.append(
            f"  {row.name:16s}  n={row.games:4d}  mean={row.mean_score:6.1f}  "
            f"median={row.median_score:6.1f}  p10={row.p10:6.1f}  "
            f"%≥{report.score_goal}={row.share_ge_goal:6.1%}  "
            f"std={row.std_score:5.1f}  min={row.minimum:.0f} max={row.maximum:.0f}  "
            f"unfinished={row.unfinished}"
        )
        summary = report.failures.get(row.name)
        if summary is not None:
            lines.append(format_failure_summary(summary, indent="    "))
    return "\n".join(lines)


def format_eval_rows(rows: Sequence[EvalRow], *, score_goal: int = SCORE_GOAL) -> str:
    """Back-compat one-liner table (no suite / failures)."""
    lines = ["Eval (same seeds):"]
    for row in rows:
        lines.append(
            f"  {row.name:16s}  games={row.games:4d}  mean={row.mean_score:6.1f}  "
            f"median={row.median_score:6.1f}  p10={row.p10:6.1f}  "
            f"%≥{score_goal}={row.share_ge_goal:6.1%}  unfinished={row.unfinished}"
        )
    return "\n".join(lines)


def eval_report_to_json(report: EvalReport) -> dict:
    return {
        "suite": suite_to_json(report.suite),
        "score_goal": report.score_goal,
        "rows": [asdict(row) for row in report.rows],
        "failures": {
            name: asdict(summary) for name, summary in report.failures.items()
        },
    }


def write_eval_json(report: EvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(eval_report_to_json(report), indent=2) + "\n", encoding="utf-8")
