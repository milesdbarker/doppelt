"""Batch solo simulation and throughput benchmarks."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from multiprocessing import Pool

from doppelt.core.phases import Phase
from doppelt.core.scoring import total_score
from doppelt.core.state import GameState
from doppelt.engine.game import apply_action, is_terminal, legal_action_ids, new_game
from doppelt.sim.policy import POLICY_NAMES, Policy, make_policy

AREA_SCORE_KEYS = ("yellow", "blue", "pink", "green", "silver", "foxes")


@dataclass(frozen=True)
class GameOutcome:
    seed: int
    terminal: bool
    total_score: int
    n_actions: int
    scores: dict[str, int]
    actions: tuple[int, ...] = ()


@dataclass(frozen=True)
class BatchResult:
    policy: str
    games: int
    workers: int
    elapsed_sec: float
    unfinished: int
    total_actions: int
    mean_score: float
    mean_area_scores: dict[str, float]
    outcomes: tuple[GameOutcome, ...]

    @property
    def games_per_sec(self) -> float:
        return self.games / self.elapsed_sec if self.elapsed_sec else float("inf")

    @property
    def actions_per_sec(self) -> float:
        return self.total_actions / self.elapsed_sec if self.elapsed_sec else float("inf")

    @property
    def mean_actions(self) -> float:
        return self.total_actions / self.games if self.games else 0.0


def play_game_with_policy(
    seed: int,
    policy: Policy,
    *,
    max_actions: int = 5_000,
) -> GameState:
    """Play one solo game with an injected policy; return the finished GameState."""
    state = new_game(seed=seed)
    for _ in range(max_actions):
        if state.phase is Phase.GAME_OVER:
            break
        legal = legal_action_ids(state)
        if not legal:
            break
        apply_action(state, policy.select(state, legal), check_legal=False)
    return state


def play_with_policy(
    seed: int,
    policy: Policy,
    *,
    max_actions: int = 5_000,
) -> GameOutcome:
    """Play one solo game with an injected policy (dice RNG still from seed)."""
    state = play_game_with_policy(seed, policy, max_actions=max_actions)
    done, scores = is_terminal(state)
    return GameOutcome(
        seed=seed,
        terminal=done,
        total_score=total_score(state.sheet),
        n_actions=len(state.action_log),
        scores=scores,
        actions=tuple(state.action_log),
    )


def resolve_worker_count(workers: int | None) -> int:
    if workers is None or workers <= 0:
        return os.cpu_count() or 1
    return workers


def _worker_play(payload: tuple[str, int, int]) -> GameOutcome:
    policy_name, seed, max_actions = payload
    return play_with_policy(seed, make_policy(policy_name, seed), max_actions=max_actions)


def run_batch(
    n_games: int,
    *,
    seed_start: int = 0,
    workers: int | None = None,
    max_actions: int = 5_000,
    policy: str = "random_legal",
) -> BatchResult:
    """Run N solo games, optionally across a process pool."""
    if n_games < 1:
        raise ValueError("n_games must be at least 1")
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {POLICY_NAMES}")

    worker_count = resolve_worker_count(workers)
    jobs = [(policy, seed_start + index, max_actions) for index in range(n_games)]
    started = time.perf_counter()
    if worker_count == 1:
        outcomes = tuple(_worker_play(job) for job in jobs)
    else:
        chunksize = max(1, n_games // (worker_count * 8))
        with Pool(processes=worker_count) as pool:
            outcomes = tuple(pool.map(_worker_play, jobs, chunksize=chunksize))
    elapsed = time.perf_counter() - started

    unfinished = sum(1 for outcome in outcomes if not outcome.terminal)
    total_actions = sum(outcome.n_actions for outcome in outcomes)
    mean_score = sum(outcome.total_score for outcome in outcomes) / n_games
    mean_area_scores = {
        key: sum(outcome.scores.get(key, 0) for outcome in outcomes) / n_games
        for key in AREA_SCORE_KEYS
    }
    return BatchResult(
        policy=policy,
        games=n_games,
        workers=worker_count,
        elapsed_sec=elapsed,
        unfinished=unfinished,
        total_actions=total_actions,
        mean_score=mean_score,
        mean_area_scores=mean_area_scores,
        outcomes=outcomes,
    )


def format_batch_result(result: BatchResult) -> str:
    lines = [
        f"Simulated {result.games} {result.policy} games",
        f"  workers:      {result.workers}",
        f"  elapsed:      {result.elapsed_sec:.3f}s",
        f"  games/s:      {result.games_per_sec:.1f}",
        f"  actions/s:    {result.actions_per_sec:.1f}",
        f"  unfinished:   {result.unfinished}",
        f"  mean score:   {result.mean_score:.1f}",
        f"  mean yellow:  {result.mean_area_scores['yellow']:.1f}",
        f"  mean blue:    {result.mean_area_scores['blue']:.1f}",
        f"  mean pink:    {result.mean_area_scores['pink']:.1f}",
        f"  mean green:   {result.mean_area_scores['green']:.1f}",
        f"  mean silver:  {result.mean_area_scores['silver']:.1f}",
        f"  mean foxes:   {result.mean_area_scores['foxes']:.1f}",
        f"  mean actions: {result.mean_actions:.1f}",
    ]
    if result.unfinished:
        seeds = [outcome.seed for outcome in result.outcomes if not outcome.terminal]
        preview = ", ".join(str(seed) for seed in seeds[:20])
        extra = "" if len(seeds) <= 20 else f" (+{len(seeds) - 20} more)"
        lines.append(f"  unfinished seeds: {preview}{extra}")
    return "\n".join(lines)
