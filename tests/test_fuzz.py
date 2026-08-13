"""Fuzz: random legal play for 10k games must always reach GAME_OVER."""

from __future__ import annotations

import time

import pytest

from doppelt.core.phases import Phase
from doppelt.engine.game import is_terminal, play_random_game

FUZZ_GAMES = 10_000
FUZZ_MAX_ACTIONS = 5_000


def _run_fuzz(n_games: int, *, seed_start: int = 0, max_actions: int = FUZZ_MAX_ACTIONS):
    unfinished: list[tuple[int, str, int]] = []
    started = time.perf_counter()
    for offset in range(n_games):
        seed = seed_start + offset
        state = play_random_game(seed=seed, max_actions=max_actions)
        done, _ = is_terminal(state)
        if not done or state.phase is not Phase.GAME_OVER:
            unfinished.append((seed, state.phase.value, len(state.action_log)))
    elapsed = time.perf_counter() - started
    return elapsed, unfinished


@pytest.mark.slow
def test_random_play_10k_all_reach_terminal(capsys):
    elapsed, unfinished = _run_fuzz(FUZZ_GAMES)
    games_per_sec = FUZZ_GAMES / elapsed if elapsed else float("inf")
    summary = (
        f"fuzz {FUZZ_GAMES} games: {elapsed:.3f}s "
        f"({games_per_sec:.1f} games/s), unfinished={len(unfinished)}"
    )
    print(summary)
    assert unfinished == [], f"{len(unfinished)} games did not finish: {unfinished[:10]}"


if __name__ == "__main__":
    elapsed, unfinished = _run_fuzz(FUZZ_GAMES)
    games_per_sec = FUZZ_GAMES / elapsed if elapsed else float("inf")
    print(f"games={FUZZ_GAMES}")
    print(f"elapsed_sec={elapsed:.3f}")
    print(f"games_per_sec={games_per_sec:.1f}")
    print(f"unfinished={len(unfinished)}")
    if unfinished:
        for seed, phase, n_actions in unfinished[:20]:
            print(f"  seed={seed} phase={phase} actions={n_actions}")
        raise SystemExit(1)
    print("all games reached GAME_OVER")
