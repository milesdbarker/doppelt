"""Batch simulator and RandomLegal policy."""

import pytest

from doppelt.core.scoring import total_score
from doppelt.engine.game import play_random_game
from doppelt.sim import RandomLegal, format_batch_result, make_policy, play_with_policy, run_batch


def test_make_policy_random_and_unknown():
    assert make_policy("random_legal", seed=1).name == "random_legal"
    assert make_policy("random", seed=1).name == "random_legal"
    assert make_policy("greedy", seed=1).name == "greedy_immediate"
    assert make_policy("mcts-lite", seed=1).name == "mcts_lite"
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("not_a_bot", seed=1)


def test_random_legal_matches_play_random_game():
    for seed in (1, 7, 99):
        engine = play_random_game(seed=seed, max_actions=5_000)
        outcome = play_with_policy(seed, RandomLegal(seed), max_actions=5_000)
        assert outcome.terminal
        assert outcome.n_actions == len(engine.action_log)
        assert outcome.actions == tuple(engine.action_log)
        assert outcome.total_score == total_score(engine.sheet)


def test_run_batch_greedy_serial_all_terminal():
    result = run_batch(4, seed_start=30, workers=1, policy="greedy_immediate")
    assert result.policy == "greedy_immediate"
    assert result.unfinished == 0
    assert result.mean_score >= 0


def test_run_batch_heuristic_serial_all_terminal():
    result = run_batch(3, seed_start=40, workers=1, policy="heuristic")
    assert result.policy == "heuristic"
    assert result.unfinished == 0
    assert result.mean_score >= 0


def test_run_batch_mcts_lite_serial_all_terminal():
    result = run_batch(2, seed_start=50, workers=1, policy="mcts_lite")
    assert result.policy == "mcts_lite"
    assert result.unfinished == 0
    assert result.mean_score >= 0


def test_run_batch_serial_all_terminal():
    result = run_batch(12, seed_start=20, workers=1)
    assert result.games == 12
    assert result.workers == 1
    assert result.unfinished == 0
    assert result.total_actions > 0
    assert result.games_per_sec > 0
    assert all(outcome.terminal for outcome in result.outcomes)


def test_run_batch_two_workers_all_terminal():
    result = run_batch(8, seed_start=100, workers=2)
    assert result.games == 8
    assert result.workers == 2
    assert result.unfinished == 0
    assert len(result.outcomes) == 8


def test_format_batch_result_includes_throughput():
    result = run_batch(3, seed_start=0, workers=1)
    text = format_batch_result(result)
    assert "games/s" in text
    assert "actions/s" in text
    assert "unfinished:   0" in text
    assert "unfinished seeds" not in text
    assert "mean yellow:" in text
    assert "mean foxes:" in text
    assert abs(sum(result.mean_area_scores.values()) - result.mean_score) < 1e-9
