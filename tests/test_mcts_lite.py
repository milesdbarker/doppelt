"""MCTS-lite: shallow UCB search with Heuristic rollouts."""

from tests.conftest import roll_hand

from doppelt.actions.catalog_v1 import passive_mark_yellow_id
from doppelt.core.phases import Phase
from doppelt.core.types import ActionTrack, Dice
from doppelt.engine.game import _begin_active_turn, legal_action_ids, new_game
from doppelt.sim import Heuristic, MctsLite, make_policy, play_with_policy


def test_make_policy_mcts_lite():
    policy = make_policy("mcts_lite", seed=1)
    assert policy.name == "mcts_lite"


def test_mcts_lite_single_legal_is_passthrough():
    state = new_game(seed=1)
    assert MctsLite(0, n_sims=4, horizon=2).select(state, [130]) == 130


def test_mcts_lite_trial_does_not_mutate_live_state():
    state = new_game(seed=4)
    state.sheet.circle_action(ActionTrack.REROLL)
    roll_hand(state)
    faces_before = dict(state.faces)
    log_before = list(state.action_log)
    hand_before = list(state.hand)
    legal = legal_action_ids(state)
    assert len(legal) > 1

    MctsLite(0, n_sims=8, horizon=2).select(state, legal)

    assert dict(state.faces) == faces_before
    assert state.action_log == log_before
    assert state.hand == hand_before


def test_mcts_lite_defers_mark_choices_to_heuristic():
    state = new_game(seed=10)
    state.sheet.yellow[0].circled = True
    state.phase = Phase.PASSIVE_MARK_YELLOW
    state.pending_die = Dice.YELLOW
    state.pending_value = 5

    legal = legal_action_ids(state)
    assert passive_mark_yellow_id(7) in legal
    assert passive_mark_yellow_id(8) in legal
    seed = 0
    assert MctsLite(seed, n_sims=4, horizon=2).select(state, legal) == Heuristic(seed).select(
        state, legal
    )


def test_mcts_lite_survives_bonus_with_pending_roll():
    """Round 4 wild: RESOLVE_BONUS + awaiting_roll + full hand must not crash."""
    state = new_game(seed=2)
    state.round_index = 4
    _begin_active_turn(state)
    assert state.phase is Phase.RESOLVE_BONUS
    assert state.awaiting_roll
    assert state.hand
    legal = legal_action_ids(state)
    assert len(legal) > 1
    choice = MctsLite(0, n_sims=8, horizon=2).select(state, legal)
    assert choice in legal


def test_mcts_lite_reaches_terminal():
    outcome = play_with_policy(13, MctsLite(13, n_sims=8, horizon=2), max_actions=5_000)
    assert outcome.terminal
    assert outcome.n_actions > 0


def test_mcts_lite_seed_875_reaches_terminal():
    outcome = play_with_policy(875, MctsLite(875), max_actions=5_000)
    assert outcome.terminal
    assert outcome.n_actions > 0
