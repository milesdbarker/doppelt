"""GreedyImmediate: one-step total_score, no extra heuristics."""

from tests.conftest import roll_hand

from doppelt.actions.catalog_v1 import passive_mark_yellow_id, passive_platter_id, passive_skip_id
from doppelt.core.phases import Phase
from doppelt.core.types import ActionTrack, Dice
from doppelt.engine.game import legal_action_ids, new_game
from doppelt.sim import GreedyImmediate, make_policy, play_with_policy


def test_make_policy_greedy():
    policy = make_policy("greedy_immediate", seed=1)
    assert policy.name == "greedy_immediate"


def test_greedy_prefers_pink_mark_over_skip():
    state = new_game(seed=1)
    state.phase = Phase.PASSIVE_PICK
    state.faces[Dice.PINK] = 6
    state.platter = [Dice.PINK]
    state.passive_pool = []
    state.use_pool_fallback = False

    legal = legal_action_ids(state)
    assert passive_platter_id(Dice.PINK) in legal
    assert passive_skip_id() in legal

    choice = GreedyImmediate(0).select(state, legal)
    assert choice == passive_platter_id(Dice.PINK)


def test_greedy_prefers_yellow_cross_over_circle():
    state = new_game(seed=2)
    state.sheet.yellow[0].circled = True  # cell 0 value 3
    state.phase = Phase.PASSIVE_MARK_YELLOW
    state.pending_die = Dice.YELLOW
    state.pending_value = 3

    legal = legal_action_ids(state)
    assert passive_mark_yellow_id(0) in legal
    assert passive_mark_yellow_id(5) in legal

    choice = GreedyImmediate(0).select(state, legal)
    assert choice == passive_mark_yellow_id(0)


def test_greedy_trial_does_not_mutate_live_state():
    state = new_game(seed=3)
    state.sheet.circle_action(ActionTrack.REROLL)
    roll_hand(state)
    faces_before = dict(state.faces)
    log_before = list(state.action_log)
    legal = legal_action_ids(state)
    assert len(legal) > 1

    GreedyImmediate(0).select(state, legal)

    assert dict(state.faces) == faces_before
    assert state.action_log == log_before


def test_greedy_immediate_reaches_terminal():
    outcome = play_with_policy(7, GreedyImmediate(7), max_actions=5_000)
    assert outcome.terminal
    assert outcome.n_actions > 0
