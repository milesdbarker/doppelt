"""Tree PUCT helpers (no torch)."""

import random

from doppelt.actions.catalog_v1 import ROLL_HAND_ID
from doppelt.engine.game import apply_action, legal_action_ids, new_game
from doppelt.ml.mcts import _apply_forced_uniques, _only_roll_legal, puct_search
from tests.conftest import roll_hand


def test_forced_uniques_do_not_auto_roll():
    state = new_game(seed=8)
    roll_hand(state)
    legal = legal_action_ids(state)
    pick = next(action_id for action_id in legal if 1 <= action_id <= 6)
    apply_action(state, pick, check_legal=False)
    _apply_forced_uniques(state)
    if not _only_roll_legal(state):
        return
    assert ROLL_HAND_ID in legal_action_ids(state)
    assert state.awaiting_roll
    faces = dict(state.faces)
    _apply_forced_uniques(state)
    assert dict(state.faces) == faces
    assert state.awaiting_roll


def test_puct_search_max_plies_one_visits():
    state = new_game(seed=3)
    roll_hand(state)
    legal = legal_action_ids(state)
    if len(legal) < 2:
        return
    priors = {action_id: 1.0 / len(legal) for action_id in legal}
    rng = random.Random(0)
    result = puct_search(
        state,
        legal,
        priors,
        lambda trial: 0.5,
        rng,
        n_sims=12,
        max_plies=1,
    )
    assert result.action_id in legal
    assert sum(result.visits) == 12
    assert dict(state.faces)
    log_before = list(state.action_log)
    puct_search(state, legal, priors, lambda trial: 0.5, rng, n_sims=4, max_plies=2)
    assert state.action_log == log_before
