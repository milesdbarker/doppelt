"""Engine integration tests — Phase 1 milestone M1.1."""

import pytest
from tests.conftest import roll_hand

from doppelt.actions.catalog_v1 import pick_die_id
from doppelt.core.phases import Phase
from doppelt.core.types import Dice
from doppelt.engine.game import (
    apply_action,
    is_terminal,
    new_game,
    play_random_game,
)


def test_new_game_starts_active_with_hand():
    state = new_game(seed=1)
    assert state.phase is Phase.ACTIVE_PICK
    assert len(state.hand) == 6
    assert state.round_index == 1
    assert state.awaiting_roll is True


def test_apply_action_rejects_illegal_by_default():
    state = new_game(seed=1)
    with pytest.raises(ValueError, match="illegal action"):
        apply_action(state, pick_die_id(Dice.BLUE))


def test_blue_pick_uses_white_sum():
    state = new_game(seed=2)
    roll_hand(state)
    state.faces[Dice.BLUE] = 3
    state.faces[Dice.WHITE] = 4
    apply_action(state, pick_die_id(Dice.BLUE))
    assert state.sheet.blue[0] == 7


def test_random_game_reaches_terminal():
    state = play_random_game(seed=123, max_actions=5000)
    done, scores = is_terminal(state)
    assert done
    assert state.phase is Phase.GAME_OVER
    assert scores["blue"] >= 0
    assert "green" in scores
