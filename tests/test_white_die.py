"""White die — yellow and pink modes."""

from tests.conftest import roll_hand

from doppelt.actions.catalog_v1 import (
    ACTION_SPACE_SIZE,
    mark_white_pink_id,
    mark_white_yellow_id,
    pick_die_id,
)
from doppelt.core.phases import Phase
from doppelt.core.types import Dice
from doppelt.engine.game import apply_action, legal_action_ids, new_game


def test_white_mark_modes_include_yellow_and_pink():
    assert mark_white_yellow_id() == 23
    assert mark_white_pink_id() == 24
    assert ACTION_SPACE_SIZE == 192


def test_active_white_pink_marks_pink_track():
    state = new_game(seed=1)
    roll_hand(state)
    state.faces[Dice.WHITE] = 4
    state.faces[Dice.BLUE] = 1
    apply_action(state, pick_die_id(Dice.WHITE))
    assert state.phase is Phase.ACTIVE_MARK_WHITE
    assert mark_white_pink_id() in legal_action_ids(state)
    apply_action(state, mark_white_pink_id())
    assert state.sheet.pink[0] == 4


def test_active_white_yellow_enters_cell_choice():
    state = new_game(seed=2)
    roll_hand(state)
    state.faces[Dice.WHITE] = 3
    state.faces[Dice.BLUE] = 6
    apply_action(state, pick_die_id(Dice.WHITE))
    apply_action(state, mark_white_yellow_id())
    assert state.phase is Phase.ACTIVE_MARK_YELLOW
    assert state.pending_die is Dice.WHITE


def test_active_white_yellow_single_cell_marks_directly():
    state = new_game(seed=3)
    roll_hand(state)
    state.faces[Dice.WHITE] = 1
    state.faces[Dice.BLUE] = 6
    apply_action(state, pick_die_id(Dice.WHITE))
    apply_action(state, mark_white_yellow_id())
    assert state.sheet.yellow[2].circled
    assert not state.sheet.yellow[2].crossed
