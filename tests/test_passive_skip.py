"""Passive player may skip a die and mark nothing (catalog ID 46)."""

from doppelt.actions.catalog_v1 import (
    ACTION_SPACE_SIZE,
    ActionKind,
    decode_action,
    encode_action,
    mark_white_pink_id,
    passive_mark_yellow_id,
    passive_platter_id,
    passive_skip_id,
)
from doppelt.core.phases import Phase
from doppelt.core.types import ActionTrack, Dice
from doppelt.engine.game import apply_action, begin_passive_turn, legal_action_ids, new_game


def test_passive_skip_catalog_id():
    assert passive_skip_id() == 46
    assert ACTION_SPACE_SIZE == 192
    action = decode_action(46)
    assert action.kind is ActionKind.PASSIVE_SKIP
    assert encode_action(action) == 46


def test_passive_skip_on_pick_marks_nothing():
    state = new_game(seed=1)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    begin_passive_turn(state)
    state.faces[Dice.PINK] = 4
    state.platter = [Dice.PINK]
    state.passive_pool = [Dice.GREEN]
    state.use_pool_fallback = False

    legal = legal_action_ids(state)
    assert passive_platter_id(Dice.PINK) in legal
    assert passive_skip_id() in legal

    apply_action(state, passive_skip_id())

    assert state.sheet.pink[0] is None
    assert state.phase is Phase.PLUS_ONE


def test_passive_skip_when_no_legal_die():
    state = new_game(seed=2)
    state.phase = Phase.PASSIVE_PICK
    state.platter = []
    state.passive_pool = []
    state.use_pool_fallback = False

    assert legal_action_ids(state) == [passive_skip_id()]
    apply_action(state, passive_skip_id())
    assert state.round_index == 2
    assert state.phase is Phase.ACTIVE_PICK


def test_passive_skip_not_legal_after_yellow_cell_choice():
    state = new_game(seed=3)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    begin_passive_turn(state)
    state.faces[Dice.YELLOW] = 3
    state.platter = [Dice.YELLOW]
    state.passive_pool = []
    state.use_pool_fallback = False

    apply_action(state, passive_platter_id(Dice.YELLOW))
    assert state.phase is Phase.PASSIVE_MARK_YELLOW
    assert passive_mark_yellow_id(0) in legal_action_ids(state)
    assert passive_skip_id() not in legal_action_ids(state)


def test_passive_skip_not_legal_after_white_mode_choice():
    state = new_game(seed=4)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    begin_passive_turn(state)
    state.faces[Dice.WHITE] = 4
    state.faces[Dice.BLUE] = 2
    state.platter = [Dice.WHITE]
    state.passive_pool = []
    state.use_pool_fallback = False

    apply_action(state, passive_platter_id(Dice.WHITE))
    assert state.phase is Phase.ACTIVE_MARK_WHITE
    assert mark_white_pink_id() in legal_action_ids(state)
    assert passive_skip_id() not in legal_action_ids(state)
