"""Yellow cell choice — passive and plus-one use catalog IDs 151–170."""

from doppelt.actions.catalog_v1 import (
    ACTION_SPACE_SIZE,
    ActionKind,
    decode_action,
    encode_action,
    passive_mark_yellow_id,
    passive_platter_id,
    plus_one_mark_yellow_id,
)
from doppelt.core.phases import Phase
from doppelt.core.types import ActionTrack, Dice
from doppelt.engine.game import (
    _mark_plus_one_yellow_cell,
    _start_plus_one_yellow_mark,
    apply_action,
    begin_passive_turn,
    legal_action_ids,
    new_game,
)


def test_passive_and_plus_one_yellow_catalog_ids():
    assert passive_mark_yellow_id(0) == 151
    assert passive_mark_yellow_id(9) == 160
    assert plus_one_mark_yellow_id(0) == 161
    assert plus_one_mark_yellow_id(9) == 170
    assert ACTION_SPACE_SIZE == 192

    passive_action = decode_action(passive_mark_yellow_id(3))
    assert passive_action.kind is ActionKind.MARK_YELLOW_PASSIVE
    assert passive_action.yellow_cell_id == 3
    assert encode_action(passive_action) == passive_mark_yellow_id(3)

    plus_one_action = decode_action(plus_one_mark_yellow_id(7))
    assert plus_one_action.kind is ActionKind.MARK_YELLOW_PLUS_ONE
    assert plus_one_action.yellow_cell_id == 7


def test_passive_yellow_enters_choice_phase_for_duplicate_values():
    state = new_game(seed=1)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    begin_passive_turn(state)
    state.faces[Dice.YELLOW] = 3
    state.platter = [Dice.YELLOW]
    state.passive_pool = []
    state.use_pool_fallback = False

    apply_action(state, passive_platter_id(Dice.YELLOW))

    assert state.phase is Phase.PASSIVE_MARK_YELLOW
    assert passive_mark_yellow_id(0) in legal_action_ids(state)
    assert passive_mark_yellow_id(5) in legal_action_ids(state)

    apply_action(state, passive_mark_yellow_id(5))

    assert state.sheet.yellow[5].circled
    assert state.phase is Phase.PLUS_ONE


def test_plus_one_yellow_enters_choice_phase_for_duplicate_values():
    state = new_game(seed=2)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    state.phase = Phase.PLUS_ONE
    state.faces[Dice.YELLOW] = 2

    _start_plus_one_yellow_mark(state)

    assert state.phase is Phase.PLUS_ONE_MARK_YELLOW
    assert plus_one_mark_yellow_id(3) in legal_action_ids(state)
    assert plus_one_mark_yellow_id(6) in legal_action_ids(state)

    state.pending_value = 2
    _mark_plus_one_yellow_cell(state, 6)

    assert state.sheet.yellow[6].circled
