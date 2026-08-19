"""White die — yellow, pink, and silver modes."""

from tests.conftest import roll_hand

from doppelt.actions.catalog_v1 import (
    ACTION_SPACE_SIZE,
    mark_white_pink_id,
    mark_white_silver_id,
    mark_white_yellow_id,
    pick_die_id,
)
from doppelt.core.phases import Phase
from doppelt.core.types import ALL_DICE, Dice
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


def test_white_as_silver_keeps_higher_silver_die_in_hand():
    """White 2 as silver must not push a silver 4 to the platter (strictly lower only)."""
    state = new_game(seed=20)
    roll_hand(state)
    for die in ALL_DICE:
        state.faces[die] = 6
    state.faces[Dice.WHITE] = 2
    state.faces[Dice.SILVER] = 4
    state.picks_made = 0
    state.hand = list(ALL_DICE)
    state.platter = []
    state.awaiting_roll = False

    apply_action(state, pick_die_id(Dice.WHITE))
    if state.phase is Phase.ACTIVE_MARK_WHITE:
        apply_action(state, mark_white_silver_id())

    assert Dice.SILVER in state.hand
    assert Dice.SILVER not in state.platter
    assert state.pending_silver_values == [2]
    assert state.pending_silver_required == [True]


def test_white_as_silver_cascades_lower_silver_die_as_joker():
    state = new_game(seed=21)
    roll_hand(state)
    for die in ALL_DICE:
        state.faces[die] = 6
    state.faces[Dice.WHITE] = 5
    state.faces[Dice.SILVER] = 3
    state.picks_made = 0
    state.hand = list(ALL_DICE)
    state.platter = []
    state.awaiting_roll = False

    apply_action(state, pick_die_id(Dice.WHITE))
    if state.phase is Phase.ACTIVE_MARK_WHITE:
        apply_action(state, mark_white_silver_id())

    assert Dice.SILVER in state.platter
    assert Dice.SILVER not in state.hand
    assert state.pending_silver_values == [5, 3]
    assert state.pending_silver_rows == [None, None]  # silver die is a joker row


def test_passive_pick_leaves_die_on_platter():
    """Passive uses a platter face without taking the die off the silver platter."""
    from doppelt.actions.catalog_v1 import passive_platter_id
    from doppelt.core.types import ActionTrack
    from doppelt.engine.game import begin_passive_turn

    state = new_game(seed=22)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    begin_passive_turn(state)
    state.faces[Dice.PINK] = 4
    state.platter = [Dice.PINK, Dice.GREEN]
    state.passive_pool = []
    state.use_pool_fallback = False

    apply_action(state, passive_platter_id(Dice.PINK))

    assert state.sheet.pink[0] == 4
    assert Dice.PINK in state.platter
    assert Dice.GREEN in state.platter


def test_passive_white_enters_mode_choice_when_multiple_legal():
    from doppelt.actions.catalog_v1 import mark_white_blue_id, mark_white_pink_id, passive_platter_id
    from doppelt.core.types import ActionTrack
    from doppelt.engine.game import begin_passive_turn

    state = new_game(seed=10)
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
    assert mark_white_blue_id() in legal_action_ids(state)

    apply_action(state, mark_white_pink_id())
    assert state.sheet.pink[0] == 4
    assert state.phase is Phase.PLUS_ONE
