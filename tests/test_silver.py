"""Silver area marking tests."""

from doppelt.actions.catalog_v1 import (
    mark_silver_id,
    passive_platter_id,
    pick_die_id,
)
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.scoring import score_silver
from doppelt.core.silver import silver_row_index
from doppelt.core.types import ALL_DICE, Color, Dice
from doppelt.engine.game import apply_action, new_game


def test_mark_silver_tracks_row_and_column():
    sheet = PlayerSheet.empty()
    sheet.mark_silver(3, Color.YELLOW)
    sheet.mark_silver(3, Color.BLUE)
    assert 3 in sheet.silver[Color.YELLOW]
    assert 3 in sheet.silver[Color.BLUE]
    assert not sheet.silver_column_complete(3)


def test_silver_row_scoring_uses_mark_count():
    sheet = PlayerSheet.empty()
    sheet.mark_silver(1, Color.YELLOW)
    sheet.mark_silver(2, Color.YELLOW)
    assert score_silver(sheet) == 4  # 2 marks in yellow row


def test_active_silver_pick_starts_resolution():
    state = new_game(seed=10)
    state.faces[Dice.SILVER] = 4
    apply_action(state, pick_die_id(Dice.SILVER))
    assert state.phase is Phase.ACTIVE_MARK_SILVER
    assert state.pending_silver_values[0] == 4
    assert state.pending_silver_required[0] is True


def test_active_silver_cascade_from_platter_sent():
    state = new_game(seed=11)
    for die in ALL_DICE:
        state.faces[die] = 6
    state.faces[Dice.SILVER] = 5
    state.faces[Dice.YELLOW] = 2
    state.faces[Dice.GREEN] = 1
    apply_action(state, pick_die_id(Dice.SILVER))
    assert state.pending_silver_values == [5, 2, 1]
    assert state.pending_silver_required == [True, False, False]
    mark_id = mark_silver_id(silver_row_index(Color.YELLOW), 5)
    apply_action(state, mark_id)
    assert 5 in state.sheet.silver[Color.YELLOW]


def test_passive_silver_single_mark_no_cascade():
    state = new_game(seed=12)
    state.phase = Phase.PASSIVE_PICK
    state.faces[Dice.SILVER] = 3
    state.platter = [Dice.SILVER]
    state.passive_pool = []
    state.use_pool_fallback = False
    apply_action(state, passive_platter_id(Dice.SILVER))
    assert state.phase is Phase.ACTIVE_MARK_SILVER
    assert state.pending_silver_values == [3]
    assert state.silver_finish == "passive"
    mark_id = mark_silver_id(silver_row_index(Color.PINK), 3)
    apply_action(state, mark_id)
    assert 3 in state.sheet.silver[Color.PINK]
    assert state.phase is Phase.ACTIVE_PICK
