"""Silver area marking tests."""

from tests.conftest import roll_hand

from doppelt.actions.catalog_v1 import (
    SILVER_SKIP_CASCADE_ID,
    mark_silver_id,
    passive_platter_id,
    pick_die_id,
)
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.scoring import score_silver
from doppelt.core.silver import silver_row_index
from doppelt.core.types import ALL_DICE, Color, Dice
from doppelt.engine.game import apply_action, is_terminal, legal_action_ids, new_game, play_random_game
from doppelt.engine.silver_flow import skip_unmarkable_cascade_heads


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
    roll_hand(state)
    state.faces[Dice.SILVER] = 4
    apply_action(state, pick_die_id(Dice.SILVER))
    assert state.phase is Phase.ACTIVE_MARK_SILVER
    assert state.pending_silver_values[0] == 4
    assert state.pending_silver_required[0] is True


def test_active_silver_cascade_from_platter_sent():
    state = new_game(seed=11)
    roll_hand(state)
    for die in ALL_DICE:
        state.faces[die] = 6
    state.faces[Dice.SILVER] = 5
    state.faces[Dice.YELLOW] = 2
    state.faces[Dice.GREEN] = 1
    apply_action(state, pick_die_id(Dice.SILVER))
    assert state.pending_silver_values == [5, 2, 1]
    assert state.pending_silver_rows == [None, Color.YELLOW, Color.GREEN]
    assert state.pending_silver_required == [True, False, False]
    mark_id = mark_silver_id(silver_row_index(Color.YELLOW), 5)
    apply_action(state, mark_id)
    assert 5 in state.sheet.silver[Color.YELLOW]

    legal = legal_action_ids(state)
    assert SILVER_SKIP_CASCADE_ID not in legal
    assert mark_silver_id(silver_row_index(Color.YELLOW), 2) in legal
    assert mark_silver_id(silver_row_index(Color.GREEN), 2) not in legal
    apply_action(state, mark_silver_id(silver_row_index(Color.YELLOW), 2))

    legal = legal_action_ids(state)
    assert mark_silver_id(silver_row_index(Color.GREEN), 1) in legal
    assert mark_silver_id(silver_row_index(Color.YELLOW), 1) not in legal


def test_last_pick_silver_cascades_all_leftover_hand_dice():
    """Pick 3 leftover dice (including equals) move to platter and cascade."""
    state = new_game(seed=14)
    roll_hand(state)
    for die in ALL_DICE:
        state.faces[die] = 6
    state.faces[Dice.SILVER] = 4
    state.faces[Dice.YELLOW] = 2
    state.faces[Dice.GREEN] = 4
    state.faces[Dice.PINK] = 6
    state.picks_made = 2
    state.hand = [Dice.SILVER, Dice.YELLOW, Dice.GREEN, Dice.PINK]
    state.platter = [Dice.WHITE, Dice.BLUE]
    state.awaiting_roll = False

    apply_action(state, pick_die_id(Dice.SILVER))

    assert Dice.YELLOW in state.platter
    assert Dice.GREEN in state.platter
    assert Dice.PINK in state.platter
    assert Dice.WHITE in state.platter
    assert Dice.BLUE in state.platter
    assert state.pending_silver_values == [4, 2, 4, 6]
    assert state.pending_silver_rows == [None, Color.YELLOW, Color.GREEN, Color.PINK]


def test_silver_does_not_cascade_dice_already_on_platter():
    state = new_game(seed=15)
    roll_hand(state)
    for die in ALL_DICE:
        state.faces[die] = 6
    state.faces[Dice.SILVER] = 4
    state.faces[Dice.YELLOW] = 2
    state.faces[Dice.GREEN] = 4  # already on platter, equal value
    state.picks_made = 1
    state.hand = [Dice.SILVER, Dice.YELLOW, Dice.PINK, Dice.WHITE]
    state.platter = [Dice.GREEN, Dice.BLUE]
    state.awaiting_roll = False

    apply_action(state, pick_die_id(Dice.SILVER))

    assert Dice.GREEN in state.platter
    assert Dice.YELLOW in state.platter
    assert Dice.PINK in state.hand
    assert state.pending_silver_values == [4, 2]
    assert state.pending_silver_rows == [None, Color.YELLOW]
    assert Color.GREEN not in state.pending_silver_rows



def test_silver_cascade_pink_die_locks_pink_row():
    state = new_game(seed=13)
    roll_hand(state)
    for die in ALL_DICE:
        state.faces[die] = 6
    state.faces[Dice.PINK] = 5
    apply_action(state, pick_die_id(Dice.SILVER))
    apply_action(state, mark_silver_id(silver_row_index(Color.YELLOW), 6))
    legal = legal_action_ids(state)
    assert legal == [mark_silver_id(silver_row_index(Color.PINK), 5)]
    apply_action(state, legal[0])
    assert 5 in state.sheet.silver[Color.PINK]


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


def test_optional_cascade_head_skipped_when_cell_already_marked():
    state = new_game(seed=21)
    roll_hand(state)
    state.phase = Phase.ACTIVE_MARK_SILVER
    state.silver_finish = "active"
    state.picks_made = 1
    state.sheet.mark_silver(1, Color.GREEN)
    state.pending_silver_values = [1, 5]
    state.pending_silver_rows = [Color.GREEN, None]
    state.pending_silver_required = [False, False]

    skip_unmarkable_cascade_heads(state)
    assert state.pending_silver_values == [5]
    assert mark_silver_id(silver_row_index(Color.YELLOW), 5) in legal_action_ids(state)


def test_bonus_occupying_cascade_cell_does_not_stuck_silver():
    """Former fuzz stuck seeds: bonus filled a cascade cell, then no legal silver action."""
    for seed in (12693, 22788, 25419, 29752):
        state = play_random_game(seed=seed, max_actions=5_000)
        done, _ = is_terminal(state)
        assert done, f"seed {seed} stuck in {state.phase.value}"
