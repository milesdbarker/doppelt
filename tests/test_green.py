"""Green area marking tests."""

from doppelt.actions.catalog_v1 import pick_die_id
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.types import Dice
from doppelt.engine.game import apply_action, new_game


def test_green_die_multiplies_by_slot_factor():
    sheet = PlayerSheet.empty()
    slot = sheet.mark_green_die(5)
    assert slot == 0
    assert sheet.green[0] == 5  # 5 × multiplier 1


def test_green_pair_star_is_difference():
    sheet = PlayerSheet.empty()
    sheet.mark_green_die(5)  # slot 0: 5 × 1
    sheet.mark_green_die(1)  # slot 1: 1 × 2
    assert sheet.green[0] == 5
    assert sheet.green[1] == 2
    assert sheet.green_stars[0] == 3


def test_active_green_pick_in_engine():
    state = new_game(seed=99)
    state.faces[Dice.GREEN] = 4
    apply_action(state, pick_die_id(Dice.GREEN))
    assert state.sheet.green[0] == 4
