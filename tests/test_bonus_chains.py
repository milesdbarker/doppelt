"""Bonus chain resolution — FIFO queue and RESOLVE_BONUS phase."""

from doppelt.actions.catalog_v1 import mark_yellow_id, pick_die_id
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import Bonus
from doppelt.core.types import BonusKind, Color, Dice
from doppelt.engine.bonus_flow import drain_auto_bonus_queue, try_enter_bonus_phase
from doppelt.engine.bonus_queue import PendingBonus, enqueue_bonus
from doppelt.engine.game import apply_action, legal_action_ids, new_game


def test_blue_field_bonus_enters_resolve_phase_for_yellow_wild():
    state = new_game(seed=1)
    state.sheet.mark_blue(10)
    state.sheet.mark_blue(8)
    state.faces[Dice.BLUE] = 3
    state.faces[Dice.WHITE] = 2
    apply_action(state, pick_die_id(Dice.BLUE))
    assert state.sheet.blue[2] == 5
    assert state.phase is Phase.RESOLVE_BONUS
    assert state.pending_bonuses[0].source == "blue:2"
    assert state.pending_bonuses[0].bonus.color is Color.YELLOW


def test_yellow_wild_bonus_requires_player_action():
    sheet = PlayerSheet.empty()
    sheet.mark_blue(10)
    sheet.mark_blue(8)
    sheet.yellow[2].circled = True
    state = new_game(seed=2)
    state.sheet = sheet
    state.faces[Dice.BLUE] = 5
    state.faces[Dice.WHITE] = 3
    apply_action(state, pick_die_id(Dice.BLUE))
    assert state.phase is Phase.RESOLVE_BONUS
    assert state.pending_bonuses[0].bonus.color is Color.YELLOW
    assert mark_yellow_id(2) in legal_action_ids(state)
    apply_action(state, mark_yellow_id(2))
    assert state.sheet.yellow[2].crossed
    assert state.phase is Phase.ACTIVE_PICK


def test_bonus_queue_fifo_yellow_before_fox():
    state = new_game(seed=3)
    state.sheet.yellow[0].circled = True
    enqueue_bonus(state, Bonus(BonusKind.BONUS_WILD, Color.YELLOW), "first")
    enqueue_bonus(state, Bonus(BonusKind.FOX), "second")
    assert try_enter_bonus_phase(state, Phase.ACTIVE_PICK)
    assert state.phase is Phase.RESOLVE_BONUS
    apply_action(state, mark_yellow_id(0))
    assert state.sheet.yellow[0].crossed
    assert state.sheet.foxes == 1
    assert not state.pending_bonuses


def test_fox_bonus_auto_resolves_without_action():
    state = new_game(seed=4)
    state.pending_bonuses = [PendingBonus(Bonus(BonusKind.FOX), "test:fox")]
    drain_auto_bonus_queue(state)
    assert state.sheet.foxes == 1
    assert not state.pending_bonuses


def test_pink_wild_auto_chain_records_events_in_order():
    state = new_game(seed=5)
    for value in (6, 6, 6):
        slot = state.sheet.mark_pink(value)
        if slot == 2:
            break
    enqueue_bonus(
        state,
        Bonus(BonusKind.BONUS_WILD, Color.GREEN),
        "pink:3",
    )
    try_enter_bonus_phase(state, Phase.ACTIVE_PICK)
    assert state.phase is Phase.ACTIVE_PICK
    assert state.sheet.green[0] == 6
    assert state.bonus_events[0].startswith("pink:3:")
