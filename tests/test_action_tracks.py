"""Action track state and usage tests."""

from tests.conftest import roll_hand

from doppelt.actions.catalog_v1 import (
    end_active_turn_id,
    pick_die_id,
    roll_hand_id,
    unlock_platter_id,
    use_reroll_id,
)
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import Bonus
from doppelt.core.types import ActionTrack, BonusKind, Dice
from doppelt.engine.bonus_flow import drain_auto_bonus_queue
from doppelt.engine.bonus_queue import enqueue_bonus
from doppelt.engine.game import advance_round_or_game_over, apply_action, legal_action_ids, new_game


def test_action_bonus_circles_track():
    sheet = PlayerSheet.empty()
    state = new_game(seed=1)
    state.sheet = sheet
    enqueue_bonus(state, Bonus(BonusKind.REROLL), "test")
    drain_auto_bonus_queue(state)
    assert sheet.action_tracks[ActionTrack.REROLL].circled == 1
    assert sheet.action_tracks[ActionTrack.REROLL].crossed == 0


def test_round_one_grants_reroll_track():
    state = new_game(seed=2)
    assert state.sheet.action_tracks[ActionTrack.REROLL].circled == 1


def test_round_two_start_grants_plus_one_not_unlock():
    state = new_game(seed=32)
    advance_round_or_game_over(state)
    assert state.round_index == 2
    assert state.sheet.action_tracks[ActionTrack.PLUS_ONE].circled == 1
    assert state.sheet.action_tracks[ActionTrack.UNLOCK].circled == 0


def test_round_three_start_grants_unlock_track():
    state = new_game(seed=32)
    state.round_index = 2
    advance_round_or_game_over(state)
    assert state.round_index == 3
    assert state.sheet.action_tracks[ActionTrack.UNLOCK].circled == 1


def test_unlock_before_roll_pulls_from_platter():
    state = new_game(seed=3)
    state.sheet.circle_action(ActionTrack.UNLOCK)
    state.platter = [Dice.YELLOW]
    state.hand = [Dice.BLUE, Dice.GREEN, Dice.PINK, Dice.WHITE, Dice.SILVER]
    state.awaiting_roll = True
    apply_action(state, unlock_platter_id(Dice.YELLOW))
    assert Dice.YELLOW in state.hand
    assert Dice.YELLOW not in state.platter
    assert state.sheet.action_tracks[ActionTrack.UNLOCK].crossed == 1
    assert state.awaiting_roll is True


def test_reroll_after_roll_rerolls_hand():
    state = new_game(seed=4)
    roll_hand(state)
    apply_action(state, use_reroll_id())
    assert state.sheet.action_tracks[ActionTrack.REROLL].crossed == 1
    assert state.sheet.action_tracks[ActionTrack.REROLL].circled == 0
    assert state.awaiting_roll is False


def test_multiple_reroll_circles_allow_multiple_uses_before_picking():
    state = new_game(seed=41)
    state.sheet.circle_action(ActionTrack.REROLL)
    roll_hand(state)
    assert state.sheet.action_tracks[ActionTrack.REROLL].circled == 2
    apply_action(state, use_reroll_id())
    assert use_reroll_id() in legal_action_ids(state)
    apply_action(state, use_reroll_id())
    assert use_reroll_id() not in legal_action_ids(state)
    assert state.sheet.action_tracks[ActionTrack.REROLL].crossed == 2


def test_active_pick_requires_roll_first():
    state = new_game(seed=5)
    assert state.awaiting_roll is True
    assert roll_hand_id() in legal_action_ids(state)
    roll_hand(state)
    state.faces[Dice.PINK] = 4
    apply_action(state, pick_die_id(Dice.PINK))
    assert state.sheet.pink[0] == 4


def test_empty_hand_with_unlock_offers_unlock_or_end_turn():
    state = new_game(seed=6)
    state.sheet.circle_action(ActionTrack.UNLOCK)
    state.hand = []
    state.platter = [Dice.YELLOW, Dice.BLUE]
    state.picks_made = 1
    state.awaiting_roll = True
    state.phase = Phase.ACTIVE_PICK

    legal = legal_action_ids(state)
    assert end_active_turn_id() in legal
    assert unlock_platter_id(Dice.YELLOW) in legal
    assert roll_hand_id() not in legal


def test_end_active_turn_finishes_without_extra_pick():
    state = new_game(seed=7)
    state.sheet.circle_action(ActionTrack.UNLOCK)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    state.hand = []
    state.platter = [Dice.GREEN]
    state.picks_made = 2
    state.awaiting_roll = True
    state.phase = Phase.ACTIVE_PICK

    apply_action(state, end_active_turn_id())
    assert state.phase is Phase.PLUS_ONE
    assert state.picks_made == 2


def test_unlock_after_empty_hand_allows_another_roll_and_pick():
    state = new_game(seed=8)
    state.sheet.circle_action(ActionTrack.UNLOCK)
    state.hand = []
    state.platter = [Dice.PINK]
    state.picks_made = 1
    state.awaiting_roll = True
    state.phase = Phase.ACTIVE_PICK

    apply_action(state, unlock_platter_id(Dice.PINK))
    assert state.hand == [Dice.PINK]
    assert state.awaiting_roll is True
    assert roll_hand_id() in legal_action_ids(state)

    apply_action(state, roll_hand_id())
    value = state.faces[Dice.PINK]
    apply_action(state, pick_die_id(Dice.PINK))
    assert state.sheet.pink[0] == value
    assert state.picks_made == 2


def test_empty_hand_without_unlock_still_ends_turn():
    state = new_game(seed=9)
    state.hand = []
    state.platter = [Dice.YELLOW]
    state.picks_made = 1
    state.awaiting_roll = True
    state.phase = Phase.ACTIVE_PICK

    assert end_active_turn_id() not in legal_action_ids(state)
    assert unlock_platter_id(Dice.YELLOW) not in legal_action_ids(state)
