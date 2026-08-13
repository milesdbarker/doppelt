"""Yellow circle/cross rules and row/column completion bonuses."""

from doppelt.actions.catalog_v1 import bonus_yellow_circle_id, bonus_yellow_cross_id
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import Bonus
from doppelt.core.scoring import score_yellow
from doppelt.core.types import ActionTrack, BonusKind, Color
from doppelt.engine.bonus_flow import apply_bonus_yellow_cross, try_enter_bonus_phase
from doppelt.engine.bonus_queue import (
    PendingBonus,
    enqueue_bonus,
    enqueue_bonuses_after_yellow_circle,
)
from doppelt.engine.game import apply_action, legal_action_ids, new_game


def test_mark_yellow_circles_then_crosses_same_cell():
    sheet = PlayerSheet.empty()

    assert sheet.mark_yellow(2, 1) == "circle"
    assert sheet.yellow[2].circled
    assert not sheet.yellow[2].crossed

    assert sheet.mark_yellow(2, 1) == "cross"
    assert sheet.yellow[2].crossed
    assert sheet.can_mark_yellow(2, 1) is False


def test_yellow_score_counts_crosses_only():
    sheet = PlayerSheet.empty()
    assert score_yellow(sheet) == 0

    sheet.mark_yellow(2, 1)
    assert score_yellow(sheet) == 0

    sheet.mark_yellow(2, 1)
    assert score_yellow(sheet) == 3


def test_row_bonus_triggers_when_row_fully_circled_not_on_cross():
    state = new_game(seed=1)
    state.sheet.mark_yellow(2, 1)  # row 1, cell 2
    enqueue_bonuses_after_yellow_circle(state, 2)
    assert not state.pending_bonuses

    state.sheet.mark_yellow(3, 2)  # row 1, cell 3 completes row
    enqueue_bonuses_after_yellow_circle(state, 3)
    assert len(state.pending_bonuses) == 1
    assert state.pending_bonuses[0].source == "yellow:row:1"
    assert state.pending_bonuses[0].bonus.kind is BonusKind.UNLOCK

    state.sheet.mark_yellow(2, 1)
    state.sheet.mark_yellow(3, 2)
    state.pending_bonuses = []
    enqueue_bonuses_after_yellow_circle(state, 2)
    enqueue_bonuses_after_yellow_circle(state, 3)
    assert not state.pending_bonuses


def test_column_bonus_triggers_when_column_circled():
    state = new_game(seed=2)
    state.sheet.mark_yellow(2, 1)  # col 0, cell 2
    enqueue_bonuses_after_yellow_circle(state, 2)
    assert not state.pending_bonuses

    state.sheet.mark_yellow(6, 2)  # col 0, cell 6 completes column
    enqueue_bonuses_after_yellow_circle(state, 6)
    assert len(state.pending_bonuses) == 1
    assert state.pending_bonuses[0].source == "yellow:col:0"
    assert state.pending_bonuses[0].bonus.kind is BonusKind.REROLL


def test_row_and_column_bonuses_enqueue_independently():
    state = new_game(seed=3)
    state.sheet.mark_yellow(2, 1)
    state.sheet.mark_yellow(3, 2)
    enqueue_bonuses_after_yellow_circle(state, 3)
    assert len(state.pending_bonuses) == 1
    assert state.pending_bonuses[0].source == "yellow:row:1"

    state.sheet.mark_yellow(7, 5)
    enqueue_bonuses_after_yellow_circle(state, 7)
    assert len(state.pending_bonuses) == 2
    assert state.pending_bonuses[1].source == "yellow:col:2"


def test_yellow_wild_cross_does_not_grant_row_bonus():
    state = new_game(seed=4)
    state.sheet.yellow[2].circled = True
    state.sheet.yellow[3].circled = True
    state.pending_bonuses = [PendingBonus(Bonus(BonusKind.BONUS_WILD, Color.YELLOW), "wild")]
    state.phase = Phase.RESOLVE_BONUS
    apply_bonus_yellow_cross(state, 2)
    assert state.sheet.action_tracks[ActionTrack.UNLOCK].circled == 0


def test_yellow_wild_circle_action_ids_and_row_bonus():
    state = new_game(seed=5)
    state.sheet.yellow[2].circled = True
    enqueue_bonus(state, Bonus(BonusKind.BONUS_WILD, Color.YELLOW), "wild")
    assert try_enter_bonus_phase(state, Phase.ACTIVE_PICK)
    assert bonus_yellow_circle_id(3) in legal_action_ids(state)
    assert bonus_yellow_cross_id(2) in legal_action_ids(state)
    assert bonus_yellow_cross_id(3) not in legal_action_ids(state)

    apply_action(state, bonus_yellow_circle_id(3))
    assert state.sheet.yellow[3].circled
    assert state.sheet.action_tracks[ActionTrack.UNLOCK].circled == 1
