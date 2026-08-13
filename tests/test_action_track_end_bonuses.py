"""Action track end bonuses when the last slot is circled."""

from doppelt.core.phases import Phase
from doppelt.core.score_sheet import Bonus, get_score_sheet
from doppelt.core.types import ActionTrack, BonusKind, Color
from doppelt.engine.action_flow import circle_action_track
from doppelt.engine.bonus_flow import drain_auto_bonus_queue, try_enter_bonus_phase
from doppelt.engine.bonus_queue import enqueue_bonus
from doppelt.engine.game import new_game


def test_score_sheet_action_track_end_bonuses():
    sheet = get_score_sheet()
    assert sheet.action_tracks[ActionTrack.REROLL].end_bonus is not None
    assert sheet.action_tracks[ActionTrack.REROLL].end_bonus.kind is BonusKind.FOX
    assert sheet.action_tracks[ActionTrack.UNLOCK].end_bonus.color is Color.PINK
    assert sheet.action_tracks[ActionTrack.PLUS_ONE].end_bonus.color is Color.SILVER


def test_reroll_track_end_bonus_grants_fox():
    state = new_game(seed=1)
    assert state.sheet.action_tracks[ActionTrack.REROLL].circled == 1
    for index in range(5):
        enqueue_bonus(state, Bonus(BonusKind.REROLL), f"fill:{index}")
    drain_auto_bonus_queue(state)
    assert state.sheet.action_tracks[ActionTrack.REROLL].circled == 6
    assert state.sheet.foxes == 1


def test_unlock_track_end_bonus_grants_pink_wild():
    state = new_game(seed=2)
    for _ in range(6):
        circle_action_track(state, ActionTrack.UNLOCK)
    assert len(state.pending_bonuses) == 1
    assert state.pending_bonuses[0].source == "action:unlock:end"
    assert state.pending_bonuses[0].bonus.color is Color.PINK
    drain_auto_bonus_queue(state)
    assert state.sheet.pink[0] == 6


def test_plus_one_track_end_bonus_enters_silver_wild_phase():
    state = new_game(seed=3)
    for _ in range(6):
        circle_action_track(state, ActionTrack.PLUS_ONE)
    assert state.pending_bonuses[0].bonus.color is Color.SILVER
    assert try_enter_bonus_phase(state, Phase.ACTIVE_PICK)
    assert state.phase is Phase.RESOLVE_BONUS
    assert state.pending_bonuses[0].bonus.color is Color.SILVER


def test_end_bonus_only_fires_once_when_track_full():
    state = new_game(seed=4)
    for _ in range(6):
        circle_action_track(state, ActionTrack.PLUS_ONE)
    assert len(state.pending_bonuses) == 1
    circle_action_track(state, ActionTrack.PLUS_ONE)
    assert len(state.pending_bonuses) == 1
