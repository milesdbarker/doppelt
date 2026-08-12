"""Tests for score_sheet_v1.yaml constants."""

from doppelt.core.score_sheet import get_score_sheet
from doppelt.core.types import ActionTrack, BonusKind, Color


def test_loads_default_score_sheet():
    sheet = get_score_sheet()
    assert sheet.version == 1
    assert sheet.name == "doppelt_v1"


def test_solo_rounds_and_grants():
    sheet = get_score_sheet()
    assert sheet.rounds_by_player_count[1] == 6
    grants = sheet.round_start_grants
    assert len(grants) == 6
    assert grants[0] is not None and grants[0].track == ActionTrack.REROLL
    assert grants[4] is None
    assert grants[5] is None


def test_silver_row_scoring_table():
    sheet = get_score_sheet()
    assert sheet.silver.row_score_by_mark_count == (2, 4, 7, 11, 16, 22)


def test_yellow_cross_scoring_table():
    sheet = get_score_sheet()
    assert sheet.yellow.score_by_cross_count == (3, 10, 21, 36, 55, 75, 96, 118, 141, 165)
    assert len(sheet.yellow.cells) == 10


def test_blue_slot_scores():
    sheet = get_score_sheet()
    assert sheet.blue.score_by_last_slot == (1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78)


def test_green_multipliers():
    sheet = get_score_sheet()
    assert sheet.green.multipliers == (1, 2, 2, 2, 1, 3, 3, 3, 2, 3, 1, 4)
    assert sheet.green.pair_count == 6


def test_pink_thresholds():
    sheet = get_score_sheet()
    assert sheet.pink.min_values == (None, None, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6)


def test_action_track_lengths():
    sheet = get_score_sheet()
    assert sheet.action_tracks[ActionTrack.REROLL].slots == 7
    assert sheet.action_tracks[ActionTrack.RETURN_DIE].slots == 6
    assert sheet.action_tracks[ActionTrack.EXTRA_DIE].slots == 6


def test_silver_column_bonuses_include_fox():
    sheet = get_score_sheet()
    fox_bonus = sheet.silver.column_bonuses[2]
    assert fox_bonus is not None
    assert fox_bonus.kind == BonusKind.FOX
    assert sheet.silver.column_bonuses[1] is not None
    assert sheet.silver.column_bonuses[1].color == Color.YELLOW
