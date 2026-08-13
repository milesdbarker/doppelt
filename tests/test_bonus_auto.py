"""Tests for automated bonus resolution."""

from doppelt.core.bonus_auto import (
    apply_auto_mark,
    automated_blue_bonus,
    automated_green_bonus,
    automated_pink_bonus,
    can_automated_wild_bonus,
)
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import get_score_sheet
from doppelt.core.types import Color


def test_blue_bonus_repeats_last_value():
    sheet = PlayerSheet.empty()
    sheet.mark_blue(10)
    sheet.mark_blue(8)
    mark = automated_blue_bonus(sheet)
    assert mark is not None
    assert mark.value == 8
    apply_auto_mark(sheet, mark)
    assert sheet.blue[2] == 8


def test_pink_bonus_always_six():
    sheet = PlayerSheet.empty()
    mark = automated_pink_bonus(sheet)
    assert mark is not None
    assert mark.value == 6
    apply_auto_mark(sheet, mark)
    assert sheet.pink[0] == 6


def test_green_bonus_positive_vs_negative_square():
    sheet = PlayerSheet.empty()
    first = automated_green_bonus(sheet)
    assert first is not None
    assert first.slot == 0
    assert first.value == 12  # 6 × multiplier 2

    apply_auto_mark(sheet, first)
    second = automated_green_bonus(sheet)
    assert second is not None
    assert second.slot == 1
    assert second.value == 2  # 1 × multiplier 2


def _fill_blue_track(sheet: PlayerSheet) -> None:
    sheet.mark_blue(12)
    for value in (11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2):
        sheet.mark_blue(value)


def test_full_blue_track_makes_blue_wild_bonus_impossible():
    sheet = PlayerSheet.empty()
    _fill_blue_track(sheet)
    assert sheet.next_blue_slot() is None
    assert automated_blue_bonus(sheet) is None
    assert not can_automated_wild_bonus(sheet, Color.BLUE)


def test_full_pink_track_makes_pink_wild_bonus_impossible():
    sheet = PlayerSheet.empty()
    for _ in range(get_score_sheet().pink.slot_count):
        sheet.mark_pink(6)
    assert automated_pink_bonus(sheet) is None
    assert not can_automated_wild_bonus(sheet, Color.PINK)


def test_full_green_track_makes_green_wild_bonus_impossible():
    sheet = PlayerSheet.empty()
    for slot in range(get_score_sheet().green.slot_count):
        sheet.mark_green(slot, 1)
    assert automated_green_bonus(sheet) is None
    assert not can_automated_wild_bonus(sheet, Color.GREEN)
