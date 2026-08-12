"""Tests for automated bonus resolution."""

from doppelt.core.bonus_auto import (
  apply_auto_mark,
  automated_blue_bonus,
  automated_green_bonus,
  automated_pink_bonus,
)
from doppelt.core.player_sheet import PlayerSheet


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
  assert first.value == 6  # 6 × multiplier 1

  apply_auto_mark(sheet, first)
  second = automated_green_bonus(sheet)
  assert second is not None
  assert second.slot == 1
  assert second.value == 2  # 1 × multiplier 2
