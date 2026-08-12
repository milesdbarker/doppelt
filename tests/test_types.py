"""Tests for Dice, Color, and white-die helpers."""

import pytest

from doppelt.core.types import (
    ALL_DICE,
    SCORING_COLORS,
    WILD_MARK_COLORS,
    Color,
    Dice,
    blue_entry_value,
    resolve_silver_mark_color,
    resolve_white_mark_color,
)


def test_all_dice_count_and_order():
    assert len(ALL_DICE) == 6
    assert ALL_DICE[0] is Dice.WHITE
    assert ALL_DICE[-1] is Dice.SILVER


def test_scoring_colors_excludes_white():
    assert len(SCORING_COLORS) == 5
    assert Color.BLUE in SCORING_COLORS


def test_non_white_dice_sheet_color():
    assert Dice.YELLOW.sheet_color is Color.YELLOW
    assert Dice.WHITE.sheet_color is None


def test_can_mark_area_fixed_colors():
    assert Dice.PINK.can_mark_area(Color.PINK)
    assert not Dice.PINK.can_mark_area(Color.GREEN)
    assert Dice.SILVER.can_mark_area(Color.SILVER)
    assert not Dice.SILVER.can_mark_area(Color.YELLOW)


def test_white_can_mark_wild_areas_only():
    for color in WILD_MARK_COLORS:
        assert Dice.WHITE.can_mark_area(color)
    assert not Dice.WHITE.can_mark_area(Color.BLUE)


def test_blue_entry_with_white_always_sums():
    assert blue_entry_value(blue_face=3, white_face=4, includes_white=True) == 7
    assert blue_entry_value(blue_face=3, white_face=4, includes_white=False) == 3


def test_resolve_white_mark_color():
    assert resolve_white_mark_color(Color.GREEN) is Color.GREEN
    with pytest.raises(ValueError):
        resolve_white_mark_color(Color.BLUE)


def test_resolve_silver_mark_color():
    assert resolve_silver_mark_color(Color.PINK) is Color.PINK
    with pytest.raises(ValueError):
        resolve_silver_mark_color(Color.SILVER)
