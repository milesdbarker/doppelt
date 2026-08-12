"""Silver grid marking helpers."""

from __future__ import annotations

from doppelt.core.types import Color

SILVER_ROW_COLORS: tuple[Color, ...] = (
    Color.YELLOW,
    Color.BLUE,
    Color.GREEN,
    Color.PINK,
)


def silver_row_index(color: Color) -> int:
    if color not in SILVER_ROW_COLORS:
        raise ValueError(f"not a silver row color: {color.value}")
    return SILVER_ROW_COLORS.index(color)


def silver_row_color(row_index: int) -> Color:
    return SILVER_ROW_COLORS[row_index]
