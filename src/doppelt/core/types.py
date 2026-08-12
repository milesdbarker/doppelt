"""Shared enums and types for the score sheet and engine."""

from enum import Enum


class Color(str, Enum):
    """Scoring-area colors on the sheet (five areas; white has no area)."""

    YELLOW = "yellow"
    BLUE = "blue"
    GREEN = "green"
    PINK = "pink"
    SILVER = "silver"


# Alias kept for score-sheet YAML field names and early code.
SheetColor = Color


class Dice(str, Enum):
    """The six physical dice in the game."""

    WHITE = "white"
    YELLOW = "yellow"
    BLUE = "blue"
    GREEN = "green"
    PINK = "pink"
    SILVER = "silver"

    @property
    def is_wild(self) -> bool:
        return self is Dice.WHITE

    @property
    def sheet_color(self) -> Color | None:
        """Fixed sheet color for non-white dice; None for white (player chooses)."""
        if self is Dice.WHITE:
            return None
        if self is Dice.SILVER:
            return Color.SILVER
        return Color(self.value)

    def can_mark_area(self, area: Color) -> bool:
        """Whether this die may be placed in the given scoring area (before white choice)."""
        if self is Dice.WHITE:
            return area in WILD_MARK_COLORS
        if self is Dice.SILVER:
            return area is Color.SILVER
        return Color(self.value) is area


class WhiteDieMode(str, Enum):
    """How the white die is used on a turn."""

    AS_COLOR = "as_color"
    AS_BLUE_SUM = "as_blue_sum"


class ActionTrack(str, Enum):
    REROLL = "reroll"
    RETURN_DIE = "return_die"
    EXTRA_DIE = "extra_die"


class BonusKind(str, Enum):
    REROLL = "reroll"
    RETURN_DIE = "return_die"
    EXTRA_DIE = "extra_die"
    BONUS_WILD = "bonus_wild"
    FOX = "fox"
    BLUE_WHITE_SUM_HINT = "blue_white_sum_hint"
    ACTION_TRACK_UNLOCK = "action_track_unlock"


ALL_DICE: tuple[Dice, ...] = tuple(Dice)
"""All six dice in a fixed order (matches catalog / RNG indexing later)."""

SCORING_COLORS: tuple[Color, ...] = (
    Color.YELLOW,
    Color.BLUE,
    Color.GREEN,
    Color.PINK,
    Color.SILVER,
)

# Colors the white die may impersonate when marking (not blue-sum mode).
WILD_MARK_COLORS: frozenset[Color] = frozenset(
    {Color.YELLOW, Color.GREEN, Color.PINK, Color.SILVER}
)

# Colors selectable when resolving a silver-area mark (value × color grid).
SILVER_MARK_COLORS: frozenset[Color] = frozenset(
    {Color.YELLOW, Color.BLUE, Color.GREEN, Color.PINK}
)


def blue_entry_value(*, blue_face: int, white_face: int, includes_white: bool) -> int:
    """Sum written in the blue track when blue and/or white dice are used.

    Rule: if either blue or white is used for a blue entry, always add both face
    values (white is counted even if it sits on the platter or an active slot).
    """
    if includes_white:
        return blue_face + white_face
    return blue_face


def resolve_white_mark_color(chosen: Color) -> Color:
    """Validate and return the sheet color chosen for a white-die color mark."""
    if chosen not in WILD_MARK_COLORS:
        raise ValueError(f"white die cannot mark {chosen.value} area as color")
    return chosen


def resolve_silver_mark_color(chosen: Color) -> Color:
    """Validate color choice when marking the silver grid."""
    if chosen not in SILVER_MARK_COLORS:
        raise ValueError(f"silver mark color must be yellow/blue/green/pink, not {chosen.value}")
    return chosen
