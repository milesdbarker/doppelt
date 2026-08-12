"""Shared enums and types for the score sheet and engine."""

from enum import Enum


class SheetColor(str, Enum):
  """Colors used on the score sheet (excludes white wild die)."""

  YELLOW = "yellow"
  BLUE = "blue"
  GREEN = "green"
  PINK = "pink"
  SILVER = "silver"


class DieColor(str, Enum):
  """Physical dice colors."""

  WHITE = "white"
  YELLOW = "yellow"
  BLUE = "blue"
  GREEN = "green"
  PINK = "pink"
  SILVER = "silver"


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
