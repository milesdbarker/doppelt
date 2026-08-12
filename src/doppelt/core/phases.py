"""Game phase identifiers."""

from enum import Enum


class Phase(str, Enum):
  ACTIVE_PICK = "active_pick"
  ACTIVE_MARK_YELLOW = "active_mark_yellow"
  PASSIVE_PICK = "passive_pick"
  GAME_OVER = "game_over"
