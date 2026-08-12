"""Game phase identifiers."""

from enum import Enum


class Phase(str, Enum):
    ACTIVE_PICK = "active_pick"
    ACTIVE_MARK_YELLOW = "active_mark_yellow"
    ACTIVE_MARK_WHITE = "active_mark_white"
    ACTIVE_MARK_SILVER = "active_mark_silver"
    RESOLVE_BONUS = "resolve_bonus"
    PASSIVE_PICK = "passive_pick"
    GAME_OVER = "game_over"
