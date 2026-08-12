"""Game phase identifiers."""

from enum import Enum


class Phase(str, Enum):
    ACTIVE_PICK = "active_pick"
    ACTIVE_MARK_YELLOW = "active_mark_yellow"
    ACTIVE_MARK_WHITE = "active_mark_white"
    ACTIVE_MARK_SILVER = "active_mark_silver"
    RESOLVE_BONUS = "resolve_bonus"
    PLUS_ONE = "plus_one"
    PASSIVE_PICK = "passive_pick"
    PASSIVE_MARK_YELLOW = "passive_mark_yellow"
    PLUS_ONE_MARK_YELLOW = "plus_one_mark_yellow"
    GAME_OVER = "game_over"
