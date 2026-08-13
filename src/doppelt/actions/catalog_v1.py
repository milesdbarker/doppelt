"""Action catalog v1 — global uint16 action IDs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from doppelt.core.silver import silver_row_color
from doppelt.core.types import SCORING_COLORS, Color, Dice


class ActionKind(str, Enum):
    PICK_DIE = "pick_die"
    MARK_YELLOW = "mark_yellow"
    MARK_YELLOW_PASSIVE = "mark_yellow_passive"
    MARK_YELLOW_PLUS_ONE = "mark_yellow_plus_one"
    MARK_WHITE_BLUE = "mark_white_blue"
    MARK_WHITE_GREEN = "mark_white_green"
    MARK_WHITE_SILVER = "mark_white_silver"
    MARK_WHITE_YELLOW = "mark_white_yellow"
    MARK_WHITE_PINK = "mark_white_pink"
    MARK_SILVER = "mark_silver"
    SILVER_SKIP_CASCADE = "silver_skip_cascade"
    FORFEIT_PICK = "forfeit_pick"
    PASSIVE_PICK = "passive_pick"
    PASSIVE_SKIP = "passive_skip"
    ROLL_HAND = "roll_hand"
    USE_REROLL = "use_reroll"
    UNLOCK_PLATTER = "unlock_platter"
    END_PLUS_ONE = "end_plus_one"
    END_ACTIVE_TURN = "end_active_turn"
    PLUS_ONE_PICK = "plus_one_pick"
    CHOOSE_WILD_COLOR = "choose_wild_color"
    BONUS_YELLOW_CIRCLE = "bonus_yellow_circle"
    BONUS_YELLOW_CROSS = "bonus_yellow_cross"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    die: Dice | None = None
    yellow_cell_id: int | None = None
    silver_row_index: int | None = None
    silver_value: int | None = None
    passive_from_pool: bool = False
    wild_color: Color | None = None


ALL_DICE_INDEX: dict[Dice, int] = {die: index for index, die in enumerate(Dice)}

# --- ID layout (stable v1) ---
# 0: forfeit
# 1-6: pick die (white..silver)
# 10-19: mark yellow cell
# 20-24: white die mode after pick
# 40-45: passive pick platter die
# 46: skip passive pick (take no platter/pool die)
# 50-55: passive pick pool die
# 100-123: mark silver cell (row × value)
# 124: skip optional silver cascade mark
# 130: roll hand (active, before pick)
# 131: use reroll action
# 132-137: unlock — pull platter die into hand
# 138: end plus-one chain
# 139: end active turn (empty hand with unlock available)
# 140-145: plus-one pick die
# 146-150: choose wild color for free-color bonus (round 4)
# 151-160: passive mark yellow cell
# 161-170: plus-one mark yellow cell
# 171-180: bonus yellow circle cell
# 181-190: bonus yellow cross cell

FORFEIT_PICK_ID = 0
PICK_DIE_BASE = 1
MARK_YELLOW_BASE = 10
MARK_WHITE_BLUE_ID = 20
MARK_WHITE_GREEN_ID = 21
MARK_WHITE_SILVER_ID = 22
MARK_WHITE_YELLOW_ID = 23
MARK_WHITE_PINK_ID = 24
MARK_SILVER_BASE = 100
SILVER_SKIP_CASCADE_ID = 124
PASSIVE_PLATTER_BASE = 40
PASSIVE_SKIP_ID = 46
PASSIVE_POOL_BASE = 50
ROLL_HAND_ID = 130
USE_REROLL_ID = 131
UNLOCK_PLATTER_BASE = 132
END_PLUS_ONE_ID = 138
END_ACTIVE_TURN_ID = 139
PLUS_ONE_PICK_BASE = 140
CHOOSE_WILD_COLOR_BASE = 146
PASSIVE_MARK_YELLOW_BASE = 151
PLUS_ONE_MARK_YELLOW_BASE = 161
BONUS_YELLOW_CIRCLE_BASE = 171
BONUS_YELLOW_CROSS_BASE = 181

CATALOG_VERSION = 1
ACTION_SPACE_SIZE = 192


def mark_white_blue_id() -> int:
    return MARK_WHITE_BLUE_ID


def mark_white_green_id() -> int:
    return MARK_WHITE_GREEN_ID


def mark_white_silver_id() -> int:
    return MARK_WHITE_SILVER_ID


def mark_white_yellow_id() -> int:
    return MARK_WHITE_YELLOW_ID


def mark_white_pink_id() -> int:
    return MARK_WHITE_PINK_ID


def pick_die_id(die: Dice) -> int:
    return PICK_DIE_BASE + ALL_DICE_INDEX[die]


def mark_yellow_id(cell_id: int) -> int:
    if not 0 <= cell_id < 10:
        raise ValueError(f"invalid yellow cell id {cell_id}")
    return MARK_YELLOW_BASE + cell_id


def passive_mark_yellow_id(cell_id: int) -> int:
    if not 0 <= cell_id < 10:
        raise ValueError(f"invalid yellow cell id {cell_id}")
    return PASSIVE_MARK_YELLOW_BASE + cell_id


def plus_one_mark_yellow_id(cell_id: int) -> int:
    if not 0 <= cell_id < 10:
        raise ValueError(f"invalid yellow cell id {cell_id}")
    return PLUS_ONE_MARK_YELLOW_BASE + cell_id


def mark_silver_id(row_index: int, value: int) -> int:
    if not 0 <= row_index < 4 or not 1 <= value <= 6:
        raise ValueError(f"invalid silver cell row={row_index} value={value}")
    return MARK_SILVER_BASE + row_index * 6 + (value - 1)


def passive_platter_id(die: Dice) -> int:
    return PASSIVE_PLATTER_BASE + ALL_DICE_INDEX[die]


def passive_pool_id(die: Dice) -> int:
    return PASSIVE_POOL_BASE + ALL_DICE_INDEX[die]


def passive_skip_id() -> int:
    return PASSIVE_SKIP_ID


def roll_hand_id() -> int:
    return ROLL_HAND_ID


def use_reroll_id() -> int:
    return USE_REROLL_ID


def unlock_platter_id(die: Dice) -> int:
    return UNLOCK_PLATTER_BASE + ALL_DICE_INDEX[die]


def end_plus_one_id() -> int:
    return END_PLUS_ONE_ID


def end_active_turn_id() -> int:
    return END_ACTIVE_TURN_ID


def plus_one_pick_id(die: Dice) -> int:
    return PLUS_ONE_PICK_BASE + ALL_DICE_INDEX[die]


def choose_wild_color_id(color: Color) -> int:
    index = SCORING_COLORS.index(color)
    return CHOOSE_WILD_COLOR_BASE + index


def bonus_yellow_circle_id(cell_id: int) -> int:
    if not 0 <= cell_id < 10:
        raise ValueError(f"invalid yellow cell id {cell_id}")
    return BONUS_YELLOW_CIRCLE_BASE + cell_id


def bonus_yellow_cross_id(cell_id: int) -> int:
    if not 0 <= cell_id < 10:
        raise ValueError(f"invalid yellow cell id {cell_id}")
    return BONUS_YELLOW_CROSS_BASE + cell_id


def decode_action(action_id: int) -> Action:
    if action_id == FORFEIT_PICK_ID:
        return Action(kind=ActionKind.FORFEIT_PICK)
    if PICK_DIE_BASE <= action_id < PICK_DIE_BASE + len(Dice):
        return Action(kind=ActionKind.PICK_DIE, die=list(Dice)[action_id - PICK_DIE_BASE])
    if MARK_YELLOW_BASE <= action_id < MARK_YELLOW_BASE + 10:
        return Action(kind=ActionKind.MARK_YELLOW, yellow_cell_id=action_id - MARK_YELLOW_BASE)
    if action_id == MARK_WHITE_BLUE_ID:
        return Action(kind=ActionKind.MARK_WHITE_BLUE)
    if action_id == MARK_WHITE_GREEN_ID:
        return Action(kind=ActionKind.MARK_WHITE_GREEN)
    if action_id == MARK_WHITE_SILVER_ID:
        return Action(kind=ActionKind.MARK_WHITE_SILVER)
    if action_id == MARK_WHITE_YELLOW_ID:
        return Action(kind=ActionKind.MARK_WHITE_YELLOW)
    if action_id == MARK_WHITE_PINK_ID:
        return Action(kind=ActionKind.MARK_WHITE_PINK)
    if MARK_SILVER_BASE <= action_id < MARK_SILVER_BASE + 24:
        offset = action_id - MARK_SILVER_BASE
        return Action(
            kind=ActionKind.MARK_SILVER,
            silver_row_index=offset // 6,
            silver_value=offset % 6 + 1,
        )
    if action_id == SILVER_SKIP_CASCADE_ID:
        return Action(kind=ActionKind.SILVER_SKIP_CASCADE)
    if PASSIVE_PLATTER_BASE <= action_id < PASSIVE_PLATTER_BASE + len(Dice):
        die = list(Dice)[action_id - PASSIVE_PLATTER_BASE]
        return Action(kind=ActionKind.PASSIVE_PICK, die=die, passive_from_pool=False)
    if PASSIVE_POOL_BASE <= action_id < PASSIVE_POOL_BASE + len(Dice):
        die = list(Dice)[action_id - PASSIVE_POOL_BASE]
        return Action(kind=ActionKind.PASSIVE_PICK, die=die, passive_from_pool=True)
    if action_id == PASSIVE_SKIP_ID:
        return Action(kind=ActionKind.PASSIVE_SKIP)
    if action_id == ROLL_HAND_ID:
        return Action(kind=ActionKind.ROLL_HAND)
    if action_id == USE_REROLL_ID:
        return Action(kind=ActionKind.USE_REROLL)
    if UNLOCK_PLATTER_BASE <= action_id < UNLOCK_PLATTER_BASE + len(Dice):
        die = list(Dice)[action_id - UNLOCK_PLATTER_BASE]
        return Action(kind=ActionKind.UNLOCK_PLATTER, die=die)
    if action_id == END_PLUS_ONE_ID:
        return Action(kind=ActionKind.END_PLUS_ONE)
    if action_id == END_ACTIVE_TURN_ID:
        return Action(kind=ActionKind.END_ACTIVE_TURN)
    if PLUS_ONE_PICK_BASE <= action_id < PLUS_ONE_PICK_BASE + len(Dice):
        die = list(Dice)[action_id - PLUS_ONE_PICK_BASE]
        return Action(kind=ActionKind.PLUS_ONE_PICK, die=die)
    if CHOOSE_WILD_COLOR_BASE <= action_id < CHOOSE_WILD_COLOR_BASE + len(SCORING_COLORS):
        color = SCORING_COLORS[action_id - CHOOSE_WILD_COLOR_BASE]
        return Action(kind=ActionKind.CHOOSE_WILD_COLOR, wild_color=color)
    if PASSIVE_MARK_YELLOW_BASE <= action_id < PASSIVE_MARK_YELLOW_BASE + 10:
        return Action(
            kind=ActionKind.MARK_YELLOW_PASSIVE,
            yellow_cell_id=action_id - PASSIVE_MARK_YELLOW_BASE,
        )
    if PLUS_ONE_MARK_YELLOW_BASE <= action_id < PLUS_ONE_MARK_YELLOW_BASE + 10:
        return Action(
            kind=ActionKind.MARK_YELLOW_PLUS_ONE,
            yellow_cell_id=action_id - PLUS_ONE_MARK_YELLOW_BASE,
        )
    if BONUS_YELLOW_CIRCLE_BASE <= action_id < BONUS_YELLOW_CIRCLE_BASE + 10:
        return Action(
            kind=ActionKind.BONUS_YELLOW_CIRCLE,
            yellow_cell_id=action_id - BONUS_YELLOW_CIRCLE_BASE,
        )
    if BONUS_YELLOW_CROSS_BASE <= action_id < BONUS_YELLOW_CROSS_BASE + 10:
        return Action(
            kind=ActionKind.BONUS_YELLOW_CROSS,
            yellow_cell_id=action_id - BONUS_YELLOW_CROSS_BASE,
        )
    raise ValueError(f"unknown action id {action_id}")


def encode_action(action: Action) -> int:
    if action.kind is ActionKind.FORFEIT_PICK:
        return FORFEIT_PICK_ID
    if action.kind is ActionKind.PICK_DIE:
        assert action.die is not None
        return pick_die_id(action.die)
    if action.kind is ActionKind.MARK_YELLOW:
        assert action.yellow_cell_id is not None
        return mark_yellow_id(action.yellow_cell_id)
    if action.kind is ActionKind.MARK_YELLOW_PASSIVE:
        assert action.yellow_cell_id is not None
        return passive_mark_yellow_id(action.yellow_cell_id)
    if action.kind is ActionKind.MARK_YELLOW_PLUS_ONE:
        assert action.yellow_cell_id is not None
        return plus_one_mark_yellow_id(action.yellow_cell_id)
    if action.kind is ActionKind.MARK_WHITE_BLUE:
        return MARK_WHITE_BLUE_ID
    if action.kind is ActionKind.MARK_WHITE_GREEN:
        return MARK_WHITE_GREEN_ID
    if action.kind is ActionKind.MARK_WHITE_SILVER:
        return MARK_WHITE_SILVER_ID
    if action.kind is ActionKind.MARK_WHITE_YELLOW:
        return MARK_WHITE_YELLOW_ID
    if action.kind is ActionKind.MARK_WHITE_PINK:
        return MARK_WHITE_PINK_ID
    if action.kind is ActionKind.MARK_SILVER:
        assert action.silver_row_index is not None and action.silver_value is not None
        return mark_silver_id(action.silver_row_index, action.silver_value)
    if action.kind is ActionKind.SILVER_SKIP_CASCADE:
        return SILVER_SKIP_CASCADE_ID
    if action.kind is ActionKind.PASSIVE_PICK:
        assert action.die is not None
        if action.passive_from_pool:
            return passive_pool_id(action.die)
        return passive_platter_id(action.die)
    if action.kind is ActionKind.PASSIVE_SKIP:
        return PASSIVE_SKIP_ID
    if action.kind is ActionKind.ROLL_HAND:
        return ROLL_HAND_ID
    if action.kind is ActionKind.USE_REROLL:
        return USE_REROLL_ID
    if action.kind is ActionKind.UNLOCK_PLATTER:
        assert action.die is not None
        return unlock_platter_id(action.die)
    if action.kind is ActionKind.END_PLUS_ONE:
        return END_PLUS_ONE_ID
    if action.kind is ActionKind.END_ACTIVE_TURN:
        return END_ACTIVE_TURN_ID
    if action.kind is ActionKind.PLUS_ONE_PICK:
        assert action.die is not None
        return plus_one_pick_id(action.die)
    if action.kind is ActionKind.CHOOSE_WILD_COLOR:
        assert action.wild_color is not None
        return choose_wild_color_id(action.wild_color)
    if action.kind is ActionKind.BONUS_YELLOW_CIRCLE:
        assert action.yellow_cell_id is not None
        return bonus_yellow_circle_id(action.yellow_cell_id)
    if action.kind is ActionKind.BONUS_YELLOW_CROSS:
        assert action.yellow_cell_id is not None
        return bonus_yellow_cross_id(action.yellow_cell_id)
    raise ValueError(f"cannot encode {action}")


def describe_action(action_id: int) -> str:
    """Human-readable label for a catalog action ID."""
    action = decode_action(action_id)
    kind = action.kind

    if kind is ActionKind.FORFEIT_PICK:
        return "forfeit remaining picks"
    if kind is ActionKind.PICK_DIE:
        return f"pick {action.die.value} die"
    if kind is ActionKind.MARK_YELLOW:
        return f"mark yellow cell {action.yellow_cell_id}"
    if kind is ActionKind.MARK_YELLOW_PASSIVE:
        return f"passive mark yellow cell {action.yellow_cell_id}"
    if kind is ActionKind.MARK_YELLOW_PLUS_ONE:
        return f"plus-one mark yellow cell {action.yellow_cell_id}"
    if kind is ActionKind.MARK_WHITE_BLUE:
        return "white die as blue sum"
    if kind is ActionKind.MARK_WHITE_GREEN:
        return "white die as green"
    if kind is ActionKind.MARK_WHITE_SILVER:
        return "white die as silver"
    if kind is ActionKind.MARK_WHITE_YELLOW:
        return "white die as yellow"
    if kind is ActionKind.MARK_WHITE_PINK:
        return "white die as pink"
    if kind is ActionKind.MARK_SILVER:
        row = silver_row_color(action.silver_row_index).value
        return f"mark silver {row} row value {action.silver_value}"
    if kind is ActionKind.SILVER_SKIP_CASCADE:
        return "skip optional silver cascade mark"
    if kind is ActionKind.PASSIVE_PICK:
        source = "pool" if action.passive_from_pool else "platter"
        return f"passive pick {action.die.value} from {source}"
    if kind is ActionKind.PASSIVE_SKIP:
        return "skip (don't take a die)"
    if kind is ActionKind.ROLL_HAND:
        return "roll hand"
    if kind is ActionKind.USE_REROLL:
        return "use reroll action"
    if kind is ActionKind.UNLOCK_PLATTER:
        return f"unlock {action.die.value} from platter"
    if kind is ActionKind.END_PLUS_ONE:
        return "end plus-one chain"
    if kind is ActionKind.END_ACTIVE_TURN:
        return "end active turn"
    if kind is ActionKind.PLUS_ONE_PICK:
        return f"extra die: use {action.die.value} at current face"
    if kind is ActionKind.CHOOSE_WILD_COLOR:
        return f"choose wild color {action.wild_color.value}"
    if kind is ActionKind.BONUS_YELLOW_CIRCLE:
        return f"bonus circle yellow cell {action.yellow_cell_id}"
    if kind is ActionKind.BONUS_YELLOW_CROSS:
        return f"bonus cross yellow cell {action.yellow_cell_id}"
    return f"unknown action {action_id}"
