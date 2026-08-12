"""Action catalog v1 — global uint16 action IDs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from doppelt.core.types import Dice


class ActionKind(str, Enum):
  PICK_DIE = "pick_die"
  MARK_YELLOW = "mark_yellow"
  FORFEIT_PICK = "forfeit_pick"
  PASSIVE_PICK = "passive_pick"


@dataclass(frozen=True)
class Action:
  kind: ActionKind
  die: Dice | None = None
  yellow_cell_id: int | None = None
  passive_from_pool: bool = False


ALL_DICE_INDEX: dict[Dice, int] = {die: index for index, die in enumerate(Dice)}

# --- ID layout (stable v1) ---
# 0: forfeit
# 1-6: pick die (white..silver)
# 10-19: mark yellow cell
# 40-45: passive pick platter die
# 50-55: passive pick pool die

FORFEIT_PICK_ID = 0
PICK_DIE_BASE = 1
MARK_YELLOW_BASE = 10
PASSIVE_PLATTER_BASE = 40
PASSIVE_POOL_BASE = 50

CATALOG_VERSION = 1
ACTION_SPACE_SIZE = 56


def pick_die_id(die: Dice) -> int:
  return PICK_DIE_BASE + ALL_DICE_INDEX[die]


def mark_yellow_id(cell_id: int) -> int:
  return MARK_YELLOW_BASE + cell_id


def passive_platter_id(die: Dice) -> int:
  return PASSIVE_PLATTER_BASE + ALL_DICE_INDEX[die]


def passive_pool_id(die: Dice) -> int:
  return PASSIVE_POOL_BASE + ALL_DICE_INDEX[die]


def decode_action(action_id: int) -> Action:
  if action_id == FORFEIT_PICK_ID:
    return Action(kind=ActionKind.FORFEIT_PICK)
  if PICK_DIE_BASE <= action_id < PICK_DIE_BASE + len(Dice):
    return Action(kind=ActionKind.PICK_DIE, die=list(Dice)[action_id - PICK_DIE_BASE])
  if MARK_YELLOW_BASE <= action_id < MARK_YELLOW_BASE + 10:
    return Action(kind=ActionKind.MARK_YELLOW, yellow_cell_id=action_id - MARK_YELLOW_BASE)
  if PASSIVE_PLATTER_BASE <= action_id < PASSIVE_PLATTER_BASE + len(Dice):
    die = list(Dice)[action_id - PASSIVE_PLATTER_BASE]
    return Action(kind=ActionKind.PASSIVE_PICK, die=die, passive_from_pool=False)
  if PASSIVE_POOL_BASE <= action_id < PASSIVE_POOL_BASE + len(Dice):
    die = list(Dice)[action_id - PASSIVE_POOL_BASE]
    return Action(kind=ActionKind.PASSIVE_PICK, die=die, passive_from_pool=True)
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
  if action.kind is ActionKind.PASSIVE_PICK:
    assert action.die is not None
    if action.passive_from_pool:
      return passive_pool_id(action.die)
    return passive_platter_id(action.die)
  raise ValueError(f"cannot encode {action}")
