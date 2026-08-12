"""FIFO bonus queue — enqueue triggers, resolve in order."""

from __future__ import annotations

from dataclasses import dataclass

from doppelt.core.score_sheet import Bonus, get_score_sheet
from doppelt.core.state import GameState
from doppelt.core.types import BonusKind


@dataclass(frozen=True)
class PendingBonus:
    """One bonus waiting in the FIFO queue."""

    bonus: Bonus
    source: str


def _field_bonus(area: str, slot: int) -> Bonus | None:
    sheet = get_score_sheet()
    if area == "blue":
        entries = sheet.blue.field_bonuses
    elif area == "pink":
        entries = sheet.pink.field_bonuses
    elif area == "green":
        entries = sheet.green.field_bonuses
    else:
        return None
    for entry in entries:
        if entry.slot == slot:
            return entry.bonus
    return None


def enqueue_bonus(state: GameState, bonus: Bonus, source: str) -> None:
    state.pending_bonuses.append(PendingBonus(bonus=bonus, source=source))


def enqueue_bonus_once(state: GameState, bonus: Bonus, source: str) -> None:
    if source in state.sheet.claimed_bonuses:
        return
    state.sheet.claimed_bonuses.add(source)
    enqueue_bonus(state, bonus, source)


def enqueue_bonuses_after_blue_mark(state: GameState, slot: int) -> None:
    bonus = _field_bonus("blue", slot)
    if bonus is None:
        return
    enqueue_bonus_once(state, bonus, f"blue:{slot}")


def enqueue_bonuses_after_green_mark(state: GameState, slot: int) -> None:
    bonus = _field_bonus("green", slot)
    if bonus is None:
        return
    enqueue_bonus_once(state, bonus, f"green:{slot}")


def enqueue_bonuses_after_pink_mark(state: GameState, slot: int, die_value: int) -> None:
    threshold = get_score_sheet().pink.min_values[slot]
    if threshold is not None and die_value < threshold:
        return
    bonus = _field_bonus("pink", slot)
    if bonus is None:
        return
    enqueue_bonus_once(state, bonus, f"pink:{slot}")


def enqueue_bonuses_after_silver_mark(state: GameState, value: int) -> None:
    if not state.sheet.silver_column_complete(value):
        return
    bonus = get_score_sheet().silver.column_bonuses[value - 1]
    if bonus is None:
        return
    enqueue_bonus_once(state, bonus, f"silver:col:{value}")


def _yellow_row_index(cell_id: int) -> int:
    return get_score_sheet().yellow.cells[cell_id].row


def _yellow_col_index(cell_id: int) -> int:
    return get_score_sheet().yellow.cells[cell_id].col


def enqueue_bonuses_after_yellow_circle(state: GameState, cell_id: int) -> None:
    """Row/column edge bonuses fire when every cell in the line is circled."""
    row_index = _yellow_row_index(cell_id)
    if state.sheet.yellow_row_circled(row_index):
        bonus = get_score_sheet().yellow.row_completion_bonuses[row_index]
        if bonus is not None:
            enqueue_bonus_once(state, bonus, f"yellow:row:{row_index}")

    col_index = _yellow_col_index(cell_id)
    if state.sheet.yellow_column_circled(col_index):
        bonus = get_score_sheet().yellow.bottom_edge_bonuses[col_index]
        if bonus is not None:
            enqueue_bonus_once(state, bonus, f"yellow:col:{col_index}")


def enqueue_followups_after_auto_mark(state: GameState, area: str, slot: int) -> None:
    if area == "blue":
        enqueue_bonuses_after_blue_mark(state, slot)
    elif area == "green":
        enqueue_bonuses_after_green_mark(state, slot)
    elif area == "pink":
        enqueue_bonuses_after_pink_mark(state, slot, state.sheet.pink[slot])


def bonus_kind_label(bonus: Bonus) -> str:
    if bonus.kind is BonusKind.BONUS_WILD:
        if bonus.color is None:
            return "wild:free"
        return f"wild:{bonus.color.value}"
    return bonus.kind.value
