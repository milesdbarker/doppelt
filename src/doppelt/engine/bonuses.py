"""Resolve bonus chains after sheet marks (queue-based)."""

from __future__ import annotations

from dataclasses import dataclass

from doppelt.core.bonus_auto import AutoMark
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.state import GameState
from doppelt.engine.bonus_queue import (
    enqueue_bonuses_after_blue_mark,
    enqueue_bonuses_after_green_mark,
    enqueue_bonuses_after_pink_mark,
    enqueue_bonuses_after_silver_mark,
    enqueue_bonuses_after_yellow_circle,
)


@dataclass(frozen=True)
class BonusEvent:
    """Record of an automated bonus resolution."""

    source: str
    mark: AutoMark


def enqueue_mark_bonuses(
    state: GameState,
    *,
    blue_slot: int | None = None,
    green_slot: int | None = None,
    pink_slot: int | None = None,
    pink_value: int | None = None,
    silver_value: int | None = None,
    yellow_cell_id: int | None = None,
    yellow_mark_result: str | None = None,
) -> None:
    if blue_slot is not None:
        enqueue_bonuses_after_blue_mark(state, blue_slot)
    if green_slot is not None:
        enqueue_bonuses_after_green_mark(state, green_slot)
    if pink_slot is not None and pink_value is not None:
        enqueue_bonuses_after_pink_mark(state, pink_slot, pink_value)
    if silver_value is not None:
        enqueue_bonuses_after_silver_mark(state, silver_value)
    if yellow_cell_id is not None and yellow_mark_result == "circle":
        enqueue_bonuses_after_yellow_circle(state, yellow_cell_id)


# Backward-compatible helpers for tests that call sheet-only APIs.
def bonuses_after_blue_mark(sheet: PlayerSheet, slot: int) -> list[BonusEvent]:
    del sheet, slot
    return []


def bonuses_after_pink_mark(sheet: PlayerSheet, slot: int, die_value: int) -> list[BonusEvent]:
    del sheet, slot, die_value
    return []


def bonuses_after_green_mark(sheet: PlayerSheet, slot: int) -> list[BonusEvent]:
    del sheet, slot
    return []


def bonuses_after_silver_mark(sheet: PlayerSheet, value: int) -> list[BonusEvent]:
    del sheet, value
    return []
