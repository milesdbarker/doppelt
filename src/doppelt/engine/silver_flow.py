"""Silver die resolution — primary mark and platter cascade."""

from __future__ import annotations

from typing import Literal

from doppelt.actions.catalog_v1 import (
    SILVER_SKIP_CASCADE_ID,
    mark_silver_id,
)
from doppelt.core.phases import Phase
from doppelt.core.silver import silver_row_index
from doppelt.core.state import GameState
from doppelt.core.types import Color
from doppelt.engine.bonus_flow import try_enter_bonus_phase
from doppelt.engine.bonus_queue import enqueue_bonuses_after_silver_mark

SilverFinish = Literal["active", "passive"]


def start_silver_resolution(
    state: GameState,
    *,
    primary_value: int,
    cascade_values: list[int],
    finish: SilverFinish,
) -> None:
    state.pending_silver_values = [primary_value, *cascade_values]
    state.pending_silver_required = [True, *[False] * len(cascade_values)]
    state.silver_finish = finish
    state.phase = Phase.ACTIVE_MARK_SILVER
    _skip_empty_cascade_heads(state)


def legal_silver_mark_action_ids(state: GameState) -> list[int]:
    if not state.pending_silver_values:
        return []
    value = state.pending_silver_values[0]
    required = state.pending_silver_required[0]
    actions = [
        mark_silver_id(silver_row_index(row), value) for row in state.sheet.legal_silver_rows(value)
    ]
    if not required:
        actions.append(SILVER_SKIP_CASCADE_ID)
    return actions


def apply_silver_mark(state: GameState, row: Color) -> None:
    if not state.pending_silver_values:
        raise ValueError("no pending silver mark")
    value = state.pending_silver_values[0]
    state.sheet.mark_silver(value, row)
    enqueue_bonuses_after_silver_mark(state, value)
    state.pending_silver_values.pop(0)
    state.pending_silver_required.pop(0)
    _skip_empty_cascade_heads(state)
    try_enter_bonus_phase(state, Phase.ACTIVE_MARK_SILVER)


def apply_silver_skip(state: GameState) -> None:
    if not state.pending_silver_values:
        raise ValueError("no pending silver mark")
    if state.pending_silver_required[0]:
        raise ValueError("primary silver mark cannot be skipped")
    state.pending_silver_values.pop(0)
    state.pending_silver_required.pop(0)
    _skip_empty_cascade_heads(state)


def silver_resolution_complete(state: GameState) -> bool:
    return state.phase is Phase.ACTIVE_MARK_SILVER and not state.pending_silver_values


def _skip_empty_cascade_heads(state: GameState) -> None:
    while state.pending_silver_values and not state.sheet.legal_silver_rows(
        state.pending_silver_values[0]
    ):
        if state.pending_silver_required[0]:
            raise ValueError("required silver mark has no legal row")
        state.pending_silver_values.pop(0)
        state.pending_silver_required.pop(0)
