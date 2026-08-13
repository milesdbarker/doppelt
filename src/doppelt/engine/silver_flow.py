"""Silver die resolution — primary mark and platter cascade."""

from __future__ import annotations

from typing import Literal

from doppelt.actions.catalog_v1 import mark_silver_id
from doppelt.core.phases import Phase
from doppelt.core.silver import silver_row_index
from doppelt.core.state import GameState
from doppelt.core.types import Color
from doppelt.engine.bonus_flow import try_enter_bonus_phase
from doppelt.engine.bonus_queue import enqueue_bonuses_after_silver_mark

SilverFinish = Literal["active", "passive", "plus_one"]
SilverCascadeMark = tuple[int, Color | None]


def _legal_rows_for_pending(state: GameState) -> list[Color]:
    value = state.pending_silver_values[0]
    row_constraint = state.pending_silver_rows[0]
    if row_constraint is None:
        return state.sheet.legal_silver_rows(value)
    if state.sheet.can_mark_silver(value, row_constraint):
        return [row_constraint]
    return []


def start_silver_resolution(
    state: GameState,
    *,
    primary_value: int,
    cascade_marks: list[SilverCascadeMark],
    finish: SilverFinish,
) -> None:
    state.pending_silver_values = [primary_value, *(value for value, _ in cascade_marks)]
    state.pending_silver_rows = [None, *(row for _, row in cascade_marks)]
    state.pending_silver_required = [True, *[False] * len(cascade_marks)]
    state.silver_finish = finish
    state.phase = Phase.ACTIVE_MARK_SILVER
    skip_unmarkable_cascade_heads(state)


def legal_silver_mark_action_ids(state: GameState) -> list[int]:
    if not state.pending_silver_values:
        return []
    value = state.pending_silver_values[0]
    return [
        mark_silver_id(silver_row_index(row), value) for row in _legal_rows_for_pending(state)
    ]


def apply_silver_mark(state: GameState, row: Color) -> None:
    if not state.pending_silver_values:
        raise ValueError("no pending silver mark")
    value = state.pending_silver_values[0]
    row_constraint = state.pending_silver_rows[0]
    if row_constraint is not None and row != row_constraint:
        raise ValueError(f"silver cascade must use {row_constraint.value} row, not {row.value}")
    if row not in _legal_rows_for_pending(state):
        raise ValueError(f"illegal silver mark value {value} row {row.value}")
    state.sheet.mark_silver(value, row)
    enqueue_bonuses_after_silver_mark(state, value)
    state.pending_silver_values.pop(0)
    state.pending_silver_rows.pop(0)
    state.pending_silver_required.pop(0)
    skip_unmarkable_cascade_heads(state)
    try_enter_bonus_phase(state, Phase.ACTIVE_MARK_SILVER)


def silver_resolution_complete(state: GameState) -> bool:
    return state.phase is Phase.ACTIVE_MARK_SILVER and not state.pending_silver_values


def skip_unmarkable_cascade_heads(state: GameState) -> None:
    """Drop optional cascade heads that no longer have a legal row."""
    while state.pending_silver_values and not _legal_rows_for_pending(state):
        if state.pending_silver_required[0]:
            raise ValueError("required silver mark has no legal row")
        state.pending_silver_values.pop(0)
        state.pending_silver_rows.pop(0)
        state.pending_silver_required.pop(0)
