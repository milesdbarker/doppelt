"""Bonus resolution phase — FIFO queue, auto-resolve where policy allows."""

from __future__ import annotations

from doppelt.actions.catalog_v1 import mark_silver_id, mark_yellow_id
from doppelt.core.bonus_auto import (
    apply_auto_mark,
    automated_blue_bonus,
    automated_green_bonus,
    automated_pink_bonus,
)
from doppelt.core.phases import Phase
from doppelt.core.silver import silver_row_index
from doppelt.core.state import GameState
from doppelt.core.types import BonusKind, Color
from doppelt.engine.bonus_queue import (
    PendingBonus,
    bonus_kind_label,
    enqueue_bonuses_after_silver_mark,
    enqueue_bonuses_after_yellow_cross,
    enqueue_followups_after_auto_mark,
)


def clear_bonus_queue(state: GameState) -> None:
    state.pending_bonuses = []
    state.resume_phase = None
    state.bonus_resume_after = None


def try_enter_bonus_phase(
    state: GameState,
    resume_phase: Phase,
    *,
    resume_after: str | None = None,
) -> bool:
    """Drain auto-resolvable bonuses; enter RESOLVE_BONUS if queue remains."""
    drain_auto_bonus_queue(state)
    if not state.pending_bonuses:
        return False
    state.resume_phase = resume_phase
    state.bonus_resume_after = resume_after
    state.phase = Phase.RESOLVE_BONUS
    return True


def finish_bonus_phase_if_empty(state: GameState) -> bool:
    """Return True when bonus queue is empty and normal phase restored."""
    drain_auto_bonus_queue(state)
    if state.pending_bonuses:
        return False
    if state.resume_phase is not None:
        state.phase = state.resume_phase
        state.resume_phase = None
    return True


def current_pending_bonus(state: GameState) -> PendingBonus | None:
    if not state.pending_bonuses:
        return None
    return state.pending_bonuses[0]


def legal_bonus_action_ids(state: GameState) -> list[int]:
    drain_auto_bonus_queue(state)
    if not state.pending_bonuses:
        return []
    pending = state.pending_bonuses[0]
    bonus = pending.bonus
    if bonus.kind is not BonusKind.BONUS_WILD:
        return []
    if bonus.color is Color.YELLOW:
        return [
            mark_yellow_id(cell_id)
            for cell_id in range(len(state.sheet.yellow))
            if state.sheet.can_bonus_cross_yellow(cell_id)
        ]
    if bonus.color is Color.SILVER:
        actions: list[int] = []
        for value in range(1, 7):
            for row in state.sheet.legal_silver_rows(value):
                actions.append(mark_silver_id(silver_row_index(row), value))
        return actions
    return []


def apply_bonus_yellow_cross(state: GameState, cell_id: int) -> None:
    pending = _pop_head_bonus(state)
    state.sheet.bonus_cross_yellow(cell_id)
    state.bonus_events.append(f"{pending.source}:yellow_cross:{cell_id}")
    enqueue_bonuses_after_yellow_cross(state, cell_id)
    finish_bonus_phase_if_empty(state)


def apply_bonus_silver_mark(state: GameState, row: Color, value: int) -> None:
    pending = _pop_head_bonus(state)
    state.sheet.mark_silver(value, row)
    state.bonus_events.append(f"{pending.source}:silver:{row.value}:{value}")
    enqueue_bonuses_after_silver_mark(state, value)
    finish_bonus_phase_if_empty(state)


def drain_auto_bonus_queue(state: GameState) -> None:
    while state.pending_bonuses and _resolve_auto_bonus_head(state):
        pass


def _pop_head_bonus(state: GameState) -> PendingBonus:
    if not state.pending_bonuses:
        raise ValueError("bonus queue is empty")
    return state.pending_bonuses.pop(0)


def _resolve_auto_bonus_head(state: GameState) -> bool:
    if not state.pending_bonuses:
        return False
    pending = state.pending_bonuses[0]
    bonus = pending.bonus

    if bonus.kind is BonusKind.FOX:
        state.sheet.foxes += 1
        state.bonus_events.append(f"{pending.source}:fox")
        state.pending_bonuses.pop(0)
        return True

    if bonus.kind in (BonusKind.REROLL, BonusKind.RETURN_DIE, BonusKind.EXTRA_DIE):
        state.bonus_events.append(f"{pending.source}:{bonus.kind.value}")
        state.pending_bonuses.pop(0)
        return True

    if bonus.kind is BonusKind.BLUE_WHITE_SUM_HINT:
        state.bonus_events.append(f"{pending.source}:hint")
        state.pending_bonuses.pop(0)
        return True

    if bonus.kind is not BonusKind.BONUS_WILD:
        state.pending_bonuses.pop(0)
        return True

    if bonus.color in (Color.BLUE, Color.GREEN, Color.PINK):
        mark = _automated_wild_mark(state, bonus.color)
        state.pending_bonuses.pop(0)
        if mark is None:
            state.bonus_events.append(f"{pending.source}:{bonus_kind_label(bonus)}:skipped")
            return True
        apply_auto_mark(state.sheet, mark)
        state.bonus_events.append(
            f"{pending.source}:{bonus_kind_label(bonus)}:{mark.area}{mark.slot}={mark.value}"
        )
        enqueue_followups_after_auto_mark(state, mark.area, mark.slot)
        return True

    return False


def _automated_wild_mark(state: GameState, color: Color):
    sheet = state.sheet
    if color is Color.BLUE:
        return automated_blue_bonus(sheet)
    if color is Color.GREEN:
        return automated_green_bonus(sheet)
    if color is Color.PINK:
        return automated_pink_bonus(sheet)
    return None
