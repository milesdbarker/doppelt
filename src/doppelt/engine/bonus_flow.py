"""Bonus resolution phase — FIFO queue, auto-resolve where policy allows."""

from __future__ import annotations

from doppelt.actions.catalog_v1 import (
    bonus_yellow_circle_id,
    bonus_yellow_cross_id,
    choose_wild_color_id,
    mark_silver_id,
)
from doppelt.core.bonus_auto import (
    apply_auto_mark,
    automated_wild_bonus,
    can_automated_wild_bonus,
)
from doppelt.core.phases import Phase
from doppelt.core.score_sheet import Bonus
from doppelt.core.silver import silver_row_index
from doppelt.core.state import GameState
from doppelt.core.types import SCORING_COLORS, BonusKind, Color
from doppelt.engine.action_flow import apply_action_bonus_circle
from doppelt.engine.bonus_queue import (
    PendingBonus,
    bonus_kind_label,
    enqueue_bonuses_after_silver_mark,
    enqueue_bonuses_after_yellow_circle,
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
    if bonus.color is None:
        return [choose_wild_color_id(color) for color in SCORING_COLORS]
    if bonus.color is Color.YELLOW:
        actions: list[int] = []
        for cell_id in range(len(state.sheet.yellow)):
            if state.sheet.can_bonus_circle_yellow(cell_id):
                actions.append(bonus_yellow_circle_id(cell_id))
            if state.sheet.can_bonus_cross_yellow(cell_id):
                actions.append(bonus_yellow_cross_id(cell_id))
        return actions
    if bonus.color is Color.SILVER:
        actions: list[int] = []
        for value in range(1, 7):
            for row in state.sheet.legal_silver_rows(value):
                actions.append(mark_silver_id(silver_row_index(row), value))
        return actions
    return []


def apply_bonus_choose_wild_color(state: GameState, color: Color) -> None:
    pending = _pop_head_bonus(state)
    bonus = pending.bonus
    if bonus.kind is not BonusKind.BONUS_WILD or bonus.color is not None:
        raise ValueError("expected free-color wild bonus at queue head")
    state.bonus_events.append(f"{pending.source}:choose:{color.value}")
    state.pending_bonuses.insert(0, PendingBonus(Bonus(BonusKind.BONUS_WILD, color), pending.source))
    finish_bonus_phase_if_empty(state)


def apply_bonus_yellow_circle(state: GameState, cell_id: int) -> None:
    pending = _pop_head_bonus(state)
    state.sheet.bonus_circle_yellow(cell_id)
    state.bonus_events.append(f"{pending.source}:yellow_circle:{cell_id}")
    enqueue_bonuses_after_yellow_circle(state, cell_id)
    finish_bonus_phase_if_empty(state)


def apply_bonus_yellow_cross(state: GameState, cell_id: int) -> None:
    pending = _pop_head_bonus(state)
    state.sheet.bonus_cross_yellow(cell_id)
    state.bonus_events.append(f"{pending.source}:yellow_cross:{cell_id}")
    finish_bonus_phase_if_empty(state)


def apply_bonus_silver_mark(state: GameState, row: Color, value: int) -> None:
    pending = _pop_head_bonus(state)
    state.sheet.mark_silver(value, row)
    state.bonus_events.append(f"{pending.source}:silver:{row.value}:{value}")
    enqueue_bonuses_after_silver_mark(state, value)
    finish_bonus_phase_if_empty(state)


def drain_auto_bonus_queue(state: GameState) -> None:
    while state.pending_bonuses:
        if _resolve_auto_bonus_head(state):
            continue
        if _skip_impossible_player_bonus(state):
            continue
        break


def _skip_impossible_player_bonus(state: GameState) -> bool:
    pending = state.pending_bonuses[0]
    bonus = pending.bonus
    if bonus.kind is not BonusKind.BONUS_WILD:
        return False
    if bonus.color is None:
        return False
    if bonus.color in (Color.BLUE, Color.GREEN, Color.PINK):
        impossible = not can_automated_wild_bonus(state.sheet, bonus.color)
    elif bonus.color is Color.YELLOW:
        impossible = not any(
            state.sheet.can_bonus_circle_yellow(cell_id)
            or state.sheet.can_bonus_cross_yellow(cell_id)
            for cell_id in range(len(state.sheet.yellow))
        )
    elif bonus.color is Color.SILVER:
        impossible = not any(
            state.sheet.can_mark_silver(value, row)
            for value in range(1, 7)
            for row in state.sheet.legal_silver_rows(value)
        )
    else:
        return False
    if not impossible:
        return False
    state.bonus_events.append(f"{pending.source}:{bonus_kind_label(bonus)}:skipped")
    state.pending_bonuses.pop(0)
    return True


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

    if bonus.kind in (BonusKind.REROLL, BonusKind.UNLOCK, BonusKind.PLUS_ONE):
        apply_action_bonus_circle(state, bonus.kind)
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

    if bonus.color is None:
        return False

    if bonus.color in (Color.BLUE, Color.GREEN, Color.PINK):
        mark = automated_wild_bonus(state.sheet, bonus.color)
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
