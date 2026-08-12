"""Deterministic bonus mark choices (no player decision)."""

from __future__ import annotations

from dataclasses import dataclass

from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import get_score_sheet


@dataclass(frozen=True)
class AutoMark:
    area: str
    slot: int
    value: int


def automated_blue_bonus(sheet: PlayerSheet) -> AutoMark | None:
    """Blue ? bonus: repeat the last blue value written."""
    slot = sheet.next_blue_slot()
    if slot is None:
        return None
    last = sheet.last_blue_value()
    if last is None:
        value = 12
    else:
        value = last
    if not sheet.can_mark_blue(value):
        return None
    return AutoMark(area="blue", slot=slot, value=value)


def automated_pink_bonus(sheet: PlayerSheet) -> AutoMark | None:
    """Pink ? bonus: always write 6 in the next pink slot."""
    slot = sheet.next_pink_slot()
    if slot is None:
        return None
    return AutoMark(area="pink", slot=slot, value=6)


def automated_green_bonus(sheet: PlayerSheet) -> AutoMark | None:
    """Green ? bonus: 6× multiplier on first slot of pair, 1× on second."""
    slot = sheet.next_green_slot()
    if slot is None:
        return None
    multiplier = get_score_sheet().green.multipliers[slot]
    die_face = 6 if slot % 2 == 0 else 1
    return AutoMark(area="green", slot=slot, value=die_face * multiplier)


def apply_auto_mark(sheet: PlayerSheet, mark: AutoMark) -> None:
    if mark.area == "blue":
        sheet.mark_blue(mark.value)
    elif mark.area == "pink":
        sheet.mark_pink(mark.value)
    elif mark.area == "green":
        sheet.mark_green(mark.slot, mark.value)
    else:
        raise ValueError(f"unsupported auto mark area {mark.area}")
