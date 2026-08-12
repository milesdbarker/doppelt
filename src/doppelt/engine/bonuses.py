"""Resolve bonus chains after sheet marks."""

from __future__ import annotations

from dataclasses import dataclass

from doppelt.core.bonus_auto import (
  AutoMark,
  apply_auto_mark,
  automated_blue_bonus,
  automated_green_bonus,
  automated_pink_bonus,
)
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import Bonus, get_score_sheet
from doppelt.core.types import BonusKind, Color


@dataclass(frozen=True)
class BonusEvent:
  """Record of an automated bonus resolution."""

  source: str
  mark: AutoMark


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


def _automated_wild_mark(sheet: PlayerSheet, color: Color | None) -> AutoMark | None:
  if color is Color.BLUE:
    return automated_blue_bonus(sheet)
  if color is Color.GREEN:
    return automated_green_bonus(sheet)
  if color is Color.PINK:
    return automated_pink_bonus(sheet)
  return None


def resolve_field_bonus(sheet: PlayerSheet, bonus: Bonus, source: str) -> list[BonusEvent]:
  events: list[BonusEvent] = []
  pending: list[tuple[Bonus, str]] = [(bonus, source)]

  while pending:
    current, src = pending.pop(0)
    if current.kind is BonusKind.FOX:
      sheet.foxes += 1
      continue
    if current.kind is not BonusKind.BONUS_WILD:
      continue

    mark = _automated_wild_mark(sheet, current.color)
    if mark is None:
      continue
    apply_auto_mark(sheet, mark)
    events.append(BonusEvent(source=src, mark=mark))

    follow_up = _field_bonus(mark.area, mark.slot)
    if follow_up is not None:
      pending.append((follow_up, f"chain:{mark.area}:{mark.slot}"))

  return events


def bonuses_after_blue_mark(sheet: PlayerSheet, slot: int) -> list[BonusEvent]:
  bonus = _field_bonus("blue", slot)
  if bonus is None:
    return []
  return resolve_field_bonus(sheet, bonus, f"blue:{slot}")


def bonuses_after_pink_mark(sheet: PlayerSheet, slot: int, die_value: int) -> list[BonusEvent]:
  threshold = get_score_sheet().pink.min_values[slot]
  if threshold is not None and die_value < threshold:
    return []
  bonus = _field_bonus("pink", slot)
  if bonus is None:
    return []
  return resolve_field_bonus(sheet, bonus, f"pink:{slot}")


def bonuses_after_green_mark(sheet: PlayerSheet, slot: int) -> list[BonusEvent]:
  bonus = _field_bonus("green", slot)
  if bonus is None:
    return []
  return resolve_field_bonus(sheet, bonus, f"green:{slot}")
