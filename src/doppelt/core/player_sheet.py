"""Mutable player score sheet state."""

from __future__ import annotations

from dataclasses import dataclass, field

from doppelt.core.score_sheet import get_score_sheet


@dataclass
class YellowCellState:
  circled: bool = False
  crossed: bool = False


@dataclass
class PlayerSheet:
  yellow: list[YellowCellState] = field(default_factory=list)
  blue: list[int | None] = field(default_factory=list)
  pink: list[int | None] = field(default_factory=list)
  green: list[int | None] = field(default_factory=list)
  green_stars: list[int | None] = field(default_factory=list)
  foxes: int = 0

  @classmethod
  def empty(cls) -> PlayerSheet:
    sheet = get_score_sheet()
    return cls(
      yellow=[YellowCellState() for _ in sheet.yellow.cells],
      blue=[None] * sheet.blue.slot_count,
      pink=[None] * sheet.pink.slot_count,
      green=[None] * sheet.green.slot_count,
      green_stars=[None] * sheet.green.pair_count,
    )

  def yellow_cell_ids_for_value(self, value: int) -> list[int]:
    sheet = get_score_sheet()
    return [cell.id for cell in sheet.yellow.cells if cell.value == value]

  def can_mark_yellow(self, cell_id: int, die_value: int) -> bool:
    sheet = get_score_sheet()
    cell_def = sheet.yellow.cells[cell_id]
    if cell_def.value != die_value:
      return False
    state = self.yellow[cell_id]
    return not state.crossed

  def mark_yellow(self, cell_id: int, die_value: int) -> str:
    """Circle first visit, cross on second. Returns 'circle' or 'cross'."""
    if not self.can_mark_yellow(cell_id, die_value):
      raise ValueError(f"illegal yellow mark cell {cell_id} value {die_value}")
    state = self.yellow[cell_id]
    if not state.circled:
      state.circled = True
      return "circle"
    state.crossed = True
    return "cross"

  def yellow_cross_count(self) -> int:
    return sum(1 for cell in self.yellow if cell.crossed)

  def next_blue_slot(self) -> int | None:
    for index, value in enumerate(self.blue):
      if value is None:
        return index
    return None

  def last_blue_value(self) -> int | None:
    for value in reversed(self.blue):
      if value is not None:
        return value
    return None

  def can_mark_blue(self, entry_value: int) -> bool:
    slot = self.next_blue_slot()
    if slot is None:
      return False
    if slot == 0:
      return 2 <= entry_value <= 12
    last = self.last_blue_value()
    assert last is not None
    return entry_value <= last

  def mark_blue(self, entry_value: int) -> int:
    if not self.can_mark_blue(entry_value):
      raise ValueError(f"illegal blue entry {entry_value}")
    slot = self.next_blue_slot()
    assert slot is not None
    self.blue[slot] = entry_value
    return slot

  def next_pink_slot(self) -> int | None:
    for index, value in enumerate(self.pink):
      if value is None:
        return index
    return None

  def can_mark_pink(self, die_value: int) -> bool:
    return self.next_pink_slot() is not None and 1 <= die_value <= 6

  def mark_pink(self, die_value: int) -> int:
    if not self.can_mark_pink(die_value):
      raise ValueError(f"illegal pink entry {die_value}")
    slot = self.next_pink_slot()
    assert slot is not None
    self.pink[slot] = die_value
    return slot

  def next_green_slot(self) -> int | None:
    for index, value in enumerate(self.green):
      if value is None:
        return index
    return None

  def mark_green(self, slot: int, entry_value: int) -> None:
    if slot < 0 or slot >= len(self.green) or self.green[slot] is not None:
      raise ValueError(f"illegal green slot {slot}")
    self.green[slot] = entry_value
    self._refresh_green_star(slot // 2)

  def _refresh_green_star(self, pair_index: int) -> None:
    left = pair_index * 2
    right = left + 1
    if self.green[left] is None or self.green[right] is None:
      return
    self.green_stars[pair_index] = self.green[left] - self.green[right]
