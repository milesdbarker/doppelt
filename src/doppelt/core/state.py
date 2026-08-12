"""Game state."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.types import Dice


@dataclass
class GameState:
  seed: int
  player_count: int
  round_index: int
  phase: Phase
  sheet: PlayerSheet
  rng: random.Random = field(repr=False)
  faces: dict[Dice, int] = field(default_factory=dict)
  hand: list[Dice] = field(default_factory=list)
  platter: list[Dice] = field(default_factory=list)
  passive_pool: list[Dice] = field(default_factory=list)
  slots: list[Dice | None] = field(default_factory=lambda: [None, None, None])
  picks_made: int = 0
  pending_die: Dice | None = None
  pending_value: int | None = None
  use_pool_fallback: bool = False
  action_log: list[int] = field(default_factory=list)
  bonus_events: list[str] = field(default_factory=list)
