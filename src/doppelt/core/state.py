"""Game state."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.types import Dice

if TYPE_CHECKING:
    from doppelt.engine.bonus_queue import PendingBonus


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
    pending_platter_sent: list[Dice] = field(default_factory=list)
    pending_silver_values: list[int] = field(default_factory=list)
    pending_silver_required: list[bool] = field(default_factory=list)
    silver_finish: str = "active"  # "active" | "passive"
    use_pool_fallback: bool = False
    pending_bonuses: list[PendingBonus] = field(default_factory=list)
    resume_phase: Phase | None = None
    bonus_resume_after: str | None = None
    action_log: list[int] = field(default_factory=list)
    bonus_events: list[str] = field(default_factory=list)
