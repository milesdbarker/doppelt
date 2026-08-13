"""Game state."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.types import Color, Dice

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
    pending_silver_rows: list[Color | None] = field(default_factory=list)
    pending_silver_required: list[bool] = field(default_factory=list)
    silver_finish: str = "active"  # "active" | "passive"
    use_pool_fallback: bool = False
    pending_bonuses: list[PendingBonus] = field(default_factory=list)
    resume_phase: Phase | None = None
    bonus_resume_after: str | None = None
    awaiting_roll: bool = False
    white_mark_resume_after: str = "after_active_pick"
    plus_one_dice_used: set[Dice] = field(default_factory=set)
    plus_one_after_passive: bool = False
    action_log: list[int] = field(default_factory=list)
    bonus_events: list[str] = field(default_factory=list)

    def copy_for_trial(self) -> GameState:
        """Clone mutable state for one-step search without sharing RNG or logs."""
        trial = GameState(
            seed=self.seed,
            player_count=self.player_count,
            round_index=self.round_index,
            phase=self.phase,
            sheet=self.sheet.copy(),
            rng=random.Random(),
            faces=dict(self.faces),
            hand=list(self.hand),
            platter=list(self.platter),
            passive_pool=list(self.passive_pool),
            slots=list(self.slots),
            picks_made=self.picks_made,
            pending_die=self.pending_die,
            pending_value=self.pending_value,
            pending_platter_sent=list(self.pending_platter_sent),
            pending_silver_values=list(self.pending_silver_values),
            pending_silver_rows=list(self.pending_silver_rows),
            pending_silver_required=list(self.pending_silver_required),
            silver_finish=self.silver_finish,
            use_pool_fallback=self.use_pool_fallback,
            pending_bonuses=list(self.pending_bonuses),
            resume_phase=self.resume_phase,
            bonus_resume_after=self.bonus_resume_after,
            awaiting_roll=self.awaiting_roll,
            white_mark_resume_after=self.white_mark_resume_after,
            plus_one_dice_used=set(self.plus_one_dice_used),
            plus_one_after_passive=self.plus_one_after_passive,
        )
        trial.rng.setstate(self.rng.getstate())
        return trial
