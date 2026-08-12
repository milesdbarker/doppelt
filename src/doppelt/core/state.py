"""Game state placeholder — rules applied in Phase 1."""

from dataclasses import dataclass


@dataclass
class GameState:
  """Minimal shell; fields expand in Phase 1."""

  seed: int
  player_count: int = 1
  round_index: int = 0
  # Per-player sheet state will live here (marks, dice, pending bonuses, etc.).
