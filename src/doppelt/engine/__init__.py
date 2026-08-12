"""Engine exports."""

from doppelt.engine.game import (
  apply_action,
  is_terminal,
  legal_action_ids,
  new_game,
  play_random_game,
)

__all__ = [
  "apply_action",
  "is_terminal",
  "legal_action_ids",
  "new_game",
  "play_random_game",
]
