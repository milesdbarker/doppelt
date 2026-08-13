"""One-step greedy: pick the legal action with the highest immediate total score."""

from __future__ import annotations

import random

from doppelt.core.scoring import total_score
from doppelt.core.state import GameState
from doppelt.engine.game import apply_action
from doppelt.sim.policy import POLICY_RNG_XOR


class GreedyImmediate:
    """Try each legal action once; keep the best ``total_score`` (random among ties)."""

    name = "greedy_immediate"

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed ^ POLICY_RNG_XOR)

    def select(self, state: GameState, legal: list[int]) -> int:
        if not legal:
            raise ValueError("GreedyImmediate received no legal actions")
        if len(legal) == 1:
            return legal[0]

        best_score: int | None = None
        best_ids: list[int] = []
        for action_id in legal:
            trial = state.copy_for_trial()
            apply_action(trial, action_id, check_legal=False)
            score = total_score(trial.sheet)
            if best_score is None or score > best_score:
                best_score = score
                best_ids = [action_id]
            elif score == best_score:
                best_ids.append(action_id)
        return self._rng.choice(best_ids)
