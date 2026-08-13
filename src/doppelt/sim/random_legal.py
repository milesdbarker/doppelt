"""Uniform random over the current legal action list."""

from __future__ import annotations

import random

from doppelt.core.state import GameState
from doppelt.sim.policy import POLICY_RNG_XOR


class RandomLegal:
    name = "random_legal"

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed ^ POLICY_RNG_XOR)

    def select(self, state: GameState, legal: list[int]) -> int:
        del state
        if not legal:
            raise ValueError("RandomLegal received no legal actions")
        return self._rng.choice(legal)
