"""Bot policy protocol and factory.

Each policy sees only the current ``GameState`` and the already-computed legal
catalog IDs. ``select`` returns one of those IDs. Dice RNG stays on the engine;
policies that need tie-breaks use a separate seeded stream (``POLICY_RNG_XOR``).
"""

from __future__ import annotations

from typing import Protocol

from doppelt.core.state import GameState

POLICY_RNG_XOR = 0xBAD5_EED

POLICY_NAMES = ("random_legal", "greedy_immediate", "heuristic")


class Policy(Protocol):
    name: str

    def select(self, state: GameState, legal: list[int]) -> int:
        """Choose one legal catalog action ID."""


def make_policy(name: str, seed: int) -> Policy:
    if name == "random_legal":
        from doppelt.sim.random_legal import RandomLegal

        return RandomLegal(seed)
    if name == "greedy_immediate":
        from doppelt.sim.greedy import GreedyImmediate

        return GreedyImmediate(seed)
    if name == "heuristic":
        from doppelt.sim.heuristic import Heuristic

        return Heuristic(seed)
    raise ValueError(f"unknown policy {name!r}; expected one of {POLICY_NAMES}")
