"""Solo passive-turn helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass

from doppelt.core.types import Dice


@dataclass(frozen=True)
class DieRoll:
  """One die showing a face value."""

  die: Dice
  value: int


@dataclass(frozen=True)
class PassiveSplit:
  """Result of assigning a solo passive roll to platter vs pool."""

  platter: tuple[DieRoll, ...]
  pool: tuple[DieRoll, ...]


def split_passive_roll(rolls: list[DieRoll], rng: random.Random) -> PassiveSplit:
  """Put the three lowest dice on the platter; break value ties at random.

  Uses the game RNG so tie-breaks are reproducible from seed + action log.
  """
  if len(rolls) != 6:
    raise ValueError("passive roll must have exactly 6 dice")

  shuffled = list(rolls)
  rng.shuffle(shuffled)
  shuffled.sort(key=lambda r: r.value)
  platter = tuple(shuffled[:3])
  pool = tuple(shuffled[3:])
  return PassiveSplit(platter=platter, pool=pool)
