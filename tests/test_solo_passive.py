"""Tests for solo passive platter assignment."""

import random

from doppelt.core.solo_passive import DieRoll, split_passive_roll
from doppelt.core.types import Dice


def _rolls(values: list[int]) -> list[DieRoll]:
    dice = list(Dice)
    return [DieRoll(die=dice[i], value=values[i]) for i in range(6)]


def test_split_lowest_three_by_value():
    split = split_passive_roll(_rolls([6, 1, 5, 2, 4, 3]), random.Random(0))
    platter_values = sorted(r.value for r in split.platter)
    pool_values = sorted(r.value for r in split.pool)
    assert platter_values == [1, 2, 3]
    assert pool_values == [4, 5, 6]


def test_tie_break_is_random_but_reproducible():
    rolls = _rolls([2, 2, 2, 2, 5, 6])
    split_a = split_passive_roll(rolls, random.Random(42))
    split_b = split_passive_roll(rolls, random.Random(42))
    split_other = split_passive_roll(rolls, random.Random(99))

    assert {r.die for r in split_a.platter} == {r.die for r in split_b.platter}
    assert len(split_a.platter) == 3
    assert all(r.value == 2 for r in split_a.platter)
    assert {r.value for r in split_a.pool} == {2, 5, 6}
    # Different seeds should usually pick a different set of three 2s.
    assert {r.die for r in split_other.platter} != {r.die for r in split_a.platter}


def test_requires_six_dice():
    short = [DieRoll(die=Dice.WHITE, value=1), DieRoll(die=Dice.YELLOW, value=2)]
    try:
        split_passive_roll(short, random.Random(0))
    except ValueError as exc:
        assert "6 dice" in str(exc)
    else:
        raise AssertionError("expected ValueError")
