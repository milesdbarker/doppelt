"""Discrete action space for the policy head (catalog v1).

The network always outputs ``ACTION_SPACE_SIZE`` logits. Only IDs in
``candidate_action_ids(state)`` can be legal; the engine's ``legal_action_ids``
is a further subset. Illegal logits are set to −inf before softmax.

Pick+mark is already a single catalog ID when the mark is unique (blue / green /
pink always; yellow and white when only one target/mode exists). Follow-up IDs
cover the remaining choice points.
"""

from __future__ import annotations

from doppelt.actions.catalog_v1 import (
    ACTION_SPACE_SIZE,
    BONUS_YELLOW_CIRCLE_BASE,
    BONUS_YELLOW_CROSS_BASE,
    CHOOSE_WILD_COLOR_BASE,
    END_ACTIVE_TURN_ID,
    END_PLUS_ONE_ID,
    FORFEIT_PICK_ID,
    MARK_SILVER_BASE,
    MARK_WHITE_BLUE_ID,
    MARK_WHITE_PINK_ID,
    MARK_YELLOW_BASE,
    PASSIVE_MARK_YELLOW_BASE,
    PASSIVE_PLATTER_BASE,
    PASSIVE_POOL_BASE,
    PASSIVE_SKIP_ID,
    PICK_DIE_BASE,
    PLUS_ONE_MARK_YELLOW_BASE,
    PLUS_ONE_PICK_BASE,
    ROLL_HAND_ID,
    SILVER_SKIP_CASCADE_ID,
    UNLOCK_PLATTER_BASE,
    USE_REROLL_ID,
    ActionKind,
    decode_action,
    encode_action,
)
from doppelt.core.phases import Phase
from doppelt.core.state import GameState
from doppelt.core.types import ALL_DICE, Dice

# Sentinels used by the policy head (true −inf; softmax maps these to 0).
ILLEGAL_LOGIT = float("-inf")

_YELLOW_CELLS = 10
_SILVER_CELLS = 24
_N_DICE = len(ALL_DICE)
_N_WILD_COLORS = 5


def _span(start: int, count: int) -> frozenset[int]:
    return frozenset(range(start, start + count))


# IDs the decoder still accepts but the engine never returns as legal.
RETIRED_ACTION_IDS: frozenset[int] = frozenset({SILVER_SKIP_CASCADE_ID})

PICK_DIE_IDS: frozenset[int] = _span(PICK_DIE_BASE, _N_DICE)
MARK_YELLOW_IDS: frozenset[int] = _span(MARK_YELLOW_BASE, _YELLOW_CELLS)
MARK_WHITE_IDS: frozenset[int] = _span(MARK_WHITE_BLUE_ID, 5)
PASSIVE_PLATTER_IDS: frozenset[int] = _span(PASSIVE_PLATTER_BASE, _N_DICE)
PASSIVE_POOL_IDS: frozenset[int] = _span(PASSIVE_POOL_BASE, _N_DICE)
MARK_SILVER_IDS: frozenset[int] = _span(MARK_SILVER_BASE, _SILVER_CELLS)
UNLOCK_IDS: frozenset[int] = _span(UNLOCK_PLATTER_BASE, _N_DICE)
PLUS_ONE_PICK_IDS: frozenset[int] = _span(PLUS_ONE_PICK_BASE, _N_DICE)
CHOOSE_WILD_COLOR_IDS: frozenset[int] = _span(CHOOSE_WILD_COLOR_BASE, _N_WILD_COLORS)
PASSIVE_MARK_YELLOW_IDS: frozenset[int] = _span(PASSIVE_MARK_YELLOW_BASE, _YELLOW_CELLS)
PLUS_ONE_MARK_YELLOW_IDS: frozenset[int] = _span(PLUS_ONE_MARK_YELLOW_BASE, _YELLOW_CELLS)
BONUS_YELLOW_CIRCLE_IDS: frozenset[int] = _span(BONUS_YELLOW_CIRCLE_BASE, _YELLOW_CELLS)
BONUS_YELLOW_CROSS_IDS: frozenset[int] = _span(BONUS_YELLOW_CROSS_BASE, _YELLOW_CELLS)

# ACTIVE_PICK splits on awaiting_roll (not a separate Phase).
ACTIVE_PRE_ROLL_IDS: frozenset[int] = UNLOCK_IDS | {ROLL_HAND_ID, END_ACTIVE_TURN_ID}
ACTIVE_POST_ROLL_IDS: frozenset[int] = PICK_DIE_IDS | {FORFEIT_PICK_ID, USE_REROLL_ID}

PHASE_ACTION_IDS: dict[Phase, frozenset[int]] = {
    Phase.ACTIVE_PICK: ACTIVE_PRE_ROLL_IDS | ACTIVE_POST_ROLL_IDS,
    Phase.ACTIVE_MARK_YELLOW: MARK_YELLOW_IDS,
    Phase.ACTIVE_MARK_WHITE: MARK_WHITE_IDS,
    Phase.ACTIVE_MARK_SILVER: MARK_SILVER_IDS,
    Phase.RESOLVE_BONUS: (
        CHOOSE_WILD_COLOR_IDS
        | BONUS_YELLOW_CIRCLE_IDS
        | BONUS_YELLOW_CROSS_IDS
        | MARK_SILVER_IDS
    ),
    Phase.PLUS_ONE: PLUS_ONE_PICK_IDS | {END_PLUS_ONE_ID},
    Phase.PASSIVE_PICK: PASSIVE_PLATTER_IDS | PASSIVE_POOL_IDS | {PASSIVE_SKIP_ID},
    Phase.PASSIVE_MARK_YELLOW: PASSIVE_MARK_YELLOW_IDS,
    Phase.PLUS_ONE_MARK_YELLOW: PLUS_ONE_MARK_YELLOW_IDS,
    Phase.GAME_OVER: frozenset(),
}

# Blue / green / pink: pick ID applies the only legal mark (no follow-up catalog ID).
COMPOSITE_PICK_DICE: frozenset[Dice] = frozenset({Dice.BLUE, Dice.GREEN, Dice.PINK})


def defined_action_ids() -> frozenset[int]:
    """Every catalog ID ``decode_action`` accepts in ``[0, ACTION_SPACE_SIZE)``."""
    ids: set[int] = set()
    for action_id in range(ACTION_SPACE_SIZE):
        try:
            decode_action(action_id)
        except ValueError:
            continue
        ids.add(action_id)
    return frozenset(ids)


def unused_action_ids() -> frozenset[int]:
    """Holes in the uint16 index space (never decoded, never legal)."""
    return frozenset(range(ACTION_SPACE_SIZE)) - defined_action_ids()


def candidate_action_ids(state: GameState) -> frozenset[int]:
    """Phase (and pre/post-roll) subset that may contain legal actions."""
    if state.phase is Phase.ACTIVE_PICK:
        if state.awaiting_roll:
            return ACTIVE_PRE_ROLL_IDS
        return ACTIVE_POST_ROLL_IDS
    return PHASE_ACTION_IDS[state.phase]


def pick_is_composite(die: Dice) -> bool:
    """True when picking this die always applies its mark with no extra catalog ID."""
    return die in COMPOSITE_PICK_DICE


def is_followup_mark(action_id: int) -> bool:
    """True for catalog IDs that only exist because a pick was not unique."""
    kind = decode_action(action_id).kind
    return kind in {
        ActionKind.MARK_YELLOW,
        ActionKind.MARK_YELLOW_PASSIVE,
        ActionKind.MARK_YELLOW_PLUS_ONE,
        ActionKind.MARK_WHITE_BLUE,
        ActionKind.MARK_WHITE_GREEN,
        ActionKind.MARK_WHITE_SILVER,
        ActionKind.MARK_WHITE_YELLOW,
        ActionKind.MARK_WHITE_PINK,
        ActionKind.MARK_SILVER,
        ActionKind.BONUS_YELLOW_CIRCLE,
        ActionKind.BONUS_YELLOW_CROSS,
        ActionKind.CHOOSE_WILD_COLOR,
    }


def assert_catalog_roundtrip() -> None:
    """Encode(decode(id)) == id for every defined ID (no duplicate semantics)."""
    for action_id in defined_action_ids():
        again = encode_action(decode_action(action_id))
        if again != action_id:
            raise AssertionError(f"catalog id {action_id} round-trips to {again}")
