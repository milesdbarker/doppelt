"""Versioned GameState encoding for the policy network (encoding_v2).

Features are a fixed-length float vector. The legal-action mask uses the same
index space as ``ACTION_SPACE_SIZE`` (catalog v1). encoding_v2 adds pick index,
hand ranks, empty-hand flags, live scores / fox floor, and round-scaled targets.
It does **not** add catalog IDs. Old encoding_v1 checkpoints will not load.
"""

from __future__ import annotations

from dataclasses import dataclass

from doppelt.actions.catalog_v1 import ACTION_SPACE_SIZE
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import get_score_sheet
from doppelt.core.scoring import score_blue, score_green, score_pink, score_silver, score_yellow
from doppelt.core.silver import SILVER_ROW_COLORS
from doppelt.core.state import GameState
from doppelt.core.types import ALL_DICE, SCORING_COLORS, ActionTrack, BonusKind, Dice
from doppelt.engine.game import legal_action_ids

ENCODING_VERSION = 2
SCORE_SCALE = 200.0
# Round-scaled area targets (3.6.3): yellow 21 = 3 crosses; blue 28 = slot 7.
TARGET_YELLOW = 21
TARGET_BLUE = 28
# First / second pink slots whose min_value is 6 (score_sheet_v1.yaml).
PINK_FIRST_SIX_SLOT = 6
PINK_SECOND_SIX_SLOT = 11
# Equal faces: earlier in DICE_ORDER ranks *lower* (white, yellow, blue, green, pink, silver).
RANK_TIEBREAK = "DICE_ORDER"

PHASE_ORDER: tuple[Phase, ...] = tuple(Phase)
TRACK_ORDER: tuple[ActionTrack, ...] = tuple(ActionTrack)
BONUS_KIND_ORDER: tuple[BonusKind, ...] = tuple(BonusKind)
DICE_ORDER: tuple[Dice, ...] = ALL_DICE

# Layout of encode_features (keep in sync with _sheet/_dice/_global/_pending).
YELLOW_CELLS = 10
BLUE_SLOTS = 12
PINK_SLOTS = 12
GREEN_SLOTS = 12
GREEN_PAIRS = 6
SILVER_ROWS = 4
SILVER_VALUES = 6
TRACK_COUNT = 3
DICE_COUNT = len(ALL_DICE)
DICE_CHANNELS = 7  # face + hand + platter + pool + 3 slots
PHASE_COUNT = len(PHASE_ORDER)
BONUS_KIND_COUNT = len(BONUS_KIND_ORDER)
COLOR_COUNT = len(SCORING_COLORS)

SHEET_SIZE = (
    YELLOW_CELLS * 2
    + BLUE_SLOTS * 2
    + PINK_SLOTS * 2
    + GREEN_SLOTS * 2
    + GREEN_PAIRS * 2
    + SILVER_ROWS * SILVER_VALUES
    + TRACK_COUNT * 3
    + 2  # foxes, claimed-bonus count
)
DICE_SIZE = DICE_COUNT * DICE_CHANNELS
GLOBAL_SIZE = 7 + PHASE_COUNT + DICE_COUNT + 1 + DICE_COUNT
PENDING_SIZE = 1 + BONUS_KIND_COUNT + (COLOR_COUNT + 1) + 4 + (COLOR_COUNT + 1)
# pick 1/2/3, per-die rank + empty-hand, live totals, fox floor, area targets, silver columns.
PROGRESS_SIZE = 3 + DICE_COUNT * 2 + COLOR_COUNT + 2 + 2 + 5 + 12 + 1
FEATURE_SIZE = SHEET_SIZE + DICE_SIZE + GLOBAL_SIZE + PENDING_SIZE + PROGRESS_SIZE

SHEET_SLICE = slice(0, SHEET_SIZE)
DICE_SLICE = slice(SHEET_SIZE, SHEET_SIZE + DICE_SIZE)
GLOBAL_SLICE = slice(SHEET_SIZE + DICE_SIZE, SHEET_SIZE + DICE_SIZE + GLOBAL_SIZE)
PENDING_SLICE = slice(
    SHEET_SIZE + DICE_SIZE + GLOBAL_SIZE,
    SHEET_SIZE + DICE_SIZE + GLOBAL_SIZE + PENDING_SIZE,
)
PROGRESS_SLICE = slice(SHEET_SIZE + DICE_SIZE + GLOBAL_SIZE + PENDING_SIZE, FEATURE_SIZE)
CONTEXT_SIZE = GLOBAL_SIZE + PENDING_SIZE + PROGRESS_SIZE


@dataclass(frozen=True)
class EncodedState:
    """Fixed-size tensors for one decision point."""

    features: tuple[float, ...]
    mask: tuple[bool, ...]
    encoding_version: int = ENCODING_VERSION

    def __post_init__(self) -> None:
        if len(self.features) != FEATURE_SIZE:
            raise ValueError(f"expected {FEATURE_SIZE} features, got {len(self.features)}")
        if len(self.mask) != ACTION_SPACE_SIZE:
            raise ValueError(f"expected {ACTION_SPACE_SIZE} mask bits, got {len(self.mask)}")


def legal_mask(state: GameState, legal: list[int] | None = None) -> tuple[bool, ...]:
    """Bitmap over the catalog; True iff the action is currently legal."""
    ids = legal if legal is not None else legal_action_ids(state)
    bits = [False] * ACTION_SPACE_SIZE
    for action_id in ids:
        if not 0 <= action_id < ACTION_SPACE_SIZE:
            raise ValueError(f"legal action id {action_id} out of catalog range")
        bits[action_id] = True
    return tuple(bits)


def encode_state(state: GameState, legal: list[int] | None = None) -> EncodedState:
    """Encode solo ``GameState`` plus a legal-action mask."""
    return EncodedState(
        features=tuple(encode_features(state)),
        mask=legal_mask(state, legal),
    )


def encode_features(state: GameState) -> list[float]:
    """Flatten sheet, dice, and globals into ``FEATURE_SIZE`` floats in [0, 1] (approx)."""
    feats: list[float] = []
    feats.extend(_sheet_features(state.sheet))
    feats.extend(_dice_features(state))
    feats.extend(_global_features(state))
    feats.extend(_pending_features(state))
    feats.extend(_progress_features(state))
    if len(feats) != FEATURE_SIZE:
        raise RuntimeError(f"encoding_v2 size drifted: got {len(feats)}, expected {FEATURE_SIZE}")
    return feats


def _one_hot(index: int | None, size: int) -> list[float]:
    row = [0.0] * size
    if index is not None and 0 <= index < size:
        row[index] = 1.0
    return row


def _sheet_features(sheet: PlayerSheet) -> list[float]:
    layout = get_score_sheet()
    feats: list[float] = []
    for cell in sheet.yellow:
        feats.append(1.0 if cell.circled else 0.0)
        feats.append(1.0 if cell.crossed else 0.0)
    for value in sheet.blue:
        feats.append(1.0 if value is not None else 0.0)
        feats.append((value or 0) / 12.0)
    for value in sheet.pink:
        feats.append(1.0 if value is not None else 0.0)
        feats.append((value or 0) / 6.0)
    for value in sheet.green:
        feats.append(1.0 if value is not None else 0.0)
        feats.append(min((value or 0) / 24.0, 1.0))
    for star in sheet.green_stars:
        feats.append(1.0 if star is not None else 0.0)
        clipped = max(-24.0, min(24.0, float(star or 0)))
        feats.append((clipped + 24.0) / 48.0)
    for row in SILVER_ROW_COLORS:
        marked = sheet.silver[row]
        for value in range(1, 7):
            feats.append(1.0 if value in marked else 0.0)
    for track in TRACK_ORDER:
        slots = sheet.action_tracks[track]
        capacity = layout.action_tracks[track].slots
        feats.append(slots.circled / capacity)
        feats.append(slots.crossed / capacity)
        feats.append(1.0 if sheet.can_use_action(track) else 0.0)
    feats.append(min(sheet.foxes / 10.0, 1.0))
    feats.append(min(len(sheet.claimed_bonuses) / 50.0, 1.0))
    return feats


def _dice_features(state: GameState) -> list[float]:
    hand = set(state.hand)
    platter = set(state.platter)
    pool = set(state.passive_pool)
    feats: list[float] = []
    for die in DICE_ORDER:
        feats.append(state.faces.get(die, 0) / 6.0)
        feats.append(1.0 if die in hand else 0.0)
        feats.append(1.0 if die in platter else 0.0)
        feats.append(1.0 if die in pool else 0.0)
        for slot_die in state.slots:
            feats.append(1.0 if slot_die is die else 0.0)
    return feats


def _global_features(state: GameState) -> list[float]:
    return [
        state.round_index / 6.0,
        state.picks_made / 3.0,
        1.0 if state.awaiting_roll else 0.0,
        1.0 if state.plus_one_after_passive else 0.0,
        1.0 if state.use_pool_fallback else 0.0,
        1.0 if state.silver_finish == "active" else 0.0,
        1.0 if state.silver_finish == "passive" else 0.0,
        *(_one_hot(PHASE_ORDER.index(state.phase), len(PHASE_ORDER))),
        *(_one_hot(
            DICE_ORDER.index(state.pending_die) if state.pending_die is not None else None,
            len(DICE_ORDER),
        )),
        (state.pending_value or 0) / 12.0,
        *([1.0 if die in state.plus_one_dice_used else 0.0 for die in DICE_ORDER]),
    ]


def _pending_features(state: GameState) -> list[float]:
    queue = state.pending_bonuses
    feats = [min(len(queue) / 8.0, 1.0)]
    kind_index: int | None = None
    color_index: int | None = None
    if queue:
        bonus = queue[0].bonus
        kind_index = BONUS_KIND_ORDER.index(bonus.kind)
        if bonus.color is not None:
            color_index = SCORING_COLORS.index(bonus.color)
        else:
            color_index = len(SCORING_COLORS)
    feats.extend(_one_hot(kind_index, len(BONUS_KIND_ORDER)))
    feats.extend(_one_hot(color_index, len(SCORING_COLORS) + 1))

    values = state.pending_silver_values
    required = state.pending_silver_required
    rows = state.pending_silver_rows
    feats.append(min(len(values) / 8.0, 1.0))
    feats.append(min(sum(1 for flag in required if flag) / 8.0, 1.0))
    if values:
        feats.append(values[0] / 6.0)
        feats.append(1.0 if required and required[0] else 0.0)
        row = rows[0] if rows else None
        row_index = SCORING_COLORS.index(row) if row is not None else len(SCORING_COLORS)
        feats.extend(_one_hot(row_index, len(SCORING_COLORS) + 1))
    else:
        feats.append(0.0)
        feats.append(0.0)
        feats.extend(_one_hot(None, len(SCORING_COLORS) + 1))
    return feats


def hand_ranks(state: GameState) -> dict[Dice, int]:
    """Rank dice currently in hand, 1 = lowest face.

    Ties break by ``DICE_ORDER`` (white, yellow, blue, green, pink, silver):
    the earlier color is treated as lower.
    """
    in_hand = [die for die in DICE_ORDER if die in state.hand]
    in_hand.sort(key=lambda die: (state.faces.get(die, 0), DICE_ORDER.index(die)))
    return {die: index + 1 for index, die in enumerate(in_hand)}


def would_empty_hand(state: GameState, die: Dice) -> bool:
    """True if picking ``die`` now would leave the hand empty.

    Matches the engine: lower-face leftovers go to the platter; pick 3 dumps
    whatever remains.
    """
    if die not in state.hand:
        return False
    if len(state.hand) == 1:
        return True
    if state.picks_made >= 2:
        return True
    value = state.faces.get(die, 0)
    return all(
        state.faces.get(other, 0) < value for other in state.hand if other is not die
    )


def _progress_features(state: GameState) -> list[float]:
    sheet = state.sheet
    yellow = score_yellow(sheet)
    blue = score_blue(sheet)
    pink = score_pink(sheet)
    green = score_green(sheet)
    silver = score_silver(sheet)
    color_scores = [yellow, blue, pink, green, silver]
    any_zero = any(score == 0 for score in color_scores)
    fox_floor = 0 if any_zero else min(color_scores)
    rounds_left = max(0, 7 - state.round_index)

    next_pick = _one_hot(None, 3)
    if state.phase is Phase.ACTIVE_PICK and 0 <= state.picks_made <= 2:
        next_pick = _one_hot(state.picks_made, 3)

    ranks = hand_ranks(state)
    die_feats: list[float] = []
    for die in DICE_ORDER:
        rank = ranks.get(die)
        die_feats.append((rank or 0) / 6.0)
        die_feats.append(1.0 if would_empty_hand(state, die) else 0.0)

    next_pink = sheet.next_pink_slot()
    pink_min = None
    if next_pink is not None:
        pink_min = get_score_sheet().pink.min_values[next_pink]
    silver_fills = []
    silver_done = []
    for value in range(1, 7):
        filled = sum(1 for row in SILVER_ROW_COLORS if value in sheet.silver[row])
        silver_fills.append(filled / 4.0)
        silver_done.append(1.0 if filled == 4 else 0.0)

    return [
        *next_pick,
        *die_feats,
        min(yellow / SCORE_SCALE, 1.0),
        min(blue / SCORE_SCALE, 1.0),
        min(pink / SCORE_SCALE, 1.0),
        min(green / SCORE_SCALE, 1.0),
        min(silver / SCORE_SCALE, 1.0),
        min(fox_floor / SCORE_SCALE, 1.0),
        1.0 if any_zero else 0.0,
        min(max(0, TARGET_YELLOW - yellow) / TARGET_YELLOW, 1.0),
        min(max(0, TARGET_BLUE - blue) / TARGET_BLUE, 1.0),
        (next_pink if next_pink is not None else PINK_SLOTS) / PINK_SLOTS,
        1.0 if pink_min == 5 else 0.0,
        1.0 if pink_min == 6 else 0.0,
        1.0 if sheet.pink[PINK_FIRST_SIX_SLOT] is not None else 0.0,
        1.0 if sheet.pink[PINK_SECOND_SIX_SLOT] is not None else 0.0,
        *silver_fills,
        *silver_done,
        rounds_left / 6.0,
    ]
