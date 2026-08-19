"""Versioned GameState encoding for the policy network (encoding_v1).

Features are a fixed-length float vector. The legal-action mask uses the same
index space as ``ACTION_SPACE_SIZE`` (catalog v1).
"""

from __future__ import annotations

from dataclasses import dataclass

from doppelt.actions.catalog_v1 import ACTION_SPACE_SIZE
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import get_score_sheet
from doppelt.core.silver import SILVER_ROW_COLORS
from doppelt.core.state import GameState
from doppelt.core.types import ALL_DICE, SCORING_COLORS, ActionTrack, BonusKind, Dice
from doppelt.engine.game import legal_action_ids

ENCODING_VERSION = 1
SCORE_SCALE = 200.0

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
FEATURE_SIZE = SHEET_SIZE + DICE_SIZE + GLOBAL_SIZE + PENDING_SIZE

SHEET_SLICE = slice(0, SHEET_SIZE)
DICE_SLICE = slice(SHEET_SIZE, SHEET_SIZE + DICE_SIZE)
GLOBAL_SLICE = slice(SHEET_SIZE + DICE_SIZE, SHEET_SIZE + DICE_SIZE + GLOBAL_SIZE)
PENDING_SLICE = slice(SHEET_SIZE + DICE_SIZE + GLOBAL_SIZE, FEATURE_SIZE)


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
    if len(feats) != FEATURE_SIZE:
        raise RuntimeError(f"encoding_v1 size drifted: got {len(feats)}, expected {FEATURE_SIZE}")
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
