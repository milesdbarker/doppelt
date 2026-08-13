"""Hand-tuned heuristic: real score + setup (silver, foxes, blue, yellow families).

One-step search like GreedyImmediate, plus a short follow-through when an action
only opens a mark-choice phase (white / yellow / silver) so those picks can be
compared to instant marks.
"""

from __future__ import annotations

import random
from functools import lru_cache

from doppelt.actions.catalog_v1 import ActionKind, decode_action
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import get_score_sheet
from doppelt.core.scoring import score_blue, score_sheet_areas, total_score
from doppelt.core.silver import SILVER_ROW_COLORS
from doppelt.core.state import GameState
from doppelt.core.types import ActionTrack, Color, Dice
from doppelt.engine.game import apply_action, legal_action_ids
from doppelt.sim.policy import POLICY_RNG_XOR

CHOICE_PHASES = frozenset(
    {
        Phase.ACTIVE_MARK_WHITE,
        Phase.ACTIVE_MARK_YELLOW,
        Phase.ACTIVE_MARK_SILVER,
        Phase.PASSIVE_MARK_YELLOW,
        Phase.PLUS_ONE_MARK_YELLOW,
    }
)
MAX_CHOICE_DEPTH = 2

FOX_PER_LIVE_COLOR = 6.0
FOX_ALL_COLORS = 12.0
FOX_BANKED = 9.0
FOX_CLAIM = 8.0
YELLOW_PENDING_CIRCLE = 3.0
YELLOW_PARTIAL_LINE = 4.0
YELLOW_FAMILY_FOCUS = 2.8
YELLOW_FAMILY_MIX = 4.0
YELLOW_EVEN_PREF = 1.8
YELLOW_EVEN_ROWS = frozenset({0, 2, 4})
YELLOW_ODD_ROWS = frozenset({1, 3})
BLUE_PROGRESS = 1.5
BLUE_OPENER = 0.9
BLUE_HEADROOM = 0.35
SILVER_CHAIN = (0.0, 1.5, 6.0, 16.0)
GREEN_OPEN_LEFT = 0.15
TRACK_CIRCLE = 1.5

YELLOW_TARGET = 21
BLUE_TARGET = 28
LATE_TARGET_ROUND = 5
LATE_YELLOW_GAP = 2.4
LATE_BLUE_GAP = 1.8

MISS_ROLL_PICK1 = -32.0
MISS_ROLL_PICK2 = -20.0
SILVER_SWEEP = 24.0
SILVER_WEAK = -12.0
PINK_BONUS_SLOTS = frozenset({4, 5, 6, 7})
PINK_HARD_SLOTS = frozenset({5, 6})
PINK_MISS_BONUS = -20.0
PINK_HIT_BONUS = 5.0
PINK_RESERVE_SETUP = 7.0
PINK_RESERVE_READY = 12.0
PINK_RESERVE_WASTE = -22.0
BLUE_PINK_WILD_SLOT = 6
YELLOW_ROW4 = 4

SILVER_FULL_SWEEP = 48.0
UNLOCK_WHITE = 10.0
UNLOCK_WEIGHT = 12.0

GREEN_GOOD = 12.0
GREEN_OK = 4.0
GREEN_BAD = -12.0
GREEN_PREFERRED: dict[int, frozenset[int]] = {
    0: frozenset({4, 5, 6}),
    1: frozenset({1, 2, 3}),
    2: frozenset({4, 5, 6}),
    4: frozenset({5, 6}),
    5: frozenset({1, 2}),
}
ROUND4_GREEN6 = 14.0
MIN_BLUE_BY_ROUND = (0, 9, 7, 6, 5, 4, 3)

REROLL_NO_ACCEPTABLE = 16.0
GREEN_LATE_LOW = -22.0

PRIOR_REROLL = 0.6
PRIOR_UNLOCK = 0.4
PRIOR_FORFEIT = -1.5
PRIOR_PASSIVE_SKIP = -1.5
PRIOR_END_PLUS_ONE = -0.3


@lru_cache(maxsize=1)
def _yellow_line_ids() -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    cells = get_score_sheet().yellow.cells
    rows: dict[int, list[int]] = {}
    cols: dict[int, list[int]] = {}
    for cell in cells:
        rows.setdefault(cell.row, []).append(cell.id)
        cols.setdefault(cell.col, []).append(cell.id)
    return (
        tuple(tuple(ids) for ids in rows.values()),
        tuple(tuple(ids) for ids in cols.values()),
    )


def _setup_scale(round_index: int) -> float:
    return max(0.35, (7 - round_index) / 6)


def _fox_setup(sheet: PlayerSheet) -> float:
    areas = score_sheet_areas(sheet)
    color_scores = [areas[name] for name in ("yellow", "blue", "pink", "green", "silver")]
    live = sum(1 for score in color_scores if score > 0)
    value = FOX_PER_LIVE_COLOR * live
    value += FOX_BANKED * sheet.foxes
    if live == 5:
        value += FOX_ALL_COLORS
    return value


def _yellow_family_counts(sheet: PlayerSheet) -> tuple[int, int]:
    even = 0
    odd = 0
    for cell in get_score_sheet().yellow.cells:
        if not sheet.yellow[cell.id].circled:
            continue
        if cell.row in YELLOW_EVEN_ROWS:
            even += 1
        elif cell.row in YELLOW_ODD_ROWS:
            odd += 1
    return even, odd


def _yellow_setup(sheet: PlayerSheet) -> float:
    pending = sum(1 for cell in sheet.yellow if cell.circled and not cell.crossed)
    value = YELLOW_PENDING_CIRCLE * pending
    rows, cols = _yellow_line_ids()
    for line in (*rows, *cols):
        circled = sum(1 for cell_id in line if sheet.yellow[cell_id].circled)
        if 0 < circled < len(line):
            value += YELLOW_PARTIAL_LINE * circled / len(line)
    even, odd = _yellow_family_counts(sheet)
    value += YELLOW_FAMILY_FOCUS * max(even, odd)
    value -= YELLOW_FAMILY_MIX * min(even, odd)
    value += YELLOW_EVEN_PREF * even
    return value


def _projected_yellow_score(sheet: PlayerSheet) -> int:
    crosses = sheet.yellow_cross_count()
    pending = sum(1 for cell in sheet.yellow if cell.circled and not cell.crossed)
    projected = min(10, crosses + pending)
    if projected == 0:
        return 0
    return get_score_sheet().yellow.score_by_cross_count[projected - 1]


def _blue_setup(sheet: PlayerSheet) -> float:
    last = sheet.last_blue_value()
    if last is None:
        return 0.0
    filled = sum(1 for entry in sheet.blue if entry is not None)
    remaining = len(sheet.blue) - filled
    value = BLUE_PROGRESS * filled
    value += BLUE_HEADROOM * last * remaining / max(remaining + filled - 1, 1)
    if filled == 1:
        value += BLUE_OPENER * last
    return value


def _silver_chains(sheet: PlayerSheet) -> float:
    value = 0.0
    for face in range(1, 7):
        filled = sum(1 for row in SILVER_ROW_COLORS if face in sheet.silver[row])
        if 1 <= filled <= 3:
            value += SILVER_CHAIN[filled]
    return value


def _green_setup(sheet: PlayerSheet) -> float:
    value = 0.0
    for pair_index in range(len(sheet.green_stars)):
        left = pair_index * 2
        right = left + 1
        if sheet.green[left] is not None and sheet.green[right] is None:
            value += GREEN_OPEN_LEFT * sheet.green[left]
    return value


def _yellow_row_circled_count(sheet: PlayerSheet, row: int) -> int:
    return sum(
        1
        for cell in get_score_sheet().yellow.cells
        if cell.row == row and sheet.yellow[cell.id].circled
    )


def _yellow_row_size(row: int) -> int:
    return sum(1 for cell in get_score_sheet().yellow.cells if cell.row == row)


def _pink_hard_reserve(state: GameState) -> float:
    """Hold yellow row-4 / blue-6 pink wilds for pink slots 5–6."""
    sheet = state.sheet
    slot = sheet.next_pink_slot()
    y4 = _yellow_row_circled_count(sheet, YELLOW_ROW4)
    y4_size = _yellow_row_size(YELLOW_ROW4)
    blue_filled = sum(1 for entry in sheet.blue if entry is not None)
    if slot is None or slot > 6:
        return 0.0
    ready = slot in PINK_HARD_SLOTS
    value = 0.0
    if ready:
        if y4 >= y4_size:
            value += PINK_RESERVE_READY
        elif y4 == y4_size - 1:
            value += PINK_RESERVE_SETUP * 0.5
        elif y4 == 0:
            value += PINK_RESERVE_SETUP * 0.35
        if blue_filled > BLUE_PINK_WILD_SLOT:
            value += PINK_RESERVE_READY
        elif blue_filled == BLUE_PINK_WILD_SLOT:
            value += PINK_RESERVE_SETUP * 0.5
        elif 4 <= blue_filled < BLUE_PINK_WILD_SLOT:
            value += PINK_RESERVE_SETUP * blue_filled / BLUE_PINK_WILD_SLOT
        return value
    if state.round_index >= 6:
        return 0.0
    if y4 >= y4_size:
        value += PINK_RESERVE_WASTE
    elif y4 == y4_size - 1:
        value += PINK_RESERVE_SETUP
    elif y4 == 0:
        value += PINK_RESERVE_SETUP * 0.2
    if blue_filled > BLUE_PINK_WILD_SLOT:
        value += PINK_RESERVE_WASTE
    elif blue_filled == BLUE_PINK_WILD_SLOT:
        value += PINK_RESERVE_SETUP
    elif blue_filled >= 3:
        value += PINK_RESERVE_SETUP * 0.45 * blue_filled / BLUE_PINK_WILD_SLOT
    return value


def _track_setup(sheet: PlayerSheet) -> float:
    return TRACK_CIRCLE * sum(sheet.action_tracks[track].circled for track in ActionTrack)


def _late_color_targets(state: GameState) -> float:
    if state.round_index < LATE_TARGET_ROUND:
        return 0.0
    value = 0.0
    projected_yellow = _projected_yellow_score(state.sheet)
    if projected_yellow < YELLOW_TARGET:
        value -= LATE_YELLOW_GAP * (YELLOW_TARGET - projected_yellow)
    blue = score_blue(state.sheet)
    if blue < BLUE_TARGET:
        value -= LATE_BLUE_GAP * (BLUE_TARGET - blue)
    return value


def evaluate_state(state: GameState) -> float:
    """Real end-game score plus round-scaled setup value."""
    sheet = state.sheet
    setup = (
        _fox_setup(sheet)
        + _yellow_setup(sheet)
        + _blue_setup(sheet)
        + _silver_chains(sheet)
        + _green_setup(sheet)
        + _track_setup(sheet)
    )
    return (
        total_score(sheet)
        + _setup_scale(state.round_index) * setup
        + _late_color_targets(state)
        + _pink_hard_reserve(state)
    )


def _action_prior(action_id: int) -> float:
    kind = decode_action(action_id).kind
    if kind is ActionKind.USE_REROLL:
        return PRIOR_REROLL
    if kind is ActionKind.UNLOCK_PLATTER:
        return PRIOR_UNLOCK
    if kind is ActionKind.FORFEIT_PICK:
        return PRIOR_FORFEIT
    if kind is ActionKind.PASSIVE_SKIP:
        return PRIOR_PASSIVE_SKIP
    if kind is ActionKind.END_PLUS_ONE:
        return PRIOR_END_PLUS_ONE
    return 0.0


def _dice_sent_by_active_pick(state: GameState, die: Dice) -> list[Dice]:
    value = state.faces[die]
    sent = [other for other in state.hand if other is not die and state.faces[other] < value]
    if state.picks_made + 1 >= 3:
        sent.extend(other for other in state.hand if other is not die and other not in sent)
    return sent


def _missed_roll_penalty(state: GameState, action_id: int) -> float:
    action = decode_action(action_id)
    if action.kind is not ActionKind.PICK_DIE or action.die is None:
        return 0.0
    if state.phase is not Phase.ACTIVE_PICK or state.awaiting_roll or state.picks_made >= 2:
        return 0.0
    value = state.faces[action.die]
    others = [other for other in state.hand if other is not action.die]
    if others and not all(state.faces[other] < value for other in others):
        return 0.0
    return MISS_ROLL_PICK1 if state.picks_made == 0 else MISS_ROLL_PICK2


def _predict_silver_marks(
    state: GameState, primary_value: int, platter_sent: list[Dice]
) -> tuple[int, bool, int]:
    silver = {row: set(state.sheet.silver[row]) for row in SILVER_ROW_COLORS}

    def can_mark(value: int, row: Color) -> bool:
        return 1 <= value <= 6 and value not in silver[row]

    def column_complete(value: int) -> bool:
        return all(value in silver[row] for row in SILVER_ROW_COLORS)

    marks = 0
    bonus = False

    def place(value: int, row_constraint: Color | None) -> None:
        nonlocal marks, bonus
        if row_constraint is not None:
            if can_mark(value, row_constraint):
                silver[row_constraint].add(value)
                marks += 1
                bonus = bonus or column_complete(value)
            return
        legal = [row for row in SILVER_ROW_COLORS if can_mark(value, row)]
        if not legal:
            return
        for row in legal:
            silver[row].add(value)
            if column_complete(value):
                bonus = True
                marks += 1
                return
            silver[row].remove(value)
        silver[legal[0]].add(value)
        marks += 1

    place(primary_value, None)
    for die in platter_sent:
        face = state.faces[die]
        if die in (Dice.WHITE, Dice.SILVER):
            place(face, None)
        else:
            place(face, Color(die.value))
    max_filled = max(
        sum(1 for row in SILVER_ROW_COLORS if face in silver[row]) for face in range(1, 7)
    )
    return marks, bonus, max_filled


def _silver_pick_adjust(state: GameState, action_id: int) -> float:
    action = decode_action(action_id)
    primary: int | None = None
    platter_sent: list[Dice] = []
    if action.kind is ActionKind.PICK_DIE and action.die is Dice.SILVER:
        primary = state.faces[Dice.SILVER]
        platter_sent = _dice_sent_by_active_pick(state, Dice.SILVER)
    elif action.kind is ActionKind.PASSIVE_PICK and action.die is Dice.SILVER:
        primary = state.faces[Dice.SILVER]
    elif action.kind is ActionKind.PLUS_ONE_PICK and action.die is Dice.SILVER:
        primary = state.faces[Dice.SILVER]
    elif action.kind is ActionKind.MARK_WHITE_SILVER:
        primary = state.faces[Dice.WHITE]
        platter_sent = list(state.pending_platter_sent)
    else:
        return 0.0
    marks, column_bonus, max_filled = _predict_silver_marks(state, primary, platter_sent)
    if (
        action.kind is ActionKind.PICK_DIE
        and state.picks_made < 2
        and state.sheet.action_tracks[ActionTrack.UNLOCK].circled >= 2
        and marks >= 6
    ):
        return SILVER_FULL_SWEEP
    if marks >= 4:
        return SILVER_SWEEP
    if marks <= 2 and not column_bonus and max_filled < 3:
        return SILVER_WEAK
    return 0.0


def _preferred_green_faces(slot: int) -> frozenset[int]:
    preferred = GREEN_PREFERRED.get(slot)
    if preferred is not None:
        return preferred
    return frozenset({4, 5, 6}) if slot % 2 == 0 else frozenset({1, 2, 3})


def _green_star_if_face(sheet: PlayerSheet, slot: int, face: int) -> int | None:
    entry = face * sheet.green_multiplier(slot)
    if slot % 2 == 0:
        right = sheet.green[slot + 1]
        if right is None:
            return None
        return entry - right
    left = sheet.green[slot - 1]
    if left is None:
        return None
    return left - entry


def _green_face_quality(sheet: PlayerSheet, face: int) -> float:
    slot = sheet.next_green_slot()
    if slot is None or not 1 <= face <= 6:
        return 0.0
    preferred = _preferred_green_faces(slot)
    star = _green_star_if_face(sheet, slot, face)
    if star is not None and star <= 0:
        return GREEN_BAD
    listed = slot in GREEN_PREFERRED
    if face in preferred:
        return GREEN_GOOD if listed else GREEN_OK
    return GREEN_BAD if listed else -3.0


def _is_low_green_slot(slot: int | None) -> bool:
    return slot is not None and slot % 2 == 1


def _late_active_pick_index(state: GameState, action) -> int | None:
    if action.kind is ActionKind.PICK_DIE and state.phase is Phase.ACTIVE_PICK:
        return state.picks_made
    if action.kind is ActionKind.MARK_WHITE_GREEN and state.phase is Phase.ACTIVE_MARK_WHITE:
        return max(0, state.picks_made - 1)
    return None


def _hand_has_higher_face(state: GameState, face: int, skip: Dice | None) -> bool:
    return any(die is not skip and state.faces[die] > face for die in state.hand)


def _green_mark_adjust(state: GameState, action_id: int) -> float:
    action = decode_action(action_id)
    face: int | None = None
    skip: Dice | None = None
    if action.kind in (ActionKind.PICK_DIE, ActionKind.PASSIVE_PICK, ActionKind.PLUS_ONE_PICK):
        if action.die is Dice.GREEN:
            face = state.faces[Dice.GREEN]
            skip = Dice.GREEN
    elif action.kind is ActionKind.MARK_WHITE_GREEN:
        face = state.faces[Dice.WHITE]
    if face is None:
        return 0.0
    quality = _green_face_quality(state.sheet, face)
    pick_index = _late_active_pick_index(state, action)
    if (
        quality > 0
        and pick_index is not None
        and pick_index >= 1
        and _is_low_green_slot(state.sheet.next_green_slot())
        and _hand_has_higher_face(state, face, skip)
    ):
        return GREEN_LATE_LOW
    return quality


def _green_auto_is_positive(sheet: PlayerSheet) -> bool:
    slot = sheet.next_green_slot()
    if slot is None:
        return False
    face = 6 if slot % 2 == 0 else 1
    return _green_face_quality(sheet, face) > 0


def _round4_wild_adjust(state: GameState, action_id: int) -> float:
    action = decode_action(action_id)
    if action.kind is not ActionKind.CHOOSE_WILD_COLOR:
        return 0.0
    if state.round_index != 4 or action.wild_color is not Color.GREEN:
        return 0.0
    if _green_auto_is_positive(state.sheet):
        return ROUND4_GREEN6
    return 0.0


def _min_blue_for_round(round_index: int) -> int:
    if 0 <= round_index < len(MIN_BLUE_BY_ROUND):
        return MIN_BLUE_BY_ROUND[round_index]
    return 4


def _blue_showing_sum(state: GameState) -> int:
    return state.faces[Dice.BLUE] + state.faces[Dice.WHITE]


def _yellow_face_hits_even(sheet: PlayerSheet, face: int) -> bool:
    even, odd = _yellow_family_counts(sheet)
    if odd > even and odd >= 2:
        return False
    for cell in get_score_sheet().yellow.cells:
        if cell.row not in YELLOW_EVEN_ROWS or cell.value != face:
            continue
        if not sheet.yellow[cell.id].crossed:
            return True
    return False


def _pink_face_hits_bonus(sheet: PlayerSheet, face: int) -> bool:
    slot = sheet.next_pink_slot()
    if slot is None:
        return False
    threshold = get_score_sheet().pink.min_values[slot]
    if threshold is None:
        return False
    return face >= threshold


def _blue_showing_is_acceptable(state: GameState) -> bool:
    sheet = state.sheet
    total = _blue_showing_sum(state)
    if not sheet.can_mark_blue(total):
        return False
    if (
        sheet.next_blue_slot() == BLUE_PINK_WILD_SLOT
        and sheet.next_pink_slot() not in PINK_HARD_SLOTS
        and state.round_index < 6
    ):
        return False
    return total >= _min_blue_for_round(state.round_index)


def _die_is_acceptable(state: GameState, die: Dice) -> bool:
    face = state.faces[die]
    sheet = state.sheet
    if die is Dice.YELLOW:
        return _yellow_face_hits_even(sheet, face)
    if die is Dice.GREEN:
        return _green_face_quality(sheet, face) > 0
    if die is Dice.BLUE:
        return _blue_showing_is_acceptable(state)
    if die is Dice.PINK:
        return _pink_face_hits_bonus(sheet, face)
    if die is Dice.WHITE:
        return (
            _yellow_face_hits_even(sheet, face)
            or _green_face_quality(sheet, face) > 0
            or _pink_face_hits_bonus(sheet, face)
            or _blue_showing_is_acceptable(state)
        )
    return False


def _any_acceptable_in_hand(state: GameState) -> bool:
    return any(_die_is_acceptable(state, die) for die in state.hand)


def _reroll_adjust(state: GameState, action_id: int) -> float:
    action = decode_action(action_id)
    if action.kind is not ActionKind.USE_REROLL:
        return 0.0
    if state.phase is not Phase.ACTIVE_PICK or state.awaiting_roll:
        return 0.0
    if _any_acceptable_in_hand(state):
        return 0.0
    return REROLL_NO_ACCEPTABLE


def _blue_unlock_p(state: GameState) -> float:
    sheet = state.sheet
    if sheet.next_blue_slot() is None:
        return 0.0
    if (
        sheet.next_blue_slot() == BLUE_PINK_WILD_SLOT
        and sheet.next_pink_slot() not in PINK_HARD_SLOTS
        and state.round_index < 6
    ):
        return 0.0
    last = sheet.last_blue_value()
    minimum = _min_blue_for_round(state.round_index)
    good = 0
    for blue in range(1, 7):
        for white in range(1, 7):
            total = blue + white
            if last is None:
                ok = total >= minimum
            else:
                ok = total <= last and total >= minimum
            if ok:
                good += 1
    return good / 36


def _yellow_even_unlock_p(sheet: PlayerSheet) -> float:
    even, odd = _yellow_family_counts(sheet)
    if odd > even and odd >= 2:
        return 0.15
    faces: set[int] = set()
    for cell in get_score_sheet().yellow.cells:
        if cell.row not in YELLOW_EVEN_ROWS:
            continue
        if sheet.yellow[cell.id].crossed:
            continue
        faces.add(cell.value)
    return len(faces) / 6


def _green_unlock_p(sheet: PlayerSheet) -> float:
    if sheet.next_green_slot() is None:
        return 0.0
    good = sum(1 for face in range(1, 7) if _green_face_quality(sheet, face) > 0)
    return good / 6


def _pink_unlock_p(sheet: PlayerSheet) -> float:
    slot = sheet.next_pink_slot()
    if slot is None:
        return 0.0
    threshold = get_score_sheet().pink.min_values[slot]
    if threshold is None:
        return 0.0
    return sum(1 for face in range(1, 7) if face >= threshold) / 6


def _unlock_accept_p(state: GameState, die: Dice) -> float:
    if die is Dice.YELLOW:
        return _yellow_even_unlock_p(state.sheet)
    if die is Dice.GREEN:
        return _green_unlock_p(state.sheet)
    if die is Dice.BLUE:
        return _blue_unlock_p(state)
    if die is Dice.PINK:
        return _pink_unlock_p(state.sheet)
    if die is Dice.SILVER:
        open_faces = sum(1 for face in range(1, 7) if state.sheet.can_use_silver_value(face))
        return 0.35 * open_faces / 6
    return 0.0


def _unlock_adjust(state: GameState, action_id: int) -> float:
    action = decode_action(action_id)
    if action.kind is not ActionKind.UNLOCK_PLATTER or action.die is None:
        return 0.0
    if action.die is Dice.WHITE:
        return UNLOCK_WHITE
    return UNLOCK_WEIGHT * _unlock_accept_p(state, action.die)


def _fox_claim_adjust(before: PlayerSheet, after: PlayerSheet) -> float:
    gained = after.foxes - before.foxes
    if gained <= 0:
        return 0.0
    return FOX_CLAIM * gained


def _pink_bonus_adjust(before: PlayerSheet, after: PlayerSheet) -> float:
    slot = before.next_pink_slot()
    if slot not in PINK_BONUS_SLOTS:
        return 0.0
    if after.next_pink_slot() == slot:
        return 0.0
    written = after.pink[slot]
    if written is None:
        return 0.0
    threshold = get_score_sheet().pink.min_values[slot]
    if threshold is None:
        return 0.0
    if written >= threshold:
        return PINK_HIT_BONUS
    return PINK_MISS_BONUS


def _value_after(state: GameState, action_id: int, *, depth: int, root: GameState) -> float:
    trial = state.copy_for_trial()
    apply_action(trial, action_id, check_legal=False)
    prior = 0.0
    if depth == 0:
        prior = (
            _action_prior(action_id)
            + _missed_roll_penalty(state, action_id)
            + _silver_pick_adjust(state, action_id)
            + _unlock_adjust(state, action_id)
            + _green_mark_adjust(state, action_id)
            + _round4_wild_adjust(state, action_id)
            + _reroll_adjust(state, action_id)
        )
    if depth >= MAX_CHOICE_DEPTH or trial.phase not in CHOICE_PHASES:
        return (
            evaluate_state(trial)
            + prior
            + _pink_bonus_adjust(root.sheet, trial.sheet)
            + _fox_claim_adjust(root.sheet, trial.sheet)
        )
    follow_ups = legal_action_ids(trial)
    if not follow_ups:
        return (
            evaluate_state(trial)
            + prior
            + _pink_bonus_adjust(root.sheet, trial.sheet)
            + _fox_claim_adjust(root.sheet, trial.sheet)
        )
    return prior + max(
        _value_after(trial, follow_id, depth=depth + 1, root=root) for follow_id in follow_ups
    )


class Heuristic:
    """Hand-tuned priorities: silver chains, fox coverage, blue descent, yellow families."""

    name = "heuristic"

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed ^ POLICY_RNG_XOR)

    def select(self, state: GameState, legal: list[int]) -> int:
        if not legal:
            raise ValueError("Heuristic received no legal actions")
        if len(legal) == 1:
            return legal[0]

        best_value = float("-inf")
        best_ids: list[int] = []
        for action_id in legal:
            value = _value_after(state, action_id, depth=0, root=state)
            if value > best_value:
                best_value = value
                best_ids = [action_id]
            elif value == best_value:
                best_ids.append(action_id)
        return self._rng.choice(best_ids)
