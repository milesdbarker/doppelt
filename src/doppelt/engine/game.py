"""Solo game engine — Phase 1 (yellow, blue, green, pink, silver)."""

from __future__ import annotations

import random
from typing import Literal

from doppelt.actions.catalog_v1 import (
    FORFEIT_PICK_ID,
    ActionKind,
    decode_action,
    mark_white_blue_id,
    mark_white_green_id,
    mark_white_silver_id,
    mark_yellow_id,
    passive_platter_id,
    passive_pool_id,
    pick_die_id,
)
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import get_score_sheet
from doppelt.core.scoring import score_sheet_areas
from doppelt.core.silver import silver_row_color
from doppelt.core.solo_passive import DieRoll, split_passive_roll
from doppelt.core.state import GameState
from doppelt.core.types import ALL_DICE, Dice, blue_entry_value
from doppelt.engine.bonus_flow import (
    apply_bonus_silver_mark,
    apply_bonus_yellow_cross,
    clear_bonus_queue,
    legal_bonus_action_ids,
    try_enter_bonus_phase,
)
from doppelt.engine.bonuses import enqueue_mark_bonuses
from doppelt.engine.silver_flow import (
    apply_silver_mark,
    apply_silver_skip,
    legal_silver_mark_action_ids,
    silver_resolution_complete,
    start_silver_resolution,
)

SUPPORTED_PICK_DICE = frozenset(
    {Dice.YELLOW, Dice.BLUE, Dice.GREEN, Dice.PINK, Dice.WHITE, Dice.SILVER}
)


WhiteMode = Literal["blue", "green", "silver"]


def new_game(seed: int, player_count: int = 1) -> GameState:

    if player_count != 1:
        raise NotImplementedError("Phase 1 engine supports solo only")

    state = GameState(
        seed=seed,
        player_count=player_count,
        round_index=1,
        phase=Phase.ACTIVE_PICK,
        sheet=PlayerSheet.empty(),
        rng=random.Random(seed),
    )

    state.faces = {die: 1 for die in ALL_DICE}

    _begin_active_turn(state)

    return state


def is_terminal(state: GameState) -> tuple[bool, dict[str, int]]:

    if state.phase is not Phase.GAME_OVER:
        return False, score_sheet_areas(state.sheet)

    return True, score_sheet_areas(state.sheet)


def legal_action_ids(state: GameState) -> list[int]:

    if state.phase is Phase.GAME_OVER:
        return []

    if state.phase is Phase.ACTIVE_PICK:
        actions: list[int] = []

        for die in state.hand:
            if die in SUPPORTED_PICK_DICE and _legal_active_pick(state, die):
                actions.append(pick_die_id(die))

        if _can_forfeit(state):
            actions.append(FORFEIT_PICK_ID)

        return actions

    if state.phase is Phase.ACTIVE_MARK_YELLOW:
        assert state.pending_die is Dice.YELLOW and state.pending_value is not None

        return [
            mark_yellow_id(cell_id)
            for cell_id in state.sheet.yellow_cell_ids_for_value(state.pending_value)
            if state.sheet.can_mark_yellow(cell_id, state.pending_value)
        ]

    if state.phase is Phase.ACTIVE_MARK_WHITE:
        modes = _white_mark_modes(state)

        actions: list[int] = []

        if "blue" in modes:
            actions.append(mark_white_blue_id())

        if "green" in modes:
            actions.append(mark_white_green_id())

        if "silver" in modes:
            actions.append(mark_white_silver_id())

        return actions

    if state.phase is Phase.ACTIVE_MARK_SILVER:
        return legal_silver_mark_action_ids(state)

    if state.phase is Phase.RESOLVE_BONUS:
        return legal_bonus_action_ids(state)

    if state.phase is Phase.PASSIVE_PICK:
        return _legal_passive_action_ids(state)

    return []


def apply_action(state: GameState, action_id: int) -> GameState:

    action = decode_action(action_id)

    if action_id not in legal_action_ids(state):
        raise ValueError(f"illegal action {action_id} in phase {state.phase}")

    state.action_log.append(action_id)

    if action.kind is ActionKind.FORFEIT_PICK:
        _forfeit_pick(state)

    elif action.kind is ActionKind.PICK_DIE:
        assert action.die is not None

        _pick_active_die(state, action.die)

    elif action.kind is ActionKind.MARK_YELLOW:
        assert action.yellow_cell_id is not None

        if state.phase is Phase.RESOLVE_BONUS:
            apply_bonus_yellow_cross(state, action.yellow_cell_id)
        else:
            _mark_pending_yellow(state, action.yellow_cell_id)

    elif action.kind is ActionKind.MARK_WHITE_BLUE:
        _mark_pending_white(state, "blue")

    elif action.kind is ActionKind.MARK_WHITE_GREEN:
        _mark_pending_white(state, "green")

    elif action.kind is ActionKind.MARK_WHITE_SILVER:
        _mark_pending_white(state, "silver")

    elif action.kind is ActionKind.MARK_SILVER:
        assert action.silver_row_index is not None
        assert action.silver_value is not None

        if state.phase is Phase.RESOLVE_BONUS:
            apply_bonus_silver_mark(
                state,
                silver_row_color(action.silver_row_index),
                action.silver_value,
            )
        else:
            apply_silver_mark(state, silver_row_color(action.silver_row_index))

            if state.phase is not Phase.RESOLVE_BONUS and silver_resolution_complete(state):
                _finish_silver_resolution(state)

    elif action.kind is ActionKind.SILVER_SKIP_CASCADE:
        apply_silver_skip(state)

        if state.phase is not Phase.RESOLVE_BONUS and silver_resolution_complete(state):
            _finish_silver_resolution(state)

    elif action.kind is ActionKind.PASSIVE_PICK:
        assert action.die is not None

        _pick_passive_die(state, action.die, from_pool=action.passive_from_pool)

    else:
        raise ValueError(f"unsupported action {action}")

    _process_bonus_resume(state)

    if state.phase is Phase.ACTIVE_MARK_SILVER and silver_resolution_complete(state):
        _finish_silver_resolution(state)

    return state


def _process_bonus_resume(state: GameState) -> None:
    if state.phase is Phase.RESOLVE_BONUS:
        return
    if state.bonus_resume_after == "after_active_pick":
        state.bonus_resume_after = None
        _after_active_pick(state)
    elif state.bonus_resume_after == "finish_passive":
        state.bonus_resume_after = None
        _finish_passive_turn(state)
    elif state.bonus_resume_after == "silver_finish":
        state.bonus_resume_after = None
        if silver_resolution_complete(state):
            _finish_silver_resolution(state)


def _roll_hand(state: GameState) -> None:

    for die in state.hand:
        state.faces[die] = state.rng.randint(1, 6)


def _clear_silver_pending(state: GameState) -> None:

    state.pending_silver_values = []

    state.pending_silver_required = []

    state.silver_finish = "active"


def _begin_active_turn(state: GameState) -> None:

    state.phase = Phase.ACTIVE_PICK

    state.hand = list(ALL_DICE)

    state.platter = []

    state.passive_pool = []

    state.slots = [None, None, None]

    state.picks_made = 0

    state.pending_die = None

    state.pending_value = None

    state.pending_platter_sent = []

    _clear_silver_pending(state)

    clear_bonus_queue(state)

    _roll_hand(state)


def _blue_entry(state: GameState) -> int:

    return blue_entry_value(
        blue_face=state.faces[Dice.BLUE],
        white_face=state.faces[Dice.WHITE],
        includes_white=True,
    )


def _white_mark_modes(state: GameState) -> list[WhiteMode]:

    modes: list[WhiteMode] = []

    if state.sheet.can_mark_blue(_blue_entry(state)):
        modes.append("blue")

    if state.sheet.can_mark_green_die(state.faces[Dice.WHITE]):
        modes.append("green")

    if state.sheet.can_use_silver_value(state.faces[Dice.WHITE]):
        modes.append("silver")

    return modes


def _legal_active_pick(state: GameState, die: Dice) -> bool:

    if die is Dice.YELLOW:
        value = state.faces[die]

        return any(
            state.sheet.can_mark_yellow(cell_id, value)
            for cell_id in state.sheet.yellow_cell_ids_for_value(value)
        )

    if die is Dice.PINK:
        return state.sheet.can_mark_pink(state.faces[die])

    if die is Dice.GREEN:
        return state.sheet.can_mark_green_die(state.faces[die])

    if die is Dice.BLUE:
        return state.sheet.can_mark_blue(_blue_entry(state))

    if die is Dice.WHITE:
        return bool(_white_mark_modes(state))

    if die is Dice.SILVER:
        return state.sheet.can_use_silver_value(state.faces[die])

    return False


def _can_forfeit(state: GameState) -> bool:

    return state.picks_made < 3 and not any(_legal_active_pick(state, die) for die in state.hand)


def _mark_die_on_sheet(
    state: GameState, die: Dice, *, white_mode: WhiteMode | None = None, resume_after: str
) -> bool:
    """Apply a die mark and maybe enter bonus phase. Returns True if bonuses pending."""

    value = state.faces[die]

    if die is Dice.YELLOW:
        legal_cells = [
            cell_id
            for cell_id in state.sheet.yellow_cell_ids_for_value(value)
            if state.sheet.can_mark_yellow(cell_id, value)
        ]

        if len(legal_cells) != 1:
            raise ValueError("yellow mark requires explicit cell choice")

        result = state.sheet.mark_yellow(legal_cells[0], value)

        enqueue_mark_bonuses(
            state,
            yellow_cell_id=legal_cells[0],
            yellow_mark_result=result,
        )

        return try_enter_bonus_phase(state, Phase.ACTIVE_PICK, resume_after=resume_after)

    if die in (Dice.BLUE, Dice.WHITE) and (die is Dice.BLUE or white_mode == "blue"):
        slot = state.sheet.mark_blue(_blue_entry(state))

        enqueue_mark_bonuses(state, blue_slot=slot)

        return try_enter_bonus_phase(state, Phase.ACTIVE_PICK, resume_after=resume_after)

    if die is Dice.GREEN or (die is Dice.WHITE and white_mode == "green"):
        face = state.faces[die]

        slot = state.sheet.mark_green_die(face)

        enqueue_mark_bonuses(state, green_slot=slot)

        return try_enter_bonus_phase(state, Phase.ACTIVE_PICK, resume_after=resume_after)

    if die is Dice.PINK:
        slot = state.sheet.mark_pink(value)

        enqueue_mark_bonuses(state, pink_slot=slot, pink_value=value)

        return try_enter_bonus_phase(state, Phase.ACTIVE_PICK, resume_after=resume_after)

    raise ValueError(f"unsupported mark for die {die}")


def _cascade_values_for_sent(state: GameState, sent: list[Dice]) -> list[int]:

    return [state.faces[die] for die in sent]


def _start_active_silver_pick(
    state: GameState, *, primary_value: int, platter_sent: list[Dice]
) -> None:

    start_silver_resolution(
        state,
        primary_value=primary_value,
        cascade_values=_cascade_values_for_sent(state, platter_sent),
        finish="active",
    )


def _start_white_as_silver(state: GameState, platter_sent: list[Dice]) -> None:

    cascade_sent = list(platter_sent)

    if Dice.SILVER in state.hand:
        state.hand.remove(Dice.SILVER)

        state.platter.append(Dice.SILVER)

        cascade_sent.append(Dice.SILVER)

    _start_active_silver_pick(
        state,
        primary_value=state.faces[Dice.WHITE],
        platter_sent=cascade_sent,
    )


def _pick_active_die(state: GameState, die: Dice) -> None:

    if die not in state.hand:
        raise ValueError("die not in hand")

    value = state.faces[die]

    if die is Dice.YELLOW:
        legal_cells = [
            cell_id
            for cell_id in state.sheet.yellow_cell_ids_for_value(value)
            if state.sheet.can_mark_yellow(cell_id, value)
        ]

        if len(legal_cells) == 1:
            _complete_pick(state, die, value)

            result = state.sheet.mark_yellow(legal_cells[0], value)

            enqueue_mark_bonuses(
                state,
                yellow_cell_id=legal_cells[0],
                yellow_mark_result=result,
            )

            if not try_enter_bonus_phase(
                state, Phase.ACTIVE_PICK, resume_after="after_active_pick"
            ):
                _after_active_pick(state)

            return

        platter_sent = _complete_pick(state, die, value)

        state.pending_die = die

        state.pending_value = value

        state.phase = Phase.ACTIVE_MARK_YELLOW

        del platter_sent

        return

    if die is Dice.SILVER:
        platter_sent = _complete_pick(state, die, value)

        _start_active_silver_pick(state, primary_value=value, platter_sent=platter_sent)

        return

    if die is Dice.WHITE:
        modes = _white_mark_modes(state)

        platter_sent = _complete_pick(state, die, value)

        if len(modes) == 1:
            mode = modes[0]

            if mode == "silver":
                _start_white_as_silver(state, platter_sent)

                return

            if not _mark_die_on_sheet(
                state, die, white_mode=mode, resume_after="after_active_pick"
            ):
                _after_active_pick(state)

            return

        state.pending_die = die

        state.pending_value = value

        state.pending_platter_sent = platter_sent

        state.phase = Phase.ACTIVE_MARK_WHITE

        return

    _complete_pick(state, die, value)

    if not _mark_die_on_sheet(state, die, resume_after="after_active_pick"):
        _after_active_pick(state)


def _mark_pending_yellow(state: GameState, cell_id: int) -> None:

    assert state.pending_value is not None

    result = state.sheet.mark_yellow(cell_id, state.pending_value)

    state.pending_die = None

    state.pending_value = None

    enqueue_mark_bonuses(
        state,
        yellow_cell_id=cell_id,
        yellow_mark_result=result,
    )

    if not try_enter_bonus_phase(state, Phase.ACTIVE_PICK, resume_after="after_active_pick"):
        _after_active_pick(state)


def _mark_pending_white(state: GameState, mode: WhiteMode) -> None:

    assert state.pending_die is Dice.WHITE

    if mode == "silver":
        _start_white_as_silver(state, state.pending_platter_sent)

    elif not _mark_die_on_sheet(
        state, Dice.WHITE, white_mode=mode, resume_after="after_active_pick"
    ):
        _after_active_pick(state)

    state.pending_die = None

    state.pending_value = None

    state.pending_platter_sent = []


def _finish_silver_resolution(state: GameState) -> None:

    if state.silver_finish == "passive":
        _finish_passive_turn(state)

        return

    _after_active_pick(state)


def _complete_pick(
    state: GameState,
    die: Dice,
    value: int,
    *,
    defer_mark: bool = False,
) -> list[Dice]:

    del defer_mark

    platter_sent: list[Dice] = []

    state.hand.remove(die)

    state.slots[state.picks_made] = die

    state.picks_made += 1

    for other in list(state.hand):
        if state.faces[other] < value:
            state.hand.remove(other)

            state.platter.append(other)

            platter_sent.append(other)

    if state.picks_made >= 3:
        for other in list(state.hand):
            state.platter.append(other)

            platter_sent.append(other)

        state.hand = []

    return platter_sent


def _after_active_pick(state: GameState) -> None:

    if state.picks_made >= 3 or not state.hand:
        _finish_active_turn(state)

        return

    _roll_hand(state)

    state.phase = Phase.ACTIVE_PICK


def _forfeit_pick(state: GameState) -> None:

    state.picks_made += 1

    if state.picks_made >= 3 or not state.hand:
        _finish_active_turn(state)

        return

    _roll_hand(state)

    state.phase = Phase.ACTIVE_PICK


def _finish_active_turn(state: GameState) -> None:

    for other in list(state.hand):
        state.hand.remove(other)

        state.platter.append(other)

    state.hand = []

    _begin_passive_turn(state)


def _begin_passive_turn(state: GameState) -> None:

    state.phase = Phase.PASSIVE_PICK

    state.platter = []

    state.passive_pool = []

    _clear_silver_pending(state)

    rolls = [DieRoll(die=die, value=state.rng.randint(1, 6)) for die in ALL_DICE]

    for roll in rolls:
        state.faces[roll.die] = roll.value

    split = split_passive_roll(rolls, state.rng)

    state.platter = [roll.die for roll in split.platter]

    state.passive_pool = [roll.die for roll in split.pool]

    state.use_pool_fallback = not any(
        _legal_passive_mark(state, die, from_pool=False) for die in state.platter
    )


def _passive_white_mode(state: GameState) -> WhiteMode | None:

    modes = _white_mark_modes(state)

    if not modes:
        return None

    if "blue" in modes:
        return "blue"

    if "green" in modes:
        return "green"

    return "silver"


def _legal_passive_mark(state: GameState, die: Dice, *, from_pool: bool) -> bool:

    del from_pool

    value = state.faces[die]

    if die is Dice.YELLOW:
        return any(
            state.sheet.can_mark_yellow(cell_id, value)
            for cell_id in state.sheet.yellow_cell_ids_for_value(value)
        )

    if die is Dice.PINK:
        return state.sheet.can_mark_pink(value)

    if die is Dice.GREEN:
        return state.sheet.can_mark_green_die(value)

    if die is Dice.BLUE:
        return state.sheet.can_mark_blue(_blue_entry(state))

    if die is Dice.WHITE:
        return _passive_white_mode(state) is not None

    if die is Dice.SILVER:
        return state.sheet.can_use_silver_value(value)

    return False


def _legal_passive_action_ids(state: GameState) -> list[int]:

    actions: list[int] = []

    for die in state.platter:
        if _legal_passive_mark(state, die, from_pool=False):
            actions.append(passive_platter_id(die))

    if state.use_pool_fallback:
        for die in state.passive_pool:
            if _legal_passive_mark(state, die, from_pool=True):
                actions.append(passive_pool_id(die))

    return actions


def _pick_passive_die(state: GameState, die: Dice, *, from_pool: bool) -> None:

    if from_pool:
        if die not in state.passive_pool:
            raise ValueError("die not in passive pool")

    elif die not in state.platter:
        raise ValueError("die not on platter")

    if die is Dice.SILVER:
        start_silver_resolution(
            state,
            primary_value=state.faces[die],
            cascade_values=[],
            finish="passive",
        )

        return

    if die is Dice.WHITE:
        mode = _passive_white_mode(state)

        assert mode is not None

        if mode == "silver":
            start_silver_resolution(
                state,
                primary_value=state.faces[die],
                cascade_values=[],
                finish="passive",
            )

            return

        _mark_die_on_sheet(state, die, white_mode=mode, resume_after="finish_passive")

    elif die is Dice.YELLOW:
        legal_cells = [
            cell_id
            for cell_id in state.sheet.yellow_cell_ids_for_value(state.faces[die])
            if state.sheet.can_mark_yellow(cell_id, state.faces[die])
        ]

        if not legal_cells:
            raise ValueError("no legal yellow cell")

        result = state.sheet.mark_yellow(legal_cells[0], state.faces[die])

        enqueue_mark_bonuses(
            state,
            yellow_cell_id=legal_cells[0],
            yellow_mark_result=result,
        )

        try_enter_bonus_phase(state, Phase.PASSIVE_PICK, resume_after="finish_passive")

    else:
        _mark_die_on_sheet(state, die, resume_after="finish_passive")

    if state.phase is not Phase.RESOLVE_BONUS:
        _finish_passive_turn(state)


def _finish_passive_turn(state: GameState) -> None:

    rounds_total = get_score_sheet().rounds_by_player_count[state.player_count]

    if state.round_index >= rounds_total:
        state.phase = Phase.GAME_OVER

        return

    state.round_index += 1

    _begin_active_turn(state)


def play_random_game(seed: int, max_actions: int = 10_000) -> GameState:
    """Play random legal moves until terminal — useful for smoke tests."""

    state = new_game(seed)

    for _ in range(max_actions):
        legal = legal_action_ids(state)

        if not legal:
            break

        action_id = state.rng.choice(legal)

        apply_action(state, action_id)

        done, _ = is_terminal(state)

        if done:
            break

    return state
