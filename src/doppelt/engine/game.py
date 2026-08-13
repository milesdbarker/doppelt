"""Solo game engine — Phase 1 (yellow, blue, green, pink, silver)."""

from __future__ import annotations

import random
from typing import Literal

from doppelt.actions.catalog_v1 import (
    FORFEIT_PICK_ID,
    ActionKind,
    decode_action,
    end_plus_one_id,
    mark_white_blue_id,
    mark_white_green_id,
    mark_white_pink_id,
    mark_white_silver_id,
    mark_white_yellow_id,
    mark_yellow_id,
    passive_mark_yellow_id,
    passive_platter_id,
    passive_pool_id,
    passive_skip_id,
    pick_die_id,
    plus_one_mark_yellow_id,
    plus_one_pick_id,
)
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import get_score_sheet
from doppelt.core.scoring import score_sheet_areas
from doppelt.core.silver import silver_row_color
from doppelt.core.solo_passive import DieRoll, split_passive_roll
from doppelt.core.state import GameState
from doppelt.core.types import ALL_DICE, ActionTrack, Color, Dice, blue_entry_value
from doppelt.engine.action_flow import (
    apply_roll_hand,
    apply_round_start_grants,
    apply_unlock_platter,
    apply_use_reroll,
    can_choose_unlock_or_end_turn,
    legal_active_post_roll_action_ids,
    legal_active_pre_roll_action_ids,
)
from doppelt.engine.bonus_flow import (
    apply_bonus_choose_wild_color,
    apply_bonus_silver_mark,
    apply_bonus_yellow_circle,
    apply_bonus_yellow_cross,
    clear_bonus_queue,
    legal_bonus_action_ids,
    try_enter_bonus_phase,
)
from doppelt.engine.bonuses import enqueue_mark_bonuses
from doppelt.engine.silver_flow import (
    apply_silver_mark,
    legal_silver_mark_action_ids,
    silver_resolution_complete,
    skip_unmarkable_cascade_heads,
    start_silver_resolution,
)

SUPPORTED_PICK_DICE = frozenset(
    {Dice.YELLOW, Dice.BLUE, Dice.GREEN, Dice.PINK, Dice.WHITE, Dice.SILVER}
)


WhiteMode = Literal["blue", "green", "pink", "silver", "yellow"]


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
        if state.awaiting_roll:
            actions.extend(legal_active_pre_roll_action_ids(state))
        else:
            for die in state.hand:
                if die in SUPPORTED_PICK_DICE and _legal_active_pick(state, die):
                    actions.append(pick_die_id(die))
            if _can_forfeit(state):
                actions.append(FORFEIT_PICK_ID)
            actions.extend(legal_active_post_roll_action_ids(state))
        return actions

    if state.phase is Phase.ACTIVE_MARK_YELLOW:
        if state.pending_value is None:
            return []

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

        if "yellow" in modes:
            actions.append(mark_white_yellow_id())

        if "pink" in modes:
            actions.append(mark_white_pink_id())

        return actions

    if state.phase is Phase.ACTIVE_MARK_SILVER:
        return legal_silver_mark_action_ids(state)

    if state.phase is Phase.RESOLVE_BONUS:
        return legal_bonus_action_ids(state)

    if state.phase is Phase.PLUS_ONE:
        return _legal_plus_one_action_ids(state)

    if state.phase is Phase.PASSIVE_MARK_YELLOW:
        assert state.pending_value is not None
        return [
            passive_mark_yellow_id(cell_id)
            for cell_id in state.sheet.yellow_cell_ids_for_value(state.pending_value)
            if state.sheet.can_mark_yellow(cell_id, state.pending_value)
        ]

    if state.phase is Phase.PLUS_ONE_MARK_YELLOW:
        assert state.pending_value is not None
        return [
            plus_one_mark_yellow_id(cell_id)
            for cell_id in state.sheet.yellow_cell_ids_for_value(state.pending_value)
            if state.sheet.can_mark_yellow(cell_id, state.pending_value)
        ]

    if state.phase is Phase.PASSIVE_PICK:
        return _legal_passive_action_ids(state)

    return []


def apply_action(state: GameState, action_id: int, *, check_legal: bool = True) -> GameState:
    """Apply a catalog action. Set ``check_legal=False`` when the caller already filtered."""

    action = decode_action(action_id)

    if check_legal and action_id not in legal_action_ids(state):
        raise ValueError(f"illegal action {action_id} in phase {state.phase}")

    state.action_log.append(action_id)

    if action.kind is ActionKind.FORFEIT_PICK:
        _forfeit_pick(state)

    elif action.kind is ActionKind.ROLL_HAND:
        apply_roll_hand(state)

    elif action.kind is ActionKind.USE_REROLL:
        apply_use_reroll(state)

    elif action.kind is ActionKind.UNLOCK_PLATTER:
        assert action.die is not None
        apply_unlock_platter(state, action.die)

    elif action.kind is ActionKind.END_PLUS_ONE:
        _end_plus_one_phase(state)

    elif action.kind is ActionKind.END_ACTIVE_TURN:
        if not state.awaiting_roll:
            raise ValueError("end active turn is only legal before rolling")
        if not can_choose_unlock_or_end_turn(state):
            raise ValueError("end active turn requires empty hand with unlock available")
        _finish_active_turn(state)

    elif action.kind is ActionKind.PLUS_ONE_PICK:
        assert action.die is not None
        _apply_plus_one_pick(state, action.die)

    elif action.kind is ActionKind.PICK_DIE:
        assert action.die is not None

        _pick_active_die(state, action.die)

    elif action.kind is ActionKind.MARK_YELLOW:
        assert action.yellow_cell_id is not None

        if state.phase is Phase.RESOLVE_BONUS:
            raise ValueError("use BONUS_YELLOW_* actions in RESOLVE_BONUS")
        elif state.phase is Phase.ACTIVE_MARK_YELLOW:
            _mark_pending_yellow(state, action.yellow_cell_id)
        else:
            raise ValueError(f"MARK_YELLOW not legal in phase {state.phase}")

    elif action.kind is ActionKind.BONUS_YELLOW_CIRCLE:
        assert action.yellow_cell_id is not None
        apply_bonus_yellow_circle(state, action.yellow_cell_id)

    elif action.kind is ActionKind.BONUS_YELLOW_CROSS:
        assert action.yellow_cell_id is not None
        apply_bonus_yellow_cross(state, action.yellow_cell_id)

    elif action.kind is ActionKind.MARK_YELLOW_PASSIVE:
        assert action.yellow_cell_id is not None
        _mark_passive_yellow_cell(state, action.yellow_cell_id)

    elif action.kind is ActionKind.MARK_YELLOW_PLUS_ONE:
        assert action.yellow_cell_id is not None
        _mark_plus_one_yellow_cell(state, action.yellow_cell_id)

    elif action.kind is ActionKind.MARK_WHITE_BLUE:
        _mark_pending_white(state, "blue")

    elif action.kind is ActionKind.MARK_WHITE_GREEN:
        _mark_pending_white(state, "green")

    elif action.kind is ActionKind.MARK_WHITE_SILVER:
        _mark_pending_white(state, "silver")

    elif action.kind is ActionKind.MARK_WHITE_YELLOW:
        _mark_pending_white(state, "yellow")

    elif action.kind is ActionKind.MARK_WHITE_PINK:
        _mark_pending_white(state, "pink")

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
        raise ValueError("optional silver cascade skips are not supported")

    elif action.kind is ActionKind.PASSIVE_PICK:
        assert action.die is not None

        _pick_passive_die(state, action.die, from_pool=action.passive_from_pool)

    elif action.kind is ActionKind.PASSIVE_SKIP:
        _apply_passive_skip(state)

    elif action.kind is ActionKind.CHOOSE_WILD_COLOR:
        assert action.wild_color is not None
        apply_bonus_choose_wild_color(state, action.wild_color)

    else:
        raise ValueError(f"unsupported action {action}")

    _process_bonus_resume(state)

    if state.phase is Phase.ACTIVE_MARK_SILVER:
        skip_unmarkable_cascade_heads(state)
        if silver_resolution_complete(state):
            _finish_silver_resolution(state)

    return state


def _process_bonus_resume(state: GameState) -> None:
    if state.phase is Phase.RESOLVE_BONUS:
        return
    if state.bonus_resume_after is None:
        return

    # Choice / resolution sub-phases own their continuation via pending state.
    # A deferred bonus resume must not overwrite them (e.g. stale
    # plus_one_continue starting the passive roll before white mode choice).
    if state.phase in {
        Phase.ACTIVE_MARK_WHITE,
        Phase.ACTIVE_MARK_YELLOW,
        Phase.ACTIVE_MARK_SILVER,
        Phase.PASSIVE_MARK_YELLOW,
        Phase.PLUS_ONE_MARK_YELLOW,
    }:
        state.bonus_resume_after = None
        return

    resume = state.bonus_resume_after
    state.bonus_resume_after = None

    if resume == "after_active_pick":
        if state.phase is Phase.ACTIVE_PICK:
            _after_active_pick(state)
    elif resume == "finish_passive":
        if state.phase is Phase.PASSIVE_PICK:
            _finish_passive_turn(state)
    elif resume == "plus_one_continue":
        if state.phase is Phase.PLUS_ONE:
            if not state.sheet.can_use_action(ActionTrack.PLUS_ONE):
                _end_plus_one_phase(state)
    elif resume == "silver_finish":
        if state.phase is Phase.ACTIVE_MARK_SILVER and silver_resolution_complete(state):
            _finish_silver_resolution(state)


def _roll_hand(state: GameState) -> None:

    for die in state.hand:
        state.faces[die] = state.rng.randint(1, 6)


def _clear_silver_pending(state: GameState) -> None:

    state.pending_silver_values = []

    state.pending_silver_rows = []

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

    apply_round_start_grants(state)

    state.awaiting_roll = True

    try_enter_bonus_phase(state, Phase.ACTIVE_PICK)


def _blue_entry(state: GameState) -> int:

    return blue_entry_value(
        blue_face=state.faces[Dice.BLUE],
        white_face=state.faces[Dice.WHITE],
        includes_white=True,
    )


def _white_mark_modes(state: GameState) -> list[WhiteMode]:

    modes: list[WhiteMode] = []
    white = state.faces[Dice.WHITE]

    if state.sheet.can_mark_blue(_blue_entry(state)):
        modes.append("blue")

    if _legal_yellow_mark_cells(state.sheet, white):
        modes.append("yellow")

    if state.sheet.can_mark_green_die(white):
        modes.append("green")

    if state.sheet.can_mark_pink(white):
        modes.append("pink")

    if state.sheet.can_use_silver_value(white):
        modes.append("silver")

    return modes


def _any_white_mark_legal(state: GameState) -> bool:
    """True if white can mark any color — cheapest checks first, early exit."""
    white = state.faces[Dice.WHITE]
    sheet = state.sheet
    if sheet.can_mark_pink(white) or sheet.can_mark_green_die(white):
        return True
    if sheet.can_mark_blue(_blue_entry(state)):
        return True
    if any(
        not sheet.yellow[cell_id].crossed
        for cell_id in sheet.yellow_cell_ids_for_value(white)
    ):
        return True
    return sheet.can_use_silver_value(white)


def _die_has_legal_mark(state: GameState, die: Dice) -> bool:
    value = state.faces[die]
    sheet = state.sheet
    if die is Dice.YELLOW:
        return any(
            not sheet.yellow[cell_id].crossed
            for cell_id in sheet.yellow_cell_ids_for_value(value)
        )
    if die is Dice.PINK:
        return sheet.can_mark_pink(value)
    if die is Dice.GREEN:
        return sheet.can_mark_green_die(value)
    if die is Dice.BLUE:
        return sheet.can_mark_blue(_blue_entry(state))
    if die is Dice.WHITE:
        return _any_white_mark_legal(state)
    if die is Dice.SILVER:
        return sheet.can_use_silver_value(value)
    return False


def _legal_active_pick(state: GameState, die: Dice) -> bool:
    return _die_has_legal_mark(state, die)


def _can_forfeit(state: GameState) -> bool:

    return state.picks_made < 3 and not any(_legal_active_pick(state, die) for die in state.hand)


def _resume_phase_for(resume_after: str) -> Phase:
    if resume_after == "finish_passive":
        return Phase.PASSIVE_PICK
    if resume_after == "plus_one_continue":
        return Phase.PLUS_ONE
    return Phase.ACTIVE_PICK


def _after_white_mark_resume(state: GameState, resume_after: str) -> None:
    if resume_after == "after_active_pick":
        _after_active_pick(state)
    elif resume_after == "finish_passive":
        _finish_passive_turn(state)
    elif resume_after == "plus_one_continue":
        _continue_plus_one_if_available(state)
    else:
        raise ValueError(f"unsupported white mark resume {resume_after!r}")


def _clear_white_mark_pending(state: GameState) -> None:
    state.pending_die = None
    state.pending_value = None
    state.pending_platter_sent = []
    state.white_mark_resume_after = "after_active_pick"


def _start_white_mode_choice(
    state: GameState,
    *,
    resume_after: str,
    platter_sent: list[Dice] | None = None,
) -> None:
    state.pending_die = Dice.WHITE
    state.pending_value = state.faces[Dice.WHITE]
    state.pending_platter_sent = list(platter_sent or [])
    state.white_mark_resume_after = resume_after
    # White mode choice tracks its own resume; drop any deferred bonus resume
    # so apply_action's trailing _process_bonus_resume cannot end the turn early.
    state.bonus_resume_after = None
    state.phase = Phase.ACTIVE_MARK_WHITE


def _apply_single_white_mode(state: GameState, mode: WhiteMode, *, resume_after: str) -> None:
    if mode == "silver":
        if resume_after == "finish_passive":
            start_silver_resolution(
                state,
                primary_value=state.faces[Dice.WHITE],
                cascade_marks=[],
                finish="passive",
            )
            return
        if resume_after == "plus_one_continue":
            start_silver_resolution(
                state,
                primary_value=state.faces[Dice.WHITE],
                cascade_marks=[],
                finish="plus_one",
            )
            return
        _start_white_as_silver(state, [])
        return

    if mode == "yellow":
        if resume_after == "finish_passive":
            _start_passive_yellow_mark(state, die_value=state.faces[Dice.WHITE])
            return
        if resume_after == "plus_one_continue":
            _start_plus_one_yellow_mark(state, die_value=state.faces[Dice.WHITE])
            return
        _begin_active_white_yellow_mark(state)
        return

    if not _mark_die_on_sheet(
        state,
        Dice.WHITE,
        white_mode=mode,
        resume_after=resume_after,
    ):
        _after_white_mark_resume(state, resume_after)


def _legal_yellow_mark_cells(sheet: PlayerSheet, value: int) -> list[int]:
    return [
        cell_id
        for cell_id in sheet.yellow_cell_ids_for_value(value)
        if sheet.can_mark_yellow(cell_id, value)
    ]


def _apply_yellow_mark_and_maybe_bonus(
    state: GameState,
    cell_id: int,
    value: int,
    *,
    resume_phase: Phase,
    resume_after: str,
) -> bool:
    result = state.sheet.mark_yellow(cell_id, value)
    enqueue_mark_bonuses(
        state,
        yellow_cell_id=cell_id,
        yellow_mark_result=result,
    )
    return try_enter_bonus_phase(state, resume_phase, resume_after=resume_after)


def _mark_die_on_sheet(
    state: GameState, die: Dice, *, white_mode: WhiteMode | None = None, resume_after: str
) -> bool:
    """Apply a die mark and maybe enter bonus phase. Returns True if bonuses pending."""

    value = state.faces[die]

    if die is Dice.YELLOW:
        legal_cells = _legal_yellow_mark_cells(state.sheet, value)

        if len(legal_cells) != 1:
            raise ValueError("yellow mark requires explicit cell choice")

        return _apply_yellow_mark_and_maybe_bonus(
            state,
            legal_cells[0],
            value,
            resume_phase=_resume_phase_for(resume_after),
            resume_after=resume_after,
        )

    if die in (Dice.BLUE, Dice.WHITE) and (die is Dice.BLUE or white_mode == "blue"):
        slot = state.sheet.mark_blue(_blue_entry(state))

        enqueue_mark_bonuses(state, blue_slot=slot)

        return try_enter_bonus_phase(
            state, _resume_phase_for(resume_after), resume_after=resume_after
        )

    if die is Dice.GREEN or (die is Dice.WHITE and white_mode == "green"):
        face = state.faces[die]

        slot = state.sheet.mark_green_die(face)

        enqueue_mark_bonuses(state, green_slot=slot)

        return try_enter_bonus_phase(
            state, _resume_phase_for(resume_after), resume_after=resume_after
        )

    if die is Dice.PINK or (die is Dice.WHITE and white_mode == "pink"):
        slot = state.sheet.mark_pink(value)

        enqueue_mark_bonuses(state, pink_slot=slot, pink_value=value)

        return try_enter_bonus_phase(
            state, _resume_phase_for(resume_after), resume_after=resume_after
        )

    if die is Dice.YELLOW or (die is Dice.WHITE and white_mode == "yellow"):
        legal_cells = _legal_yellow_mark_cells(state.sheet, value)

        if len(legal_cells) != 1:
            raise ValueError("yellow mark requires explicit cell choice")

        return _apply_yellow_mark_and_maybe_bonus(
            state,
            legal_cells[0],
            value,
            resume_phase=_resume_phase_for(resume_after),
            resume_after=resume_after,
        )

    raise ValueError(f"unsupported mark for die {die}")


def _silver_row_for_platter_die(die: Dice) -> Color | None:
    """Row fixed by die color on platter cascade; white/silver dice are jokers."""
    if die in (Dice.WHITE, Dice.SILVER):
        return None
    color = die.sheet_color
    if color is None or color is Color.SILVER:
        return None
    return color


def _cascade_marks_for_sent(state: GameState, sent: list[Dice]) -> list[tuple[int, Color | None]]:
    return [(state.faces[die], _silver_row_for_platter_die(die)) for die in sent]


def _start_active_silver_pick(
    state: GameState, *, primary_value: int, platter_sent: list[Dice]
) -> None:

    start_silver_resolution(
        state,
        primary_value=primary_value,
        cascade_marks=_cascade_marks_for_sent(state, platter_sent),
        finish="active",
    )


def _start_white_as_silver(state: GameState, platter_sent: list[Dice]) -> None:

    cascade_sent = list(platter_sent)

    # White used as silver sends the physical silver die to the platter — that
    # die *was* moved by this pick, so it is a cascade mark (joker row).
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

    if state.awaiting_roll:
        raise ValueError("must roll before picking a die")

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

            if mode == "yellow":
                _begin_active_white_yellow_mark(state)

                return

            if not _mark_die_on_sheet(
                state, die, white_mode=mode, resume_after="after_active_pick"
            ):
                _after_active_pick(state)

            return

        _start_white_mode_choice(
            state,
            resume_after="after_active_pick",
            platter_sent=platter_sent,
        )

        return

    _complete_pick(state, die, value)

    if not _mark_die_on_sheet(state, die, resume_after="after_active_pick"):
        _after_active_pick(state)


def _begin_active_white_yellow_mark(state: GameState) -> None:
    value = state.faces[Dice.WHITE]
    legal_cells = _legal_yellow_mark_cells(state.sheet, value)

    if not legal_cells:
        raise ValueError("no legal yellow cell for white")

    if len(legal_cells) == 1:
        if not _apply_yellow_mark_and_maybe_bonus(
            state,
            legal_cells[0],
            value,
            resume_phase=Phase.ACTIVE_PICK,
            resume_after="after_active_pick",
        ):
            _after_active_pick(state)

        return

    state.pending_die = Dice.WHITE
    state.pending_value = value
    state.phase = Phase.ACTIVE_MARK_YELLOW


def _mark_pending_yellow(state: GameState, cell_id: int) -> None:

    assert state.pending_value is not None

    value = state.pending_value
    state.pending_die = None
    state.pending_value = None

    if not _apply_yellow_mark_and_maybe_bonus(
        state,
        cell_id,
        value,
        resume_phase=Phase.ACTIVE_PICK,
        resume_after="after_active_pick",
    ):
        _after_active_pick(state)


def _mark_passive_yellow_cell(state: GameState, cell_id: int) -> None:
    assert state.pending_value is not None

    value = state.pending_value
    state.pending_die = None
    state.pending_value = None

    if not _apply_yellow_mark_and_maybe_bonus(
        state,
        cell_id,
        value,
        resume_phase=Phase.PASSIVE_PICK,
        resume_after="finish_passive",
    ):
        _finish_passive_turn(state)


def _mark_plus_one_yellow_cell(state: GameState, cell_id: int) -> None:
    assert state.pending_value is not None

    value = state.pending_value
    state.pending_die = None
    state.pending_value = None

    if not _apply_yellow_mark_and_maybe_bonus(
        state,
        cell_id,
        value,
        resume_phase=Phase.PLUS_ONE,
        resume_after="plus_one_continue",
    ):
        _continue_plus_one_if_available(state)


def _start_passive_yellow_mark(state: GameState, *, die_value: int | None = None) -> None:
    value = die_value if die_value is not None else state.faces[Dice.YELLOW]
    legal_cells = _legal_yellow_mark_cells(state.sheet, value)

    if not legal_cells:
        raise ValueError("no legal yellow cell")

    if len(legal_cells) == 1:
        if not _apply_yellow_mark_and_maybe_bonus(
            state,
            legal_cells[0],
            value,
            resume_phase=Phase.PASSIVE_PICK,
            resume_after="finish_passive",
        ):
            _finish_passive_turn(state)
        return

    state.pending_die = Dice.YELLOW
    state.pending_value = value
    state.phase = Phase.PASSIVE_MARK_YELLOW


def _start_plus_one_yellow_mark(state: GameState, *, die_value: int | None = None) -> None:
    value = die_value if die_value is not None else state.faces[Dice.YELLOW]
    legal_cells = _legal_yellow_mark_cells(state.sheet, value)

    if not legal_cells:
        raise ValueError("no legal yellow cell for plus-one")

    if len(legal_cells) == 1:
        if not _apply_yellow_mark_and_maybe_bonus(
            state,
            legal_cells[0],
            value,
            resume_phase=Phase.PLUS_ONE,
            resume_after="plus_one_continue",
        ):
            _continue_plus_one_if_available(state)
        return

    state.pending_die = Dice.YELLOW
    state.pending_value = value
    state.phase = Phase.PLUS_ONE_MARK_YELLOW


def _mark_pending_white(state: GameState, mode: WhiteMode) -> None:
    resume = state.white_mark_resume_after
    assert state.pending_die is Dice.WHITE

    if mode == "silver":
        if resume == "finish_passive":
            start_silver_resolution(
                state,
                primary_value=state.faces[Dice.WHITE],
                cascade_marks=[],
                finish="passive",
            )
        elif resume == "plus_one_continue":
            start_silver_resolution(
                state,
                primary_value=state.faces[Dice.WHITE],
                cascade_marks=[],
                finish="plus_one",
            )
        else:
            _start_white_as_silver(state, state.pending_platter_sent)
        _clear_white_mark_pending(state)
        return

    if mode == "yellow":
        state.pending_platter_sent = []
        if resume == "finish_passive":
            _start_passive_yellow_mark(state, die_value=state.faces[Dice.WHITE])
        elif resume == "plus_one_continue":
            _start_plus_one_yellow_mark(state, die_value=state.faces[Dice.WHITE])
        else:
            _begin_active_white_yellow_mark(state)
        return

    if not _mark_die_on_sheet(state, Dice.WHITE, white_mode=mode, resume_after=resume):
        _after_white_mark_resume(state, resume)

    _clear_white_mark_pending(state)


def _finish_silver_resolution(state: GameState) -> None:

    if state.silver_finish == "passive":
        _finish_passive_turn(state)

        return

    if state.silver_finish == "plus_one":
        if state.sheet.can_use_action(ActionTrack.PLUS_ONE):
            state.phase = Phase.PLUS_ONE

            return

        _end_plus_one_phase(state)

        return

    _after_active_pick(state)


def _complete_pick(
    state: GameState,
    die: Dice,
    value: int,
    *,
    defer_mark: bool = False,
) -> list[Dice]:
    """Place the chosen die and send leftovers to the platter.

    Returns dice that **moved onto the platter because of this pick**:
    strictly lower dice, plus on pick 3 any remaining hand dice (end of
    active turn). Dice already on the platter are not included.
    """

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

    if state.picks_made >= 3:
        _finish_active_turn(state)

        return

    if not state.hand:
        if can_choose_unlock_or_end_turn(state):
            state.awaiting_roll = True
            state.phase = Phase.ACTIVE_PICK
            return
        _finish_active_turn(state)
        return

    state.awaiting_roll = True

    state.phase = Phase.ACTIVE_PICK


def _forfeit_pick(state: GameState) -> None:

    if state.awaiting_roll:
        raise ValueError("cannot forfeit before rolling")

    state.picks_made += 1

    if state.picks_made >= 3 or not state.hand:
        _finish_active_turn(state)

        return

    state.awaiting_roll = True

    state.phase = Phase.ACTIVE_PICK


def _finish_active_turn(state: GameState) -> None:

    for other in list(state.hand):
        state.hand.remove(other)

        state.platter.append(other)

    state.hand = []

    _begin_plus_one_phase(state, after_passive=False)


def _begin_plus_one_phase(state: GameState, *, after_passive: bool) -> None:

    state.plus_one_dice_used = set()

    state.plus_one_after_passive = after_passive

    if not state.sheet.can_use_action(ActionTrack.PLUS_ONE):
        _end_plus_one_phase(state)

        return

    state.phase = Phase.PLUS_ONE


def _end_plus_one_phase(state: GameState) -> None:

    state.plus_one_dice_used = set()

    if state.plus_one_after_passive:
        advance_round_or_game_over(state)

        return

    begin_passive_turn(state)


def _apply_plus_one_pick(state: GameState, die: Dice) -> None:
    """Use one extra die at its current face (no re-roll)."""

    if die in state.plus_one_dice_used:
        raise ValueError("die already used in this plus-one chain")
    if not state.sheet.can_use_action(ActionTrack.PLUS_ONE):
        raise ValueError("no plus-one action available")
    if not _legal_passive_mark(state, die, from_pool=False):
        raise ValueError(f"plus-one {die.value} is not a legal mark at current faces")

    state.plus_one_dice_used.add(die)
    state.sheet.use_action(ActionTrack.PLUS_ONE)

    if die is Dice.SILVER:
        start_silver_resolution(
            state,
            primary_value=state.faces[die],
            cascade_marks=[],
            finish="plus_one",
        )

        return

    if die is Dice.WHITE:
        modes = _white_mark_modes(state)
        if not modes:
            raise ValueError("no legal plus-one mark for white")
        if len(modes) == 1:
            _apply_single_white_mode(state, modes[0], resume_after="plus_one_continue")
        else:
            _start_white_mode_choice(state, resume_after="plus_one_continue")
        return

    if die is Dice.YELLOW:
        _start_plus_one_yellow_mark(state)

        return

    if not _mark_die_on_sheet(state, die, resume_after="plus_one_continue"):
        _continue_plus_one_if_available(state)


def _continue_plus_one_if_available(state: GameState) -> None:

    if state.sheet.can_use_action(ActionTrack.PLUS_ONE):
        state.phase = Phase.PLUS_ONE

        return

    _end_plus_one_phase(state)


def begin_passive_turn(state: GameState) -> None:

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


def _plus_one_die_can_mark(state: GameState, die: Dice) -> bool:
    """True if this die can be marked using its current face (extra-die action)."""
    return _legal_passive_mark(state, die, from_pool=False)


def _legal_plus_one_action_ids(state: GameState) -> list[int]:
    actions: list[int] = [end_plus_one_id()]
    if not state.sheet.can_use_action(ActionTrack.PLUS_ONE):
        return actions
    for die in Dice:
        if die in state.plus_one_dice_used:
            continue
        if _plus_one_die_can_mark(state, die):
            actions.append(plus_one_pick_id(die))
    return actions


def _legal_passive_mark(state: GameState, die: Dice, *, from_pool: bool) -> bool:

    del from_pool
    return _die_has_legal_mark(state, die)


def _legal_passive_action_ids(state: GameState) -> list[int]:

    actions: list[int] = []

    for die in state.platter:
        if _legal_passive_mark(state, die, from_pool=False):
            actions.append(passive_platter_id(die))

    if state.use_pool_fallback:
        for die in state.passive_pool:
            if _legal_passive_mark(state, die, from_pool=True):
                actions.append(passive_pool_id(die))

    actions.append(passive_skip_id())
    return actions


def _apply_passive_skip(state: GameState) -> None:
    """Decline to take a platter/pool die; mark nothing."""
    if state.phase is not Phase.PASSIVE_PICK:
        raise ValueError(f"passive skip not legal in phase {state.phase}")

    _finish_passive_turn(state)


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
            cascade_marks=[],
            finish="passive",
        )

        return

    if die is Dice.WHITE:
        modes = _white_mark_modes(state)
        if not modes:
            raise ValueError("no legal passive mark for white")
        if len(modes) == 1:
            _apply_single_white_mode(state, modes[0], resume_after="finish_passive")
        else:
            _start_white_mode_choice(state, resume_after="finish_passive")
        return

    elif die is Dice.YELLOW:
        _start_passive_yellow_mark(state)

    else:
        _mark_die_on_sheet(state, die, resume_after="finish_passive")

    if state.phase is Phase.PASSIVE_PICK:
        _finish_passive_turn(state)


def _finish_passive_turn(state: GameState) -> None:

    _begin_plus_one_phase(state, after_passive=True)


def advance_round_or_game_over(state: GameState) -> None:

    rounds_total = get_score_sheet().rounds_by_player_count[state.player_count]

    if state.round_index >= rounds_total:
        if state.sheet.can_use_action(ActionTrack.PLUS_ONE):
            state.plus_one_after_passive = True
            state.plus_one_dice_used = set()
            state.phase = Phase.PLUS_ONE

            return

        state.phase = Phase.GAME_OVER

        return

    state.round_index += 1

    _begin_active_turn(state)


def play_random_game(seed: int, max_actions: int = 10_000) -> GameState:
    """Play random legal moves until terminal — useful for smoke tests."""

    state = new_game(seed)
    choice_rng = random.Random(seed ^ 0xBAD5_EED)

    for _ in range(max_actions):
        legal = legal_action_ids(state)

        if not legal:
            break

        action_id = choice_rng.choice(legal)

        apply_action(state, action_id, check_legal=False)

        if state.phase is Phase.GAME_OVER:
            break

    return state
