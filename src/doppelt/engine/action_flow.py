"""Action track usage — reroll, unlock, plus one."""

from __future__ import annotations

from doppelt.actions.catalog_v1 import (
    end_active_turn_id,
    end_plus_one_id,
    plus_one_pick_id,
    roll_hand_id,
    unlock_platter_id,
    use_reroll_id,
)
from doppelt.core.action_tracks import ACTION_BONUS_TO_TRACK, track_capacity
from doppelt.core.score_sheet import Bonus, get_score_sheet
from doppelt.core.state import GameState
from doppelt.core.types import ActionTrack, BonusKind, Dice
from doppelt.engine.bonus_queue import enqueue_bonus, enqueue_bonus_once


def _action_track_is_full(state: GameState, track: ActionTrack) -> bool:
    slots = state.sheet.action_tracks[track]
    return slots.circled + slots.crossed >= track_capacity(track)


def circle_action_track(state: GameState, track: ActionTrack) -> bool:
    """Circle one slot on an action track; enqueue end bonus if the track fills."""
    if not state.sheet.circle_action(track):
        return False
    if _action_track_is_full(state, track):
        end_bonus = get_score_sheet().action_tracks[track].end_bonus
        if end_bonus is not None:
            enqueue_bonus_once(state, end_bonus, f"action:{track.value}:end")
    return True


def apply_action_bonus_circle(state: GameState, bonus: BonusKind) -> None:
    track = ACTION_BONUS_TO_TRACK.get(bonus)
    if track is None:
        return
    circle_action_track(state, track)


def apply_round_start_grants(state: GameState) -> None:
    grants = get_score_sheet().round_start_grants
    index = state.round_index - 1
    if index < 0 or index >= len(grants):
        return
    grant = grants[index]
    if grant is None:
        return
    if grant.kind is BonusKind.ACTION_TRACK_UNLOCK:
        if grant.track is not None:
            circle_action_track(state, grant.track)
        return
    if grant.kind is BonusKind.BONUS_WILD and grant.color is None:
        enqueue_bonus(
            state,
            Bonus(BonusKind.BONUS_WILD, None),
            f"round:{state.round_index}:wild",
        )


def can_choose_unlock_or_end_turn(state: GameState) -> bool:
    """Empty hand with an unused unlock and dice on the platter — player chooses next."""
    return (
        state.picks_made < 3
        and not state.hand
        and state.sheet.can_use_action(ActionTrack.UNLOCK)
        and bool(state.platter)
    )


def legal_active_pre_roll_action_ids(state: GameState) -> list[int]:
    actions: list[int] = []
    if state.sheet.can_use_action(ActionTrack.UNLOCK):
        for die in state.platter:
            actions.append(unlock_platter_id(die))
    if state.hand:
        actions.append(roll_hand_id())
    if can_choose_unlock_or_end_turn(state):
        actions.append(end_active_turn_id())
    return actions


def legal_active_post_roll_action_ids(state: GameState) -> list[int]:
    if state.hand and state.sheet.can_use_action(ActionTrack.REROLL):
        return [use_reroll_id()]
    return []


def apply_roll_hand(state: GameState) -> None:
    if not state.awaiting_roll:
        raise ValueError("roll is only legal before picking")
    if not state.hand:
        raise ValueError("no dice in hand to roll")
    for die in state.hand:
        state.faces[die] = state.rng.randint(1, 6)
    state.awaiting_roll = False


def apply_use_reroll(state: GameState) -> None:
    if state.awaiting_roll:
        raise ValueError("reroll is only legal after rolling")
    if not state.hand:
        raise ValueError("no dice in hand to reroll")
    state.sheet.use_action(ActionTrack.REROLL)
    for die in state.hand:
        state.faces[die] = state.rng.randint(1, 6)


def apply_unlock_platter(state: GameState, die: Dice) -> None:
    if not state.awaiting_roll:
        raise ValueError("unlock is only legal before rolling")
    if die not in state.platter:
        raise ValueError("die not on platter")
    state.sheet.use_action(ActionTrack.UNLOCK)
    state.platter.remove(die)
    state.hand.append(die)


def legal_plus_one_action_ids(state: GameState) -> list[int]:
    actions: list[int] = [end_plus_one_id()]
    if not state.sheet.can_use_action(ActionTrack.PLUS_ONE):
        return actions
    for die in Dice:
        if die in state.plus_one_dice_used:
            continue
        actions.append(plus_one_pick_id(die))
    return actions


def apply_end_plus_one(state: GameState) -> None:
    state.plus_one_dice_used = set()


def apply_plus_one_pick_start(state: GameState, die: Dice) -> None:
    if die in state.plus_one_dice_used:
        raise ValueError("die already used in this plus-one chain")
    if not state.sheet.can_use_action(ActionTrack.PLUS_ONE):
        raise ValueError("no plus-one action available")
    state.faces[die] = state.rng.randint(1, 6)
    state.plus_one_dice_used.add(die)
    state.sheet.use_action(ActionTrack.PLUS_ONE)
