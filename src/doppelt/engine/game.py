"""Solo game engine — Phase 1 (yellow, blue, pink; automated bonuses)."""

from __future__ import annotations

import random

from doppelt.actions.catalog_v1 import (
  FORFEIT_PICK_ID,
  ActionKind,
  decode_action,
  mark_yellow_id,
  passive_platter_id,
  passive_pool_id,
  pick_die_id,
)
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import get_score_sheet
from doppelt.core.scoring import score_sheet_areas
from doppelt.core.solo_passive import DieRoll, split_passive_roll
from doppelt.core.state import GameState
from doppelt.core.types import ALL_DICE, Dice, blue_entry_value
from doppelt.engine.bonuses import bonuses_after_blue_mark, bonuses_after_pink_mark

SUPPORTED_PICK_DICE = frozenset({Dice.YELLOW, Dice.BLUE, Dice.PINK, Dice.WHITE})


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
    _mark_pending_yellow(state, action.yellow_cell_id)
  elif action.kind is ActionKind.PASSIVE_PICK:
    assert action.die is not None
    _pick_passive_die(state, action.die, from_pool=action.passive_from_pool)
  else:
    raise ValueError(f"unsupported action {action}")

  return state


def _roll_hand(state: GameState) -> None:
  for die in state.hand:
    state.faces[die] = state.rng.randint(1, 6)


def _begin_active_turn(state: GameState) -> None:
  state.phase = Phase.ACTIVE_PICK
  state.hand = list(ALL_DICE)
  state.platter = []
  state.passive_pool = []
  state.slots = [None, None, None]
  state.picks_made = 0
  state.pending_die = None
  state.pending_value = None
  _roll_hand(state)


def _blue_entry(state: GameState) -> int:
  return blue_entry_value(
    blue_face=state.faces[Dice.BLUE],
    white_face=state.faces[Dice.WHITE],
    includes_white=True,
  )


def _legal_active_pick(state: GameState, die: Dice) -> bool:
  if die is Dice.YELLOW:
    value = state.faces[die]
    return any(state.sheet.can_mark_yellow(cell_id, value) for cell_id in state.sheet.yellow_cell_ids_for_value(value))
  if die is Dice.PINK:
    return state.sheet.can_mark_pink(state.faces[die])
  if die in (Dice.BLUE, Dice.WHITE):
    return state.sheet.can_mark_blue(_blue_entry(state))
  return False


def _can_forfeit(state: GameState) -> bool:
  return state.picks_made < 3 and not any(_legal_active_pick(state, die) for die in state.hand)


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
      state.sheet.mark_yellow(legal_cells[0], value)
      _after_active_pick(state)
      return
    _complete_pick(state, die, value, defer_mark=True)
    state.pending_die = die
    state.pending_value = value
    state.phase = Phase.ACTIVE_MARK_YELLOW
    return

  _complete_pick(state, die, value)
  if die in (Dice.BLUE, Dice.WHITE):
    slot = state.sheet.mark_blue(_blue_entry(state))
    events = bonuses_after_blue_mark(state.sheet, slot)
    state.bonus_events.extend(f"{event.source}:{event.mark.area}{event.mark.slot}={event.mark.value}" for event in events)
  elif die is Dice.PINK:
    slot = state.sheet.mark_pink(value)
    events = bonuses_after_pink_mark(state.sheet, slot, value)
    state.bonus_events.extend(f"{event.source}:{event.mark.area}{event.mark.slot}={event.mark.value}" for event in events)
  _after_active_pick(state)


def _mark_pending_yellow(state: GameState, cell_id: int) -> None:
  assert state.pending_value is not None
  state.sheet.mark_yellow(cell_id, state.pending_value)
  state.pending_die = None
  state.pending_value = None
  _after_active_pick(state)


def _complete_pick(
  state: GameState,
  die: Dice,
  value: int,
  *,
  defer_mark: bool = False,
) -> None:
  del defer_mark  # platter rules always apply; mark may follow in another phase
  state.hand.remove(die)
  state.slots[state.picks_made] = die
  state.picks_made += 1

  for other in list(state.hand):
    if state.faces[other] < value:
      state.hand.remove(other)
      state.platter.append(other)

  if state.picks_made >= 3:
    for other in list(state.hand):
      state.platter.append(other)
    state.hand = []


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
  rolls = [DieRoll(die=die, value=state.rng.randint(1, 6)) for die in ALL_DICE]
  for roll in rolls:
    state.faces[roll.die] = roll.value
  split = split_passive_roll(rolls, state.rng)
  state.platter = [roll.die for roll in split.platter]
  state.passive_pool = [roll.die for roll in split.pool]
  state.use_pool_fallback = not any(_legal_passive_mark(state, die, from_pool=False) for die in state.platter)


def _legal_passive_mark(state: GameState, die: Dice, *, from_pool: bool) -> bool:
  value = state.faces[die]
  if die is Dice.YELLOW:
    return any(state.sheet.can_mark_yellow(cell_id, value) for cell_id in state.sheet.yellow_cell_ids_for_value(value))
  if die is Dice.PINK:
    return state.sheet.can_mark_pink(value)
  if die in (Dice.BLUE, Dice.WHITE):
    return state.sheet.can_mark_blue(_blue_entry(state))
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

  value = state.faces[die]
  if die is Dice.YELLOW:
    legal_cells = [
      cell_id
      for cell_id in state.sheet.yellow_cell_ids_for_value(value)
      if state.sheet.can_mark_yellow(cell_id, value)
    ]
    if not legal_cells:
      raise ValueError("no legal yellow cell")
    state.sheet.mark_yellow(legal_cells[0], value)
  elif die in (Dice.BLUE, Dice.WHITE):
    slot = state.sheet.mark_blue(_blue_entry(state))
    events = bonuses_after_blue_mark(state.sheet, slot)
    state.bonus_events.extend(f"{event.source}:{event.mark.area}{event.mark.slot}={event.mark.value}" for event in events)
  elif die is Dice.PINK:
    slot = state.sheet.mark_pink(value)
    events = bonuses_after_pink_mark(state.sheet, slot, value)
    state.bonus_events.extend(f"{event.source}:{event.mark.area}{event.mark.slot}={event.mark.value}" for event in events)
  else:
    raise ValueError(f"unsupported passive die {die}")

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
