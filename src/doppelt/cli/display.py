"""CLI formatting helpers."""

from __future__ import annotations

from doppelt.actions.catalog_v1 import describe_action
from doppelt.core.scoring import score_sheet_areas, total_score
from doppelt.core.state import GameState
from doppelt.core.types import Dice


def format_dice_faces(state: GameState) -> str:
    parts = [f"{die.value}={state.faces.get(die, '?')}" for die in Dice]
    return ", ".join(parts)


def format_status(state: GameState) -> str:
    lines = [
        f"Round {state.round_index}  Phase: {state.phase.value}",
        f"Dice: {format_dice_faces(state)}",
    ]
    if state.hand:
        hand = ", ".join(die.value for die in state.hand)
        lines.append(f"Hand: {hand}")
    if state.platter:
        platter = ", ".join(die.value for die in state.platter)
        lines.append(f"Platter: {platter}")
    if state.passive_pool:
        pool = ", ".join(die.value for die in state.passive_pool)
        lines.append(f"Passive pool: {pool}")
    if state.pending_bonuses:
        lines.append(f"Pending bonuses: {len(state.pending_bonuses)}")
    lines.append(f"Foxes collected: {state.sheet.foxes}")
    return "\n".join(lines)


def format_scores(state: GameState) -> str:
    areas = score_sheet_areas(state.sheet)
    lines = [
        "Final scores:",
        f"  yellow: {areas['yellow']}",
        f"  blue:   {areas['blue']}",
        f"  pink:   {areas['pink']}",
        f"  green:  {areas['green']}",
        f"  silver: {areas['silver']}",
        f"  foxes:  {areas['foxes']}  ({state.sheet.foxes} fox(es))",
        f"  total:  {total_score(state.sheet)}",
    ]
    return "\n".join(lines)


def format_action_menu(action_ids: list[int]) -> str:
    lines = ["Legal actions:"]
    for index, action_id in enumerate(action_ids):
        lines.append(f"  [{index}] id={action_id}: {describe_action(action_id)}")
    lines.append("Enter list index, action id, or 'q' to quit.")
    return "\n".join(lines)
