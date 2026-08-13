"""CLI formatting helpers."""

from __future__ import annotations

from doppelt.actions.catalog_v1 import ActionKind, decode_action, describe_action
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.scoring import score_sheet_areas, total_score
from doppelt.core.silver import SILVER_ROW_COLORS
from doppelt.core.state import GameState
from doppelt.core.types import Dice


def format_dice_faces(state: GameState) -> str:
    parts = [f"{die.value}={state.faces.get(die, '?')}" for die in Dice]
    return ", ".join(parts)


def format_hand_dice(state: GameState) -> str:
    """Hand dice with current face values, lowest face first."""
    ordered = sorted(state.hand, key=lambda die: (state.faces.get(die, 0), die.value))
    return ", ".join(f"{die.value}={state.faces.get(die, '?')}" for die in ordered)


def format_silver_grid(sheet: PlayerSheet) -> str:
    """Exact silver cells marked — useful for verifying column bonuses."""
    lines = ["Silver grid (X = marked):", "         1 2 3 4 5 6"]
    for row in SILVER_ROW_COLORS:
        cells = ["X" if value in sheet.silver[row] else "." for value in range(1, 7)]
        lines.append(f"  {row.value:<6} {' '.join(cells)}")
    marked_parts: list[str] = []
    for row in SILVER_ROW_COLORS:
        values = sorted(sheet.silver[row])
        if values:
            marked_parts.append(f"{row.value}={','.join(str(v) for v in values)}")
    lines.append(
        "  marked: " + ("; ".join(marked_parts) if marked_parts else "(none)")
    )
    return "\n".join(lines)


def _sheet_has_silver_marks(sheet: PlayerSheet) -> bool:
    return any(sheet.silver[row] for row in SILVER_ROW_COLORS)


def format_silver_pending(state: GameState) -> str | None:
    """Explain primary + cascade marks while resolving a silver pick."""
    if not state.pending_silver_values:
        return None
    lines = ["Silver marks from this pick (only dice just moved to platter):"]
    for index, (value, row) in enumerate(
        zip(state.pending_silver_values, state.pending_silver_rows, strict=True)
    ):
        if index == 0:
            lines.append(f"  primary {value} (any row)")
        elif row is None:
            lines.append(f"  platter die {value} (joker / any row)")
        else:
            lines.append(f"  platter {row.value} die {value} ({row.value} row only)")
    return "\n".join(lines)


def format_status(state: GameState) -> str:
    lines = [
        f"Round {state.round_index}  Phase: {state.phase.value}",
    ]
    active_pick = _active_pick_number(state)
    if active_pick is not None:
        lines.append(f"Active pick: {active_pick} of 3")
    lines.append(f"Dice: {format_dice_faces(state)}")
    if state.hand:
        lines.append(f"Hand: {format_hand_dice(state)}")
    if state.platter:
        platter = ", ".join(die.value for die in state.platter)
        lines.append(f"Platter: {platter}")
    if state.passive_pool:
        pool = ", ".join(die.value for die in state.passive_pool)
        lines.append(f"Passive pool: {pool}")
    if state.pending_bonuses:
        lines.append(f"Pending bonuses: {len(state.pending_bonuses)}")
    lines.append(f"Foxes collected: {state.sheet.foxes}")
    pending_silver = format_silver_pending(state)
    if pending_silver:
        lines.append(pending_silver)
    if _sheet_has_silver_marks(state.sheet) or state.pending_silver_values:
        lines.append(format_silver_grid(state.sheet))
    return "\n".join(lines)


def _active_pick_number(state: GameState) -> int | None:
    """1–3 while the active player is rolling, picking, or resolving a mark."""
    from doppelt.core.phases import Phase

    if state.phase is Phase.ACTIVE_PICK:
        if state.picks_made >= 3:
            return None
        return state.picks_made + 1
    if state.phase in {
        Phase.ACTIVE_MARK_YELLOW,
        Phase.ACTIVE_MARK_WHITE,
        Phase.ACTIVE_MARK_SILVER,
    }:
        return max(1, min(state.picks_made, 3))
    if state.phase is Phase.RESOLVE_BONUS and state.resume_phase in {
        Phase.ACTIVE_PICK,
        Phase.ACTIVE_MARK_YELLOW,
        Phase.ACTIVE_MARK_WHITE,
        Phase.ACTIVE_MARK_SILVER,
    }:
        # Bonus mid-pick: picks_made already counts the die just taken.
        if state.picks_made <= 0:
            return 1
        return min(state.picks_made, 3)
    return None



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
    if _sheet_has_silver_marks(state.sheet):
        lines.append(format_silver_grid(state.sheet))
    return "\n".join(lines)


def sort_actions_for_display(state: GameState, action_ids: list[int]) -> list[int]:
    """Sort active pick-die actions by face value ascending; other actions keep relative order."""

    def sort_key(action_id: int) -> tuple[int, int, int]:
        action = decode_action(action_id)
        if action.kind is ActionKind.PICK_DIE and action.die is not None:
            face = state.faces.get(action.die, 99)
            return (0, face, action_id)
        return (1, 0, action_id)

    return sorted(action_ids, key=sort_key)


def format_action_menu(action_ids: list[int], state: GameState | None = None) -> str:
    lines = ["Legal actions:"]
    for index, action_id in enumerate(action_ids):
        label = describe_action(action_id)
        if state is not None:
            action = decode_action(action_id)
            if action.kind is ActionKind.PICK_DIE and action.die is not None:
                face = state.faces.get(action.die)
                if face is not None:
                    label = f"pick {action.die.value} die ({face})"
            elif action.kind is ActionKind.PLUS_ONE_PICK and action.die is not None:
                face = state.faces.get(action.die)
                if face is not None:
                    if action.die is Dice.BLUE:
                        white = state.faces.get(Dice.WHITE, "?")
                        entry = face + white if isinstance(white, int) else "?"
                        label = (
                            f"extra die: blue={face} + white={white} → blue track {entry}"
                        )
                    else:
                        label = f"extra die: use {action.die.value}={face}"
        lines.append(f"  [{index}] id={action_id}: {label}")
    lines.append("Enter list index, action id, or 'q' to quit.")
    return "\n".join(lines)
