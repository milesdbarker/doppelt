"""Score-sheet overlay marks for replay visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import PACKAGE_ROOT
from doppelt.core.scoring import score_sheet_areas, total_score
from doppelt.core.silver import SILVER_ROW_COLORS
from doppelt.core.state import GameState
from doppelt.core.phases import Phase
from doppelt.core.types import ALL_DICE, ActionTrack, Dice

_PASSIVE_PICKED_PHASES = frozenset(
    {
        Phase.PASSIVE_PICK,
        Phase.PASSIVE_MARK_YELLOW,
    }
)

LAYOUT_PATH = PACKAGE_ROOT / "data" / "score_sheet" / "board_layout_v1.json"
BOARD_IMAGE_PATH = PACKAGE_ROOT / "data" / "score_sheet" / "board.png"


@dataclass(frozen=True)
class OverlayMark:
    kind: str  # "circle" | "cross" | "text"
    x: float
    y: float
    radius: float = 16
    text: str | None = None
    highlight: bool = False
    key: str = ""


@dataclass(frozen=True)
class ShownDie:
    color: str
    face: int | None
    highlight: bool = False
    key: str = ""
    badge: str | None = None


@dataclass(frozen=True)
class DicePanelState:
    rolled: tuple[ShownDie, ...]
    platter: tuple[ShownDie, ...]
    picked: tuple[ShownDie, ...] = ()


@lru_cache(maxsize=1)
def load_board_layout() -> dict[str, Any]:
    with LAYOUT_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def board_image_path() -> Path:
    return BOARD_IMAGE_PATH


def _circle(x: float, y: float, radius: float, *, key: str, highlight: bool) -> OverlayMark:
    return OverlayMark("circle", x, y, radius, key=key, highlight=highlight)


def _cross(x: float, y: float, radius: float, *, key: str, highlight: bool) -> OverlayMark:
    return OverlayMark("cross", x, y, radius, key=key, highlight=highlight)


def _text(
    x: float, y: float, text: str, radius: float, *, key: str, highlight: bool
) -> OverlayMark:
    return OverlayMark("text", x, y, radius, text=text, key=key, highlight=highlight)


def sheet_highlight_keys(previous: PlayerSheet | None, current: PlayerSheet) -> set[str]:
    if previous is None:
        return set()
    keys: set[str] = set()
    for index, (before, after) in enumerate(zip(previous.yellow, current.yellow, strict=True)):
        if before.circled != after.circled or before.crossed != after.crossed:
            keys.add(f"yellow:{index}")
    for row in SILVER_ROW_COLORS:
        added = current.silver[row] - previous.silver[row]
        for value in added:
            keys.add(f"silver:{row.value}:{value}")
    for index, (before, after) in enumerate(zip(previous.blue, current.blue, strict=True)):
        if before != after:
            keys.add(f"blue:{index}")
    for index, (before, after) in enumerate(zip(previous.green, current.green, strict=True)):
        if before != after:
            keys.add(f"green:{index}")
    for index, (before, after) in enumerate(zip(previous.pink, current.pink, strict=True)):
        if before != after:
            keys.add(f"pink:{index}")
    for track in ActionTrack:
        before = previous.action_tracks[track]
        after = current.action_tracks[track]
        if before.circled != after.circled or before.crossed != after.crossed:
            used = after.crossed
            available = after.circled
            start = min(before.crossed, used)
            end = used + available
            for slot in range(start, end):
                keys.add(f"track:{track.value}:{slot}")
    keys.update(current.claimed_bonuses - previous.claimed_bonuses)
    if previous.foxes != current.foxes:
        keys.add("score:foxes")
    return keys


def marks_from_sheet(
    sheet: PlayerSheet,
    *,
    layout: dict[str, Any] | None = None,
    highlights: set[str] | None = None,
    round_index: int | None = None,
    game_over: bool = False,
    include_scores: bool = True,
) -> list[OverlayMark]:
    layout = layout or load_board_layout()
    hot = highlights or set()
    marks: list[OverlayMark] = []

    rounds = layout["rounds"]
    for index, x in enumerate(rounds["xs"]):
        round_no = index + 1
        key = f"round:{round_no}"
        if game_over or (round_index is not None and round_no < round_index):
            marks.append(_cross(x, rounds["y"], rounds["radius"], key=key, highlight=key in hot))
        elif round_index is not None and round_no == round_index:
            marks.append(_circle(x, rounds["y"], rounds["radius"], key=key, highlight=key in hot))

    for track in ActionTrack:
        spec = layout["action_tracks"][track.value]
        slots = sheet.action_tracks[track]
        for slot, x in enumerate(spec["xs"]):
            key = f"track:{track.value}:{slot}"
            hl = key in hot
            if slot < slots.crossed:
                marks.append(_cross(x, spec["y"], spec["radius"], key=key, highlight=hl))
            elif slot < slots.crossed + slots.circled:
                marks.append(_circle(x, spec["y"], spec["radius"], key=key, highlight=hl))

    silver = layout["silver"]
    for row_index, row in enumerate(SILVER_ROW_COLORS):
        y = silver["ys"][row_index]
        for value in sheet.silver[row]:
            x = silver["xs"][value - 1]
            key = f"silver:{row.value}:{value}"
            marks.append(_cross(x, y, silver["radius"], key=key, highlight=key in hot))

    radius = layout["yellow_radius"]
    for cell in layout["yellow_cells"]:
        state = sheet.yellow[cell["id"]]
        key = f"yellow:{cell['id']}"
        hl = key in hot
        if state.crossed:
            marks.append(_circle(cell["x"], cell["y"], radius, key=key, highlight=hl))
            marks.append(_cross(cell["x"], cell["y"], radius, key=key, highlight=hl))
        elif state.circled:
            marks.append(_circle(cell["x"], cell["y"], radius, key=key, highlight=hl))

    for area_name in ("blue", "green", "pink"):
        spec = layout[area_name]
        values = getattr(sheet, area_name)
        for index, value in enumerate(values):
            if value is None:
                continue
            key = f"{area_name}:{index}"
            marks.append(
                _text(
                    spec["xs"][index],
                    spec["y"],
                    str(value),
                    spec["radius"],
                    key=key,
                    highlight=key in hot,
                )
            )

    bonus_r = layout["bonus_radius"]
    for source, (x, y) in layout["claimed_bonuses"].items():
        if source in sheet.claimed_bonuses:
            marks.append(_cross(x, y, bonus_r, key=source, highlight=source in hot))

    if include_scores:
        areas = score_sheet_areas(sheet)
        areas["total"] = total_score(sheet)
        for name, (x, y) in layout["scores"].items():
            key = f"score:{name}"
            marks.append(_text(x, y, str(areas[name]), 22, key=key, highlight=key in hot))

    return marks


def marks_from_state(
    state: GameState,
    *,
    previous_sheet: PlayerSheet | None = None,
    layout: dict[str, Any] | None = None,
) -> list[OverlayMark]:
    from doppelt.core.phases import Phase

    highlights = sheet_highlight_keys(previous_sheet, state.sheet)
    return marks_from_sheet(
        state.sheet,
        layout=layout,
        highlights=highlights,
        round_index=state.round_index,
        game_over=state.phase is Phase.GAME_OVER,
    )


def _shown_group(
    dice: list[Dice],
    faces: dict[Dice, int],
    *,
    group: str,
    show_faces: bool,
    previous: set[Dice] | None,
) -> tuple[ShownDie, ...]:
    shown: list[ShownDie] = []
    present = {die for die in dice}
    for die in ALL_DICE:
        if die not in present:
            continue
        face = faces.get(die) if show_faces else None
        key = f"{group}:{die.value}"
        highlight = previous is not None and die not in previous
        shown.append(ShownDie(die.value, face, highlight=highlight, key=key))
    return tuple(shown)


def _use_passive_picked(
    *,
    phase: Phase | None,
    resume_phase: Phase | None,
    plus_one_after_passive: bool,
    bonus_resume_after: str | None,
) -> bool:
    """True only while resolving the passive turn (not leftover flags on a later active turn)."""
    if phase in _PASSIVE_PICKED_PHASES or resume_phase in _PASSIVE_PICKED_PHASES:
        return True
    if bonus_resume_after == "finish_passive":
        return True
    if plus_one_after_passive and phase in {
        Phase.PLUS_ONE,
        Phase.PLUS_ONE_MARK_YELLOW,
        Phase.RESOLVE_BONUS,
        Phase.GAME_OVER,
    }:
        return True
    return False


def _picked_slots(
    slots: list[Dice | None],
    faces: dict[Dice, int],
    previous_slots: list[Dice | None] | None,
) -> tuple[ShownDie, ...]:
    prev = {die for die in (previous_slots or []) if die is not None}
    track_prev = previous_slots is not None
    shown: list[ShownDie] = []
    padded = list(slots) + [None] * max(0, 3 - len(slots))
    for index, die in enumerate(padded[:3]):
        badge = str(index + 1)
        key = f"picked:slot:{index}"
        if die is None:
            shown.append(ShownDie("empty", None, key=key, badge=badge))
            continue
        highlight = track_prev and die not in prev
        shown.append(
            ShownDie(die.value, faces.get(die), highlight=highlight, key=key, badge=badge)
        )
    return tuple(shown)


def dice_panel_from_parts(
    *,
    hand: list[Dice],
    platter: list[Dice],
    faces: dict[Dice, int],
    awaiting_roll: bool,
    slots: list[Dice | None] | None = None,
    passive_pool: list[Dice] | None = None,
    phase: Phase | None = None,
    resume_phase: Phase | None = None,
    plus_one_after_passive: bool = False,
    bonus_resume_after: str | None = None,
    previous_hand: list[Dice] | None = None,
    previous_platter: list[Dice] | None = None,
    previous_slots: list[Dice | None] | None = None,
    previous_pool: list[Dice] | None = None,
) -> DicePanelState:
    """Hand = rolled; slots/pool = picked; platter = silver platter."""
    show_hand_faces = bool(hand) and not awaiting_roll
    prev_hand = set(previous_hand) if previous_hand is not None else None
    prev_platter = set(previous_platter) if previous_platter is not None else None
    pool = list(passive_pool or [])
    slot_list = list(slots) if slots is not None else [None, None, None]
    if _use_passive_picked(
        phase=phase,
        resume_phase=resume_phase,
        plus_one_after_passive=plus_one_after_passive,
        bonus_resume_after=bonus_resume_after,
    ):
        picked = _shown_group(
            pool,
            faces,
            group="picked",
            show_faces=True,
            previous=set(previous_pool) if previous_pool is not None else None,
        )
        rolled: tuple[ShownDie, ...] = ()
    else:
        picked = _picked_slots(slot_list, faces, previous_slots)
        rolled = _shown_group(
            hand,
            faces,
            group="rolled",
            show_faces=show_hand_faces,
            previous=prev_hand,
        )
    return DicePanelState(
        rolled=rolled,
        platter=_shown_group(
            platter,
            faces,
            group="platter",
            show_faces=True,
            previous=prev_platter,
        ),
        picked=picked,
    )


def dice_panel_from_state(
    state: GameState,
    *,
    previous_hand: list[Dice] | None = None,
    previous_platter: list[Dice] | None = None,
    previous_slots: list[Dice | None] | None = None,
    previous_pool: list[Dice] | None = None,
) -> DicePanelState:
    return dice_panel_from_parts(
        hand=state.hand,
        platter=state.platter,
        faces=state.faces,
        awaiting_roll=state.awaiting_roll,
        slots=state.slots,
        passive_pool=state.passive_pool,
        phase=state.phase,
        resume_phase=state.resume_phase,
        plus_one_after_passive=state.plus_one_after_passive,
        bonus_resume_after=state.bonus_resume_after,
        previous_hand=previous_hand,
        previous_platter=previous_platter,
        previous_slots=previous_slots,
        previous_pool=previous_pool,
    )
