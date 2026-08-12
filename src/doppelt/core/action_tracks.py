"""Action track slot state (reroll / unlock / plus one)."""

from __future__ import annotations

from dataclasses import dataclass

from doppelt.core.score_sheet import get_score_sheet
from doppelt.core.types import ActionTrack, BonusKind

ACTION_BONUS_TO_TRACK: dict[BonusKind, ActionTrack] = {
    BonusKind.REROLL: ActionTrack.REROLL,
    BonusKind.UNLOCK: ActionTrack.UNLOCK,
    BonusKind.PLUS_ONE: ActionTrack.PLUS_ONE,
}


@dataclass
class ActionTrackSlots:
    circled: int = 0
    crossed: int = 0


def track_capacity(track: ActionTrack) -> int:
    return get_score_sheet().action_tracks[track].slots


def empty_action_tracks() -> dict[ActionTrack, ActionTrackSlots]:
    return {track: ActionTrackSlots() for track in ActionTrack}


def can_circle_track(slots: ActionTrackSlots, track: ActionTrack) -> bool:
    return slots.circled + slots.crossed < track_capacity(track)


def circle_track(slots: ActionTrackSlots, track: ActionTrack) -> bool:
    if not can_circle_track(slots, track):
        return False
    slots.circled += 1
    return True


def can_use_track(slots: ActionTrackSlots) -> bool:
    return slots.circled > 0


def use_track(slots: ActionTrackSlots) -> None:
    if slots.circled <= 0:
        raise ValueError("no circled action available on track")
    slots.circled -= 1
    slots.crossed += 1
