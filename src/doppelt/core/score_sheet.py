"""Load score sheet constants from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from doppelt.core.types import ActionTrack, BonusKind, Color

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCORE_SHEET_PATH = PACKAGE_ROOT / "data" / "score_sheet" / "score_sheet_v1.yaml"


@dataclass(frozen=True)
class Bonus:
    kind: BonusKind
    color: Color | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Bonus | None:
        if data is None:
            return None
        color = data.get("color")
        return cls(
            kind=BonusKind(data["kind"]),
            color=Color(color) if color else None,
        )


@dataclass(frozen=True)
class YellowCell:
    id: int
    value: int
    row: int
    col: int


@dataclass(frozen=True)
class FieldBonus:
    slot: int
    bonus: Bonus


@dataclass(frozen=True)
class ActionTrackDef:
    slots: int
    end_bonus: Bonus | None


@dataclass(frozen=True)
class SilverArea:
    rows: tuple[Color, ...]
    columns: tuple[int, ...]
    row_score_by_mark_count: tuple[int, ...]
    column_bonuses: tuple[Bonus | None, ...]


@dataclass(frozen=True)
class YellowArea:
    cells: tuple[YellowCell, ...]
    score_by_cross_count: tuple[int, ...]
    row_completion_bonuses: tuple[Bonus | None, ...]
    column_completion_bonuses: tuple[Bonus | None, ...]
    bottom_edge_bonuses: tuple[Bonus | None, ...]


@dataclass(frozen=True)
class BlueArea:
    slot_count: int
    score_by_last_slot: tuple[int, ...]
    field_bonuses: tuple[FieldBonus, ...]


@dataclass(frozen=True)
class GreenArea:
    slot_count: int
    multipliers: tuple[int, ...]
    pair_count: int
    field_bonuses: tuple[FieldBonus, ...]


@dataclass(frozen=True)
class PinkArea:
    slot_count: int
    min_values: tuple[int | None, ...]
    field_bonuses: tuple[FieldBonus, ...]


@dataclass(frozen=True)
class RoundStartGrant:
    """Grant at the beginning of a round; None means no grant (solo rounds 5–6)."""

    kind: BonusKind | None
    track: ActionTrack | None = None
    color: Color | None = None

    @classmethod
    def from_raw(cls, data: dict[str, Any] | None) -> RoundStartGrant | None:
        if data is None:
            return None
        track = data.get("track")
        color = data.get("color")
        return cls(
            kind=BonusKind(data["kind"]),
            track=ActionTrack(track) if track else None,
            color=Color(color) if color else None,
        )


@dataclass(frozen=True)
class ScoreSheet:
    version: int
    name: str
    rounds_by_player_count: dict[int, int]
    round_start_grants: tuple[RoundStartGrant | None, ...]
    round_track_decorations: dict[int, str]
    action_tracks: dict[ActionTrack, ActionTrackDef]
    silver: SilverArea
    yellow: YellowArea
    blue: BlueArea
    green: GreenArea
    pink: PinkArea


def _tuple_bonuses(items: list[Any]) -> tuple[Bonus | None, ...]:
    return tuple(Bonus.from_dict(item) for item in items)


def _parse_action_tracks(raw: dict[str, Any]) -> dict[ActionTrack, ActionTrackDef]:
    tracks: dict[ActionTrack, ActionTrackDef] = {}
    for key, value in raw.items():
        tracks[ActionTrack(key)] = ActionTrackDef(
            slots=value["slots"],
            end_bonus=Bonus.from_dict(value.get("end_bonus")),
        )
    return tracks


def _parse_field_bonuses(raw: list[dict[str, Any]]) -> tuple[FieldBonus, ...]:
    return tuple(
        FieldBonus(slot=item["slot"], bonus=Bonus.from_dict(item["bonus"])) for item in raw
    )


def load_score_sheet(path: Path | None = None) -> ScoreSheet:
    sheet_path = path or DEFAULT_SCORE_SHEET_PATH
    with sheet_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    yellow_raw = raw["yellow"]
    cells = tuple(
        YellowCell(id=c["id"], value=c["value"], row=c["row"], col=c["col"])
        for c in yellow_raw["cells"]
    )

    silver_raw = raw["silver"]
    silver = SilverArea(
        rows=tuple(Color(c) for c in silver_raw["rows"]),
        columns=tuple(silver_raw["columns"]),
        row_score_by_mark_count=tuple(silver_raw["row_score_by_mark_count"]),
        column_bonuses=_tuple_bonuses(silver_raw["column_bonuses"]),
    )

    yellow = YellowArea(
        cells=cells,
        score_by_cross_count=tuple(yellow_raw["score_by_cross_count"]),
        row_completion_bonuses=_tuple_bonuses(yellow_raw["row_completion_bonuses"]),
        column_completion_bonuses=_tuple_bonuses(yellow_raw["column_completion_bonuses"]),
        bottom_edge_bonuses=_tuple_bonuses(yellow_raw["bottom_edge_bonuses"]),
    )

    blue_raw = raw["blue"]
    blue = BlueArea(
        slot_count=blue_raw["slot_count"],
        score_by_last_slot=tuple(blue_raw["score_by_last_slot"]),
        field_bonuses=_parse_field_bonuses(blue_raw["field_bonuses"]),
    )

    green_raw = raw["green"]
    green = GreenArea(
        slot_count=green_raw["slot_count"],
        multipliers=tuple(green_raw["multipliers"]),
        pair_count=green_raw["pair_count"],
        field_bonuses=_parse_field_bonuses(green_raw["field_bonuses"]),
    )

    pink_raw = raw["pink"]
    pink = PinkArea(
        slot_count=pink_raw["slot_count"],
        min_values=tuple(pink_raw["min_values"]),
        field_bonuses=_parse_field_bonuses(pink_raw["field_bonuses"]),
    )

    grants = tuple(RoundStartGrant.from_raw(g) for g in raw["round_start_grants"])

    return ScoreSheet(
        version=raw["version"],
        name=raw["name"],
        rounds_by_player_count=dict(raw["rounds_by_player_count"]),
        round_start_grants=grants,
        round_track_decorations=dict(raw["round_track_decorations"]),
        action_tracks=_parse_action_tracks(raw["action_tracks"]),
        silver=silver,
        yellow=yellow,
        blue=blue,
        green=green,
        pink=pink,
    )


@lru_cache(maxsize=1)
def get_score_sheet() -> ScoreSheet:
    return load_score_sheet()
