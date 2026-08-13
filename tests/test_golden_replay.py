"""Golden replay: human solo game (seed + action IDs → terminal sheet/score)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from doppelt.actions.catalog_v1 import CATALOG_VERSION
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.scoring import score_sheet_areas, total_score
from doppelt.engine.game import is_terminal
from doppelt.replay import decode_log, replay_game

GOLDEN_LOG = Path(__file__).parent / "fixtures" / "golden_game.bin"

# Human solo play through `doppelt play`, saved as game.bin.
EXPECTED_SEED = 732
EXPECTED_ACTION_COUNT = 94
EXPECTED_TOTAL = 347
EXPECTED_SCORES = {
    "yellow": 36,
    "blue": 36,
    "pink": 35,
    "green": 49,
    "silver": 51,
    "foxes": 140,
}
EXPECTED_SHEET_HASH = "c709686b56b07343cc2128db2ced0e5be63e786dbcae41ca7a85f439dad20f6d"
EXPECTED_SHEET = {
    "yellow": [
        [True, True],
        [True, True],
        [False, False],
        [False, False],
        [True, True],
        [True, False],
        [False, False],
        [False, False],
        [True, True],
        [True, False],
    ],
    "blue": [10, 7, 7, 7, 7, 6, 6, 4, None, None, None, None],
    "pink": [1, 1, 1, 4, 6, 6, 6, 5, 5, None, None, None],
    "green": [12, 2, 12, 1, 18, 6, 18, 2, 6, None, None, None],
    "green_stars": [10, 11, 12, 16, None, None],
    "silver": {
        "yellow": [1, 2, 3, 5],
        "blue": [1, 2, 3, 4, 5, 6],
        "green": [1, 2, 3],
        "pink": [1, 2, 3, 5],
    },
    "foxes": 4,
    "claimed_bonuses": [
        "action:plus_one:end",
        "blue:1",
        "blue:2",
        "blue:4",
        "blue:5",
        "blue:6",
        "green:1",
        "green:3",
        "green:4",
        "green:6",
        "green:7",
        "green:8",
        "pink:3",
        "pink:4",
        "pink:5",
        "pink:6",
        "pink:7",
        "pink:8",
        "silver:col:1",
        "silver:col:2",
        "silver:col:3",
        "yellow:col:1",
        "yellow:col:3",
        "yellow:row:0",
        "yellow:row:2",
        "yellow:row:4",
    ],
    "action_tracks": {
        "reroll": [1, 2],
        "unlock": [0, 4],
        "plus_one": [0, 6],
    },
}


def _sheet_snapshot(sheet: PlayerSheet) -> dict:
    return {
        "yellow": [[cell.circled, cell.crossed] for cell in sheet.yellow],
        "blue": list(sheet.blue),
        "pink": list(sheet.pink),
        "green": list(sheet.green),
        "green_stars": list(sheet.green_stars),
        "silver": {color.value: sorted(values) for color, values in sheet.silver.items()},
        "foxes": sheet.foxes,
        "claimed_bonuses": sorted(sheet.claimed_bonuses),
        "action_tracks": {
            track.value: [slots.circled, slots.crossed]
            for track, slots in sheet.action_tracks.items()
        },
    }


def _sheet_hash(sheet: PlayerSheet) -> str:
    payload = json.dumps(_sheet_snapshot(sheet), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_golden_human_game_replays_to_frozen_sheet_and_score():
    data = GOLDEN_LOG.read_bytes()
    log = decode_log(data)

    assert log.seed == EXPECTED_SEED
    assert log.player_count == 1
    assert log.catalog_version == CATALOG_VERSION
    assert len(log.actions) == EXPECTED_ACTION_COUNT

    state = replay_game(log)
    done, scores = is_terminal(state)

    assert state.phase is Phase.GAME_OVER
    assert done
    assert scores == EXPECTED_SCORES
    assert score_sheet_areas(state.sheet) == EXPECTED_SCORES
    assert total_score(state.sheet) == EXPECTED_TOTAL
    assert _sheet_snapshot(state.sheet) == EXPECTED_SHEET
    assert _sheet_hash(state.sheet) == EXPECTED_SHEET_HASH

    replayed_again = replay_game(data)
    assert replayed_again.sheet == state.sheet
    assert total_score(replayed_again.sheet) == EXPECTED_TOTAL
