"""Binary log export/import and deterministic replay."""

import struct

import pytest

from doppelt.actions.catalog_v1 import ACTION_SPACE_SIZE, CATALOG_VERSION
from doppelt.core.phases import Phase
from doppelt.core.scoring import total_score
from doppelt.engine.game import is_terminal, new_game, play_random_game
from doppelt.replay import (
    HEADER_SIZE,
    MAGIC,
    GameLog,
    ReplayError,
    decode_log,
    encode_log,
    export_log,
    import_log,
    replay_game,
)


def test_header_layout_is_20_bytes():
    assert HEADER_SIZE == 20
    log = GameLog(seed=42, player_count=1, catalog_version=CATALOG_VERSION, actions=(0, 130))
    header = encode_log(log)[:HEADER_SIZE]
    magic, catalog_version, player_count, flags, seed, action_count = struct.unpack(
        "<4sHBBQI", header
    )
    assert magic == MAGIC
    assert catalog_version == CATALOG_VERSION
    assert player_count == 1
    assert flags == 0
    assert seed == 42
    assert action_count == 2


def test_encode_decode_round_trip():
    actions = (130, 1, 20, 0)
    log = GameLog(seed=999, player_count=1, catalog_version=CATALOG_VERSION, actions=actions)
    data = encode_log(log)
    restored = decode_log(data)
    assert restored == log
    assert import_log(data) == log


def test_export_log_uses_state_action_log():
    state = new_game(seed=7)
    state.action_log = [130, 1, 20]
    data = export_log(state)
    log = decode_log(data)
    assert log.seed == 7
    assert log.player_count == 1
    assert log.actions == (130, 1, 20)


def test_decode_rejects_bad_magic():
    log = GameLog(seed=1, player_count=1, catalog_version=CATALOG_VERSION, actions=(0,))
    data = bytearray(encode_log(log))
    data[0:4] = b"BAD!"
    with pytest.raises(ReplayError, match="invalid magic"):
        decode_log(bytes(data))


def test_decode_rejects_truncated_payload():
    log = GameLog(seed=1, player_count=1, catalog_version=CATALOG_VERSION, actions=(0, 1))
    data = encode_log(log)[:-1]
    with pytest.raises(ReplayError, match="size mismatch"):
        decode_log(data)


def test_decode_rejects_out_of_range_action_id():
    log = GameLog(
        seed=1,
        player_count=1,
        catalog_version=CATALOG_VERSION,
        actions=(ACTION_SPACE_SIZE,),
    )
    with pytest.raises(ReplayError, match="out of range"):
        encode_log(log)


def test_random_game_replay_matches_terminal_state():
    original = play_random_game(seed=123, max_actions=5000)
    replayed = replay_game(export_log(original))

    assert original.phase == replayed.phase
    assert original.round_index == replayed.round_index
    assert original.action_log == replayed.action_log
    assert original.sheet == replayed.sheet

    done_original, scores_original = is_terminal(original)
    done_replayed, scores_replayed = is_terminal(replayed)
    assert done_original == done_replayed
    assert scores_original == scores_replayed
    assert total_score(original.sheet) == total_score(replayed.sheet)


def test_replay_from_bytes_matches_scores():
    original = play_random_game(seed=456, max_actions=5000)
    data = export_log(original)
    replayed = replay_game(data)

    assert replayed.phase is Phase.GAME_OVER
    done, scores = is_terminal(replayed)
    assert done
    assert scores == is_terminal(original)[1]
    assert total_score(replayed.sheet) == total_score(original.sheet)
