"""Binary action log encoding and replay.

Logs store ``seed``, ``player_count``, catalog version, and a ``uint16`` action
array. Dice and tie-break randomness come only from ``GameState.rng`` during
``apply_action`` — bot or training code must not consume that RNG between actions.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from doppelt.actions.catalog_v1 import ACTION_SPACE_SIZE, CATALOG_VERSION
from doppelt.core.state import GameState
from doppelt.engine.game import apply_action, new_game

MAGIC = b"DPL1"
HEADER_STRUCT = struct.Struct("<4sHBBQI")
HEADER_SIZE = HEADER_STRUCT.size  # 20 bytes


class ReplayError(ValueError):
    """Invalid log bytes or replay failure."""


@dataclass(frozen=True)
class GameLog:
    seed: int
    player_count: int
    catalog_version: int
    actions: tuple[int, ...]


def encode_log(log: GameLog) -> bytes:
    if log.catalog_version != CATALOG_VERSION:
        raise ReplayError(
            f"unsupported catalog version {log.catalog_version}; expected {CATALOG_VERSION}"
        )
    if log.player_count < 1:
        raise ReplayError("player_count must be at least 1")
    if len(log.actions) > 0xFFFFFFFF:
        raise ReplayError("action log exceeds uint32 capacity")

    for action_id in log.actions:
        if not 0 <= action_id < ACTION_SPACE_SIZE:
            raise ReplayError(f"action id {action_id} out of range [0, {ACTION_SPACE_SIZE})")

    header = HEADER_STRUCT.pack(
        MAGIC,
        log.catalog_version,
        log.player_count,
        0,
        log.seed & 0xFFFFFFFFFFFFFFFF,
        len(log.actions),
    )
    body = struct.pack(f"<{len(log.actions)}H", *log.actions) if log.actions else b""
    return header + body


def decode_log(data: bytes) -> GameLog:
    if len(data) < HEADER_SIZE:
        raise ReplayError(f"log too short: {len(data)} bytes (need at least {HEADER_SIZE})")

    magic, catalog_version, player_count, flags, seed, action_count = HEADER_STRUCT.unpack_from(
        data, 0
    )
    if magic != MAGIC:
        raise ReplayError(f"invalid magic {magic!r}; expected {MAGIC!r}")
    if flags != 0:
        raise ReplayError(f"unsupported header flags {flags}")
    if catalog_version != CATALOG_VERSION:
        raise ReplayError(
            f"unsupported catalog version {catalog_version}; expected {CATALOG_VERSION}"
        )
    if player_count < 1:
        raise ReplayError(f"invalid player_count {player_count}")

    expected_size = HEADER_SIZE + action_count * 2
    if len(data) != expected_size:
        raise ReplayError(
            f"log size mismatch: got {len(data)} bytes, expected {expected_size} "
            f"for {action_count} actions"
        )

    if action_count:
        actions = struct.unpack_from(f"<{action_count}H", data, HEADER_SIZE)
    else:
        actions = ()

    for action_id in actions:
        if not 0 <= action_id < ACTION_SPACE_SIZE:
            raise ReplayError(f"action id {action_id} out of range [0, {ACTION_SPACE_SIZE})")

    return GameLog(
        seed=seed,
        player_count=player_count,
        catalog_version=catalog_version,
        actions=actions,
    )


def export_log(state: GameState) -> bytes:
    """Serialize seed, player count, and recorded actions from a finished or in-progress game."""
    return encode_log(
        GameLog(
            seed=state.seed,
            player_count=state.player_count,
            catalog_version=CATALOG_VERSION,
            actions=tuple(state.action_log),
        )
    )


def import_log(data: bytes) -> GameLog:
    """Parse a binary action log."""
    return decode_log(data)


def replay_game(log: GameLog | bytes) -> GameState:
    """Reconstruct game state by replaying a binary log from its initial seed."""
    if isinstance(log, bytes):
        log = decode_log(log)

    state = new_game(seed=log.seed, player_count=log.player_count)
    for action_id in log.actions:
        apply_action(state, action_id)
    return state
