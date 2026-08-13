"""Bulk solo dataset shards: seed + uint16 actions + score metadata.

Shard magic ``DPLD`` (format v1). One policy per shard. Replay any record via
``doppelt.replay.replay_game`` after converting to ``GameLog``.
"""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from doppelt.actions.catalog_v1 import ACTION_SPACE_SIZE, CATALOG_VERSION
from doppelt.replay.binary import GameLog
from doppelt.sim.batch import AREA_SCORE_KEYS, GameOutcome, resolve_worker_count, run_batch

DATASET_MAGIC = b"DPLD"
DATASET_FORMAT_VERSION = 1
DATASET_POLICIES = ("random_legal", "greedy_immediate", "heuristic")
DEFAULT_MIX = "heuristic:5,greedy_immediate:3,random_legal:2"
DEFAULT_SHARD_SIZE = 50_000

_HEADER_PREFIX = struct.Struct("<4sHH")  # magic, format_version, catalog_version
_COUNTS = struct.Struct("<II")  # n_records, flags
_RECORD_PREFIX = struct.Struct("<QBB7hH")  # seed, terminal, player_count, scores×7, n_actions


@dataclass(frozen=True)
class DatasetRecord:
    seed: int
    policy: str
    terminal: bool
    player_count: int
    total_score: int
    scores: dict[str, int]
    actions: tuple[int, ...]


class DatasetError(ValueError):
    """Invalid dataset shard bytes or generate arguments."""


def parse_mix(spec: str) -> dict[str, int]:
    """Parse ``heuristic:5,greedy_immediate:3,random_legal:2`` into weights."""
    mix: dict[str, int] = {}
    for part in spec.split(","):
        piece = part.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise DatasetError(f"invalid mix entry {piece!r}; expected name:weight")
        name, weight_text = piece.split(":", 1)
        name = name.strip()
        if name not in DATASET_POLICIES:
            raise DatasetError(
                f"dataset policy {name!r} is not allowed; expected one of {DATASET_POLICIES}"
            )
        try:
            weight = int(weight_text.strip())
        except ValueError as exc:
            raise DatasetError(f"invalid mix weight in {piece!r}") from exc
        if weight < 1:
            raise DatasetError(f"mix weight must be positive in {piece!r}")
        if name in mix:
            raise DatasetError(f"duplicate mix policy {name!r}")
        mix[name] = weight
    if not mix:
        raise DatasetError("mix is empty")
    return mix


def allocate_mix(n_games: int, mix: dict[str, int]) -> dict[str, int]:
    if n_games < 1:
        raise DatasetError("n_games must be at least 1")
    total_w = sum(mix.values())
    counts = {name: n_games * weight // total_w for name, weight in mix.items()}
    remainder = n_games - sum(counts.values())
    for name, _weight in sorted(mix.items(), key=lambda item: (-item[1], item[0])):
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1
    return counts


def outcome_to_record(outcome: GameOutcome, policy: str) -> DatasetRecord:
    return DatasetRecord(
        seed=outcome.seed,
        policy=policy,
        terminal=outcome.terminal,
        player_count=1,
        total_score=outcome.total_score,
        scores=dict(outcome.scores),
        actions=outcome.actions,
    )


def record_to_game_log(record: DatasetRecord) -> GameLog:
    return GameLog(
        seed=record.seed,
        player_count=record.player_count,
        catalog_version=CATALOG_VERSION,
        actions=record.actions,
    )


def encode_shard(policy: str, records: Sequence[DatasetRecord]) -> bytes:
    if policy not in DATASET_POLICIES:
        raise DatasetError(f"unsupported shard policy {policy!r}")
    name_bytes = policy.encode("utf-8")
    if len(name_bytes) > 255:
        raise DatasetError("policy name too long")
    parts = [
        _HEADER_PREFIX.pack(DATASET_MAGIC, DATASET_FORMAT_VERSION, CATALOG_VERSION),
        struct.pack("<H", len(name_bytes)),
        name_bytes,
        _COUNTS.pack(len(records), 0),
    ]
    for record in records:
        if record.policy != policy:
            raise DatasetError(f"record policy {record.policy!r} != shard policy {policy!r}")
        if record.player_count < 1:
            raise DatasetError("player_count must be at least 1")
        if len(record.actions) > 0xFFFF:
            raise DatasetError(f"seed {record.seed}: too many actions for uint16 count")
        for action_id in record.actions:
            if not 0 <= action_id < ACTION_SPACE_SIZE:
                raise DatasetError(f"seed {record.seed}: action id {action_id} out of range")
        scores = [record.total_score, *[record.scores.get(key, 0) for key in AREA_SCORE_KEYS]]
        parts.append(
            _RECORD_PREFIX.pack(
                record.seed & 0xFFFFFFFFFFFFFFFF,
                1 if record.terminal else 0,
                record.player_count,
                *scores,
                len(record.actions),
            )
        )
        if record.actions:
            parts.append(struct.pack(f"<{len(record.actions)}H", *record.actions))
    return b"".join(parts)


def decode_shard(data: bytes) -> tuple[str, tuple[DatasetRecord, ...]]:
    if len(data) < _HEADER_PREFIX.size + 2 + _COUNTS.size:
        raise DatasetError("shard too short")
    magic, format_version, catalog_version = _HEADER_PREFIX.unpack_from(data, 0)
    if magic != DATASET_MAGIC:
        raise DatasetError(f"invalid magic {magic!r}; expected {DATASET_MAGIC!r}")
    if format_version != DATASET_FORMAT_VERSION:
        raise DatasetError(f"unsupported dataset format {format_version}")
    if catalog_version != CATALOG_VERSION:
        raise DatasetError(f"unsupported catalog version {catalog_version}")
    offset = _HEADER_PREFIX.size
    (name_len,) = struct.unpack_from("<H", data, offset)
    offset += 2
    if offset + name_len + _COUNTS.size > len(data):
        raise DatasetError("shard header truncated")
    policy = data[offset : offset + name_len].decode("utf-8")
    offset += name_len
    n_records, flags = _COUNTS.unpack_from(data, offset)
    offset += _COUNTS.size
    if flags != 0:
        raise DatasetError(f"unsupported shard flags {flags}")
    if policy not in DATASET_POLICIES:
        raise DatasetError(f"unsupported shard policy {policy!r}")

    records: list[DatasetRecord] = []
    for _ in range(n_records):
        if offset + _RECORD_PREFIX.size > len(data):
            raise DatasetError("shard record prefix truncated")
        unpacked = _RECORD_PREFIX.unpack_from(data, offset)
        offset += _RECORD_PREFIX.size
        seed, terminal, player_count, *score_values, n_actions = unpacked
        total_score = score_values[0]
        area_scores = dict(zip(AREA_SCORE_KEYS, score_values[1:], strict=True))
        nbytes = n_actions * 2
        if offset + nbytes > len(data):
            raise DatasetError(f"seed {seed}: actions truncated")
        if n_actions:
            actions = struct.unpack_from(f"<{n_actions}H", data, offset)
        else:
            actions = ()
        offset += nbytes
        records.append(
            DatasetRecord(
                seed=seed,
                policy=policy,
                terminal=bool(terminal),
                player_count=player_count,
                total_score=total_score,
                scores=area_scores,
                actions=actions,
            )
        )
    if offset != len(data):
        raise DatasetError(f"shard has {len(data) - offset} trailing bytes")
    return policy, tuple(records)


def write_shard(path: Path, policy: str, records: Sequence[DatasetRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_shard(policy, records))


def read_shard(path: Path) -> tuple[str, tuple[DatasetRecord, ...]]:
    return decode_shard(path.read_bytes())


def list_shard_paths(out_dir: Path) -> list[Path]:
    """Shard paths from ``manifest.json`` when present, else ``**/*.dpld``."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = [out_dir / shard["path"] for shard in manifest.get("shards", [])]
        return [path for path in paths if path.is_file()]
    return sorted(out_dir.glob("**/*.dpld"))


def iter_dataset_records(out_dir: Path) -> Iterator[DatasetRecord]:
    paths = list_shard_paths(out_dir)
    if not paths:
        raise DatasetError(f"no DPLD shards in {out_dir}")
    for path in paths:
        _policy, records = read_shard(path)
        yield from records


def _shard_path(out_dir: Path, policy: str, shard_index: int) -> Path:
    return out_dir / policy / f"shard_{shard_index:05d}.dpld"


def _write_manifest(out_dir: Path, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def generate_dataset(
    out_dir: Path,
    *,
    n_games: int,
    mix: str = DEFAULT_MIX,
    shard_size: int = DEFAULT_SHARD_SIZE,
    seed_start: int = 0,
    workers: int | None = None,
    max_actions: int = 5_000,
    resume: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Generate mixed-bot DPLD shards. Skips finished shards when ``resume`` is true."""
    if shard_size < 1:
        raise DatasetError("shard_size must be at least 1")
    weights = parse_mix(mix)
    counts = allocate_mix(n_games, weights)
    worker_count = resolve_worker_count(workers)
    out_dir = Path(out_dir)

    manifest: dict = {
        "format": "DPLD",
        "format_version": DATASET_FORMAT_VERSION,
        "catalog_version": CATALOG_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed_start": seed_start,
        "games": n_games,
        "mix_weights": weights,
        "mix_counts": counts,
        "shard_size": shard_size,
        "workers": worker_count,
        "max_actions": max_actions,
        "shards": [],
    }

    def log(message: str) -> None:
        if on_progress is not None:
            on_progress(message)
    seed_cursor = seed_start
    started = time.perf_counter()
    total_unfinished = 0

    for policy, policy_games in counts.items():
        shard_index = 0
        remaining = policy_games
        policy_seed = seed_cursor
        while remaining > 0:
            chunk = min(shard_size, remaining)
            rel = _shard_path(out_dir, policy, shard_index).relative_to(out_dir).as_posix()
            shard_file = out_dir / rel
            log(
                f"start {rel}: {chunk} {policy} games, seeds "
                f"{policy_seed}-{policy_seed + chunk - 1}"
            )
            if resume and shard_file.is_file():
                stored_policy, stored = read_shard(shard_file)
                if stored_policy == policy and len(stored) == chunk:
                    unfinished = sum(1 for record in stored if not record.terminal)
                    total_unfinished += unfinished
                    manifest["shards"].append(
                        {
                            "path": rel,
                            "policy": policy,
                            "games": chunk,
                            "seed_start": policy_seed,
                            "unfinished": unfinished,
                            "resumed": True,
                        }
                    )
                    log(f"skip existing {rel} ({chunk} games)")
                    _write_manifest(out_dir, manifest)
                    remaining -= chunk
                    policy_seed += chunk
                    shard_index += 1
                    continue

            result = run_batch(
                chunk,
                seed_start=policy_seed,
                workers=worker_count,
                max_actions=max_actions,
                policy=policy,
            )
            records = [outcome_to_record(outcome, policy) for outcome in result.outcomes]
            write_shard(shard_file, policy, records)
            total_unfinished += result.unfinished
            manifest["shards"].append(
                {
                    "path": rel,
                    "policy": policy,
                    "games": chunk,
                    "seed_start": policy_seed,
                    "unfinished": result.unfinished,
                    "mean_score": round(result.mean_score, 2),
                    "elapsed_sec": round(result.elapsed_sec, 3),
                    "games_per_sec": round(result.games_per_sec, 1),
                }
            )
            _write_manifest(out_dir, manifest)
            log(
                f"wrote {rel}: {chunk} {policy} games, "
                f"{result.games_per_sec:.0f}/s, unfinished={result.unfinished}"
            )
            remaining -= chunk
            policy_seed += chunk
            shard_index += 1
        seed_cursor = policy_seed

    manifest["elapsed_sec"] = round(time.perf_counter() - started, 3)
    manifest["unfinished"] = total_unfinished
    _write_manifest(out_dir, manifest)
    log(
        f"dataset complete: {n_games} games -> {out_dir} "
        f"in {manifest['elapsed_sec']:.1f}s, unfinished={total_unfinished}"
    )
    return manifest
