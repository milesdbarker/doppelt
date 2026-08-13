"""DPLD shard format and mixed-bot dataset generation."""

import io
import json
from pathlib import Path

import pytest

from doppelt.cli.commands import run_dataset_generate
from doppelt.core.scoring import total_score
from doppelt.replay.binary import replay_game
from doppelt.sim.dataset import (
    DATASET_POLICIES,
    DatasetError,
    DatasetRecord,
    allocate_mix,
    decode_shard,
    encode_shard,
    generate_dataset,
    parse_mix,
    read_shard,
    record_to_game_log,
)
from doppelt.sim.batch import play_with_policy
from doppelt.sim.random_legal import RandomLegal


def test_parse_mix_default_and_rejects_mcts():
    mix = parse_mix("heuristic:5,greedy_immediate:3,random_legal:2")
    assert mix == {"heuristic": 5, "greedy_immediate": 3, "random_legal": 2}
    with pytest.raises(DatasetError, match="not allowed"):
        parse_mix("mcts_lite:1")
    with pytest.raises(DatasetError, match="not allowed"):
        parse_mix("heuristic:1,mcts_lite:1")


def test_allocate_mix_exact_and_remainder():
    mix = {"heuristic": 5, "greedy_immediate": 3, "random_legal": 2}
    assert allocate_mix(1_000_000, mix) == {
        "heuristic": 500_000,
        "greedy_immediate": 300_000,
        "random_legal": 200_000,
    }
    assert sum(allocate_mix(11, mix).values()) == 11
    assert allocate_mix(11, mix)["heuristic"] == 6


def test_encode_decode_shard_roundtrip():
    record = DatasetRecord(
        seed=7,
        policy="random_legal",
        terminal=True,
        player_count=1,
        total_score=64,
        scores={"yellow": 10, "blue": 12, "pink": 8, "green": 14, "silver": 16, "foxes": 4},
        actions=(0, 1, 2, 191),
    )
    blob = encode_shard("random_legal", [record])
    policy, decoded = decode_shard(blob)
    assert policy == "random_legal"
    assert decoded == (record,)


def test_encode_shard_rejects_mcts_policy():
    record = DatasetRecord(
        seed=1,
        policy="mcts_lite",
        terminal=True,
        player_count=1,
        total_score=0,
        scores={},
        actions=(),
    )
    with pytest.raises(DatasetError, match="unsupported shard policy"):
        encode_shard("mcts_lite", [record])


def test_record_replays_to_same_score():
    outcome = play_with_policy(41, RandomLegal(41), max_actions=5_000)
    record = DatasetRecord(
        seed=outcome.seed,
        policy="random_legal",
        terminal=outcome.terminal,
        player_count=1,
        total_score=outcome.total_score,
        scores=dict(outcome.scores),
        actions=outcome.actions,
    )
    state = replay_game(record_to_game_log(record))
    assert outcome.terminal
    assert total_score(state.sheet) == outcome.total_score
    assert tuple(state.action_log) == outcome.actions


def test_generate_dataset_writes_shards_and_manifest(tmp_path: Path):
    out_dir = tmp_path / "solo_v1"
    messages: list[str] = []
    manifest = generate_dataset(
        out_dir,
        n_games=6,
        mix="heuristic:1,greedy_immediate:1,random_legal:1",
        shard_size=2,
        seed_start=20,
        workers=1,
        on_progress=messages.append,
    )
    assert manifest["games"] == 6
    assert manifest["mix_counts"] == {
        "heuristic": 2,
        "greedy_immediate": 2,
        "random_legal": 2,
    }
    assert manifest["unfinished"] == 0
    assert (out_dir / "manifest.json").is_file()
    stored = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stored["catalog_version"] == manifest["catalog_version"]
    assert len(stored["shards"]) == 3
    for shard_info in stored["shards"]:
        policy, records = read_shard(out_dir / shard_info["path"])
        assert policy == shard_info["policy"]
        assert policy in DATASET_POLICIES
        assert len(records) == 2
        assert all(record.terminal for record in records)
        assert all(record.actions for record in records)
    assert any("wrote" in message for message in messages)

    again = generate_dataset(
        out_dir,
        n_games=6,
        mix="heuristic:1,greedy_immediate:1,random_legal:1",
        shard_size=2,
        seed_start=20,
        workers=1,
        resume=True,
        on_progress=messages.append,
    )
    assert all(shard.get("resumed") for shard in again["shards"])
    assert any("skip existing" in message for message in messages)


def test_dataset_generate_cli_rejects_mcts(tmp_path: Path):
    buf = io.StringIO()
    code = run_dataset_generate(
        out_dir=tmp_path / "bad",
        games=4,
        mix="mcts_lite:1",
        shard_size=4,
        workers=1,
        output=buf,
    )
    assert code == 2
    assert "dataset error" in buf.getvalue()


def test_dataset_generate_cli_small_run(tmp_path: Path):
    out_dir = tmp_path / "cli_ds"
    buf = io.StringIO()
    code = run_dataset_generate(
        out_dir=out_dir,
        games=3,
        mix="random_legal:1",
        shard_size=3,
        seed=90,
        workers=1,
        output=buf,
    )
    assert code == 0
    assert (out_dir / "random_legal" / "shard_00000.dpld").is_file()
    assert "manifest.json" in buf.getvalue()
