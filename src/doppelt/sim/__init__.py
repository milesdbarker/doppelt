"""Batch simulation and baseline bots."""

from doppelt.sim.analytics import (
    DatasetAnalytics,
    PolicyAnalytics,
    analyze_dataset,
    analyze_records,
    format_analytics_report,
    write_analytics_json,
)
from doppelt.sim.batch import (
    AREA_SCORE_KEYS,
    BatchResult,
    GameOutcome,
    format_batch_result,
    play_with_policy,
    resolve_worker_count,
    run_batch,
)
from doppelt.sim.dataset import (
    DATASET_POLICIES,
    DatasetError,
    DatasetRecord,
    decode_shard,
    encode_shard,
    generate_dataset,
    iter_dataset_records,
    list_shard_paths,
    read_shard,
    record_to_game_log,
    write_shard,
)
from doppelt.sim.greedy import GreedyImmediate
from doppelt.sim.heuristic import Heuristic
from doppelt.sim.mcts_lite import MctsLite
from doppelt.sim.policy import POLICY_NAMES, POLICY_RNG_XOR, Policy, make_policy
from doppelt.sim.random_legal import RandomLegal

__all__ = [
    "AREA_SCORE_KEYS",
    "DATASET_POLICIES",
    "DatasetAnalytics",
    "DatasetError",
    "DatasetRecord",
    "POLICY_NAMES",
    "POLICY_RNG_XOR",
    "BatchResult",
    "GameOutcome",
    "GreedyImmediate",
    "Heuristic",
    "MctsLite",
    "Policy",
    "PolicyAnalytics",
    "RandomLegal",
    "analyze_dataset",
    "analyze_records",
    "decode_shard",
    "encode_shard",
    "format_analytics_report",
    "format_batch_result",
    "generate_dataset",
    "iter_dataset_records",
    "list_shard_paths",
    "make_policy",
    "play_with_policy",
    "read_shard",
    "record_to_game_log",
    "resolve_worker_count",
    "run_batch",
    "write_analytics_json",
    "write_shard",
]
