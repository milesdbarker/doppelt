"""Behavioral-cloning examples from DPLD shards or live bot games."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from doppelt.core.phases import Phase
from doppelt.core.scoring import total_score
from doppelt.engine.game import apply_action, legal_action_ids, new_game
from doppelt.ml.encoding import SCORE_SCALE, encode_state
from doppelt.sim.batch import play_game_with_policy
from doppelt.sim.dataset import DatasetRecord, iter_dataset_records, record_to_game_log
from doppelt.sim.policy import make_policy


@dataclass(frozen=True)
class BCExample:
    features: tuple[float, ...]
    mask: tuple[bool, ...]
    action_id: int
    return_score: float
    seed: int


def examples_from_record(record: DatasetRecord) -> list[BCExample]:
    """Replay a finished game and label each state with the logged action."""
    if not record.terminal or not record.actions:
        return []
    log = record_to_game_log(record)
    state = new_game(seed=log.seed, player_count=log.player_count)
    scaled = record.total_score / SCORE_SCALE
    examples: list[BCExample] = []
    for action_id in log.actions:
        if state.phase is Phase.GAME_OVER:
            break
        legal = legal_action_ids(state)
        encoded = encode_state(state, legal)
        if not encoded.mask[action_id]:
            raise ValueError(f"logged action {action_id} illegal at seed={record.seed}")
        examples.append(
            BCExample(
                features=encoded.features,
                mask=encoded.mask,
                action_id=action_id,
                return_score=scaled,
                seed=record.seed,
            )
        )
        apply_action(state, action_id, check_legal=False)
    return examples


def iter_examples_from_dataset(
    in_dir: Path,
    *,
    policy: str | None = "heuristic",
    max_games: int | None = None,
) -> Iterator[BCExample]:
    seen = 0
    for record in iter_dataset_records(in_dir):
        if policy is not None and record.policy != policy:
            continue
        if not record.terminal:
            continue
        for example in examples_from_record(record):
            yield example
        seen += 1
        if max_games is not None and seen >= max_games:
            return


def examples_from_live_games(
    n_games: int,
    *,
    policy: str = "heuristic",
    seed_start: int = 0,
    max_actions: int = 5_000,
) -> list[BCExample]:
    """Play bots live and clone their actions (no DPLD shards required)."""
    examples: list[BCExample] = []
    for index in range(n_games):
        seed = seed_start + index
        bot = make_policy(policy, seed)
        state = play_game_with_policy(seed, bot, max_actions=max_actions)
        record = DatasetRecord(
            seed=seed,
            policy=bot.name,
            terminal=state.phase is Phase.GAME_OVER,
            player_count=1,
            total_score=total_score(state.sheet),
            scores={},
            actions=tuple(state.action_log),
        )
        examples.extend(examples_from_record(record))
    return examples


def examples_from_neural_games(
    n_games: int,
    *,
    checkpoint: Path | None = None,
    net=None,
    payload: dict | None = None,
    seed_start: int = 0,
    mcts_sims: int = 32,
    mcts_plies: int = 2,
    visit_sample: bool = False,
    max_actions: int = 5_000,
) -> list[BCExample]:
    """Play neural (+ optional PUCT) games and clone the actions search actually took."""
    from doppelt.ml.policy import NeuralPolicy

    teacher = NeuralPolicy(
        checkpoint,
        seed=seed_start,
        net=net,
        payload=payload,
        mcts_sims=mcts_sims,
        mcts_plies=mcts_plies,
        visit_sample=visit_sample,
    )
    examples: list[BCExample] = []
    for index in range(n_games):
        seed = seed_start + index
        teacher._search_rng.seed(seed ^ 0x4C75)
        teacher._generator.manual_seed(seed)
        state = play_game_with_policy(seed, teacher, max_actions=max_actions)
        record = DatasetRecord(
            seed=seed,
            policy=teacher.name,
            terminal=state.phase is Phase.GAME_OVER,
            player_count=1,
            total_score=total_score(state.sheet),
            scores={},
            actions=tuple(state.action_log),
        )
        if not record.terminal:
            continue
        examples.extend(examples_from_record(record))
    return examples


def split_by_seed(
    examples: Sequence[BCExample],
    *,
    val_frac: float = 0.1,
) -> tuple[list[BCExample], list[BCExample]]:
    """Hold out seeds with ``seed % 10 < val_frac * 10`` (no shuffle leakage)."""
    if not 0.0 <= val_frac < 1.0:
        raise ValueError("val_frac must be in [0, 1)")
    threshold = val_frac * 10.0
    train: list[BCExample] = []
    val: list[BCExample] = []
    for example in examples:
        if (example.seed % 10) < threshold:
            val.append(example)
        else:
            train.append(example)
    if not train:
        return list(examples), []
    return train, val
