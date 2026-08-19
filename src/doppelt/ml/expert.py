"""Expert iteration: PUCT search as teacher, BC distill onto the net (3.6.1).

A2C self-play does not run MCTS. This loop generates search games, clones the
visit-greedy (or visit-sampled) actions, and fine-tunes the current checkpoint.
Best weights are kept by greedy eval on the dev seed range, not search play.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from doppelt.ml.bc_data import examples_from_neural_games, split_by_seed
from doppelt.ml.eval import eval_policy_scores
from doppelt.ml.model import load_checkpoint, save_checkpoint
from doppelt.ml.policy import NeuralPolicy
from doppelt.ml.train import TrainResult, train_bc


@dataclass(frozen=True)
class ExpertResult:
    checkpoint_path: Path
    rounds: int
    games: int
    mcts_sims: int
    best_eval_score: float
    last_train_loss: float
    last_n_train: int


def _greedy_eval(net, payload: dict, *, games: int, seed: int, max_actions: int) -> float:
    policy = NeuralPolicy(net=net, payload=payload, seed=seed, sample=False, mcts_sims=0)
    row, _failures = eval_policy_scores(
        policy,
        games=games,
        seed_start=seed,
        max_actions=max_actions,
        collect_failures=False,
    )
    return row.mean_score


def train_expert(
    *,
    init_checkpoint: Path,
    out_path: Path,
    rounds: int = 3,
    games_per_round: int = 32,
    mcts_sims: int = 32,
    mcts_plies: int = 2,
    visit_sample: bool = False,
    seed: int = 0,
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_frac: float = 0.1,
    eval_games: int = 16,
    eval_seed: int = 20_000,
    max_actions: int = 5_000,
    on_progress: Callable[[str], None] | None = None,
) -> ExpertResult:
    """Generate PUCT games from the current net, distill, repeat."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if games_per_round < 1:
        raise ValueError("games_per_round must be at least 1")
    if mcts_sims < 1:
        raise ValueError("mcts_sims must be at least 1")
    if mcts_plies < 1:
        raise ValueError("mcts_plies must be at least 1")

    init_checkpoint = Path(init_checkpoint)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_path = out_path.with_name(out_path.stem + "_round.pt")
    shutil.copy2(init_checkpoint, work_path)

    net, payload = load_checkpoint(work_path)
    hidden = int(payload.get("hidden", net.hidden))
    architecture = getattr(net, "architecture", "mlp")

    def log(message: str) -> None:
        if on_progress is not None:
            on_progress(message)

    best_eval = float("-inf")
    if eval_games >= 1:
        best_eval = _greedy_eval(
            net, payload, games=eval_games, seed=eval_seed, max_actions=max_actions
        )
        save_checkpoint(
            out_path,
            net,
            hidden=hidden,
            architecture=architecture,
            extra={
                **{key: payload[key] for key in ("score_scale",) if key in payload},
                "expert_round": 0,
                "eval_score": best_eval,
                "mcts_sims": mcts_sims,
            },
        )
        log(f"round 0: greedy eval mean={best_eval:.1f} (init)")
    else:
        shutil.copy2(work_path, out_path)

    last_train = TrainResult(out_path, 0, 0.0, 0.0, 0, 0)
    total_games = 0
    teacher_path = work_path

    for round_index in range(1, rounds + 1):
        log(
            f"round {round_index}/{rounds}: playing {games_per_round} games "
            f"with mcts_sims={mcts_sims} mcts_plies={mcts_plies}..."
        )
        examples = examples_from_neural_games(
            games_per_round,
            checkpoint=teacher_path,
            seed_start=seed + total_games,
            mcts_sims=mcts_sims,
            mcts_plies=mcts_plies,
            visit_sample=visit_sample,
            max_actions=max_actions,
        )
        total_games += games_per_round
        if not examples:
            raise RuntimeError("expert iteration collected no terminal-game examples")
        train_ex, val_ex = split_by_seed(examples, val_frac=val_frac)
        last_train = train_bc(
            train_ex,
            val_ex,
            out_path=work_path,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed + round_index,
            init_checkpoint=teacher_path,
        )
        net, payload = load_checkpoint(work_path)
        hidden = int(payload.get("hidden", hidden))
        architecture = getattr(net, "architecture", architecture)
        teacher_path = work_path

        if eval_games < 1:
            shutil.copy2(work_path, out_path)
            log(
                f"round {round_index}: distilled {last_train.n_train} actions, "
                f"train_loss={last_train.train_loss:.4f}"
            )
            continue

        mean = _greedy_eval(
            net, payload, games=eval_games, seed=eval_seed, max_actions=max_actions
        )
        if mean >= best_eval:
            best_eval = mean
            shutil.copy2(work_path, out_path)
        log(
            f"round {round_index}: search games={games_per_round}  "
            f"actions={last_train.n_train}  train_loss={last_train.train_loss:.4f}  "
            f"greedy eval mean={mean:.1f} (best={best_eval:.1f})"
        )

    if best_eval == float("-inf"):
        best_eval = 0.0

    return ExpertResult(
        checkpoint_path=Path(out_path),
        rounds=rounds,
        games=total_games,
        mcts_sims=mcts_sims,
        best_eval_score=best_eval,
        last_train_loss=last_train.train_loss,
        last_n_train=last_train.n_train,
    )


def format_expert_result(result: ExpertResult) -> str:
    return (
        f"expert-iteration: {result.games} search games, {result.rounds} rounds, "
        f"mcts_sims={result.mcts_sims}, last_n_train={result.last_n_train}, "
        f"last_train_loss={result.last_train_loss:.4f}, "
        f"best_eval={result.best_eval_score:.1f}\n"
        f"checkpoint: {result.checkpoint_path}"
    )
