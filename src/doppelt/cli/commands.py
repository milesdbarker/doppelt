"""CLI command implementations."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from doppelt.actions.catalog_v1 import ActionKind, CATALOG_VERSION, decode_action, describe_action
from doppelt.cli.display import (
    format_action_menu,
    format_scores,
    format_silver_grid,
    format_silver_pending,
    format_status,
    sort_actions_for_display,
)
from doppelt.cli.render import require_pillow, save_overlay
from doppelt.core.types import Dice
from doppelt.engine.game import (
    apply_action,
    is_terminal,
    legal_action_ids,
    new_game,
)
from doppelt.replay import ReplayError, decode_log, export_log, import_log, replay_game
from doppelt.sim import (
    format_batch_result,
    make_policy,
    play_game_with_policy,
    resolve_policy_name,
    resolve_worker_count,
    run_batch,
)
from doppelt.sim.analytics import analyze_dataset, format_analytics_report, write_analytics_json
from doppelt.sim.dataset import (
    DEFAULT_MIX,
    DEFAULT_SHARD_SIZE,
    DatasetError,
    generate_dataset,
)


def _write_log(path: Path, state) -> None:
    path.write_bytes(export_log(state))
    print(f"Saved log to {path}")


def _parse_action_choice(raw: str, legal: list[int]) -> int | None:
    text = raw.strip().lower()
    if text in {"q", "quit", "exit"}:
        return None
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"invalid choice: {raw!r}") from None
    if 0 <= value < len(legal):
        return legal[value]
    if value in legal:
        return value
    raise ValueError(f"choice {value} is not a legal action")


def _should_print_silver_grid(action_id: int) -> bool:
    action = decode_action(action_id)
    if action.kind is ActionKind.MARK_SILVER:
        return True
    if action.kind is ActionKind.MARK_WHITE_SILVER:
        return True
    if action.kind in (
        ActionKind.PICK_DIE,
        ActionKind.PASSIVE_PICK,
        ActionKind.PLUS_ONE_PICK,
    ):
        return action.die is Dice.SILVER
    return False


def run_play(
    *,
    seed: int,
    save_path: Path | None = None,
    input_fn: Callable[[str], str] = input,
    output: TextIO | None = None,
) -> int:
    """Interactive solo game."""
    out = output or sys.stdout
    state = new_game(seed=seed)

    print(f"New solo game (seed={seed})", file=out)
    while True:
        done, _ = is_terminal(state)
        if done:
            break

        legal = sort_actions_for_display(state, legal_action_ids(state))
        if not legal:
            print("No legal actions; stopping.", file=out)
            return 1

        print(file=out)
        print(format_status(state), file=out)

        if len(legal) == 1:
            action_id = legal[0]
            print(
                f"Auto: id={action_id}: {describe_action(action_id)}",
                file=out,
            )
        else:
            print(format_action_menu(legal, state), file=out)

            while True:
                try:
                    raw = input_fn("> ")
                except EOFError:
                    print("Interrupted.", file=out)
                    return 130

                try:
                    action_id = _parse_action_choice(raw, legal)
                except ValueError as error:
                    print(error, file=out)
                    continue

                if action_id is None:
                    print("Quit without finishing.", file=out)
                    return 0
                break

        apply_action(state, action_id)
        if _should_print_silver_grid(action_id):
            print(file=out)
            pending_silver = format_silver_pending(state)
            if pending_silver:
                print(pending_silver, file=out)
            print(format_silver_grid(state.sheet), file=out)

    print(file=out)
    print(format_scores(state), file=out)
    if save_path is not None:
        _write_log(save_path, state)
    return 0


def run_random(
    *,
    seed: int,
    max_actions: int = 10_000,
    save_path: Path | None = None,
    policy: str = "random",
    checkpoint: Path | None = None,
    mcts_sims: int = 0,
    mcts_plies: int = 2,
    output: TextIO | None = None,
) -> int:
    """Play one solo game with a named bot (dice RNG stays on the engine)."""
    out = output or sys.stdout
    if policy in {"neural", "nn"}:
        if checkpoint is None:
            print("neural policy requires --checkpoint PATH", file=out)
            return 2
        from doppelt.ml.policy import NeuralPolicy

        try:
            bot = NeuralPolicy(checkpoint, seed=seed, mcts_sims=mcts_sims, mcts_plies=mcts_plies)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Could not load checkpoint: {error}", file=out)
            return 1
        canonical = "neural"
    else:
        canonical = resolve_policy_name(policy)
        bot = make_policy(canonical, seed)
    state = play_game_with_policy(seed, bot, max_actions=max_actions)
    done, _ = is_terminal(state)

    print(f"{canonical} solo game (seed={seed})", file=out)
    print(f"Actions played: {len(state.action_log)}", file=out)
    print(f"Terminal: {done}  phase={state.phase.value}", file=out)
    print(format_scores(state), file=out)

    if save_path is not None:
        _write_log(save_path, state)
    return 0 if done else 1


def run_simulate(
    *,
    games: int,
    seed: int = 0,
    workers: int = 0,
    max_actions: int = 5_000,
    policy: str = "random_legal",
    output: TextIO | None = None,
) -> int:
    """Batch-simulate solo games with a named policy and print throughput."""
    out = output or sys.stdout
    worker_count = resolve_worker_count(workers)
    result = run_batch(
        games,
        seed_start=seed,
        workers=worker_count,
        max_actions=max_actions,
        policy=policy,
    )
    print(format_batch_result(result), file=out)
    return 0 if result.unfinished == 0 else 1


def run_dataset_generate(
    *,
    out_dir: Path,
    games: int,
    mix: str = DEFAULT_MIX,
    shard_size: int = DEFAULT_SHARD_SIZE,
    seed: int = 0,
    workers: int = 0,
    max_actions: int = 5_000,
    resume: bool = True,
    output: TextIO | None = None,
) -> int:
    """Generate mixed-bot DPLD shards (no mcts_lite)."""
    out = output or sys.stdout
    worker_count = resolve_worker_count(workers)
    try:
        manifest = generate_dataset(
            out_dir,
            n_games=games,
            mix=mix,
            shard_size=shard_size,
            seed_start=seed,
            workers=worker_count,
            max_actions=max_actions,
            resume=resume,
            on_progress=lambda message: print(message, file=out, flush=True),
        )
    except DatasetError as error:
        print(f"dataset error: {error}", file=out)
        return 2
    print(f"manifest: {out_dir / 'manifest.json'}", file=out)
    return 0 if manifest.get("unfinished", 0) == 0 else 1


def run_train_bc(
    *,
    in_dir: Path | None,
    live_games: int,
    policy: str,
    max_games: int,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden: int,
    architecture: str = "pvn_v1",
    lr: float,
    val_frac: float,
    out_path: Path,
    output: TextIO | None = None,
) -> int:
    """Clone bot actions into a masked policy/value checkpoint."""
    out = output or sys.stdout
    if in_dir is None and live_games < 1:
        print("train bc requires --in DIR and/or --live-games N", file=out)
        return 2
    from doppelt.ml.bc_data import (
        examples_from_live_games,
        iter_examples_from_dataset,
        split_by_seed,
    )
    from doppelt.ml.train import format_train_result, train_bc

    examples = []
    try:
        if live_games > 0:
            print(f"playing {live_games} {policy} games for BC labels...", file=out, flush=True)
            examples.extend(
                examples_from_live_games(live_games, policy=policy, seed_start=seed)
            )
        if in_dir is not None:
            print(
                f"replaying up to {max_games} {policy} shard games from {in_dir}...",
                file=out,
                flush=True,
            )
            examples.extend(
                list(iter_examples_from_dataset(in_dir, policy=policy, max_games=max_games))
            )
    except (DatasetError, OSError, RuntimeError, ValueError) as error:
        print(f"train error: {error}", file=out)
        return 1
    if not examples:
        print("no BC examples collected", file=out)
        return 1
    train_ex, val_ex = split_by_seed(examples, val_frac=val_frac)
    try:
        result = train_bc(
            train_ex,
            val_ex,
            out_path=out_path,
            epochs=epochs,
            batch_size=batch_size,
            hidden=hidden,
            architecture=architecture,
            lr=lr,
            seed=seed,
        )
    except RuntimeError as error:
        print(error, file=out)
        return 1
    print(format_train_result(result), file=out)
    return 0


def run_train_selfplay(
    *,
    init_checkpoint: Path | None,
    out_path: Path,
    iterations: int,
    games_per_iter: int,
    seed: int,
    lr: float,
    entropy_coef: float,
    value_coef: float,
    kl_coef: float,
    eval_games: int,
    eval_seed: int,
    eval_every: int,
    hidden: int,
    architecture: str,
    output: TextIO | None = None,
) -> int:
    """Actor-critic self-play; keep the checkpoint with the best greedy eval score."""
    out = output or sys.stdout
    from doppelt.ml.selfplay import format_selfplay_result, train_selfplay

    try:
        result = train_selfplay(
            out_path=out_path,
            init_checkpoint=init_checkpoint,
            iterations=iterations,
            games_per_iter=games_per_iter,
            seed=seed,
            lr=lr,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            kl_coef=kl_coef,
            eval_games=eval_games,
            eval_seed=eval_seed,
            eval_every=eval_every,
            hidden=hidden,
            architecture=architecture,
            on_progress=lambda message: print(message, file=out, flush=True),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"self-play error: {error}", file=out)
        return 1
    print(format_selfplay_result(result), file=out)
    return 0


def run_train_expert(
    *,
    init_checkpoint: Path,
    out_path: Path,
    rounds: int,
    games_per_round: int,
    mcts_sims: int,
    mcts_plies: int,
    visit_sample: bool,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    eval_games: int,
    eval_seed: int,
    output: TextIO | None = None,
) -> int:
    """PUCT search as teacher; distill onto the net; keep best greedy eval."""
    out = output or sys.stdout
    from doppelt.ml.expert import format_expert_result, train_expert

    try:
        result = train_expert(
            init_checkpoint=init_checkpoint,
            out_path=out_path,
            rounds=rounds,
            games_per_round=games_per_round,
            mcts_sims=mcts_sims,
            mcts_plies=mcts_plies,
            visit_sample=visit_sample,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            val_frac=val_frac,
            eval_games=eval_games,
            eval_seed=eval_seed,
            on_progress=lambda message: print(message, file=out, flush=True),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"expert-iteration error: {error}", file=out)
        return 1
    print(format_expert_result(result), file=out)
    return 0


def run_eval(
    *,
    checkpoint: Path,
    suite: str | None,
    games: int | None,
    seed: int | None,
    baselines: str,
    sample: bool = False,
    mcts_sims: int = 0,
    mcts_plies: int = 2,
    json_path: Path | None = None,
    collect_failures: bool = True,
    output: TextIO | None = None,
) -> int:
    """Evaluate a checkpoint against baseline bots on a frozen or custom suite."""
    out = output or sys.stdout
    names = [name.strip() for name in baselines.split(",") if name.strip()]
    from doppelt.ml.eval import eval_checkpoint, format_eval_report, write_eval_json
    from doppelt.ml.eval_suite import resolve_eval_suite

    try:
        resolved = resolve_eval_suite(suite, games=games, seed=seed)
        report = eval_checkpoint(
            checkpoint,
            games=resolved.games,
            seed_start=resolved.seed_start,
            baselines=names,
            sample=sample,
            mcts_sims=mcts_sims,
            mcts_plies=mcts_plies,
            suite=resolved,
            collect_failures=collect_failures,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"eval error: {error}", file=out)
        return 1
    print(format_eval_report(report), file=out)
    if json_path is not None:
        write_eval_json(report, json_path)
        print(f"json: {json_path}", file=out)
    return 0


def run_dataset_analyze(
    *,
    in_dir: Path,
    json_path: Path | None = None,
    output: TextIO | None = None,
) -> int:
    """Print score/strategy analytics for DPLD shards."""
    out = output or sys.stdout
    try:
        report = analyze_dataset(in_dir)
    except DatasetError as error:
        print(f"dataset error: {error}", file=out)
        return 2
    except OSError as error:
        print(f"Could not read {in_dir}: {error}", file=out)
        return 1
    print(format_analytics_report(report), file=out, end="")
    if json_path is not None:
        write_analytics_json(report, json_path)
        print(f"json: {json_path}", file=out)
    return 0


def run_replay(
    log_path: Path,
    *,
    verbose: bool = False,
    output: TextIO | None = None,
) -> int:
    """Replay a binary log and print the final result."""
    out = output or sys.stdout
    try:
        data = log_path.read_bytes()
        log = import_log(data)
    except OSError as error:
        print(f"Could not read {log_path}: {error}", file=out)
        return 1
    except ReplayError as error:
        print(f"Invalid log: {error}", file=out)
        return 1

    print(
        f"Replay seed={log.seed} player_count={log.player_count} "
        f"catalog=v{log.catalog_version} actions={len(log.actions)}",
        file=out,
    )

    if verbose:
        for index, action_id in enumerate(log.actions):
            print(f"  {index:4d}  {action_id:3d}  {describe_action(action_id)}", file=out)

    try:
        state = replay_game(log)
    except ValueError as error:
        print(f"Replay failed: {error}", file=out)
        return 1

    done, _ = is_terminal(state)
    print(format_scores(state), file=out)
    print(f"Replay complete (terminal={done}).", file=out)
    return 0 if done else 1


def run_visualize(
    log_path: Path,
    *,
    out_path: Path | None = None,
    window: bool = True,
    step: int | None = None,
    output: TextIO | None = None,
) -> int:
    """Replay a log onto the photographed score sheet, optionally step by step."""
    out = output or sys.stdout
    try:
        require_pillow()
    except RuntimeError as error:
        print(error, file=out)
        return 1

    try:
        data = log_path.read_bytes()
        log = import_log(data)
    except OSError as error:
        print(f"Could not read {log_path}: {error}", file=out)
        return 1
    except ReplayError as error:
        print(f"Invalid log: {error}", file=out)
        return 1

    from doppelt.cli.viewer import build_frames, run_tk_viewer

    try:
        frames = build_frames(log)
    except ValueError as error:
        print(f"Replay failed: {error}", file=out)
        return 1

    last_index = len(frames) - 1
    show_index = last_index if step is None else max(0, min(step, last_index))
    print(
        f"Visualize seed={log.seed} actions={len(log.actions)} frames={len(frames)}",
        file=out,
    )
    print(frames[show_index].caption, file=out)

    if out_path is not None:
        save_overlay(frames[show_index].marks(), out_path, dice=frames[show_index].dice_panel())
        print(f"Saved sheet to {out_path}", file=out)

    if window:
        def on_save(index: int) -> None:
            target = out_path or Path(f"visualize-step-{index}.png")
            save_overlay(frames[index].marks(), target, dice=frames[index].dice_panel())
            print(f"Saved sheet to {target}", file=out)

        try:
            run_tk_viewer(frames, start_at=show_index, on_save=on_save)
        except _tk_errors() as error:
            print(f"Could not open window: {error}", file=out)
            if out_path is None:
                return 1
    return 0


def _tk_errors() -> tuple[type[BaseException], ...]:
    try:
        import tkinter
    except ImportError:
        return (ImportError,)
    return (tkinter.TclError, ImportError)


def run_decode(log_path: Path, *, output: TextIO | None = None) -> int:
    """Pretty-print a binary log without replaying."""
    out = output or sys.stdout
    try:
        data = log_path.read_bytes()
        log = decode_log(data)
    except OSError as error:
        print(f"Could not read {log_path}: {error}", file=out)
        return 1
    except ReplayError as error:
        print(f"Invalid log: {error}", file=out)
        return 1

    print(f"seed={log.seed}", file=out)
    print(f"catalog=v{log.catalog_version} (engine v{CATALOG_VERSION})", file=out)
    print(f"player_count={log.player_count}", file=out)
    print(f"actions={list(log.actions)}", file=out)
    return 0
