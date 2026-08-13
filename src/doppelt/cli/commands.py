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
from doppelt.core.types import Dice
from doppelt.engine.game import (
    apply_action,
    is_terminal,
    legal_action_ids,
    new_game,
    play_random_game,
)
from doppelt.replay import ReplayError, decode_log, export_log, import_log, replay_game
from doppelt.sim import format_batch_result, resolve_worker_count, run_batch


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
    output: TextIO | None = None,
) -> int:
    """Play a random legal solo game (bot uses a separate RNG from dice)."""
    out = output or sys.stdout
    state = play_random_game(seed=seed, max_actions=max_actions)
    done, _ = is_terminal(state)

    print(f"Random solo game (seed={seed})", file=out)
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
