"""CLI command tests."""

import io
from pathlib import Path

import pytest

from doppelt.actions.catalog_v1 import ROLL_HAND_ID, describe_action
from doppelt.cli.commands import run_decode, run_play, run_random, run_replay, run_simulate
from doppelt.cli.main import main
from doppelt.engine.game import play_random_game
from doppelt.replay import export_log


def test_describe_action_roll_hand():
    assert describe_action(ROLL_HAND_ID) == "roll hand"


def test_describe_action_passive_skip():
    from doppelt.actions.catalog_v1 import passive_skip_id

    assert describe_action(passive_skip_id()) == "skip (don't take a die)"


def test_main_requires_subcommand():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_random_command_prints_total():
    buf = io.StringIO()
    code = run_random(seed=42, output=buf)
    assert code == 0
    text = buf.getvalue()
    assert "total:" in text
    assert "random_legal solo game" in text


def test_random_command_heuristic_policy():
    buf = io.StringIO()
    code = run_random(seed=42, policy="heuristic", output=buf)
    assert code == 0
    text = buf.getvalue()
    assert "heuristic solo game" in text
    assert "total:" in text


def test_random_command_greedy_alias(tmp_path: Path):
    log_path = tmp_path / "greedy.bin"
    buf = io.StringIO()
    code = run_random(seed=11, policy="greedy", save_path=log_path, output=buf)
    assert code == 0
    assert "greedy_immediate solo game" in buf.getvalue()
    assert log_path.is_file()


def test_random_save_and_replay(tmp_path: Path, capsys):
    log_path = tmp_path / "game.bin"
    code = run_random(seed=99, save_path=log_path)
    assert code == 0
    assert log_path.is_file()

    code = run_replay(log_path)
    assert code == 0
    captured = capsys.readouterr()
    assert "Replay complete" in captured.out
    assert "total:" in captured.out


def test_decode_log(tmp_path: Path, capsys):
    state = play_random_game(seed=7, max_actions=100)
    log_path = tmp_path / "game.bin"
    log_path.write_bytes(export_log(state))

    code = run_decode(log_path)
    assert code == 0
    captured = capsys.readouterr()
    assert "seed=7" in captured.out
    assert "actions=" in captured.out


def test_replay_missing_file(tmp_path: Path, capsys):
    code = run_replay(tmp_path / "missing.bin")
    assert code == 1
    assert "Could not read" in capsys.readouterr().out


def test_play_quit_early(capsys):
    inputs = iter(["q"])
    code = run_play(seed=1, input_fn=lambda _: next(inputs), output=io.StringIO())
    assert code == 0


def test_active_pick_number_in_status():
    from tests.conftest import roll_hand

    from doppelt.cli.display import format_status
    from doppelt.engine.game import new_game

    state = new_game(seed=2)
    assert "Active pick: 1 of 3" in format_status(state)
    roll_hand(state)
    assert "Active pick: 1 of 3" in format_status(state)
    state.picks_made = 1
    assert "Active pick: 2 of 3" in format_status(state)
    state.picks_made = 2
    assert "Active pick: 3 of 3" in format_status(state)


def test_hand_status_shows_face_values():
    from tests.conftest import roll_hand

    from doppelt.cli.display import format_status
    from doppelt.core.types import Dice
    from doppelt.engine.game import new_game

    state = new_game(seed=2)
    roll_hand(state)
    state.faces[Dice.YELLOW] = 1
    state.faces[Dice.BLUE] = 6
    text = format_status(state)
    assert "Hand:" in text
    assert "yellow=1" in text
    assert "blue=6" in text


def test_silver_pending_lists_cascade_marks():
    from tests.conftest import roll_hand

    from doppelt.actions.catalog_v1 import pick_die_id
    from doppelt.cli.display import format_silver_pending
    from doppelt.core.types import ALL_DICE, Dice
    from doppelt.engine.game import apply_action, new_game

    state = new_game(seed=16)
    roll_hand(state)
    for die in ALL_DICE:
        state.faces[die] = 6
    state.faces[Dice.SILVER] = 4
    state.faces[Dice.YELLOW] = 2
    apply_action(state, pick_die_id(Dice.SILVER))
    text = format_silver_pending(state)
    assert text is not None
    assert "primary 4" in text
    assert "platter yellow die 2" in text

    from doppelt.cli.display import format_silver_grid
    from doppelt.core.player_sheet import PlayerSheet
    from doppelt.core.types import Color

    sheet = PlayerSheet.empty()
    sheet.mark_silver(3, Color.YELLOW)
    sheet.mark_silver(3, Color.BLUE)
    sheet.mark_silver(5, Color.PINK)
    text = format_silver_grid(sheet)
    assert "X = marked" in text
    assert "yellow=3" in text
    assert "blue=3" in text
    assert "pink=5" in text
    assert "green=" not in text.split("marked:")[1]



def test_active_pick_actions_sorted_by_face_value():
    from tests.conftest import roll_hand

    from doppelt.actions.catalog_v1 import ActionKind, decode_action, pick_die_id
    from doppelt.cli.display import sort_actions_for_display
    from doppelt.core.types import Dice
    from doppelt.engine.game import legal_action_ids, new_game

    state = new_game(seed=2)
    roll_hand(state)
    state.faces[Dice.PINK] = 2
    state.faces[Dice.GREEN] = 5
    state.faces[Dice.YELLOW] = 3
    state.faces[Dice.BLUE] = 4
    state.faces[Dice.WHITE] = 6
    state.faces[Dice.SILVER] = 1

    legal = sort_actions_for_display(state, legal_action_ids(state))
    pick_faces = [
        state.faces[decode_action(action_id).die]
        for action_id in legal
        if decode_action(action_id).kind is ActionKind.PICK_DIE
        and decode_action(action_id).die is not None
    ]
    assert pick_faces == sorted(pick_faces)
    assert pick_die_id(Dice.SILVER) in legal



def test_play_auto_selects_when_only_one_action():
    buf = io.StringIO()
    inputs = iter(["q"])

    def input_fn(_prompt: str) -> str:
        return next(inputs, "q")

    code = run_play(seed=3, input_fn=input_fn, output=buf)
    assert code == 0
    assert "Auto: id=130: roll hand" in buf.getvalue()


def test_play_one_action_then_quit():
    inputs = iter(["130", "q"])

    def fake_input(_prompt: str) -> str:
        return next(inputs)

    code = run_play(seed=2, input_fn=fake_input, output=io.StringIO())
    assert code == 0


def test_main_random_subcommand(capsys):
    code = main(["random", "--seed", "5"])
    assert code == 0


def test_main_random_policy_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["random", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--policy" in text
    assert "heuristic" in text
    assert "mcts-lite" in text


def test_main_random_rejects_unknown_policy():
    with pytest.raises(SystemExit) as exc:
        main(["random", "--policy", "not-a-bot"])
    assert exc.value.code == 2


def test_simulate_command_reports_throughput():
    buf = io.StringIO()
    code = run_simulate(games=4, seed=3, workers=1, policy="random_legal", output=buf)
    assert code == 0
    text = buf.getvalue()
    assert "4 random_legal games" in text
    assert "games/s" in text
    assert "unfinished:   0" in text


def test_main_train_bc_requires_source():
    from doppelt.cli.commands import run_train_bc

    buf = io.StringIO()
    code = run_train_bc(
        in_dir=None,
        live_games=0,
        policy="heuristic",
        max_games=1,
        seed=0,
        epochs=1,
        batch_size=8,
        hidden=32,
        architecture="mlp",
        lr=1e-3,
        val_frac=0.1,
        out_path=Path("data/models/unused.pt"),
        output=buf,
    )
    assert code == 2
    assert "live-games" in buf.getvalue()


def test_main_eval_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "checkpoint" in text
    assert "--mcts-sims" in text


def test_main_train_bc_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train", "bc", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--live-games" in text
    assert "--in" in text
    assert "--arch" in text


def test_main_train_selfplay_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train", "selfplay", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--init" in text
    assert "--iters" in text
    assert "--kl" in text


def test_main_dataset_generate_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["dataset", "generate", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "mcts_lite" not in text or "no mcts_lite" in text
    assert "--mix" in text
