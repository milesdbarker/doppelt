"""CLI command tests."""

import io
from pathlib import Path

import pytest

from doppelt.actions.catalog_v1 import ROLL_HAND_ID, describe_action
from doppelt.cli.commands import run_decode, run_play, run_random, run_replay
from doppelt.cli.main import main
from doppelt.engine.game import play_random_game
from doppelt.replay import export_log


def test_describe_action_roll_hand():
    assert describe_action(ROLL_HAND_ID) == "roll hand"


def test_main_requires_subcommand():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_random_command_prints_total():
    buf = io.StringIO()
    code = run_random(seed=42, output=buf)
    assert code == 0
    assert "total:" in buf.getvalue()


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
