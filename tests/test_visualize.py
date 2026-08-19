"""Score-sheet visualization from replay logs."""

from pathlib import Path

from PIL import Image

from doppelt.cli.commands import run_visualize
from doppelt.cli.overlay import board_image_path, load_board_layout, marks_from_sheet
from doppelt.cli.render import render_overlay
from doppelt.cli.viewer import build_frames
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.types import ActionTrack, Color
from doppelt.engine.game import play_random_game
from doppelt.replay import export_log, import_log


def test_layout_covers_sheet_cells():
    layout = load_board_layout()
    assert board_image_path().is_file()
    assert len(layout["yellow_cells"]) == 10
    assert len(layout["silver"]["xs"]) == 6
    assert len(layout["blue"]["xs"]) == 12
    assert len(layout["green"]["xs"]) == 12
    assert len(layout["pink"]["xs"]) == 12
    for track in ("reroll", "unlock", "plus_one"):
        assert len(layout["action_tracks"][track]["xs"]) == 6


def test_marks_match_requested_overlay_rules():
    sheet = PlayerSheet.empty()
    sheet.circle_action(ActionTrack.REROLL)
    sheet.use_action(ActionTrack.REROLL)
    sheet.circle_action(ActionTrack.PLUS_ONE)
    sheet.mark_silver(3, Color.YELLOW)
    sheet.yellow[2].circled = True
    sheet.yellow[0].circled = True
    sheet.yellow[0].crossed = True
    sheet.mark_blue(8)
    sheet.mark_green_die(5)  # slot 0 multiplier 2 → 10
    sheet.mark_pink(4)
    sheet.claimed_bonuses.add("blue:1")

    marks = marks_from_sheet(sheet, round_index=1, include_scores=False)
    by_key = {}
    for mark in marks:
        by_key.setdefault(mark.key, []).append(mark.kind)

    assert by_key["track:reroll:0"] == ["cross"]
    assert by_key["track:plus_one:0"] == ["circle"]
    assert by_key["silver:yellow:3"] == ["cross"]
    assert by_key["yellow:2"] == ["circle"]
    assert by_key["yellow:0"] == ["circle", "cross"]
    assert by_key["blue:0"] == ["text"]
    assert by_key["green:0"] == ["text"]
    assert by_key["pink:0"] == ["text"]
    assert by_key["blue:1"] == ["cross"]

    texts = {mark.key: mark.text for mark in marks if mark.kind == "text"}
    assert texts["blue:0"] == "8"
    assert texts["green:0"] == "10"
    assert texts["pink:0"] == "4"


def test_dice_panel_shows_rolled_and_platter():
    from tests.conftest import roll_hand

    from doppelt.actions.catalog_v1 import pick_die_id
    from doppelt.cli.overlay import dice_panel_from_state
    from doppelt.core.types import Dice
    from doppelt.engine.game import apply_action, new_game

    state = new_game(seed=2)
    panel = dice_panel_from_state(state)
    assert len(panel.rolled) == 6
    assert panel.platter == ()
    assert all(die.face is None for die in panel.rolled)
    assert [die.badge for die in panel.picked] == ["1", "2", "3"]
    assert all(die.color == "empty" for die in panel.picked)

    roll_hand(state)
    panel = dice_panel_from_state(state)
    assert len(panel.rolled) == 6
    assert all(die.face in range(1, 7) for die in panel.rolled)

    for die in Dice:
        state.faces[die] = 6
    state.faces[Dice.YELLOW] = 5
    state.faces[Dice.WHITE] = 2
    state.faces[Dice.BLUE] = 3
    apply_action(state, pick_die_id(Dice.YELLOW))
    panel = dice_panel_from_state(
        state,
        previous_hand=list(Dice),
        previous_platter=[],
        previous_slots=[None, None, None],
    )
    assert "yellow" not in {die.color for die in panel.rolled}
    platter_colors = {die.color for die in panel.platter}
    assert platter_colors == {"white", "blue"}
    assert all(die.highlight for die in panel.platter)
    assert panel.picked[0].color == "yellow"
    assert panel.picked[0].badge == "1"
    assert panel.picked[0].highlight
    assert panel.picked[1].color == "empty"
    assert panel.picked[2].color == "empty"


def test_dice_panel_passive_shows_non_platter_dice():
    from doppelt.cli.overlay import dice_panel_from_state
    from doppelt.engine.game import begin_passive_turn, new_game

    state = new_game(seed=4)
    begin_passive_turn(state)
    panel = dice_panel_from_state(state)
    assert len(panel.platter) == 3
    assert len(panel.picked) == 3
    assert all(die.badge is None for die in panel.picked)
    assert {die.color for die in panel.picked}.isdisjoint({die.color for die in panel.platter})
    assert len(panel.picked) + len(panel.platter) == 6
    assert panel.rolled == ()


def test_passive_platter_pick_stays_on_platter_in_viewer():
    from doppelt.actions.catalog_v1 import passive_platter_id
    from doppelt.cli.overlay import dice_panel_from_state
    from doppelt.core.types import ActionTrack, Dice
    from doppelt.engine.game import apply_action, begin_passive_turn, new_game

    state = new_game(seed=22)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    begin_passive_turn(state)
    state.faces[Dice.PINK] = 4
    state.platter = [Dice.PINK, Dice.GREEN, Dice.BLUE]
    state.passive_pool = [Dice.WHITE, Dice.YELLOW, Dice.SILVER]
    state.use_pool_fallback = False
    apply_action(state, passive_platter_id(Dice.PINK))

    panel = dice_panel_from_state(state)
    assert {die.color for die in panel.platter} == {"pink", "green", "blue"}


def test_rolled_hand_not_hidden_by_stale_passive_flag():
    from doppelt.cli.overlay import dice_panel_from_parts
    from doppelt.core.phases import Phase
    from doppelt.core.types import Dice

    panel = dice_panel_from_parts(
        hand=list(Dice),
        platter=[],
        faces={die: 4 for die in Dice},
        awaiting_roll=False,
        phase=Phase.ACTIVE_PICK,
        plus_one_after_passive=True,
        passive_pool=[],
    )
    assert len(panel.rolled) == 6
    assert all(die.face == 4 for die in panel.rolled)
    assert [die.badge for die in panel.picked] == ["1", "2", "3"]


def test_later_round_active_frames_still_show_rolled_hand():
    from doppelt.core.phases import Phase

    state = play_random_game(seed=21, max_actions=400)
    frames = build_frames(import_log(export_log(state)))
    round_two = [
        frame
        for frame in frames
        if frame.round_index >= 2 and frame.phase is Phase.ACTIVE_PICK
    ]
    assert round_two
    panel = round_two[0].dice_panel()
    assert len(panel.rolled) == 6


def test_plus_one_keeps_active_dice_before_passive_roll():
    from doppelt.core.phases import Phase

    found = None
    for seed in range(1, 80):
        state = play_random_game(seed=seed, max_actions=500)
        frames = build_frames(import_log(export_log(state)))
        for index, frame in enumerate(frames[:-1]):
            nxt = frames[index + 1]
            if frame.note != "plus-one applied (before passive roll)":
                continue
            if nxt.note != "passive dice rolled":
                continue
            found = (frame, nxt)
            break
        if found is not None:
            break

    assert found is not None
    before_passive, after_roll = found
    assert before_passive.phase in {Phase.PLUS_ONE, Phase.PLUS_ONE_MARK_YELLOW}
    assert after_roll.phase is Phase.PASSIVE_PICK
    assert [die.badge for die in before_passive.dice_panel().picked] == ["1", "2", "3"]
    assert len(after_roll.dice_panel().platter) == 3
    assert len(after_roll.dice_panel().picked) == 3
    assert before_passive.faces != after_roll.faces or before_passive.platter != after_roll.platter



def test_render_overlay_writes_png(tmp_path: Path):
    sheet = PlayerSheet.empty()
    sheet.mark_pink(6)
    image = render_overlay(marks_from_sheet(sheet, include_scores=True, round_index=1))
    assert image.size == (768, 1024)
    path = tmp_path / "sheet.png"
    image.save(path)
    loaded = Image.open(path)
    assert loaded.size == (768, 1024)


def test_render_includes_dice_panel_width():
    from doppelt.cli.overlay import ShownDie, DicePanelState, load_board_layout

    dice = DicePanelState(
        rolled=(ShownDie("yellow", 4), ShownDie("blue", 2)),
        platter=(ShownDie("silver", 1),),
        picked=(ShownDie("green", 5, badge="1"),),
    )
    image = render_overlay([], dice=dice)
    panel_w = load_board_layout()["dice_panel"]["width"]
    assert image.size == (768 + panel_w, 1024)


def test_visualize_command_saves_final_png(tmp_path: Path, capsys):
    state = play_random_game(seed=21, max_actions=400)
    log_path = tmp_path / "game.bin"
    out_path = tmp_path / "final.png"
    log_path.write_bytes(export_log(state))

    code = run_visualize(log_path, out_path=out_path, window=False)
    assert code == 0
    assert out_path.is_file()
    captured = capsys.readouterr().out
    assert "Visualize seed=21" in captured
    assert "Saved sheet" in captured
    assert Image.open(out_path).size[1] == 1024
    assert Image.open(out_path).size[0] > 768


def test_build_frames_starts_before_first_action():
    state = play_random_game(seed=8, max_actions=80)
    log = import_log(export_log(state))
    frames = build_frames(log)
    assert frames[0].action_id is None
    assert frames[0].index == 0
    assert len(frames) >= len(log.actions) + 1
    assert frames[-1].index == len(frames) - 1


def test_main_visualize_help(capsys):
    from doppelt.cli.main import main
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["visualize", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--out" in text
    assert "--no-window" in text
