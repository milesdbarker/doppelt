"""End-game scoring: color areas, fox valuation, and totals."""

from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.scoring import score_sheet_areas, total_score
from doppelt.core.types import Color
from doppelt.engine.game import is_terminal, new_game


def _sheet_with_all_color_scores() -> PlayerSheet:
    sheet = PlayerSheet.empty()
    sheet.mark_yellow(2, 1)
    sheet.mark_yellow(2, 1)  # 1 cross → 3 points
    sheet.blue[0] = 5  # last slot 0 → 1 point
    sheet.pink[0] = 2  # sum → 2 points
    sheet.mark_green_die(5)
    sheet.mark_green_die(1)  # pair star → 3 points
    sheet.mark_silver(1, Color.YELLOW)  # 1 mark → 2 points
    return sheet


def test_foxes_score_lowest_color_times_fox_count():
    sheet = _sheet_with_all_color_scores()
    sheet.foxes = 2
    areas = score_sheet_areas(sheet)
    assert areas["yellow"] == 3
    assert areas["blue"] == 1
    assert areas["pink"] == 2
    assert areas["green"] == 3
    assert areas["silver"] == 2
    assert areas["foxes"] == 2  # min color is 1 (blue) × 2 foxes


def test_foxes_score_zero_when_any_color_is_zero():
    sheet = _sheet_with_all_color_scores()
    sheet.foxes = 3
    assert score_sheet_areas(sheet)["foxes"] == 3

    sheet = PlayerSheet.empty()
    sheet.blue[0] = 5
    sheet.pink[0] = 2
    sheet.mark_green_die(5)
    sheet.mark_green_die(1)
    sheet.mark_silver(1, Color.YELLOW)
    sheet.foxes = 2
    assert score_sheet_areas(sheet)["foxes"] == 0


def test_foxes_score_zero_without_foxes():
    sheet = _sheet_with_all_color_scores()
    assert score_sheet_areas(sheet)["foxes"] == 0


def test_total_score_includes_fox_points():
    sheet = _sheet_with_all_color_scores()
    sheet.foxes = 2
    areas = score_sheet_areas(sheet)
    assert total_score(sheet) == sum(areas.values())
    assert total_score(sheet) == 3 + 1 + 2 + 3 + 2 + 2


def test_is_terminal_reports_fox_breakdown_at_game_over():
    state = new_game(seed=1)
    state.sheet = _sheet_with_all_color_scores()
    state.sheet.foxes = 2
    state.phase = Phase.GAME_OVER

    done, scores = is_terminal(state)
    assert done
    assert scores["foxes"] == 2
    assert total_score(state.sheet) == sum(scores.values())


def test_is_terminal_not_done_before_game_over():
    state = new_game(seed=2)
    state.sheet.foxes = 1
    done, scores = is_terminal(state)
    assert not done
    assert scores["foxes"] == 0  # yellow still 0 at new game
