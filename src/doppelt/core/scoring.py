"""End-game scoring helpers."""

from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.score_sheet import get_score_sheet
from doppelt.core.silver import SILVER_ROW_COLORS


def score_yellow(sheet: PlayerSheet) -> int:
    count = sheet.yellow_cross_count()
    if count == 0:
        return 0
    table = get_score_sheet().yellow.score_by_cross_count
    return table[count - 1]


def score_blue(sheet: PlayerSheet) -> int:
    for index in range(len(sheet.blue) - 1, -1, -1):
        if sheet.blue[index] is not None:
            return get_score_sheet().blue.score_by_last_slot[index]
    return 0


def score_pink(sheet: PlayerSheet) -> int:
    return sum(value for value in sheet.pink if value is not None)


def score_green(sheet: PlayerSheet) -> int:
    return sum(star for star in sheet.green_stars if star is not None)


def score_silver(sheet: PlayerSheet) -> int:
    table = get_score_sheet().silver.row_score_by_mark_count
    total = 0
    for row in SILVER_ROW_COLORS:
        count = sheet.silver_row_mark_count(row)
        if count > 0:
            total += table[count - 1]
    return total


def _color_area_scores(sheet: PlayerSheet) -> list[int]:
    return [
        score_yellow(sheet),
        score_blue(sheet),
        score_pink(sheet),
        score_green(sheet),
        score_silver(sheet),
    ]


def score_foxes_from_colors(fox_count: int, color_scores: list[int]) -> int:
    """Each fox scores the lowest color total; 0 if any color scores 0."""
    if fox_count == 0:
        return 0
    if any(score == 0 for score in color_scores):
        return 0
    return fox_count * min(color_scores)


def score_foxes(sheet: PlayerSheet) -> int:
    """Each fox scores the lowest color total; 0 if any color scores 0."""
    return score_foxes_from_colors(sheet.foxes, _color_area_scores(sheet))


def score_sheet_areas(sheet: PlayerSheet) -> dict[str, int]:
    yellow = score_yellow(sheet)
    blue = score_blue(sheet)
    pink = score_pink(sheet)
    green = score_green(sheet)
    silver = score_silver(sheet)
    return {
        "yellow": yellow,
        "blue": blue,
        "pink": pink,
        "green": green,
        "silver": silver,
        "foxes": score_foxes_from_colors(sheet.foxes, [yellow, blue, pink, green, silver]),
    }


def total_score(sheet: PlayerSheet) -> int:
    return sum(score_sheet_areas(sheet).values())
