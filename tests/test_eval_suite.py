"""Frozen eval suites (3.6.16) and failure flags (3.6.15)."""

import pytest

from doppelt.actions.catalog_v1 import ROLL_HAND_ID, pick_die_id
from doppelt.cli.main import main
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.types import ActionTrack, Dice
from doppelt.ml.eval import _summarize, eval_report_to_json, format_eval_report
from doppelt.ml.eval_suite import (
    EVAL_SUITES,
    SCORE_GOAL,
    custom_eval_suite,
    get_eval_suite,
    resolve_eval_suite,
)
from doppelt.sim.failures import (
    PINK_FIRST_SIX_SLOT,
    flags_from_sheet,
    summarize_failures,
)


def test_suites_are_disjoint_except_report_superset():
    report = EVAL_SUITES["report"]
    holdout = EVAL_SUITES["holdout"]
    dev = EVAL_SUITES["dev"]
    assert report.games == 256
    assert holdout.games == 256
    assert EVAL_SUITES["report_1k"].seed_start == report.seed_start
    assert EVAL_SUITES["report_1k"].games == 1000
    report_end = report.seed_start + report.games
    holdout_end = holdout.seed_start + holdout.games
    assert report_end <= holdout.seed_start or holdout_end <= report.seed_start
    assert not report.retune
    assert not holdout.retune
    assert dev.retune
    assert dev.seed_start == 20_000


def test_resolve_eval_suite_truncation_and_custom():
    full = resolve_eval_suite("report")
    assert full.games == 256
    assert full.seed_start == 1_000_000
    truncated = resolve_eval_suite("report", games=8)
    assert truncated.games == 8
    assert truncated.retune
    assert "truncated" in truncated.purpose
    custom = resolve_eval_suite("custom", games=4, seed=99)
    assert custom.name == "custom"
    assert custom.seed_start == 99
    with pytest.raises(ValueError, match="omit --seed"):
        resolve_eval_suite("report", seed=10_000)
    with pytest.raises(ValueError, match="custom eval"):
        resolve_eval_suite("custom")


def test_eval_row_percentiles_and_goal_share():
    row = _summarize("net", [200, 250, 300, 350], unfinished=0, goal=SCORE_GOAL)
    assert row.mean_score == 275.0
    assert row.median_score == 275.0
    assert row.p10 == pytest.approx(215.0)
    assert row.share_ge_goal == 0.5
    assert row.minimum == 200
    assert row.maximum == 350


def test_failure_flags_cover_roadmap_cases():
    silver = pick_die_id(Dice.SILVER)
    yellow = pick_die_id(Dice.YELLOW)
    pink = [None] * 12
    dead = flags_from_sheet(
        seed=1,
        total_score=180,
        scores={"yellow": 10, "blue": 10, "pink": 10, "green": 10, "silver": 0, "foxes": 0},
        fox_count=2,
        pink=pink,
        plus_one_circled=1,
        actions=(ROLL_HAND_ID, silver),
    )
    assert dead.zero_color
    assert dead.silver_first
    assert dead.skipped_pink_6
    assert dead.unused_plus_one
    assert dead.fox_zero_four_colors_alive
    assert dead.fox_killed_by_zero_color
    pink[PINK_FIRST_SIX_SLOT] = 6
    healthy = flags_from_sheet(
        seed=2,
        total_score=320,
        scores={"yellow": 40, "blue": 40, "pink": 40, "green": 40, "silver": 40, "foxes": 40},
        fox_count=2,
        pink=pink,
        plus_one_circled=0,
        actions=(ROLL_HAND_ID, yellow),
    )
    assert not healthy.zero_color
    assert not healthy.silver_first
    assert not healthy.skipped_pink_6
    assert not healthy.unused_plus_one
    assert not healthy.fox_zero_four_colors_alive
    assert not healthy.fox_killed_by_zero_color
    summary = summarize_failures([dead, healthy])
    assert summary.zero_color_rate == 0.5
    assert summary.skipped_pink_6_rate == 0.5
    assert summary.share_ge_goal == 0.5
    assert any("zero-color" in note for note in summary.notes)
    assert not any("silver-first" in note for note in summary.notes)


def test_flags_from_empty_sheet_defaults():
    sheet = PlayerSheet.empty()
    flags = flags_from_sheet(
        seed=0,
        total_score=0,
        scores={"yellow": 0, "blue": 0, "pink": 0, "green": 0, "silver": 0, "foxes": 0},
        fox_count=sheet.foxes,
        pink=sheet.pink,
        plus_one_circled=sheet.action_tracks[ActionTrack.PLUS_ONE].circled,
        actions=(),
    )
    assert flags.skipped_pink_6
    assert flags.zero_color
    assert not flags.unused_plus_one


def test_custom_suite_json_roundtrip():
    suite = custom_eval_suite(seed_start=3, games=2)
    from doppelt.ml.eval import EvalReport, EvalRow
    from doppelt.sim.failures import FailureSummary

    report = EvalReport(
        suite=suite,
        score_goal=300,
        rows=(
            EvalRow("net", 2, 100.0, 100.0, 0.0, 100.0, 100.0, 100.0, 0.0, 0),
        ),
        failures={
            "net": FailureSummary(2, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, notes=["zero-color"]),
        },
    )
    text = format_eval_report(report)
    assert "suite=custom" in text
    assert "DO NOT retune" not in text
    payload = eval_report_to_json(report)
    assert payload["suite"]["games"] == 2
    assert payload["failures"]["net"]["zero_color_rate"] == 1.0


def test_main_eval_help_lists_suite(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["eval", "--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    assert "--suite" in text
    assert "report" in text
    assert "--no-failures" in text
    assert "--json" in text


def test_get_eval_suite_unknown():
    with pytest.raises(ValueError, match="unknown eval suite"):
        get_eval_suite("prod")
