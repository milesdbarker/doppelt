"""Dataset analytics: score summaries, histograms, and strategy flags."""

import io
from pathlib import Path

from doppelt.actions.catalog_v1 import (
    MARK_WHITE_SILVER_ID,
    PASSIVE_SKIP_ID,
    ROLL_HAND_ID,
    pick_die_id,
)
from doppelt.cli.commands import run_dataset_analyze
from doppelt.core.types import Dice
from doppelt.sim.analytics import (
    DatasetAnalytics,
    analyze_dataset,
    analyze_records,
    format_analytics_report,
)
from doppelt.sim.dataset import DatasetRecord, generate_dataset


def test_analyze_records_percentiles_and_silver_flag():
    silver = pick_die_id(Dice.SILVER)
    yellow = pick_die_id(Dice.YELLOW)
    records = [
        DatasetRecord(
            seed=1,
            policy="heuristic",
            terminal=True,
            player_count=1,
            total_score=120,
            scores={"yellow": 10, "blue": 20, "pink": 15, "green": 25, "silver": 40, "foxes": 10},
            actions=(ROLL_HAND_ID, silver, MARK_WHITE_SILVER_ID),
        ),
        DatasetRecord(
            seed=2,
            policy="heuristic",
            terminal=True,
            player_count=1,
            total_score=80,
            scores={"yellow": 0, "blue": 10, "pink": 10, "green": 10, "silver": 50, "foxes": 0},
            actions=(ROLL_HAND_ID, silver, PASSIVE_SKIP_ID),
        ),
        DatasetRecord(
            seed=3,
            policy="random_legal",
            terminal=True,
            player_count=1,
            total_score=40,
            scores={"yellow": 8, "blue": 8, "pink": 8, "green": 8, "silver": 8, "foxes": 0},
            actions=(ROLL_HAND_ID, yellow),
        ),
    ]
    reports = analyze_records(records)
    heuristic = reports["heuristic"]
    assert heuristic.games == 2
    assert heuristic.total.p50 == 100
    assert heuristic.total.minimum == 80
    assert heuristic.silver_first_pick_rate == 1.0
    assert heuristic.silver_highest_color_rate == 1.0
    assert heuristic.zero_color_rate == 0.5
    assert heuristic.fox_nonzero_rate == 0.5
    assert any("silver-first" in note for note in heuristic.notes)
    random = reports["random_legal"]
    assert random.silver_pick_rate == 0.0
    assert random.pick_share["yellow"] == 1.0
    text = format_analytics_report(
        DatasetAnalytics(
            dataset="test",
            games=3,
            policies=reports,
            comparisons=["heuristic vs random_legal: 100.0 vs 40.0 (+60.0)"],
        )
    )
    assert "heuristic vs random_legal" in text
    assert "total score histogram" in text


def test_analyze_small_generated_dataset(tmp_path: Path):
    out_dir = tmp_path / "solo"
    generate_dataset(
        out_dir,
        n_games=6,
        mix="heuristic:1,random_legal:1",
        shard_size=3,
        seed_start=40,
        workers=1,
    )
    report = analyze_dataset(out_dir)
    assert report.games == 6
    assert set(report.policies) == {"heuristic", "random_legal"}
    for analytics in report.policies.values():
        assert analytics.games == 3
        assert analytics.total.n == 3
        assert 0 <= analytics.silver_pick_rate <= 1
        assert analytics.mean_actions > 0
        assert analytics.total_histogram
        assert "yellow" in analytics.area_histograms
    assert any("heuristic vs random_legal" in line for line in report.comparisons)


def test_dataset_analyze_cli(tmp_path: Path):
    out_dir = tmp_path / "cli"
    generate_dataset(
        out_dir,
        n_games=3,
        mix="random_legal:1",
        shard_size=3,
        seed_start=70,
        workers=1,
    )
    json_path = tmp_path / "report.json"
    buf = io.StringIO()
    code = run_dataset_analyze(in_dir=out_dir, json_path=json_path, output=buf)
    assert code == 0
    text = buf.getvalue()
    assert "random_legal" in text
    assert "mean=" in text
    assert json_path.is_file()
    assert "json:" in text
