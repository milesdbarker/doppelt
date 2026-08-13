"""Baseline analytics over DPLD solo shards (scores, histograms, strategy flags)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from doppelt.actions.catalog_v1 import (
    BONUS_YELLOW_CIRCLE_BASE,
    BONUS_YELLOW_CROSS_BASE,
    CHOOSE_WILD_COLOR_BASE,
    MARK_SILVER_BASE,
    MARK_WHITE_BLUE_ID,
    MARK_WHITE_PINK_ID,
    MARK_WHITE_SILVER_ID,
    PASSIVE_POOL_BASE,
    PASSIVE_PLATTER_BASE,
    PASSIVE_SKIP_ID,
    PICK_DIE_BASE,
    ROLL_HAND_ID,
    pick_die_id,
)
from doppelt.core.types import ALL_DICE, Dice, SCORING_COLORS
from doppelt.sim.batch import AREA_SCORE_KEYS
from doppelt.sim.dataset import DatasetError, DatasetRecord, iter_dataset_records

SILVER_PICK_ID = pick_die_id(Dice.SILVER)
DIE_BY_PICK_ID = {pick_die_id(die): die.value for die in ALL_DICE}
COLOR_KEYS = AREA_SCORE_KEYS[:-1]  # yellow..silver, exclude foxes
WHITE_MODE_IDS = frozenset(range(MARK_WHITE_BLUE_ID, MARK_WHITE_PINK_ID + 1))


@dataclass(frozen=True)
class ScoreSummary:
    n: int
    mean: float
    std: float
    minimum: int
    maximum: int
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p99: float


@dataclass(frozen=True)
class HistogramBin:
    lo: int
    hi: int
    count: int
    share: float


@dataclass
class PolicyAnalytics:
    policy: str
    games: int
    unfinished: int
    total: ScoreSummary
    areas: dict[str, ScoreSummary]
    mean_actions: float
    mean_active_picks: float
    pick_share: dict[str, float]
    silver_pick_rate: float
    silver_first_pick_rate: float
    white_as_silver_rate: float
    passive_skip_rate: float
    mean_silver_marks: float
    mean_yellow_bonus_actions: float
    max_yellow_bonus_streak: int
    yellow_bonus_game_rate: float
    round4_wild_rate: float
    fox_nonzero_rate: float
    mean_fox_score: float
    mean_total_with_fox: float | None
    mean_total_without_fox: float | None
    fox_score_share: float
    silver_highest_color_rate: float
    zero_color_rate: float
    total_histogram: list[HistogramBin]
    area_histograms: dict[str, list[HistogramBin]]
    notes: list[str] = field(default_factory=list)


@dataclass
class DatasetAnalytics:
    dataset: str
    games: int
    policies: dict[str, PolicyAnalytics]
    comparisons: list[str]


def _percentile(sorted_vals: Sequence[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    idx = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _summary(values: Sequence[int]) -> ScoreSummary:
    n = len(values)
    if n == 0:
        return ScoreSummary(0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mean = sum(values) / n
    var = sum((value - mean) ** 2 for value in values) / n
    ordered = sorted(values)
    return ScoreSummary(
        n=n,
        mean=round(mean, 4),
        std=round(math.sqrt(var), 4),
        minimum=ordered[0],
        maximum=ordered[-1],
        p10=round(_percentile(ordered, 10), 2),
        p25=round(_percentile(ordered, 25), 2),
        p50=round(_percentile(ordered, 50), 2),
        p75=round(_percentile(ordered, 75), 2),
        p90=round(_percentile(ordered, 90), 2),
        p99=round(_percentile(ordered, 99), 2),
    )


def _histogram(values: Sequence[int], bin_width: int) -> list[HistogramBin]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    start = (minimum // bin_width) * bin_width
    counts: dict[int, int] = defaultdict(int)
    for value in values:
        counts[(value // bin_width) * bin_width] += 1
    n = len(values)
    bins: list[HistogramBin] = []
    lo = start
    while lo <= maximum:
        count = counts.get(lo, 0)
        if count:
            bins.append(HistogramBin(lo, lo + bin_width, count, round(count / n, 6)))
        lo += bin_width
    return bins


def _is_bonus_yellow(action_id: int) -> bool:
    return (
        BONUS_YELLOW_CIRCLE_BASE <= action_id < BONUS_YELLOW_CIRCLE_BASE + 10
        or BONUS_YELLOW_CROSS_BASE <= action_id < BONUS_YELLOW_CROSS_BASE + 10
    )


def _is_passive_take(action_id: int) -> bool:
    return (
        PASSIVE_PLATTER_BASE <= action_id < PASSIVE_PLATTER_BASE + len(ALL_DICE)
        or PASSIVE_POOL_BASE <= action_id < PASSIVE_POOL_BASE + len(ALL_DICE)
    )


def _scan_actions(actions: Sequence[int]) -> dict[str, float | int]:
    pick_counts = {die.value: 0 for die in ALL_DICE}
    active_picks = 0
    silver_first = 0
    first_picks = 0
    awaiting_first = False
    white_modes = 0
    white_as_silver = 0
    passive_takes = 0
    passive_skips = 0
    silver_marks = 0
    yellow_bonus = 0
    max_bonus_streak = 0
    bonus_streak = 0
    has_yellow_bonus = 0
    round4_wild = 0

    for action_id in actions:
        if action_id == ROLL_HAND_ID:
            awaiting_first = True
            bonus_streak = 0
            continue
        if PICK_DIE_BASE <= action_id < PICK_DIE_BASE + len(ALL_DICE):
            die_name = DIE_BY_PICK_ID[action_id]
            pick_counts[die_name] += 1
            active_picks += 1
            if awaiting_first:
                first_picks += 1
                if action_id == SILVER_PICK_ID:
                    silver_first += 1
                awaiting_first = False
            bonus_streak = 0
            continue
        awaiting_first = False
        if action_id in WHITE_MODE_IDS:
            white_modes += 1
            if action_id == MARK_WHITE_SILVER_ID:
                white_as_silver += 1
        if _is_passive_take(action_id):
            passive_takes += 1
        elif action_id == PASSIVE_SKIP_ID:
            passive_skips += 1
        if MARK_SILVER_BASE <= action_id < MARK_SILVER_BASE + 24:
            silver_marks += 1
        if _is_bonus_yellow(action_id):
            yellow_bonus += 1
            has_yellow_bonus = 1
            bonus_streak += 1
            if bonus_streak > max_bonus_streak:
                max_bonus_streak = bonus_streak
        else:
            bonus_streak = 0
        if CHOOSE_WILD_COLOR_BASE <= action_id < CHOOSE_WILD_COLOR_BASE + len(SCORING_COLORS):
            round4_wild = 1

    return {
        "active_picks": active_picks,
        "pick_counts": pick_counts,
        "silver_first": silver_first,
        "first_picks": first_picks,
        "white_modes": white_modes,
        "white_as_silver": white_as_silver,
        "passive_takes": passive_takes,
        "passive_skips": passive_skips,
        "silver_marks": silver_marks,
        "yellow_bonus": yellow_bonus,
        "max_bonus_streak": max_bonus_streak,
        "has_yellow_bonus": has_yellow_bonus,
        "round4_wild": round4_wild,
    }


def _notes(analytics: PolicyAnalytics) -> list[str]:
    notes: list[str] = []
    if analytics.silver_first_pick_rate >= 0.35:
        notes.append(
            f"silver-first bias: {analytics.silver_first_pick_rate:.1%} of roll-1 picks are silver"
        )
    if analytics.silver_pick_rate >= 0.40:
        notes.append(f"over-picks silver: {analytics.silver_pick_rate:.1%} of active picks")
    for die, share in analytics.pick_share.items():
        if die in {"white", "silver"}:
            continue
        if share >= 0.30:
            notes.append(f"{die}-pick bias: {share:.1%} of active picks")
    if analytics.silver_highest_color_rate >= 0.50:
        notes.append(
            f"silver is the top color in {analytics.silver_highest_color_rate:.1%} of games"
        )
    if analytics.zero_color_rate >= 0.70:
        notes.append(
            f"leaves a color at 0 in {analytics.zero_color_rate:.1%} of games (foxes stay 0)"
        )
    if analytics.fox_nonzero_rate <= 0.05 and analytics.games >= 20:
        notes.append(f"almost never banks foxes ({analytics.fox_nonzero_rate:.1%} of games)")
    if analytics.passive_skip_rate >= 0.70:
        notes.append(f"skips most passive picks ({analytics.passive_skip_rate:.1%})")
    if analytics.white_as_silver_rate >= 0.40:
        notes.append(f"white often played as silver ({analytics.white_as_silver_rate:.1%})")
    return notes


def analyze_records(records: Iterable[DatasetRecord]) -> dict[str, PolicyAnalytics]:
    totals: dict[str, list[int]] = defaultdict(list)
    areas: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    unfinished: dict[str, int] = defaultdict(int)
    action_counts: dict[str, list[int]] = defaultdict(list)
    active_picks: dict[str, int] = defaultdict(int)
    pick_counts: dict[str, dict[str, int]] = defaultdict(lambda: {die.value: 0 for die in ALL_DICE})
    silver_first: dict[str, int] = defaultdict(int)
    first_picks: dict[str, int] = defaultdict(int)
    white_modes: dict[str, int] = defaultdict(int)
    white_as_silver: dict[str, int] = defaultdict(int)
    passive_takes: dict[str, int] = defaultdict(int)
    passive_skips: dict[str, int] = defaultdict(int)
    silver_marks: dict[str, list[int]] = defaultdict(list)
    yellow_bonus: dict[str, list[int]] = defaultdict(list)
    max_bonus_streak: dict[str, int] = defaultdict(int)
    yellow_bonus_games: dict[str, int] = defaultdict(int)
    round4_wild: dict[str, int] = defaultdict(int)
    fox_totals: dict[str, list[int]] = defaultdict(list)
    no_fox_totals: dict[str, list[int]] = defaultdict(list)
    silver_highest: dict[str, int] = defaultdict(int)
    zero_color: dict[str, int] = defaultdict(int)

    for record in records:
        policy = record.policy
        totals[policy].append(record.total_score)
        if not record.terminal:
            unfinished[policy] += 1
        for key in AREA_SCORE_KEYS:
            areas[policy][key].append(int(record.scores.get(key, 0)))
        action_counts[policy].append(len(record.actions))
        scan = _scan_actions(record.actions)
        active_picks[policy] += int(scan["active_picks"])
        for die_name, count in scan["pick_counts"].items():
            pick_counts[policy][die_name] += int(count)
        silver_first[policy] += int(scan["silver_first"])
        first_picks[policy] += int(scan["first_picks"])
        white_modes[policy] += int(scan["white_modes"])
        white_as_silver[policy] += int(scan["white_as_silver"])
        passive_takes[policy] += int(scan["passive_takes"])
        passive_skips[policy] += int(scan["passive_skips"])
        silver_marks[policy].append(int(scan["silver_marks"]))
        yellow_bonus[policy].append(int(scan["yellow_bonus"]))
        max_bonus_streak[policy] = max(max_bonus_streak[policy], int(scan["max_bonus_streak"]))
        yellow_bonus_games[policy] += int(scan["has_yellow_bonus"])
        round4_wild[policy] += int(scan["round4_wild"])
        fox = int(record.scores.get("foxes", 0))
        if fox > 0:
            fox_totals[policy].append(record.total_score)
        else:
            no_fox_totals[policy].append(record.total_score)
        color_scores = [int(record.scores.get(key, 0)) for key in COLOR_KEYS]
        if color_scores and color_scores[-1] >= max(color_scores):
            silver_highest[policy] += 1
        if any(score == 0 for score in color_scores):
            zero_color[policy] += 1

    reports: dict[str, PolicyAnalytics] = {}
    for policy, total_scores in totals.items():
        n = len(total_scores)
        picks = active_picks[policy]
        first = first_picks[policy]
        modes = white_modes[policy]
        passive_n = passive_takes[policy] + passive_skips[policy]
        fox_summary = _summary(areas[policy]["foxes"])
        total_summary = _summary(total_scores)
        analytics = PolicyAnalytics(
            policy=policy,
            games=n,
            unfinished=unfinished[policy],
            total=total_summary,
            areas={key: _summary(areas[policy][key]) for key in AREA_SCORE_KEYS},
            mean_actions=round(sum(action_counts[policy]) / n, 3),
            mean_active_picks=round(picks / n, 3) if n else 0.0,
            pick_share={
                die: round(pick_counts[policy][die] / picks, 4) if picks else 0.0
                for die in DIE_BY_PICK_ID.values()
            },
            silver_pick_rate=round(pick_counts[policy]["silver"] / picks, 4) if picks else 0.0,
            silver_first_pick_rate=round(silver_first[policy] / first, 4) if first else 0.0,
            white_as_silver_rate=round(white_as_silver[policy] / modes, 4) if modes else 0.0,
            passive_skip_rate=round(passive_skips[policy] / passive_n, 4) if passive_n else 0.0,
            mean_silver_marks=round(sum(silver_marks[policy]) / n, 3),
            mean_yellow_bonus_actions=round(sum(yellow_bonus[policy]) / n, 3),
            max_yellow_bonus_streak=max_bonus_streak[policy],
            yellow_bonus_game_rate=round(yellow_bonus_games[policy] / n, 4),
            round4_wild_rate=round(round4_wild[policy] / n, 4),
            fox_nonzero_rate=round(len(fox_totals[policy]) / n, 4),
            mean_fox_score=fox_summary.mean,
            mean_total_with_fox=(
                round(sum(fox_totals[policy]) / len(fox_totals[policy]), 2)
                if fox_totals[policy]
                else None
            ),
            mean_total_without_fox=(
                round(sum(no_fox_totals[policy]) / len(no_fox_totals[policy]), 2)
                if no_fox_totals[policy]
                else None
            ),
            fox_score_share=round(fox_summary.mean / total_summary.mean, 4)
            if total_summary.mean
            else 0.0,
            silver_highest_color_rate=round(silver_highest[policy] / n, 4),
            zero_color_rate=round(zero_color[policy] / n, 4),
            total_histogram=_histogram(total_scores, 10),
            area_histograms={
                key: _histogram(areas[policy][key], 5 if key != "foxes" else 5)
                for key in AREA_SCORE_KEYS
            },
        )
        analytics.notes = _notes(analytics)
        reports[policy] = analytics
    return reports


def _comparisons(policies: dict[str, PolicyAnalytics]) -> list[str]:
    lines: list[str] = []
    names = [name for name in ("random_legal", "greedy_immediate", "heuristic") if name in policies]
    if "random_legal" in policies and "heuristic" in policies:
        delta = policies["heuristic"].total.mean - policies["random_legal"].total.mean
        lines.append(
            f"heuristic vs random_legal: {policies['heuristic'].total.mean:.1f} vs "
            f"{policies['random_legal'].total.mean:.1f} ({delta:+.1f})"
        )
    if "random_legal" in policies and "greedy_immediate" in policies:
        delta = policies["greedy_immediate"].total.mean - policies["random_legal"].total.mean
        lines.append(
            f"greedy_immediate vs random_legal: {policies['greedy_immediate'].total.mean:.1f} vs "
            f"{policies['random_legal'].total.mean:.1f} ({delta:+.1f})"
        )
    if "greedy_immediate" in policies and "heuristic" in policies:
        delta = policies["heuristic"].total.mean - policies["greedy_immediate"].total.mean
        lines.append(
            f"heuristic vs greedy_immediate: {policies['heuristic'].total.mean:.1f} vs "
            f"{policies['greedy_immediate'].total.mean:.1f} ({delta:+.1f})"
        )
    if len(names) >= 2:
        fox_bits = ", ".join(f"{name} {policies[name].fox_nonzero_rate:.1%}" for name in names)
        lines.append(f"fox nonzero rate: {fox_bits}")
        silver_bits = ", ".join(f"{name} {policies[name].silver_pick_rate:.1%}" for name in names)
        lines.append(f"silver active-pick share: {silver_bits}")
    return lines


def analyze_dataset(out_dir: Path) -> DatasetAnalytics:
    out_dir = Path(out_dir)
    policies = analyze_records(iter_dataset_records(out_dir))
    if not policies:
        raise DatasetError(f"no records in {out_dir}")
    games = sum(report.games for report in policies.values())
    return DatasetAnalytics(
        dataset=str(out_dir),
        games=games,
        policies=policies,
        comparisons=_comparisons(policies),
    )


def analytics_to_json(report: DatasetAnalytics) -> dict:
    return {
        "dataset": report.dataset,
        "games": report.games,
        "comparisons": report.comparisons,
        "policies": {name: asdict(analytics) for name, analytics in report.policies.items()},
    }


def _fmt_hist(bins: Sequence[HistogramBin], *, min_share: float = 0.005) -> list[str]:
    if not bins:
        return ["    (empty)"]
    shown = [row for row in bins if row.share >= min_share]
    if not shown:
        shown = [bins[0], bins[-1]] if len(bins) > 1 else list(bins)
    width = max(len(f"{row.lo}-{row.hi - 1}") for row in shown)
    lines = []
    for row in shown:
        bar = "#" * max(1, round(row.share * 40)) if row.share > 0 else ""
        label = f"{row.lo}-{row.hi - 1}".ljust(width)
        lines.append(f"    {label}  {row.share:6.1%}  {bar}")
    omitted = len(bins) - len(shown)
    if omitted:
        lines.append(f"    ({omitted} sparse bins omitted)")
    return lines


def format_analytics_report(report: DatasetAnalytics) -> str:
    lines = [
        f"Dataset analytics: {report.dataset}",
        f"  games: {report.games}",
        "",
    ]
    if report.comparisons:
        lines.append("Comparisons")
        for item in report.comparisons:
            lines.append(f"  {item}")
        lines.append("")
    for name in ("heuristic", "greedy_immediate", "random_legal"):
        if name not in report.policies:
            continue
        analytics = report.policies[name]
        total = analytics.total
        lines.extend(
            [
                f"{analytics.policy}  n={analytics.games}  unfinished={analytics.unfinished}",
                f"  score  mean={total.mean:.1f}  std={total.std:.1f}  "
                f"p10={total.p10:.0f} p25={total.p25:.0f} p50={total.p50:.0f} "
                f"p75={total.p75:.0f} p90={total.p90:.0f} p99={total.p99:.0f}  "
                f"min={total.minimum} max={total.maximum}",
                "  colors "
                + "  ".join(
                    f"{key}={analytics.areas[key].mean:.1f}" for key in AREA_SCORE_KEYS
                ),
                f"  actions/game={analytics.mean_actions:.1f}  "
                f"active picks={analytics.mean_active_picks:.1f}  "
                f"silver marks={analytics.mean_silver_marks:.1f}",
                "  pick share "
                + "  ".join(f"{die}={share:.1%}" for die, share in analytics.pick_share.items()),
                f"  silver pick={analytics.silver_pick_rate:.1%}  "
                f"silver-first={analytics.silver_first_pick_rate:.1%}  "
                f"white-as-silver={analytics.white_as_silver_rate:.1%}  "
                f"passive skip={analytics.passive_skip_rate:.1%}",
                f"  yellow bonus acts={analytics.mean_yellow_bonus_actions:.2f}  "
                f"bonus games={analytics.yellow_bonus_game_rate:.1%}  "
                f"max streak={analytics.max_yellow_bonus_streak}  "
                f"round4 wild={analytics.round4_wild_rate:.1%}",
                f"  fox nonzero={analytics.fox_nonzero_rate:.1%}  "
                f"mean fox={analytics.mean_fox_score:.1f}  "
                f"share of total={analytics.fox_score_share:.1%}  "
                f"mean total fox/no-fox="
                f"{analytics.mean_total_with_fox}/{analytics.mean_total_without_fox}",
                f"  silver highest color={analytics.silver_highest_color_rate:.1%}  "
                f"any color zero={analytics.zero_color_rate:.1%}",
                "  total score histogram (bin 10):",
                *_fmt_hist(analytics.total_histogram),
            ]
        )
        if analytics.notes:
            lines.append("  notes:")
            for note in analytics.notes:
                lines.append(f"    - {note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_analytics_json(report: DatasetAnalytics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(analytics_to_json(report), indent=2) + "\n", encoding="utf-8")
