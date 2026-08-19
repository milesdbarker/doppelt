"""Failure mining on finished solo games (3.6.15).

Flags answer whether a ~268 mean is one systematic bug or mixed tactics:
zero-color (foxes die), skipped first pink-6 slot, leftover plus-one circles,
and fox score 0 while four colors still score.

Silver-first is tracked as a tactic, not a failure: taking silver on pick 1
is strong when two unlocks are available.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

from doppelt.actions.catalog_v1 import (
    PICK_DIE_BASE,
    ROLL_HAND_ID,
    pick_die_id,
)
from doppelt.core.scoring import score_sheet_areas
from doppelt.core.state import GameState
from doppelt.core.types import ActionTrack, Dice
from doppelt.sim.batch import AREA_SCORE_KEYS

COLOR_KEYS = AREA_SCORE_KEYS[:-1]
SILVER_PICK_ID = pick_die_id(Dice.SILVER)
N_DICE = 6
# First pink slot whose min_value is 6 (0-based); see score_sheet_v1.yaml.
PINK_FIRST_SIX_SLOT = 6
SCORE_GOAL = 300


@dataclass(frozen=True)
class GameFailureFlags:
    seed: int
    total_score: int
    zero_color: bool
    silver_first: bool
    skipped_pink_6: bool
    unused_plus_one: bool
    fox_zero_four_colors_alive: bool
    fox_killed_by_zero_color: bool
    n_nonzero_colors: int
    fox_count: int
    fox_score: int


@dataclass
class FailureSummary:
    games: int
    zero_color_rate: float
    silver_first_rate: float
    skipped_pink_6_rate: float
    unused_plus_one_rate: float
    fox_zero_four_colors_alive_rate: float
    fox_killed_by_zero_color_rate: float
    share_ge_goal: float
    notes: list[str] = field(default_factory=list)


def _rate(count: int, n: int) -> float:
    return round(count / n, 4) if n else 0.0


def silver_first_in_actions(actions: Sequence[int]) -> bool:
    """True if any post-roll first pick is silver."""
    awaiting_first = False
    for action_id in actions:
        if action_id == ROLL_HAND_ID:
            awaiting_first = True
            continue
        if PICK_DIE_BASE <= action_id < PICK_DIE_BASE + N_DICE:
            if awaiting_first and action_id == SILVER_PICK_ID:
                return True
            awaiting_first = False
            continue
        awaiting_first = False
    return False


def flags_from_sheet(
    *,
    seed: int,
    total_score: int,
    scores: dict[str, int],
    fox_count: int,
    pink: Sequence[int | None],
    plus_one_circled: int,
    actions: Sequence[int],
) -> GameFailureFlags:
    color_scores = [int(scores.get(key, 0)) for key in COLOR_KEYS]
    n_nonzero = sum(1 for value in color_scores if value > 0)
    fox_score = int(scores.get("foxes", 0))
    skipped_pink_6 = True
    if len(pink) > PINK_FIRST_SIX_SLOT:
        skipped_pink_6 = pink[PINK_FIRST_SIX_SLOT] is None
    return GameFailureFlags(
        seed=seed,
        total_score=total_score,
        zero_color=any(value == 0 for value in color_scores),
        silver_first=silver_first_in_actions(actions),
        skipped_pink_6=skipped_pink_6,
        unused_plus_one=plus_one_circled > 0,
        fox_zero_four_colors_alive=fox_score == 0 and n_nonzero >= 4,
        fox_killed_by_zero_color=fox_count > 0 and fox_score == 0,
        n_nonzero_colors=n_nonzero,
        fox_count=fox_count,
        fox_score=fox_score,
    )


def flags_from_state(state: GameState, *, seed: int | None = None) -> GameFailureFlags:
    scores = score_sheet_areas(state.sheet)
    total = sum(scores.values())
    plus_one = state.sheet.action_tracks[ActionTrack.PLUS_ONE]
    return flags_from_sheet(
        seed=state.seed if seed is None else seed,
        total_score=total,
        scores=scores,
        fox_count=state.sheet.foxes,
        pink=state.sheet.pink,
        plus_one_circled=plus_one.circled,
        actions=state.action_log,
    )


def summarize_failures(flags: Iterable[GameFailureFlags], *, goal: int = SCORE_GOAL) -> FailureSummary:
    rows = list(flags)
    n = len(rows)
    if n == 0:
        return FailureSummary(
            games=0,
            zero_color_rate=0.0,
            silver_first_rate=0.0,
            skipped_pink_6_rate=0.0,
            unused_plus_one_rate=0.0,
            fox_zero_four_colors_alive_rate=0.0,
            fox_killed_by_zero_color_rate=0.0,
            share_ge_goal=0.0,
        )
    summary = FailureSummary(
        games=n,
        zero_color_rate=_rate(sum(1 for row in rows if row.zero_color), n),
        silver_first_rate=_rate(sum(1 for row in rows if row.silver_first), n),
        skipped_pink_6_rate=_rate(sum(1 for row in rows if row.skipped_pink_6), n),
        unused_plus_one_rate=_rate(sum(1 for row in rows if row.unused_plus_one), n),
        fox_zero_four_colors_alive_rate=_rate(
            sum(1 for row in rows if row.fox_zero_four_colors_alive), n
        ),
        fox_killed_by_zero_color_rate=_rate(
            sum(1 for row in rows if row.fox_killed_by_zero_color), n
        ),
        share_ge_goal=_rate(sum(1 for row in rows if row.total_score >= goal), n),
    )
    summary.notes = failure_notes(summary)
    return summary


def failure_notes(summary: FailureSummary) -> list[str]:
    notes: list[str] = []
    if summary.zero_color_rate >= 0.25:
        notes.append(
            f"zero-color in {summary.zero_color_rate:.1%} of games (foxes stay 0)"
        )
    if summary.skipped_pink_6_rate >= 0.50:
        notes.append(
            f"never reached first pink-6 slot in {summary.skipped_pink_6_rate:.1%}"
        )
    if summary.unused_plus_one_rate >= 0.20:
        notes.append(
            f"leftover plus-one circles in {summary.unused_plus_one_rate:.1%} of games"
        )
    if summary.fox_zero_four_colors_alive_rate >= 0.20:
        notes.append(
            f"fox score 0 with ≥4 colors alive in {summary.fox_zero_four_colors_alive_rate:.1%}"
        )
    if summary.fox_killed_by_zero_color_rate >= 0.10:
        notes.append(
            f"banked foxes then zeroed them in {summary.fox_killed_by_zero_color_rate:.1%}"
        )
    return notes


def format_failure_summary(summary: FailureSummary, *, indent: str = "  ") -> str:
    lines = [
        f"{indent}failures  zero-color={summary.zero_color_rate:.1%}  "
        f"skip-pink-6={summary.skipped_pink_6_rate:.1%}  "
        f"unused-plus-one={summary.unused_plus_one_rate:.1%}",
        f"{indent}          fox=0 with 4+ colors={summary.fox_zero_four_colors_alive_rate:.1%}  "
        f"foxes killed by 0-color={summary.fox_killed_by_zero_color_rate:.1%}  "
        f"%≥{SCORE_GOAL}={summary.share_ge_goal:.1%}",
        f"{indent}tactics   silver-first={summary.silver_first_rate:.1%}  "
        f"(not a failure; strong when two unlocks are available)",
    ]
    if summary.notes:
        for note in summary.notes:
            lines.append(f"{indent}          - {note}")
    return "\n".join(lines)


def failure_summary_to_json(summary: FailureSummary) -> dict:
    return asdict(summary)
