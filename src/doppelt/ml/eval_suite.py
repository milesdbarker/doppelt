"""Frozen eval seed suites for the 300-point claim (3.6.16).

``report`` is the only suite that may be used to claim mean 300. Do not retune
hyperparameters, pick checkpoints, or iterate encodings against it. Use ``dev``
for overnight / self-play selection. Look at ``holdout`` once after you would
otherwise ship a report number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SCORE_GOAL = 300


@dataclass(frozen=True)
class EvalSuite:
    name: str
    seed_start: int
    games: int
    retune: bool
    purpose: str


# Seeds sit far from train (0…) and legacy eval (10_000 / 20_000).
EVAL_SUITES: dict[str, EvalSuite] = {
    "dev": EvalSuite(
        name="dev",
        seed_start=20_000,
        games=64,
        retune=True,
        purpose="checkpoint selection and overnight; not a 300 claim",
    ),
    "report": EvalSuite(
        name="report",
        seed_start=1_000_000,
        games=256,
        retune=False,
        purpose="frozen 300-claim suite; do not retune on these seeds",
    ),
    "holdout": EvalSuite(
        name="holdout",
        seed_start=2_000_000,
        games=256,
        retune=False,
        purpose="second freeze; inspect after report, never during tuning",
    ),
    "report_1k": EvalSuite(
        name="report_1k",
        seed_start=1_000_000,
        games=1_000,
        retune=False,
        purpose="same start as report, 1000 games (superset; still not for retune)",
    ),
}

DEFAULT_EVAL_SUITE = "report"
SUITE_NAMES = tuple(EVAL_SUITES)


def get_eval_suite(name: str) -> EvalSuite:
    key = name.strip().lower()
    if key not in EVAL_SUITES:
        allowed = ", ".join(SUITE_NAMES)
        raise ValueError(f"unknown eval suite {name!r}; expected one of {allowed}")
    return EVAL_SUITES[key]


def custom_eval_suite(*, seed_start: int, games: int) -> EvalSuite:
    if games < 1:
        raise ValueError("games must be at least 1")
    return EvalSuite(
        name="custom",
        seed_start=seed_start,
        games=games,
        retune=True,
        purpose="ad-hoc seed range; not a frozen 300 claim",
    )


def resolve_eval_suite(
    suite: str | None,
    *,
    games: int | None = None,
    seed: int | None = None,
) -> EvalSuite:
    """Named suite, or a custom range if ``suite`` is omitted / ``custom``."""
    if suite is None or suite.strip().lower() == "custom":
        if games is None or seed is None:
            raise ValueError("custom eval requires both --games and --seed")
        return custom_eval_suite(seed_start=seed, games=games)
    resolved = get_eval_suite(suite)
    if games is not None and games != resolved.games:
        if games < 1:
            raise ValueError("games must be at least 1")
        return EvalSuite(
            name=resolved.name,
            seed_start=resolved.seed_start,
            games=games,
            retune=True,
            purpose=f"truncated {resolved.name} ({games}/{resolved.games} games); not a 300 claim",
        )
    if seed is not None and seed != resolved.seed_start:
        raise ValueError(
            f"suite {resolved.name!r} uses seed_start={resolved.seed_start}; "
            "omit --seed or use --suite custom"
        )
    return resolved


def suite_to_json(suite: EvalSuite) -> dict:
    payload = asdict(suite)
    payload["score_goal"] = SCORE_GOAL
    return payload
