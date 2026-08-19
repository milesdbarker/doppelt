"""PUCT search using a neural policy prior and value leaf (play-time only).

Self-play / BC training does not call this. Each simulation copies the state,
applies one root action (plus forced unique follow-ups), then scores the leaf
with the value head — or true total if the trial ended.
"""

from __future__ import annotations

import math
import random

from doppelt.core.phases import Phase
from doppelt.core.scoring import total_score
from doppelt.core.state import GameState
from doppelt.engine.game import apply_action, legal_action_ids
from doppelt.ml.encoding import SCORE_SCALE

DEFAULT_MCTS_SIMS = 32
C_PUCT = 1.5
MAX_FORCED = 16


def _fork_trial(state: GameState, rng: random.Random) -> GameState:
    trial = state.copy_for_trial()
    trial.ensure_private_rng()
    trial.rng.seed(rng.randrange(1 << 30))
    return trial


def _puct(total_visits: int, visits: int, value_sum: float, prior: float, c_puct: float) -> float:
    q = 0.0 if visits == 0 else value_sum / visits
    return q + c_puct * prior * math.sqrt(total_visits + 1e-8) / (1 + visits)


def _safe_apply(trial: GameState, action_id: int) -> bool:
    try:
        apply_action(trial, action_id, check_legal=False)
    except ValueError:
        return False
    return True


def _apply_forced_uniques(trial: GameState) -> None:
    for _ in range(MAX_FORCED):
        if trial.phase is Phase.GAME_OVER:
            return
        legal = legal_action_ids(trial)
        if len(legal) != 1:
            return
        if not _safe_apply(trial, legal[0]):
            return


def puct_select(
    state: GameState,
    legal: list[int],
    priors: dict[int, float],
    evaluate_leaf,
    rng: random.Random,
    *,
    n_sims: int,
    c_puct: float = C_PUCT,
) -> int:
    """Choose a legal catalog ID by most-visited PUCT child at the root."""
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    if len(legal) == 1:
        return legal[0]

    visits = [0] * len(legal)
    values = [0.0] * len(legal)
    prior_list = [max(float(priors.get(action_id, 0.0)), 1e-8) for action_id in legal]
    prior_sum = sum(prior_list)
    prior_list = [p / prior_sum for p in prior_list]
    total_visits = 0

    for _ in range(n_sims):
        index = max(
            range(len(legal)),
            key=lambda i: (
                _puct(total_visits, visits[i], values[i], prior_list[i], c_puct),
                rng.random(),
            ),
        )
        trial = _fork_trial(state, rng)
        leaf = 0.0
        if _safe_apply(trial, legal[index]):
            _apply_forced_uniques(trial)
            if trial.phase is Phase.GAME_OVER:
                leaf = total_score(trial.sheet) / SCORE_SCALE
            else:
                leaf = float(evaluate_leaf(trial))
        visits[index] += 1
        values[index] += leaf
        total_visits += 1

    best = max(visits)
    candidates = [i for i, count in enumerate(visits) if count == best]
    return legal[rng.choice(candidates)]
