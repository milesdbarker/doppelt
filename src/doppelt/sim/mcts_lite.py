"""Single-player MCTS-lite: small UCB search with cheap rollouts.

Mark-choice and bonus phases keep Heuristic (already 1–2 ply / low branch).
On active pick only, run a handful of simulations: apply the root action,
sample dice rolls, play a short random rollout, backup ``evaluate_state``.
"""

from __future__ import annotations

import math
import random

from doppelt.actions.catalog_v1 import roll_hand_id
from doppelt.core.phases import Phase
from doppelt.core.scoring import total_score
from doppelt.core.state import GameState
from doppelt.engine.game import apply_action, legal_action_ids
from doppelt.sim.heuristic import Heuristic, evaluate_state
from doppelt.sim.policy import POLICY_RNG_XOR

SEARCH_PHASES = frozenset({Phase.ACTIVE_PICK})
MAX_SEARCH_BRANCH = 10

DEFAULT_SIMS = 32
DEFAULT_HORIZON = 4
UCB_C = 1.25
SCORE_SCALE = 200.0


def _fork_trial(state: GameState, rng: random.Random) -> GameState:
    trial = state.copy_for_trial()
    trial.rng.seed(rng.randrange(1 << 30))
    return trial


def _ucb(total_visits: int, visits: int, value_sum: float) -> float:
    if visits == 0:
        return float("inf")
    return value_sum / visits / SCORE_SCALE + UCB_C * math.sqrt(
        math.log(total_visits + 1) / visits
    )


def _safe_apply(trial: GameState, action_id: int) -> bool:
    try:
        apply_action(trial, action_id, check_legal=False)
    except ValueError:
        return False
    return True


class MctsLite:
    """Shallow UCB search on active picks; Heuristic everywhere else."""

    name = "mcts_lite"

    def __init__(
        self,
        seed: int,
        *,
        n_sims: int = DEFAULT_SIMS,
        horizon: int = DEFAULT_HORIZON,
    ) -> None:
        if n_sims < 1:
            raise ValueError("n_sims must be at least 1")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        self._rng = random.Random(seed ^ POLICY_RNG_XOR)
        self._heuristic = Heuristic(seed)
        self.n_sims = n_sims
        self.horizon = horizon

    def select(self, state: GameState, legal: list[int]) -> int:
        if not legal:
            raise ValueError("MctsLite received no legal actions")
        if len(legal) == 1:
            return legal[0]
        if state.phase not in SEARCH_PHASES or len(legal) > MAX_SEARCH_BRANCH:
            return self._heuristic.select(state, legal)
        return self._search(state, legal)

    def _search(self, state: GameState, legal: list[int]) -> int:
        visits = [0] * len(legal)
        values = [0.0] * len(legal)
        total_visits = 0

        for _ in range(self.n_sims):
            index = max(
                range(len(legal)),
                key=lambda i: (
                    _ucb(total_visits, visits[i], values[i]),
                    self._rng.random(),
                ),
            )
            trial = _fork_trial(state, self._rng)
            if not _safe_apply(trial, legal[index]):
                visits[index] += 1
                values[index] += evaluate_state(state)
                total_visits += 1
                continue
            score = self._rollout(trial)
            visits[index] += 1
            values[index] += score
            total_visits += 1

        best_visits = max(visits)
        candidates = [i for i, count in enumerate(visits) if count == best_visits]
        if len(candidates) > 1:
            best_mean = max(values[i] / visits[i] for i in candidates)
            candidates = [i for i in candidates if values[i] / visits[i] >= best_mean - 1e-9]
        return legal[self._rng.choice(candidates)]

    def _rollout(self, trial: GameState) -> float:
        for _ in range(self.horizon):
            if trial.phase is Phase.GAME_OVER:
                return float(total_score(trial.sheet))
            if trial.phase is Phase.ACTIVE_PICK and trial.awaiting_roll and trial.hand:
                if not _safe_apply(trial, roll_hand_id()):
                    break
                continue
            legal = legal_action_ids(trial)
            if not legal:
                break
            action_id = legal[0] if len(legal) == 1 else self._rng.choice(legal)
            if not _safe_apply(trial, action_id):
                break
        if trial.phase is Phase.GAME_OVER:
            return float(total_score(trial.sheet))
        return evaluate_state(trial)
