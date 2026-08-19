"""PUCT search with a real tree and sampled chance rolls (3.6.2).

Play-time eval and expert-iteration labels use this. A2C self-play does not.

Each simulation copies the state, walks decision nodes with PUCT, samples
``roll_hand`` as a chance node (never as a forced unique), and may take a
short reply ply before the value leaf.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from doppelt.actions.catalog_v1 import ROLL_HAND_ID, USE_REROLL_ID
from doppelt.core.phases import Phase
from doppelt.core.scoring import total_score
from doppelt.core.state import GameState
from doppelt.engine.game import apply_action, legal_action_ids
from doppelt.ml.encoding import SCORE_SCALE

DEFAULT_MCTS_SIMS = 32
DEFAULT_MAX_PLIES = 2
C_PUCT = 1.5
MAX_FORCED = 16
CHANCE_ACTION_IDS = frozenset({ROLL_HAND_ID, USE_REROLL_ID})


@dataclass(frozen=True)
class PuctResult:
    """Root PUCT outcome: visit-greedy action plus raw visit counts (3.6.1 / 3.6.21)."""

    action_id: int
    legal: tuple[int, ...]
    visits: tuple[int, ...]


class _Edge:
    __slots__ = ("prior", "visits", "value_sum", "chance", "decision")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.chance: _ChanceNode | None = None
        self.decision: _DecisionNode | None = None


class _ChanceNode:
    __slots__ = ("outcomes",)

    def __init__(self) -> None:
        self.outcomes: dict[tuple, _DecisionNode] = {}


class _DecisionNode:
    __slots__ = ("edges", "expanded")

    def __init__(self) -> None:
        self.edges: dict[int, _Edge] = {}
        self.expanded = False


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
    """Collapse forced marks/bonuses. Do not auto-apply rolls (those are chance)."""
    for _ in range(MAX_FORCED):
        if trial.phase is Phase.GAME_OVER:
            return
        legal = legal_action_ids(trial)
        if len(legal) != 1 or legal[0] in CHANCE_ACTION_IDS:
            return
        if not _safe_apply(trial, legal[0]):
            return


def _only_roll_legal(trial: GameState) -> bool:
    if trial.phase is Phase.GAME_OVER:
        return False
    legal = legal_action_ids(trial)
    return legal == [ROLL_HAND_ID]


def _chance_key(trial: GameState) -> tuple:
    hand = tuple(sorted((die.value, trial.faces.get(die, 0)) for die in trial.hand))
    platter = tuple(die.value for die in trial.platter)
    return (trial.round_index, trial.picks_made, hand, platter)


def _sample_chance_roll(trial: GameState) -> None:
    if not _only_roll_legal(trial):
        return
    trial.ensure_private_rng()
    _safe_apply(trial, ROLL_HAND_ID)
    _apply_forced_uniques(trial)


def _terminal_value(trial: GameState, score_scale: float) -> float:
    return total_score(trial.sheet) / score_scale


def _expand(node: _DecisionNode, legal: list[int], priors: dict[int, float]) -> None:
    prior_list = [max(float(priors.get(action_id, 0.0)), 1e-8) for action_id in legal]
    total = sum(prior_list) or 1.0
    for action_id, prior in zip(legal, prior_list):
        node.edges[action_id] = _Edge(prior / total)
    node.expanded = True


def _select_edge(node: _DecisionNode, rng: random.Random, c_puct: float) -> tuple[int, _Edge]:
    total_visits = sum(edge.visits for edge in node.edges.values())
    action_id = max(
        node.edges,
        key=lambda aid: (
            _puct(
                total_visits,
                node.edges[aid].visits,
                node.edges[aid].value_sum,
                node.edges[aid].prior,
                c_puct,
            ),
            rng.random(),
        ),
    )
    return action_id, node.edges[action_id]


def _child_decision(edge: _Edge, trial: GameState, action_id: int) -> _DecisionNode:
    if action_id in CHANCE_ACTION_IDS or _only_roll_legal(trial):
        if _only_roll_legal(trial):
            _sample_chance_roll(trial)
        if edge.chance is None:
            edge.chance = _ChanceNode()
        key = _chance_key(trial)
        child = edge.chance.outcomes.get(key)
        if child is None:
            child = _DecisionNode()
            edge.chance.outcomes[key] = child
        return child
    if edge.decision is None:
        edge.decision = _DecisionNode()
    return edge.decision


def _leaf_value(trial: GameState, infer, evaluate_leaf, score_scale: float) -> float:
    if trial.phase is Phase.GAME_OVER:
        return _terminal_value(trial, score_scale)
    _sample_chance_roll(trial)
    if trial.phase is Phase.GAME_OVER:
        return _terminal_value(trial, score_scale)
    legal = legal_action_ids(trial)
    if not legal:
        return _terminal_value(trial, score_scale)
    if infer is not None:
        _priors, value = infer(trial, legal)
        return float(value)
    return float(evaluate_leaf(trial))


def _simulate(
    node: _DecisionNode,
    trial: GameState,
    *,
    plies_left: int,
    infer,
    evaluate_leaf,
    rng: random.Random,
    c_puct: float,
    score_scale: float,
) -> float:
    if trial.phase is Phase.GAME_OVER:
        return _terminal_value(trial, score_scale)
    _apply_forced_uniques(trial)
    if trial.phase is Phase.GAME_OVER:
        return _terminal_value(trial, score_scale)

    legal = legal_action_ids(trial)
    if not legal:
        return _terminal_value(trial, score_scale)

    if not node.expanded:
        if infer is not None:
            priors, value = infer(trial, legal)
        else:
            priors = {action_id: 1.0 / len(legal) for action_id in legal}
            value = float(evaluate_leaf(trial))
        _expand(node, legal, priors)
        return float(value)

    action_id, edge = _select_edge(node, rng, c_puct)
    if not _safe_apply(trial, action_id):
        value = 0.0
        edge.visits += 1
        edge.value_sum += value
        return value
    _apply_forced_uniques(trial)

    if trial.phase is Phase.GAME_OVER or plies_left <= 1:
        value = _leaf_value(trial, infer, evaluate_leaf, score_scale)
    else:
        child = _child_decision(edge, trial, action_id)
        value = _simulate(
            child,
            trial,
            plies_left=plies_left - 1,
            infer=infer,
            evaluate_leaf=evaluate_leaf,
            rng=rng,
            c_puct=c_puct,
            score_scale=score_scale,
        )
    edge.visits += 1
    edge.value_sum += value
    return value


def puct_search(
    state: GameState,
    legal: list[int],
    priors: dict[int, float],
    evaluate_leaf,
    rng: random.Random,
    *,
    n_sims: int,
    c_puct: float = C_PUCT,
    sample: bool = False,
    max_plies: int = DEFAULT_MAX_PLIES,
    infer=None,
    score_scale: float = SCORE_SCALE,
) -> PuctResult:
    """Tree PUCT. ``sample`` draws from visit counts; otherwise visit-greedy.

    ``max_plies`` counts player decisions in a simulation (1 = root action then
    leaf, after sampling any pending roll). Default 2 is root + one reply.
    ``infer(state, legal) -> (priors, value)`` is used when expanding inner
    nodes; root uses ``priors`` / ``evaluate_leaf`` if ``infer`` is omitted.
    """
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    if max_plies < 1:
        raise ValueError("max_plies must be at least 1")
    if len(legal) == 1:
        return PuctResult(legal[0], (legal[0],), (n_sims,))

    root = _DecisionNode()
    _expand(root, legal, priors)

    for _ in range(n_sims):
        trial = _fork_trial(state, rng)
        _simulate(
            root,
            trial,
            plies_left=max_plies,
            infer=infer,
            evaluate_leaf=evaluate_leaf,
            rng=rng,
            c_puct=c_puct,
            score_scale=score_scale,
        )

    visits = tuple(root.edges[action_id].visits for action_id in legal)
    if sample:
        index = rng.choices(range(len(legal)), weights=visits, k=1)[0]
    else:
        best = max(visits)
        candidates = [i for i, count in enumerate(visits) if count == best]
        index = rng.choice(candidates)
    return PuctResult(legal[index], tuple(legal), visits)


def puct_select(
    state: GameState,
    legal: list[int],
    priors: dict[int, float],
    evaluate_leaf,
    rng: random.Random,
    *,
    n_sims: int,
    c_puct: float = C_PUCT,
    sample: bool = False,
    max_plies: int = DEFAULT_MAX_PLIES,
    infer=None,
    score_scale: float = SCORE_SCALE,
) -> int:
    """Choose a legal catalog ID by most-visited PUCT child at the root."""
    return puct_search(
        state,
        legal,
        priors,
        evaluate_leaf,
        rng,
        n_sims=n_sims,
        c_puct=c_puct,
        sample=sample,
        max_plies=max_plies,
        infer=infer,
        score_scale=score_scale,
    ).action_id
