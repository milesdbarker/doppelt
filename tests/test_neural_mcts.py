"""Neural PUCT search (requires torch)."""

from tests.conftest import roll_hand

import pytest

from doppelt.core.types import ActionTrack
from doppelt.engine.game import legal_action_ids, new_game

torch = pytest.importorskip("torch")


def test_puct_select_is_legal_and_does_not_mutate_state():
    from doppelt.ml.model import build_net
    from doppelt.ml.policy import NeuralPolicy

    state = new_game(seed=4)
    state.sheet.circle_action(ActionTrack.REROLL)
    roll_hand(state)
    faces_before = dict(state.faces)
    log_before = list(state.action_log)
    legal = legal_action_ids(state)
    assert len(legal) > 1

    net = build_net(hidden=32, architecture="mlp")
    policy = NeuralPolicy(net=net, seed=0, mcts_sims=8)
    choice = policy.select(state, legal)
    assert choice in legal
    assert dict(state.faces) == faces_before
    assert state.action_log == log_before


def test_puct_passthrough_single_legal():
    from doppelt.ml.model import build_net
    from doppelt.ml.policy import NeuralPolicy

    net = build_net(hidden=32, architecture="mlp")
    policy = NeuralPolicy(net=net, seed=1, mcts_sims=8)
    state = new_game(seed=1)
    assert policy.select(state, [130]) == 130
