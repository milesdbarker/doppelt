"""Expert iteration: PUCT teacher + distill (requires torch)."""

from pathlib import Path

import pytest

from doppelt.engine.game import legal_action_ids, new_game
from doppelt.ml.bc_data import examples_from_live_games, split_by_seed
from doppelt.ml.train import train_bc

from tests.conftest import roll_hand

torch = pytest.importorskip("torch")


def test_puct_search_visits_sum_to_sims():
    from doppelt.ml.mcts import puct_search
    from doppelt.ml.model import build_net
    from doppelt.ml.policy import NeuralPolicy

    state = new_game(seed=7)
    roll_hand(state)
    legal = legal_action_ids(state)
    assert len(legal) > 1
    net = build_net(hidden=32, architecture="mlp")
    policy = NeuralPolicy(net=net, seed=0, mcts_sims=8)
    logits, _value = policy._forward(state, legal)
    probs = torch.softmax(logits.cpu(), dim=-1).squeeze(0)
    priors = {action_id: float(probs[action_id].item()) for action_id in legal}
    result = puct_search(
        state,
        legal,
        priors,
        policy._leaf_value,
        policy._search_rng,
        n_sims=8,
    )
    assert result.action_id in legal
    assert sum(result.visits) == 8
    assert result.legal == tuple(legal)


def test_train_expert_fine_tunes_init(tmp_path: Path):
    from doppelt.ml.expert import train_expert
    from doppelt.ml.model import load_checkpoint

    examples = examples_from_live_games(1, policy="random_legal", seed_start=5)
    train, val = split_by_seed(examples, val_frac=0.0)
    init = tmp_path / "init.pt"
    train_bc(
        train,
        val,
        out_path=init,
        epochs=1,
        batch_size=32,
        hidden=32,
        architecture="mlp",
        lr=1e-2,
    )
    out = tmp_path / "expert.pt"
    result = train_expert(
        init_checkpoint=init,
        out_path=out,
        rounds=1,
        games_per_round=1,
        mcts_sims=4,
        epochs=1,
        eval_games=0,
        seed=9,
        batch_size=32,
        lr=1e-2,
    )
    assert out.is_file()
    assert result.games == 1
    assert result.mcts_sims == 4
    assert result.last_n_train > 0
    net, payload = load_checkpoint(out)
    assert payload.get("architecture") == "mlp"
    assert net is not None
