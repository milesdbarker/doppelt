"""Neural policy/value net and BC training (requires torch)."""

from pathlib import Path

import pytest

from doppelt.engine.game import legal_action_ids, new_game
from doppelt.ml.bc_data import examples_from_live_games, split_by_seed
from doppelt.ml.encoding import FEATURE_SIZE, encode_state

torch = pytest.importorskip("torch")


def test_forward_masks_illegal_actions():
    from doppelt.ml.model import build_net

    state = new_game(seed=5)
    encoded = encode_state(state)
    net = build_net(hidden=32, architecture="pvn_v1")
    net.eval()
    features = torch.tensor([encoded.features], dtype=torch.float32)
    mask = torch.tensor([encoded.mask], dtype=torch.bool)
    logits, value = net(features, mask)
    assert logits.shape == (1, len(encoded.mask))
    assert value.shape == (1,)
    illegal = ~mask
    assert torch.isneginf(logits[illegal]).all()
    legal = mask
    assert torch.isfinite(logits[legal]).all()


def test_default_architecture_param_count():
    from doppelt.ml.model import (
        DEFAULT_ARCHITECTURE,
        MAX_PARAM_TARGET,
        MIN_PARAM_TARGET,
        build_net,
        count_parameters,
    )

    net = build_net()
    n_params = count_parameters(net)
    assert net.architecture == DEFAULT_ARCHITECTURE
    assert MIN_PARAM_TARGET <= n_params <= MAX_PARAM_TARGET


def test_bc_overfit_and_legal_select(tmp_path: Path):
    from doppelt.ml.model import load_checkpoint
    from doppelt.ml.policy import NeuralPolicy
    from doppelt.ml.train import train_bc

    examples = examples_from_live_games(2, policy="random_legal", seed_start=30)
    assert examples
    train, _val = split_by_seed(examples, val_frac=0.0)
    out = tmp_path / "bc.pt"
    result = train_bc(train, [], out_path=out, epochs=4, batch_size=32, hidden=64, lr=1e-2)
    assert out.is_file()
    assert result.n_train == len(train)
    net, payload = load_checkpoint(out)
    assert payload["feature_size"] == FEATURE_SIZE
    assert net is not None

    policy = NeuralPolicy(out, seed=30)
    state = new_game(seed=30)
    legal = legal_action_ids(state)
    chosen = policy.select(state, legal)
    assert chosen in legal
