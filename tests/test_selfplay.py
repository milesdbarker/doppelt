"""Solo self-play actor-critic (requires torch)."""

from pathlib import Path

import pytest

from doppelt.engine.game import apply_action, legal_action_ids, new_game

torch = pytest.importorskip("torch")


def test_collect_episode_stays_legal_and_finishes():
    from doppelt.ml.model import build_net
    from doppelt.ml.selfplay import collect_episode

    net = build_net(hidden=32, architecture="mlp")
    episode = collect_episode(net, seed=11, max_actions=5_000)
    assert episode.terminal
    assert episode.actions
    assert episode.score >= 0
    state = new_game(seed=11)
    for action_id in episode.actions:
        legal = legal_action_ids(state)
        assert action_id in legal
        apply_action(state, action_id, check_legal=False)


def test_train_selfplay_writes_checkpoint(tmp_path: Path):
    from doppelt.ml.model import load_checkpoint
    from doppelt.ml.selfplay import train_selfplay

    out = tmp_path / "sp.pt"
    result = train_selfplay(
        out_path=out,
        iterations=1,
        games_per_iter=1,
        seed=3,
        eval_games=0,
        hidden=32,
        architecture="mlp",
        lr=1e-3,
    )
    assert out.is_file()
    assert result.games == 1
    assert result.iterations == 1
    net, payload = load_checkpoint(out)
    assert payload["architecture"] == "mlp"
    assert net is not None
