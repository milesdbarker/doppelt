"""encoding_v1: fixed features + catalog-aligned legal mask."""

from doppelt.actions.catalog_v1 import ACTION_SPACE_SIZE
from doppelt.core.phases import Phase
from doppelt.engine.game import apply_action, legal_action_ids, new_game, play_random_game
from doppelt.ml.encoding import (
    ENCODING_VERSION,
    FEATURE_SIZE,
    EncodedState,
    encode_features,
    encode_state,
    legal_mask,
)
from doppelt.ml.bc_data import BCExample, examples_from_record, split_by_seed
from doppelt.sim.batch import play_with_policy
from doppelt.sim.dataset import DatasetRecord
from doppelt.sim.random_legal import RandomLegal


def test_infer_architecture_from_state_keys():
    from doppelt.ml.model import infer_architecture

    assert infer_architecture({"backbone.0.weight": None}) == "mlp"
    assert infer_architecture({"sheet_encoder.0.weight": None}) == "pvn_v1"


def test_encoding_size_and_version():
    from doppelt.ml.encoding import (
        DICE_SIZE,
        FEATURE_SIZE,
        GLOBAL_SIZE,
        PENDING_SIZE,
        SHEET_SIZE,
    )

    state = new_game(seed=1)
    encoded = encode_state(state)
    assert encoded.encoding_version == ENCODING_VERSION
    assert SHEET_SIZE + DICE_SIZE + GLOBAL_SIZE + PENDING_SIZE == FEATURE_SIZE
    assert len(encoded.features) == FEATURE_SIZE
    assert len(encoded.mask) == ACTION_SPACE_SIZE
    assert all(0.0 <= value <= 1.0 for value in encoded.features)


def test_legal_mask_matches_engine():
    state = new_game(seed=3)
    legal = legal_action_ids(state)
    mask = legal_mask(state, legal)
    assert {index for index, bit in enumerate(mask) if bit} == set(legal)
    encoded = encode_state(state, legal)
    assert encoded.mask == mask


def test_encoding_is_deterministic():
    a = encode_features(new_game(seed=9))
    b = encode_features(new_game(seed=9))
    assert a == b


def test_encoding_changes_after_roll():
    from tests.conftest import roll_hand

    state = new_game(seed=4)
    before = encode_features(state)
    roll_hand(state)
    after = encode_features(state)
    assert before != after
    assert encode_state(state).mask[130] is False


def test_mask_tracks_random_legal_game():
    state = new_game(seed=21)
    for action_id in play_random_game(seed=21, max_actions=5_000).action_log:
        encoded = encode_state(state)
        legal = legal_action_ids(state)
        assert {i for i, bit in enumerate(encoded.mask) if bit} == set(legal)
        assert encoded.mask[action_id]
        apply_action(state, action_id, check_legal=False)
    assert state.phase is Phase.GAME_OVER
    assert legal_action_ids(state) == []
    assert not any(encode_state(state).mask)


def test_bc_examples_from_record_label_logged_actions():
    outcome = play_with_policy(17, RandomLegal(17), max_actions=5_000)
    record = DatasetRecord(
        seed=17,
        policy="random_legal",
        terminal=outcome.terminal,
        player_count=1,
        total_score=outcome.total_score,
        scores=outcome.scores,
        actions=outcome.actions,
    )
    examples = examples_from_record(record)
    assert examples
    assert all(ex.mask[ex.action_id] for ex in examples)
    assert all(ex.seed == 17 for ex in examples)
    assert len(examples) == len(outcome.actions)


def test_split_by_seed_holds_out_mod_10():
    mask = (True,) + (False,) * (ACTION_SPACE_SIZE - 1)
    dummy = (0.0,) * FEATURE_SIZE
    holdout = BCExample(dummy, mask, 0, 0.1, seed=0)
    keep = BCExample(dummy, mask, 0, 0.2, seed=1)
    train, val = split_by_seed([holdout, keep], val_frac=0.1)
    assert val == [holdout]
    assert train == [keep]


def test_encoded_state_rejects_wrong_sizes():
    try:
        EncodedState(features=(0.0,), mask=(False,) * ACTION_SPACE_SIZE)
    except ValueError as error:
        assert "features" in str(error)
    else:
        raise AssertionError("expected ValueError")
