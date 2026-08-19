"""encoding_v2: fixed features + catalog-aligned legal mask."""

import pytest

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
        PROGRESS_SIZE,
        SHEET_SIZE,
    )

    state = new_game(seed=1)
    encoded = encode_state(state)
    assert encoded.encoding_version == 2
    assert SHEET_SIZE + DICE_SIZE + GLOBAL_SIZE + PENDING_SIZE + PROGRESS_SIZE == FEATURE_SIZE
    assert ENCODING_VERSION == 2
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
def test_pick_index_rank_empty_hand_and_fox_floor():
    from doppelt.core.types import Dice
    from doppelt.ml.encoding import (
        DICE_ORDER,
        PROGRESS_SLICE,
        TARGET_BLUE,
        TARGET_YELLOW,
        hand_ranks,
        would_empty_hand,
    )
    from tests.conftest import roll_hand

    state = new_game(seed=4)
    roll_hand(state)
    state.hand = [Dice.YELLOW, Dice.BLUE, Dice.PINK]
    state.faces[Dice.YELLOW] = 2
    state.faces[Dice.BLUE] = 4
    state.faces[Dice.PINK] = 6
    state.picks_made = 0
    ranks = hand_ranks(state)
    assert ranks[Dice.YELLOW] == 1
    assert ranks[Dice.BLUE] == 2
    assert ranks[Dice.PINK] == 3
    assert would_empty_hand(state, Dice.YELLOW) is False
    assert would_empty_hand(state, Dice.BLUE) is False
    assert would_empty_hand(state, Dice.PINK) is True
    state.picks_made = 2
    assert would_empty_hand(state, Dice.YELLOW) is True

    state.picks_made = 0
    progress = encode_features(state)[PROGRESS_SLICE]
    assert progress[0:3] == [1.0, 0.0, 0.0]
    yellow_off = 3 + DICE_ORDER.index(Dice.YELLOW) * 2
    assert progress[yellow_off] == pytest.approx(1 / 6)
    assert progress[yellow_off + 1] == 0.0
    pink_off = 3 + DICE_ORDER.index(Dice.PINK) * 2
    assert progress[pink_off] == pytest.approx(3 / 6)
    assert progress[pink_off + 1] == 1.0
    assert progress[20] == 0.0  # fox floor
    assert progress[21] == 1.0  # any color zero
    assert progress[22] == 1.0  # remaining to yellow 21
    assert progress[23] == 1.0  # remaining to blue 28
    assert TARGET_YELLOW == 21
    assert TARGET_BLUE == 28


def test_silver_column_progress_and_pink_six():
    from doppelt.core.silver import SILVER_ROW_COLORS
    from doppelt.ml.encoding import PINK_FIRST_SIX_SLOT, PROGRESS_SLICE

    state = new_game(seed=2)
    for row in SILVER_ROW_COLORS:
        state.sheet.silver[row].add(3)
    state.sheet.pink[PINK_FIRST_SIX_SLOT] = 6
    progress = encode_features(state)[PROGRESS_SLICE]
    assert progress[27] == 1.0  # first pink-6 slot filled
    assert progress[31] == 1.0  # silver column 3 fill 4/4
    assert progress[37] == 1.0  # silver column 3 complete


def test_encoded_state_rejects_wrong_sizes():
    try:
        EncodedState(features=(0.0,), mask=(False,) * ACTION_SPACE_SIZE)
    except ValueError as error:
        assert "features" in str(error)
    else:
        raise AssertionError("expected ValueError")
