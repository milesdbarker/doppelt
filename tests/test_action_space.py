"""Catalog v1 action space: defined IDs, phase subsets, composites."""

from doppelt.actions.catalog_v1 import ACTION_SPACE_SIZE, decode_action, roll_hand_id
from doppelt.actions.space import (
    ACTIVE_POST_ROLL_IDS,
    ACTIVE_PRE_ROLL_IDS,
    ILLEGAL_LOGIT,
    PHASE_ACTION_IDS,
    RETIRED_ACTION_IDS,
    assert_catalog_roundtrip,
    candidate_action_ids,
    defined_action_ids,
    is_followup_mark,
    pick_is_composite,
    unused_action_ids,
)
from doppelt.core.phases import Phase
from doppelt.core.types import Dice
from doppelt.engine.game import apply_action, legal_action_ids, new_game, play_random_game


def test_catalog_size_covers_defined_ids():
    defined = defined_action_ids()
    unused = unused_action_ids()
    assert max(defined) < ACTION_SPACE_SIZE
    assert defined | unused == frozenset(range(ACTION_SPACE_SIZE))
    assert not (defined & unused)
    assert RETIRED_ACTION_IDS <= defined
    assert_catalog_roundtrip()


def test_no_duplicate_catalog_semantics():
    seen: dict[tuple, int] = {}
    for action_id in defined_action_ids():
        action = decode_action(action_id)
        key = (
            action.kind,
            action.die,
            action.yellow_cell_id,
            action.silver_row_index,
            action.silver_value,
            action.passive_from_pool,
            action.wild_color,
        )
        assert key not in seen, f"{action_id} duplicates {seen[key]}"
        seen[key] = action_id


def test_phase_subsets_cover_defined_live_ids():
    live = defined_action_ids() - RETIRED_ACTION_IDS
    covered = set()
    for ids in PHASE_ACTION_IDS.values():
        covered |= set(ids)
    assert covered == live
    for phase, ids in PHASE_ACTION_IDS.items():
        if phase is Phase.GAME_OVER:
            assert ids == frozenset()
            continue
        assert ids, phase


def test_candidate_ids_split_active_pre_and_post_roll():
    state = new_game(seed=1)
    assert state.awaiting_roll
    assert candidate_action_ids(state) == ACTIVE_PRE_ROLL_IDS
    legal = legal_action_ids(state)
    assert set(legal) <= ACTIVE_PRE_ROLL_IDS
    apply_action(state, roll_hand_id())
    assert not state.awaiting_roll
    assert candidate_action_ids(state) == ACTIVE_POST_ROLL_IDS
    assert set(legal_action_ids(state)) <= ACTIVE_POST_ROLL_IDS


def test_legal_actions_stay_inside_phase_subset():
    for seed in (1, 8, 42, 99):
        state = new_game(seed=seed)
        for action_id in play_random_game(seed=seed, max_actions=5_000).action_log:
            allowed = candidate_action_ids(state)
            legal = legal_action_ids(state)
            assert set(legal) <= allowed
            assert action_id in legal
            assert action_id not in unused_action_ids()
            assert action_id not in RETIRED_ACTION_IDS
            apply_action(state, action_id, check_legal=False)
        assert state.phase is Phase.GAME_OVER
        assert legal_action_ids(state) == []
        assert candidate_action_ids(state) == frozenset()


def test_blue_green_pink_picks_are_composite():
    assert pick_is_composite(Dice.BLUE)
    assert pick_is_composite(Dice.GREEN)
    assert pick_is_composite(Dice.PINK)
    assert not pick_is_composite(Dice.YELLOW)
    assert not pick_is_composite(Dice.WHITE)
    assert not pick_is_composite(Dice.SILVER)


def test_followup_ids_are_choice_points():
    from doppelt.actions.catalog_v1 import mark_white_blue_id, mark_yellow_id, pick_die_id

    assert is_followup_mark(mark_yellow_id(0))
    assert is_followup_mark(mark_white_blue_id())
    assert not is_followup_mark(pick_die_id(Dice.BLUE))


def test_illegal_logit_is_negative_infinity():
    assert ILLEGAL_LOGIT == float("-inf")
