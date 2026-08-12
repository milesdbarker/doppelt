"""Round 4 free-color wild bonus — player chooses color via catalog IDs 146–150."""

from doppelt.actions.catalog_v1 import (
    ACTION_SPACE_SIZE,
    ActionKind,
    choose_wild_color_id,
    decode_action,
    encode_action,
)
from doppelt.core.phases import Phase
from doppelt.core.types import Color
from doppelt.engine.game import _begin_active_turn, apply_action, legal_action_ids, new_game


def test_choose_wild_color_catalog_ids():
    assert choose_wild_color_id(Color.YELLOW) == 146
    assert choose_wild_color_id(Color.SILVER) == 150
    assert ACTION_SPACE_SIZE == 192
    for color in Color:
        action_id = choose_wild_color_id(color)
        action = decode_action(action_id)
        assert action.kind is ActionKind.CHOOSE_WILD_COLOR
        assert action.wild_color is color
        assert encode_action(action) == action_id


def test_round_four_begins_in_bonus_phase_for_color_choice():
    state = new_game(seed=1)
    state.round_index = 4
    _begin_active_turn(state)
    assert state.phase is Phase.RESOLVE_BONUS
    assert state.pending_bonuses[0].source == "round:4:wild"
    assert state.pending_bonuses[0].bonus.color is None
    assert choose_wild_color_id(Color.PINK) in legal_action_ids(state)
    assert state.awaiting_roll is True


def test_round_four_pink_wild_resolves_and_returns_to_active_pick():
    state = new_game(seed=2)
    state.round_index = 4
    _begin_active_turn(state)
    apply_action(state, choose_wild_color_id(Color.PINK))
    assert state.phase is Phase.ACTIVE_PICK
    assert state.awaiting_roll is True
    assert state.sheet.pink[0] == 6
    assert any(event.endswith(":choose:pink") for event in state.bonus_events)


def test_round_four_yellow_wild_requires_follow_up_mark():
    state = new_game(seed=3)
    state.sheet.yellow[2].circled = True
    state.round_index = 4
    _begin_active_turn(state)
    apply_action(state, choose_wild_color_id(Color.YELLOW))
    assert state.phase is Phase.RESOLVE_BONUS
    assert state.pending_bonuses[0].bonus.color is Color.YELLOW
