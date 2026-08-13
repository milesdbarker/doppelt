"""Heuristic bot: setup value beyond immediate total_score."""

from tests.conftest import roll_hand

from doppelt.actions.catalog_v1 import (
    choose_wild_color_id,
    mark_white_blue_id,
    mark_white_pink_id,
    passive_mark_yellow_id,
    passive_platter_id,
    passive_skip_id,
    pick_die_id,
    plus_one_pick_id,
    unlock_platter_id,
    use_reroll_id,
)
from doppelt.core.phases import Phase
from doppelt.core.types import ALL_DICE, ActionTrack, Color, Dice
from doppelt.engine.game import _begin_active_turn, legal_action_ids, new_game
from doppelt.sim import GreedyImmediate, Heuristic, make_policy, play_with_policy


def test_make_policy_heuristic():
    policy = make_policy("heuristic", seed=1)
    assert policy.name == "heuristic"


def test_heuristic_prefers_yellow_circle_over_skip():
    state = new_game(seed=1)
    state.phase = Phase.PASSIVE_PICK
    state.faces[Dice.YELLOW] = 1
    state.platter = [Dice.YELLOW]
    state.passive_pool = []
    state.use_pool_fallback = False

    legal = legal_action_ids(state)
    assert passive_platter_id(Dice.YELLOW) in legal
    assert passive_skip_id() in legal

    choice = Heuristic(0).select(state, legal)
    assert choice == passive_platter_id(Dice.YELLOW)


def test_heuristic_prefers_high_blue_opener_over_pink():
    """Greedy takes pink 6 pts; heuristic prefers opening blue on 11."""
    state = new_game(seed=2)
    state.round_index = 1
    state.phase = Phase.ACTIVE_MARK_WHITE
    state.pending_die = Dice.WHITE
    state.pending_value = 6
    state.faces[Dice.WHITE] = 6
    state.faces[Dice.BLUE] = 5
    state.white_mark_resume_after = "after_active_pick"

    legal = legal_action_ids(state)
    assert mark_white_blue_id() in legal
    assert mark_white_pink_id() in legal
    assert GreedyImmediate(0).select(state, legal) == mark_white_pink_id()
    assert Heuristic(0).select(state, legal) == mark_white_blue_id()


def test_heuristic_prefers_silver_column_progress_over_low_pink():
    state = new_game(seed=3)
    roll_hand(state)
    state.round_index = 1
    state.awaiting_roll = False
    state.picks_made = 2  # last pick: no missed-roll penalty
    state.hand = [Dice.SILVER, Dice.PINK]
    state.platter = []
    state.faces[Dice.SILVER] = 3
    state.faces[Dice.PINK] = 2
    state.sheet.silver[Color.YELLOW].add(3)
    state.sheet.silver[Color.BLUE].add(3)

    legal = legal_action_ids(state)
    assert pick_die_id(Dice.SILVER) in legal
    assert pick_die_id(Dice.PINK) in legal
    assert Heuristic(0).select(state, legal) == pick_die_id(Dice.SILVER)


def test_heuristic_avoids_pick_that_skips_remaining_rolls():
    state = new_game(seed=5)
    roll_hand(state)
    state.awaiting_roll = False
    state.picks_made = 0
    state.hand = [Dice.PINK, Dice.YELLOW]
    state.platter = []
    state.faces[Dice.PINK] = 6
    state.faces[Dice.YELLOW] = 1

    legal = legal_action_ids(state)
    assert pick_die_id(Dice.PINK) in legal
    assert pick_die_id(Dice.YELLOW) in legal
    assert use_reroll_id() in legal
    choice = Heuristic(0).select(state, legal)
    assert choice != pick_die_id(Dice.PINK)
    assert choice == use_reroll_id()


def test_heuristic_prefers_big_silver_sweep_on_last_pick():
    state = new_game(seed=6)
    roll_hand(state)
    state.awaiting_roll = False
    state.picks_made = 2
    state.hand = [Dice.SILVER, Dice.YELLOW, Dice.BLUE, Dice.GREEN, Dice.PINK]
    state.platter = []
    state.faces[Dice.SILVER] = 6
    state.faces[Dice.YELLOW] = 2
    state.faces[Dice.BLUE] = 3
    state.faces[Dice.GREEN] = 1
    state.faces[Dice.PINK] = 4

    legal = legal_action_ids(state)
    assert pick_die_id(Dice.SILVER) in legal
    assert pick_die_id(Dice.PINK) in legal
    assert Heuristic(0).select(state, legal) == pick_die_id(Dice.SILVER)


def test_heuristic_avoids_weak_silver_without_bonus():
    state = new_game(seed=7)
    state.phase = Phase.PLUS_ONE
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    state.faces[Dice.SILVER] = 2
    state.faces[Dice.PINK] = 6
    state.plus_one_dice_used = set()

    legal = legal_action_ids(state)
    assert plus_one_pick_id(Dice.SILVER) in legal
    assert plus_one_pick_id(Dice.PINK) in legal
    assert Heuristic(0).select(state, legal) == plus_one_pick_id(Dice.PINK)


def test_heuristic_avoids_pink_that_misses_slot_4_to_7_bonus():
    state = new_game(seed=8)
    state.phase = Phase.PASSIVE_PICK
    state.faces[Dice.PINK] = 3
    state.faces[Dice.YELLOW] = 1
    state.platter = [Dice.PINK, Dice.YELLOW]
    state.passive_pool = []
    state.use_pool_fallback = False
    state.sheet.pink[0] = 1
    state.sheet.pink[1] = 1
    state.sheet.pink[2] = 2
    state.sheet.pink[3] = 3  # next slot 4 needs 4+ for plus-one

    legal = legal_action_ids(state)
    assert passive_platter_id(Dice.PINK) in legal
    assert passive_platter_id(Dice.YELLOW) in legal
    assert Heuristic(0).select(state, legal) == passive_platter_id(Dice.YELLOW)


def test_heuristic_late_game_prefers_yellow_when_below_21():
    state = new_game(seed=9)
    state.round_index = 5
    state.phase = Phase.PASSIVE_PICK
    state.faces[Dice.PINK] = 6
    state.faces[Dice.YELLOW] = 1
    state.platter = [Dice.PINK, Dice.YELLOW]
    state.passive_pool = []
    state.use_pool_fallback = False

    legal = legal_action_ids(state)
    assert passive_platter_id(Dice.PINK) in legal
    assert passive_platter_id(Dice.YELLOW) in legal
    assert Heuristic(0).select(state, legal) == passive_platter_id(Dice.YELLOW)


def test_heuristic_stays_on_one_yellow_family():
    state = new_game(seed=10)
    state.sheet.yellow[0].circled = True  # even-row family
    state.phase = Phase.PASSIVE_MARK_YELLOW
    state.pending_die = Dice.YELLOW
    state.pending_value = 5  # cell 7 odd row, cell 8 even row

    legal = legal_action_ids(state)
    assert passive_mark_yellow_id(7) in legal
    assert passive_mark_yellow_id(8) in legal
    assert Heuristic(0).select(state, legal) == passive_mark_yellow_id(8)


def test_heuristic_prefers_even_yellow_rows_when_open():
    state = new_game(seed=12)
    state.phase = Phase.PASSIVE_MARK_YELLOW
    state.pending_die = Dice.YELLOW
    state.pending_value = 5  # cell 7 odd row, cell 8 even row

    legal = legal_action_ids(state)
    assert passive_mark_yellow_id(7) in legal
    assert passive_mark_yellow_id(8) in legal
    assert Heuristic(0).select(state, legal) == passive_mark_yellow_id(8)


def test_heuristic_sets_up_yellow_row_4_for_pink_5_and_6():
    state = new_game(seed=13)
    state.phase = Phase.PASSIVE_MARK_YELLOW
    state.pending_die = Dice.YELLOW
    state.pending_value = 4  # cell 4 row 2, cell 9 row 4

    legal = legal_action_ids(state)
    assert passive_mark_yellow_id(4) in legal
    assert passive_mark_yellow_id(9) in legal
    assert Heuristic(0).select(state, legal) == passive_mark_yellow_id(9)


def test_heuristic_holds_yellow_row_4_until_pink_5_or_6():
    state = new_game(seed=14)
    state.sheet.yellow[8].circled = True
    state.phase = Phase.PASSIVE_MARK_YELLOW
    state.pending_die = Dice.YELLOW
    state.pending_value = 4

    legal = legal_action_ids(state)
    assert passive_mark_yellow_id(4) in legal
    assert passive_mark_yellow_id(9) in legal
    assert Heuristic(0).select(state, legal) == passive_mark_yellow_id(4)


def test_heuristic_fires_yellow_row_4_when_pink_is_on_5():
    state = new_game(seed=15)
    for slot in range(5):
        state.sheet.pink[slot] = 4
    state.sheet.yellow[8].circled = True
    state.phase = Phase.PASSIVE_MARK_YELLOW
    state.pending_die = Dice.YELLOW
    state.pending_value = 4

    legal = legal_action_ids(state)
    assert passive_mark_yellow_id(4) in legal
    assert passive_mark_yellow_id(9) in legal
    assert Heuristic(0).select(state, legal) == passive_mark_yellow_id(9)


def test_heuristic_holds_blue_slot_6_until_pink_5_or_6():
    state = new_game(seed=16)
    for index, value in enumerate((12, 11, 10, 9, 8, 7)):
        state.sheet.blue[index] = value
    state.phase = Phase.PASSIVE_PICK
    state.faces[Dice.BLUE] = 3
    state.faces[Dice.WHITE] = 3
    state.faces[Dice.YELLOW] = 3
    state.platter = [Dice.BLUE, Dice.YELLOW]
    state.passive_pool = []
    state.use_pool_fallback = False

    legal = legal_action_ids(state)
    assert passive_platter_id(Dice.BLUE) in legal
    assert passive_platter_id(Dice.YELLOW) in legal
    assert Heuristic(0).select(state, legal) == passive_platter_id(Dice.YELLOW)


def test_heuristic_takes_six_mark_silver_when_two_unlocks_ready():
    state = new_game(seed=17)
    roll_hand(state)
    state.sheet.circle_action(ActionTrack.UNLOCK)
    state.sheet.circle_action(ActionTrack.UNLOCK)
    state.awaiting_roll = False
    state.picks_made = 0
    state.hand = list(ALL_DICE)
    state.platter = []
    state.faces[Dice.SILVER] = 6
    state.faces[Dice.WHITE] = 5
    state.faces[Dice.PINK] = 4
    state.faces[Dice.GREEN] = 3
    state.faces[Dice.BLUE] = 2
    state.faces[Dice.YELLOW] = 1

    legal = legal_action_ids(state)
    assert pick_die_id(Dice.SILVER) in legal
    assert pick_die_id(Dice.YELLOW) in legal
    assert Heuristic(0).select(state, legal) == pick_die_id(Dice.SILVER)


def test_heuristic_unlocks_white_before_other_dice():
    state = new_game(seed=18)
    state.phase = Phase.ACTIVE_PICK
    state.awaiting_roll = True
    state.picks_made = 1
    state.hand = []
    state.platter = [Dice.WHITE, Dice.YELLOW, Dice.GREEN]
    state.sheet.circle_action(ActionTrack.UNLOCK)
    state.sheet.circle_action(ActionTrack.UNLOCK)

    legal = legal_action_ids(state)
    assert unlock_platter_id(Dice.WHITE) in legal
    assert unlock_platter_id(Dice.YELLOW) in legal
    assert Heuristic(0).select(state, legal) == unlock_platter_id(Dice.WHITE)


def test_heuristic_unlocks_even_yellow_over_bonusless_pink():
    state = new_game(seed=19)
    state.phase = Phase.ACTIVE_PICK
    state.awaiting_roll = True
    state.picks_made = 1
    state.hand = []
    state.platter = [Dice.YELLOW, Dice.PINK]
    state.sheet.circle_action(ActionTrack.UNLOCK)

    legal = legal_action_ids(state)
    assert unlock_platter_id(Dice.YELLOW) in legal
    assert unlock_platter_id(Dice.PINK) in legal
    assert Heuristic(0).select(state, legal) == unlock_platter_id(Dice.YELLOW)


def test_heuristic_prefers_high_green_on_slot_0():
    state = new_game(seed=20)
    state.phase = Phase.PLUS_ONE
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    state.faces[Dice.GREEN] = 6
    state.faces[Dice.YELLOW] = 1
    state.plus_one_dice_used = set()

    legal = legal_action_ids(state)
    assert plus_one_pick_id(Dice.GREEN) in legal
    assert plus_one_pick_id(Dice.YELLOW) in legal
    assert Heuristic(0).select(state, legal) == plus_one_pick_id(Dice.GREEN)


def test_heuristic_avoids_low_green_on_slot_0():
    state = new_game(seed=21)
    state.phase = Phase.PLUS_ONE
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    state.faces[Dice.GREEN] = 2
    state.faces[Dice.YELLOW] = 3
    state.plus_one_dice_used = set()

    legal = legal_action_ids(state)
    assert plus_one_pick_id(Dice.GREEN) in legal
    assert plus_one_pick_id(Dice.YELLOW) in legal
    assert Heuristic(0).select(state, legal) == plus_one_pick_id(Dice.YELLOW)


def test_heuristic_rerolls_when_no_acceptable_dice():
    state = new_game(seed=23)
    roll_hand(state)
    state.awaiting_roll = False
    state.picks_made = 0
    state.hand = [Dice.YELLOW, Dice.GREEN, Dice.PINK, Dice.BLUE, Dice.WHITE]
    state.platter = []
    state.faces[Dice.YELLOW] = 1
    state.faces[Dice.GREEN] = 2
    state.faces[Dice.PINK] = 3
    state.faces[Dice.BLUE] = 1
    state.faces[Dice.WHITE] = 1

    legal = legal_action_ids(state)
    assert use_reroll_id() in legal
    assert Heuristic(0).select(state, legal) == use_reroll_id()


def test_heuristic_does_not_reroll_when_even_yellow_is_showing():
    state = new_game(seed=24)
    roll_hand(state)
    state.awaiting_roll = False
    state.picks_made = 0
    state.hand = [Dice.YELLOW, Dice.PINK]
    state.platter = []
    state.faces[Dice.YELLOW] = 3
    state.faces[Dice.PINK] = 6

    legal = legal_action_ids(state)
    assert use_reroll_id() in legal
    assert pick_die_id(Dice.YELLOW) in legal
    assert Heuristic(0).select(state, legal) == pick_die_id(Dice.YELLOW)


def test_heuristic_defers_low_green_on_active_roll_2():
    state = new_game(seed=25)
    roll_hand(state)
    state.sheet.green[0] = 12
    state.awaiting_roll = False
    state.picks_made = 1
    state.hand = [Dice.GREEN, Dice.YELLOW, Dice.PINK]
    state.platter = []
    state.faces[Dice.GREEN] = 2
    state.faces[Dice.YELLOW] = 6
    state.faces[Dice.PINK] = 5

    legal = legal_action_ids(state)
    assert pick_die_id(Dice.GREEN) in legal
    assert pick_die_id(Dice.YELLOW) in legal
    assert pick_die_id(Dice.PINK) in legal
    choice = Heuristic(0).select(state, legal)
    assert choice != pick_die_id(Dice.GREEN)
    assert choice in {pick_die_id(Dice.YELLOW), pick_die_id(Dice.PINK)}


def test_heuristic_takes_low_green_on_first_active_roll():
    state = new_game(seed=26)
    roll_hand(state)
    state.sheet.green[0] = 12
    state.awaiting_roll = False
    state.picks_made = 0
    state.hand = [Dice.GREEN, Dice.PINK]
    state.platter = []
    state.faces[Dice.GREEN] = 2
    state.faces[Dice.PINK] = 3

    legal = legal_action_ids(state)
    assert pick_die_id(Dice.GREEN) in legal
    assert pick_die_id(Dice.PINK) in legal
    assert Heuristic(0).select(state, legal) == pick_die_id(Dice.GREEN)


def test_heuristic_takes_low_green_on_passive():
    state = new_game(seed=27)
    state.sheet.green[0] = 12
    state.phase = Phase.PASSIVE_PICK
    state.faces[Dice.GREEN] = 2
    state.faces[Dice.PINK] = 3
    state.platter = [Dice.GREEN, Dice.PINK]
    state.passive_pool = []
    state.use_pool_fallback = False

    legal = legal_action_ids(state)
    assert passive_platter_id(Dice.GREEN) in legal
    assert passive_platter_id(Dice.PINK) in legal
    assert Heuristic(0).select(state, legal) == passive_platter_id(Dice.GREEN)


def test_heuristic_round4_wild_prefers_positive_green_6():
    state = new_game(seed=22)
    state.round_index = 4
    _begin_active_turn(state)
    legal = legal_action_ids(state)
    assert choose_wild_color_id(Color.GREEN) in legal
    assert Heuristic(0).select(state, legal) == choose_wild_color_id(Color.GREEN)


def test_heuristic_trial_does_not_mutate_live_state():
    state = new_game(seed=4)
    state.sheet.circle_action(ActionTrack.REROLL)
    roll_hand(state)
    faces_before = dict(state.faces)
    log_before = list(state.action_log)
    legal = legal_action_ids(state)
    assert len(legal) > 1

    Heuristic(0).select(state, legal)

    assert dict(state.faces) == faces_before
    assert state.action_log == log_before


def test_heuristic_reaches_terminal():
    outcome = play_with_policy(11, Heuristic(11), max_actions=5_000)
    assert outcome.terminal
    assert outcome.n_actions > 0
