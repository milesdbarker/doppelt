"""Plus-one (extra die) — uses current face values, no re-roll."""

from doppelt.actions.catalog_v1 import plus_one_pick_id
from doppelt.core.phases import Phase
from doppelt.core.types import ActionTrack, Dice
from doppelt.engine.game import apply_action, legal_action_ids, new_game


def test_plus_one_blue_not_offered_when_no_legal_sum():
    state = new_game(seed=1)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    state.phase = Phase.PLUS_ONE
    state.sheet.mark_blue(10)
    state.sheet.mark_blue(2)
    state.faces[Dice.BLUE] = 1
    state.faces[Dice.WHITE] = 6

    assert plus_one_pick_id(Dice.BLUE) not in legal_action_ids(state)


def test_plus_one_yellow_not_offered_when_no_legal_cell():
    state = new_game(seed=2)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    state.phase = Phase.PLUS_ONE
    state.faces[Dice.YELLOW] = 3
    for cell in state.sheet.yellow:
        cell.circled = True
        cell.crossed = True

    assert plus_one_pick_id(Dice.YELLOW) not in legal_action_ids(state)


def test_plus_one_blue_marks_with_current_blue_white_sum():
    state = new_game(seed=3)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    state.phase = Phase.PLUS_ONE
    state.sheet.mark_blue(10)
    state.sheet.mark_blue(8)
    state.faces[Dice.BLUE] = 2
    state.faces[Dice.WHITE] = 1

    apply_action(state, plus_one_pick_id(Dice.BLUE))

    assert state.sheet.blue[2] == 3  # 2 + 1, no re-roll
    assert state.faces[Dice.BLUE] == 2
    assert state.faces[Dice.WHITE] == 1
    assert state.sheet.action_tracks[ActionTrack.PLUS_ONE].crossed == 1


def test_plus_one_does_not_reroll_die():
    state = new_game(seed=4)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    state.phase = Phase.PLUS_ONE
    state.faces[Dice.PINK] = 5

    apply_action(state, plus_one_pick_id(Dice.PINK))

    assert state.phase is Phase.PLUS_ONE
    assert state.faces[Dice.PINK] == 5
    assert state.sheet.pink[0] == 5


def test_plus_one_white_enters_mode_choice_without_changing_faces():
    from doppelt.actions.catalog_v1 import mark_white_pink_id

    state = new_game(seed=5)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    state.phase = Phase.PLUS_ONE
    state.plus_one_after_passive = False
    state.faces = {die: 4 for die in Dice}
    state.faces[Dice.BLUE] = 6  # blue sum 10 may still be legal; pink 4 is too

    faces_before = dict(state.faces)
    apply_action(state, plus_one_pick_id(Dice.WHITE))

    assert state.phase is Phase.ACTIVE_MARK_WHITE
    assert state.white_mark_resume_after == "plus_one_continue"
    assert mark_white_pink_id() in legal_action_ids(state)
    assert state.faces == faces_before


def test_plus_one_white_ignores_stale_bonus_resume():
    """A leftover plus_one_continue must not start passive before white mode choice."""
    from doppelt.actions.catalog_v1 import mark_white_pink_id

    state = new_game(seed=6)
    state.sheet.circle_action(ActionTrack.PLUS_ONE)
    state.phase = Phase.PLUS_ONE
    state.plus_one_after_passive = False
    state.bonus_resume_after = "plus_one_continue"
    state.faces = {die: 4 for die in Dice}

    faces_before = dict(state.faces)
    apply_action(state, plus_one_pick_id(Dice.WHITE))

    assert state.phase is Phase.ACTIVE_MARK_WHITE
    assert state.bonus_resume_after is None
    assert mark_white_pink_id() in legal_action_ids(state)
    assert state.faces == faces_before

    apply_action(state, mark_white_pink_id())
    assert state.sheet.pink[0] == 4
    assert state.phase is Phase.PASSIVE_PICK
