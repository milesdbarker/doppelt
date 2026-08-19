"""Discrete action space: catalog subsets, composites, legal mask."""

from doppelt.actions.space import (
    ACTIVE_POST_ROLL_IDS,
    ACTIVE_PRE_ROLL_IDS,
    COMPOSITE_PICK_DICE,
    ILLEGAL_LOGIT,
    PHASE_ACTION_IDS,
    RETIRED_ACTION_IDS,
    candidate_action_ids,
    defined_action_ids,
    is_followup_mark,
    pick_is_composite,
    unused_action_ids,
)

__all__ = [
    "ACTIVE_POST_ROLL_IDS",
    "ACTIVE_PRE_ROLL_IDS",
    "COMPOSITE_PICK_DICE",
    "ILLEGAL_LOGIT",
    "PHASE_ACTION_IDS",
    "RETIRED_ACTION_IDS",
    "candidate_action_ids",
    "defined_action_ids",
    "is_followup_mark",
    "pick_is_composite",
    "unused_action_ids",
]
