"""Shared test helpers."""

from doppelt.actions.catalog_v1 import roll_hand_id
from doppelt.engine.game import apply_action


def roll_hand(state) -> None:
    apply_action(state, roll_hand_id())
