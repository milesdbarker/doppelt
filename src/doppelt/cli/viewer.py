"""Step through a replay on the photographed score sheet."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from doppelt.actions.catalog_v1 import describe_action
from doppelt.cli.overlay import (
    DicePanelState,
    OverlayMark,
    dice_panel_from_parts,
    sheet_highlight_keys,
)
from doppelt.core.phases import Phase
from doppelt.core.player_sheet import PlayerSheet
from doppelt.core.scoring import total_score
from doppelt.core.state import GameState
from doppelt.core.types import Dice
from doppelt.engine.game import apply_action, new_game
from doppelt.replay.binary import GameLog


@dataclass
class VizFrame:
    index: int
    action_id: int | None
    sheet: PlayerSheet
    previous_sheet: PlayerSheet | None
    round_index: int
    phase: Phase
    seed: int
    total: int
    hand: list[Dice] = field(default_factory=list)
    platter: list[Dice] = field(default_factory=list)
    slots: list[Dice | None] = field(default_factory=lambda: [None, None, None])
    passive_pool: list[Dice] = field(default_factory=list)
    faces: dict[Dice, int] = field(default_factory=dict)
    awaiting_roll: bool = False
    plus_one_after_passive: bool = False
    resume_phase: Phase | None = None
    bonus_resume_after: str | None = None
    previous_hand: list[Dice] | None = None
    previous_platter: list[Dice] | None = None
    previous_slots: list[Dice | None] | None = None
    previous_pool: list[Dice] | None = None
    note: str | None = None

    @property
    def caption(self) -> str:
        if self.action_id is None:
            action = "start of game"
        else:
            action = describe_action(self.action_id)
        text = (
            f"step {self.index}  round {self.round_index}  {self.phase.value}  "
            f"{action}  total={self.total}"
        )
        if self.note:
            return f"{text}  {self.note}"
        return text

    def marks(self) -> list[OverlayMark]:
        highlights = sheet_highlight_keys(self.previous_sheet, self.sheet)
        return marks_from_sheet_frame(self, highlights)

    def dice_panel(self) -> DicePanelState:
        return dice_panel_from_parts(
            hand=self.hand,
            platter=self.platter,
            faces=self.faces,
            awaiting_roll=self.awaiting_roll,
            slots=self.slots,
            passive_pool=self.passive_pool,
            phase=self.phase,
            resume_phase=self.resume_phase,
            plus_one_after_passive=self.plus_one_after_passive,
            bonus_resume_after=self.bonus_resume_after,
            previous_hand=self.previous_hand,
            previous_platter=self.previous_platter,
            previous_slots=self.previous_slots,
            previous_pool=self.previous_pool,
        )


def marks_from_sheet_frame(frame: VizFrame, highlights: set[str]) -> list[OverlayMark]:
    from doppelt.cli.overlay import marks_from_sheet

    return marks_from_sheet(
        frame.sheet,
        highlights=highlights,
        round_index=frame.round_index,
        game_over=frame.phase is Phase.GAME_OVER,
    )


_PLUS_ONE_VIEW_PHASES = frozenset(
    {
        Phase.PLUS_ONE,
        Phase.PLUS_ONE_MARK_YELLOW,
    }
)


def _should_split_plus_one_from_passive(before: dict, after_phase: Phase) -> bool:
    """True when plus-one finished and the engine immediately rolled the passive dice."""
    if after_phase is not Phase.PASSIVE_PICK:
        return False
    if before["plus_one_after_passive"]:
        return False
    if before["phase"] in _PLUS_ONE_VIEW_PHASES:
        return True
    if before["bonus_resume_after"] == "plus_one_continue":
        return True
    if before["silver_finish"] == "plus_one":
        return True
    return before["white_mark_resume_after"] == "plus_one_continue"


def build_frames(log: GameLog) -> list[VizFrame]:
    state = new_game(seed=log.seed, player_count=log.player_count)
    frames = [_frame(0, None, None, state)]
    previous = _dice_snapshot(state)
    previous_sheet = state.sheet.copy()
    frame_index = 1
    for action_id in log.actions:
        before = _dice_snapshot(state)
        apply_action(state, action_id)
        if _should_split_plus_one_from_passive(before, state.phase):
            view_phase = (
                before["phase"] if before["phase"] in _PLUS_ONE_VIEW_PHASES else Phase.PLUS_ONE
            )
            bridge_dice = dict(before)
            bridge_dice["plus_one_after_passive"] = False
            bridge_dice["resume_phase"] = None
            bridge_dice["bonus_resume_after"] = None
            frames.append(
                _frame(
                    frame_index,
                    action_id,
                    previous,
                    state,
                    previous_sheet=previous_sheet,
                    dice=bridge_dice,
                    phase=view_phase,
                    note="plus-one applied (before passive roll)",
                )
            )
            frame_index += 1
            previous_sheet = state.sheet.copy()
            previous = before
            frames.append(
                _frame(
                    frame_index,
                    action_id,
                    previous,
                    state,
                    previous_sheet=previous_sheet,
                    note="passive dice rolled",
                )
            )
        else:
            frames.append(
                _frame(
                    frame_index,
                    action_id,
                    previous,
                    state,
                    previous_sheet=previous_sheet,
                )
            )
        previous_sheet = state.sheet.copy()
        previous = _dice_snapshot(state)
        frame_index += 1
    return frames


def _dice_snapshot(state: GameState) -> dict:
    return {
        "hand": list(state.hand),
        "platter": list(state.platter),
        "slots": list(state.slots),
        "pool": list(state.passive_pool),
        "faces": dict(state.faces),
        "awaiting_roll": state.awaiting_roll,
        "plus_one_after_passive": state.plus_one_after_passive,
        "resume_phase": state.resume_phase,
        "bonus_resume_after": state.bonus_resume_after,
        "phase": state.phase,
        "silver_finish": state.silver_finish,
        "white_mark_resume_after": state.white_mark_resume_after,
    }


def _frame(
    index: int,
    action_id: int | None,
    previous: dict | None,
    state: GameState,
    *,
    previous_sheet: PlayerSheet | None = None,
    dice: dict | None = None,
    phase: Phase | None = None,
    plus_one_after_passive: bool | None = None,
    resume_phase: Phase | None = None,
    bonus_resume_after: str | None = None,
    note: str | None = None,
) -> VizFrame:
    prev = previous or {}
    d = dice if dice is not None else _dice_snapshot(state)
    return VizFrame(
        index=index,
        action_id=action_id,
        sheet=state.sheet.copy(),
        previous_sheet=previous_sheet,
        round_index=state.round_index,
        phase=state.phase if phase is None else phase,
        seed=state.seed,
        total=total_score(state.sheet),
        hand=list(d["hand"]),
        platter=list(d["platter"]),
        slots=list(d["slots"]),
        passive_pool=list(d["pool"]),
        faces=dict(d["faces"]),
        awaiting_roll=d["awaiting_roll"],
        plus_one_after_passive=(
            d["plus_one_after_passive"]
            if plus_one_after_passive is None
            else plus_one_after_passive
        ),
        resume_phase=d["resume_phase"] if resume_phase is None else resume_phase,
        bonus_resume_after=(
            d["bonus_resume_after"] if bonus_resume_after is None else bonus_resume_after
        ),
        previous_hand=None if previous is None else list(prev["hand"]),
        previous_platter=None if previous is None else list(prev["platter"]),
        previous_slots=None if previous is None else list(prev["slots"]),
        previous_pool=None if previous is None else list(prev["pool"]),
        note=note,
    )


def run_tk_viewer(
    frames: list[VizFrame],
    *,
    start_at: int | None = None,
    on_save: Callable[[int], None] | None = None,
) -> None:
    import tkinter as tk
    from PIL import ImageTk

    from doppelt.cli.render import load_board_image, render_overlay

    base = load_board_image()
    index = len(frames) - 1 if start_at is None else max(0, min(start_at, len(frames) - 1))

    root = tk.Tk()
    root.title("doppelt visualize")
    caption = tk.StringVar()
    help_text = "←/→ step   Home/End first/last   S save PNG   Q quit"
    tk.Label(root, textvariable=caption, font=("Segoe UI", 11), anchor="w", justify="left").pack(
        fill="x", padx=8, pady=(8, 0)
    )
    tk.Label(root, text=help_text, fg="#444", anchor="w").pack(fill="x", padx=8)
    panel = tk.Label(root)
    panel.pack(padx=8, pady=8)

    photo_holder: dict[str, object] = {}

    def show() -> None:
        frame = frames[index]
        image = render_overlay(frame.marks(), base=base, dice=frame.dice_panel())
        max_h = min(920, root.winfo_screenheight() - 140)
        if image.height > max_h:
            scale = max_h / image.height
            image = image.resize((int(image.width * scale), max_h))
        photo = ImageTk.PhotoImage(image)
        photo_holder["img"] = photo
        panel.configure(image=photo)
        caption.set(f"{frame.caption}   ({index + 1}/{len(frames)})")
        root.title(f"doppelt visualize — {frame.caption}")

    def step(delta: int) -> None:
        nonlocal index
        index = max(0, min(len(frames) - 1, index + delta))
        show()

    def go(target: int) -> None:
        nonlocal index
        index = max(0, min(len(frames) - 1, target))
        show()

    root.bind("<Left>", lambda _e: step(-1))
    root.bind("<Right>", lambda _e: step(1))
    root.bind("<Home>", lambda _e: go(0))
    root.bind("<End>", lambda _e: go(len(frames) - 1))
    root.bind("q", lambda _e: root.destroy())
    root.bind("Q", lambda _e: root.destroy())
    if on_save is not None:
        root.bind("s", lambda _e: on_save(index))
        root.bind("S", lambda _e: on_save(index))

    show()
    root.mainloop()
