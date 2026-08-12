# Rules notes — edge cases and engine conventions

**Scope:** Solo (1 player, 6 rounds) first. Multiplayer differences noted where relevant.

---

## Round track (rounds 5 & 6)

Solo uses the same player sheet as multiplayer. The round track shows six round
numbers. **Rounds 5 and 6 display player-count symbols only** (3 players / 2 & 1
players) — they are **not** bonus rounds in solo. No round-start grant on rounds 5–6.

Round-start grants apply to rounds 1–4 only:

| Round | Grant |
|-------|--------|
| 1 | Unlock reroll action track |
| 2 | Unlock plus one (+1) action track |
| 3 | Unlock unlock action track |
| 4 | Black `?` bonus — player chooses wild color (catalog IDs 146–150) |

## Solo game structure

- **6 rounds.** Each round has one **active** phase then one **passive** phase (same
  player, both roles).
- Order: active → passive → active → passive … (6 of each per game).
- Goal: maximize total score (no opponent).
- **Passive phase cannot use reroll actions** (rulebook solo restriction).

---

## Active turn — dice flow

The active player rolls up to **three times**, picking one die after each roll.

### Picking a die

1. Choose one die from the dice in hand.
2. Place it on an empty **dice slot** (max 3 slots on the sheet).
3. **Mark the sheet** using that die’s color/value (see area rules + white die).
4. Send every die showing a **strictly lower** value than the chosen die to the
   **silver platter**.
   - If the chosen die is the **lowest** among dice in hand, **nothing** goes to the
     platter on that pick.

### After the third pick (or early stop)

- After the third pick, **all remaining dice** not on the sheet go to the platter
  (even if they are not lower than the last pick).

### Fewer than three rolls

If an early pick is high enough that all other dice in hand go to the platter, there
may be **no dice left to reroll**. The active turn ends with fewer than three picks.
This is legal; avoid it when possible.

### Forfeit pick

If the active player **cannot or chooses not** to use any die from the current roll,
they **skip that pick** (leave a dice slot empty for that roll). The roll still
consumes one of the three pick opportunities.

---

## Active turn — silver platter (multiplayer rules; solo active phase)

During the **active** phase, platter contents come from the active pick/roll flow
above — **not** from “lowest 3 of 6.”

Platter dice during active play:

- Cannot be picked again by the active player (except **unlock** action).
- After the active phase ends, they become available for the **passive** phase.

---

## Solo passive turn — platter and fallback pick

Solo passive is **not** “take one die from the active platter.” Instead:

1. Roll all **6 dice**.
2. Place the **3 lowest** face values on the silver platter.
3. The other **3 dice** stay in the **passive pool** (not on the platter).

### Tie-break for “lowest 3”

Physical rules: if multiple dice tie for a borderline spot, the die **closest to the
platter** goes on the platter. We have no physical platter in the engine.

**Engine convention:** when assigning the three platter dice, break ties **at random**
using the game RNG (same seeded stream as dice rolls so replays stay deterministic).
Typical approach: shuffle tied candidates with `rng`, then take the three lowest by
value (random order among equal values).

Helper: `split_passive_roll()` in `doppelt.core.solo_passive`.

### Passive mark

1. Try to pick **one die from the platter** and mark it legally.
2. **Fallback (solo simplification):** if **no** platter die has a legal mark, pick
   **one die from the passive pool** (the other 3 from the roll) instead.

This replaces the multiplayer **steal from active player’s dice slots** rule for solo.

### Voluntary skip

If the player **declines** to use any platter die when a legal platter mark exists,
they **cannot** use the fallback pool (same spirit as the multiplayer “voluntary pass”
rule). Engine: only offer fallback when `legal_platter_moves` is empty.

### Passive actions

- **No reroll** on passive turns.
- Unlock / plus one timing follows general action rules if ever unlocked during
  passive (rare in solo); reroll remains blocked.

---

## Silver die — platter cascade (active or passive)

When the **silver die** is chosen:

1. Mark its value on the silver grid in a **chosen row color** (yellow / blue / green /
   pink).
2. Every die that moves to the platter **because of this pick** (strictly lower than
   silver) must be marked on the silver grid at that die's **face value in the matching
   row color** (yellow die → yellow row, pink die → pink row, etc.). White and silver
   dice sent to the platter are **jokers** — any row at that face value.
3. Dice already on the platter **before** this pick **cannot** be marked from this
   silver pick.
4. Optional cascade marks are skipped automatically when no legal row remains; otherwise
   the player must mark them (no explicit skip action).

White/silver interactions: see [White die](#white-die).

---

## Bonuses — immediate chains

When a mark triggers a bonus (`?` under a field, or completing a row/column):

1. Resolve the bonus **immediately** — it cannot be saved.
2. The bonus mark may trigger **another** bonus → repeat until the queue is empty.
3. Wild (`?`) bonus: choose a color (round 4 black `?` is fully free; field `?` bonuses
   are tied to a color icon on the sheet).
4. Yellow bonus marks require the target cell to **already be circled** before it can
   be crossed. A yellow wild may **circle** any uncircled cell or **cross** any circled
   uncrossed cell (catalog IDs 171–190).

**Engine:** maintain a `pending_bonuses` FIFO queue on `GameState`; enter
`RESOLVE_BONUS` phase while the queue is non-empty. Auto-resolvable bonuses (fox,
action grants, blue/green/pink wild) drain from the head without player actions;
yellow and silver wilds expose catalog mark actions. Follow-up bonuses enqueue at
the **back** of the queue.

**Round 4 black `?`:** at round start, enqueue `Bonus(BONUS_WILD, color=None)`; player
chooses a color via `choose_wild_color_id()` (catalog IDs 146–150), then the chosen wild
resolves through the same queue.

Helpers: `bonus_queue.py`, `bonus_flow.py`.

### Automated bonus marks (engine policy)

For training and simulation, blue / green / pink wild bonuses do **not** require a
player choice:

| Bonus area | Automated mark |
|------------|----------------|
| **Blue** | Write the **last blue value** again (or 12 if blue is still empty). Must still satisfy blue legality (≤ previous slot). |
| **Green** | Write **6 × multiplier** on the first slot of a pair, **1 × multiplier** on the second slot (positive vs negative square). |
| **Pink** | Always write **6** in the next pink slot. |

If the target track is full (12 slots) or no legal automated mark exists, the bonus is
**skipped** so the queue cannot deadlock. Same for impossible yellow/silver wild marks.

Helpers: `automated_blue_bonus()`, `automated_green_bonus()`, `automated_pink_bonus()`,
`can_automated_wild_bonus()` in `doppelt.core.bonus_auto`.

Yellow / silver wild bonuses and action/fox bonuses still need explicit choices or
separate rules (Phase 1.2+).

---

## Actions — timing

Three action tracks: **reroll**, **unlock**, **plus one** (+1).

| Action | Who | When |
|--------|-----|------|
| **Reroll** | Active only | After a roll, before picking; reroll **all** dice currently in hand (not on platter, not on sheet slots). Must reroll all — cannot keep some. |
| **Unlock** | Active only | **Before** rolling; pull one die from platter back into hand. When the hand is empty but unlock is available, the player must choose unlock or end the active turn. |
| **Plus one** | Active or passive | **End of turn**, after normal picks/marks are done; choose any of the 6 dice (even one already used this turn). Each physical die at most once per plus-one action chain. |

Unlock: circling the next slot on an action track when a sheet mark unlocks it.
Using an action crosses off the leftmost circled slot.

**Action track end bonuses:** when a circle fills the last open slot on a track
(`circled + crossed == capacity`), enqueue the pad’s `end_bonus` immediately
(fox on reroll, pink wild on unlock, silver wild on plus-one). Helper:
`circle_action_track()` in `action_flow.py`.

**Replay:** ``seed`` + binary action log (``export_log`` / ``replay_game`` in
``doppelt.replay``) fully determines outcome. Only ``GameState.rng`` inside
``apply_action`` may consume randomness; action pickers must use a separate RNG.

---

## Scoring edge cases

| Area | Edge case |
|------|-----------|
| **Green** | Empty second slot of a pair → 0 for that pair at game end. Stars can go negative mid-game. |
| **Pink** | Any value may be written; field `?` bonus only if value ≥ threshold. |
| **Blue** | Each new value must be ≤ previous slot. Score = star above **last filled** slot. |
| **Yellow** | Circles unlock row/column bonuses when a full line is circled; only **crosses** count toward score. |
| **Silver** | Score each **row** by mark count using the row table; sum rows. |
| **Foxes** | Each fox = points equal to **lowest** of the five color totals. If any color is 0, foxes score 0. |

---

## White die

The white die is wild. It has **no scoring area of its own**. On each use, pick one
mode:

### Mode 1 — Mark as a color (`WhiteDieMode.AS_COLOR`)

Treat the white die as **yellow, green, pink, or silver** (not blue directly).

- Yellow: circle or cross the matching number in the yellow grid (cell choice when ambiguous).
- Green / pink: write the face value in the next slot of that track.
- Silver: mark the face value on the silver grid in a chosen row color
  (yellow, blue, green, or pink row — not the silver die’s own “silver area” row).

Active mode choice catalog IDs: 20 blue sum, 21 green, 22 silver, 23 yellow, 24 pink.

Engine helper: `resolve_white_mark_color()`.

### Mode 2 — Blue sum (`WhiteDieMode.AS_BLUE_SUM`)

When **either** the blue die **or** the white die is used for a blue-track entry,
**always** write `blue_face + white_face` — even if one die is on the platter or
still on the active player’s dice slots. You cannot record only one die’s value.

Engine helper: `blue_entry_value(includes_white=...)`.

The active player may fill **two** blue slots in one round by picking blue on one
roll and white on another (each entry still uses both face values).

### Silver ↔ white interactions

- White used **as silver**: the physical silver die goes to the platter; the silver
  grid mark may use any row color.
- Silver die chosen: white (if discarded to platter) counts as a joker row color for
  extra platter marks triggered by silver.

### Not allowed

- White cannot impersonate blue via color mode (blue always uses sum mode).
- White cannot mark the blue track alone without the sum rule.

---

## Multiplayer (deferred)

When adding 2–4 players later:

| Topic | Multiplayer rule |
|-------|------------------|
| Passive platter | Use dice the **active player** left on the platter (not a fresh roll). |
| Passive steal | If no legal platter die, take from **active player’s dice slots** on the sheet. |
| Passive pick | Each opponent picks one platter die; **same die** may be chosen by multiple players. |
| Turn order | Active rotates left; round ends when each player has been active once. |

---

## Player counts and round count

| Players | Rounds |
|---------|--------|
| 1–2 | 6 |
| 3 | 5 |
| 4 | 4 |
