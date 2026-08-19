# Doppelt So Clever — Project Roadmap

**Goal:** Build a game-playing agent that plays *Doppelt So Clever* (Twice as Clever) well.

**Strategy:** Engine first → reproducible game logs → mass simulation → neural network → evaluate and iterate.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Phase 0 — Foundations](#phase-0--foundations)
4. [Phase 1 — Game Engine](#phase-1--game-engine)
5. [Phase 2 — Simulation at Scale](#phase-2--simulation-at-scale)
6. [Phase 3 — Neural Network Agent](#phase-3--neural-network-agent)
7. [Phase 4 — Evaluate & Iterate](#phase-4--evaluate--iterate)
8. [Rule Parity Checklist (solo)](#rule-parity-checklist-solo)
9. [Milestone Checklist](#milestone-checklist)
10. [Risks & Decisions](#risks--decisions)

---



## 1. Project Overview



### What we're building


| Layer          | Purpose                                                                                         |
| -------------- | ----------------------------------------------------------------------------------------------- |
| **Engine**     | Correct rules, legal moves, scoring, deterministic replay from a seed                           |
| **Action log** | Compact representation of every decision in a full game                                         |
| **Simulator**  | Run millions of games quickly (CPU-first, optional GPU later)                                   |
| **Agent**      | Policy network that chooses moves; trained via self-play or supervised learning on strong games |
| **Evaluation** | Benchmark scores, head-to-head play, analysis tools                                             |




### Game summary (for implementation)

- **Dice:** 6 dice — white (wild), yellow, blue, green, pink, silver.
- **Rounds:** 6 (1–2 players), 5 (3 players), or 4 (4 players). Start with **solo (6 rounds)**.
- **Active turn:** Roll all available dice → pick one → mark score sheet → lower dice go to silver platter → repeat up to 3 picks/rolls.
- **Passive turn:** Each other player picks one die from the platter (same die can be chosen by multiple players).
- **Scoring areas:** Each color has distinct placement and scoring rules (see Phase 1).
- **Bonuses & actions:** Immediate chain resolution (reroll, unlock from platter, plus one pick). This is the hardest part of the engine.
- **Foxes:** Each fox = points equal to your **lowest** color score at game end.
- **Win condition:** Highest total score (solo: maximize your own score).



### Success criteria (high level)

- [ ] Any finished game can be replayed from seed + action log and produce identical final state and score.
- [ ] Engine runs ≥ 10k solo games/sec on your machine (order-of-magnitude target; tune after baseline).
- [x] Trained agent beats random and heuristic (greedy neural ~268 mean, 2026-08-18).
- [ ] Solo greedy mean **300** on a frozen eval suite ([§3.6](#36-algorithm-improvements-268--300)).
- [ ] Clear metrics dashboard: average score, score distribution, training curves.

---



## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      doppelt project                        │
├─────────────────────────────────────────────────────────────┤
│  doppelt/                                                   │
│    core/          # GameState, rules, legal moves, scoring  │
│    actions/       # Action types, encoding/decoding         │
│    engine/        # Step(), apply_action(), game loop       │
│    replay/        # Log format, load, replay, validate      │
│    sim/           # Batch runners, bots (random, heuristic) │
│    ml/            # State tensors, dataset, model, train    │
│    cli/           # play, simulate, train, replay commands  │
│  tests/           # Rule tests, golden replays, property tests│
│  data/            # Score sheet constants, optional logs    │
│  notebooks/       # Exploration & analysis (optional)       │
└─────────────────────────────────────────────────────────────┘
```



### Recommended stack


| Choice     | Recommendation                                                                         | Why                                               |
| ---------- | -------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Language   | **Python 3.11+**                                                                       | Fast iteration, PyTorch ecosystem, easy profiling |
| ML         | **PyTorch**                                                                            | Standard for policy networks; good debugging      |
| Tests      | **pytest**                                                                             | Golden replays and rule regression                |
| Speed path | **NumPy** state arrays → **Rust extension** or **Cython** only if profiling shows need |                                                   |
| Config     | **YAML or TOML**                                                                       | Score sheet layout, training hyperparameters      |




### Core design principles

1. **Determinism:** `seed` + `action_list` → identical outcome. Required for debugging and ML.
2. **Legal moves only:** The engine exposes `legal_actions(state)`; agents never guess validity.
3. **Small action vocabulary:** Encode decisions as discrete actions (die pick, mark target, bonus resolution choices).
4. **Separate RNG from rules:** Dice rolls from a seeded RNG stream; actions consume randomness in a fixed order.
5. **Solo first:** 1-player, 6 rounds avoids multi-agent coupling until the core loop is solid.

---



## Phase 0 — Foundations

**Objective:** Repo structure, score sheet data, and a shared vocabulary before writing rules.

### Tasks

- [x] **0.1** Finalize repo layout (`core`, `engine`, `tests`, etc.) and `pyproject.toml` / dependencies.
- [x] **0.2** Digitize the **score sheet** as data:
  - Yellow: numbers 1–6, circle/cross grid, row/column bonuses.
  - Blue: descending sequence slots, star thresholds.
  - Green: multipliers per slot, pairwise subtraction stars.
  - Pink: threshold per slot, sum scoring.
  - Silver: 4×6 grid (value × color), row scoring table.
  - Fox positions, action tracks (reroll / unlock / plus one), round-track bonuses.
- [x] **0.3** Define **Dice** enum and **Color** enum; document white-die behavior.
- [x] **0.4** Write `docs/rules-notes.md` — edge cases pulled from the rulebook (silver platter timing, passive steal from active sheet, etc.).
- [ ] **0.5** Collect 2–3 **manual score sheets** from real games (your own plays) as future golden-test references. *(Deferred — use point-value tests for now.)*



### Deliverable

Score sheet constants loadable in code; empty `GameState` struct with no rules yet.

**Estimated effort:** 1–2 sessions.

---



## Phase 1 — Game Engine

**Objective:** A correct, replayable engine that records actions that fully determine a game.

### 1.1 State model

Represent explicitly (avoid hiding state in implicit globals):

```
GameState
├── round_index, active_player_index, phase (ROUND_START | ACTIVE_ROLL | ACTIVE_PICK | ...
├── dice_pool: which dice are in hand, on platter, on active sheet slots
├── dice_values: current face values (after each roll)
├── sheets[player]: per-color marks (circles, crosses, numbers, stars, foxes, actions unlocked/used)
├── rng_state (or external seed + roll counter)
└── pending: bonus/action queue (for chained resolutions)
```



### 1.2 Rules modules (implement in order)


| Order | Module                 | Complexity | Notes                                                           |
| ----- | ---------------------- | ---------- | --------------------------------------------------------------- |
| 1     | **Turn structure**     | Medium     | Active 3-phase roll/pick; platter; passive picks; round advance |
| 2     | **Yellow**             | Low        | Circle or cross (only if circled)                               |
| 3     | **Pink**               | Low        | Left-to-right fill; threshold for field bonuses                 |
| 4     | **Blue**               | Medium     | White+blue sum; non-increasing sequence                         |
| 5     | **Green**              | Medium     | Multiply by factor; pairwise star math                          |
| 6     | **Silver**             | **High**   | Color choice + extra platter marks from newly discarded dice    |
| 7     | **White die**          | Medium     | Wild color vs blue-sum modes                                    |
| 8     | **Bonuses**            | **High**   | Immediate free marks; chain until queue empty                   |
| 9     | **Actions**            | **High**   | Reroll, unlock from platter, plus one (timing constraints)     |
| 10    | **Foxes**              | Low        | Mark foxes; end-game valuation                                  |
| 11    | **Scoring**            | Medium     | Per-area formulas + fox multiplier                              |
| 12    | **Passive edge cases** | Medium     | Steal from active sheet if platter unusable                     |




### 1.3 Action log format

Design for **complete replay**, **ML training**, and **minimal storage at scale**.

#### Principle: numbers, not objects

Every decision in a game is an integer index. The log is a **fixed header** plus a **flat array of action IDs**. No JSON per move, no string labels, no redundant roll data (dice come from the seeded RNG).

#### Global action catalog

Define a versioned **Action Catalog** (`action_catalog_v1`) — a fixed, total ordering of every semantically distinct action the engine can ever apply. Examples:


| Global ID | Action (conceptual)         |
| --------- | --------------------------- |
| 0         | Pick white die from hand    |
| 1         | Pick yellow die from hand   |
| …         | …                           |
| N         | Mark pink slot 0            |
| …         | …                           |
| M         | Use reroll action (track 1) |
| …         | …                           |


Rules for the catalog:

1. **Stable across games** — action ID 42 always means the same thing (e.g. “cross yellow number 4”).
2. **Versioned** — bump catalog version when ordering or semantics change; old logs stay decodable.
3. **Same order as ML mask** — the policy network’s action head uses this same index space; legal moves are a bitmask over the catalog.
4. **Sorted definition** — document ordering explicitly (e.g. by phase → color → slot → sub-choice) so two implementations agree.

At each step the agent picks a **global catalog ID**. The engine validates it is legal; replay applies the same ID. We do **not** store “index into current legal list” — that would shrink bytes slightly but breaks direct use as training labels and requires recomputing legality to interpret a log.

#### Binary log layout

```
┌──────────────────────────────────────────┐
│ Header (fixed size, e.g. 16 bytes)     │
│   magic, catalog_version, player_count,  │
│   seed (u64), action_count (u32)         │
├──────────────────────────────────────────┤
│ Actions: action_count × uint16 LE        │
│   [id₀, id₁, id₂, …]                     │
└──────────────────────────────────────────┘
```


| Field      | Size           | Notes                                              |
| ---------- | -------------- | -------------------------------------------------- |
| Header     | ~16 B          | Once per game                                      |
| Per action | 2 B (`uint16`) | ~50–150 actions/game typical → **~100–300 B/game** |
| 1M games   | ~100–300 MB    | Raw; batch with zstd often cuts further            |


Use `uint8` only if catalog stays < 256 actions; `uint16` is safer once silver, bonuses, and passive play are included.

#### What is *not* in the log


| Omitted                  | Why                                                                   |
| ------------------------ | --------------------------------------------------------------------- |
| Dice roll results        | Deterministic from `seed` + roll counter in engine                    |
| Timestamps, player names | Not needed for replay or training                                     |
| Full state snapshots     | Recompute via replay; store separately only for ML datasets if needed |
| JSON wrappers            | Debug tooling can export human-readable views on demand               |




#### Replay

```python
state = new_game(seed=log.seed, player_count=log.player_count)
for action_id in log.actions:
    state = apply_action(state, action_id)  # must be legal
assert is_terminal(state)
```

Golden tests: `seed + action_ids → final score + sheet hash`.

#### Debug / human-readable export (optional)

For development only — not stored at scale:

```text
seed=42  catalog=v1  actions=[12, 45, 45, 102, …]
# or: doppelt decode log.bin --pretty
```



#### Catalog maintenance

- [ ] Ship `action_catalog_v1.json` (or YAML) listing every ID and its meaning — source of truth for tests and docs.
- [x] Unit test: catalog is contiguous-enough (holes allowed), no duplicate semantics, size matches `ACTION_SPACE_SIZE`.
- [x] Unit test: random legal play → encode IDs → replay → identical terminal state.



#### Relation to Phase 3

The neural network’s policy output dimension = `ACTION_SPACE_SIZE`. The legal-action mask is a bitmap over the same indices. Training rows can be `(state_encoding, action_id, final_score)` without translation layers.

### 1.4 Engine API

```python
state = new_game(seed=42, player_count=1)
legal = legal_actions(state)
state = apply_action(state, action)
done, scores = is_terminal(state)
replay = export_log(state)       # bytes: header + uint16 action array
state2 = replay_game(log)        # seed + action_ids → must match
```



### 1.5 Testing strategy

- [ ] **Unit tests** per color area (placement legality, scoring).
- [x] **Golden replay tests** — load action log, assert final score and sheet state.
- [ ] **Property tests** — `apply` only legal actions; score monotonicity where applicable.
- [ ] **Rulebook examples** — encode official examples from the PDF as tests.
- [x] **Fuzz test** — random legal play for 10k games; no crashes, terminal always reached.
  Baseline 2026-08-12: 10k random solo games in **18.424s** (~**543 games/s**); all reached `GAME_OVER`. Re-run: `python tests/test_fuzz.py`. After sim loop + trusted apply + legality early-exit, single-process ~**916 games/s**.



### Phase 1 milestones


| Milestone | Definition of done                                                |
| --------- | ----------------------------------------------------------------- |
| **M1.1**  | Solo game completes with yellow/pink/blue only (stub other areas) |
| **M1.2**  | Full solo game with all colors, no bonuses/actions                |
| **M1.3**  | Bonuses + action tracks working with chain resolution             |
| **M1.4**  | Full rule parity with rulebook; golden replays pass               |
| **M1.5**  | CLI: `doppelt play` (human) and `doppelt replay <log>`            |


**Estimated effort:** 4–8 weeks part-time (rules complexity is real).

---



## Phase 2 — Simulation at Scale

**Objective:** Run millions of games; collect datasets; measure baseline strength.

### 2.1 Performance

- [x] Profile hot path (`legal_actions`, `apply_action`, scoring).
  - ~2× `legal_action_ids` per step (`play` loop + `apply_action` re-check).
  - `_white_mark_modes` / silver legality checks dominate `legal_action_ids`.
  - Was scoring every step via `is_terminal()`; random loop now checks `phase == GAME_OVER`.
- [x] Batch simulation: `doppelt.sim.run_batch` + `doppelt simulate --games N --workers W`.
- [x] Target: benchmark **games/sec** and **actions/sec** on your hardware.
  - 2026-08-12 (this machine): **1 worker 10k = 916 games/s, 59.8k actions/s**;
    **16 workers 10k = 4.71k games/s, 307k actions/s**
    (was 667 / 3.94k before trusted `apply_action` + early-exit white/silver legality).
  - Cheap wins done: `apply_action(..., check_legal=False)` in sim/random loops;
    `_any_white_mark_legal` / `can_use_silver_value` early-exit.
  - Remaining: still ~2× short of 10k games/s; next would be deeper `legal_action_ids` work or numba/Rust.
- [ ] Optional: `numba` / Rust extension if Python is too slow after optimization.



### 2.2 Bots (baselines)

Each bot implements ``Policy.select(state, legal) -> action_id``. Factory: ``make_policy(name, seed)``.
CLI: ``doppelt simulate --policy random_legal|greedy_immediate|heuristic|mcts_lite``.

| Bot                      | Status | Purpose                                                        |
| ------------------------ | ------ | -------------------------------------------------------------- |
| **RandomLegal**          | ✅     | Uniform over legal actions                                     |
| **GreedyImmediate**      | ✅     | One-step ``total_score`` max; random among ties                |
| **Heuristic**            | ✅     | Score + setup + 1–2 ply on mark choices; see below             |
| **MCTS-lite**            | ✅     | Shallow UCB + Heuristic rollouts; chance-sampled dice          |

#### Heuristic rules (`doppelt.sim.heuristic`)

Scores `total_score` plus round-scaled setup (fox coverage, yellow lines/families, blue descent, silver chains, green lefts, unused action circles). Instant marks are compared to white/yellow/silver choice follow-through (depth ≤ 2).

- **Keep rolling.** Emptying the hand on pick 1 or 2 (no 2nd/3rd roll) is heavily penalized.
- **Acceptable dice / reroll.** Even-row yellow, good green, not-too-low blue (by round), or pink that hits its field bonus count as “acceptable” (white can impersonate those). If none are showing and a reroll is available, reroll.
- **Yellow families.** Commit to even rows (0/2/4) *or* odd (1/3); even is the default bias. Mixing families is penalized.
- **Late color targets.** From round 5, push toward ≥21 yellow and ≥28 blue if those aren’t available yet.
- **Silver.** ≥4 open boxes from a pick is a sweep (very good). 1–2 marks with no column bonus and no column at 3 is weak. **Two unlocks + 6 silver marks** is a special line: take it even if the hand empties, then unlock white, then the die with the best expected acceptable hit.
- **Pink 4–7.** Hitting the slot bonus is good; missing it is bad. **Pink 5 & 6** (need 5+ / 6+) are hard: save yellow row-4 and blue slot-6 pink wilds for those slots, and set those wilds up so they can fire then.
- **Green faces.** Prefer 4–6 on slots 0/2, 5–6 on 4, 1–3 on 1, 1–2 on 5. **Low green** (right half of a pair) is easy on roll 1 or passive; on active rolls 2–3, prefer a higher die instead.
- **Round 4 wild.** If auto-green would be a positive 6 (left slot), take green most of the time.
- **Foxes.** Slight extra value for live colors, banked foxes, and claiming a fox (each fox × lowest color at end).




#### MCTS-lite (`doppelt.sim.mcts_lite`)

Same `Policy` interface. Mark-choice, bonus, passive, and plus-one still use Heuristic. On **active pick** only (≤10 legal actions) it runs a small UCB1 tree at the root (~32 sims):

1. Pick a legal catalog ID via UCB (normalized by ~200 pts).
2. `copy_for_trial` and **reseed** trial dice RNG so rollouts don’t share faces.
3. Apply the action (`check_legal=False`).
4. Short **random** rollout (default horizon 4): sample `roll_hand` only in `ACTIVE_PICK`, otherwise random legal.
5. Backup `evaluate_state` (or `total_score` if the trial ended).

Play the most-visited root action. Intended as a stronger baseline than Heuristic, not a 1M-game generator.

### 2.3 Dataset generation

CLI: `doppelt dataset generate` → `data/datasets/solo_v1/` (gitignored).

- [x] Compact **DPLD** shards: magic `DPLD`, catalog v1, one policy per shard, per record `seed` + terminal + scores (total + 6 areas) + `uint16[]` actions. Manifest `manifest.json` lists mix, seed ranges, unfinished.
- [x] Metadata in-record + manifest (bot type / policy folder, final score, per-color breakdown). Replay via `record_to_game_log` → `replay_game`.
- [x] Default mix **heuristic:5 / greedy_immediate:3 / random_legal:2**. `mcts_lite` is not a dataset policy (too slow).
- [x] Run **1M+** mixed-bot solo games (shard size 50k, resume skips existing shards). 2026-08-12: `data/datasets/solo_v1/` — 500k heuristic / 300k greedy / 200k random; 78 unfinished (heuristic only).
- [ ] Optional Parquet export (pyarrow `[ml]` extra) if columnar training I/O is needed later.



### 2.4 Analytics

CLI: `doppelt dataset analyze [--json analytics.json]` reads DPLD shards (no full replay).

- [x] Score distributions per bot (mean/std/percentiles + total-score histogram).
- [x] Color-area histograms and fox impact (nonzero rate, fox share of total, mean total with vs without fox).
- [x] Average actions per game; yellow-bonus action count / streak (logged chain proxy); round-4 wild rate.
- [x] Degenerate-strategy flags: silver-first / silver-pick share, silver as top color, zero-color (no fox), passive skip, white-as-silver.



### Phase 2 milestones


| Milestone | Definition of done                                |
| --------- | ------------------------------------------------- |
| **M2.1**  | 100k games batch run completes reliably ✅ (2026-08-12, 16 workers, 0 unfinished) |
| **M2.2**  | 1M game dataset with logs on disk ✅ (2026-08-12, `data/datasets/solo_v1`, 78 unfinished heuristic) |
| **M2.3**  | Baseline report: random vs heuristic scores ✅ (`doppelt dataset analyze`) |
| **M2.4**  | ≥ 10k games/sec (or documented bottleneck + plan). Current: ~916/s single-process, ~4.7k/s ×16 workers. |


**Estimated effort:** 2–4 weeks after Phase 1.

---



## Phase 3 — Neural Network Agent

**Objective:** Learn a policy that improves with training data and/or self-play.

### 3.1 State representation

Encode `GameState` as fixed-size tensors (for solo first):

- **Sheet channels:** per-area occupancy, numbers, circles, crosses (multi-channel binary/float).
- **Dice:** values + location (hand / platter / slot) × 6 dice.
- **Global:** round, phase, unlocked actions, pending queue summary.
- **Mask:** legal action bitmap (same size as action space).

Keep encoding versioned (`encoding_v2`) so you can change without breaking old models.

**Shipped:** `doppelt.ml.encoding` (`ENCODING_VERSION = 2`) plus a catalog-sized legal mask. Default net is `pvn_v1` (see 3.3). Train with `doppelt train bc`; play with `--policy neural --checkpoint`. encoding_v1 checkpoints will not load.

### 3.2 Action space

Fixed discrete space with **legal mask** (`ACTION_SPACE_SIZE = 192`, same indices as the catalog):

- [x] **Phase-specific subsets** — `doppelt.actions.space.PHASE_ACTION_IDS`; active pick splits pre-roll vs post-roll via `candidate_action_ids(state)`.
- [x] **Composite pick+mark** — blue / green / pink picks apply the only mark; yellow / white auto-apply when the cell or mode is unique; follow-up IDs cover remaining choices.
- [x] **Illegal actions masked to −inf** before softmax (`ILLEGAL_LOGIT` in the policy head).

Holes in 0–191 are unused padding (stable IDs). `SILVER_SKIP_CASCADE` (124) stays in the catalog for old logs but is never legal.

Size may be large (silver color choices, bonus targets). Start with explicit enumeration; compress later if needed.

### 3.3 Model architecture (starting point)


| Component   | Shipped (`pvn_v1`)                                      |
| ----------- | ------------------------------------------------------- |
| Backbone    | Sheet MLP + 1D CNN over 6 dice + context MLP, fused     |
| Policy head | Linear → masked softmax (`ILLEGAL_LOGIT` = −inf)        |
| Value head  | Linear → scalar (final score / `SCORE_SCALE`)           |
| Size        | Default hidden 256 is ~100k–1M params; `--arch mlp` kept for old BC checkpoints |


**Shipped:** `doppelt.ml.model` — `build_net(architecture="pvn_v1")`. Retrain BC after this change; checkpoints without `architecture` still load as `mlp`.




### 3.4 Training approaches (try in order)

1. **Behavioral cloning** — imitate heuristic bot games from Phase 2 (`doppelt train bc`).
2. **Supervised on best-of-N** — label actions from best rollout among N random continuations. *(not yet)*
3. **Policy gradient / Actor-Critic** — reward = final score / `SCORE_SCALE` (`doppelt train selfplay`).
4. **Self-play** — current policy samples solo games vs the dice RNG; optional `--kl` toward the BC init; best checkpoint by greedy eval score.
5. **Play-time PUCT** — `--mcts-sims N` on `doppelt eval` / `doppelt random --policy neural` (does not slow training).

For solo roll-and-write, **final score RL** is natural; consider auxiliary rewards for fox setup only if training is too sparse.

### 3.5 Training loop

- [x] `Dataset` from DPLD shard replay or live bot games; optional Parquet later.
- [x] Train/val split by seed (`seed % 10`, `--val-frac`).
- [x] Log: train/val loss; eval vs heuristic/random via `doppelt eval`.
- [x] Checkpoint best BC model by **val loss**; self-play keeps best by **greedy eval score**.



### 3.6 Algorithm improvements (268 → 300)

**Status (2026-08-18):** Greedy neural policy **~268** mean, heuristic far behind (~100 pts). Target **300** mean on a fixed eval seed set (same seeds, ≥256 games; 1k later). Dice variance means 300 is an **expected-score** goal, not a score every game.

Heuristic BC is **low leverage** now (weaker teacher). Do not clone heuristic shards as the main path. Overnight A2C from the current net still helps a bit, then plateaus. Items below are **highest leverage first**. Each should be judged by greedy mean on the **same** `--seed` suite before stacking the next.

| # | Change | Why it moves 268→300 | Cost | Status |
|---|--------|----------------------|------|--------|
| **3.6.1** | **Expert iteration (search as teacher)** — play/train with PUCT (`--mcts-sims`), store visit-greedy (or visit-sampled) actions, BC or distill onto the net, repeat | Net is already stronger than heuristic; the remaining teacher is **itself + search**. This is the AlphaZero loop. Play-time search alone does not update weights. | High (slow games if used in the inner loop). Start: generate a few k search games, BC, eval. Do **not** put MCTS inside every A2C iter until rollouts are batched. | ✅ `doppelt train expert --init … --mcts-sims 32` (greedy eval on `dev`; A2C still has no MCTS) |
| **3.6.2** | **Deeper PUCT + chance nodes** — current search is root-only: one action, forced uniques, value leaf. Add a real tree, sample `roll_hand` as chance, maybe 1–2 ply of replies | 268-level blunders are often “this pick vs the *distribution* of remaining dice,” not one-step value error. | Medium–high. Keep `--mcts-sims` for eval; cap depth. | ✅ tree PUCT, `roll_hand` as chance (not a forced unique), `--mcts-plies` default 2 (root + one reply). Expert labels and `--mcts-sims` eval share it. A2C still has no MCTS. |
| **3.6.3** | **`encoding_v2` (features, not new action IDs)** — explicit pick 1/2/3; die rank low→high (tie-break documented); “would this empty the hand”; live color totals / fox floor; remaining round-scaled targets (yellow 21, blue 28, pink 5–6, silver columns) | Policy still has to *infer* pick index and rank from a flat vector. Extra catalog IDs for rank×pick are the wrong tool (sparse, v2 logs). | Low–medium. Retrain or fine-tune; bump `ENCODING_VERSION`. | ✅ `ENCODING_VERSION = 2`; progress block after pending. Old v1 `.pt` files will not load. |
| **3.6.4** | **Better return estimator (TD / GAE); treat PPO as optional** — bootstrap value between picks (`TD(λ)` as in TD-Gammon) or GAE. **Do not assume PPO beats A2C**: Yahtzee (Papé 2025, arXiv:2601.00007) found A2C more robust than PPO on a stochastic scorecard under a fixed budget. Keep A2C; add bootstrapping first. | Sparse 6-round MC return is noisy. Tesauro’s breakthrough was *per-turn* TD, not waiting for the game to end. | Medium. Same collector, different backup. | ⬜ |
| **3.6.5** | **Auxiliary value heads** — predict per-color totals + foxes + “any color is 0”; main head stays grand total | Foxes = count × min(colors) (0 if any color is 0). A scalar value often misses “don’t leave pink at 0.” | Low–medium. Extra losses, same encoder. | ⬜ |
| **3.6.6** | **Self-distill high-score games** — from the current net, keep top percentile of greedy (or search) games, BC on those actions only | Stronger than heuristic BC; cheap compared to search-in-the-loop. Filter unfinished games if analytics still flags them. | Low (replay logs you already know how to write). | ⬜ |
| **3.6.7** | **Batched GPU / faster leaves** — vectorize encode + forward; later numba/Rust on `legal_action_ids` | Unlocks 3.6.1–2 at overnight scale. Does not raise the ceiling by itself. | Medium. Profile first (encode vs engine vs torch). | ⬜ |
| **3.6.8** | **KL / entropy schedule** — decay `--kl` and entropy as eval rises; freeze a *BC-or-best* anchor separately from stage init | Constant `kl=0.01` toward last stage can freeze a 268 local max **or** slowly forget tactics. Schedule + keep `stage_*.pt`. | Low (script/CLI). | ⬜ |
| **3.6.9** | **Light shaping, eval unshaped** — small penalties for a 0-color at end, emptying hand on pick 1–2 without unlock plan; optional fox-coverage bonus. **Report only raw `total_score`** | Speeds learning the fox/zero-color cliff that caps many 250s. Overweighting distorts 300. | Low. Easy to get wrong — A/B on the eval suite. | ⬜ |
| **3.6.10** | **Per-area encoders / attention** — replace flat sheet MLP with yellow/blue/green/pink/silver towers (or a small transformer over slots) | 300 play is “which *region* is the bottleneck this round.” `pvn_v1` is a start; capacity may underfit sheet structure. | Medium. New `--arch`, new BC or fine-tune. | ⬜ |
| **3.6.11** | **Checkpoint league** — mix rollouts vs frozen older nets (and vs greedy-self) so the policy does not overfit one value function | Overnight stages can overfit eval seed 20000. League = more diverse states. | Medium. | ⬜ |
| **3.6.12** | **Root expectimax on active picks** — exact one-ply over legal picks, chance over remaining faces / platter, net value at leaves | Lower variance than PUCT for the 3-pick decision; expensive branching. Use only at play/eval or for labels. | High at large width. | ⬜ |
| **3.6.13** | **Phase-specific heads** — separate logits for pick vs mark vs bonus (still global catalog IDs + mask) | Less wasted capacity on never-legal IDs in that phase. | Low–medium. | ⬜ |
| **3.6.14** | **Human / target games** — log your own 280–300 solos; BC a small mix so the net sees “300-shaped” sheets | Tiny dataset, high unique tactics (pink 5–6, late fox). Easy to overfit — mix with self-play, never replace eval seeds. | Low. | ⬜ |
| **3.6.15** | **Failure mining** — `dataset analyze`-style stats on neural logs: zero-color rate, silver-first, skipped pink-6, unused plus-one, fox=0 with 4 colors alive | Tells you whether 268 is **tactics** (encoding/search) or **one systematic bug**. | Low. Build on existing analytics. | ✅ `doppelt eval` (default) + `sim.failures`; analyze also reports %≥300 and fox=0/4+ colors |
| **3.6.16** | **Eval discipline for 300** — freeze 256–1000 seeds; report mean, median, p10, %≥300; never retune on that set; use a second holdout | Otherwise “300” is noise or leakage. | Low. Do this **before** claiming a new SOTA. | ✅ suites `report` / `holdout` / `report_1k` / `dev`; `doppelt eval --suite report` |
| **3.6.17** | **Ensemble at eval** — average 2–3 snapshots’ logits (or vote after short PUCT) | Cheap 2–5 pts sometimes; not a training story. | Low at eval time. | ⬜ |
| **3.6.18** | **Do not** explode the catalog (pick × rank × pick-index IDs); **do not** BC the 1M mixed random/greedy set; **do not** 15k A2C iters as the main 300 plan | Those spend compute without new information. | — | — |

#### From related work (methods we were not using)

Closest published cousins: **solitaire Yahtzee** (optimal DP ~254.6 expected; A2C nets ~242), **Qwixx** (exact/approx chance EV), **TD-Gammon** (dice + neural value + 1-ply afterstates), **Stochastic/Gumbel MuZero** (chance nodes; search with few sims), **2048** (stochastic puzzle, learned chance). No serious public *Doppelt so clever* / *Ganz schön clever* research agent turned up — the official apps are not this.

| # | Change | Source / why it is new for us | Cost | Status |
|---|--------|-------------------------------|------|--------|
| **3.6.19** | **Afterstate (post-decision) value + 1-ply greedy** — for each legal pick, `apply` in a trial, score `V(afterstate)`; play argmax. Train `V` on afterstates, not only pre-decision states | **TD-Gammon**: Tesauro did not learn Q(s,a); he evaluated the *board after the move* and 1-ply searched. Our PUCT still evaluates mixed pre/post states and samples. Afterstates make pick-1 vs pick-2 comparable given the *current* faces (chance is the *next* roll). | Low–medium. Engine already has `copy_for_trial`. | ⬜ |
| **3.6.20** | **Gumbel / Sequential Halving at the root** — replace PUCT when `--mcts-sims` is 8–64 | **Gumbel MuZero** (Danihelka et al., ICLR 2022): PUCT is a poor simple-regret algorithm at *low* sim counts. We cannot afford 800 sims/move; Gumbel is built for that budget. | Medium. Swap root selection; keep the engine. | ⬜ |
| **3.6.21** | **Train on the search *policy*, not only the argmax** — KL/CE to visit counts (or completed Q); temperature then greedy later in the game | **AlphaZero / OpenSpiel**: labels are the full π_search. We planned “expert iteration” as cloning the chosen move only. Distilling the distribution teaches “yellow and white were both fine.” | Medium. Needs 3.6.1 logging of visits. | ⬜ |
| **3.6.22** | **Enumerate small chance, sample large chance** — remaining hand size 1–3: exact 6^k (or 6^k / symmetries) expected value; only Monte-Carlo the 4–6 die rolls | **Qwixx solvers**: 1000 sampled rolls ≈ exact EV; 10 samples are *biased high* in the planner and worse in real play. Our MCTS currently samples chance like “N = sims,” which is the 10-sample regime. | Medium. Tables or on-the-fly product of d6. | ⬜ |
| **3.6.23** | **Round-6 / plus-one endgame DP** — exact (or depth-complete) backup when few marks remain; net only for earlier rounds | **Yahtzee / Qwixx**: optimal play is computed *backwards from the terminal sheet*. Full Doppelt is too big, but **last active + passive + plus-one** may be enumerable with the real engine. Use as labels or as the leaf instead of `V`. | High to engineer; huge if it fits in RAM. Probe state-space first. | ⬜ |
| **3.6.24** | **Distributional / quantile value** — predict a histogram (or quantiles) of final score, not one mean | Dice: two picks can share E[score] but differ in P(≥300) and P(fox=0). **C51 / QR-DQN**. Optional train on P(≥300) or CVaR if the *goal* is the 300 line, not the mean. | Medium. Extra head; eval still reports mean. | ⬜ |
| **3.6.25** | **Hierarchical option: choose a color (or “fox insurance”) then a die** — two-level policy; low-level still catalog IDs | **Yahtzee “Dynamic Intuition” / hierarchical RL**: first commit to a *category*, then tactical keep/roll. Matches “which area is the bottleneck.” | Medium. Risk of bad options; mask options that have no legal mark. | ⬜ |
| **3.6.26** | **Curriculum from the end** — train only on round 5–6 (or plus-one) with random/net prefixes, then unlock earlier rounds | Same Yahtzee work: long-horizon *upper bonus* never learned well from full games. Fox × min(color) is our upper-bonus analogue. | Low–medium. Need a “start at round k” engine hook. | ⬜ |
| **3.6.27** | **Reanalyse** — replay stored games with the *current* net (and cheap 1-ply/Gumbel) to refresh value/policy targets without new dice | **MuZero Reanalyse**: env steps are our bottleneck; we already have action logs. | Medium. Replay is CPU; no new overnight games required. | ⬜ |
| **3.6.28** | **Hand-crafted *progress* features** (Tesauro: raw board → intermediate; features → master) — e.g. expected points if this color is abandoned, “fox dead” bit, silver-column-at-3, pink slot threshold remaining | Encoding_v2 occupancy is still “raw sheet.” TD-Gammon’s jump was *domain features*, not a bigger MLP. | Low. Overlaps 3.6.3; this is the *heuristic-shaped* slice. | ⬜ |

**Suggested attack order:** 3.6.16 (measure) → 3.6.15 (why 268) → **3.6.19 (afterstate 1-ply)** → 3.6.3/28 (features) → 3.6.6 (self-distill) → **3.6.4 TD/GAE, keep A2C** → **3.6.20 Gumbel** → 3.6.22 exact small chance → 3.6.1+21 search teacher → 3.6.23 endgame if feasible. Overnight A2C can run in parallel but is not the 300 path by itself.

**Play-time vs train:** `--mcts-sims` already helps a *played* game without touching weights. Hitting **300 greedy mean** almost certainly needs **3.6.1 / 3.6.21** (weights trained on search), not only more sims at eval.



### Phase 3 milestones


| Milestone | Definition of done                                          |
| --------- | ----------------------------------------------------------- |
| **M3.1**  | State/action encoding implemented and tested ✅ (`encoding_v2`) |
| **M3.2**  | BC model beats RandomLegal by wide margin ✅            |
| **M3.3**  | RL/self-play improves over BC on eval suite ✅ (greedy net ~268 vs heuristic, 2026-08-18) |
| **M3.4**  | 10k eval games: report mean/median score + comparison table |
| **M3.5**  | Mean **300** on a frozen ≥256-seed greedy suite (see [3.6](#36-algorithm-improvements-268--300)) |


**Estimated effort:** 3–6 weeks (tuning-heavy).

---



## Phase 4 — Evaluate & Iterate

**Objective:** Decide what's next based on data — don't guess.

### 4.1 Evaluation suite

- [x] **Fixed seed suite** — `report` (256 from 1_000_000), `holdout` (256 from 2_000_000), `report_1k` (1000 from 1_000_000). Self-play uses `dev` (20_000).
- [x] **Score stats** — mean, median, std, p10, %≥300 on `doppelt eval` (histogram still on `dataset analyze`).
- [ ] **Head-to-head** — if multi-player engine exists.
- [ ] **Human comparison** — your own solo scores on same seeds (optional).



### 4.2 Analysis questions to answer


| Question                              | Tool                             |
| ------------------------------------- | -------------------------------- |
| Which colors are weakest?             | Per-color score breakdown        |
| Does the agent misuse silver / foxes? | Action frequency by phase        |
| Are bonus chains handled well?        | Chain depth vs score correlation |
| Is it overfitting to bot style?       | Eval on human replay logs        |




### 4.3 Possible next steps (decide after data)

- Multi-player (3–4 players) engine + opponent modeling.
- Stronger search: MCTS with policy prior ✅ play-time PUCT (`--mcts-sims`; not used in `train selfplay`).
- Larger model / transformer over sheet regions.
- Distributed self-play on cloud GPU.
- Human UI: web score sheet + agent suggestions.
- Export to mobile for pass-and-play practice.



### Phase 4 milestones


| Milestone | Definition of done                                                    |
| --------- | --------------------------------------------------------------------- |
| **M4.1**  | Eval report document with charts                                      |
| **M4.2**  | Top 3 bottlenecks identified (rules bug, encoding, exploration, etc.) |
| **M4.3**  | Next phase chosen with concrete success metric                        |


---

## Rule Parity Checklist (solo)

Cross-check against `docs/rules-notes.md`, `data/score_sheet/score_sheet_v1.yaml`, and the
official rulebook. **Checked** = implemented and covered by tests where noted; **Open** = not
yet done or known incorrect vs rulebook.

### Turn & round structure

| Status | Rule |
|--------|------|
| ✅ | 6 rounds; active phase then passive phase each round |
| ✅ | Active turn: up to 3 roll/pick cycles; must roll before pick |
| ✅ | On pick: mark sheet; dice strictly lower than pick → platter |
| ✅ | After 3rd pick (or no dice left): remaining hand dice → platter |
| ✅ | Forfeit pick — skip mark but consume one pick slot |
| ✅ | Round-start action unlocks: reroll (R1), plus one (R2), unlock (R3) |
| ✅ | **Round 4 black `?`** — free-color wild; player chooses color (catalog IDs 146–150) |
| ✅ | Passive: roll 6, three lowest to platter; RNG tie-break |
| ✅ | Passive platter pick; pool fallback only when no legal platter mark |
| ✅ | Passive voluntary skip — catalog ID 46; no pool fallback when platter mark exists |
| ✅ | Passive reroll blocked |
| ✅ | Plus-one phase after active and after passive |
| ✅ | End-game plus-one — unused plus-one actions still usable after round 6 |

### Color areas — marking

| Status | Rule |
|--------|------|
| ✅ | Yellow: circle on first visit, cross on second |
| ✅ | Yellow: player chooses cell when multiple match (active: 10–19; passive: 151–160; plus-one: 161–170) |
| ✅ | Blue: left-to-right, non-increasing values, white+blue sum always |
| ✅ | Green: die face × slot multiplier; pairwise star = difference |
| ✅ | Pink: left-to-right fill; any value 1–6 |
| ✅ | Silver: value + row color; column-complete bonuses |

### White die

| Status | Rule |
|--------|------|
| ✅ | Blue sum mode (`blue + white`, even if blue on platter/sheet) |
| ✅ | Yellow mode — circle/cross via yellow cell choice (catalog 23 → 10–19 / 151–160 / 161–170) |
| ✅ | Green mode |
| ✅ | Pink mode — write face in next pink slot (catalog 24) |
| ✅ | Silver mode — any row color; physical silver die stays in hand unless strictly lower |

### Silver die

| Status | Rule |
|--------|------|
| ✅ | Primary mark required; cascade marks for dice sent to platter on this pick |
| ✅ | Optional cascade skip removed — auto-skip only when no legal row; otherwise must mark |
| ✅ | Silver cascade marks locked to platter die color row (white/silver dice = joker) |
| ✅ | Cannot choose silver if face value already fully marked |
| ✅ | Passive / plus-one silver — single mark, no cascade |
| ✅ | White-as-silver: silver die discarded to platter; cascade includes it |

### Bonuses & chains

| Status | Rule |
|--------|------|
| ✅ | FIFO queue; resolve immediately; follow-ups append to back |
| ✅ | Field `?` under blue / green / pink marks (pink threshold enforced) |
| ✅ | Silver column-complete bonuses |
| ✅ | Auto-resolve: fox, action-track circles, blue/green/pink wild |
| ✅ | Player choice: yellow wild (cross), silver wild (grid mark) |
| ✅ | Skip impossible wild bonuses (full track / no legal mark) |
| ✅ | **Yellow wild can circle** — catalog IDs 171–180 circle, 181–190 cross |
| ✅ | **Yellow row completion** — triggers when row fully **circled** (not cross) |
| ✅ | **Yellow column completion** — bottom-edge bonus when column fully **circled** |
| ✅ | **Yellow bottom-edge bonuses** — wired via `bottom_edge_bonuses[col]` on column circle |
| ✅ | **Action track end bonuses** — fox (reroll bar), pink wild (unlock bar), silver wild (plus-one bar) when last slot circled |
| ✅ | **Round 4 free-color wild** — enqueue + player color choice |

### Actions

| Status | Rule |
|--------|------|
| ✅ | Reroll — active only, after roll, reroll entire hand |
| ✅ | Unlock — active only, before roll; empty hand + unlock available → unlock or end turn |
| ✅ | Plus one — end of turn; use any of 6 dice at **current face** (no re-roll); each die once per chain |
| ✅ | Action track state — circle on grant, cross leftmost on use |

### Scoring & game end

| Status | Rule |
|--------|------|
| ✅ | Yellow — points from cross-count table |
| ✅ | Blue — star above last filled slot |
| ✅ | Pink — sum of written values |
| ✅ | Green — sum of pairwise stars (empty pair slot → 0) |
| ✅ | Silver — row mark-count table, sum rows |
| ✅ | Fox counter incremented when fox bonus resolves |
| ✅ | **Fox end-game score** — each fox = lowest of five color totals; 0 if any color is 0 |
| ✅ | **`total_score()`** — include fox points in grand total |
| ✅ | **`is_terminal()` scores** — return correct fox breakdown (currently hard-coded 0) |

### Engine infrastructure (Phase 1 deliverables)

| Status | Item |
|--------|------|
| ✅ | `GameState.action_log` appended on every `apply_action` |
| ✅ | Binary log export / import (`replay` module) |
| ✅ | CLI `doppelt play` / `doppelt replay` / `doppelt random` / `doppelt decode` |
| ✅ | Golden replay tests (seed + action IDs → final sheet/score) |
| ⬜ | Shipped `action_catalog_v1.json` + catalog size test |
| ⬜ | Manual score-sheet references for golden tests (Phase 0.5) |

### Multiplayer (explicitly deferred)

| Status | Rule |
|--------|------|
| ⬜ | 2–4 players; 6 / 5 / 4 rounds by count |
| ⬜ | Passive uses active player's platter (not fresh roll) |
| ⬜ | Passive steal from active dice slots when platter unusable |
| ⬜ | Multiple passives may take the same platter die |

---

## Milestone Checklist

Use this as a living progress tracker. See [Rule Parity Checklist (solo)](#rule-parity-checklist-solo) for the full rule-by-rule list.

### Phase 0

- [x] Repo + dependencies
- [x] Score sheet data file + **action catalog v1**
- [x] Rules notes doc



### Phase 1

- [x] Turn loop (solo) — active pick + passive pick, 6 rounds
- [x] Yellow / pink / blue / **green** / **silver** marking (core legality)
- [x] Bonuses + chains — FIFO queue; field bonuses; auto blue/green/pink wild; yellow/silver wild
- [x] Action tracks — reroll / unlock / plus one state, timing, round 1–3 grants
- [x] Scoring + foxes — color area totals, fox = lowest × count, `total_score()`, `is_terminal()`
- [x] Action log replay (binary export/import)
- [x] CLI play / replay / random / decode
- [x] Golden tests pass



### Phase 2

- [x] Batch simulator
- [x] Baseline bots (RandomLegal, GreedyImmediate, Heuristic, MctsLite)
- [x] 1M game dataset
- [x] Baseline report



### Phase 3

- [x] Encoding + model (`encoding_v2`, `pvn_v1` policy/value, legal mask)
- [x] BC training works (beats RandomLegal)
- [x] Self-play / RL loop (`doppelt train selfplay`); greedy net ~268 vs heuristic (2026-08-18)
- [x] Checkpoints + eval suite (`doppelt eval`, `doppelt random --policy neural`)
- [ ] Algorithm improvements toward mean 300 ([§3.6](#36-algorithm-improvements-268--300))



### Phase 4

- [ ] Full eval report
- [ ] Next steps documented

---



## Risks & Decisions



### Technical risks


| Risk                                     | Mitigation                                           |
| ---------------------------------------- | ---------------------------------------------------- |
| Rules bugs in bonus/action chains        | Golden tests from rulebook; replay human games       |
| Action space explosion (silver, bonuses) | Phase-specific masks; hierarchical actions if needed |
| Slow simulation                          | Profile early; solo-only; multiprocessing            |
| RL reward sparsity                       | BC pretrain; reward shaping on area completion       |
| Overfitting to weak bots                 | Mix bots; self-play; human logs                      |




### Decisions to make soon


| Decision        | Options                    | Recommendation                                                           |
| --------------- | -------------------------- | ------------------------------------------------------------------------ |
| Player count v1 | Solo only vs full 4-player | **Solo first**                                                           |
| Log format      | JSON vs binary numeric     | **Binary** (`uint16` action array + header); pretty-print for debug only |
| ML framework    | PyTorch vs JAX             | **PyTorch** unless you prefer JAX                                        |
| First training  | BC vs pure RL              | **BC on heuristic** then RL                                              |




### Out of scope (for now)

- Physical dice / UI polish
- Online multiplayer
- Perfect play / provably optimal solver
- Mobile app

---



## Suggested work order (next 3 sessions)

1. **Session 1:** Run `doppelt train expert --init CHECKPOINT --rounds 3 --games 32 --mcts-sims 32` (dev eval only). Then `doppelt eval … --suite report` once, not during tuning.
2. **Session 2:** If greedy mean moved, inspect failure lines (not silver-first). If stuck, go to afterstate 1-ply ([3.6.19](#36-algorithm-improvements-268--300)).
3. **Session 3:** Afterstate 1-ply greedy ([3.6.19](#36-algorithm-improvements-268--300)) — TD-Gammon style, no catalog change.

---



## References

- [Twice as Clever — BoardGameGeek](https://boardgamegeek.com/boardgame/269210/twice-as-clever)
- Official rulebook (PDF): search "Twice as Clever rulebook" or use your physical copy
- Related work: *Ganz schön clever* digital implementations and AI discussions (mechanics overlap; silver area is the main delta)

---

*Last updated: 2026-08-18 — encoding_v2 (3.6.3); v1 checkpoints will not load.*