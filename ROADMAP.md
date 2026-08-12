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
8. [Milestone Checklist](#milestone-checklist)
9. [Risks & Decisions](#risks--decisions)

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
- **Bonuses & actions:** Immediate chain resolution (reroll, return die from platter, extra die pick). This is the hardest part of the engine.
- **Foxes:** Each fox = points equal to your **lowest** color score at game end.
- **Win condition:** Highest total score (solo: maximize your own score).



### Success criteria (high level)

- [ ] Any finished game can be replayed from seed + action log and produce identical final state and score.
- [ ] Engine runs ≥ 10k solo games/sec on your machine (order-of-magnitude target; tune after baseline).
- [ ] Trained agent beats random play consistently and approaches human-reasonable solo scores.
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
  - Fox positions, action tracks (reroll / return / extra die), round-track bonuses.
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
| 9     | **Actions**            | **High**   | Reroll, return from platter, extra die (timing constraints)     |
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
- [ ] Unit test: catalog is contiguous, no duplicate semantics, size matches `ACTION_SPACE_SIZE`.
- [ ] Unit test: random legal play → encode IDs → replay → identical terminal state.



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
- [ ] **Golden replay tests** — load action log, assert final score and sheet state.
- [ ] **Property tests** — `apply` only legal actions; score monotonicity where applicable.
- [ ] **Rulebook examples** — encode official examples from the PDF as tests.
- [ ] **Fuzz test** — random legal play for 10k games; no crashes, terminal always reached.



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

- [ ] Profile hot path (`legal_actions`, `apply_action`, scoring).
- [ ] Batch simulation: run N games with vectorized or multiprocessing workers.
- [ ] Target: benchmark **games/sec** and **actions/sec** on your hardware.
- [ ] Optional: `numba` / Rust extension if Python is too slow after optimization.



### 2.2 Bots (baselines)

Implement simple policies for data generation and evaluation:


| Bot                      | Purpose                                                        |
| ------------------------ | -------------------------------------------------------------- |
| **RandomLegal**          | Uniform over legal actions                                     |
| **GreedyImmediate**      | Maximize immediate marks / stars                               |
| **Heuristic**            | Hand-tuned priorities (silver chains, fox setup, blue descent) |
| **MCTS-lite** (optional) | Small rollout search — strong baseline without NN              |




### 2.3 Dataset generation

- [ ] Run **1M+** solo games with mixed bots; save compact binary logs (`seed` + `uint16[]` actions).
- [ ] Store bulk logs as `.bin` shards or columnar **Parquet** (`seed`, `actions` as fixed-length or variable-length int column, `final_score`).
- [ ] Record metadata: bot type, final score, per-color breakdown (metadata separate from per-move logs).



### 2.4 Analytics

- [ ] Score distributions per bot.
- [ ] Histogram of color scores and fox impact.
- [ ] Average actions per game, bonus chain depth.
- [ ] Identify degenerate strategies (e.g., always picking silver).



### Phase 2 milestones


| Milestone | Definition of done                                |
| --------- | ------------------------------------------------- |
| **M2.1**  | 100k games batch run completes reliably           |
| **M2.2**  | 1M game dataset with logs on disk                 |
| **M2.3**  | Baseline report: random vs heuristic scores       |
| **M2.4**  | ≥ 10k games/sec (or documented bottleneck + plan) |


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

Keep encoding versioned (`encoding_v1`) so you can change without breaking old models.

### 3.2 Action space

Fixed discrete space with **legal mask**:

- Phase-specific subsets (e.g., only `PICK_DIE` actions during pick phase).
- Composite actions where safe (pick + mark bundled if uniqueness holds).
- Illegal actions masked to −inf before softmax.

Size may be large (silver color choices, bonus targets). Start with explicit enumeration; compress later if needed.

### 3.3 Model architecture (starting point)


| Component   | Suggestion                                           |
| ----------- | ---------------------------------------------------- |
| Backbone    | Small MLP or 1D CNN over flattened sheet + dice      |
| Policy head | Linear → softmax over action space (masked)          |
| Value head  | Linear → scalar (expected final score)               |
| Size        | Small first (~100k–1M params); scale if underfitting |




### 3.4 Training approaches (try in order)

1. **Behavioral cloning** — imitate heuristic bot games from Phase 2 (fast sanity check).
2. **Supervised on best-of-N** — label actions from best rollout among N random continuations.
3. **Policy gradient / Actor-Critic** — reward = final score (or shaped intermediate rewards).
4. **Self-play** — current policy vs itself; optional league of checkpoint opponents.

For solo roll-and-write, **final score RL** is natural; consider auxiliary rewards for fox setup only if training is too sparse.

### 3.5 Training loop

- [ ] `Dataset` from Parquet logs OR on-the-fly self-play generator.
- [ ] Train/val split by seed ranges (avoid leakage).
- [ ] Log: loss, policy entropy, mean episode score, eval vs heuristic every N steps.
- [ ] Checkpoint best model by **eval score**, not train loss.



### Phase 3 milestones


| Milestone | Definition of done                                          |
| --------- | ----------------------------------------------------------- |
| **M3.1**  | State/action encoding implemented and tested                |
| **M3.2**  | BC model beats RandomLegal by wide margin                   |
| **M3.3**  | RL/self-play improves over BC on eval suite                 |
| **M3.4**  | 10k eval games: report mean/median score + comparison table |


**Estimated effort:** 3–6 weeks (tuning-heavy).

---



## Phase 4 — Evaluate & Iterate

**Objective:** Decide what's next based on data — don't guess.

### 4.1 Evaluation suite

- [ ] **Fixed seed suite** — 1000 seeds, same for all agents.
- [ ] **Score stats** — mean, median, std, percentiles, histogram.
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
- Stronger search: MCTS with policy prior, depth-limited rollouts.
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



## Milestone Checklist

Use this as a living progress tracker.

### Phase 0

- [x] Repo + dependencies
- [x] Score sheet data file + **action catalog v1**
- [x] Rules notes doc



### Phase 1

- [x] Turn loop (solo) — active pick + passive pick, 6 rounds
- [x] Yellow / pink / blue / **green** / **silver** marking
- [x] Bonuses + chains (FIFO queue, auto blue/green/pink wild; yellow/silver wild via actions)
- [ ] Scoring + foxes
- [ ] Action log replay
- [ ] CLI play/replay
- [ ] Golden tests pass



### Phase 2

- [ ] Batch simulator
- [ ] 1M game dataset
- [ ] Baseline bots + report



### Phase 3

- [ ] Encoding + model
- [ ] BC training works
- [ ] Self-play / RL improves score
- [ ] Checkpoints + eval suite



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

1. **Session 1:** Phase 0 — `pyproject.toml`, score sheet YAML, enums, empty `GameState`.
2. **Session 2:** Phase 1.1–1.2 — turn loop + yellow/pink/blue without bonuses.
3. **Session 3:** CLI random self-play + binary action log + first golden replay test.

---



## References

- [Twice as Clever — BoardGameGeek](https://boardgamegeek.com/boardgame/269210/twice-as-clever)
- Official rulebook (PDF): search "Twice as Clever rulebook" or use your physical copy
- Related work: *Ganz schön clever* digital implementations and AI discussions (mechanics overlap; silver area is the main delta)

---

*Last updated: 2026-08-11 — action log: binary uint16 catalog IDs.*