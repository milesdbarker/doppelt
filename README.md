# doppelt

AI agent for **Doppelt So Clever** (Twice as Clever).

See [ROADMAP.md](ROADMAP.md) for the full build plan.

## Setup

Requires Python 3.11+.

```powershell
cd doppelt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Optional ML extra for Phase 3 (encoding tests run without it; training needs PyTorch):

```powershell
pip install -e ".[ml]"
```

## Commands

```powershell
pytest                    # run tests
ruff check src tests

doppelt play              # interactive solo game
doppelt play --seed 42 --save game.bin
doppelt random --seed 42  # random legal self-play (smoke test / log generation)
doppelt random --policy heuristic --seed 42 --save heuristic.bin
doppelt random --policy greedy --save greedy.bin
doppelt random --policy mcts-lite --seed 1 --save mcts.bin
doppelt visualize heuristic.bin
doppelt simulate --games 10000 --workers 0   # batch random games; 0 workers = CPU count
doppelt simulate --games 1000 --policy greedy_immediate
doppelt simulate --games 1000 --policy heuristic
doppelt simulate --games 200 --policy mcts_lite
doppelt dataset generate                     # 1M mixed heuristic/greedy/random DPLD shards (no mcts_lite)
doppelt dataset generate --games 10000 --out data/datasets/smoke
doppelt dataset analyze                      # baseline report from solo_v1 shards
doppelt dataset analyze --json data/datasets/solo_v1/analytics.json
doppelt train bc --live-games 200 --out data/models/bc_v1.pt
doppelt train bc --in data/datasets/solo_v1 --policy heuristic --max-games 10000 --arch pvn_v1
doppelt train selfplay --init data/models/bc_v1.pt --iters 50 --games 32 --out data/models/selfplay_v1.pt
doppelt eval data/models/selfplay_v1.pt --games 64 --baselines random_legal,heuristic
doppelt eval data/models/selfplay_v1.pt --games 16 --mcts-sims 32
doppelt random --policy neural --checkpoint data/models/selfplay_v1.pt --mcts-sims 32
python scripts/overnight_selfplay.py --init data/models/bc_v1.pt --out-dir data/models/overnight
doppelt random --policy neural --checkpoint data/models/bc_v1.pt --seed 42
doppelt replay game.bin   # replay a binary log and print scores
doppelt decode game.bin   # print log header and action ids
doppelt visualize game.bin                  # step through a replay on the score sheet
doppelt visualize game.bin --out sheet.png --no-window
```