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

Optional ML dependencies (PyTorch, NumPy, Parquet) — install later when needed:

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
doppelt simulate --games 10000 --workers 0   # batch random games; 0 workers = CPU count
doppelt simulate --games 1000 --policy greedy_immediate
doppelt simulate --games 1000 --policy heuristic
doppelt simulate --games 200 --policy mcts_lite
doppelt dataset generate                     # 1M mixed heuristic/greedy/random DPLD shards (no mcts_lite)
doppelt dataset generate --games 10000 --out data/datasets/smoke
doppelt dataset analyze                      # baseline report from solo_v1 shards
doppelt dataset analyze --json data/datasets/solo_v1/analytics.json
doppelt replay game.bin   # replay a binary log and print scores
doppelt decode game.bin   # print log header and action ids
```