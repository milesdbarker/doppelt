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
doppelt replay game.bin   # replay a binary log and print scores
doppelt decode game.bin   # print log header and action ids
```