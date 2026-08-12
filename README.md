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
pytest          # run tests
doppelt         # CLI (stub for now)
ruff check src tests
```