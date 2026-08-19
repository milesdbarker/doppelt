"""Overnight training now uses expert iteration (search as teacher).

This filename is kept so old commands still work. Prefer:

    python scripts/overnight_expert.py --init data/models/selfplay_v1.pt
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "overnight_expert.py"
    namespace = runpy.run_path(str(target), run_name="overnight_expert")
    raise SystemExit(namespace["main"]())
