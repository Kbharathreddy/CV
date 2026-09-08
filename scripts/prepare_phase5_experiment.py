"""CLI wrapper for Phase 5 manifest and scene preparation."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.phase5_experiments import main


if __name__ == "__main__":
    raise SystemExit(main())
