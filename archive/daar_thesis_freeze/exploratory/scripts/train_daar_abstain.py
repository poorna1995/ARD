"""Deprecated alias — use ``scripts/train_gate1a_solvability.py``."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_gate1a_solvability.py")), run_name="__main__")
