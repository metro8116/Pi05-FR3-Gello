#!/usr/bin/env python3
"""Train using the selected existing Python/uv environment; never installs packages."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from fr3_pi05.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
