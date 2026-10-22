#!/usr/bin/env python3
"""Read-only offline preflight; no large model allocation or hardware access."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from fr3_pi05.doctor import main

if __name__ == "__main__":
    raise SystemExit(main())
