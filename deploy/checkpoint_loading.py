"""Compatibility import for the single shared native checkpoint loader."""
from pathlib import Path
import sys

_training_src = Path(__file__).resolve().parents[1] / 'src'
if str(_training_src) not in sys.path:
    sys.path.insert(0, str(_training_src))

from fr3_pi05.runtime import load_native_weights_strict

__all__ = ['load_native_weights_strict']
