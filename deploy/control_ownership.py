"""Compatibility module alias for the repository's shared FR3 control lease."""
from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from common import fr3_control_ownership as _shared

if __name__ == '__main__':
    raise SystemExit(_shared.main())
else:
    # Use the actual shared module, including its globals and test overrides.
    sys.modules[__name__] = _shared
