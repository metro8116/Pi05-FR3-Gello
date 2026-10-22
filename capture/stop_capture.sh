#!/usr/bin/env bash
# No process sweep: the Python helper targets one recorded capture session.
set -euo pipefail
capture_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$capture_dir/.." && pwd)
stop_python=${CAPTURE_PYTHON:-${PI05_HOST_PYTHON:-$repo_root/.venv-robot/bin/python}}
if [[ ! -x $stop_python ]]; then stop_python=python3; fi
exec "$stop_python" "$capture_dir/session_stop.py" "$@"
