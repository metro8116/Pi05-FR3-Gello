#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
interpreter="${PI05_TEST_PYTHON:-${PI05_ML_PYTHON:-python3}}"
if [[ -z "${PI05_TEST_PYTHON:-}" && ! -x "$interpreter" ]]; then
  interpreter=python3
  for candidate in "${PI05_CONVERTER_PYTHON:-$repo/.venv-converter/bin/python}" "${PI05_HOST_PYTHON:-$repo/.venv-robot/bin/python}"; do
    if [[ -x "$candidate" ]]; then interpreter="$candidate"; break; fi
  done
fi
export CUDA_VISIBLE_DEVICES=''
"$interpreter" -B -c 'import sys; assert sys.version_info >= (3,10), "CPU regression tests require Python >=3.10; set PI05_TEST_PYTHON"'
echo "CPU tests: $interpreter (optional dependency skips are reported)"
cd "$repo"
"$interpreter" -B -m unittest discover -s tests -v
"$interpreter" -B -m unittest discover -s deploy/tests -v
if [[ -d capture/tests ]]; then "$interpreter" -B -m unittest discover -s capture/tests -v; fi
