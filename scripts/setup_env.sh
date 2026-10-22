#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-}"
case "$profile" in
  robot) python_version=3.10 ;;
  converter|ml) python_version=3.12 ;;
  *) echo 'Usage: ./run.sh setup robot|converter|ml [--resume]'; exit 2 ;;
esac
resume=0
if [[ $# -eq 2 && "$2" == --resume ]]; then
  resume=1
elif [[ $# -ne 1 ]]; then
  echo 'Usage: ./run.sh setup robot|converter|ml [--resume]' >&2
  exit 2
fi
if ! command -v uv >/dev/null; then echo 'Install uv first; see docs/environments.md.' >&2; exit 2; fi
target="$repo/.venv-$profile"
if [[ -L "$target" ]]; then
  echo "Refusing to install through an environment symlink: $target" >&2
  exit 2
fi
if [[ -e "$target" && $resume -eq 0 ]]; then
  echo "Refusing to modify existing environment: $target. Use it as-is, or choose/remove it yourself." >&2
  echo "For an interrupted installation in this new repository, retry: ./run.sh setup $profile --resume" >&2
  exit 2
fi
if ((resume)); then
  if [[ ! -d "$target" || ! -f "$target/pyvenv.cfg" || ! -x "$target/bin/python" ]]; then
    echo "Cannot resume: $target is not an existing Python virtual environment." >&2
    exit 2
  fi
  # Check the actual isolated prefix, not just a directory bearing a venv name.
  "$target/bin/python" -I -c '
import pathlib,sys
target, version = pathlib.Path(sys.argv[1]).resolve(), sys.argv[2]
if (pathlib.Path(sys.prefix).resolve() != target or sys.prefix == sys.base_prefix
        or ".".join(map(str,sys.version_info[:2])) != version):
    raise SystemExit("Refusing to resume: interpreter prefix/version does not match the repository environment")
' "$target" "$python_version"
else
  uv venv --python "$python_version" "$target"
fi
trap 'echo "Installation did not finish. Environment retained. After fixing the error, retry: ./run.sh setup '"$profile"' --resume" >&2' ERR
requirements="$repo/requirements/$profile.lock.txt"
if [[ ! -f "$requirements" ]]; then requirements="$repo/requirements/$profile.txt"; fi
uv pip install --python "$target/bin/python" -r "$requirements"
uv pip install --python "$target/bin/python" --no-deps --editable "$repo"
uv pip check --python "$target/bin/python"
echo "Ready: $target; other environments were not modified. Use ./run.sh doctor next."
