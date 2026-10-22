#!/usr/bin/env bash
# Retry an idempotent dependency-fetch command; keep its TLS verification intact.
set -uo pipefail
attempts="${1:-}"
shift || exit 2
[[ "$attempts" =~ ^[1-9][0-9]*$ && $# -gt 0 ]] || { echo 'Usage: retry.sh ATTEMPTS COMMAND [ARGS...]' >&2; exit 2; }
result=1
for ((attempt=1; attempt<=attempts; attempt++)); do
  if "$@"; then exit 0; else result=$?; fi
  if ((attempt < attempts)); then
    delay=$((attempt * 3))
    echo "Dependency command failed (exit $result); retry $((attempt+1))/$attempts in ${delay}s: $1" >&2
    sleep "$delay"
  fi
done
exit "$result"
