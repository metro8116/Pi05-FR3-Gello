#!/usr/bin/env bash
# One entry point. Environment creation is explicit; no command installs implicitly.
set -euo pipefail
PI05_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PI05_REPO_ROOT
for env_file in "$PI05_REPO_ROOT/configs/local.env" "$PI05_REPO_ROOT/configs/autodl.env"; do
  if [[ -f "$env_file" ]]; then source "$env_file"; fi
done
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PI05_REPO_ROOT:$PI05_REPO_ROOT/src:$PI05_REPO_ROOT/common${PYTHONPATH:+:$PYTHONPATH}"
export PI05_DOCKER_HOST="${PI05_DOCKER_HOST:-unix:///var/run/docker.sock}"
export LOCAL_ROS2_DOCKER_HOST="${LOCAL_ROS2_DOCKER_HOST:-$PI05_DOCKER_HOST}"
export PI05_HOST_PYTHON="${PI05_ROBOT_PYTHON:-$PI05_REPO_ROOT/.venv-robot/bin/python}"
export CAPTURE_PYTHON="$PI05_HOST_PYTHON"
export PI05_ML_PYTHON="${PI05_ML_PYTHON:-$PI05_REPO_ROOT/.venv-ml/bin/python}"
export PI05_CONVERTER_PYTHON="${PI05_CONVERTER_PYTHON:-$PI05_REPO_ROOT/.venv-converter/bin/python}"

help() {
  cat <<'EOF'
Pi05-FR3-Gello — run from any working directory
  ./run.sh setup robot|converter|ml [--resume]
                                  Create a repository-local uv environment, or resume its failed setup
  ./run.sh doctor                  Read-only paths/dependency inventory; no hardware
  ./run.sh ros-image               Build the dedicated ROS Humble Docker image
  ./run.sh ros-up                  Start its idle container (no robot nodes)
  ./run.sh ros-build               Build the collection ROS packages (no hardware)
  ./run.sh prepare                 Build/test the deployment overlay (no hardware)
  ./run.sh collect [--check|--output-dir PATH|--no-camera]
                                  Open collection GUI; enable teleoperation manually
  ./run.sh stop-collect            Stop only this repository's recorded capture session
  ./run.sh validate [ARGS]         Validate raw recordings; no conversion
  ./run.sh convert [ARGS]          Export a NEW LeRobot v3.0 dataset
  ./run.sh check-train [ARGS]      Inspect data, weights and ML runtime
  ./run.sh train [ARGS]            Official LeRobot training (fresh output directory)
  ./run.sh resume --resume-from PATH [ARGS]
                                  Restore an official checkpoint and training state
  ./run.sh serve --checkpoint PATH [ARGS]
                                  GPU host: loopback inference server
  ./run.sh deploy [--check|--network-check|--no-tunnel]
                                  Robot PC: supervised ROS + deployment GUI
  ./run.sh stop-deploy             Stop the recorded deployment session
  ./run.sh test                    CPU regression tests; no robot/GPU allocation
  ./run.sh help

Machine overrides: configs/local.env and configs/autodl.env (ignored by Git).
See README.md and docs/. Never run collect and deploy simultaneously.
EOF
}
python_for() {
  local interpreter="$1"; shift
  if [[ ! -x "$interpreter" ]]; then
    echo "Python not found: $interpreter. Run ./run.sh setup PROFILE or configure PI05_*_PYTHON." >&2
    exit 2
  fi
  exec "$interpreter" "$@"
}
command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  help|-h|--help) help ;;
  setup) exec bash "$PI05_REPO_ROOT/scripts/setup_env.sh" "$@" ;;
  doctor) exec python3 "$PI05_REPO_ROOT/scripts/doctor.py" "$@" ;;
  ros-image|ros-up|ros-build) exec bash "$PI05_REPO_ROOT/scripts/ros.sh" "$command" "$@" ;;
  collect) exec bash "$PI05_REPO_ROOT/capture/start_capture.sh" "$@" ;;
  stop-collect) exec bash "$PI05_REPO_ROOT/capture/stop_capture.sh" "$@" ;;
  validate) python_for "$PI05_CONVERTER_PYTHON" "$PI05_REPO_ROOT/tools/convert_to_lerobot_v30.py" --dry-run "$@" ;;
  convert) python_for "$PI05_CONVERTER_PYTHON" "$PI05_REPO_ROOT/tools/convert_to_lerobot_v30.py" "$@" ;;
  check-train) python_for "$PI05_ML_PYTHON" "$PI05_REPO_ROOT/check_env.py" "$@" ;;
  train) python_for "$PI05_ML_PYTHON" "$PI05_REPO_ROOT/train.py" "$@" ;;
  resume) python_for "$PI05_ML_PYTHON" "$PI05_REPO_ROOT/train.py" --resume "$@" ;;
  serve) python_for "$PI05_ML_PYTHON" "$PI05_REPO_ROOT/deploy/serve_pi05.py" "$@" ;;
  prepare) python_for "$PI05_HOST_PYTHON" "$PI05_REPO_ROOT/deploy/local_session.py" --prepare "$@" ;;
  deploy) python_for "$PI05_HOST_PYTHON" "$PI05_REPO_ROOT/deploy/local_session.py" "$@" ;;
  stop-deploy) python_for "$PI05_HOST_PYTHON" "$PI05_REPO_ROOT/deploy/local_session.py" --stop "$@" ;;
  test) exec bash "$PI05_REPO_ROOT/scripts/test.sh" "$@" ;;
  *) echo "Unknown command: $command" >&2; help >&2; exit 2 ;;
esac
