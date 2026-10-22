#!/usr/bin/env bash
# Local GUI only. Robot activation always requires the GUI's explicit start.
set -euo pipefail
capture_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$capture_dir/.." && pwd)
capture_python=${CAPTURE_PYTHON:-${PI05_HOST_PYTHON:-$repo_root/.venv-robot/bin/python}}
check_only=0
while (($#)); do
  case "$1" in
    --output-dir)
      (($# >= 2)) || { echo '--output-dir needs a path' >&2; exit 2; }
      export TAKE_DATA_OUTPUT_DIR="$2"; shift 2 ;;
    --output-dir=*) export TAKE_DATA_OUTPUT_DIR="${1#*=}"; shift ;;
    --no-camera) export TAKE_DATA_DISABLE_CAMERAS=1; shift ;;
    --check) check_only=1; shift ;;
    --help|-h)
      echo 'Usage: start_capture.sh [--output-dir PATH] [--no-camera] [--check]'
      echo 'CAPTURE_PYTHON defaults to REPO/.venv-robot/bin/python; --check opens no camera/robot.'
      exit 0 ;;
    *) echo "Unknown capture option: $1" >&2; exit 2 ;;
  esac
done
[[ -x $capture_python ]] || { echo "Capture Python missing: $capture_python; set up the repository robot environment first." >&2; exit 2; }
export PYTHONPATH="$repo_root/common:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export CAPTURE_PYTHON="$capture_python"
export TAKE_DATA_OUTPUT_DIR="${TAKE_DATA_OUTPUT_DIR:-$repo_root/data/raw}"
export TAKE_DATA_TASK="${TAKE_DATA_TASK-Put the avocado in the basket.}"
export TAKE_DATA_CAMERA_SERIALS="${TAKE_DATA_CAMERA_SERIALS:-233522074047,405622074065}"
export TAKE_DATA_DISABLE_CAMERAS="${TAKE_DATA_DISABLE_CAMERAS:-0}"
export TAKE_DATA_AUTOSTART_ROBOT=0
export PI05_CONTAINER="${PI05_CONTAINER:-pi05-fr3-gello-ros2-1}"
export LOCAL_ROS2_WS="$repo_root/ros2_ws"
export LOCAL_ROS2_DOCKER_COMPOSE="$repo_root/docker/compose.yaml"
export LOCAL_ROS2_DOCKER_SERVICE=ros2
export LOCAL_ROS2_DOCKER_HOST="${LOCAL_ROS2_DOCKER_HOST:-unix:///var/run/docker.sock}"
export LOCAL_ROS2_GRIPPER_CFG=example_fr3_config_franka_hand_move.yaml
export LOCAL_ROS2_EXEC_MODE=container
export FRANKA_IP="${FRANKA_IP:-${ROBOT_IP:-172.16.0.2}}"
export PYTHONUNBUFFERED=1
if ((check_only)); then
  exec "$capture_python" -c 'import cv2, pyrealsense2, tkinter, h5py, numpy, PIL, yaml, fr3_control_ownership; print("Capture imports OK; no hardware was opened.")'
fi
fault_lock=${FR3_HARDWARE_FAULT_LOCK:-$repo_root/runtime/gripper_hardware_fault.lock}
[[ ! -f $fault_lock ]] || { echo "Hardware inspection lock present: $fault_lock" >&2; exit 41; }
exec "$capture_python" "$capture_dir/video_action_recorder.py"
