#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
container="${PI05_CONTAINER:-pi05-fr3-gello-ros2-1}"
docker_cmd=(docker -H "${PI05_DOCKER_HOST:-unix:///var/run/docker.sock}")
compose=("${docker_cmd[@]}" compose -f "$repo/docker/compose.yaml")
if [[ $# -ne 1 ]]; then echo 'Usage: ./run.sh ros-image|ros-up|ros-build' >&2; exit 2; fi
case "$1" in
  ros-image)
    python3 "$repo/docker/prepare_sources.py"
    exec "${compose[@]}" build ros2 ;;
  ros-up) exec "${compose[@]}" up -d --no-build ros2 ;;
  ros-build)
    exec "${docker_cmd[@]}" exec "$container" bash -lc '
      set -eo pipefail
      source /opt/ros/humble/setup.bash
      source /opt/ros/humble/franka/setup.bash
      cd /workspace/ros2
      colcon build --symlink-install --packages-select franka_gello_state_publisher franka_fr3_arm_controllers franka_gripper_manager --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
    ' ;;
  *) exit 2 ;;
esac
