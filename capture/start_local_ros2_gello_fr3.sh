#!/usr/bin/env bash
set -euo pipefail

sanitize_path() {
    local path="${1:-}"
    local filtered=""
    IFS=":" read -r -a parts <<< "$path"
    for p in "${parts[@]}"; do
        case "$p" in
            "$HOME/anaconda3/bin"|"$HOME/anaconda3/condabin"|"$HOME/miniconda3/bin"|"$HOME/miniconda3/condabin") ;;
            "") ;;
            *) filtered="${filtered:+$filtered:}$p" ;;
        esac
    done
    echo "$filtered"
}

sanitize_env() {
    PATH="$(sanitize_path "$PATH")"
    export PATH
    unset LD_PRELOAD || true
    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_SHLVL _CE_CONDA _CE_M || true
    if [ -n "${PYTHONPATH:-}" ]; then
        PYTHONPATH="$(
            echo "$PYTHONPATH" \
                | tr ':' '\n' \
                | awk 'length($0) > 0 && $0 !~ /(ana|mini)?conda[0-9]?\//' \
                | paste -sd ':' -
        )"
        if [ -n "$PYTHONPATH" ]; then
            export PYTHONPATH
        else
            unset PYTHONPATH
        fi
    fi
}

safe_source() {
    local file="$1"
    set +u
    # shellcheck disable=SC1090
    source "$file"
    set -u
}

source_ros2_environment() {
    sanitize_env
    if [ -n "${LOCAL_ROS2_SETUP_CMD:-}" ]; then
        echo "Using LOCAL_ROS2_SETUP_CMD"
        set +u
        eval "$LOCAL_ROS2_SETUP_CMD"
        set -u
        sanitize_env
        return
    fi

    local ros_setup="/opt/ros/${ROS_DISTRO}/setup.bash"
    local franka_setup="/opt/ros/${ROS_DISTRO}/franka/setup.bash"
    local ws_setup="${LOCAL_ROS2_WS}/install/setup.bash"

    if [ -f "$ros_setup" ]; then
        safe_source "$ros_setup"
    fi
    if [ -f "$franka_setup" ]; then
        safe_source "$franka_setup"
    fi
    if [ -f "$ws_setup" ]; then
        safe_source "$ws_setup"
    fi
    sanitize_env
}

resolve_config() {
    local package="$1"
    local name="$2"
    local installed="${LOCAL_ROS2_WS}/install/${package}/share/${package}/config/${name}"
    local source="${LOCAL_ROS2_WS}/src/${package}/config/${name}"
    if [ -f "$installed" ]; then
        echo "$installed"
    elif [ -f "$source" ]; then
        echo "$source"
    else
        return 1
    fi
}

detect_gello_port() {
    local matches=()
    local pattern
    for pattern in /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter* /dev/serial/by-id/usb-ROBOTIS_OpenRB-150*; do
        [ -e "$pattern" ] && matches+=("$pattern")
    done
    if [ "${#matches[@]}" -eq 1 ]; then
        echo "${matches[0]}"
        return 0
    fi
    return 1
}

wait_for_gello_topic() {
    local timeout_sec="${1:-15}"
    local pgid="${2:-}"
    local deadline=$((SECONDS + timeout_sec))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if [ -n "$pgid" ] && ! pgrep -g "$pgid" >/dev/null 2>&1; then
            echo "GELLO publisher launch exited before /gello/joint_states became available."
            return 1
        fi
        if timeout 2s ros2 topic echo --once /gello/joint_states >/dev/null 2>&1 \
            && { [ -z "$pgid" ] || pgrep -g "$pgid" -f 'gello_publisher' >/dev/null 2>&1; }; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

set_rt_priority_for_ros2_control() {
    local pgid="$1"
    local prio="${2:-80}"
    local pid=""
    command -v chrt >/dev/null 2>&1 || return 0
    for _ in $(seq 1 30); do
        pid="$(pgrep -g "$pgid" -f 'ros2_control_node' | head -n 1 || true)"
        if [ -n "$pid" ]; then
            chrt -f -p "$prio" "$pid" >/dev/null 2>&1 || true
            return 0
        fi
        sleep 0.2
    done
}

pin_cpu_for_ros2_control() {
    local pgid="$1"
    local cpu="$2"
    local pid=""
    [ -n "$cpu" ] || return 0
    command -v taskset >/dev/null 2>&1 || return 0
    for _ in $(seq 1 30); do
        pid="$(pgrep -g "$pgid" -f 'ros2_control_node' | head -n 1 || true)"
        if [ -n "$pid" ]; then
            taskset -pc "$cpu" "$pid" >/dev/null 2>&1 || true
            return 0
        fi
        sleep 0.2
    done
}

guarded_launch() {
    if [ -n "$LOCAL_ROS2_SESSION_ID" ]; then
        local session_lock_fd
        exec {session_lock_fd}>"/tmp/pi05_capture_session_${LOCAL_ROS2_SESSION_ID}.lock"
        flock -x "$session_lock_fd"
        if [ -e "/tmp/pi05_capture_cancel_${LOCAL_ROS2_SESSION_ID}" ]; then
            flock -u "$session_lock_fd"
            exec {session_lock_fd}>&-
            echo "ROS2 session was canceled; refusing to launch." >&2
            exit 130
        fi
        setsid "$@" {session_lock_fd}>&- &
        LAST_LAUNCH_PID=$!
        flock -u "$session_lock_fd"
        exec {session_lock_fd}>&-
    else
        setsid "$@" &
        LAST_LAUNCH_PID=$!
    fi
}

cleanup() {
    local status=$?
    [ "${CLEANUP_STARTED:-0}" = "0" ] || return
    CLEANUP_STARTED=1
    trap - EXIT
    trap '' HUP INT TERM
    echo "Stopping local ROS2 GELLO/FR3 stack..."
    local groups=()
    local pg
    for pg in "${PGID_GRIPPER:-}" "${PGID_FR3:-}" "${PGID_GELLO:-}" "${LAST_LAUNCH_PID:-}"; do
        if [[ "$pg" =~ ^[0-9]+$ ]] && [ "$pg" -gt 1 ]; then
            case " ${groups[*]} " in *" $pg "*) ;; *) groups+=("$pg") ;; esac
        fi
    done
    local sig timeout
    for sig in INT TERM KILL; do
        case "$sig" in INT) timeout=5 ;; TERM) timeout=3 ;; KILL) timeout=1 ;; esac
        for pg in "${groups[@]}"; do
            kill -"$sig" -- "-$pg" 2>/dev/null || true
        done
        local deadline=$((SECONDS + timeout))
        local remaining
        while true; do
            remaining=0
            for pg in "${groups[@]}"; do
                if kill -0 -- "-$pg" 2>/dev/null; then remaining=1; fi
            done
            [ "$remaining" = "0" ] && break
            [ "$SECONDS" -ge "$deadline" ] && break
            sleep 0.1
        done
        [ "$remaining" = "0" ] && break
    done
    for pg in "${groups[@]}"; do
        if kill -0 -- "-$pg" 2>/dev/null; then
            echo "ROS2 process group $pg did not exit after SIGKILL" >&2
            status=1
        else
            wait "$pg" 2>/dev/null || true
        fi
    done
    [ -z "${LOCAL_ROS2_READY_FILE:-}" ] || rm -f "$LOCAL_ROS2_READY_FILE"
    exit "$status"
}
LOCAL_ROS2_SESSION_ID="${LOCAL_ROS2_SESSION_ID:-}"
export PI05_CAPTURE_OWNER="pi05-fr3-gello"
[[ -n "$LOCAL_ROS2_SESSION_ID" ]] || { echo "An explicit capture session is required" >&2; exit 2; }
if [ -n "$LOCAL_ROS2_SESSION_ID" ]; then
    if [[ ! "$LOCAL_ROS2_SESSION_ID" =~ ^[a-f0-9]{32}$ ]]; then
        echo "Invalid LOCAL_ROS2_SESSION_ID: expected 32 lowercase hexadecimal characters" >&2
        exit 2
    fi
    command -v flock >/dev/null 2>&1 || { echo "flock is required for session control" >&2; exit 2; }
    if [ -e "/tmp/pi05_capture_cancel_${LOCAL_ROS2_SESSION_ID}" ]; then
        echo "ROS2 session was canceled; refusing to start wrapper." >&2
        exit 130
    fi
fi

CLEANUP_STARTED=0
PGID_GELLO=""
PGID_FR3=""
PGID_GRIPPER=""
LAST_LAUNCH_PID=""
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "${1:-}" = "--data-bridge" ]; then
    [ "$#" = "2" ] || { echo "--data-bridge requires one command argument" >&2; exit 2; }
    guarded_launch bash -lc "$2"
    wait "$LAST_LAUNCH_PID"
    exit 0
fi

ROS_DISTRO="${ROS_DISTRO:-humble}"
LOCAL_ROS2_WS="${LOCAL_ROS2_WS:-/workspace/ros2}"
if [ -f "${FR3_HARDWARE_FAULT_LOCK:-/workspace/repo/runtime/gripper_hardware_fault.lock}" ]; then
    echo "BLOCKED: inspect the configured hardware fault lock." >&2
    exit 41
fi
GELLO_CFG="${GELLO_CFG:-my_gello.yaml}"
FR3_CFG="${FR3_CFG:-example_fr3_config.yaml}"
LOCAL_ROS2_USE_GRIPPER="${LOCAL_ROS2_USE_GRIPPER:-1}"
LOCAL_ROS2_GRIPPER_TYPE="${LOCAL_ROS2_GRIPPER_TYPE:-franka_hand}"
LOCAL_ROS2_READY_FILE="${LOCAL_ROS2_READY_FILE:-}"
[ -n "$LOCAL_ROS2_READY_FILE" ] && rm -f "$LOCAL_ROS2_READY_FILE"

LOCAL_ROS2_GRIPPER_TYPE="franka_hand"
LOCAL_ROS2_USE_GRIPPER="1"
GRIPPER_CFG="example_fr3_config_franka_hand_move.yaml"
GRIPPER_LAUNCH="franka_gripper_client.launch.py"
GRIPPER_LABEL="Franka Hand Move client"

source_ros2_environment
if ! command -v ros2 >/dev/null 2>&1; then
    echo "ros2 command not found. Start this from the ROS2 Humble environment or set LOCAL_ROS2_SETUP_CMD."
    exit 1
fi
if ! /usr/bin/python3 -c 'import rclpy' >/dev/null 2>&1; then
    echo "/usr/bin/python3 cannot import rclpy. Start this from the ROS2 Humble environment or set LOCAL_ROS2_SETUP_CMD."
    exit 1
fi

GELLO_CFG_PATH="$(resolve_config franka_gello_state_publisher "$GELLO_CFG")"
FR3_CFG_PATH="$(resolve_config franka_fr3_arm_controllers "$FR3_CFG")"
if [ "${LOCAL_ROS2_USE_GRIPPER}" != "0" ]; then
    GRIPPER_CFG_PATH="$(resolve_config franka_gripper_manager "$GRIPPER_CFG")"
fi

if [ -n "${GELLO_PORT:-}" ]; then
    GELLO_PORT_RESOLVED="$GELLO_PORT"
else
    GELLO_PORT_RESOLVED="$(detect_gello_port || true)"
fi
export GELLO_PORT="${GELLO_PORT_RESOLVED:-}"

ROBOT_IP="$(awk -F: '/robot_ip:/ {gsub(/"/, "", $2); gsub(/[[:space:]]/, "", $2); print $2; exit}' "$FR3_CFG_PATH" || true)"
echo "GELLO ROS2 workspace: $LOCAL_ROS2_WS"
echo "GELLO config: $GELLO_CFG"
echo "FR3 config: $FR3_CFG"
echo "Gripper type: $LOCAL_ROS2_GRIPPER_TYPE"
if [ "${LOCAL_ROS2_USE_GRIPPER}" != "0" ]; then
    echo "Gripper config: $GRIPPER_CFG"
fi
[ -n "$ROBOT_IP" ] && echo "FR3 robot_ip: $ROBOT_IP"
[ -n "$GELLO_PORT_RESOLVED" ] && echo "GELLO port: $GELLO_PORT_RESOLVED"

echo "Launching GELLO publisher..."
guarded_launch ros2 launch franka_gello_state_publisher main.launch.py config_file:="$GELLO_CFG"
PID_GELLO=$LAST_LAUNCH_PID
PGID_GELLO=$PID_GELLO

echo "Waiting for /gello/joint_states..."
if ! wait_for_gello_topic "${LOCAL_ROS2_GELLO_TOPIC_TIMEOUT:-20}" "$PGID_GELLO"; then
    echo "No /gello/joint_states received. Check GELLO USB/power and $GELLO_CFG_PATH."
    exit 1
fi

echo "Launching FR3 joint impedance controller..."
guarded_launch ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py robot_config_file:="$FR3_CFG"
PID_FR3=$LAST_LAUNCH_PID
PGID_FR3=$PID_FR3

CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
ROS2_CONTROL_CPU="${ROS2_CONTROL_CPU:-}"
if [ -z "$ROS2_CONTROL_CPU" ] && [ "$CPU_COUNT" -ge 4 ]; then
    ROS2_CONTROL_CPU="2"
fi
if [ "${ENABLE_RT:-1}" = "1" ]; then
    set_rt_priority_for_ros2_control "$PGID_FR3" "${RT_PRIORITY:-80}"
fi
if [ "${PIN_CPU:-1}" = "1" ]; then
    pin_cpu_for_ros2_control "$PGID_FR3" "$ROS2_CONTROL_CPU"
fi

if [ "${LOCAL_ROS2_USE_GRIPPER}" != "0" ]; then
    echo "Launching $GRIPPER_LABEL..."
    guarded_launch ros2 launch franka_gripper_manager "$GRIPPER_LAUNCH" config_file:="$GRIPPER_CFG"
    PID_GRIPPER=$LAST_LAUNCH_PID
    PGID_GRIPPER=$PID_GRIPPER
fi

if [ -n "$LOCAL_ROS2_READY_FILE" ]; then
    touch "$LOCAL_ROS2_READY_FILE"
fi

# Any launch exiting ends this session; clean up the other launched groups too.
launch_pids=("$PID_GELLO" "$PID_FR3")
[ -z "${PID_GRIPPER:-}" ] || launch_pids+=("$PID_GRIPPER")
while true; do
    for pid in "${launch_pids[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            status=0
            wait "$pid" || status=$?
            exit "$status"
        fi
    done
    sleep 0.1
done
