#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频+动作数据同步录制系统
Integrated Video and Robot Action Recording System
"""

import cv2
import pyrealsense2 as rs
import time
import math
import numpy as np
import os
import shutil
import socket
import json
import h5py
import copy
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, font as tkfont
import threading
from PIL import Image, ImageTk
import subprocess
import signal
import sys
import atexit
import shlex
import re
import uuid
from collections import deque

def _gripper_control_profile(goal):
    """Validate versioned execution parameters; state/action stay eight-dimensional.

    Keep this small wire validator identical in bridge, recorder and converter:
    the ROS bridge is copied into the container as a standalone file.
    """
    def check(condition, message):
        if not condition:
            raise ValueError(message)

    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    schema = goal.get("schema_version")
    check(isinstance(schema, int) and not isinstance(schema, bool) and schema in (1, 2),
          "Unsupported gripper target schema")
    if schema == 1:
        return {"schema_version": 1, "control_policy": "move_width_v1", "policy_config": None}
    required = ("control_policy", "control_state", "command_type", "force_n", "epsilon_inner_m",
                "epsilon_outer_m", "policy_config", "result_cancellation_reason")
    check(all(key in goal for key in required), "Missing schema-2 gripper control fields")
    policy = goal["control_policy"]
    check(policy in ("grasp_hold_v1", "grasp_hold_v2"), "Unsupported gripper control policy")
    check(goal["control_state"] in ("idle", "moving", "grasping", "holding", "canceling", "fault"),
          "Invalid gripper control state")
    check(goal["status"] in ("pending", "accepted", "completed", "not_reached", "error"),
          "Invalid schema-2 gripper status")
    maximum = goal.get("max_width_m")
    check(finite(maximum) and 0 < maximum <= 0.08 + 1e-9, "Invalid Franka Hand physical maximum width")
    config = goal["policy_config"]
    keys = {"grasp_force_n", "grasp_epsilon_inner_m", "grasp_epsilon_outer_m",
            "close_trigger_width_m", "release_trigger_width_m", "measurement_max_age_s"}
    if policy == "grasp_hold_v2":
        keys |= {"grasp_force_limit_n", "command_speed_m_s", "command_speed_limit_m_s"}
    check(isinstance(config, dict) and set(config) == keys and all(finite(v) for v in config.values()),
          "Invalid gripper policy_config fields or non-finite values")
    if policy == "grasp_hold_v1":
        # Preserve the historical recording profile, not an accessory load rating.
        check(30 <= config["grasp_force_n"] <= 70, "Historical grasp_hold_v1 force must be in 30..70 N")
    else:
        # These are software ceilings; passing them does not establish physical safety.
        check(0 < config["grasp_force_n"] <= config["grasp_force_limit_n"] <= 70,
              "Grasp force must be positive and within its configured software limit (<= 70 N)")
        check(0 < config["command_speed_m_s"] <= config["command_speed_limit_m_s"] <= 0.1,
              "Command speed must be positive and within its configured software limit (<= 0.1 m/s)")
    check(0 <= config["grasp_epsilon_inner_m"] <= maximum
          and 0 <= config["grasp_epsilon_outer_m"] <= maximum, "Invalid Grasp width tolerance")
    check(0 <= config["close_trigger_width_m"] < config["release_trigger_width_m"] <= maximum,
          "Invalid gripper close/release hysteresis")
    check(0 < config["measurement_max_age_s"] <= 0.5, "Invalid gripper measurement freshness limit")
    command_type = goal["command_type"]
    check(command_type in (None, "move", "grasp"), "Invalid accepted gripper command_type")
    check(goal.get("valid") is True if command_type is not None else goal.get("valid") is False,
          "Accepted gripper command_type and valid disagree")
    if goal.get("valid") is True:
        check(finite(goal.get("speed_m_s")) and 0 < goal["speed_m_s"] <= 0.1,
              "Accepted Franka Hand speed must be in (0, 0.1] m/s")
    if goal.get("requested_width_m") is not None:
        check(finite(goal["requested_width_m"]) and 0 <= goal["requested_width_m"] <= maximum
              and finite(goal.get("requested_speed_m_s")) and 0 < goal["requested_speed_m_s"] <= 0.1,
              "Invalid requested Franka Hand width/speed")
    if policy == "grasp_hold_v2":
        if goal.get("valid") is True:
            check(goal["speed_m_s"] == config["command_speed_m_s"],
                  "Accepted speed differs from its policy_config")
        if goal.get("requested_speed_m_s") is not None:
            check(finite(goal["requested_speed_m_s"])
                  and goal["requested_speed_m_s"] == config["command_speed_m_s"],
                  "Requested speed differs from its policy_config")
    fields = ("force_n", "epsilon_inner_m", "epsilon_outer_m")
    if command_type == "grasp":
        for field, config_field in zip(fields, ("grasp_force_n", "grasp_epsilon_inner_m", "grasp_epsilon_outer_m")):
            check(finite(goal[field]) and goal[field] == config[config_field],
                  "Accepted Grasp parameter differs from its policy_config: " + field)
    else:
        check(all(goal[field] is None for field in fields), "Move/unaccepted command cannot claim Grasp parameters")
    reason = goal["result_cancellation_reason"]
    check(reason in (None, "operator_open"), "Invalid gripper cancellation reason")
    if goal["control_state"] == "holding":
        check(command_type == "grasp" and goal["status"] in ("completed", "pending"),
              "Holding requires a completed accepted Grasp")
    if reason is not None:
        accepted, request = goal.get("accepted_sequence"), goal.get("sequence")
        # Provenance remains valid when a later freshness/withdrawal fault stops
        # release. Forward that real fault intact; recordability is checked below.
        check(isinstance(accepted, int) and not isinstance(accepted, bool) and accepted > 0
              and isinstance(request, int) and not isinstance(request, bool) and request >= accepted
              and command_type == "grasp"
              and isinstance(goal.get("result_sequence"), int) and not isinstance(goal.get("result_sequence"), bool)
              and goal.get("result_sequence") == accepted
              and isinstance(goal.get("result_status"), int) and not isinstance(goal.get("result_status"), bool)
              and goal.get("result_status") == 5 and isinstance(goal.get("result_success"), bool)
              and goal.get("result_error") in ("", "libfranka gripper: Command aborted!"),
              "operator_open must retain the exact canceled accepted Grasp result")
        normal_cancel = (goal["status"] == "pending" and goal["control_state"] == "canceling"
                         and goal.get("error") is None)
        cancel_fault = (goal["status"] == "error" and goal["control_state"] == "fault"
                        and isinstance(goal.get("error"), str) and bool(goal["error"]))
        check(normal_cancel or cancel_fault, "Invalid canceled Grasp control status/error")
    return {"schema_version": 2, "control_policy": policy, "policy_config": dict(config)}


def _gripper_execution_kind(goal):
    """Return only explicitly supported recordable non-success diagnostics."""
    profile = _gripper_control_profile(goal)
    def check(condition, message):
        if not condition:
            raise ValueError(message)
    status = goal.get("status")
    accepted, request = goal.get("accepted_sequence"), goal.get("sequence")
    check(goal.get("valid") is True and isinstance(accepted, int) and not isinstance(accepted, bool)
          and accepted > 0 and isinstance(request, int) and not isinstance(request, bool) and request >= accepted,
          "Missing/invalid accepted gripper target")
    raw = [goal.get("result_" + key) for key in ("success", "status", "sequence", "error")]
    has_result = any(value is not None for value in raw)
    check(not has_result or all(value is not None for value in raw), "Incomplete raw accepted gripper result")
    if has_result:
        check(isinstance(raw[0], bool) and isinstance(raw[1], int) and not isinstance(raw[1], bool)
              and raw[1] >= 0 and isinstance(raw[2], int) and not isinstance(raw[2], bool)
              and raw[2] == accepted and isinstance(raw[3], str), "Invalid/mismatched raw accepted gripper result")
    is_move = profile["schema_version"] == 1 or goal["command_type"] == "move"
    completed = has_result and raw == [True, 4, accepted, ""]
    ordinary_not_reached = has_result and raw == [False, 6, accepted, ""] and is_move
    kfail = (has_result and raw == [False, 6, accepted, "libfranka gripper: Command failed!"] and is_move)
    cancellation = profile["schema_version"] == 2 and goal["result_cancellation_reason"] == "operator_open"
    if cancellation:
        check(status == "pending" and goal["control_state"] == "canceling" and goal.get("error") is None
              and request > accepted and isinstance(goal.get("requested_width_m"), (int, float))
              and not isinstance(goal.get("requested_width_m"), bool)
              and math.isfinite(goal["requested_width_m"])
              and profile["policy_config"]["release_trigger_width_m"] <= goal["requested_width_m"] <= goal["max_width_m"],
              "Canceled Grasp is faulty or no longer has a valid explicit opening request")
        return "operator_open_canceled_grasp"
    if kfail:
        check((status == "error" and goal.get("error") == "Move failed (status=6): libfranka gripper: Command failed!")
              or (status == "pending" and request > accepted and goal.get("error") in (None, "")),
              "Accepted Move kFail requires exact execution diagnostics or pending later request")
        return "accepted_move_kfail"
    check(profile["schema_version"] != 2 or goal["control_state"] != "fault", "Gripper control state is fault")
    check(status in ("pending", "accepted", "completed", "not_reached") and goal.get("error") in (None, ""),
          "Missing/failed accepted gripper target")
    check(not has_result or completed or ordinary_not_reached, "Unsupported/failed raw accepted gripper result")
    check(status != "not_reached" or ordinary_not_reached, "not_reached requires an ordinary accepted Move result")
    check(status != "completed" or completed or profile["schema_version"] == 1 and not has_result,
          "Completed gripper target disagrees with its raw result")
    if profile["schema_version"] == 2 and goal["control_state"] == "holding":
        check(completed, "Holding requires a successful accepted Grasp result")
    return None


# 相机序列号。可用 TAKE_DATA_CAMERA_SERIALS 覆盖，例如：
# export TAKE_DATA_CAMERA_SERIALS=233522074047,405622074065
CAMERA_SERIAL_NUMBERS = [
    '243722070232',  # 相机1
    '243622073040',  # 相机2
    '243722073691'   # 相机3
]

# Local monorepo configuration. ROS and hardware execute only after the GUI's explicit start.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAPTURE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TAKE_DATA_DIR = REPO_ROOT
LOCAL_DATA_DIR = os.path.join(REPO_ROOT, "data", "raw")
LOCAL_BRIDGE_SCRIPT = os.path.join(CAPTURE_DIR, "remote_data_bridge.py")
LOCAL_GELLO_BRIDGE_SCRIPT = ""
ROBOT_BACKEND = "local_ros2"
LOCAL_GELLO_ROOT = REPO_ROOT
LOCAL_GELLO_PYTHON = ""
LOCAL_GELLO_PORT = os.environ.get("GELLO_PORT", os.environ.get("LOCAL_GELLO_PORT", ""))
LOCAL_ACTION_HOST = os.environ.get("LOCAL_ACTION_HOST", "127.0.0.1")
LOCAL_ROS2_CONTROL_SCRIPT = os.path.join(CAPTURE_DIR, "start_local_ros2_gello_fr3.sh")
LOCAL_ROS2_RELATIVE_CONFIG = ""
LOCAL_ROS2_RELATIVE_NODE = ""
LOCAL_ROS2_LOG = os.environ.get("LOCAL_ROS2_LOG", "/tmp/pi05_capture_ros2.log")
LOCAL_ROS2_BRIDGE_LOG = os.environ.get("LOCAL_ROS2_BRIDGE_LOG", "/tmp/pi05_capture_bridge.log")
LOCAL_ROS2_DISTRO = "humble"
LOCAL_ROS2_WS = os.environ.get("LOCAL_ROS2_WS", os.path.join(REPO_ROOT, "ros2_ws"))
LOCAL_ROS2_FRANKA_WS = LOCAL_ROS2_GELLO_WS = LOCAL_ROS2_WS
LOCAL_ROS2_GELLO_CFG = os.environ.get("LOCAL_ROS2_GELLO_CFG", "my_gello.yaml")
LOCAL_ROS2_FR3_CFG = os.environ.get("LOCAL_ROS2_FR3_CFG", "example_fr3_config.yaml")
LOCAL_ROS2_GRIPPER_TYPE = "franka_hand"
LOCAL_ROS2_DEFAULT_GRIPPER_CFG = "example_fr3_config_franka_hand_move.yaml"
LOCAL_ROS2_GRIPPER_CFG = LOCAL_ROS2_DEFAULT_GRIPPER_CFG
LOCAL_ROS2_SETUP_CMD = ""
LOCAL_ROS2_USE_GRIPPER = "1"
LOCAL_ROS2_EXEC_MODE = "container"
LOCAL_ROS2_DOCKER_COMPOSE = os.environ.get("LOCAL_ROS2_DOCKER_COMPOSE", os.path.join(REPO_ROOT, "docker", "compose.yaml"))
LOCAL_ROS2_DOCKER_HOST = os.environ.get("LOCAL_ROS2_DOCKER_HOST", "unix:///var/run/docker.sock").strip()
LOCAL_ROS2_DOCKER_SERVICE = os.environ.get("LOCAL_ROS2_DOCKER_SERVICE", "ros2")
LOCAL_ROS2_CONTAINER_WS = "/workspace/ros2"
LOCAL_ROS2_CONTAINER_CONTROL_SCRIPT = "/tmp/pi05_capture_ros2_launcher.sh"
LOCAL_ROS2_CONTAINER_BRIDGE_SCRIPT = "/tmp/pi05_capture_remote_data_bridge.py"
LOCAL_ROS2_CONTAINER_AUTOSTART = os.environ.get("LOCAL_ROS2_CONTAINER_AUTOSTART", "1") != "0"
LOCAL_ROS2_READY_FILE = "/tmp/pi05_capture_ros2_ready"
LOCAL_ROS2_STOP_SCRIPT = os.path.join(CAPTURE_DIR, "stop_capture.sh")
CAPTURE_SESSION_FILE = os.environ.get("PI05_CAPTURE_SESSION_FILE", os.path.join(REPO_ROOT, "runs", "capture", "session.json"))
REMOTE_PORT = int(os.environ.get("PI05_CAPTURE_PORT", "9999"))
REMOTE_HOST = LOCAL_ACTION_HOST
REMOTE_USER = REMOTE_PASSWORD = ""
REMOTE_ABSOLUTE_SCRIPT = REMOTE_RELATIVE_SCRIPT = REMOTE_RELATIVE_CONFIG = ""
REMOTE_BRIDGE_DIR = REMOTE_STOP_SCRIPT = REMOTE_REPLAY_SCRIPT = REMOTE_REPLAY_DIR = ""
REMOTE_REPLAY_ACTION_DIR = REMOTE_REPLAY_VIDEO_DIR = ""
REMOTE_ROBOT_IP = os.environ.get("FRANKA_IP", os.environ.get("ROBOT_IP", "172.16.0.2"))
GRIPPER_MAX_WIDTH = 0.085
FR3_JOINT_ORDER = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]
FINGER_JOINT_ORDER = ["fr3_finger_joint1", "fr3_finger_joint2"]
CAPTURE_CONTRACT = "fr3_joint_position_v1"
GRIPPER_MOVE_KFAIL = "libfranka gripper: Command failed!"
GRIPPER_EXECUTION_POLICY = "accepted_move_kfail_with_raw_diagnostics_v1"
GRIPPER_EXECUTION_POLICY_V2 = "grasp_hold_and_accepted_move_with_raw_diagnostics_v2"
STATE_FEATURE_NAMES = FR3_JOINT_ORDER + ["gripper_width_m"]
ACTION_FEATURE_NAMES = FR3_JOINT_ORDER + ["gripper_target_width_m"]
CAPTURE_SOURCE_KEYS = ["gello_joints", "franka_joints", "gripper_joints", "gripper_command",
                       "end_effector_pose", "arm_command", "gripper_target"]
JOINT_NUMBER_RE = re.compile(r"joint([1-7])$", re.IGNORECASE)


class VideoActionRecorder:
    def __init__(self):
        # 基础路径
        self.base_path = os.environ.get("TAKE_DATA_OUTPUT_DIR", LOCAL_DATA_DIR)
        self.record_base_path = self.base_path
        self.legacy_video_base_path = os.path.join(self.base_path, "video")
        self.legacy_action_base_path = os.path.join(self.base_path, "action")
        os.makedirs(self.record_base_path, exist_ok=True)
        self.current_video_path = self.record_base_path
        self.current_action_path = self.record_base_path

        # 录制状态
        self.is_recording = False
        self.running = True
        self.session_dir = None
        self.session_index = None
        self.record_start_time = None
        self.record_lock = threading.Lock()
        self.replay_in_progress = False

        # 视频相关
        self.pipelines = []
        self.preview_thread = None
        self.video_writers = []
        self.frame_lock = threading.Lock()
        self.preview_fps = 30.0
        self.record_fps = 15.0
        self.frame_dirs = []
        self.video_output_paths = []
        self.frame_index = 0
        self.disable_cameras = os.environ.get("TAKE_DATA_DISABLE_CAMERAS", "0") == "1"
        self.last_action_record_time = 0.0
        self.camera_serials = []
        self.camera_ready = False
        self.camera_last_frame_times = []
        self.camera_frame_error = None
        self.video_frame_metadata = []
        self.record_metadata = None
        self.recording_error = None
        self.task_text = os.environ.get("TAKE_DATA_TASK", "Put the avocado in the basket.").strip()
        self.outcome = "success"
        self.record_activation_id = None

        # 动作数据相关
        self.action_socket = None
        self.action_data_buffer = []
        self.action_thread = None
        self.robot_ready = False
        self.autostart_robot = False  # Explicit GUI start is required for every session.
        self._robot_starting = False
        self._robot_start_attempted = False
        self._robot_start_lock = threading.Lock()
        self._robot_stopping = False
        self._robot_stop_requested = threading.Event()
        self._robot_stop_error = None
        self._exit_requested = False
        self.local_ros2_container_id = None
        self.local_ros2_session_id = None
        self.latest_action_data = None
        self.action_sample_lock = threading.Lock()
        self.action_sample_buffer = deque(maxlen=512)
        self.action_sample_window_sec = 5.0
        self.action_time_offsets = deque(maxlen=200)
        self.action_time_offset = None
        self.control_mode_var = None
        self.active_control_mode = None
        self.local_gello_process = None
        self.local_gello_log_file = None
        self.local_ros2_control_process = None
        self.local_ros2_bridge_process = None
        self.local_ros2_log_file = None
        self.local_ros2_bridge_log_file = None
        self.relative_zero_timeout_sec = 10.0
        self.relative_zero_tolerance = 0.05
        self._action_redraw_pending = False
        self.action_redraw_interval_ms = 80
        self.action_history = deque(maxlen=600)
        self._last_history_append_time = -float("inf")
        self.action_history_lock = threading.Lock()
        self._joint_order_cache = {}
        self.history_window_sec = 10.0
        self.preview_size = (0, 0)
        self.ui_state_path = os.path.join(
            os.path.expanduser("~"),
            ".config",
            "pi05-fr3-gello",
            "capture_ui_state.json"
        )
        self.ui_state = self.load_ui_state()
        self._splitter_restored = False

        # Own the shared old/new FR3 lock before either camera is opened.
        self._acquire_fr3_control_lease()

        # 创建GUI
        self.create_gui()

        # 初始化相机
        if self.disable_cameras:
            self.camera_status_label.config(text="相机: 已禁用（仅录动作）", fg="#95a5a6")
            self.preview_label.config(text="相机已禁用，仅显示和录制动作数据", fg="white")
        else:
            self.init_cameras()

        # 设置清理钩子
        self.setup_cleanup_hooks()

        # 开始预览
        if not self.disable_cameras:
            self.start_preview()

        # 启动远程数据流
        if self.autostart_robot:
            self.start_action_stream()

    def get_chinese_font(self, size, weight="normal"):
        """获取中文字体"""
        if not hasattr(self, "_font_cache"):
            self._font_cache = {}

        cache_key = (size, weight)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        if not hasattr(self, "_font_family"):
            available = set(tkfont.families(self.root))
            preferred = [
                "思源黑体",
                "Source Han Sans SC",
                "Source Han Sans CN",
                "Source Han Sans",
                "Noto Sans CJK SC",
                "Noto Sans CJK",
                "WenQuanYi Zen Hei",
                "WenQuanYi Micro Hei",
                "Droid Sans Fallback",
                "Microsoft YaHei",
                "PingFang SC",
                "SimHei"
            ]
            self._font_family = next((f for f in preferred if f in available), None)

        if self._font_family:
            font_obj = tkfont.Font(
                root=self.root,
                family=self._font_family,
                size=size,
                weight=weight
            )
            actual = font_obj.actual()
            if actual.get("family") != self._font_family:
                font_obj = tkfont.Font(
                    root=self.root,
                    family=self._font_family,
                    size=size,
                    weight="normal"
                )
        else:
            font_obj = tkfont.Font(root=self.root, size=size, weight=weight)
        self._font_cache[cache_key] = font_obj
        return font_obj

    def setup_cleanup_hooks(self):
        """设置清理钩子，确保程序退出时清理机器人进程"""
        atexit.register(self.cleanup_remote_processes)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """信号处理器"""
        print("\n接收到退出信号，正在清理...")
        self.cleanup_and_exit()

    def load_ui_state(self):
        """加载界面布局状态"""
        try:
            if os.path.exists(self.ui_state_path):
                with open(self.ui_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _set_label_text(self, label, text, fg=None):
        """线程安全更新标签文字"""
        def _update():
            if fg is None:
                label.config(text=text)
            else:
                label.config(text=text, fg=fg)

        try:
            self.root.after(0, _update)
        except Exception:
            pass

    def _on_content_resize(self, event):
        """保持上下区域50/50高度"""
        total = event.height
        if total <= 0:
            return
        half = max(int(total / 2), 1)
        event.widget.grid_rowconfigure(0, minsize=half)
        event.widget.grid_rowconfigure(1, minsize=half)

    def _build_action_canvases(self):
        """创建动作曲线画布"""
        self.action_canvases = []
        self.action_canvas_specs = [
            {"key": "pos_x", "title": "位置 X (m)", "color": "#e74c3c", "ymin": -1.0, "ymax": 1.0, "min_span": 0.02},
            {"key": "pos_y", "title": "位置 Y (m)", "color": "#2ecc71", "ymin": -1.0, "ymax": 1.0, "min_span": 0.02},
            {"key": "pos_z", "title": "位置 Z (m)", "color": "#3498db", "ymin": -1.0, "ymax": 1.0, "min_span": 0.02},
            {"key": "grip", "title": "夹爪开合 (m)", "color": "#ecf0f1", "ymin": 0.0, "ymax": 0.085, "min_span": 0.01},
            {"key": "ori_x", "title": "姿态 Qx", "color": "#9b59b6", "ymin": -1.0, "ymax": 1.0, "min_span": 0.1},
            {"key": "ori_y", "title": "姿态 Qy", "color": "#1abc9c", "ymin": -1.0, "ymax": 1.0, "min_span": 0.1},
            {"key": "ori_z", "title": "姿态 Qz", "color": "#f1c40f", "ymin": -1.0, "ymax": 1.0, "min_span": 0.1},
            {"key": "ori_w", "title": "姿态 Qw", "color": "#e67e22", "ymin": -1.0, "ymax": 1.0, "min_span": 0.1},
        ]

        cols = 2
        rows = (len(self.action_canvas_specs) + cols - 1) // cols
        for i in range(len(self.action_canvas_specs)):
            canvas = tk.Canvas(self.action_canvas_frame, bg="#1f2d3a", highlightthickness=0)
            row = i % rows
            col = i // rows
            canvas.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
            self.action_canvas_frame.grid_rowconfigure(row, weight=1)
            self.action_canvases.append(canvas)
        for col in range(cols):
            self.action_canvas_frame.grid_columnconfigure(col, weight=1)

    def _compute_series_range(self, points, default_min, default_max, min_span):
        """根据数据自动计算显示范围"""
        min_v = None
        max_v = None
        for _, v in points:
            try:
                v = float(v)
            except Exception:
                continue
            if min_v is None or v < min_v:
                min_v = v
            if max_v is None or v > max_v:
                max_v = v
        if min_v is None or max_v is None:
            return default_min, default_max

        span = max_v - min_v
        if span < min_span:
            center = (min_v + max_v) / 2
            span = min_span
            min_v = center - span / 2
            max_v = center + span / 2
        else:
            pad = span * 0.1
            min_v -= pad
            max_v += pad
        return min_v, max_v

    def save_ui_state(self):
        """保存界面布局状态"""
        try:
            os.makedirs(os.path.dirname(self.ui_state_path), exist_ok=True)
            with open(self.ui_state_path, "w", encoding="utf-8") as f:
                json.dump(self.ui_state, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _format_path_label(self, prefix, path):
        safe_path = (path or "").rstrip("/")
        if safe_path:
            return f"{prefix}: {safe_path}/"
        return f"{prefix}: --"

    def set_current_paths(self, video_path=None, action_path=None):
        """更新当前路径并刷新标签"""
        if video_path:
            self.current_video_path = video_path
            if hasattr(self, "video_path_label"):
                self.video_path_label.config(
                    text=self._format_path_label("视频", video_path)
                )
        if action_path:
            self.current_action_path = action_path
            if hasattr(self, "action_path_label"):
                self.action_path_label.config(
                    text=self._format_path_label("动作", action_path)
                )

    def open_path_in_file_manager(self, path):
        """打开保存路径目录"""
        if not path:
            messagebox.showinfo("提示", "暂无可打开的路径")
            return
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            messagebox.showwarning("提示", f"路径不存在: {abs_path}")
            return
        try:
            subprocess.Popen(["xdg-open", abs_path])
        except Exception as e:
            messagebox.showerror("错误", f"无法打开路径: {str(e)}")

    def _get_split_ratio(self):
        """获取分割条当前位置占比"""
        try:
            total = self.main_paned.winfo_height()
            if total <= 0:
                return None
            _, sash_y = self.main_paned.sash_coord(0)
            ratio = sash_y / total
            return max(0.2, min(0.85, ratio))
        except Exception:
            return None

    def _apply_split_ratio(self, ratio):
        """按占比设置分割条位置"""
        try:
            total = self.main_paned.winfo_height()
            if total <= 0:
                return
            sash_y = int(total * ratio)
            self.main_paned.sash_place(0, 0, sash_y)
        except Exception:
            pass

    def _restore_splitter(self):
        """恢复分割条位置"""
        ratio = self.ui_state.get("split_ratio", 0.75)
        self._apply_split_ratio(ratio)

    def _on_splitter_release(self, event):
        """拖动分割条后自动保存"""
        ratio = self._get_split_ratio()
        if ratio is not None:
            self.ui_state["split_ratio"] = ratio
            self.save_ui_state()

    def _on_paned_configure(self, event):
        """首次布局完成后恢复分割条位置"""
        if not self._splitter_restored:
            self._splitter_restored = True
            self._restore_splitter()

    def start_action_stream(self):
        """启动远程系统并持续接收动作数据（后台运行）"""
        with self._robot_start_lock:
            if self._robot_starting or self._robot_stopping or self._robot_stop_error or self.action_socket:
                return
            self._robot_stop_requested.clear()
            self._robot_starting = True
            self._robot_start_attempted = True
        self._update_teleop_button()

        def background_start():
            try:
                if not self.start_remote_robot(control_mode=self._get_selected_control_mode()):
                    return
                if self._robot_stop_requested.is_set():
                    return
                if not self.action_thread or not self.action_thread.is_alive():
                    self.action_thread = threading.Thread(target=self.receive_action_data, daemon=True)
                    self.action_thread.start()
            finally:
                with self._robot_start_lock:
                    self._robot_starting = False
                self._update_teleop_button()

        # 在后台线程启动，不阻塞GUI
        startup_thread = threading.Thread(target=background_start, daemon=True)
        startup_thread.start()

    def _update_teleop_button(self):
        def update():
            connected = self.action_socket is not None
            self.teleop_button.config(
                state=tk.DISABLED if self._robot_starting or self._robot_stopping or self._robot_stop_error or connected else tk.NORMAL,
                text="正在停止遥操作..." if self._robot_stopping else "正在启动遥操作..." if self._robot_starting else (
                    "遥操作已启动" if connected else "启动遥操作"
                ),
            )
            if hasattr(self, "stop_teleop_button"):
                self.stop_teleop_button.config(
                    state=tk.DISABLED if self._robot_stopping else tk.NORMAL,
                    text="正在停止并核验..." if self._robot_stopping else (
                        "重试停止遥操作" if self._robot_stop_error else "停止遥操作"
                    ),
                )
        if self.running:
            self.root.after(0, update)

    def stop_teleoperation(self, exit_after=False):
        """在容器内停止并核验控制进程；停止失败时保留窗口。"""
        if ROBOT_BACKEND != "local_ros2" or not self._local_ros2_use_container():
            return
        with self._robot_start_lock:
            self._exit_requested = self._exit_requested or exit_after
            if self._robot_stopping:
                return
            self._robot_stopping = True
            self._robot_stop_requested.set()
            self._robot_stop_error = None
        self._set_label_text(self.robot_ready_label, "机器人状态: 正在停止并核验容器进程...", "#f39c12")
        self._update_teleop_button()

        def worker():
            error = None
            try:
                self.cleanup_remote_processes(force=True)
            except Exception as exc:
                error = str(exc)

            def finish():
                self._robot_stopping = False
                self._robot_stop_error = error
                if error:
                    self._exit_requested = False
                    self._set_label_text(self.robot_ready_label, "机器人状态: 停止失败，未确认停机", "#e74c3c")
                    messagebox.showerror("停止遥操作失败", "未确认机械臂控制已停止，窗口将保持打开。\n请保持急停可用并重试停止。\n\n" + error)
                else:
                    self._robot_start_attempted = False
                    self._set_label_text(self.robot_connection_label, "机器人连接: 遥操作已停止", "#95a5a6")
                    self._set_label_text(self.robot_ready_label, "机器人状态: 已停止（控制进程已核验）", "#27ae60")
                    if self.is_recording:
                        self.stop_recording()
                    if self._exit_requested:
                        self._finish_exit()
                        return
                self._update_teleop_button()
            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def _validate_replay_index(self, value):
        if value == "":
            return True
        return value.isdigit()

    def _on_replay_index_change(self):
        try:
            value = int(self.replay_index_var.get())
        except (tk.TclError, ValueError):
            return
        if value < 1:
            self.replay_index_var.set(1)

    def _update_replay_controls(self):
        if ROBOT_BACKEND in ("local_gello", "local_ros2"):
            if hasattr(self, "replay_button"):
                self.replay_button.config(state=tk.DISABLED)
            if hasattr(self, "replay_spinbox"):
                self.replay_spinbox.config(state=tk.DISABLED)
            return

        last_index = self.get_last_record_index()
        if last_index:
            self.replay_spinbox.config(to=last_index)
            if self.replay_index_var.get() > last_index:
                self.replay_index_var.set(last_index)
        else:
            self.replay_spinbox.config(to=1)
            self.replay_index_var.set(1)

        if self.replay_in_progress or self.is_recording or not last_index:
            self.replay_button.config(state=tk.DISABLED)
        else:
            self.replay_button.config(state=tk.NORMAL)

        spinbox_state = tk.DISABLED if self.replay_in_progress else tk.NORMAL
        self.replay_spinbox.config(state=spinbox_state)

    def _get_selected_control_mode(self):
        if ROBOT_BACKEND == "local_ros2":
            return "absolute"
        if self.control_mode_var is None:
            return "relative"
        value = self.control_mode_var.get()
        return value if value in ("absolute", "relative") else "relative"

    def _set_mode_controls_state(self, state):
        if ROBOT_BACKEND in ("local_gello", "local_ros2"):
            state = tk.DISABLED
        if hasattr(self, "mode_abs_radio"):
            self.mode_abs_radio.config(state=state)
        if hasattr(self, "mode_rel_radio"):
            self.mode_rel_radio.config(state=state)

    def _on_control_mode_change(self):
        if ROBOT_BACKEND in ("local_gello", "local_ros2"):
            if self.active_control_mode and self.control_mode_var is not None:
                self.control_mode_var.set(self.active_control_mode)
            return
        new_mode = self._get_selected_control_mode()
        if self.is_recording or self.replay_in_progress:
            messagebox.showwarning("提示", "录制或回放中无法切换控制模式")
            if self.active_control_mode:
                self.control_mode_var.set(self.active_control_mode)
            return
        if self.active_control_mode == new_mode:
            return
        self._set_mode_controls_state(tk.DISABLED)
        switch_thread = threading.Thread(
            target=self._switch_control_mode_worker,
            args=(new_mode,),
            daemon=True,
        )
        switch_thread.start()

    def _switch_control_mode_worker(self, new_mode):
        try:
            self._set_label_text(self.robot_ready_label, "机器人状态: 切换控制模式...", "#f39c12")
            if self.robot_ready or self.action_socket:
                try:
                    self._pause_remote_control()
                except Exception as e:
                    self._set_label_text(self.robot_ready_label, f"机器人状态: {e}", "#e74c3c")
            if not self.start_remote_robot(control_mode=new_mode):
                raise RuntimeError("切换控制模式失败")
            if new_mode == "relative":
                self._set_relative_control_enabled(False)
            self.active_control_mode = new_mode
        except Exception as e:
            def _revert():
                if self.active_control_mode:
                    self.control_mode_var.set(self.active_control_mode)
                messagebox.showerror("错误", f"切换控制模式失败: {e}")
            self.root.after(0, _revert)
        finally:
            self.root.after(0, lambda: self._set_mode_controls_state(tk.NORMAL))

    def _get_replay_index(self):
        try:
            value = int(self.replay_index_var.get())
        except (tk.TclError, ValueError):
            return None
        if value < 1:
            return None
        return value

    def _find_action_json(self, action_dir):
        candidates = []
        for name in os.listdir(action_dir):
            if name.startswith("action_data_") and name.endswith(".json"):
                candidates.append(os.path.join(action_dir, name))
        if not candidates:
            return None
        return max(candidates, key=os.path.getmtime)

    def _get_replay_sources(self, session_index):
        record_name = f"record_{session_index:03d}"
        record_dir = os.path.join(self.record_base_path, record_name)
        action_dir = os.path.join(record_dir, "action")
        video_dir = os.path.join(record_dir, "video")
        if not os.path.isdir(action_dir):
            action_dir = os.path.join(self.legacy_action_base_path, record_name)
        if not os.path.isdir(video_dir):
            video_dir = os.path.join(self.legacy_video_base_path, record_name)
        if not os.path.isdir(action_dir):
            raise FileNotFoundError(f"未找到动作数据目录: {record_name}")
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(f"未找到视频数据目录: {record_name}")

        json_path = self._find_action_json(action_dir)
        if not json_path:
            raise FileNotFoundError(f"{record_name} 中未找到动作JSON文件")

        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("schema_version", 1) >= 2 or payload.get("capture_contract") == CAPTURE_CONTRACT:
            raise ValueError("新版采集保存的是 FR3 绝对关节与夹爪目标；旧回放器未实现此动作契约，已禁止运动回放")

        frames = payload.get("data") or []
        if not frames:
            raise ValueError(f"{record_name} 动作数据为空")
        return json_path, action_dir, video_dir

    def _ensure_remote_replay_script(self, *args, **kwargs):
        raise RuntimeError("此仓库仅支持本机 ROS2 容器采集；旧远程/回放入口已禁用。")


    def _prepare_replay_json(self, json_path):
        """补齐回放所需的夹爪关节字段"""
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("schema_version", 1) >= 2 or payload.get("capture_contract") == CAPTURE_CONTRACT:
            raise ValueError("旧回放器不支持新版绝对目标契约，已禁止运动回放")

        frames = payload.get("data") or []
        if not frames:
            return json_path

        needs_rewrite = False
        for frame in frames:
            gripper_data = frame.get("gripper_joints") or {}
            names = gripper_data.get("names") or []
            positions = gripper_data.get("position") or []

            has_canonical = (
                isinstance(names, list)
                and "fr3_finger_joint1" in names
                and "fr3_finger_joint2" in names
                and isinstance(positions, list)
                and len(positions) >= 2
            )
            if has_canonical:
                continue

            width = None
            if frame.get("gripper_command") is not None:
                try:
                    width = float(frame["gripper_command"]) * GRIPPER_MAX_WIDTH
                except Exception:
                    width = None
            if width is None:
                width = self._extract_gripper_width(names, positions)
            if width is None:
                width = 0.0

            half = float(width) / 2.0
            frame["gripper_joints"] = {
                "names": ["fr3_finger_joint1", "fr3_finger_joint2"],
                "position": [half, half],
                "velocity": [0.0, 0.0],
                "effort": [0.0, 0.0],
            }
            needs_rewrite = True

        if not needs_rewrite:
            return json_path

        payload["packet_count"] = len(frames)
        base, _ = os.path.splitext(json_path)
        replay_path = f"{base}_replay.json"
        with open(replay_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return replay_path
    def _run_remote_replay(self, *args, **kwargs):
        raise RuntimeError("此仓库仅支持本机 ROS2 容器采集；旧远程/回放入口已禁用。")


    def _remote_process_exists(self, pattern):
        cmd = f"pgrep -f {shlex.quote(pattern)} >/dev/null 2>&1"
        result = subprocess.run(
            self._ssh_base_cmd() + [f"{REMOTE_USER}@{REMOTE_HOST}", cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0

    def _remote_any_process_exists(self, patterns):
        if not patterns:
            return False
        escaped = [re.escape(p) for p in patterns]
        regex = "|".join(escaped)
        return self._remote_process_exists(regex)

    def _wait_for_remote_process(self, patterns, timeout, status_prefix):
        start = time.monotonic()
        if isinstance(patterns, str):
            patterns = [patterns]
        while time.monotonic() - start < timeout:
            if self._remote_any_process_exists(patterns):
                return True
            elapsed = int(time.monotonic() - start)
            self._set_label_text(
                self.robot_ready_label,
                f"{status_prefix} ({elapsed}s)",
                "#f39c12",
            )
            time.sleep(1)
        return False

    def _pause_remote_control(self):
        self._stop_local_ros2_container()


    def start_replay(self):
        messagebox.showwarning("运动回放不可用", "此采集界面不发送录制数据作为运动命令；查看请打开 MP4。")


    def _replay_worker(self, *args, **kwargs):
        raise RuntimeError("此仓库仅支持本机 ROS2 容器采集；旧远程/回放入口已禁用。")


    def create_gui(self):
        """创建GUI界面"""
        self.root = tk.Tk()
        self.root.title("视频+动作数据同步录制系统")
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        window_w = min(1920, max(1, screen_w - 40))
        window_h = max(1, screen_h - 40)
        self.root.geometry(f"{window_w}x{window_h}")
        self.root.configure(bg="#2c3e50")

        # 设置窗口关闭协议
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)

        # 字体
        self.font_title = self.get_chinese_font(18, "bold")
        self.font_large = self.get_chinese_font(14, "bold")
        self.font_medium = self.get_chinese_font(12)
        self.font_small = self.get_chinese_font(10)
        self.font_tiny = self.get_chinese_font(9)
        self.font_panel_title = self.get_chinese_font(13, "bold")
        self.font_panel_section = self.get_chinese_font(11, "bold")
        self.font_panel_text = self.get_chinese_font(10)
        self.font_panel_small = self.get_chinese_font(9)

        # ====================
        # 中间内容区域
        # ====================
        content_frame = tk.Frame(self.root, bg="#2c3e50")
        content_frame.pack(fill="both", expand=True, padx=8, pady=8)
        content_frame.grid_rowconfigure(0, weight=1, uniform="main")
        content_frame.grid_rowconfigure(1, weight=1, uniform="main")
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.bind("<Configure>", self._on_content_resize)

        # 上方：预览区域
        video_frame = tk.Frame(content_frame, bg="#34495e", relief="raised", borderwidth=2)
        video_frame.grid(row=0, column=0, sticky="nsew")
        video_frame.grid_rowconfigure(0, weight=0)
        video_frame.grid_rowconfigure(1, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        preview_title = tk.Label(
            video_frame,
            text="实时预览（相机拼接）",
            font=self.font_large,
            fg="white",
            bg="#34495e"
        )
        preview_title.grid(row=0, column=0, pady=6)

        preview_container = tk.Frame(video_frame, bg="#34495e")
        preview_container.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        # 预览画布
        self.preview_label = tk.Label(preview_container, bg="#34495e", anchor="center")
        self.preview_label.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.preview_label.bind("<Configure>", self._on_preview_resize)

        # 下方：动作可视化 + 控制面板
        bottom_frame = tk.Frame(content_frame, bg="#2c3e50")
        bottom_frame.grid(row=1, column=0, sticky="nsew")
        bottom_frame.grid_rowconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=0, minsize=420)

        action_frame = tk.Frame(bottom_frame, bg="#34495e", relief="raised", borderwidth=2)
        action_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        action_frame.grid_rowconfigure(0, weight=0)
        action_frame.grid_rowconfigure(1, weight=1)
        action_frame.grid_columnconfigure(0, weight=1)

        action_title = tk.Label(
            action_frame,
            text="动作可视化",
            font=self.font_large,
            fg="white",
            bg="#34495e"
        )
        action_title.grid(row=0, column=0, pady=6)

        self.action_canvas_frame = tk.Frame(action_frame, bg="#1f2d3a")
        self.action_canvas_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self._build_action_canvases()
        self._schedule_action_redraw()

        # 右下：控制区域
        right_frame = tk.Frame(bottom_frame, bg="#34495e", relief="raised", borderwidth=2, width=420)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_frame.grid_propagate(False)
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        control_canvas = tk.Canvas(
            right_frame,
            bg="#34495e",
            highlightthickness=0,
            borderwidth=0
        )
        control_scrollbar = tk.Scrollbar(
            right_frame,
            orient="vertical",
            command=control_canvas.yview
        )
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        control_canvas.grid(row=0, column=0, sticky="nsew")
        control_scrollbar.grid(row=0, column=1, sticky="ns")

        control_content = tk.Frame(control_canvas, bg="#34495e")
        control_window_id = control_canvas.create_window(
            (0, 0),
            window=control_content,
            anchor="nw"
        )

        def _on_control_canvas_configure(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))

        def _on_control_canvas_width(event):
            control_canvas.itemconfigure(control_window_id, width=event.width)

        control_content.bind("<Configure>", _on_control_canvas_configure)
        control_canvas.bind("<Configure>", _on_control_canvas_width)

        control_title = tk.Label(
            control_content,
            text="控制面板",
            font=self.font_large,
            fg="white",
            bg="#34495e"
        )
        control_title.pack(pady=6)

        # ====================
        # 系统状态框
        # ====================
        status_frame = tk.LabelFrame(
            control_content,
            text="系统状态",
            font=self.font_panel_section,
            fg="white",
            bg="#34495e",
            padx=8,
            pady=6
        )
        status_frame.pack(padx=10, pady=5, fill="x")
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_columnconfigure(1, weight=1)

        # 相机状态
        self.camera_status_label = tk.Label(
            status_frame,
            text="相机: 初始化中...",
            font=self.font_panel_text,
            fg="#f39c12",
            bg="#34495e",
            anchor="w"
        )
        self.camera_status_label.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)

        # 机器人连接状态
        self.robot_connection_label = tk.Label(
            status_frame,
            text="机器人连接: 未连接",
            font=self.font_panel_text,
            fg="#95a5a6",
            bg="#34495e",
            anchor="w"
        )
        self.robot_connection_label.grid(row=0, column=1, sticky="w", padx=(6, 0), pady=2)

        # 机器人准备状态
        self.robot_ready_label = tk.Label(
            status_frame,
            text="机器人状态: 待机中",
            font=self.font_panel_text,
            fg="#95a5a6",
            bg="#34495e",
            anchor="w"
        )
        self.robot_ready_label.grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)

        # 录制状态
        self.status_label = tk.Label(
            status_frame,
            text="录制状态: 等待录制",
            font=self.font_panel_text,
            fg="#27ae60",
            bg="#34495e",
            anchor="w"
        )
        self.status_label.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=2)

        # 控制模式
        if self.control_mode_var is None:
            self.control_mode_var = tk.StringVar(value="relative")
        mode_row = tk.Frame(status_frame, bg="#34495e")
        mode_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        mode_row.grid_columnconfigure(1, weight=1)

        mode_label = tk.Label(
            mode_row,
            text="控制模式:",
            font=self.font_panel_text,
            fg="#bdc3c7",
            bg="#34495e",
            anchor="w"
        )
        mode_label.grid(row=0, column=0, sticky="w")

        self.mode_abs_radio = tk.Radiobutton(
            mode_row,
            text="绝对",
            value="absolute",
            variable=self.control_mode_var,
            command=self._on_control_mode_change,
            font=self.font_panel_text,
            fg="white",
            bg="#34495e",
            selectcolor="#34495e",
            activebackground="#34495e",
            activeforeground="white"
        )
        self.mode_abs_radio.grid(row=0, column=1, sticky="w", padx=(6, 8))

        self.mode_rel_radio = tk.Radiobutton(
            mode_row,
            text="相对",
            value="relative",
            variable=self.control_mode_var,
            command=self._on_control_mode_change,
            font=self.font_panel_text,
            fg="white",
            bg="#34495e",
            selectcolor="#34495e",
            activebackground="#34495e",
            activeforeground="white"
        )
        self.mode_rel_radio.grid(row=0, column=2, sticky="w")

        if ROBOT_BACKEND == "local_gello":
            self.control_mode_var.set("absolute")
            self.mode_abs_radio.config(text="本机GELLO", state=tk.DISABLED)
            self.mode_rel_radio.config(state=tk.DISABLED)
        elif ROBOT_BACKEND == "local_ros2":
            self.control_mode_var.set("absolute")
            self.mode_abs_radio.config(text="本机ROS2", state=tk.DISABLED)
            self.mode_rel_radio.config(state=tk.DISABLED)

        task_frame = tk.LabelFrame(control_content, text="采集任务与结果", font=self.font_panel_section,
                                   fg="white", bg="#34495e", padx=8, pady=6)
        task_frame.pack(padx=10, pady=5, fill="x")
        tk.Label(task_frame, text="任务描述（必填）", fg="white", bg="#34495e").pack(anchor="w")
        self.task_var = tk.StringVar(value=self.task_text)
        self.task_entry = tk.Entry(task_frame, textvariable=self.task_var, font=self.font_panel_text)
        self.task_entry.pack(fill="x", pady=(2, 6))
        tk.Label(task_frame, text="本段结果（停止录制前可修改，默认 success）", fg="white", bg="#34495e").pack(anchor="w")
        self.outcome_var = tk.StringVar(value="success")
        self.outcome_menu = tk.OptionMenu(task_frame, self.outcome_var, "success", "unknown", "failure", "aborted")
        self.outcome_menu.pack(fill="x")

        # ====================
        # 录制信息框
        # ====================
        info_frame = tk.LabelFrame(
            control_content,
            text="录制信息",
            font=self.font_panel_section,
            fg="white",
            bg="#34495e",
            padx=8,
            pady=6
        )
        info_frame.pack(padx=10, pady=5, fill="x")

        # 录制时长
        time_row = tk.Frame(info_frame, bg="#34495e")
        time_row.pack(fill="x", pady=2)
        time_row.grid_columnconfigure(0, weight=1)
        time_row.grid_columnconfigure(1, weight=1)

        self.time_label = tk.Label(
            time_row,
            text="录制时长: 00:00:00",
            font=self.font_panel_text,
            fg="white",
            bg="#34495e",
            anchor="w"
        )
        self.time_label.grid(row=0, column=0, sticky="w")

        # 录制序号
        self.session_label = tk.Label(
            time_row,
            text="录制序号: --",
            font=self.font_panel_text,
            fg="#ecf0f1",
            bg="#34495e",
            anchor="e"
        )
        self.session_label.grid(row=0, column=1, sticky="e")

        # 保存路径
        self.path_title_label = tk.Label(
            info_frame,
            text="保存路径:",
            font=self.font_panel_text,
            fg="#95a5a6",
            bg="#34495e",
            anchor="w"
        )
        self.path_title_label.pack(fill="x", pady=(6, 2))

        self.video_path_label = tk.Label(
            info_frame,
            text=self._format_path_label("视频", self.current_video_path),
            font=self.font_panel_text,
            fg="#5dade2",
            bg="#34495e",
            anchor="w",
            cursor="hand2"
        )
        self.video_path_label.pack(fill="x", pady=2)

        self.action_path_label = tk.Label(
            info_frame,
            text=self._format_path_label("动作", self.current_action_path),
            font=self.font_panel_text,
            fg="#5dade2",
            bg="#34495e",
            anchor="w",
            cursor="hand2"
        )
        self.action_path_label.pack(fill="x", pady=(2, 4))
        self.video_path_label.bind(
            "<Button-1>",
            lambda event: self.open_path_in_file_manager(self.current_video_path)
        )
        self.action_path_label.bind(
            "<Button-1>",
            lambda event: self.open_path_in_file_manager(self.current_action_path)
        )

        # ====================
        # 控制按钮区域
        # ====================
        last_index = self.get_last_record_index()
        default_replay_index = last_index if last_index else 1
        self.replay_index_var = tk.IntVar(value=default_replay_index)

        button_frame = tk.Frame(control_content, bg="#34495e")
        button_frame.pack(pady=4, padx=10, fill="x")
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=0)

        self.teleop_button = tk.Button(
            button_frame,
            text="启动遥操作",
            font=self.font_panel_section,
            bg="#2980b9",
            fg="white",
            height=1,
            command=self.start_action_stream,
            cursor="hand2",
            relief="raised",
            borderwidth=3,
        )
        self.teleop_button.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        if ROBOT_BACKEND == "local_ros2" and self._local_ros2_use_container():
            self.stop_teleop_button = tk.Button(
                button_frame,
                text="停止遥操作",
                font=self.font_panel_section,
                bg="#c0392b",
                fg="white",
                height=1,
                command=self.stop_teleoperation,
                cursor="hand2",
                relief="raised",
                borderwidth=3,
            )
            self.stop_teleop_button.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=(0, 4))

        # 开始录制按钮
        self.start_button = tk.Button(
            button_frame,
            text="开始录制",
            font=self.font_panel_section,
            bg="#27ae60",
            fg="white",
            height=1,
            command=self.start_recording,
            cursor="hand2",
            relief="raised",
            borderwidth=3
        )
        self.start_button.grid(row=1, column=0, sticky="ew", pady=(4, 4))

        # 停止录制按钮
        self.stop_button = tk.Button(
            button_frame,
            text="停止录制",
            font=self.font_panel_section,
            bg="#e74c3c",
            fg="white",
            height=1,
            state=tk.DISABLED,
            command=self.stop_recording,
            cursor="hand2",
            relief="raised",
            borderwidth=3
        )
        self.stop_button.grid(row=2, column=0, sticky="ew", pady=(0, 4))

        # 删除上次录制按钮
        self.delete_last_button = tk.Button(
            button_frame,
            text="删除上次录制",
            font=self.font_panel_text,
            bg="#f39c12",
            fg="white",
            height=1,
            command=self.delete_last_recording,
            cursor="hand2",
            relief="raised",
            borderwidth=3
        )
        self.delete_last_button.grid(row=3, column=0, sticky="ew", pady=(0, 4))

        # 重放按钮与选择序号
        self.replay_button = tk.Button(
            button_frame,
            text="重放",
            font=self.font_panel_section,
            bg="#2980b9",
            fg="white",
            height=1,
            command=self.start_replay,
            cursor="hand2",
            relief="raised",
            borderwidth=3
        )
        self.replay_button.grid(row=4, column=0, sticky="ew", pady=(0, 4))

        validate_cmd = (self.root.register(self._validate_replay_index), "%P")
        self.replay_spinbox = tk.Spinbox(
            button_frame,
            from_=1,
            to=999,
            width=6,
            textvariable=self.replay_index_var,
            font=self.font_panel_text,
            justify="center",
            command=self._on_replay_index_change,
            validate="key",
            validatecommand=validate_cmd
        )
        self.replay_spinbox.grid(row=4, column=1, sticky="e", pady=(0, 4))
        self._update_replay_controls()


    def init_cameras(self):
        """初始化所有相机"""
        started_pipelines = []
        try:
            camera_serials = self._resolve_camera_serial_numbers()
            if not camera_serials:
                raise RuntimeError("未检测到 RealSense 相机")
            if len(set(camera_serials)) != len(camera_serials):
                raise RuntimeError("相机序列号重复，请检查 TAKE_DATA_CAMERA_SERIALS")
            self.camera_serials = list(camera_serials)
            self.camera_last_frame_times = [0.0] * len(camera_serials)

            for sn in camera_serials:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(sn)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                started_pipelines.append(pipeline)
                pipeline.start(config)
                print(f"相机 {sn} 初始化成功")

            self.pipelines = started_pipelines
            self.camera_ready = True
            self.camera_frame_error = None
            self.camera_status_label.config(
                text=f"相机: 已连接 {len(camera_serials)} 台",
                fg="#27ae60"
            )
        except Exception as e:
            for pipeline in started_pipelines:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self.pipelines = []
            self.camera_ready = False
            self.camera_frame_error = str(e)
            self.camera_status_label.config(
                text=f"相机: 初始化失败 {str(e)}",
                fg="#e74c3c"
            )
            messagebox.showerror("错误", f"相机初始化失败: {str(e)}")

    def _check_camera_recording_ready(self):
        if self.disable_cameras:
            return
        with self.frame_lock:
            frame_times = list(self.camera_last_frame_times)
        expected = len(self.camera_serials)
        if not self.camera_ready or not expected or len(self.pipelines) != expected:
            raise RuntimeError("相机未全部初始化成功，禁止开始录制")
        if self.camera_frame_error:
            raise RuntimeError("相机采集异常，禁止开始录制: " + self.camera_frame_error)
        now = time.time()
        if len(frame_times) != expected or any(ts <= 0 or now - ts > 1.0 for ts in frame_times):
            raise RuntimeError("相机帧未就绪或已超过 1 秒未更新，禁止开始录制")

    def _resolve_camera_serial_numbers(self):
        env_serials = os.environ.get("TAKE_DATA_CAMERA_SERIALS", "").strip()
        if env_serials:
            serials = [sn.strip() for sn in env_serials.split(",") if sn.strip()]
            print(f"使用 TAKE_DATA_CAMERA_SERIALS 指定相机: {serials}")
            return serials

        connected = []
        try:
            context = rs.context()
            for device in context.query_devices():
                try:
                    connected.append(device.get_info(rs.camera_info.serial_number))
                except Exception:
                    pass
        except Exception as exc:
            print(f"检测 RealSense 相机失败: {exc}")

        if connected:
            configured = list(CAMERA_SERIAL_NUMBERS)
            if all(sn in connected for sn in configured):
                return configured
            missing = [sn for sn in configured if sn not in connected]
            if missing:
                print(f"配置中的相机未连接: {missing}")
            print(f"自动使用当前检测到的相机: {connected}")
            return connected

        return list(CAMERA_SERIAL_NUMBERS)

    def start_preview(self):
        """开始视频预览"""
        if not self.camera_ready or not self.pipelines:
            return
        def preview_loop():
            last_update_time = 0
            next_record_time = 0
            update_interval = 1.0 / self.preview_fps
            record_interval = 1.0 / self.record_fps

            while self.running:
                current_time = time.time()

                # 控制帧率
                if current_time - last_update_time < update_interval:
                    time.sleep(0.001)
                    continue

                images = []
                camera_frame_info = []
                valid = True

                try:
                    # 从所有相机获取帧
                    for pipeline in self.pipelines:
                        if not self.running:
                            break
                        frames = pipeline.wait_for_frames(timeout_ms=1000)
                        if not self.running:
                            break
                        color_frame = frames.get_color_frame()
                        if not color_frame:
                            valid = False
                            break
                        color_image = np.asanyarray(color_frame.get_data()).copy()
                        images.append(color_image)
                        info = {"serial": self.camera_serials[len(images) - 1], "host_timestamp": time.time()}
                        try:
                            info["device_timestamp_ms"] = color_frame.get_timestamp()
                            info["device_frame_number"] = color_frame.get_frame_number()
                            info["timestamp_domain"] = str(color_frame.get_frame_timestamp_domain())
                        except (AttributeError, RuntimeError):
                            pass
                        camera_frame_info.append(info)

                    if not self.running:
                        break
                    if not valid or len(images) != len(self.pipelines):
                        raise RuntimeError("未收到全部相机的彩色帧")
                    if images:
                        with self.frame_lock:
                            self.camera_last_frame_times = [info["host_timestamp"] for info in camera_frame_info]
                        if self.camera_frame_error:
                            self.camera_frame_error = None
                            self._set_label_text(self.camera_status_label, f"相机: 已连接 {len(images)} 台", "#27ae60")
                        # 如果正在录制，保存原始帧（无REC标识）并记录对应动作
                        if self.is_recording and not self._robot_stopping:
                            frame_timestamp = time.time()
                            if not self.recording_error and frame_timestamp >= next_record_time:
                                with self.record_lock:
                                    if self.is_recording and not self._robot_stopping:
                                        for i, color_image in enumerate(images):
                                            if i < len(self.frame_dirs):
                                                frame_path = os.path.join(
                                                    self.frame_dirs[i],
                                                    f"frame_{self.frame_index:06d}.jpg"
                                                )
                                                if not cv2.imwrite(frame_path, color_image):
                                                    raise RuntimeError(f"写入相机帧失败: {frame_path}")
                                        self._capture_action_sample(frame_timestamp, camera_frame_info)
                                        self.video_frame_metadata.append({
                                            "frame_index": self.frame_index,
                                            "frame_timestamp": frame_timestamp,
                                            "cameras": camera_frame_info,
                                        })
                                        self.frame_index += 1
                                        next_record_time = next_record_time + record_interval if next_record_time else frame_timestamp + record_interval
                                        if next_record_time <= frame_timestamp - record_interval:
                                            next_record_time = frame_timestamp + record_interval
                        else:
                            next_record_time = 0

                        # 为预览添加录制标识（不影响保存的视频）
                        if self.is_recording:
                            for color_image in images:
                                cv2.circle(color_image, (30, 30), 15, (0, 0, 255), -1)
                                cv2.putText(color_image, "REC", (55, 40),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if valid and images:
                        # 拼接所有已配置相机的画面。
                        combined = self._compose_preview_images(images)

                        # 转换为RGB用于显示
                        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

                        # 转换为PIL Image
                        pil_img = Image.fromarray(combined_rgb)

                        # 在主线程中更新GUI
                        self.root.after(0, self.update_preview_label, pil_img)

                        last_update_time = current_time

                except Exception as e:
                    if not self.running:
                        break
                    print(f"预览错误: {e}")
                    self.camera_frame_error = str(e)
                    self._set_label_text(self.camera_status_label, "相机: 采集异常 " + str(e), "#e74c3c")
                    self._report_recording_error(str(e))
                    time.sleep(0.1)

        self.preview_thread = threading.Thread(target=preview_loop, daemon=True)
        self.preview_thread.start()

    def _on_preview_resize(self, event):
        """缓存预览区域尺寸，避免线程中调用Tk方法"""
        if event.width > 0 and event.height > 0:
            self.preview_size = (event.width, event.height)

    def _compose_preview_images(self, images):
        """拼接预览画面并添加间隔"""
        if not images:
            return None
        if len(images) != 3:
            try:
                return cv2.hconcat(images)
            except Exception:
                return images[0]

        h, w = images[0].shape[:2]
        gap = max(8, int(w * 0.02))
        cache = getattr(self, "_preview_gap_cache", {})
        if (
            cache.get("h") != h
            or cache.get("gap") != gap
            or cache.get("dtype") != images[0].dtype
        ):
            # 使用与预览背景一致的颜色（RGB #34495e -> BGR 94,73,52）
            gap_color = (94, 73, 52)
            gap_img = np.full((h, gap, 3), gap_color, dtype=images[0].dtype)
            cache = {"h": h, "gap": gap, "dtype": images[0].dtype, "img": gap_img}
            self._preview_gap_cache = cache

        gap_img = cache["img"]
        try:
            return cv2.hconcat([gap_img, images[0], gap_img, images[1], gap_img, images[2], gap_img])
        except Exception:
            return cv2.hconcat(images)

    def update_preview_label(self, pil_img):
        """在主线程中更新预览标签"""
        try:
            target_w, target_h = self.preview_size
            if target_w <= 0 or target_h <= 0:
                return

            img_w, img_h = pil_img.size
            scale = min(target_w / img_w, target_h / img_h)
            new_w = max(int(img_w * scale), 1)
            new_h = max(int(img_h * scale), 1)
            if new_w != img_w or new_h != img_h:
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=pil_img)
            self.preview_label.imgtk = imgtk
            self.preview_label.configure(image=imgtk)
        except:
            pass

    def request_action_redraw(self):
        """请求动作可视化刷新"""
        return

    def _schedule_action_redraw(self):
        """按固定间隔刷新动作曲线，避免高频触发导致卡顿"""
        if self._action_redraw_pending:
            return
        self._action_redraw_pending = True
        try:
            self.root.after(self.action_redraw_interval_ms, self._action_redraw_loop)
        except Exception:
            self._action_redraw_pending = False

    def _action_redraw_loop(self):
        """动作曲线刷新循环"""
        self._action_redraw_pending = False
        self.update_action_canvas()
        if self.running:
            self._schedule_action_redraw()

    def _extract_gripper_width(self, joint_names, positions):
        """提取夹爪开合"""
        if not positions:
            return None
        try:
            if not joint_names:
                if len(positions) >= 2:
                    return float(positions[0]) + float(positions[1])
                return float(positions[0])
            lower_names = [name.lower() for name in joint_names]
            left_idx = next(
                (
                    i for i, name in enumerate(lower_names)
                    if "left" in name and any(token in name for token in ("finger", "gripper", "knuckle"))
                ),
                None
            )
            right_idx = next(
                (
                    i for i, name in enumerate(lower_names)
                    if "right" in name and any(token in name for token in ("finger", "gripper", "knuckle"))
                ),
                None
            )
            if left_idx is not None and right_idx is not None and left_idx != right_idx:
                return float(positions[left_idx]) + float(positions[right_idx])

            indices = [
                i for i, name in enumerate(lower_names)
                if any(token in name for token in ("finger", "gripper", "knuckle"))
            ]
            if not indices:
                return None
            if len(indices) >= 2:
                return float(positions[indices[0]]) + float(positions[indices[1]])
            return float(positions[indices[0]])
        except Exception:
            return None

    def _append_action_history(self, data):
        """追加动作历史数据"""
        # Plotting needs at most 30 Hz. Keep its allocation and canvas workload
        # independent of the incoming ROS joint/pose publication rates.
        now = time.monotonic()
        if now - self._last_history_append_time < 1.0 / 30.0:
            return
        self._last_history_append_time = now
        try:
            ts = data.get('timestamp') or time.time()
            franka = data.get('franka_joints') or {}
            joints = franka.get('position') or []

            # 从gripper_joints获取夹爪数据
            gripper_data = data.get('gripper_joints') or {}
            gripper_positions = gripper_data.get('position') or []
            gripper_names = gripper_data.get('names')
            gripper = self._extract_gripper_width(gripper_names, gripper_positions)
            target = data.get('gripper_target') or {}
            if gripper is None and target.get('valid') and target.get('target_width_m') is not None:
                gripper = float(target['target_width_m'])
            if gripper is None and data.get('gripper_command') is not None:
                try:
                    fallback_max = 0.08 if ROBOT_BACKEND == "local_ros2" and LOCAL_ROS2_GRIPPER_TYPE in ("franka", "franka_hand", "franka-hand", "hand", "builtin") else GRIPPER_MAX_WIDTH
                    max_width = float(target.get('max_width_m') or fallback_max)
                    gripper = float(data['gripper_command']) * max_width
                except Exception:
                    gripper = None

            ee = data.get('end_effector_pose') or {}
            pos = ee.get('position') or {}
            ori = ee.get('orientation') or {}

            sample = {
                "t": float(ts),
                "joints": list(joints[:7]) if joints else [],
                "pos": (
                    float(pos.get('x', 0.0)),
                    float(pos.get('y', 0.0)),
                    float(pos.get('z', 0.0))
                ),
                "ori": (
                    float(ori.get('x', 0.0)),
                    float(ori.get('y', 0.0)),
                    float(ori.get('z', 0.0)),
                    float(ori.get('w', 0.0))
                ),
                "gripper": None if gripper is None else float(gripper)
            }
        except Exception:
            return

        with self.action_history_lock:
            self.action_history.append(sample)
    def _draw_time_series_canvas(self, canvas, title, points, color, y_min, y_max, t0, t1, show_time=False, time_text=None):
        """绘制单条时间序列"""
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w <= 10 or h <= 10:
            return

        pad = 6
        canvas.create_text(
            pad, pad,
            anchor="nw",
            text=title,
            font=self.font_small,
            fill="#ecf0f1"
        )
        if show_time and time_text:
            canvas.create_text(
                w - pad, pad,
                anchor="ne",
                text=f"时间: {time_text}",
                font=self.font_small,
                fill="#bdc3c7"
            )

        title_h = 18
        left_pad = 40
        right_pad = 10
        top = pad + title_h
        bottom = h - pad
        left = pad + left_pad
        right = w - right_pad
        if bottom <= top or right <= left:
            return

        canvas.create_rectangle(
            left, top,
            right, bottom,
            outline="#3b5166",
            fill="#1f2d3a"
        )

        if y_max <= y_min:
            return

        if y_min < 0 < y_max:
            zero_y = top + (1.0 - ((0 - y_min) / (y_max - y_min))) * (bottom - top)
            canvas.create_line(left, zero_y, right, zero_y, fill="#7f8c8d")

        canvas.create_text(
            pad, top - 2,
            anchor="nw",
            text=f"{y_max:.2f}",
            font=self.font_tiny,
            fill="#bdc3c7"
        )
        canvas.create_text(
            pad, bottom - 12,
            anchor="nw",
            text=f"{y_min:.2f}",
            font=self.font_tiny,
            fill="#bdc3c7"
        )

        if not points:
            return

        window = max(t1 - t0, 1e-6)
        width = right - left
        height = bottom - top

        max_points = 240
        step = max(1, len(points) // max_points)
        coords = []
        for t, v in points[::step]:
            if t < t0 or t > t1:
                continue
            try:
                v = float(v)
            except Exception:
                continue
            x_pos = left + ((t - t0) / window) * width
            y_pos = top + (1.0 - ((v - y_min) / (y_max - y_min))) * height
            coords.extend([x_pos, y_pos])

        if len(coords) >= 4:
            canvas.create_line(coords, fill=color, width=1.5)

    def update_action_canvas(self):
        """更新动作可视化"""
        if not self.action_canvases:
            return
        if not self.action_canvases[0].winfo_exists():
            return

        with self.action_history_lock:
            if self.action_history:
                t1 = self.action_history[-1]["t"]
            else:
                t1 = time.time()
            t0 = t1 - self.history_window_sec
            while self.action_history and self.action_history[0]["t"] < t0:
                self.action_history.popleft()
            samples = list(self.action_history)

        if not samples:
            for canvas in self.action_canvases:
                canvas.delete("all")
                w = canvas.winfo_width()
                h = canvas.winfo_height()
                if w <= 10 or h <= 10:
                    continue
                canvas.create_text(
                    w / 2, h / 2,
                    text="等待动作数据...",
                    font=self.font_medium,
                    fill="#bdc3c7"
                )
            return

        time_text = time.strftime("%H:%M:%S", time.localtime(t1))
        series_map = {
            "pos_x": [],
            "pos_y": [],
            "pos_z": [],
            "ori_x": [],
            "ori_y": [],
            "ori_z": [],
            "ori_w": [],
            "grip": []
        }

        for sample in samples:
            t = sample.get("t")
            if t is None:
                continue
            pos = sample.get("pos") or (0.0, 0.0, 0.0)
            ori = sample.get("ori") or (0.0, 0.0, 0.0, 1.0)
            grip = sample.get("gripper")

            series_map["pos_x"].append((t, pos[0]))
            series_map["pos_y"].append((t, pos[1]))
            series_map["pos_z"].append((t, pos[2]))
            series_map["ori_x"].append((t, ori[0]))
            series_map["ori_y"].append((t, ori[1]))
            series_map["ori_z"].append((t, ori[2]))
            series_map["ori_w"].append((t, ori[3]))
            if grip is not None:
                series_map["grip"].append((t, grip))

        for idx, spec in enumerate(self.action_canvas_specs):
            canvas = self.action_canvases[idx]
            key = spec["key"]
            points = series_map.get(key, [])
            y_min, y_max = self._compute_series_range(
                points,
                spec["ymin"],
                spec["ymax"],
                spec["min_span"]
            )
            show_time = idx == 0
            self._draw_time_series_canvas(
                canvas,
                spec["title"],
                points,
                spec["color"],
                y_min,
                y_max,
                t0,
                t1,
                show_time=show_time,
                time_text=time_text
            )

    def get_next_record_index(self):
        """获取下一个录制序号"""
        last_index = self.get_last_record_index()
        return (last_index or 0) + 1

    def get_last_record_index(self):
        """获取最后一次录制序号"""
        max_index = 0

        # 新结构: /data/record_XXX/
        if os.path.exists(self.record_base_path):
            for name in os.listdir(self.record_base_path):
                if not name.startswith("record_"):
                    continue
                path = os.path.join(self.record_base_path, name)
                if not os.path.isdir(path):
                    continue
                try:
                    idx = int(name.split("_")[1])
                    max_index = max(max_index, idx)
                except Exception:
                    pass

        # 兼容旧结构: /data/video/record_XXX 和 /data/action/record_XXX
        for legacy_base in (self.legacy_video_base_path, self.legacy_action_base_path):
            if os.path.exists(legacy_base):
                for name in os.listdir(legacy_base):
                    if name.startswith("record_"):
                        try:
                            idx = int(name.split("_")[1])
                            max_index = max(max_index, idx)
                        except Exception:
                            pass

        if max_index <= 0:
            return None
        return max_index

    def delete_last_recording(self):
        """删除上一次录制数据"""
        if self.is_recording:
            messagebox.showwarning("提示", "录制中无法删除上次数据")
            return
        if self.replay_in_progress:
            messagebox.showwarning("提示", "回放中无法删除上次数据")
            return

        last_index = self.get_last_record_index()
        if not last_index:
            messagebox.showinfo("提示", "没有找到可删除的录制数据")
            return

        record_name = f"record_{last_index:03d}"
        record_dir = os.path.join(self.record_base_path, record_name)
        video_dir = os.path.join(record_dir, "video")
        action_dir = os.path.join(record_dir, "action")
        legacy_video_dir = os.path.join(self.legacy_video_base_path, record_name)
        legacy_action_dir = os.path.join(self.legacy_action_base_path, record_name)

        confirm = messagebox.askyesno(
            "确认删除",
            f"确定删除上一次录制数据？\n{record_name}"
        )
        if not confirm:
            return

        errors = []
        deleted_any = False

        # 删除原始数据（视频 + 动作）
        paths_to_delete = []
        if os.path.exists(record_dir):
            paths_to_delete.append(record_dir)
        else:
            if os.path.exists(legacy_video_dir):
                paths_to_delete.append(legacy_video_dir)
            if os.path.exists(legacy_action_dir):
                paths_to_delete.append(legacy_action_dir)

        for path in paths_to_delete:
            try:
                shutil.rmtree(path)
                deleted_any = True
            except Exception as e:
                errors.append(f"{path}: {e}")

        if errors:
            messagebox.showerror("错误", "删除失败:\n" + "\n".join(errors))
            return

        if not deleted_any:
            messagebox.showinfo("提示", "未找到需要删除的数据")
            return

        new_last = self.get_last_record_index()
        if new_last:
            new_record = f"record_{new_last:03d}"
            self.session_label.config(text=f"录制序号: {new_record}")
            record_dir = os.path.join(self.record_base_path, new_record)
            if os.path.isdir(record_dir):
                video_path = os.path.join(record_dir, "video")
                action_path = os.path.join(record_dir, "action")
            else:
                video_path = os.path.join(self.legacy_video_base_path, new_record)
                action_path = os.path.join(self.legacy_action_base_path, new_record)
            self.set_current_paths(video_path, action_path)
            self.session_index = new_last
        else:
            self.session_label.config(text="录制序号: --")
            self.set_current_paths(self.record_base_path, self.record_base_path)
            self.session_index = None

        if new_last:
            self.replay_index_var.set(new_last)
        self._update_replay_controls()

        messagebox.showinfo("完成", f"已删除 {record_name} 的数据")

    def _get_remote_script_for_mode(self, *args, **kwargs):
        raise RuntimeError("此仓库仅支持本机 ROS2 容器采集；旧远程/回放入口已禁用。")


    def _ssh_base_cmd(self, *args, **kwargs):
        raise RuntimeError("此仓库仅支持本机 ROS2 容器采集；旧远程/回放入口已禁用。")


    def _scp_base_cmd(self, *args, **kwargs):
        raise RuntimeError("此仓库仅支持本机 ROS2 容器采集；旧远程/回放入口已禁用。")


    def _run_remote_command(self, *args, **kwargs):
        raise RuntimeError("此仓库仅支持本机 ROS2 容器采集；旧远程/回放入口已禁用。")


    def _is_local_ros2_backend(self):
        return ROBOT_BACKEND == "local_ros2"

    def _local_ros2_host_setup_exists(self):
        return os.path.isfile(f"/opt/ros/{LOCAL_ROS2_DISTRO}/setup.bash")

    def _local_ros2_use_container(self):
        if not self._is_local_ros2_backend():
            return False
        if LOCAL_ROS2_EXEC_MODE in ("container", "docker", "devcontainer"):
            return True
        if LOCAL_ROS2_EXEC_MODE in ("host", "local", "native"):
            return False
        return not self._local_ros2_host_setup_exists()

    def _local_ros2_compose_cmd(self):
        if not os.path.isfile(LOCAL_ROS2_DOCKER_COMPOSE):
            raise FileNotFoundError(f"缺少 ROS2 Dev Container compose 文件: {LOCAL_ROS2_DOCKER_COMPOSE}")
        if not shutil.which("docker"):
            raise RuntimeError("找不到 docker 命令，无法启动 ROS2 Humble 容器。")
        if subprocess.run(
            ["docker", "compose", "version"],
            env=self._local_ros2_docker_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0:
            return ["docker", "compose", "-f", LOCAL_ROS2_DOCKER_COMPOSE]
        if shutil.which("docker-compose"):
            return ["docker-compose", "-f", LOCAL_ROS2_DOCKER_COMPOSE]
        raise RuntimeError("找不到 docker compose 或 docker-compose，无法启动 ROS2 Humble 容器。")

    def _local_ros2_container_id(self, start=False):
        container = os.environ.get("PI05_CONTAINER", "pi05-fr3-gello-ros2-1")
        docker_env = self._local_ros2_docker_env()
        def running():
            result = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container], env=docker_env,
                                    capture_output=True, text=True, check=False, timeout=10)
            return result.returncode == 0 and result.stdout.strip() == "true"
        if running():
            return container
        if not start:
            return None
        if not LOCAL_ROS2_CONTAINER_AUTOSTART:
            raise RuntimeError("ROS2 容器未运行，请先启动本仓库的 ros2 服务。")
        result = subprocess.run(self._local_ros2_compose_cmd() + ["up", "-d", "--no-build", LOCAL_ROS2_DOCKER_SERVICE],
                                cwd=os.path.dirname(LOCAL_ROS2_DOCKER_COMPOSE), env=docker_env,
                                capture_output=True, text=True, check=False, timeout=60)
        if result.returncode != 0 or not running():
            raise RuntimeError("ROS2 容器不可用；请先完成本仓库 Docker/ROS 构建。" + (result.stderr or result.stdout))
        return container


    def _local_ros2_prepare_container_scripts(self):
        container_id = self._local_ros2_container_id(start=True)
        self.local_ros2_container_id = container_id
        serial_env = self._local_ros2_docker_env()
        if LOCAL_GELLO_PORT:
            serial_env["GELLO_PORT"] = LOCAL_GELLO_PORT
        subprocess.run(["bash", os.path.join(CAPTURE_DIR, "refresh_gello_serial.sh"), "--container", container_id],
                       env=serial_env, check=True, timeout=20)
        copies = (
            (LOCAL_ROS2_CONTROL_SCRIPT, LOCAL_ROS2_CONTAINER_CONTROL_SCRIPT),
            (LOCAL_BRIDGE_SCRIPT, LOCAL_ROS2_CONTAINER_BRIDGE_SCRIPT),
        )
        for source, target in copies:
            if not os.path.isfile(source):
                raise FileNotFoundError(f"缺少脚本: {source}")
            subprocess.run(
                ["docker", "cp", source, f"{container_id}:{target}"],
                env=self._local_ros2_docker_env(),
                check=True,
                timeout=20,
            )
        subprocess.run(
            [
                "docker",
                "exec",
                container_id,
                "chmod",
                "+x",
                LOCAL_ROS2_CONTAINER_CONTROL_SCRIPT,
                LOCAL_ROS2_CONTAINER_BRIDGE_SCRIPT,
            ],
            env=self._local_ros2_docker_env(),
            check=False,
            timeout=20,
        )
        return container_id

    def _local_ros2_docker_env(self, base_env=None):
        env = dict(os.environ if base_env is None else base_env)
        if LOCAL_ROS2_DOCKER_HOST and LOCAL_ROS2_DOCKER_HOST.lower() != "inherit":
            env["DOCKER_HOST"] = LOCAL_ROS2_DOCKER_HOST
        return env

    def _local_ros2_container_source_prefix(self):
        return (
            f"source /opt/ros/{LOCAL_ROS2_DISTRO}/setup.bash && "
            f"source /opt/ros/{LOCAL_ROS2_DISTRO}/franka/setup.bash && "
            f"source {shlex.quote(LOCAL_ROS2_CONTAINER_WS)}/install/setup.bash"
        )

    def _local_ros2_container_env(self, env):
        merged = dict(env)
        merged.update({
            "ROS_DISTRO": LOCAL_ROS2_DISTRO,
            "LOCAL_ROS2_WS": LOCAL_ROS2_CONTAINER_WS,
            "FRANKA_WS": LOCAL_ROS2_CONTAINER_WS,
            "GELLO_WS": LOCAL_ROS2_CONTAINER_WS,
            "LOCAL_ROS2_SETUP_CMD": self._local_ros2_container_source_prefix(),
            "FR3_HARDWARE_FAULT_LOCK": os.environ.get("FR3_CONTAINER_HARDWARE_FAULT_LOCK", "/workspace/repo/runtime/gripper_hardware_fault.lock"),
        })
        return merged

    def _local_ros2_docker_exec_base(self, env=None, workdir=None, start=True):
        container_id = self._local_ros2_container_id(start=start)
        if not container_id:
            return None
        self.local_ros2_container_id = container_id
        cmd = ["docker", "exec"]
        if workdir:
            cmd.extend(["-w", workdir])
        env = env or {}
        pass_keys = (
            "PYTHONUNBUFFERED",
            "PI05_CAPTURE_OWNER",
            "PI05_CAPTURE_PORT",
            "FR3_HARDWARE_FAULT_LOCK",
            "FRANKA_IP",
            "ROS_DISTRO",
            "LOCAL_ROS2_WS",
            "FRANKA_WS",
            "GELLO_WS",
            "GELLO_CFG",
            "FR3_CFG",
            "GRIPPER_CFG",
            "LOCAL_ROS2_USE_GRIPPER",
            "LOCAL_ROS2_GRIPPER_TYPE",
            "LOCAL_ROS2_SETUP_CMD",
            "LOCAL_ROS2_READY_FILE",
            "GELLO_PORT",
            "LOCAL_ROS2_GELLO_TOPIC_TIMEOUT",
            "LOCAL_ROS2_CONTROLLER_TIMEOUT",
            "LOCAL_ROS2_SESSION_ID",
            "ENABLE_RT",
            "PIN_CPU",
            "RT_PRIORITY",
            "ROS2_CONTROL_CPU",
        )
        for key in pass_keys:
            if key in env and env[key] is not None:
                cmd.extend(["-e", f"{key}={env[key]}"])
        cmd.append(container_id)
        return cmd

    def _local_ros2_source_prefix(self):
        if self._local_ros2_use_container():
            return self._local_ros2_container_source_prefix()
        if LOCAL_ROS2_SETUP_CMD:
            return LOCAL_ROS2_SETUP_CMD

        setup_files = []
        ros_setup = f"/opt/ros/{LOCAL_ROS2_DISTRO}/setup.bash"
        franka_overlay = f"/opt/ros/{LOCAL_ROS2_DISTRO}/franka/setup.bash"
        workspace_setup = os.path.join(LOCAL_ROS2_WS, "install", "setup.bash")
        workspace_local_setup = os.path.join(LOCAL_ROS2_WS, "install", "local_setup.bash")

        for path in (ros_setup, franka_overlay, workspace_setup):
            if os.path.isfile(path):
                setup_files.append(path)
        if not setup_files and os.path.isfile(workspace_local_setup):
            setup_files.append(workspace_local_setup)
        if not setup_files:
            return "true"
        return " && ".join(f"source {shlex.quote(path)}" for path in setup_files)

    def _run_local_ros2_command(self, command, timeout=10, check=False):
        full_cmd = f"{self._local_ros2_source_prefix()} && {command}"
        if self._local_ros2_use_container():
            exec_base = self._local_ros2_docker_exec_base(
                workdir=LOCAL_ROS2_CONTAINER_WS,
                start=True,
            )
            return subprocess.run(
                exec_base + ["bash", "-lc", full_cmd],
                env=self._local_ros2_docker_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=check,
            )
        return subprocess.run(
            ["bash", "-lc", full_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )

    def _run_robot_command(self, command, timeout=10, check=False):
        if self._is_local_ros2_backend():
            return self._run_local_ros2_command(command, timeout=timeout, check=check)
        return self._run_remote_command(command, timeout=timeout, check=check)

    def _local_ros2_ready_file_exists(self, use_container):
        if use_container:
            exec_base = self._local_ros2_docker_exec_base(start=False)
            if not exec_base:
                return False
            result = subprocess.run(
                exec_base + ["test", "-f", LOCAL_ROS2_READY_FILE],
                env=self._local_ros2_docker_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
            return result.returncode == 0
        return os.path.isfile(LOCAL_ROS2_READY_FILE)

    def _cleanup_remote_for_start(self, *args, **kwargs):
        raise RuntimeError("此仓库仅支持本机 ROS2 容器采集；旧远程/回放入口已禁用。")


    def _parse_number_list(self, text):
        if not text:
            return None
        values = []
        for token in re.split(r"[\s,]+", text.strip()):
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values or None

    def _get_remote_robot_zero(self):
        # relative_start zeroes are captured by the controller at each activation.
        return None


    def _extract_franka_positions(self, data):
        if not isinstance(data, dict):
            return None
        franka = data.get("franka_joints")
        if not isinstance(franka, dict):
            return None
        positions = franka.get("position")
        if not positions:
            return None
        names = franka.get("names") or []
        order = self._get_franka_order(names)
        if order and len(positions) >= len(order):
            return [positions[i] for i in order]
        return list(positions[:7])

    def _wait_for_robot_zero(self, robot_zero, timeout_sec):
        if not robot_zero or len(robot_zero) < 7:
            return False
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            with self.action_sample_lock:
                latest = copy.deepcopy(self.latest_action_data)
            positions = self._extract_franka_positions(latest)
            if positions:
                diffs = [abs(p - z) for p, z in zip(positions, robot_zero)]
                max_diff = max(diffs) if diffs else None
                if max_diff is not None and max_diff <= self.relative_zero_tolerance:
                    return True
                if max_diff is not None:
                    self._set_label_text(
                        self.robot_ready_label,
                        f"机器人状态: 等待回零 (偏差 {max_diff:.3f}rad)",
                        "#f39c12",
                    )
            time.sleep(0.1)
        return False

    def _set_relative_control_enabled(self, enabled, retries=5, wait_sec=1.0, raise_on_failure=False):
        raise RuntimeError("控制器使用 relative_start；不支持旧相对发布器服务。")


    def _cleanup_local_gello_for_start(self, *args, **kwargs):
        raise RuntimeError("此仓库仅支持本机 ROS2 容器采集；旧远程/回放入口已禁用。")


    def _terminate_process_group(self, process, timeout=5):
        if not process or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            try:
                process.wait(timeout=2)
            except Exception:
                pass

    def _acquire_fr3_control_lease(self):
        if getattr(self, "_fr3_control_lease", None) is not None:
            return
        from fr3_control_ownership import ControlLease
        lease = ControlLease.from_environment("gello")
        self._fr3_control_lease = lease


    def _fr3_cleanup_allowed(self):
        try:
            # Hold the lock through cleanup, avoiding a check-then-clean race
            # with a deployment session starting on another process.
            self._acquire_fr3_control_lease()
            return True
        except Exception as exc:
            print(f"其他会话正在控制 FR3 或检查不可用，跳过采集程序清理: {exc}")
            return False

    def _cleanup_local_ros2_for_start(self):
        # A fresh GUI has no session to stop. Never sweep unrelated processes.
        if not self.local_ros2_session_id:
            if os.path.isfile(CAPTURE_SESSION_FILE):
                raise RuntimeError("存在先前采集会话状态，请先运行 capture/stop_capture.sh 核验停止；不会自动清理其他会话。")
            return
        if not self._fr3_cleanup_allowed():
            raise RuntimeError("控制权已被其他会话占用，拒绝清理")
        self._stop_local_ros2_container()


    def _stop_local_ros2_container(self):
        """先确认容器任务退出，再释放主机上的 docker exec 句柄。"""
        stop_cmd = ["bash", LOCAL_ROS2_STOP_SCRIPT]
        if self.local_ros2_container_id:
            stop_cmd.extend(["--container", self.local_ros2_container_id])
        if self.local_ros2_session_id:
            stop_cmd.extend(["--session", self.local_ros2_session_id])
        try:
            result = subprocess.run(
                stop_cmd,
                env=self._local_ros2_docker_env(),
                capture_output=True,
                text=True,
                check=False,
                timeout=18,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("停止容器控制进程超时，未确认停机。") from exc
        if result.returncode != 0:
            detail = "\n".join(part.strip() for part in (result.stderr, result.stdout) if part and part.strip())
            raise RuntimeError("停止脚本未确认控制进程全部退出: " + (detail[-2000:] or f"exit {result.returncode}"))
        print(result.stdout.strip() or "容器内控制进程已全部退出")

        if self.action_socket:
            try:
                self.action_socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.action_socket.close()
            except Exception:
                pass
            self.action_socket = None
        self.robot_ready = False
        self._terminate_process_group(self.local_ros2_bridge_process, timeout=0.2)
        self._terminate_process_group(self.local_ros2_control_process, timeout=0.2)
        self.local_ros2_bridge_process = None
        self.local_ros2_control_process = None
        for log_file in (self.local_ros2_bridge_log_file, self.local_ros2_log_file):
            if log_file:
                log_file.close()
        self.local_ros2_bridge_log_file = None
        self.local_ros2_log_file = None

    def _check_local_ros2_requirements(self):
        fault_lock = os.environ.get("FR3_HARDWARE_FAULT_LOCK", os.path.join(REPO_ROOT, "runtime", "gripper_hardware_fault.lock"))
        if os.path.isfile(fault_lock):
            raise RuntimeError("Hardware inspection lock present: " + fault_lock)
        checks = [
            (LOCAL_ROS2_WS, f"缺少本机 ROS2 工作区: {LOCAL_ROS2_WS}"),
            (os.path.join(LOCAL_ROS2_WS, "install"), f"缺少 ROS2 install 目录: {LOCAL_ROS2_WS}/install"),
            (LOCAL_ROS2_CONTROL_SCRIPT, f"缺少本机 ROS2 启动脚本: {LOCAL_ROS2_CONTROL_SCRIPT}"),
            (LOCAL_BRIDGE_SCRIPT, f"缺少本机数据桥接脚本: {LOCAL_BRIDGE_SCRIPT}"),
        ]
        for path, message in checks:
            if not os.path.exists(path):
                raise FileNotFoundError(message)

        ros_check_cmd = (
            "command -v ros2 && "
            "/usr/bin/python3 -c 'import rclpy' && "
            "ros2 pkg prefix franka_gello_state_publisher && "
            "ros2 pkg prefix franka_fr3_arm_controllers"
        )
        if LOCAL_ROS2_USE_GRIPPER != "0":
            ros_check_cmd += " && ros2 pkg prefix franka_gripper_manager"

        result = self._run_local_ros2_command(
            ros_check_cmd,
            timeout=10,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "import rclpy failed"
            raise RuntimeError(
                "本机 ROS2 环境不可用，无法启动 GELLO/FR3 控制。"
                "这个工作区需要 ROS2 Humble、franka_ros2 和 rclpy。"
                "如果之前是在 VS Code Dev Container 里调通，请在同一个 ROS2 环境中启动 GUI，"
                "或设置 LOCAL_ROS2_SETUP_CMD 为完整 source 命令。"
                f"详情: {detail}"
            )

    def _start_local_ros2_robot(self, control_mode=None):
        try:
            self._acquire_fr3_control_lease()
        except Exception as exc:
            self._set_label_text(self.robot_ready_label, f"机器人状态: 控制权被占用: {exc}", "#e74c3c")
            print(f"无法启动遥操作: {exc}")
            return False
        try:
            print("启动本机 ROS2 GELLO/FR3 控制...")
            self._set_label_text(self.robot_connection_label, "机器人连接: 启动本机ROS2...", "#f39c12")
            self._set_label_text(self.robot_ready_label, "机器人状态: 检查ROS2环境...", "#f39c12")
            self._cleanup_local_ros2_for_start()
            time.sleep(0.5)
            use_container = self._local_ros2_use_container()
            if use_container:
                self._set_label_text(self.robot_ready_label, "机器人状态: 准备ROS2容器...", "#f39c12")
                self._local_ros2_prepare_container_scripts()
            self._check_local_ros2_requirements()
            with self._robot_start_lock:
                if self._robot_stop_requested.is_set():
                    raise RuntimeError("已取消启动遥操作")
                self.local_ros2_session_id = uuid.uuid4().hex
                os.makedirs(os.path.dirname(CAPTURE_SESSION_FILE), exist_ok=True)
                session_state = {"session_id": self.local_ros2_session_id,
                                 "container": os.environ.get("PI05_CONTAINER", "pi05-fr3-gello-ros2-1"),
                                 "owner": "pi05-fr3-gello"}
                with open(CAPTURE_SESSION_FILE + ".tmp", "w", encoding="utf-8") as state_file:
                    json.dump(session_state, state_file)
                os.replace(CAPTURE_SESSION_FILE + ".tmp", CAPTURE_SESSION_FILE)

            self.local_ros2_log_file = open(LOCAL_ROS2_LOG, "wb", buffering=0)
            self.local_ros2_bridge_log_file = open(LOCAL_ROS2_BRIDGE_LOG, "wb", buffering=0)

            env = os.environ.copy()
            env.update({
                "PYTHONUNBUFFERED": "1",
                "PI05_CAPTURE_OWNER": "pi05-fr3-gello",
                "PI05_CAPTURE_PORT": str(REMOTE_PORT),
                "FRANKA_IP": REMOTE_ROBOT_IP,
                "ROS_DISTRO": LOCAL_ROS2_DISTRO,
                "LOCAL_ROS2_WS": LOCAL_ROS2_WS,
                "FRANKA_WS": LOCAL_ROS2_FRANKA_WS,
                "GELLO_WS": LOCAL_ROS2_GELLO_WS,
                "GELLO_CFG": LOCAL_ROS2_GELLO_CFG,
                "FR3_CFG": LOCAL_ROS2_FR3_CFG,
                "GRIPPER_CFG": LOCAL_ROS2_GRIPPER_CFG,
                "LOCAL_ROS2_USE_GRIPPER": LOCAL_ROS2_USE_GRIPPER,
                "LOCAL_ROS2_GRIPPER_TYPE": LOCAL_ROS2_GRIPPER_TYPE,
                "LOCAL_ROS2_READY_FILE": LOCAL_ROS2_READY_FILE,
                "LOCAL_ROS2_SESSION_ID": self.local_ros2_session_id,
                "RELATIVE_CFG": LOCAL_ROS2_RELATIVE_CONFIG,
                "RELATIVE_NODE": LOCAL_ROS2_RELATIVE_NODE,
                "RELATIVE_AUTO_ENABLE": "0",
            })
            if LOCAL_ROS2_SETUP_CMD:
                env["LOCAL_ROS2_SETUP_CMD"] = LOCAL_ROS2_SETUP_CMD
            if LOCAL_GELLO_PORT:
                env["GELLO_PORT"] = LOCAL_GELLO_PORT
            if use_container:
                env = self._local_ros2_container_env(env)

            self._set_label_text(self.robot_ready_label, "机器人状态: 启动FR3 ROS2控制...", "#f39c12")
            if use_container:
                control_cmd = (
                    self._local_ros2_docker_exec_base(
                        env=env,
                        workdir=LOCAL_ROS2_CONTAINER_WS,
                        start=True,
                    )
                    + ["bash", LOCAL_ROS2_CONTAINER_CONTROL_SCRIPT]
                )
                control_cwd = os.path.dirname(LOCAL_ROS2_DOCKER_COMPOSE)
            else:
                control_cmd = ["bash", LOCAL_ROS2_CONTROL_SCRIPT]
                control_cwd = os.path.dirname(LOCAL_ROS2_CONTROL_SCRIPT)
            with self._robot_start_lock:
                if self._robot_stop_requested.is_set():
                    raise RuntimeError("已取消启动遥操作")
                self.local_ros2_control_process = subprocess.Popen(
                    control_cmd,
                    cwd=control_cwd,
                    stdout=self.local_ros2_log_file,
                    stderr=subprocess.STDOUT,
                    env=env if not use_container else self._local_ros2_docker_env(),
                    start_new_session=True,
                )

            bridge_script = (
                LOCAL_ROS2_CONTAINER_BRIDGE_SCRIPT if use_container else LOCAL_BRIDGE_SCRIPT
            )
            bridge_cmd = f"{self._local_ros2_source_prefix()} && exec /usr/bin/python3 {shlex.quote(bridge_script)}"
            self._set_label_text(self.robot_ready_label, "机器人状态: 启动本机数据桥接...", "#f39c12")
            if use_container:
                bridge_process_cmd = (
                    self._local_ros2_docker_exec_base(
                        env=env,
                        workdir=LOCAL_ROS2_CONTAINER_WS,
                        start=True,
                    )
                    + ["bash", LOCAL_ROS2_CONTAINER_CONTROL_SCRIPT, "--data-bridge", bridge_cmd]
                )
                bridge_cwd = os.path.dirname(LOCAL_ROS2_DOCKER_COMPOSE)
            else:
                bridge_process_cmd = ["bash", "-lc", bridge_cmd]
                bridge_cwd = os.path.dirname(LOCAL_BRIDGE_SCRIPT)
            with self._robot_start_lock:
                if self._robot_stop_requested.is_set():
                    raise RuntimeError("已取消启动遥操作")
                self.local_ros2_bridge_process = subprocess.Popen(
                    bridge_process_cmd,
                    cwd=bridge_cwd,
                    stdout=self.local_ros2_bridge_log_file,
                    stderr=subprocess.STDOUT,
                    env=env if not use_container else self._local_ros2_docker_env(),
                    start_new_session=True,
                )

            self.action_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.action_socket.settimeout(2)

            connected = False
            last_error = None
            for attempt in range(90):
                if self._robot_stop_requested.is_set():
                    raise RuntimeError("已取消启动遥操作")
                if self.local_ros2_control_process.poll() is not None:
                    raise RuntimeError(f"本机ROS2控制已退出，请查看 {LOCAL_ROS2_LOG}")
                if self.local_ros2_bridge_process.poll() is not None:
                    raise RuntimeError(f"本机ROS2数据桥接已退出，请查看 {LOCAL_ROS2_BRIDGE_LOG}")
                try:
                    self.action_socket.connect((LOCAL_ACTION_HOST, REMOTE_PORT))
                    connected = True
                    break
                except (socket.timeout, ConnectionRefusedError, OSError) as exc:
                    last_error = exc
                    self._set_label_text(
                        self.robot_ready_label,
                        f"机器人状态: 等待本机ROS2桥接... ({attempt + 1}/90)",
                        "#f39c12",
                    )
                    time.sleep(0.5)
                    try:
                        self.action_socket.close()
                    except Exception:
                        pass
                    self.action_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.action_socket.settimeout(2)

            if not connected:
                raise RuntimeError(f"无法连接本机ROS2数据桥接: {last_error}")

            ready_deadline = time.monotonic() + float(os.environ.get("LOCAL_ROS2_READY_TIMEOUT", "45"))
            while time.monotonic() < ready_deadline:
                if self._robot_stop_requested.is_set():
                    raise RuntimeError("已取消启动遥操作")
                if self.local_ros2_control_process.poll() is not None:
                    raise RuntimeError(f"本机ROS2控制已退出，请查看 {LOCAL_ROS2_LOG}")
                if self._local_ros2_ready_file_exists(use_container):
                    break
                self._set_label_text(
                    self.robot_ready_label,
                    "机器人状态: 等待ROS2控制就绪...",
                    "#f39c12",
                )
                time.sleep(0.5)
            else:
                raise RuntimeError(f"本机ROS2控制启动超时，请查看 {LOCAL_ROS2_LOG}")

            if self._robot_stop_requested.is_set():
                raise RuntimeError("已取消启动遥操作")
            self.action_socket.settimeout(None)
            self.active_control_mode = "absolute"
            self.control_mode_var.set("absolute")
            self._set_label_text(self.robot_connection_label, "机器人连接: 本机ROS2已连接", "#27ae60")
            self._set_label_text(self.robot_ready_label, "机器人状态: 等待数据...", "#f39c12")
            print("本机 ROS2 GELLO/FR3 控制启动成功")
            return True

        except Exception as e:
            if self.action_socket:
                try:
                    self.action_socket.close()
                except Exception:
                    pass
                self.action_socket = None
            if self._robot_stop_requested.is_set():
                print(f"本机 ROS2 启动已取消: {e}")
                return False
            try:
                self._cleanup_local_ros2_for_start()
            except Exception as cleanup_error:
                e = RuntimeError(f"{e}; 清理失败，未确认停止: {cleanup_error}")
                self._robot_stop_error = str(e)
            self._set_label_text(self.robot_connection_label, "机器人连接: 连接失败", "#e74c3c")
            self._set_label_text(self.robot_ready_label, f"机器人状态: {str(e)}", "#e74c3c")
            self.robot_ready = False
            print(f"本机 ROS2 启动失败: {e}")
            return False

    def _start_local_gello_robot(self, *args, **kwargs):
        raise RuntimeError("此仓库仅支持本机 ROS2 容器采集；旧远程/回放入口已禁用。")


    def start_remote_robot(self, control_mode=None):
        return self._start_local_ros2_robot(control_mode)


    def receive_action_data(self):
        """接收动作数据"""
        sock = self.action_socket
        if not sock:
            return
        buffer = ''
        while self.running and sock:
            try:
                chunk = sock.recv(65536).decode('utf-8')
                if not chunk:
                    break

                buffer += chunk

                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            recv_ts = time.time()
                            data = json.loads(line)
                            with self.action_sample_lock:
                                self.latest_action_data = data
                                remote_ts = data.get("timestamp")
                                if isinstance(remote_ts, (int, float)):
                                    offset = recv_ts - remote_ts
                                    self.action_time_offsets.append(offset)
                                    self.action_time_offset = min(self.action_time_offsets)
                                self.action_sample_buffer.append({
                                    "remote_ts": remote_ts if isinstance(remote_ts, (int, float)) else None,
                                    "local_recv_ts": recv_ts,
                                    "data": data,
                                })
                                cutoff = recv_ts - self.action_sample_window_sec
                                while (
                                    self.action_sample_buffer
                                    and self.action_sample_buffer[0]["local_recv_ts"] < cutoff
                                ):
                                    self.action_sample_buffer.popleft()
                            self._append_action_history(data)
                            self.request_action_redraw()
                            if self.disable_cameras and self.is_recording and not self._robot_stopping:
                                try:
                                    with self.record_lock:
                                        if self.is_recording and not self._robot_stopping and recv_ts - self.last_action_record_time >= 1.0 / self.record_fps:
                                            self._capture_action_sample(recv_ts)
                                            self.last_action_record_time = recv_ts
                                except Exception as exc:
                                    self._report_recording_error(str(exc))
                            if not self.robot_ready:
                                self.robot_ready = True
                                self._set_label_text(
                                    self.robot_ready_label,
                                    "机器人状态: 就绪",
                                    "#27ae60"
                                )
                        except json.JSONDecodeError:
                            pass

            except Exception as e:
                print(f"接收数据错误: {e}")
                break
        try:
            sock.close()
        except Exception:
            pass
        if self.action_socket is sock:
            self.action_socket = None
            self.robot_ready = False
            self._set_label_text(self.robot_connection_label, "机器人连接: 未连接", "#95a5a6")
            self._set_label_text(self.robot_ready_label, "机器人状态: 待机中", "#95a5a6")
            self._update_teleop_button()
            if not self._robot_stopping:
                self._report_recording_error("机器人数据连接断开，录制已中止")

    def _get_aligned_action_time(self, sample):
        """Estimate local time for an action sample."""
        remote_ts = sample.get("remote_ts")
        if remote_ts is None or self.action_time_offset is None:
            return sample["local_recv_ts"]
        return remote_ts + self.action_time_offset

    def _get_task_text(self):
        return (self.task_var.get() if hasattr(self, "task_var") else self.task_text).strip()

    def _get_outcome(self):
        outcome = self.outcome_var.get() if hasattr(self, "outcome_var") else self.outcome
        return outcome if outcome in ("unknown", "success", "failure", "aborted") else "unknown"

    def _report_recording_error(self, message):
        if self.is_recording and not self.recording_error:
            self.recording_error = message
            if self.record_metadata and getattr(self, "_last_capture_diagnostics", None):
                self.record_metadata["capture_failure_diagnostics"] = copy.deepcopy(self._last_capture_diagnostics)
            self.root.after(0, self.stop_recording)

    @staticmethod
    def _capture_vector(value, length, field):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (length,) or not np.isfinite(array).all():
            raise RuntimeError(f"采集缺少有效 {field} ({length} 维)")
        return array.tolist()

    @staticmethod
    def _capture_gripper_result(target):
        """Encode optional raw Move results without conflating missing and false."""
        success = target.get("result_success")
        status = target.get("result_status")
        sequence = target.get("result_sequence")
        error = target.get("result_error")
        if success is not None and not isinstance(success, bool):
            raise RuntimeError("夹爪 result_success 必须为布尔值或空值")
        for field, value, minimum in (("result_status", status, 0), ("result_sequence", sequence, 1)):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < minimum):
                raise RuntimeError("夹爪 " + field + " 必须为有效整数或空值")
        if error is not None and not isinstance(error, str):
            raise RuntimeError("夹爪 result_error 必须为字符串或空值")
        return (-1 if success is None else int(success), -1 if status is None else status,
                -1 if sequence is None else sequence, "" if error is None else error)

    @classmethod
    def _capture_gripper_execution(cls, target):
        """Keep accepted action semantics separate from versioned execution results."""
        cls._capture_gripper_result(target)
        try:
            return _gripper_execution_kind(target)
        except ValueError as exc:
            raise RuntimeError("夹爪最近接受的目标无效或状态错误: " + str(target.get("status")) + "；" + str(exc)) from exc

    @classmethod
    def _capture_gripper_execution_diagnostics(cls, rows):
        events = {}
        for index, row in enumerate(rows):
            goal = row["gripper_target"]
            kind = cls._capture_gripper_execution(goal)
            if not kind:
                continue
            sequence = goal["accepted_sequence"]
            key = (kind, sequence)
            if key not in events:
                events[key] = {
                    "kind": kind, "accepted_sequence": sequence,
                    "result_status": goal["result_status"], "result_success": goal["result_success"],
                    "result_error": goal["result_error"],
                    "first_frame_index": index, "last_frame_index": index, "frame_count": 0,
                    "first_frame_timestamp": row["frame_timestamp"], "last_frame_timestamp": row["frame_timestamp"],
                }
            event = events[key]
            event.update(last_frame_index=index, last_frame_timestamp=row["frame_timestamp"])
            event["frame_count"] += 1
        return list(events.values())

    def _select_capture_sample(self, cutoff):
        """按各源的实际时间选最近过去样本，禁止缓存包刷新源数据年龄。"""
        with self.action_sample_lock:
            packets = list(self.action_sample_buffer)
        latest = packets[-1] if packets else {}
        latest_data = latest.get("data") or {}
        self._last_capture_diagnostics = {
            "reference_timestamp": cutoff,
            "checked_at": time.time(),
            "buffered_packets": len(packets),
            "latest_packet_local_received_ts": latest.get("local_recv_ts"),
            "latest_packet_bridge_timestamp": latest_data.get("timestamp"),
            "latest_packet_transport": copy.deepcopy(latest_data.get("transport")),
            "latest_packet_source_timestamps": copy.deepcopy(latest_data.get("source_timestamps")),
            "latest_gripper_target": copy.deepcopy(latest_data.get("gripper_target")),
        }
        cutoff_ns = int(round(cutoff * 1e9))
        selected, source_times, bridge_metadata = {}, {}, {}
        for key in CAPTURE_SOURCE_KEYS:
            candidates = []
            for packet in packets:
                data = packet["data"]
                value = data.get(key)
                source = (data.get("source_timestamps") or {}).get(key) or {}
                stamp = source.get("source_stamp_ns")
                if value is None:
                    continue
                if stamp is not None:
                    try:
                        source_ns = int(stamp)
                    except (ValueError, TypeError):
                        continue
                elif key == "gripper_command":
                    source_ns = int(round(source.get("host_received_ts", 0.0) * 1e9))
                else:
                    continue
                if source_ns > 0 and source_ns <= cutoff_ns:
                    candidates.append((source_ns, value, source, data))
            if candidates:
                source_time, value, source, original = max(candidates, key=lambda item: item[0])
                selected[key] = copy.deepcopy(value)
                source_times[key] = copy.deepcopy(source)
                if key == "arm_command":
                    bridge_metadata = {"bridge_packet_timestamp": original.get("timestamp"),
                                       "datetime": original.get("datetime")}
        for key in CAPTURE_SOURCE_KEYS:
            selected.setdefault(key, None)
        limits = {"arm_command": 0.2, "gripper_joints": 0.5, "gripper_target": 0.5}
        ages = {}
        for key, max_age in limits.items():
            source = source_times.get(key) or {}
            stamp = source.get("source_stamp_ns")
            if key not in selected or not stamp:
                raise RuntimeError(f"缺少图像时刻之前的 {key}，禁止录制")
            age_ns = cutoff_ns - int(stamp)
            ages[key] = age_ns / 1e9
            if age_ns > int(round(max_age * 1e9)):
                raise RuntimeError(f"{key} 源数据已过期 ({ages[key]:.3f}s > {max_age}s)，禁止录制")
        arm = selected["arm_command"]
        if arm.get("schema_version") != 1:
            raise RuntimeError("不支持的控制器快照版本")
        if arm.get("joint_names") != FR3_JOINT_ORDER:
            raise RuntimeError("控制器关节顺序不符合 fr3_joint1…7")
        if int(arm.get("stamp_ns", 0)) != int(source_times["arm_command"]["source_stamp_ns"]):
            raise RuntimeError("控制器快照与源时间戳不一致")
        for key in ("q_measured", "dq_measured", "q_target", "tau_command", "gello_q", "robot_start_q", "gello_start_q"):
            arm[key] = self._capture_vector(arm.get(key), 7, "arm_command." + key)
        activation_id = str(arm.get("activation_id")) if arm.get("activation_id") is not None else ""
        if not activation_id or int(arm.get("sequence", 0)) <= 0:
            raise RuntimeError("控制器 activation_id 或 sequence 无效")
        if not isinstance(arm.get("relative_start"), bool):
            raise RuntimeError("控制器快照缺少 relative_start")
        gripper = selected["gripper_joints"]
        names = gripper.get("names") or []
        if not all(name in names for name in FINGER_JOINT_ORDER):
            raise RuntimeError("缺少两个真实手指关节测量，禁止用命令代替观测")
        indices = [names.index(name) for name in FINGER_JOINT_ORDER]
        positions = gripper.get("position") or []
        try:
            fingers = self._capture_vector([positions[i] for i in indices], 2, "gripper_joints.position")
        except IndexError as exc:
            raise RuntimeError("手指测量长度无效") from exc
        if min(fingers) < -1e-5:
            raise RuntimeError("手指宽度测量为负数")
        normalized_gripper = {"names": list(FINGER_JOINT_ORDER), "position": fingers}
        normalized_gripper["source_names"] = gripper.get("source_names", names)
        for key in ("velocity", "effort"):
            values = gripper.get(key)
            normalized_gripper[key] = [values[i] for i in indices] if values and max(indices) < len(values) else None
        target = selected["gripper_target"]
        self._capture_gripper_execution(target)
        profile = _gripper_control_profile(target)
        if self.is_recording and self.record_metadata and self.record_metadata.get("gripper_control_profile") != profile:
            raise RuntimeError("同一 episode 内夹爪控制版本或参数发生变化，禁止录制")
        if int(target.get("stamp_ns", 0)) != int(source_times["gripper_target"]["source_stamp_ns"]):
            raise RuntimeError("夹爪 heartbeat 与源时间戳不一致")
        command_stamp = int(target.get("command_stamp_ns", 0))
        if not command_stamp or command_stamp > cutoff_ns or command_stamp > int(target["stamp_ns"]):
            raise RuntimeError("缺少图像之前已接受的夹爪命令")
        width = float(target.get("target_width_m"))
        max_width = float(target.get("max_width_m"))
        speed = float(target.get("speed_m_s"))
        if not np.isfinite(speed) or speed <= 0:
            raise RuntimeError("夹爪目标速度无效")
        if not np.isfinite([width, max_width]).all() or max_width <= 0 or not 0 <= width <= max_width + 1e-6:
            raise RuntimeError("夹爪目标宽度无效")
        if sum(fingers) > max_width + 0.005:
            raise RuntimeError("真实夹爪测量超过配置最大宽度")
        observation = np.asarray(arm["q_measured"] + [sum(fingers)], dtype=np.float32)
        action = np.asarray(arm["q_target"] + [width], dtype=np.float32)
        if not np.isfinite(observation).all() or not np.isfinite(action).all():
            raise RuntimeError("观测或动作超出有限 float32 范围")
        selected.update(bridge_metadata)
        selected["gripper_joints"] = normalized_gripper
        selected["source_timestamps"] = source_times
        selected["timestamp"] = int(arm["stamp_ns"]) / 1e9
        selected["observation.state"] = observation.tolist()
        selected["action"] = action.tolist()
        required_times = [int(source_times[key]["source_stamp_ns"]) for key in limits]
        selected["alignment"] = {
            "reference_timestamp": cutoff,
            "source_age_s": ages,
            "cross_stream_skew_s": (max(required_times) - min(required_times)) / 1e9,
            "selection": "latest_source_sample_at_or_before_image_reference",
        }
        return selected

    def _capture_action_sample(self, frame_timestamp, camera_frame_info=None):
        cutoff = frame_timestamp
        basis = "host_received" if camera_frame_info else "host_capture_no_images"
        if camera_frame_info:
            exposure_times = []
            for info in camera_frame_info:
                domain = str(info.get("timestamp_domain", "")).lower()
                timestamp = info.get("device_timestamp_ms")
                if timestamp is None or not any(name in domain for name in ("system_time", "global_time")):
                    break
                exposure = float(timestamp) / 1000.0
                if not np.isfinite(exposure) or not 0 <= frame_timestamp - exposure <= 1.0:
                    raise RuntimeError("相机曝光时间不在有效系统时间范围内")
                exposure_times.append(exposure)
            if len(exposure_times) == len(camera_frame_info):
                cutoff = min(exposure_times)
                basis = "earliest_camera_exposure_system_epoch"
        sample = self._select_capture_sample(cutoff)
        activation_id = str(sample["arm_command"]["activation_id"])
        if self.record_activation_id is not None and activation_id != self.record_activation_id:
            raise RuntimeError("控制器 activation_id 已变化，当前录制必须结束并重新开始")
        if self.record_activation_id is None:
            self.record_activation_id = activation_id
        sample["frame_timestamp"] = frame_timestamp
        sample["alignment"]["image_time_basis"] = basis
        sample["alignment"]["image_host_timestamp"] = frame_timestamp
        self.action_data_buffer.append(sample)
        if self.record_metadata:
            bases = self.record_metadata.setdefault("image_time_bases", [])
            if basis not in bases:
                bases.append(basis)

    def _check_action_recording_ready(self):
        return self._select_capture_sample(time.time())

    def _reorder_franka_block(self, joint_block):
        """Return a reordered Franka joint block for saving."""
        names = joint_block.get("names") or []
        order = self._get_franka_order(names)
        if not order:
            return joint_block, order, False

        new_block = joint_block.copy()
        for field in ("position", "velocity", "effort"):
            values = joint_block.get(field)
            if values is None:
                continue
            if len(values) < len(order):
                continue
            new_block[field] = [values[i] for i in order]

        new_block["names"] = list(FR3_JOINT_ORDER)
        return new_block, order, new_block != joint_block

    def _get_franka_order(self, names):
        if not names:
            return None
        key = tuple(names)
        if key in self._joint_order_cache:
            return self._joint_order_cache[key]

        order = None
        if all(name in names for name in FR3_JOINT_ORDER):
            order = [names.index(name) for name in FR3_JOINT_ORDER]
        else:
            index_map = {}
            for idx, name in enumerate(names):
                match = JOINT_NUMBER_RE.search(name)
                if not match:
                    continue
                joint_num = int(match.group(1))
                if 1 <= joint_num <= 7 and joint_num not in index_map:
                    index_map[joint_num] = idx
            if len(index_map) == 7:
                order = [index_map[i] for i in range(1, 8)]

        self._joint_order_cache[key] = order
        return order

    def start_recording(self):
        """开始录制"""
        if self.is_recording:
            return
        if self.replay_in_progress:
            messagebox.showwarning("提示", "回放中无法录制")
            return

        try:
            if not self.autostart_robot and not self.action_socket and not self._robot_starting:
                messagebox.showwarning("提示", "请先点击“启动遥操作”，等待机器人就绪后再录制")
                return
            if ROBOT_BACKEND in ("local_gello", "local_ros2") and (not self.robot_ready or not self.action_socket):
                self.start_button.config(state=tk.DISABLED)
                wait_thread = threading.Thread(
                    target=self._wait_and_start_local_recording_worker,
                    daemon=True,
                )
                wait_thread.start()
                return

            if not self.robot_ready or not self.action_socket:
                messagebox.showwarning("提示", "机器人未就绪，请稍后再试")
                return
            if (
                ROBOT_BACKEND != "local_gello"
                and self.active_control_mode
                and self.active_control_mode != self._get_selected_control_mode()
            ):
                messagebox.showwarning("提示", "控制模式切换中，请稍后再试")
                return

            if ROBOT_BACKEND != "local_gello" and self._get_selected_control_mode() == "relative":
                self.start_button.config(state=tk.DISABLED)
                self._set_mode_controls_state(tk.DISABLED)
                prep_thread = threading.Thread(
                    target=self._start_relative_recording_worker,
                    daemon=True
                )
                prep_thread.start()
                return

            self._start_recording_core()

        except Exception as e:
            messagebox.showerror("错误", f"启动录制失败: {str(e)}")

    def _wait_and_start_local_recording_worker(self, timeout_sec=20.0):
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            if self.robot_ready and self.action_socket:
                if ROBOT_BACKEND != "local_gello" and self._get_selected_control_mode() == "relative":
                    self._start_relative_recording_worker()
                else:
                    self.root.after(0, self._start_recording_core)
                return
            if self.local_gello_process and self.local_gello_process.poll() is not None:
                break
            if self.local_ros2_control_process and self.local_ros2_control_process.poll() is not None:
                break
            if self.local_ros2_bridge_process and self.local_ros2_bridge_process.poll() is not None:
                break
            time.sleep(0.2)

        def _fail():
            self.start_button.config(state=tk.NORMAL)
            log_path = (
                LOCAL_ROS2_LOG if ROBOT_BACKEND == "local_ros2"
                else "/tmp/local_gello_data_bridge.log"
            )
            messagebox.showwarning(
                "提示",
                f"机器人未连接，请查看 {log_path}"
            )
        self.root.after(0, _fail)

    def _start_recording_core(self):
        """实际执行录制启动逻辑"""
        try:
            task = self._get_task_text()
            if not task:
                raise RuntimeError("请填写真实任务描述后再开始录制")
            self._check_camera_recording_ready()
            initial_sample = self._check_action_recording_ready()
            self.record_activation_id = str(initial_sample["arm_command"]["activation_id"])
            self.outcome = "success"
            if hasattr(self, "outcome_var"):
                self.outcome_var.set("success")
            # 获取录制序号
            self.session_index = self.get_next_record_index()
            self.record_start_time = time.time()

            # 创建录制目录
            record_dir = os.path.join(
                self.record_base_path,
                f"record_{self.session_index:03d}"
            )
            video_session_dir = os.path.join(record_dir, "video")
            action_session_dir = os.path.join(record_dir, "action")

            os.makedirs(video_session_dir, exist_ok=True)
            os.makedirs(action_session_dir, exist_ok=True)

            self.session_dir = {
                'record': record_dir,
                'video': video_session_dir,
                'action': action_session_dir
            }

            # 更新UI
            record_name = f"record_{self.session_index:03d}"
            self.session_label.config(text=f"录制序号: {record_name}")
            self.set_current_paths(video_session_dir, action_session_dir)

            # 清空数据缓冲
            self.action_data_buffer = []
            self.last_action_record_time = 0.0
            self.video_frame_metadata = []
            self.recording_error = None

            # 创建帧目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.frame_dirs = []
            self.video_output_paths = []
            self.frame_index = 0

            for i in range(len(self.pipelines)):
                frame_dir = os.path.join(
                    video_session_dir,
                    f"camera_{i+1}_{timestamp}_frames"
                )
                os.makedirs(frame_dir, exist_ok=True)
                self.frame_dirs.append(frame_dir)
                video_path = os.path.join(
                    video_session_dir,
                    f"camera_{i+1}_{timestamp}.mp4"
                )
                self.video_output_paths.append(video_path)
                print(f"创建帧目录: {frame_dir}")

            self.record_metadata = {
                "schema_version": 2,
                "capture_contract": CAPTURE_CONTRACT,
                "final_status_authority": "metadata.json",
                "session": record_name,
                "task": task,
                "outcome": "success",
                "observation_state_features": list(STATE_FEATURE_NAMES),
                "action_features": list(ACTION_FEATURE_NAMES),
                "observation_state_units": ["rad"] * 7 + ["m"],
                "action_units": ["rad"] * 7 + ["m"],
                "control_units": {"q_measured": "rad", "dq_measured": "rad/s", "q_target": "rad",
                                  "tau_command": "Nm", "gello_q": "rad", "robot_start_q": "rad", "gello_start_q": "rad"},
                "action_semantics": "absolute_FR3_joint_position_targets_and_last_accepted_gripper_width",
                "gripper_execution_policy": (GRIPPER_EXECUTION_POLICY_V2 if initial_sample["gripper_target"]["schema_version"] == 2 else GRIPPER_EXECUTION_POLICY),
                "gripper_execution_diagnostics": [],
                "gripper_control_profile": _gripper_control_profile(initial_sample["gripper_target"]),
                "clock_domain": "ROS_system_epoch_seconds_and_nanoseconds",
                "causal_alignment": {"arm_max_age_s": 0.2, "gripper_measurement_max_age_s": 0.5,
                                     "gripper_heartbeat_max_age_s": 0.5,
                                     "image_reference": "earliest_valid_system_or_global_exposure_else_host_receipt",
                                     "host_fallback_limitation": "past_before_host_receipt_does_not_prove_past_before_exposure"},
                "controller": {key: initial_sample["arm_command"].get(key) for key in
                               ("activation_id", "relative_start", "robot_start_q", "gello_start_q", "joint_names", "sequence", "stamp_ns")},
                "gripper_contract": {"max_width_m": initial_sample["gripper_target"]["max_width_m"],
                                     "observation": "sum_of_two_measured_finger_positions_m",
                                     "action": "last_accepted_target_width_m",
                                     "heartbeat_hz": 10},
                "controller_config": self._capture_controller_config(),
                "started_at": datetime.now().isoformat(),
                "color_resolution": [640, 480],
                "capture_fps": 30,
                "record_fps": self.record_fps,
                "cameras_disabled": self.disable_cameras,
                "cameras": [
                    {
                        "camera_index": i + 1,
                        "serial": serial,
                        "video_path": os.path.relpath(self.video_output_paths[i], record_dir),
                        "frames_directory": os.path.relpath(self.frame_dirs[i], record_dir),
                    }
                    for i, serial in enumerate(self.camera_serials)
                ],
            }
            self._save_recording_metadata("recording", [])
            if hasattr(self, "task_entry"):
                self.task_entry.config(state=tk.DISABLED)

            # 开始录制
            with self.record_lock:
                self.is_recording = True

            # 更新UI
            self.status_label.config(text="录制状态: 正在录制", fg="#e74c3c")
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.delete_last_button.config(state=tk.DISABLED)
            self._update_replay_controls()

            # 开始更新录制时长
            self.update_duration()

            print(f"开始录制: record_{self.session_index:03d}")
            return True

        except Exception as e:
            messagebox.showerror("错误", f"启动录制失败: {str(e)}")
            if self.is_recording:
                self.stop_recording()
            return False

    def _capture_controller_config(self):
        path = os.path.join(LOCAL_ROS2_WS, "src", "franka_fr3_arm_controllers", "config", "controllers.yaml")
        result = {"path": path, "source": "workspace_configuration_not_runtime_parameter_readback"}
        try:
            import yaml
            with open(path, encoding="utf-8") as handle:
                parameters = yaml.safe_load(handle)["/**"]["joint_impedance_controller"]["ros__parameters"]
            result["parameters"] = {key: parameters[key] for key in
                                    ("relative_start", "k_gains", "d_gains", "k_alpha") if key in parameters}
        except Exception as exc:
            result["unavailable_reason"] = str(exc)
        return result

    def _start_recording_with_relative(self):
        """相对控制准备完成后执行录制"""
        ok = self._start_recording_core()
        if not ok:
            self._set_relative_control_enabled(False)
            self.start_button.config(state=tk.NORMAL)
            self._set_mode_controls_state(tk.NORMAL)

    def _start_relative_recording_worker(self):
        error = None
        try:
            self._set_label_text(self.robot_ready_label, "机器人状态: 相对模式准备中...", "#f39c12")
            if not self._set_relative_control_enabled(False, raise_on_failure=True):
                raise RuntimeError("关闭相对控制失败")
            robot_zero = self._get_remote_robot_zero()
            if not robot_zero:
                raise RuntimeError("无法读取相对控制初始位姿")
            if not self._wait_for_robot_zero(robot_zero, self.relative_zero_timeout_sec):
                raise RuntimeError("等待机器人回到初始位姿超时")
            if not self._set_relative_control_enabled(True, raise_on_failure=True):
                raise RuntimeError("启用相对控制失败")
            self.root.after(0, self._start_recording_with_relative)
            return
        except Exception as exc:
            error = str(exc)
        finally:
            if error:
                def _fail():
                    messagebox.showerror("错误", f"相对控制准备失败: {error}")
                    self.start_button.config(state=tk.NORMAL)
                self.root.after(0, _fail)
                try:
                    self._set_relative_control_enabled(False)
                except Exception:
                    pass
            self.root.after(0, lambda: self._set_mode_controls_state(tk.NORMAL))

    def stop_recording(self):
        """停止录制"""
        if not self.is_recording:
            return

        try:
            if ROBOT_BACKEND != "local_gello" and self.active_control_mode == "relative":
                self._set_relative_control_enabled(False)

            # 关闭录制
            with self.record_lock:
                self.is_recording = False
            self.outcome = self._get_outcome()
            if self.record_metadata:
                self.record_metadata["outcome"] = self.outcome

            errors = [self.recording_error] if self.recording_error else []
            if not self.disable_cameras and self.frame_index != len(self.action_data_buffer):
                errors.append("视频帧组与动作样本数量不一致")
            # 视频失败也继续保存已收到的动作数据与错误元信息。
            try:
                self._convert_frames_to_videos()
            except Exception as error:
                errors.append(str(error))

            # 保存动作数据
            if self.record_metadata:
                self.record_metadata.update(status="error" if errors else "complete", errors=list(errors),
                                            frame_count=self.frame_index, action_count=len(self.action_data_buffer))
            if self.action_data_buffer and self.session_dir:
                try:
                    self.save_action_data()
                except Exception as error:
                    errors.append("保存动作失败: " + str(error))
            elif self.session_dir:
                errors.append("没有可保存的动作数据")
            try:
                self._save_recording_metadata("error" if errors else "complete", errors)
            except Exception as error:
                errors.append("保存录制元信息失败: " + str(error))

            # 更新UI
            self.status_label.config(text="录制状态: 保存失败，请检查" if errors else "录制状态: 录制已停止", fg="#e74c3c" if errors else "#95a5a6")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.delete_last_button.config(state=tk.NORMAL)
            if self.session_index:
                self.replay_index_var.set(self.session_index)
            self._update_replay_controls()
            if hasattr(self, "task_entry"):
                self.task_entry.config(state=tk.NORMAL)

            if errors:
                messagebox.showerror("录制数据不完整", "本次录制未通过完整性检查，请勿作为正式样本使用。\n\n" + "\n".join(errors))
                print("录制数据不完整: " + "; ".join(errors))
            else:
                messagebox.showinfo("完成", f"录制完成！\n已保存到: record_{self.session_index:03d}")
                print(f"录制完成: record_{self.session_index:03d}")

        except Exception as e:
            messagebox.showerror("错误", f"停止录制失败: {str(e)}")

    def _convert_frames_to_videos(self):
        """将帧目录转换为视频文件"""
        if not self.frame_dirs:
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        errors = []
        for i, frame_dir in enumerate(self.frame_dirs):
            try:
                frame_files = [
                    name for name in os.listdir(frame_dir)
                    if name.startswith("frame_") and name.endswith(".jpg")
                ]
                frame_files.sort()
                if not frame_files:
                    raise RuntimeError("没有录到任何帧")
                first_path = os.path.join(frame_dir, frame_files[0])
                first_img = cv2.imread(first_path)
                if first_img is None:
                    raise RuntimeError("无法读取首帧")
                height, width = first_img.shape[:2]
                video_path = (
                    self.video_output_paths[i]
                    if i < len(self.video_output_paths)
                    else os.path.join(self.session_dir["video"], f"camera_{i+1}.mp4")
                )
                writer = cv2.VideoWriter(video_path, fourcc, self.record_fps, (width, height))
                try:
                    if not writer.isOpened():
                        raise RuntimeError("MP4 编码器无法打开输出文件")
                    for name in frame_files:
                        frame_path = os.path.join(frame_dir, name)
                        img = cv2.imread(frame_path)
                        if img is None:
                            raise RuntimeError(f"无法读取帧: {name}")
                        if writer.write(img) is False:
                            raise RuntimeError(f"MP4 写帧失败: {name}")
                finally:
                    writer.release()
                verify = cv2.VideoCapture(video_path)
                try:
                    if not verify.isOpened() or int(verify.get(cv2.CAP_PROP_FRAME_COUNT)) != len(frame_files):
                        raise RuntimeError("MP4 无法回读或帧数与原始帧不一致")
                finally:
                    verify.release()
                print(f"生成视频文件: {video_path}")
            except Exception as e:
                print(f"生成视频失败: {frame_dir} ({e})")
                errors.append(f"相机 {i + 1} 视频失败: {e}")
        if errors:
            raise RuntimeError("\n".join(errors))

    def _save_recording_metadata(self, status, errors):
        if not self.session_dir or not self.record_metadata:
            return
        self.record_metadata.update(status=status, errors=errors, frame_count=self.frame_index,
                                    action_count=len(self.action_data_buffer), outcome=self.outcome)
        record_dir = self.session_dir["record"]
        with open(os.path.join(record_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(self.record_metadata, handle, indent=2, ensure_ascii=False)
        if status != "recording":
            with open(os.path.join(self.session_dir["video"], "frame_timestamps.json"), "w", encoding="utf-8") as handle:
                json.dump(self.video_frame_metadata, handle, indent=2, ensure_ascii=False)
    def save_action_data(self):
        """保存原始遥测与有明确物理语义的八维训练观测/动作。"""
        if not self.action_data_buffer or not self.session_dir:
            return
        action_dir = self.session_dir['action']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_to_save = []
        for frame in self.action_data_buffer:
            copied = copy.deepcopy(frame)
            if isinstance(copied.get("franka_joints"), dict):
                copied["franka_joints"], _, _ = self._reorder_franka_block(copied["franka_joints"])
            self._capture_vector(copied.get("observation.state"), 8, "observation.state")
            self._capture_vector(copied.get("action"), 8, "action")
            data_to_save.append(copied)
        metadata = copy.deepcopy(self.record_metadata or {})
        metadata.update(schema_version=2, capture_contract=CAPTURE_CONTRACT,
                        task=metadata.get("task", self._get_task_text()), outcome=self._get_outcome())
        execution_diagnostics = self._capture_gripper_execution_diagnostics(data_to_save)
        profiles = [_gripper_control_profile(row["gripper_target"]) for row in data_to_save]
        if any(profile != profiles[0] for profile in profiles):
            raise RuntimeError("同一 episode 内夹爪控制版本或参数不一致，拒绝保存")
        if metadata.get("gripper_control_profile", profiles[0]) != profiles[0]:
            raise RuntimeError("夹爪控制元数据与实际遥测不一致，拒绝保存")
        execution_metadata = {"gripper_execution_policy": GRIPPER_EXECUTION_POLICY_V2 if profiles[0]["schema_version"] == 2 else GRIPPER_EXECUTION_POLICY,
                              "gripper_execution_diagnostics": execution_diagnostics,
                              "gripper_control_profile": profiles[0]}
        metadata.update(execution_metadata)
        if self.record_metadata is not None:
            self.record_metadata.update(copy.deepcopy(execution_metadata))
        if execution_diagnostics:
            print(f"夹爪执行诊断：{len(execution_diagnostics)} 个事件（Move 未完成或操作者张开取消 Grasp）；"
                  "保留原始结果、目标宽度和实测开口；任务结果由操作者确认。")
        metadata["final_status_authority"] = "metadata.json"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        json_file = os.path.join(action_dir, f"action_data_{timestamp}.json")
        hdf5_file = os.path.join(action_dir, f"action_data_{timestamp}.h5")
        if os.path.exists(json_file) or os.path.exists(hdf5_file):
            raise RuntimeError("动作输出路径已存在，拒绝覆盖")
        json_tmp, hdf5_tmp = json_file + ".tmp", hdf5_file + ".tmp"
        payload = {
            'schema_version': 2, 'capture_contract': CAPTURE_CONTRACT,
            'session': f"record_{self.session_index:03d}", 'timestamp': timestamp,
            'task': metadata['task'], 'outcome': metadata['outcome'], 'save_status': 'complete',
            'packet_count': len(data_to_save), 'metadata': metadata, 'data': data_to_save,
        }
        committed = []
        try:
            with open(json_tmp, 'w', encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)

            def raw_vector(frame, block, field, length):
                values = (frame.get(block) or {}).get(field)
                if values is None or len(values) != length:
                    return [float("nan")] * length
                return values

            gripper_results = [self._capture_gripper_result(frame['gripper_target']) for frame in data_to_save]
            with h5py.File(hdf5_tmp, 'w') as handle:
                handle.attrs['save_status'] = 'complete'
                handle.attrs['schema_version'] = 2
                handle.attrs['capture_contract'] = CAPTURE_CONTRACT
                handle.attrs['session'] = f"record_{self.session_index:03d}"
                handle.attrs['timestamp'] = timestamp
                handle.attrs['packet_count'] = len(data_to_save)
                handle.attrs['task'] = metadata['task']
                handle.attrs['outcome'] = metadata['outcome']
                handle.attrs['record_fps'] = self.record_fps
                handle.attrs['metadata_json'] = json.dumps(metadata, ensure_ascii=False)
                handle.attrs['observation_state_features'] = json.dumps(STATE_FEATURE_NAMES)
                handle.attrs['action_features'] = json.dumps(ACTION_FEATURE_NAMES)
                handle.attrs['feature_units'] = json.dumps(["rad"] * 7 + ["m"])
                handle.attrs['action_semantics'] = 'absolute_joint_position_targets_and_last_accepted_gripper_width'
                handle.create_dataset('timestamps', data=np.asarray([frame['frame_timestamp'] for frame in data_to_save], dtype=np.float64))
                handle.create_dataset('observation/state', data=np.asarray([frame['observation.state'] for frame in data_to_save], dtype=np.float32))
                handle.create_dataset('action', data=np.asarray([frame['action'] for frame in data_to_save], dtype=np.float32))
                for block, prefix, length in (("gello_joints", "gello", 7), ("franka_joints", "franka", 7), ("gripper_joints", "gripper", 2)):
                    for field, dataset in (("position", "positions"), ("velocity", "velocities"), ("effort", "efforts")):
                        handle.create_dataset(prefix + '/' + dataset,
                                              data=np.asarray([raw_vector(frame, block, field, length) for frame in data_to_save], dtype=np.float32))
                for field in ("q_target", "tau_command", "q_measured", "dq_measured", "gello_q", "robot_start_q", "gello_start_q"):
                    handle.create_dataset('control/' + field, data=np.asarray([frame['arm_command'][field] for frame in data_to_save], dtype=np.float32))
                for field in ("stamp_ns", "sequence"):
                    handle.create_dataset('control/' + field, data=np.asarray([frame['arm_command'][field] for frame in data_to_save], dtype=np.int64))
                handle.create_dataset('control/activation_id', data=np.asarray([str(frame['arm_command']['activation_id']) for frame in data_to_save], dtype=object), dtype=h5py.string_dtype('utf-8'))
                handle.create_dataset('control/relative_start', data=np.asarray([frame['arm_command']['relative_start'] for frame in data_to_save], dtype=np.bool_))
                handle.create_dataset('gripper/width_m', data=np.asarray([sum(frame['gripper_joints']['position']) for frame in data_to_save], dtype=np.float32))
                for field in ("target_width_m", "max_width_m", "speed_m_s"):
                    handle.create_dataset('gripper/' + field, data=np.asarray([frame['gripper_target'][field] for frame in data_to_save], dtype=np.float32))
                for field in ("stamp_ns", "command_stamp_ns", "sequence"):
                    handle.create_dataset('gripper/target_' + field, data=np.asarray([frame['gripper_target'][field] for frame in data_to_save], dtype=np.int64))
                handle.create_dataset('gripper/target_accepted_sequence', data=np.asarray([frame['gripper_target'].get('accepted_sequence', -1) for frame in data_to_save], dtype=np.int64))
                handle.create_dataset('gripper/target_status', data=np.asarray([frame['gripper_target']['status'] for frame in data_to_save], dtype=object), dtype=h5py.string_dtype('utf-8'))
                result_success = handle.create_dataset('gripper/target_result_success', data=np.asarray([result[0] for result in gripper_results], dtype=np.int8))
                result_success.attrs['encoding'] = '-1=no_result; 0=false; 1=true'
                for field, index in (("status", 1), ("sequence", 2)):
                    result_dataset = handle.create_dataset('gripper/target_result_' + field, data=np.asarray([result[index] for result in gripper_results], dtype=np.int64))
                    result_dataset.attrs['missing_value'] = -1
                handle.create_dataset('gripper/target_result_error', data=np.asarray([result[3] for result in gripper_results], dtype=object), dtype=h5py.string_dtype('utf-8'))
                # v2 keeps complete accepted mode/parameters alongside the unchanged Nx8 features.
                if profiles[0]["schema_version"] == 2:
                    goals = [frame["gripper_target"] for frame in data_to_save]
                    handle.create_dataset('gripper/target_schema_version', data=np.asarray([goal['schema_version'] for goal in goals], dtype=np.int64))
                    for field in ('control_policy', 'control_state', 'command_type', 'result_cancellation_reason'):
                        handle.create_dataset('gripper/target_' + field, data=np.asarray([goal[field] or '' for goal in goals], dtype=object), dtype=h5py.string_dtype('utf-8'))
                    for field in ('force_n', 'epsilon_inner_m', 'epsilon_outer_m'):
                        dataset = handle.create_dataset('gripper/target_' + field, data=np.asarray([-1.0 if goal[field] is None else goal[field] for goal in goals], dtype=np.float64))
                        dataset.attrs['missing_value'] = -1.0
                    handle.create_dataset('gripper/target_policy_config_json', data=np.asarray([json.dumps(goal['policy_config'], sort_keys=True, allow_nan=False) for goal in goals], dtype=object), dtype=h5py.string_dtype('utf-8'))
                handle.create_dataset('gripper/raw_command_percent', data=np.asarray([frame.get('gripper_command', np.nan) if frame.get('gripper_command') is not None else np.nan for frame in data_to_save], dtype=np.float32))
                positions, orientations = [], []
                for frame in data_to_save:
                    pose = frame.get('end_effector_pose') or {}
                    positions.append([pose.get('position', {}).get(axis, np.nan) for axis in ('x', 'y', 'z')])
                    orientations.append([pose.get('orientation', {}).get(axis, np.nan) for axis in ('x', 'y', 'z', 'w')])
                handle.create_dataset('end_effector/positions', data=np.asarray(positions, dtype=np.float32))
                handle.create_dataset('end_effector/orientations', data=np.asarray(orientations, dtype=np.float32))
                handle.create_dataset('alignment/reference_timestamp', data=np.asarray([frame['alignment']['reference_timestamp'] for frame in data_to_save], dtype=np.float64))
                handle.create_dataset('alignment/cross_stream_skew_s', data=np.asarray([frame['alignment']['cross_stream_skew_s'] for frame in data_to_save], dtype=np.float32))
                handle.create_dataset('alignment/image_time_basis', data=np.asarray([frame['alignment']['image_time_basis'] for frame in data_to_save], dtype=object), dtype=h5py.string_dtype('utf-8'))
                for key in CAPTURE_SOURCE_KEYS:
                    sources = [(frame.get('source_timestamps') or {}).get(key) or {} for frame in data_to_save]
                    handle.create_dataset('source_timestamps/' + key + '/source_stamp_ns', data=np.asarray([source.get('source_stamp_ns') if source.get('source_stamp_ns') is not None else -1 for source in sources], dtype=np.int64))
                    handle.create_dataset('source_timestamps/' + key + '/host_received_ts', data=np.asarray([source.get('host_received_ts', np.nan) for source in sources], dtype=np.float64))
                for key in ("arm_command", "gripper_joints", "gripper_target"):
                    handle.create_dataset('alignment/source_age_s/' + key, data=np.asarray([frame['alignment']['source_age_s'][key] for frame in data_to_save], dtype=np.float32))
            os.replace(hdf5_tmp, hdf5_file)
            committed.append(hdf5_file)
            os.replace(json_tmp, json_file)
            committed.append(json_file)
        except Exception as exc:
            for path in (json_tmp, hdf5_tmp, *committed):
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except OSError:
                    pass
            # Retain raw rows for recovery, outside the canonical action directory.
            diagnostic_dir = os.path.join(os.path.dirname(action_dir), "diagnostics")
            metadata.update(status="error", errors=["动作双写失败: " + str(exc)])
            payload['save_status'] = 'error'
            try:
                os.makedirs(diagnostic_dir, exist_ok=True)
                with open(os.path.join(diagnostic_dir, os.path.basename(json_file)), 'w', encoding='utf-8') as handle:
                    json.dump(payload, handle, indent=2, ensure_ascii=False)
            except Exception as diagnostic_error:
                print(f"诊断数据保存失败: {diagnostic_error}")
            raise RuntimeError("动作 JSON/HDF5 未全部成功保存，未提交正式动作文件: " + str(exc)) from exc
        print(f"动作数据已保存: {action_dir}")

    def update_duration(self):
        """更新录制时长"""
        if self.is_recording and self.record_start_time:
            elapsed = int(time.time() - self.record_start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60

            self.time_label.config(text=f"录制时长: {hours:02d}:{minutes:02d}:{seconds:02d}")
            self.root.after(1000, self.update_duration)

    def cleanup_remote_processes(self, force=False):
        if not self.local_ros2_session_id:
            return
        self._robot_stop_requested.set()
        self._cleanup_local_ros2_for_start()


    def on_window_close(self):
        """窗口关闭事件"""
        if ROBOT_BACKEND == "local_ros2" and self._local_ros2_use_container():
            self.cleanup_and_exit()
            return
        if self.is_recording:
            if messagebox.askyesno("确认", "正在录制中，确定要退出吗？\n（数据将会保存）"):
                self.stop_recording()
                self.cleanup_and_exit()
        else:
            if messagebox.askyesno("确认", "确定要退出吗？"):
                self.cleanup_and_exit()

    def on_exit_button(self):
        """退出按钮事件"""
        self.on_window_close()

    def cleanup_and_exit(self):
        """清理并退出"""
        if ROBOT_BACKEND == "local_ros2" and self._local_ros2_use_container():
            if self._robot_start_attempted or self._robot_starting or self._robot_stopping or self._robot_stop_error:
                self.stop_teleoperation(exit_after=True)
            else:
                self._finish_exit()
            return
        self.cleanup_remote_processes()
        self._finish_exit()

    def _finish_exit(self):
        """仅在控制停止成功或从未启动时销毁窗口。"""
        if getattr(self, "_exit_finalizing", False):
            return
        self._exit_finalizing = True
        print("正在退出...")
        self.running = False

        if self.is_recording:
            self.stop_recording()

        if self.action_socket:
            try:
                self.action_socket.close()
            except Exception:
                pass
            self.action_socket = None

        self._release_cameras_and_destroy()

    def _release_cameras_and_destroy(self):
        # Keep Tk's event loop alive while the reader finishes an outstanding
        # frame wait / root.after call. Stopping RealSense or destroying Tk while
        # that thread still owns native frame buffers can crash the process.
        preview_thread = getattr(self, "preview_thread", None)
        if preview_thread is not None and preview_thread.is_alive():
            self.root.after(50, self._release_cameras_and_destroy)
            return
        if preview_thread is not None:
            preview_thread.join(timeout=0)
            self.preview_thread = None
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except:
                pass
        self.pipelines = []

        # 关闭窗口
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass

    def run(self):
        """运行主循环"""
        self.root.mainloop()


def main():
    try:
        app = VideoActionRecorder()
        app.run()
    except KeyboardInterrupt:
        print("\n程序被中断")
        sys.exit(0)
    except Exception as e:
        print(f"程序错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
