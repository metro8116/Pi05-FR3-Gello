#!/usr/bin/env python3
"""Convert schema-2 GELLO/FR3 captures to an official LeRobot Dataset v3.

Requires numpy, opencv-python and h5py for validation; export additionally needs
LeRobot (tested version documented by --help). No robot connection or Hub upload.
Images remain RGB at their captured resolution. Model resizing / key renaming
belong to training and deployment processors, not this converter.

Examples::

    python tools/convert_to_lerobot_v30.py --dry-run --report runs/conversion_check.json
    python tools/convert_to_lerobot_v30.py --repo-id local/fr3_avocado_pi05 \
        --output-dir ./data/lerobot/fr3_avocado_v30

Only complete, successful schema-2 episodes are selected by default. Old captures
cannot recover the actual robot target and are reported as skipped. An invalid
selected episode aborts the whole export instead of silently changing the set.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any


CONTRACT = "fr3_joint_position_v1"
JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
FINGERS = ["fr3_finger_joint1", "fr3_finger_joint2"]
STATE_NAMES = JOINTS + ["gripper_width_m"]
ACTION_NAMES = JOINTS + ["gripper_target_width_m"]
UNITS = ["rad"] * 7 + ["m"]
DEFAULT_SERIALS = ["233522074047", "405622074065"]
DEFAULT_KEYS = ["observation.images.camera_1", "observation.images.camera_2"]
SOURCE_LIMITS = {"arm_command": 0.2, "gripper_joints": 0.5, "gripper_target": 0.5}
ARM_FIELDS = ("q_measured", "dq_measured", "q_target", "tau_command", "gello_q",
              "robot_start_q", "gello_start_q")
ACTION_SEMANTICS = "absolute_FR3_joint_position_targets_and_last_accepted_gripper_width"
MOVE_KFAIL = "libfranka gripper: Command failed!"
MOVE_KFAIL_ERROR = "Move failed (status=6): " + MOVE_KFAIL
GRIPPER_EXECUTION_POLICY = "grasp_hold_and_accepted_move_with_raw_diagnostics_v2"
RESULT_FIELDS = ("result_success", "result_status", "result_sequence", "result_error")


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


class ConversionError(RuntimeError):
    """Invalid input or an unsupported dataset writer."""


@dataclass
class Episode:
    record: str
    path: Path
    metadata: dict
    state: Any
    action: Any
    timestamps: Any
    video_paths: list[Path]
    image_shape: tuple[int, int, int]
    provenance: dict[str, Any]
    fingerprints: dict[str, dict]
    timing: dict
    warnings: list[str] = field(default_factory=list)
    gripper_execution_diagnostics: list[dict] = field(default_factory=list)
    gripper_control_profile: dict = field(default_factory=dict)
    # Set only for a reviewed interval view, after validating the complete raw
    # episode. Video paths and fingerprints always refer to original raw bytes.
    source_frame_count: int | None = None
    original_metadata: dict | None = None
    trim: dict | None = None
    gripper_execution_frame_kinds: list[str | None] = field(default_factory=list, repr=False)

    @property
    def frames(self):
        return len(self.timestamps)


def require(condition, message):
    if not condition:
        raise ConversionError(message)


def number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{name} must be a finite number")
    return float(value)


def integer(value, name, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"{name} must be an integer >= {minimum}")
    return value


def vector(value, size, name):
    require(isinstance(value, list) and len(value) == size, f"{name} must have {size} values")
    return np.asarray([number(x, name) for x in value], dtype=np.float64)


def equal_values(actual, expected, name, atol=1e-6):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all()
            and np.allclose(actual, expected, rtol=1e-5, atol=atol),
            f"{name}: values/shape disagree")


def read_json(path):
    try:
        # Optional raw fields can contain NaN in historical diagnostics. Required
        # numeric values are checked explicitly; none can reach exported features.
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConversionError(f"Cannot read {path}: {exc}") from exc


def inside_file(record, relative):
    require(isinstance(relative, str) and relative, "Missing relative source-file path")
    path = (record / relative).resolve()
    require(path.is_relative_to(record.resolve()) and path.is_file(),
            f"Source file missing or outside episode: {relative}")
    return path


def fingerprint(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"Source changed while reading: {path}")
    return {"size": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest.hexdigest()}


def source_stamp(row, key, reference):
    source = (row.get("source_timestamps") or {}).get(key) or {}
    stamp = integer(source.get("source_stamp_ns"), f"{key}.source_stamp_ns", 1)
    age = reference - stamp / 1e9
    require(-1e-6 <= age <= SOURCE_LIMITS[key] + 1e-6,
            f"{key}: future or stale source (age {age:.6f}s)")
    return stamp, max(0.0, age)


def gripper_result(goal):
    """Preserve the optional raw result, including the distinction false/missing."""
    success = goal.get("result_success")
    status = goal.get("result_status")
    sequence = goal.get("result_sequence")
    error = goal.get("result_error")
    require(success is None or isinstance(success, bool), "gripper.result_success must be a boolean or null")
    for field, value, minimum in (("result_status", status, 0), ("result_sequence", sequence, 1)):
        if value is not None:
            integer(value, "gripper." + field, minimum)
    require(error is None or isinstance(error, str), "gripper.result_error must be a string or null")
    return (-1 if success is None else int(success), -1 if status is None else status,
            -1 if sequence is None else sequence, "" if error is None else error)


def gripper_target_recordable(goal, request_sequence, accepted_sequence):
    """Validate accepted targets without changing the original action vector."""
    require(goal.get("sequence") == request_sequence and goal.get("accepted_sequence") == accepted_sequence,
            "Gripper request/accepted sequence arguments disagree")
    try:
        return _gripper_execution_kind(goal)
    except ValueError as exc:
        raise ConversionError(str(exc)) from exc


def diagnostic_warnings(diagnostics):
    return [
        f"Gripper diagnostic {item['kind']}: accepted_sequence={item['accepted_sequence']}, "
        f"frames={item['first_frame_index']}..{item['last_frame_index']}, count={item['frame_count']}, "
        f"capture_timestamp={item['first_frame_timestamp']:.9f}..{item['last_frame_timestamp']:.9f}. "
        + ("Accepted target and raw execution failure retained; cause is undetermined and does not establish grasp success."
         if item["kind"] == "accepted_move_kfail" else "Operator opening canceled the previous Grasp; cancellation retained independently and does not establish task success.")
        for item in diagnostics]


def validate_episode(record, metadata, args):
    require(metadata.get("capture_contract") == CONTRACT, "Unsupported capture_contract")
    require(not metadata.get("errors"), "Root metadata contains recording errors")
    require(not metadata.get("cameras_disabled"), "Camera-disabled episode cannot train this two-camera policy")
    task = metadata.get("task")
    require(isinstance(task, str) and task.strip(), "Missing task text")
    for key, expected in (("observation_state_features", STATE_NAMES), ("action_features", ACTION_NAMES),
                          ("observation_state_units", UNITS), ("action_units", UNITS)):
        require(metadata.get(key) == expected, f"Unsupported {key}")
    require(metadata.get("action_semantics") == ACTION_SEMANTICS, "Unsupported action semantics")
    require(abs(number(metadata.get("record_fps"), "record_fps") - args.fps) < 0.01,
            "record_fps differs from --fps; conversion does not resample actions")
    count = integer(metadata.get("action_count"), "action_count", args.min_frames)
    require(metadata.get("frame_count") == count, "metadata frame/action counts differ")
    cameras = metadata.get("cameras")
    require(isinstance(cameras, list) and len(cameras) == 2, "Exactly two camera entries are required")
    cameras = sorted(cameras, key=lambda camera: camera.get("camera_index", -1))
    require([camera.get("camera_index") for camera in cameras] == [1, 2], "Invalid camera_index mapping")
    require([camera.get("serial") for camera in cameras] == args.camera_serials,
            "Camera serial/order differs from --camera-serials")
    resolution = metadata.get("color_resolution")
    require(isinstance(resolution, list) and len(resolution) == 2, "Missing color_resolution")
    width, height = [integer(x, "color_resolution", 1) for x in resolution]
    video_paths = [inside_file(record, camera.get("video_path")) for camera in cameras]
    require(video_paths[0] != video_paths[1], "Both cameras refer to the same video")
    json_paths = sorted((record / "action").glob("action_data_*.json"))
    h5_paths = sorted((record / "action").glob("action_data_*.h5"))
    require(len(json_paths) == len(h5_paths) == 1, "Need exactly one canonical JSON/HDF5 action pair")
    require(json_paths[0].stem == h5_paths[0].stem, "JSON/HDF5 action filenames do not form a pair")
    timestamp_path = inside_file(record, "video/frame_timestamps.json")
    paths = [record / "metadata.json", json_paths[0], h5_paths[0], timestamp_path, *video_paths]
    signatures = {str(path.relative_to(record)): fingerprint(path) for path in paths}
    payload, frame_metadata = read_json(json_paths[0]), read_json(timestamp_path)
    require(payload.get("schema_version") == 2 and payload.get("capture_contract") == CONTRACT,
            "Action JSON has a different schema/contract")
    require(payload.get("save_status") == "complete", "Action pair was not successfully saved")
    require(payload.get("task") == task and payload.get("outcome") == metadata.get("outcome"),
            "Action task/outcome disagrees with root metadata")
    require(payload.get("packet_count") == count, "Action packet_count differs from metadata")
    rows = payload.get("data")
    require(isinstance(rows, list) and len(rows) == count, "Action row count differs from metadata")
    require(isinstance(frame_metadata, list) and len(frame_metadata) == count,
            "Camera timestamp count differs from actions")
    states, actions, timestamps, references, arm_stamps, sequences = [], [], [], [], [], []
    finger_stamps, target_stamps, command_stamps, finger_widths, target_widths = [], [], [], [], []
    accepted_sequences, accepted_events = [], []
    target_statuses, target_results, request_sequences, target_errors, target_raw_results = [], [], [], [], []
    execution_diagnostics = {}
    execution_frame_kinds = []
    control_profiles, raw_control_profiles = [], []
    camera_times, camera_numbers, time_bases = [], [], []
    arm_values = {name: [] for name in ARM_FIELDS}
    activation = None
    zeroes = None
    previous_camera_numbers = [-1, -1]
    maximum_width = number((metadata.get("gripper_contract") or {}).get("max_width_m"), "gripper_contract.max_width_m")
    for index, (row, video_frame) in enumerate(zip(rows, frame_metadata, strict=True)):
        prefix = f"frame {index}"
        require(isinstance(row, dict) and isinstance(video_frame, dict), f"{prefix}: invalid row")
        stamp = number(row.get("frame_timestamp"), prefix + ".frame_timestamp")
        require(video_frame.get("frame_index") == index, f"{prefix}: camera frame index mismatch")
        require(abs(number(video_frame.get("frame_timestamp"), "video timestamp") - stamp) <= 1e-6,
                f"{prefix}: video/action timestamps differ")
        alignment = row.get("alignment") or {}
        reference = number(alignment.get("reference_timestamp"), "alignment.reference_timestamp")
        basis = alignment.get("image_time_basis")
        require(basis in ("earliest_camera_exposure_system_epoch", "host_received"),
                f"{prefix}: unsupported image_time_basis")
        require(basis != "host_received" or args.allow_host_timestamps,
                "Host-receipt alignment cannot prove exposure causality; inspect then opt in with --allow-host-timestamps")
        infos = video_frame.get("cameras")
        require(isinstance(infos, list) and len(infos) == 2, f"{prefix}: missing camera timestamps")
        require([info.get("serial") for info in infos] == args.camera_serials,
                f"{prefix}: camera timestamp order/serial mismatch")
        exposure, frame_numbers = [], []
        for camera_index, info in enumerate(infos):
            frame_number = integer(info.get("device_frame_number"), "device_frame_number")
            require(frame_number > previous_camera_numbers[camera_index],
                    f"{prefix}: repeated/backward camera frame number")
            frame_numbers.append(frame_number)
            previous_camera_numbers[camera_index] = frame_number
            if basis == "earliest_camera_exposure_system_epoch":
                domain = str(info.get("timestamp_domain", ""))
                require("global_time" in domain or "system_time" in domain, f"{prefix}: wrong camera clock domain")
                exposure_time = number(info.get("device_timestamp_ms"), "device_timestamp_ms") / 1000.0
                require(-1e-6 <= stamp - exposure_time <= 1.0 + 1e-6, f"{prefix}: camera exposure outside host clock range")
                exposure.append(exposure_time)
            else:
                exposure.append(float("nan"))
        if basis == "earliest_camera_exposure_system_epoch":
            require(abs(reference - min(exposure)) <= 1e-6, f"{prefix}: exposure reference disagrees with camera timestamps")
            require(max(exposure) - min(exposure) <= args.max_camera_skew,
                    f"{prefix}: camera exposure skew exceeds --max-camera-skew")
        else:
            require(abs(reference - stamp) <= 1e-6, f"{prefix}: host reference differs from frame time")
        arm = row.get("arm_command") or {}
        require(arm.get("schema_version") == 1 and arm.get("joint_names") == JOINTS,
                f"{prefix}: invalid controller telemetry schema/joint order")
        current_activation = arm.get("activation_id")
        require(isinstance(current_activation, str) and current_activation, "Missing controller activation_id")
        activation = current_activation if activation is None else activation
        require(activation == current_activation, "Controller reactivated within the episode")
        sequence = integer(arm.get("sequence"), "arm.sequence", 1)
        require(not sequences or sequence >= sequences[-1], "Controller sequence moved backwards")
        arm_stamp, arm_age = source_stamp(row, "arm_command", reference)
        require(arm.get("stamp_ns") == arm_stamp, "Arm payload and source timestamp differ")
        require(not arm_stamps or arm_stamp >= arm_stamps[-1], "Controller source time moved backwards")
        require(not sequences or sequence != sequences[-1] or arm_stamp == arm_stamps[-1],
                "Same controller sequence has two source times")
        require(isinstance(arm.get("relative_start"), bool), "Missing relative_start")
        for name in ARM_FIELDS:
            arm_values[name].append(vector(arm.get(name), 7, name))
            if sequences and sequence == sequences[-1]:
                require(np.array_equal(arm_values[name][-1], arm_values[name][-2]),
                        "Same controller snapshot ID contains different measured/command values")
        current_zeroes = (arm["relative_start"], arm["robot_start_q"], arm["gello_start_q"])
        zeroes = current_zeroes if zeroes is None else zeroes
        require(current_zeroes == zeroes, "Controller startup zeroes changed within one activation")
        gripper = row.get("gripper_joints") or {}
        names = gripper.get("names")
        require(isinstance(names, list) and sorted(names) == sorted(FINGERS), "Need exactly two measured finger joints")
        positions = vector(gripper.get("position"), 2, "finger positions")
        require(float(positions.min()) >= -1e-5, "Negative measured finger position")
        width_m = float(positions.sum())
        finger_stamp, finger_age = source_stamp(row, "gripper_joints", reference)
        require(not finger_stamps or finger_stamp >= finger_stamps[-1], "Finger measurement time moved backwards")
        goal = row.get("gripper_target") or {}
        require(goal.get("valid") is True, "Missing/invalid accepted gripper target")
        profile = _gripper_control_profile(goal)
        require(not control_profiles or profile == control_profiles[0], "Gripper control version/policy_config changed within episode")
        control_profiles.append(profile)
        raw_control_profiles.append({key: goal.get(key) for key in ("schema_version", "control_policy", "control_state", "command_type", "force_n", "epsilon_inner_m", "epsilon_outer_m", "policy_config", "result_cancellation_reason")})
        request_sequence = integer(goal.get("sequence"), "gripper.sequence", 1)
        accepted_sequence = integer(goal.get("accepted_sequence"), "gripper.accepted_sequence", 1)
        require(accepted_sequence <= request_sequence, "Gripper accepted sequence exceeds request sequence")
        result = gripper_result(goal)
        execution_kind = gripper_target_recordable(goal, request_sequence, accepted_sequence)
        execution_frame_kinds.append(execution_kind)
        if execution_kind:
            diagnostic = execution_diagnostics.setdefault((execution_kind, accepted_sequence), {
                "kind": execution_kind, "accepted_sequence": accepted_sequence,
                "result_status": goal["result_status"], "result_success": goal["result_success"],
                "result_error": goal["result_error"], "first_frame_index": index,
                "last_frame_index": index, "frame_count": 0,
                "first_frame_timestamp": stamp, "last_frame_timestamp": stamp})
            diagnostic.update(last_frame_index=index, last_frame_timestamp=stamp,
                              frame_count=diagnostic["frame_count"] + 1)
        target_stamp, goal_age = source_stamp(row, "gripper_target", reference)
        require(goal.get("stamp_ns") == target_stamp, "Gripper heartbeat source time differs")
        command_stamp = integer(goal.get("command_stamp_ns"), "gripper.command_stamp_ns", 1)
        require(command_stamp <= target_stamp, "Accepted gripper goal is newer than its heartbeat")
        target_width = number(goal.get("target_width_m"), "target_width_m")
        maximum = number(goal.get("max_width_m"), "max_width_m")
        speed = number(goal.get("speed_m_s"), "speed_m_s")
        require(maximum > 0 and 0 <= target_width <= maximum + 1e-6
                and width_m <= maximum + 0.005 and speed > 0,
                "Gripper measurement/target exceeds configured physical width")
        require(abs(maximum - maximum_width) <= 1e-6, "Gripper maximum width changed from root contract")
        event = (command_stamp, target_width, speed, goal.get("command_type"), goal.get("force_n"),
                 goal.get("epsilon_inner_m"), goal.get("epsilon_outer_m"))
        if accepted_sequences:
            require(accepted_sequence >= accepted_sequences[-1] and command_stamp >= command_stamps[-1]
                    and target_stamp >= target_stamps[-1], "Gripper accepted sequence or source/event time moved backwards")
            if accepted_sequence == accepted_sequences[-1]:
                require(event == accepted_events[-1], "Same accepted gripper goal was relabelled with different time/width/speed")
        state, action = vector(row.get("observation.state"), 8, "observation.state"), vector(row.get("action"), 8, "action")
        equal_values(state, [*arm_values["q_measured"][-1], width_m], "Observation must use actual measured joints/fingers")
        equal_values(action, [*arm_values["q_target"][-1], target_width], "Action must use actual controller/accepted gripper target")
        for key, age in (("arm_command", arm_age), ("gripper_joints", finger_age), ("gripper_target", goal_age)):
            recorded_age = (alignment.get("source_age_s") or {}).get(key)
            require(abs(number(recorded_age, "source_age_s") - age) <= 1e-6, "Recorded source age disagrees with timestamps")
        states.append(state); actions.append(action); timestamps.append(stamp); references.append(reference)
        arm_stamps.append(arm_stamp); sequences.append(sequence); finger_stamps.append(finger_stamp)
        target_stamps.append(target_stamp); command_stamps.append(command_stamp)
        finger_widths.append(width_m); target_widths.append(target_width)
        accepted_sequences.append(accepted_sequence); accepted_events.append(event)
        target_statuses.append(goal["status"]); target_results.append(result)
        request_sequences.append(request_sequence); target_errors.append(goal.get("error"))
        # JSON strings preserve absent versus explicit null fields without pickle.
        target_raw_results.append(json.dumps({key: goal[key] for key in
                                  ("status", "error", *RESULT_FIELDS, "result_cancellation_reason") if key in goal},
                                  ensure_ascii=False, sort_keys=True, allow_nan=False))
        camera_times.append(exposure); camera_numbers.append(frame_numbers); time_bases.append(basis)
    state, action = np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)
    require(np.isfinite(state).all() and np.isfinite(action).all(), "State/action overflows float32")
    timestamps = np.asarray(timestamps, dtype=np.float64)
    dt = np.diff(timestamps)
    require(np.all(dt > 0), "Frame timestamps must be strictly increasing")
    require(np.all(np.diff(references) > 0), "Image reference timestamps must be strictly increasing")
    observed_fps = (count - 1) / (timestamps[-1] - timestamps[0]) if count > 1 else args.fps
    require(abs(observed_fps / args.fps - 1) <= args.fps_tolerance,
            f"Mean source FPS {observed_fps:.3f} differs too much from {args.fps}; do not silently retime this episode")
    require(not len(dt) or float(dt.max()) <= args.max_gap_factor / args.fps,
            "Frame timestamp gap exceeds --max-gap-factor / --fps")
    controller = metadata.get("controller") or {}
    require(controller.get("activation_id") == activation, "Root controller activation disagrees with action rows")
    for key, expected in (("joint_names", JOINTS), ("relative_start", zeroes[0]),
                          ("robot_start_q", zeroes[1]), ("gello_start_q", zeroes[2])):
        require(controller.get(key) == expected, f"Root controller {key} differs")
    profile = control_profiles[0]
    require(profile["schema_version"] == 1 or metadata.get("gripper_control_profile") == profile,
            "Root gripper control profile missing/differs from actual rows")
    require(metadata.get("gripper_control_profile", profile) == profile,
            "Root gripper control profile differs from actual rows")
    require((payload.get("metadata") or {}).get("gripper_control_profile", profile if profile["schema_version"] == 1 else None) == profile,
            "Action metadata gripper control profile differs from actual rows")
    with h5py.File(h5_paths[0], "r") as handle:
        for key, expected in (("schema_version", 2), ("capture_contract", CONTRACT), ("save_status", "complete"),
                              ("task", task), ("outcome", metadata["outcome"]), ("packet_count", count)):
            require(handle.attrs.get(key) == expected, f"HDF5 attribute {key} differs")
        for key, expected in (("observation/state", state), ("action", action), ("timestamps", timestamps),
                              ("control/stamp_ns", arm_stamps), ("control/sequence", sequences),
                              ("gripper/width_m", finger_widths), ("gripper/target_width_m", target_widths),
                              ("gripper/target_sequence", request_sequences),
                              ("gripper/target_accepted_sequence", accepted_sequences),
                              ("gripper/target_stamp_ns", target_stamps), ("gripper/target_command_stamp_ns", command_stamps)):
            require(key in handle, f"Missing HDF5 dataset {key}")
            if key.endswith("stamp_ns") or key in ("control/sequence", "gripper/target_sequence", "gripper/target_accepted_sequence"):
                require(np.array_equal(handle[key][:], np.asarray(expected, dtype=np.int64)), f"HDF5 {key} differs")
            elif key == "timestamps":
                require(handle[key].shape == timestamps.shape and np.isfinite(handle[key][:]).all()
                        and np.allclose(handle[key][:], timestamps, rtol=0, atol=1e-6),
                        "HDF5 timestamps differ")
            else:
                equal_values(handle[key][:], expected, f"HDF5 {key}", atol=1e-6)
        for key in ARM_FIELDS:
            require("control/" + key in handle, f"Missing HDF5 control/{key}")
            equal_values(handle["control/" + key][:], arm_values[key], "HDF5 control/" + key)
        # Older schema-2 episodes can lack raw result datasets. Both recorded
        # execution exceptions require all four JSON/HDF5 diagnostics to agree.
        needs_results = "not_reached" in target_statuses or bool(execution_diagnostics) or profile["schema_version"] == 2
        for field, index in (("success", 0), ("status", 1), ("sequence", 2), ("error", 3)):
            key = "gripper/target_result_" + field
            if key not in handle:
                require(not needs_results, f"Missing HDF5 dataset {key} for recorded Move outcome")
                continue
            dataset = handle[key]
            require(dataset.shape == (count,), f"HDF5 {key} has an invalid shape")
            expected = [result[index] for result in target_results]
            if field == "error":
                require(h5py.check_string_dtype(dataset.dtype) is not None, f"HDF5 {key} must contain strings")
                require(list(dataset.asstr()[:]) == expected, f"HDF5 {key} differs")
            else:
                require(dataset.dtype == np.dtype("int8") if field == "success" else dataset.dtype.kind == "i",
                        f"HDF5 {key} has an invalid integer type")
                require(np.array_equal(dataset[:], np.asarray(expected, dtype=np.int64)), f"HDF5 {key} differs")
        if profile["schema_version"] == 2:
            h5_metadata = json.loads(handle.attrs.get("metadata_json", "{}"))
            require(h5_metadata.get("gripper_control_profile") == profile, "HDF5 gripper control profile missing/differs")
            for field in ("schema_version", "control_policy", "control_state", "command_type", "force_n",
                          "epsilon_inner_m", "epsilon_outer_m", "policy_config", "result_cancellation_reason"):
                key = "gripper/target_" + ("policy_config_json" if field == "policy_config" else field)
                require(key in handle and handle[key].shape == (count,), "Missing/invalid HDF5 v2 dataset " + key)
                dataset = handle[key]
                expected = [item[field] for item in raw_control_profiles]
                if field in ("force_n", "epsilon_inner_m", "epsilon_outer_m"):
                    require(dataset.dtype.kind == "f" and dataset.attrs.get("missing_value") == -1.0,
                            "HDF5 nullable Grasp parameter encoding differs: " + key)
                    require(np.array_equal(dataset[:], np.asarray([-1.0 if value is None else value for value in expected], dtype=np.float64)),
                            "HDF5 " + key + " differs from the exact accepted parameter")
                elif field == "schema_version":
                    require(dataset.dtype.kind == "i" and np.array_equal(dataset[:], expected), "HDF5 schema version differs")
                else:
                    require(h5py.check_string_dtype(dataset.dtype) is not None, "HDF5 v2 field must contain strings: " + key)
                    expected = [json.dumps(value, sort_keys=True, allow_nan=False) if field == "policy_config" else value or "" for value in expected]
                    require(list(dataset.asstr()[:]) == expected, "HDF5 " + key + " differs")
            require("gripper/target_status" in handle, "Missing HDF5 v2 target_status")
        if "gripper/target_status" in handle:
            dataset = handle["gripper/target_status"]
            require(dataset.shape == (count,) and h5py.check_string_dtype(dataset.dtype) is not None,
                    "HDF5 gripper/target_status has an invalid shape/type")
            require(list(dataset.asstr()[:]) == target_statuses, "HDF5 gripper/target_status differs")
        for key, expected in (("arm_command", arm_stamps), ("gripper_joints", finger_stamps), ("gripper_target", target_stamps)):
            dataset = f"source_timestamps/{key}/source_stamp_ns"
            require(dataset in handle and np.array_equal(handle[dataset][:], expected), f"HDF5 {dataset} differs")
        require("control/activation_id" in handle
                and list(handle["control/activation_id"].asstr()[:]) == [activation] * count,
                "HDF5 controller activation differs")
    diagnostics = list(execution_diagnostics.values())
    warnings = diagnostic_warnings(diagnostics)
    for camera, video_path in zip(cameras, video_paths, strict=True):
        cap = cv2.VideoCapture(str(video_path))
        try:
            require(cap.isOpened(), f"Cannot open video {video_path}")
            require(abs(cap.get(cv2.CAP_PROP_FPS) - args.fps) < 0.05, f"Video FPS differs: {video_path}")
            decoded = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                require(image.shape == (height, width, 3), f"Video image shape differs: {video_path}")
                decoded += 1
            require(decoded == count, f"Video decoded frame count {decoded} != {count}: {video_path}")
        finally:
            cap.release()
        # Raw JPEGs are optional when transferring data to the training machine.
        raw_name = camera.get("frames_directory")
        if raw_name:
            raw = (record / raw_name).resolve()
            require(raw.is_relative_to(record.resolve()), "Raw frame directory is outside episode")
            if raw.is_dir():
                expected = {f"frame_{i:06d}.jpg" for i in range(count)}
                require({p.name for p in raw.glob("frame_*.jpg")} == expected, "Raw JPEG frame indices/count differ")
            else:
                warnings.append(f"Optional raw JPEG directory not present: {raw_name}")
    provenance = {
        "capture_timestamp": timestamps, "image_reference_timestamp": np.asarray(references),
        "arm_stamp_ns": np.asarray(arm_stamps, dtype=np.int64), "arm_sequence": np.asarray(sequences, dtype=np.int64),
        "gripper_measurement_stamp_ns": np.asarray(finger_stamps, dtype=np.int64),
        "gripper_heartbeat_stamp_ns": np.asarray(target_stamps, dtype=np.int64),
        "gripper_accepted_command_stamp_ns": np.asarray(command_stamps, dtype=np.int64),
        "gripper_request_sequence": np.asarray(request_sequences, dtype=np.int64),
        "gripper_accepted_sequence": np.asarray(accepted_sequences, dtype=np.int64),
        "gripper_target_status": np.asarray(target_statuses),
        "gripper_target_error": np.asarray(["" if value is None else value for value in target_errors]),
        "gripper_target_result_success": np.asarray([value[0] for value in target_results], dtype=np.int8),
        "gripper_target_result_status": np.asarray([value[1] for value in target_results], dtype=np.int64),
        "gripper_target_result_sequence": np.asarray([value[2] for value in target_results], dtype=np.int64),
        "gripper_target_result_error": np.asarray([value[3] for value in target_results]),
        "gripper_target_raw_result_json": np.asarray(target_raw_results),
        "gripper_target_schema_version": np.asarray([item["schema_version"] for item in raw_control_profiles], dtype=np.int64),
        "gripper_target_control_policy": np.asarray([item["control_policy"] or "move_width_v1" for item in raw_control_profiles]),
        "gripper_target_control_state": np.asarray([item["control_state"] or "" for item in raw_control_profiles]),
        "gripper_target_command_type": np.asarray([item["command_type"] or ("move" if item["schema_version"] == 1 else "") for item in raw_control_profiles]),
        "gripper_target_result_cancellation_reason": np.asarray([item["result_cancellation_reason"] or "" for item in raw_control_profiles]),
        "gripper_target_policy_config_json": np.asarray([json.dumps(item["policy_config"], sort_keys=True, allow_nan=False) for item in raw_control_profiles]),
        **{"gripper_target_" + field: np.asarray([-1.0 if item[field] is None else item[field] for item in raw_control_profiles], dtype=np.float64)
           for field in ("force_n", "epsilon_inner_m", "epsilon_outer_m")},
        "camera_exposure_timestamp": np.asarray(camera_times, dtype=np.float64),
        "camera_device_frame_number": np.asarray(camera_numbers, dtype=np.int64),
        "image_time_basis": np.asarray(time_bases),
    }
    timing = {"observed_fps": float(observed_fps), "duration_s": float(timestamps[-1] - timestamps[0]),
              "max_frame_gap_s": float(dt.max()) if len(dt) else 0.0,
              "max_camera_skew_s": float(np.nanmax(np.ptp(camera_times, axis=1)))
              if "earliest_camera_exposure_system_epoch" in time_bases else None}
    return Episode(record.name, record, metadata, state, action, timestamps, video_paths,
                   (height, width, 3), provenance, signatures, timing, warnings, diagnostics, profile,
                   gripper_execution_frame_kinds=execution_frame_kinds)


def discover(args, report):
    root = args.input_dir
    require(root.is_dir(), f"Input directory does not exist: {root}")
    records = [root] if (root / "metadata.json").is_file() else sorted(
        [path for path in root.iterdir() if path.is_dir() and re.fullmatch(r"record_\d+", path.name)],
        key=lambda path: int(path.name.split("_")[-1]))
    require(records, f"No record_* episodes found in {root}")
    if args.episodes:
        missing = set(args.episodes) - {path.name for path in records}
        require(not missing, f"Unknown --episodes: {sorted(missing)}")
        records = [path for path in records if path.name in args.episodes]
    episodes = []
    for record in records:
        item = {"record": record.name, "path": str(record)}
        try:
            metadata = read_json(record / "metadata.json")
            require(isinstance(metadata, dict), "metadata.json must be an object")
            if metadata.get("schema_version") != 2:
                report["skipped"].append({**item, "reason": "Not schema 2; actual robot targets/task labels cannot be reconstructed"})
                continue
            if metadata.get("status") != "complete":
                report["skipped"].append({**item, "reason": "Recording status is not complete"})
                continue
            if metadata.get("outcome") not in args.include_outcomes:
                report["skipped"].append({**item, "reason": f"Outcome {metadata.get('outcome')!r} not selected"})
                continue
            print(f"Validating {record.name} ...", flush=True)
            episode = validate_episode(record, metadata, args)
            require(read_json(record / "metadata.json") == metadata, "Root metadata changed during validation; stop recording before converting")
            if episodes:
                require(episode.image_shape == episodes[0].image_shape, "Image resolution changed between episodes")
                require(episode.metadata.get("gripper_contract") == episodes[0].metadata.get("gripper_contract"),
                        "Gripper contract changed between episodes")
            episodes.append(episode)
            report["selected"].append({**item, "frames": episode.frames, "task": metadata["task"],
                                       "outcome": metadata["outcome"], "timing": episode.timing,
                                       "warnings": episode.warnings,
                                       "gripper_execution_diagnostics": episode.gripper_execution_diagnostics,
                                       "gripper_control_profile": episode.gripper_control_profile})
        except (ConversionError, OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
            report["errors"].append({**item, "error": str(exc)})
    profiles = {json.dumps(episode.gripper_control_profile, sort_keys=True) for episode in episodes}
    if len(profiles) > 1:
        warning = "Mixed gripper control profiles in dataset: " + "; ".join(sorted(profiles)) + ". Features retain absolute last-accepted width; deployment must select its execution policy explicitly."
        report.setdefault("warnings", []).append(warning)
        for episode in episodes:
            episode.warnings.append(warning)
    report["total_frames"] = sum(episode.frames for episode in episodes)
    return episodes


def interval_diagnostics(episode, start, end):
    """Regroup the actual retained diagnostic frames, including sparse events."""
    require(len(episode.gripper_execution_frame_kinds) == episode.frames,
            f"{episode.record}: missing validated per-frame diagnostic classifications")
    events = {}
    provenance = episode.provenance
    for index in range(start, end):
        kind = episode.gripper_execution_frame_kinds[index]
        if kind is None:
            continue
        accepted = int(provenance["gripper_accepted_sequence"][index])
        stamp = float(episode.timestamps[index])
        key = (kind, accepted)
        diagnostic = events.setdefault(key, {
            "kind": kind, "accepted_sequence": accepted,
            "result_status": int(provenance["gripper_target_result_status"][index]),
            "result_success": bool(provenance["gripper_target_result_success"][index]),
            "result_error": str(provenance["gripper_target_result_error"][index]),
            "first_frame_index": index - start, "last_frame_index": index - start, "frame_count": 0,
            "original_first_frame_index": index, "original_last_frame_index": index,
            "first_frame_timestamp": stamp, "last_frame_timestamp": stamp,
        })
        diagnostic.update(last_frame_index=index - start, original_last_frame_index=index,
                          frame_count=diagnostic["frame_count"] + 1,
                          last_frame_timestamp=stamp)
    return list(events.values())


def apply_trim_plan(episodes, args, report):
    """Create interval views only after every selected raw episode was validated.

    Source paths, videos, original metadata and their hashes are never rewritten.
    The official writer will still decode each full source video to EOF, while
    emitting only the reviewed interval. No source JPEG directory is copied.
    """
    if not getattr(args, "trim_plan", None):
        return episodes
    require(not report.get("errors"), "Cannot trim before complete raw validation succeeds")
    plan_path = args.trim_plan.expanduser().resolve()
    plan_signature = fingerprint(plan_path)
    plan_bytes = plan_path.read_bytes()
    require(hashlib.sha256(plan_bytes).hexdigest() == plan_signature["sha256"],
            "Trim plan changed while reading")
    plan = json.loads(plan_bytes)
    require(isinstance(plan, dict) and plan.get("schema") == "fr3_tail_trim_v1",
            "Trim plan must use schema fr3_tail_trim_v1")
    require(type(plan.get("fps")) is int and plan["fps"] == args.fps == 15,
            "Trim plan and capture must both use exactly 15 FPS")
    entries = plan.get("episodes")
    require(isinstance(entries, list), "Trim plan episodes must be a list")
    require(all(isinstance(entry, dict) and isinstance(entry.get("record"), str) for entry in entries),
            "Every trim plan episode must identify its record")
    names = [entry["record"] for entry in entries]
    require(len(set(names)) == len(names), "Trim plan contains duplicate records")
    require(set(names) == {episode.record for episode in episodes},
            "Trim plan must cover every selected record exactly once, with no additional records")
    require(episodes, "Trim plan selected no eligible episodes")
    by_name = {entry["record"]: entry for entry in entries}
    views = []
    for episode in episodes:
        require(episode.trim is None and episode.source_frame_count is None,
                f"{episode.record}: refusing to trim an already-derived view")
        entry = by_name[episode.record]
        original_count = integer(entry.get("original_frames"), "original_frames", args.min_frames)
        require(original_count == episode.frames,
                f"{episode.record}: trim plan original_frames differs from validated raw count")
        start = integer(entry.get("keep_start_inclusive", 0), "keep_start_inclusive", 0)
        end = integer(entry.get("keep_end_exclusive"), "keep_end_exclusive", 0)
        require(0 <= start < end <= original_count,
                f"{episode.record}: require 0 <= keep_start_inclusive < keep_end_exclusive <= original_frames")
        count = end - start
        require(count >= args.min_frames,
                f"{episode.record}: retained interval is shorter than min_frames")
        reason, review = entry.get("reason"), entry.get("review")
        require(isinstance(reason, str) and bool(reason.strip()),
                f"{episode.record}: trim plan requires a nonempty reason")
        require((isinstance(review, str) and bool(review.strip())) or (isinstance(review, dict) and bool(review)),
                f"{episode.record}: trim plan requires its review evidence")
        sources = entry.get("source_files")
        require(isinstance(sources, dict) and set(sources) == set(episode.fingerprints),
                f"{episode.record}: trim plan must contain the complete original source_files dictionary")
        for relative, actual in episode.fingerprints.items():
            expected = sources[relative]
            require(isinstance(expected, dict) and set(expected) == {"size", "mtime_ns", "sha256"},
                    f"{episode.record}/{relative}: source fingerprint must contain size, mtime_ns, sha256")
            integer(expected["size"], "source fingerprint size", 0)
            integer(expected["mtime_ns"], "source fingerprint mtime_ns", 0)
            require(isinstance(expected["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]),
                    f"{episode.record}/{relative}: invalid source SHA256")
            require(expected["sha256"] == actual["sha256"] and expected["size"] == actual["size"],
                    f"{episode.record}/{relative}: source fingerprint differs from trim plan")
        timestamps = episode.timestamps[start:end].copy()
        dt = np.diff(timestamps)
        observed_fps = (count - 1) / float(timestamps[-1] - timestamps[0])
        require(np.all(dt > 0) and abs(observed_fps / args.fps - 1) <= args.fps_tolerance,
                f"{episode.record}: retained interval has invalid capture timing/FPS")
        require(float(dt.max()) <= args.max_gap_factor / args.fps,
                f"{episode.record}: retained interval has a capture gap")
        provenance = {}
        for key, values in episode.provenance.items():
            array = np.asarray(values)
            require(array.ndim > 0 and len(array) == original_count,
                    f"{episode.record}: provenance {key} is not an original per-frame array")
            provenance[key] = array[start:end].copy()
        provenance["original_frame_index"] = np.arange(start, end, dtype=np.int64)
        diagnostics = interval_diagnostics(episode, start, end)
        trim = {
            "operation": "retain_reviewed_contiguous_interval_remove_stationary_edges",
            "original_frames": original_count,
            "keep_start_inclusive": start,
            "keep_end_exclusive": end,
            "removed_frames": original_count - count,
            "removed_initial_frames": start,
            "removed_tail_frames": original_count - end,
            "retained_capture_start_timestamp": float(timestamps[0]),
            "retained_capture_end_timestamp": float(timestamps[-1]),
            "reason": reason,
            "review": copy.deepcopy(review),
            "plan_sha256": plan_signature["sha256"],
            "source_files_scope": "complete_original_raw_bytes_before_interval_trimming",
            "original_validation": {
                "timing": copy.deepcopy(episode.timing),
                "warnings": list(episode.warnings),
                "gripper_execution_diagnostics": copy.deepcopy(episode.gripper_execution_diagnostics),
            },
        }
        metadata = copy.deepcopy(episode.metadata)
        metadata.update(frame_count=count, action_count=count,
                        gripper_execution_diagnostics=copy.deepcopy(diagnostics),
                        derivation=copy.deepcopy(trim))
        exposure = provenance["camera_exposure_timestamp"]
        bases = provenance["image_time_basis"]
        timing = {"observed_fps": float(observed_fps),
                  "duration_s": float(timestamps[-1] - timestamps[0]),
                  "max_frame_gap_s": float(dt.max()),
                  "max_camera_skew_s": float(np.nanmax(np.ptp(exposure, axis=1)))
                  if "earliest_camera_exposure_system_epoch" in bases else None}
        warnings = [warning for warning in episode.warnings if not warning.startswith("Gripper diagnostic ")]
        warnings.extend(diagnostic_warnings(diagnostics))
        view = Episode(
            episode.record, episode.path, metadata, episode.state[start:end].copy(), episode.action[start:end].copy(),
            timestamps, list(episode.video_paths), episode.image_shape, provenance,
            copy.deepcopy(episode.fingerprints), timing, warnings, diagnostics,
            copy.deepcopy(episode.gripper_control_profile),
            source_frame_count=original_count, original_metadata=copy.deepcopy(episode.metadata),
            trim=trim, gripper_execution_frame_kinds=episode.gripper_execution_frame_kinds[start:end],
        )
        views.append(view)
    report["total_original_frames"] = sum(episode.frames for episode in episodes)
    report["total_frames"] = sum(episode.frames for episode in views)
    report["removed_frames"] = report["total_original_frames"] - report["total_frames"]
    report["removed_initial_frames"] = sum(view.trim["removed_initial_frames"] for view in views)
    report["removed_tail_frames"] = sum(view.trim["removed_tail_frames"] for view in views)
    report["trim_plan"] = {
        "schema": "fr3_tail_trim_v1", "source_path": str(plan_path),
        "sha256": plan_signature["sha256"], "output_copy": "meta/tail_trim_plan.json",
        "source_fingerprint_match": "sha256_and_size; source mtime_ns retained as provenance",
        "raw_validation": "every selected episode validated in full before applying intervals",
    }
    for selected, view in zip(report["selected"], views, strict=True):
        require(selected["record"] == view.record, "Selection/report order changed while trimming")
        selected.update(frames=view.frames, original_frames=view.source_frame_count, timing=view.timing,
                        warnings=view.warnings, gripper_execution_diagnostics=view.gripper_execution_diagnostics,
                        trim=copy.deepcopy(view.trim))
    # Private byte snapshot is copied exactly into the new output. It never
    # becomes an instruction or a write to the plan/raw input paths.
    args._trim_plan_bytes = plan_bytes
    args._trim_plan_context = report["trim_plan"]
    return views


def split_episodes(count, ratio, seed):
    if count < 2 or ratio == 0:
        return list(range(count)), []
    validation_count = min(count - 1, max(1, round(count * ratio)))
    order = np.random.default_rng(seed).permutation(count)
    validation = sorted(int(i) for i in order[:validation_count])
    return [i for i in range(count) if i not in validation], validation


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temp, path)


def publish_directory(source, destination):
    """Atomically publish on Linux without replacing even an empty directory."""
    if sys.platform.startswith("linux"):
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        require(rename is not None, "Atomic no-overwrite publication requires Linux renameat2")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        # AT_FDCWD = -100; RENAME_NOREPLACE = 1.
        if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
    elif os.name == "nt":
        os.rename(source, destination)  # Windows rename refuses existing destinations.
    else:
        raise ConversionError("Atomic no-overwrite publication is supported on Linux/Windows")


def make_parser():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Validation: numpy opencv-python h5py. Export profile: lerobot==0.6.1 (Python 3.12+, CPU PyTorch/PyAV). Never installs packages or uploads data.")
    parser.add_argument("--input-dir", type=Path, default=base / "data/raw", help="Directory containing record_* or one episode; default is this repository's data/raw")
    parser.add_argument("--output-dir", type=Path, default=base / "data/lerobot/fr3_avocado_v30", help="New output directory; existing paths are refused")
    parser.add_argument("--repo-id", default="local/fr3_avocado_pi05", help="LeRobot dataset identifier; local only, no upload")
    parser.add_argument("--dry-run", action="store_true", help="Validate sources and show selection; does not import LeRobot or create output")
    parser.add_argument("--report", type=Path, help="Optional JSON report outside raw input and output directories")
    parser.add_argument("--episodes", nargs="+", help="Optional source directory names, e.g. record_002 record_003")
    parser.add_argument("--trim-plan", type=Path,
                        help="Optional reviewed fr3_tail_trim_v1 plan; validates complete raw episodes first, then exports each [keep_start_inclusive, keep_end_exclusive) interval")
    parser.add_argument("--include-outcomes", nargs="+", default=["success"], choices=["success", "failure", "unknown", "aborted"], help="Default: success only; other labels require explicit opt-in")
    parser.add_argument("--fps", type=int, default=15, help="Must equal capture record_fps; no resampling")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--fps-tolerance", type=float, default=0.10, help="Allowed relative error of mean source FPS (default 0.10)")
    parser.add_argument("--max-gap-factor", type=float, default=3.0, help="Maximum capture gap in frame periods (default 3)")
    parser.add_argument("--max-camera-skew", type=float, default=0.10, help="Maximum exposure skew between two images, seconds")
    parser.add_argument("--allow-host-timestamps", action="store_true", help="Explicitly accept documented host-receipt fallback instead of exposure causality")
    parser.add_argument("--camera-serials", nargs=2, default=DEFAULT_SERIALS, metavar=("CAMERA_1", "CAMERA_2"))
    parser.add_argument("--camera-keys", nargs=2, default=DEFAULT_KEYS, metavar=("KEY_1", "KEY_2"), help="Two stable observation.images.* keys; no implicit wrist/exterior role")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Held-out whole episodes; default 0 uses all episodes for this task's training and statistics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-writer-threads", type=int, default=2)
    parser.add_argument("--encoder-threads", type=int, default=2)
    return parser


def load_validation_dependencies():
    global np, cv2, h5py
    try:
        import numpy as np
        import cv2
        import h5py
    except ImportError as exc:
        raise ConversionError("Validation needs numpy, opencv-python and h5py in the selected Python environment") from exc


def iter_source_video_frames(episode, camera_keys):
    """Decode all original frames, yielding aligned RGB only for the interval.

    Yielded indices address the derived state/action arrays. Source indices are
    retained separately in provenance, and all original frames plus EOF are
    checked even when the discarded tail is long.
    """
    source_frames = episode.source_frame_count if episode.source_frame_count is not None else episode.frames
    start = episode.trim["keep_start_inclusive"] if episode.trim is not None else 0
    end = episode.trim["keep_end_exclusive"] if episode.trim is not None else episode.frames
    require(0 <= start < end <= source_frames and end - start == episode.frames,
            "Retained interval does not match its validated source video")
    caps = [cv2.VideoCapture(str(path)) for path in episode.video_paths]
    try:
        require(all(cap.isOpened() for cap in caps), "Source video became unreadable after validation")
        for source_index in range(source_frames):
            images = []
            for cap in caps:
                ok, bgr = cap.read()
                require(ok and bgr.shape == episode.image_shape, "Source video changed during export")
                images.append(bgr)
            if not start <= source_index < end:
                continue
            yield source_index - start, {
                key: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                for key, bgr in zip(camera_keys, images, strict=True)
            }
        require(not any(cap.read()[0] for cap in caps), "Source video gained frames during export")
    finally:
        for cap in caps:
            cap.release()


def export_dataset(episodes, args, report):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        version = importlib.metadata.version("lerobot")
    except Exception as exc:
        raise ConversionError("Export needs a separate LeRobot environment (tested: lerobot==0.6.1, "
                              "Python 3.12+, datasets==4.8.5, pyarrow==26.0.0, av==15.1.0). "
                              "Use --dry-run in the collection environment; no packages were installed. "
                              f"Import failed: {exc}") from exc
    params = inspect.signature(LeRobotDataset.create).parameters
    require(all(key in params for key in ("repo_id", "fps", "features", "root", "use_videos", "batch_encoding_size"))
            and all(callable(getattr(LeRobotDataset, key, None)) for key in ("add_frame", "save_episode", "finalize")),
            f"Unsupported LeRobot {version} writer API; current tested profile is 0.6.1")
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES},
        **{key: {"dtype": "video", "shape": episodes[0].image_shape, "names": ["height", "width", "channels"]}
           for key in args.camera_keys},
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir.parent / ("." + args.output_dir.name + ".conversion.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ConversionError(f"Another conversion may be using this output: {lock_path}") from exc
    workspace = None
    writer = None
    finalized = False
    try:
        with os.fdopen(lock_fd, "w") as handle:
            handle.write(f"pid={os.getpid()}\nrepo_id={args.repo_id}\n")
        require(not args.output_dir.exists() and not args.output_dir.is_symlink(), "Output path appeared during validation; refusing overwrite")
        workspace = Path(tempfile.mkdtemp(prefix="." + args.output_dir.name + ".converting-", dir=args.output_dir.parent))
        stage = workspace / "dataset"  # The official create() requires a nonexistent root.
        options = {"repo_id": args.repo_id, "fps": args.fps, "features": features, "root": stage,
                   "robot_type": "franka_fr3", "use_videos": True,
                   "batch_encoding_size": 1, "video_backend": "pyav",
                   "image_writer_threads": args.image_writer_threads,
                   "image_writer_processes": 0, "encoder_threads": args.encoder_threads,
                   "streaming_encoding": False}
        if "vcodec" in params:
            options["vcodec"] = "h264"
        elif "rgb_encoder" in params:
            # Newer LeRobot releases expose codec settings through a config.
            from lerobot.configs import RGBEncoderConfig
            options["rgb_encoder"] = RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p")
        else:
            raise ConversionError(f"Unsupported LeRobot {version} video encoder interface")
        writer = LeRobotDataset.create(**{key: value for key, value in options.items() if key in params})
        info = read_json(stage / "meta/info.json")
        require(str(info.get("codebase_version", "")).startswith("v3."), "Installed LeRobot did not create Dataset v3")
        for episode_index, episode in enumerate(episodes):
            print(f"Exporting {episode.record} -> episode {episode_index}: {episode.frames} frames ...", flush=True)
            with closing(iter_source_video_frames(episode, args.camera_keys)) as frames:
                for index, images in frames:
                    frame = {"observation.state": episode.state[index].copy(),
                             "action": episode.action[index].copy(), "task": episode.metadata["task"],
                             **images}
                    # Official API assigns timestamp=i/fps and default indices.
                    # Passing epoch timestamps here would create an invalid timeline.
                    writer.add_frame(frame)
            writer.save_episode(parallel_encoding=False)
        writer.finalize()
        finalized = True
        stop_images = getattr(writer, "stop_image_writer", None)
        if callable(stop_images):
            stop_images()

        train, validation = report["train_episode_indices"], report["validation_episode_indices"]
        stats_path = stage / "meta/stats.json"
        stats = read_json(stats_path)
        for key, attribute in (("observation.state", "state"), ("action", "action")):
            values = np.concatenate([getattr(episodes[index], attribute) for index in train]).astype(np.float64)
            # Exact global quantiles over training frames, including constant axes.
            stats[key] = {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
                          "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0, ddof=0).tolist(),
                          "count": [len(values)],
                          **{name: np.quantile(values, quantile, axis=0).tolist() for name, quantile in
                             (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99))}}
        write_json(stats_path, stats)
        split_info = {"seed": args.seed, "requested_validation_ratio": args.val_ratio,
                      "train_episode_indices": train, "validation_episode_indices": validation,
                      "train_frames": sum(episodes[index].frames for index in train),
                      "validation_frames": sum(episodes[index].frames for index in validation),
                      "usage": "Explicitly pass these episode indices to LeRobotDataset(episodes=...). "
                               "This sidecar does not change the default all-episodes loader or info.json splits.",
                      "state_action_stats_scope": "train_episode_indices",
                      "image_stats_scope": "official_writer_all_selected_episodes"}
        write_json(stage / "meta/episode_splits.json", split_info)
        provenance_paths = []
        for index, episode in enumerate(episodes):
            relative = Path("meta/source_timestamps") / f"episode_{index:06d}.npz"
            (stage / relative).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(stage / relative, **episode.provenance)
            provenance_paths.append(relative.as_posix())

        print("Verifying official LeRobot reader, scalar rows and both encoded video streams ...", flush=True)
        restored = LeRobotDataset(repo_id=args.repo_id, root=stage, video_backend="pyav")
        require(len(restored) == report["total_frames"], "LeRobot readback frame count differs")
        stored = restored.hf_dataset.select_columns(["observation.state", "action", "timestamp", "episode_index"]).with_format("numpy")[:]
        equal_values(stored["observation.state"], np.concatenate([episode.state for episode in episodes]), "LeRobot readback state")
        equal_values(stored["action"], np.concatenate([episode.action for episode in episodes]), "LeRobot readback action")
        expected_indices = np.concatenate([np.full(episode.frames, index) for index, episode in enumerate(episodes)])
        require(np.array_equal(np.asarray(stored["episode_index"]).reshape(-1), expected_indices), "LeRobot episode boundaries differ")
        expected_time = np.concatenate([np.arange(episode.frames) / args.fps for episode in episodes])
        require(np.allclose(np.asarray(stored["timestamp"]).reshape(-1), expected_time, rtol=1e-6, atol=1e-4),
                "LeRobot time axis is not the per-episode i/fps grid")
        offset = 0
        for episode in episodes:
            for row_index in sorted({offset, offset + episode.frames - 1}):
                sample = restored[row_index]  # Actually decodes both video keys through PyAV.
                for key in args.camera_keys:
                    image = np.asarray(sample[key])
                    require(image.shape == (3, *episode.image_shape[:2]) and np.isfinite(image).all(),
                            f"LeRobot cannot decode {key} at row {row_index}")
            offset += episode.frames
        info = read_json(stage / "meta/info.json")
        require(info.get("total_episodes") == len(episodes) and info.get("total_frames") == report["total_frames"],
                "LeRobot metadata counts differ")
        for key in args.camera_keys:
            encoded = sorted((stage / "videos" / key).rglob("*.mp4"))
            require(encoded, f"No encoded videos for {key}")
            count = 0
            for path in encoded:
                cap = cv2.VideoCapture(str(path))
                try:
                    require(cap.isOpened(), f"Cannot decode output video {path}")
                    while True:
                        ok, image = cap.read()
                        if not ok:
                            break
                        require(image.shape == episodes[0].image_shape, "Encoded image dimensions differ")
                        count += 1
                finally:
                    cap.release()
            require(count == report["total_frames"], f"Encoded video frame count differs for {key}")
        for episode in episodes:
            for relative, signature in episode.fingerprints.items():
                source_path = episode.path / relative
                stat = source_path.stat()
                require((stat.st_size, stat.st_mtime_ns) == (signature["size"], signature["mtime_ns"]),
                        f"Source changed during conversion: {episode.record}/{relative}")
                if episode.trim is not None:
                    require(fingerprint(source_path)["sha256"] == signature["sha256"],
                            f"Source bytes changed after the reviewed trim plan: {episode.record}/{relative}")
        if getattr(args, "trim_plan", None):
            context = args._trim_plan_context
            require(fingerprint(args.trim_plan.expanduser().resolve())["sha256"] == context["sha256"],
                    "Trim plan changed during export")
            (stage / "meta/tail_trim_plan.json").write_bytes(args._trim_plan_bytes)
        manifest = {"conversion_schema_version": 1, "lerobot_version": version,
                    "dataset_format": info["codebase_version"], "repo_id": args.repo_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "converter_sha256": fingerprint(Path(__file__).resolve())["sha256"],
                    "capture_contract": CONTRACT, "fps": args.fps, "image_shape_hwc": list(episodes[0].image_shape),
                    "observation_state_features": STATE_NAMES, "action_features": ACTION_NAMES,
                    "feature_units": UNITS, "action_semantics": ACTION_SEMANTICS,
                    "gripper_execution_policy": GRIPPER_EXECUTION_POLICY,
                    "gripper_control_profiles": [json.loads(value) for value in sorted({json.dumps(episode.gripper_control_profile, sort_keys=True) for episode in episodes})],
                    "gripper_execution_interpretation": "Execution outcomes remain raw diagnostics. Accepted Move kFail has undetermined cause; an operator_open canceled Grasp records explicit release provenance. Neither establishes contact, grasp or task success. Episode outcome remains the operator label.",
                    "camera_mapping": [{"camera_index": i + 1, "serial": serial, "feature": key}
                                       for i, (serial, key) in enumerate(zip(args.camera_serials, args.camera_keys, strict=True))],
                    "timestamp_semantics": "Official per-episode i/fps grid; original wall/source/exposure times in source_timestamps/*.npz; no action interpolation or shifting",
                    "split_file": "meta/episode_splits.json", "stats_scope": split_info,
                    "sources": [{"episode_index": index, "record": episode.record, "source_path": str(episode.path),
                                 "task": episode.metadata["task"], "outcome": episode.metadata["outcome"],
                                 "frames": episode.frames, "metadata": episode.metadata,
                                 "gripper_execution_diagnostics": episode.gripper_execution_diagnostics,
                                 "gripper_control_profile": episode.gripper_control_profile,
                                 "warnings": episode.warnings,
                                 "source_files": episode.fingerprints, "source_timestamps_file": provenance_paths[index],
                                 **({"metadata_scope": "retained_contiguous_interval",
                                     "source_files_scope": "complete_original_raw_bytes_before_interval_trimming",
                                     "source_frame_count": episode.source_frame_count,
                                     "original_metadata": episode.original_metadata,
                                     "retained_frame_range": {"start_inclusive": episode.trim["keep_start_inclusive"],
                                                              "end_exclusive": episode.trim["keep_end_exclusive"]},
                                     "trim": episode.trim} if episode.trim is not None else {})}
                                for index, episode in enumerate(episodes)],
                    "validation": {**report, "status": "complete"},
                    "verification": {"scalar_rows": "all", "video_frames": "all", "official_reader_images": "episode boundaries"}}
        if getattr(args, "trim_plan", None):
            manifest["cleaning"] = {**args._trim_plan_context,
                                    "original_frames": report["total_original_frames"],
                                    "retained_frames": report["total_frames"],
                                    "removed_frames": report["removed_frames"],
                                    "removed_initial_frames": report["removed_initial_frames"],
                                    "removed_tail_frames": report["removed_tail_frames"],
                                    "full_original_video_decode_during_export": True,
                                    "original_source_sha256_rechecked_before_publication": True,
                                    "state_action_values": "unchanged retained intervals; no interpolation, substitution or shifting"}
        write_json(stage / "meta/conversion_manifest.json", manifest)
        # Official metadata keeps relative data/video paths, so the finalized tree
        # can be moved as a unit. Publication never merges with an existing dataset.
        require(not args.output_dir.exists() and not args.output_dir.is_symlink(), "Output appeared before publication; refusing overwrite")
        publish_directory(stage, args.output_dir)
        report["lerobot_version"] = version
        report["dataset_format"] = info["codebase_version"]
        report["manifest"] = str(args.output_dir / "meta/conversion_manifest.json")
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise ConversionError(f"Export failed; no completed output published: {exc}") from exc
    finally:
        if writer is not None:
            if not finalized:
                try:
                    writer.finalize()
                except Exception:
                    pass
            try:
                stop_images = getattr(writer, "stop_image_writer", None)
                if callable(stop_images):
                    stop_images()
            except Exception:
                pass
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)
        lock_path.unlink(missing_ok=True)


def main(argv=None):
    args = make_parser().parse_args(argv)
    report = {"converter_schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "input_dir": str(args.input_dir.expanduser().resolve()),
              "output_dir": str(args.output_dir.expanduser().resolve()), "repo_id": args.repo_id,
              "fps": args.fps, "selected": [], "skipped": [], "errors": [], "total_frames": 0,
              "dry_run": args.dry_run, "status": "checking"}
    report_path_validated = False
    try:
        args.input_dir = args.input_dir.expanduser().resolve()
        args.output_dir = args.output_dir.expanduser().absolute()
        if args.report:
            # Normalize before any other validation can fail. Error reporting
            # must never turn an early error into a write inside raw input.
            args.report = args.report.expanduser().resolve()
            resolved_output = args.output_dir.resolve()
            require(not args.report.is_relative_to(args.input_dir) and not args.report.is_relative_to(resolved_output),
                    "--report must be outside raw input and output")
            if args.report.exists():
                previous = read_json(args.report)
                require(isinstance(previous, dict) and previous.get("converter_schema_version") == 1,
                        "--report exists and is not a converter report; refusing overwrite")
            report_path_validated = True
        require(not args.output_dir.exists() and not args.output_dir.is_symlink(), f"Output path already exists: {args.output_dir}")
        args.output_dir = args.output_dir.resolve()
        require(not args.output_dir.is_relative_to(args.input_dir) and not args.input_dir.is_relative_to(args.output_dir),
                "Output must be separate from raw input")
        require(re.fullmatch(r"[^/\s]+/[^/\s]+", args.repo_id), "--repo-id must be namespace/name")
        require(args.fps > 0 and args.min_frames >= 2 and 0 <= args.val_ratio < 1, "Invalid FPS/min-frames/val-ratio")
        require(0 <= args.fps_tolerance < 1 and args.max_gap_factor >= 1 and args.max_camera_skew > 0,
                "Invalid timing tolerances")
        require(args.image_writer_threads >= 0 and args.encoder_threads > 0, "Invalid writer thread count")
        require(len(set(args.camera_serials)) == 2 and len(set(args.camera_keys)) == 2
                and all(re.fullmatch(r"observation\.images\.[A-Za-z0-9_]+", key) for key in args.camera_keys),
                "Camera serials/keys must be distinct; keys must start observation.images.")
        load_validation_dependencies()
        episodes = discover(args, report)
        require(not report["errors"], "One or more selected episodes failed validation")
        episodes = apply_trim_plan(episodes, args, report)
        train, validation = split_episodes(len(episodes), args.val_ratio, args.seed)
        report["train_episode_indices"], report["validation_episode_indices"] = train, validation
        if args.dry_run:
            report["status"] = "validated" if episodes else "no_eligible_episodes"
        else:
            require(episodes, "No eligible episodes. Record schema-2 data and label successful demonstrations first; use --dry-run for details")
            export_dataset(episodes, args, report)
            report["status"] = "complete"
        result = 0 if episodes else 2
        if not episodes:
            print("ERROR: No eligible episodes; validation is not ready for conversion", file=sys.stderr)
    except (ConversionError, OSError, ValueError, TypeError, ImportError) as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        result = 2
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        print("Conversion interrupted; no completed dataset published", file=sys.stderr)
        result = 130
    if args.report and report_path_validated:
        # An explicitly requested report is the only write in dry-run mode.
        try:
            require(not args.report.resolve().is_relative_to(args.input_dir) and not args.report.resolve().is_relative_to(args.output_dir),
                    "Refusing a report inside input/output")
            write_json(args.report, report)
        except (ConversionError, OSError, TypeError) as exc:
            print(f"Cannot write report: {exc}", file=sys.stderr)
            result = 2
    print(f"Selected {len(report['selected'])} episodes / {report['total_frames']} frames; skipped {len(report['skipped'])}; invalid {len(report['errors'])}")
    for item in report["skipped"]:
        print(f"  SKIP {item['record']}: {item['reason']}")
    for item in report["errors"]:
        print(f"  INVALID {item['record']}: {item['error']}", file=sys.stderr)
    if report["status"] == "complete":
        print(f"LeRobot v3 dataset saved to {args.output_dir}")
    return result


if __name__ == "__main__":
    sys.exit(main())
