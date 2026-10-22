#!/usr/bin/env python3
"""
Remote ROS2 Data Bridge
Subscribes to ROS2 topics and streams data via TCP to local machine
"""

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, String
import copy
import math
import re
import socket
import json
import threading
import time
from datetime import datetime

FR3_JOINT_NAMES = [f'fr3_joint{i}' for i in range(1, 8)]
FINGER_JOINT_NAMES = ['fr3_finger_joint1', 'fr3_finger_joint2']
ARM_NAME = re.compile(r'(?:^|[/_])fr3_joint([1-7])$')
FINGER_NAME = re.compile(r'(?:^|[/_])finger_joint([12])$')


def _finite_vector(value, length):
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                    and math.isfinite(item) for item in value))


def _positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _reject_json_constant(value):
    raise ValueError(f'Non-finite JSON value: {value}')


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


class ROS2DataBridge(Node):
    def __init__(self, port=9999, stream_hz=50.0):
        super().__init__('ros2_data_bridge')

        # TCP server setup
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.running = True
        self.data_lock = threading.Lock()
        self.client_lock = threading.Lock()
        self.stop_event = threading.Event()
        if not math.isfinite(stream_hz) or not 1 <= stream_hz <= 100:
            raise ValueError('stream_hz must be between 1 and 100')
        self.stream_hz = float(stream_hz)
        self.data_generation = 0
        self.last_sent_generation = -1

        # Data buffers
        self.latest_data = {
            'timestamp': None,
            'gello_joints': None,
            'franka_joints': None,
            'gripper_joints': None,
            'gripper_command': None,
            'end_effector_pose': None,
            'arm_command': None,
            'gripper_target': None,
            'source_timestamps': {},
        }

        # Create subscribers
        self.gello_sub = self.create_subscription(
            JointState,
            '/gello/joint_states',
            self.gello_callback,
            10
        )

        self.franka_sub = self.create_subscription(
            JointState,
            '/franka/joint_states',
            self.franka_callback,
            10
        )

        self.gripper_sub = self.create_subscription(
            JointState,
            '/gripper_joint_states',
            lambda msg: self.gripper_callback(msg, '/gripper_joint_states'),
            10
        )
        self.gripper_ns_sub = self.create_subscription(
            JointState,
            '/gripper/gripper_joint_states',
            lambda msg: self.gripper_callback(msg, '/gripper/gripper_joint_states'),
            10
        )
        self.gripper_ns_joint_sub = self.create_subscription(
            JointState,
            '/gripper/joint_states',
            lambda msg: self.gripper_callback(msg, '/gripper/joint_states'),
            10
        )
        self.gripper_franka_sub = self.create_subscription(
            JointState,
            '/franka_gripper/joint_states',
            lambda msg: self.gripper_callback(msg, '/franka_gripper/joint_states'),
            10
        )
        self.gripper_fallback_sub = self.create_subscription(
            JointState,
            '/joint_states',
            lambda msg: self.gripper_callback(msg, '/joint_states'),
            10
        )
        self.gripper_command_sub = self.create_subscription(
            Float32,
            '/gripper/gripper_client/target_gripper_width_percent',
            lambda msg: self.gripper_command_callback(
                msg, '/gripper/gripper_client/target_gripper_width_percent'),
            10
        )
        self.gripper_command_fallback_sub = self.create_subscription(
            Float32,
            '/gripper_client/target_gripper_width_percent',
            lambda msg: self.gripper_command_callback(
                msg, '/gripper_client/target_gripper_width_percent'),
            10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
            self.pose_callback,
            10
        )
        self.arm_command_sub = self.create_subscription(
            String, '/gello_capture/arm_command', self.arm_command_callback, 10)
        self.gripper_target_sub = self.create_subscription(
            String, '/gello_capture/gripper_command', self.gripper_target_callback, 10)

        self.get_logger().info(f'ROS2 Data Bridge initialized; latest snapshots at {self.stream_hz:g} Hz')

        # Start TCP server in separate thread
        self.server_thread = threading.Thread(target=self.run_server, daemon=True)
        self.server_thread.start()
        self.sender_thread = threading.Thread(target=self.stream_data, daemon=True)
        self.sender_thread.start()

    @staticmethod
    def _header_stamp_ns(msg):
        stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
        if stamp is None:
            return None
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        return value if value > 0 else None

    def _store(self, key, value, topic, source_stamp_ns, received_at=None):
        received_at = time.time() if received_at is None else received_at
        with self.data_lock:
            self.latest_data[key] = value
            self.latest_data['source_timestamps'][key] = {
                'source_stamp_ns': source_stamp_ns,
                'host_received_ts': received_at,
                'topic': topic,
            }
            # Packet transport time is separate from every source's acquisition time.
            self.latest_data['timestamp'] = received_at
            self.data_generation += 1
        # Never serialize or write to TCP in a ROS subscription callback. Joint
        # state/pose topics may arrive at 1 kHz; resending the entire cached JSON
        # for each one starves the receiver and delays all independent sources.

    @staticmethod
    def _joint_block(msg, indices, names):
        positions = [msg.position[index] for index in indices
                     if index < len(msg.position)]
        if not _finite_vector(positions, len(indices)):
            raise ValueError('Missing or non-finite measured joint positions')
        block = {'position': positions, 'names': names,
                 'source_names': [msg.name[index] for index in indices]}
        for field in ('velocity', 'effort'):
            values = getattr(msg, field)
            selected = [values[index] for index in indices if index < len(values)]
            block[field] = selected if _finite_vector(selected, len(indices)) else None
        return block

    @staticmethod
    def _indices_for_names(msg, pattern, count):
        found = {}
        for index, name in enumerate(msg.name):
            match = pattern.search(name)
            if match:
                number = int(match.group(1))
                if number in found:
                    raise ValueError('Duplicate measured joint name')
                found[number] = index
        if set(found) != set(range(1, count + 1)):
            raise ValueError('Message does not contain all required measured joints')
        return [found[number] for number in range(1, count + 1)]

    def _arm_joint_callback(self, msg, key, topic):
        received_at = time.time()
        try:
            indices = self._indices_for_names(msg, ARM_NAME, 7)
            block = self._joint_block(msg, indices, list(FR3_JOINT_NAMES))
        except (ValueError, TypeError, IndexError) as error:
            self.get_logger().warning(f'Invalid {key}: {error}')
            block = None
        self._store(key, block, topic, self._header_stamp_ns(msg), received_at)

    def gello_callback(self, msg):
        self._arm_joint_callback(msg, 'gello_joints', '/gello/joint_states')

    def franka_callback(self, msg):
        """Callback for Franka joint states"""
        self._arm_joint_callback(msg, 'franka_joints', '/franka/joint_states')

    def gripper_callback(self, msg, topic='/gripper_joint_states'):
        """Callback for Franka gripper joint states"""
        if not self._looks_like_gripper_joint_state(msg):
            return
        received_at = time.time()
        try:
            indices = self._indices_for_names(msg, FINGER_NAME, 2)
            block = self._joint_block(msg, indices, list(FINGER_JOINT_NAMES))
            block['position_unit'] = 'm'
            block['width_m'] = sum(block['position'])
        except (ValueError, TypeError, IndexError) as error:
            self.get_logger().warning(f'Invalid gripper measurement: {error}')
            block = None
        self._store('gripper_joints', block, topic, self._header_stamp_ns(msg), received_at)

    def gripper_command_callback(self, msg, topic='/gripper/gripper_client/target_gripper_width_percent'):
        """Callback for gripper command (percent open)."""
        try:
            value = float(msg.data)
            if not math.isfinite(value):
                return
        except (TypeError, ValueError):
            return
        # Float32 has no ROS header. Do not manufacture a source timestamp or
        # present this raw GELLO ratio as an accepted/measured gripper width.
        self._store('gripper_command', value, topic, None)

    @staticmethod
    def _looks_like_gripper_joint_state(msg):
        """Filter non-gripper joint state updates."""
        if not msg.name:
            return False
        return any(FINGER_NAME.search(name) for name in msg.name)

    @staticmethod
    def _validate_arm_command(payload):
        if not isinstance(payload, dict) or payload.get('schema_version') != 1:
            raise ValueError('Unsupported arm command schema')
        if not _positive_integer(payload.get('sequence')) or not _positive_integer(payload.get('stamp_ns')):
            raise ValueError('Arm command needs a positive sequence and source stamp')
        if not isinstance(payload.get('activation_id'), str) or not payload['activation_id']:
            raise ValueError('Arm command needs its activation ID')
        if payload.get('joint_names') != FR3_JOINT_NAMES:
            raise ValueError('Arm command joint order must be fr3_joint1 through fr3_joint7')
        for field in ('q_measured', 'dq_measured', 'q_target', 'tau_command',
                      'gello_q', 'robot_start_q', 'gello_start_q'):
            if not _finite_vector(payload.get(field), 7):
                raise ValueError(f'Invalid seven-joint arm command field {field}')
        if not isinstance(payload.get('relative_start'), bool):
            raise ValueError('Arm command relative_start must be boolean')

    @staticmethod
    def _validate_gripper_target(payload):
        if not isinstance(payload, dict):
            raise ValueError('Invalid gripper target object')
        _gripper_control_profile(payload)
        sequence = payload.get('sequence')
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise ValueError('Invalid gripper sequence')
        if not _positive_integer(payload.get('stamp_ns')):
            raise ValueError('Missing gripper heartbeat source stamp')
        if payload.get('status') not in ('pending', 'accepted', 'completed', 'not_reached', 'error'):
            raise ValueError('Invalid gripper goal status')
        if not isinstance(payload.get('valid'), bool):
            raise ValueError('Gripper validity must be boolean')
        for key in ('result_sequence', 'result_status'):
            value = payload.get(key)
            minimum = 1 if key == 'result_sequence' else 0
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < minimum):
                raise ValueError(f'Invalid nullable gripper {key}')
        if payload.get('result_success') is not None and not isinstance(payload['result_success'], bool):
            raise ValueError('Gripper result_success must be boolean or null')
        if payload.get('result_error') is not None and not isinstance(payload['result_error'], str):
            raise ValueError('Gripper result_error must be string or null')
        maximum = payload.get('max_width_m')
        if not _finite_vector([maximum], 1) or maximum <= 0:
            raise ValueError('Invalid actual gripper maximum width')
        if payload['valid']:
            width, speed = payload.get('target_width_m'), payload.get('speed_m_s')
            accepted_sequence = payload.get('accepted_sequence')
            if (not _positive_integer(payload.get('command_stamp_ns'))
                    or payload['command_stamp_ns'] > payload['stamp_ns']
                    or not _positive_integer(accepted_sequence)
                    or accepted_sequence > sequence
                    or not _finite_vector([width, speed], 2)
                    or width < 0 or width > maximum + 1e-9 or speed <= 0):
                raise ValueError('Invalid accepted gripper goal')
        elif (any(payload.get(key) is not None for key in
                  ('target_width_m', 'command_stamp_ns', 'accepted_sequence', 'speed_m_s'))
              or payload['status'] not in ('pending', 'error')):
            raise ValueError('Unaccepted gripper goal must not claim an accepted target')
        if payload['status'] == 'not_reached':
            if (not payload['valid'] or not _positive_integer(payload.get('result_sequence'))
                    or payload.get('result_sequence') != payload.get('accepted_sequence')
                    or not isinstance(payload.get('result_status'), int)
                    or isinstance(payload.get('result_status'), bool)
                    or payload.get('result_status') != 6 or payload.get('result_success') is not False
                    or payload.get('result_error') != '' or payload.get('error') not in (None, '')):
                raise ValueError('not_reached must retain the exact accepted ordinary unsuccessful Move result')

    def _telemetry_callback(self, msg, key, topic, validator):
        received_at = time.time()
        try:
            payload = json.loads(msg.data, parse_constant=_reject_json_constant)
            validator(payload)
        except (ValueError, TypeError, KeyError) as error:
            self.get_logger().warning(f'Invalid {key} telemetry: {error}')
            self._store(key, None, topic, None, received_at)
            return
        # Preserve the original JSON object, including every source/event stamp.
        self._store(key, payload, topic, payload['stamp_ns'], received_at)

    def arm_command_callback(self, msg):
        self._telemetry_callback(msg, 'arm_command', '/gello_capture/arm_command',
                                 self._validate_arm_command)

    def gripper_target_callback(self, msg):
        self._telemetry_callback(msg, 'gripper_target', '/gello_capture/gripper_command',
                                 self._validate_gripper_target)

    def pose_callback(self, msg):
        """Callback for end effector pose"""
        pose = {
            'position': {
                'x': msg.pose.position.x,
                'y': msg.pose.position.y,
                'z': msg.pose.position.z
            },
            'orientation': {
                'x': msg.pose.orientation.x,
                'y': msg.pose.orientation.y,
                'z': msg.pose.orientation.z,
                'w': msg.pose.orientation.w
            }
        }
        if not _finite_vector(list(pose['position'].values()) + list(pose['orientation'].values()), 7):
            pose = None
        self._store('end_effector_pose', pose,
                    '/franka_robot_state_broadcaster/current_pose', self._header_stamp_ns(msg))

    def send_data(self):
        """Send one latest snapshot; called only by the bounded sender thread."""
        with self.client_lock:
            client_socket = self.client_socket
        if client_socket is None:
            return

        try:
            # Prepare data packet
            with self.data_lock:
                generation = self.data_generation
                if generation == self.last_sent_generation or self.latest_data['timestamp'] is None:
                    return
                data_packet = copy.deepcopy(self.latest_data)
            data_packet['datetime'] = datetime.fromtimestamp(data_packet['timestamp']).isoformat()
            data_packet['transport'] = {'sent_at': time.time(), 'stream_hz': self.stream_hz,
                                        'source_generation': generation}

            # Serialize and send
            json_data = json.dumps(data_packet, allow_nan=False)
            message = json_data + '\n'  # Use newline as delimiter
            client_socket.sendall(message.encode('utf-8'))
            self.last_sent_generation = generation

        except OSError as e:
            self.get_logger().warn(f'Client disconnected: {e}')
            with self.client_lock:
                if self.client_socket is client_socket:
                    self.client_socket = None
                    self.client_address = None
            try:
                client_socket.close()
            except OSError:
                pass

    def stream_data(self):
        """Coalesce source updates at 50 Hz, without a queue of stale packets."""
        interval = 1.0 / self.stream_hz
        while not self.stop_event.is_set():
            started = time.monotonic()
            self.send_data()
            self.stop_event.wait(max(0.0, interval - (time.monotonic() - started)))

    def run_server(self):
        """Run TCP server to accept connections"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(0.5)

        self.get_logger().info(f'TCP server listening on port {self.port}')

        while self.running:
            try:
                self.get_logger().info('Waiting for client connection...')
                client_socket, client_address = self.server_socket.accept()
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
                client_socket.settimeout(0.1)
                with self.client_lock:
                    self.client_socket = client_socket
                    self.client_address = client_address
                    self.last_sent_generation = -1
                self.get_logger().info(f'Client connected from {client_address}')

                # Keep connection alive
                while self.running and self.client_socket is not None:
                    self.stop_event.wait(0.1)

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.get_logger().error(f'Server error: {e}')
                    time.sleep(1)

    def shutdown(self):
        """Cleanup on shutdown"""
        self.running = False
        self.stop_event.set()
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        for worker in (self.server_thread, self.sender_thread):
            if worker is not threading.current_thread():
                worker.join(timeout=1.0)


def main(args=None):
    rclpy.init(args=args)

    # Create bridge node
    bridge = ROS2DataBridge(port=int(os.environ.get("PI05_CAPTURE_PORT", "9999")))

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        bridge.shutdown()
        bridge.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
