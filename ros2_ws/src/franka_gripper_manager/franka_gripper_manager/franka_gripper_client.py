# Modified for Pi05-FR3-Gello: relocatable hardware-fault lock.
import os
import rclpy
import math
import time
import json
from pathlib import Path
from rclpy.node import Node
from franka_msgs.action import Move, Grasp
from franka_msgs.action import Homing
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
from std_msgs.msg import Float32, String

from .grasp_hold_policy import GraspHoldPolicy, measured_finger_width

DEFAULT_GRASP_ACTION_TOPIC = "franka_gripper/grasp"
DEFAULT_MOVE_ACTION_TOPIC = "franka_gripper/move"
DEFAULT_HOMING_ACTION_TOPIC = "franka_gripper/homing"
DEFAULT_JOINT_STATES_TOPIC = "franka_gripper/joint_states"
DEFAULT_GRIPPER_COMMAND_TOPIC = "gripper/gripper_client/target_gripper_width_percent"
DEFAULT_GRIPPER_MOVE_SPEED = 0.0
DEFAULT_COMMAND_DEADBAND_PERCENT = 0.02
DEFAULT_MAX_WIDTH_FALLBACK = 0.08


class GripperClient(Node):
    def __init__(self):
        for lock in (Path(os.environ.get('FR3_HARDWARE_FAULT_LOCK', '/workspace/repo/runtime/gripper_hardware_fault.lock')),):
            if lock.is_file():
                raise RuntimeError('Damaged gripper fingers: control disabled. Repair and verify finger load limits before re-enabling.')
        super().__init__("gripper_client")

        self.declare_parameter("control_mode", "disabled")
        self.declare_parameter("grasp_action_topic", DEFAULT_GRASP_ACTION_TOPIC)
        # No operational force default after the finger-damage incident.
        self.declare_parameter("grasp_force_n", 0.0)
        self.declare_parameter("grasp_force_limit_n", 0.0)
        self.declare_parameter("command_speed_limit_m_s", 0.0)
        self.declare_parameter("force_configuration_verified", False)
        self.declare_parameter("grasp_epsilon_inner_m", 0.005)
        self.declare_parameter("grasp_epsilon_outer_m", 0.08)
        self.declare_parameter("close_trigger_width_m", 0.03)
        self.declare_parameter("release_trigger_width_m", 0.06)
        self.declare_parameter("measurement_max_age_s", 0.5)
        self.declare_parameter("move_action_topic", DEFAULT_MOVE_ACTION_TOPIC)
        self.declare_parameter("homing_action_topic", DEFAULT_HOMING_ACTION_TOPIC)
        self.declare_parameter("gripper_command_topic", DEFAULT_GRIPPER_COMMAND_TOPIC)
        self.declare_parameter("joint_states_topic", DEFAULT_JOINT_STATES_TOPIC)
        self.declare_parameter("gripper_move_speed", DEFAULT_GRIPPER_MOVE_SPEED)
        self.declare_parameter("command_deadband_percent", DEFAULT_COMMAND_DEADBAND_PERCENT)
        self.declare_parameter("max_width_fallback", DEFAULT_MAX_WIDTH_FALLBACK)
        self.declare_parameter("invert_gripper_command", False)
        self.declare_parameter("home_on_startup", False)

        move_action_topic = (
            self.get_parameter("move_action_topic").get_parameter_value().string_value
        )
        homing_action_topic = (
            self.get_parameter("homing_action_topic").get_parameter_value().string_value
        )
        gripper_command_topic = (
            self.get_parameter("gripper_command_topic").get_parameter_value().string_value
        )
        joint_states_topic = (
            self.get_parameter("joint_states_topic").get_parameter_value().string_value
        )
        self._gripper_move_speed = (
            self.get_parameter("gripper_move_speed").get_parameter_value().double_value
        )
        self._command_deadband_percent = (
            self.get_parameter("command_deadband_percent").get_parameter_value().double_value
        )
        self._max_width_fallback = (
            self.get_parameter("max_width_fallback").get_parameter_value().double_value
        )
        self._invert_gripper_command = (
            self.get_parameter("invert_gripper_command").get_parameter_value().bool_value
        )
        self._home_on_startup = (
            self.get_parameter("home_on_startup").get_parameter_value().bool_value
        )

        self._control_mode = self.get_parameter("control_mode").value
        if self._control_mode == 'disabled':
            raise RuntimeError('Gripper control is disabled pending finger repair and a verified force configuration')
        if self._control_mode not in ('move', 'grasp_hold'):
            raise ValueError("control_mode must be move or grasp_hold")
        if (self._control_mode == 'grasp_hold'
                and self.get_parameter('force_configuration_verified').value is not True):
            raise RuntimeError('Actuator force support and installed finger load limits have not been verified')
        if self._control_mode == 'grasp_hold' and self._home_on_startup:
            raise ValueError('Automatic homing has no force limit and is not allowed in grasp_hold_v2')
        self._grasp_policy = None

        self._ACTION_SERVER_TIMEOUT = 10.0
        self._MIN_GRIPPER_WIDTH_PERCENT = 0.0
        self._MAX_GRIPPER_WIDTH_PERCENT = 1.0
        self._gripper_command_transmitted = True
        self._max_width = 0.0
        self._last_gripper_command_percent = None
        # Capture telemetry observes the existing action path. It never sends a
        # goal, changes a command limit, or unlocks the in-flight command gate.
        self._capture_sequence = 0
        self._capture_accepted_sequence = None
        self._capture_command_stamp_ns = None
        self._capture_target_width_m = None
        self._capture_speed_m_s = None
        self._capture_requested_width_m = None
        self._capture_requested_speed_m_s = None
        self._capture_status = 'pending'
        self._capture_error = None
        self._capture_valid = False
        self._capture_result_sequence = None
        self._capture_result_status = None
        self._capture_result_success = None
        self._capture_result_error = None
        self._capture_pending_goal = None

        if self._control_mode == 'grasp_hold':
            # Validate every physical limit before homing could send a goal.
            self._grasp_policy = self._create_grasp_policy(self._max_width_fallback)

        self.get_logger().info("Initializing gripper client...")
        if self._home_on_startup:
            self._home_gripper(homing_action_topic)
            if self._control_mode == 'move':
                self._get_max_gripper_width(joint_states_topic)
            else:
                # This policy uses the validated configured physical travel;
                # its feedback subscriber independently reads both fingers.
                self._max_width = self._max_width_fallback
        else:
            self._max_width = self._max_width_fallback
            self.get_logger().info(
                "Skipping gripper homing on startup. Using max width fallback: "
                f"{self._max_width}"
            )

        self._capture_publisher = self.create_publisher(
            String, '/gello_capture/gripper_command', 10)
        self._capture_timer = self.create_timer(0.1, self._publish_capture_telemetry)

        if self._control_mode == 'grasp_hold':
            if self._grasp_policy.max_width != self._max_width:
                self._grasp_policy = self._create_grasp_policy(self._max_width)
            self._measured_gripper_subscription = self.create_subscription(
                JointState, joint_states_topic, self._measured_gripper_callback, 10)

        self.get_logger().info("Subscribing to gripper commands...")
        self._gripper_command_subscription = self.create_subscription(
            Float32, gripper_command_topic, self._gripper_command_callback, 10
        )
        self._action_client = ActionClient(self, Move, move_action_topic)

        self.get_logger().info("Waiting for gripper move action server...")
        if not self._action_client.wait_for_server(timeout_sec=self._ACTION_SERVER_TIMEOUT):
            raise RuntimeError(
                f"Move action server not available after {self._ACTION_SERVER_TIMEOUT} seconds!"
            )

        if self._control_mode == 'grasp_hold':
            grasp_action_topic = self.get_parameter("grasp_action_topic").value
            self._grasp_action_client = ActionClient(self, Grasp, grasp_action_topic)
            if not self._grasp_action_client.wait_for_server(
                    timeout_sec=self._ACTION_SERVER_TIMEOUT):
                raise RuntimeError("Grasp action server is not available")
        self.get_logger().info("Gripper client initialized: " + self._control_mode)

    def _create_grasp_policy(self, max_width):
        return GraspHoldPolicy(
            self._send_policy_goal,
            max_width=max_width, speed=self._gripper_move_speed,
            deadband_percent=self._command_deadband_percent,
            **{name: self.get_parameter(name).value for name in (
                'grasp_force_n', 'grasp_force_limit_n', 'command_speed_limit_m_s',
                'grasp_epsilon_inner_m', 'grasp_epsilon_outer_m',
                'close_trigger_width_m', 'release_trigger_width_m',
                'measurement_max_age_s')},
            stamp_ns=lambda: self.get_clock().now().nanoseconds,
            notify=self._publish_capture_telemetry)

    def _home_gripper(self, homing_action_topic: str) -> None:
        self.get_logger().info("Starting gripper homing...")
        homing_client = ActionClient(self, Homing, homing_action_topic)

        self.get_logger().info(f"Waiting for homing action server {homing_action_topic}...")
        if not homing_client.wait_for_server(timeout_sec=self._ACTION_SERVER_TIMEOUT):
            raise RuntimeError(
                f"Homing action server not available after {self._ACTION_SERVER_TIMEOUT} seconds!"
            )

        self.get_logger().info("Homing action server found!")
        goal_msg = Homing.Goal()
        future = homing_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            raise RuntimeError("Homing action rejected!")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        time.sleep(2)

        if result.result.success:
            self.get_logger().info("Gripper homing successful!")
        else:
            raise RuntimeError("Gripper homing failed!")

    def _get_max_gripper_width(self, joint_states_topic: str) -> None:
        self.get_logger().info("Readout maximum gripper width...")
        future = rclpy.task.Future()

        def joint_state_callback(msg):
            _INDEX_FINGER_LEFT = 0
            self._max_width = 2 * msg.position[_INDEX_FINGER_LEFT]
            if self._max_width <= 0.0:
                self.get_logger().warning(
                    "Invalid gripper width from joint state; using fallback "
                    f"{self._max_width_fallback}"
                )
                self._max_width = self._max_width_fallback
            self.get_logger().info(f"Maximum gripper width determined: {self._max_width}")
            if not future.done():
                future.set_result(True)

        self.get_logger().info(f"Subscribing to {joint_states_topic}...")
        gripper_subscription = self.create_subscription(
            JointState, joint_states_topic, joint_state_callback, 10
        )

        self.get_logger().info(f"Waiting for {joint_states_topic}...")
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(f"Unsubscribing from {joint_states_topic}")
        self.destroy_subscription(gripper_subscription)

    def _measured_gripper_callback(self, msg: JointState) -> None:
        try:
            width = measured_finger_width(msg.name, msg.position, self._max_width)
            stamp = msg.header.stamp
            stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            now_ns = self.get_clock().now().nanoseconds
            age_s = (now_ns - stamp_ns) / 1e9
            max_age_s = self._grasp_policy.config['measurement_max_age_s']
            # The native driver publishes a ROS source stamp. Queue delivery
            # must never give old encoder feedback a fresh receipt timestamp.
            if (stamp_ns <= 0 or not 0 <= int(stamp.nanosec) < 1_000_000_000
                    or age_s < -0.001 or age_s > max_age_s):
                raise ValueError('Finger JointState source stamp is missing, stale or future')
            measured_at = time.monotonic() - max(0.0, age_s)
        except (ValueError, TypeError, AttributeError, OverflowError) as error:
            self._grasp_policy.set_measurement(None)
            self.get_logger().warning('Ignoring invalid finger feedback: ' + str(error))
            return
        self._grasp_policy.set_measurement(width, received_at=measured_at)

    def _send_policy_goal(self, kind, parameters):
        self._grasp_policy.validate_goal(kind, parameters)
        goal = Grasp.Goal() if kind == 'grasp' else Move.Goal()
        goal.width = parameters['width']
        goal.speed = parameters['speed']
        if kind == 'grasp':
            goal.force = parameters['force']
            goal.epsilon.inner = parameters['epsilon_inner']
            goal.epsilon.outer = parameters['epsilon_outer']
            return self._grasp_action_client.send_goal_async(goal)
        return self._action_client.send_goal_async(goal)

    def _gripper_command_callback(self, msg: Float32) -> None:
        if self._control_mode == 'grasp_hold':
            # Check before min/max: Python's clamp idiom maps NaN to a
            # seemingly valid endpoint, which could unintentionally open.
            if not math.isfinite(msg.data):
                self._grasp_policy.request(float('nan'))
                return
            percent = max(0.0, min(1.0, float(msg.data)))
            if self._invert_gripper_command:
                percent = 1.0 - percent
            self._grasp_policy.request(percent * self._max_width)
            return
        new_open_width_percent = max(
            self._MIN_GRIPPER_WIDTH_PERCENT,
            min(self._MAX_GRIPPER_WIDTH_PERCENT, msg.data),
        )
        if self._invert_gripper_command:
            new_open_width_percent = self._MAX_GRIPPER_WIDTH_PERCENT - new_open_width_percent

        if (
            self._last_gripper_command_percent is not None
            and abs(new_open_width_percent - self._last_gripper_command_percent)
            < self._command_deadband_percent
        ):
            return

        if self._gripper_command_transmitted:
            new_open_width = self._max_width * new_open_width_percent
            self._send_gripper_command(new_open_width)
            self._last_gripper_command_percent = new_open_width_percent
            self._gripper_command_transmitted = False

    def _send_gripper_command(self, gripper_position: float) -> None:
        goal_msg = Move.Goal()
        goal_msg.width = gripper_position
        goal_msg.speed = self._gripper_move_speed
        self._capture_sequence += 1
        sequence = self._capture_sequence
        self._capture_pending_goal = (sequence, float(goal_msg.width), float(goal_msg.speed))
        self._capture_requested_width_m = float(goal_msg.width)
        self._capture_requested_speed_m_s = float(goal_msg.speed)
        self._capture_status = 'pending'
        self._capture_error = None
        self._publish_capture_telemetry()
        try:
            self._future = self._action_client.send_goal_async(goal_msg)
            self._future.add_done_callback(
                lambda future: self._gripper_response_callback(
                    future, sequence, float(goal_msg.width), float(goal_msg.speed)))
        except Exception as error:
            self._capture_status = 'error'
            self._capture_error = 'Move goal request failed: ' + str(error)
            self._publish_capture_telemetry()
            raise

    def _gripper_response_callback(self, future: rclpy.task.Future,
                                   sequence=None, requested_width=None, requested_speed=None) -> None:
        if sequence is not None and sequence != self._capture_sequence:
            self.get_logger().warning('Ignoring late gripper response for an older request')
            return
        try:
            goal_handle = future.result()
        except Exception as error:
            self._capture_status = 'error'
            self._capture_error = 'Move goal response failed: ' + str(error)
            self._publish_capture_telemetry()
            raise

        if not goal_handle.accepted:
            self.get_logger().warning(f"Goal rejected with status: {goal_handle.status}")
            self._gripper_command_transmitted = True
            self._capture_status = 'error'
            self._capture_error = 'Move goal rejected'
            self._publish_capture_telemetry()
            return

        if sequence is None:
            sequence, requested_width, requested_speed = self._capture_pending_goal
        self._capture_accepted_sequence = sequence
        self._capture_command_stamp_ns = self.get_clock().now().nanoseconds
        self._capture_target_width_m = requested_width
        self._capture_speed_m_s = requested_speed
        self._capture_valid = True
        self._capture_status = 'accepted'
        self._capture_error = None
        self._capture_result_sequence = None
        self._capture_result_status = None
        self._capture_result_success = None
        self._capture_result_error = None
        self._publish_capture_telemetry()
        try:
            self._get_result_future = goal_handle.get_result_async()
            self._get_result_future.add_done_callback(
                lambda result_future: self._get_result_callback(result_future, sequence))
        except Exception as error:
            self._capture_status = 'error'
            self._capture_error = 'Move result request failed: ' + str(error)
            self._publish_capture_telemetry()
            raise

    def _get_result_callback(self, future: rclpy.task.Future, sequence=None) -> None:
        sequence = self._capture_accepted_sequence if sequence is None else sequence
        if sequence != self._capture_accepted_sequence:
            self.get_logger().warning('Ignoring late gripper result for an older accepted goal')
            return
        try:
            response = future.result()
            result = response.result
        except Exception as error:
            self._capture_status = 'error'
            self._capture_error = 'Move result failed: ' + str(error)
            self._publish_capture_telemetry()
            raise
        self.get_logger().debug("Result: {0}".format(result))
        self._gripper_command_transmitted = True
        self._capture_result_sequence = sequence
        self._capture_result_status = getattr(response, 'status', None)
        self._capture_result_success = getattr(result, 'success', None)
        self._capture_result_error = getattr(result, 'error', None)
        # franka_gripper 2.0.2 aborts both ordinary move(false) and exceptions.
        # Only exceptions carry result.error. Keep an ordinary unsuccessful Move
        # distinct: it does not prove contact/grasp success, and the accepted
        # command remains the intended teleoperation target for learning.
        if (self._capture_result_status == 4 and self._capture_result_success is True
                and self._capture_result_error == ''):
            self._capture_status = 'completed'
            self._capture_error = None
        elif (self._capture_result_status == 6 and self._capture_result_success is False
              and self._capture_result_error == ''):
            self._capture_status = 'not_reached'
            self._capture_error = None
            self.get_logger().warning(
                f'Move target not reached: accepted_sequence={sequence}, '
                f'target_width_m={self._capture_target_width_m}; '
                'ordinary unsuccessful result, cause not specified by gripper')
        else:
            self._capture_status = 'error'
            self._capture_error = (
                f'Move failed (status={getattr(response, "status", None)}): '
                + str(getattr(result, 'error', 'unsuccessful action result')))
            self.get_logger().error(self._capture_error)
        self._publish_capture_telemetry()

    def _publish_capture_telemetry(self) -> None:
        """Publish a heartbeat while retaining the last actually accepted goal."""
        if self._grasp_policy is not None:
            self._grasp_policy.tick()
            payload = self._grasp_policy.snapshot()
        else:
            payload = self._legacy_capture_payload()
        try:
            message = String()
            message.data = json.dumps(payload, allow_nan=False)
            self._capture_publisher.publish(message)
        except Exception as error:
            # Capture failure never changes goal execution. The recorder
            # independently rejects stale or missing telemetry.
            self.get_logger().warning('Could not publish gripper capture telemetry: ' + str(error))

    def _legacy_capture_payload(self):
        return {
            'schema_version': 1,
            'sequence': self._capture_sequence,
            'accepted_sequence': self._capture_accepted_sequence,
            'stamp_ns': self.get_clock().now().nanoseconds,
            'command_stamp_ns': self._capture_command_stamp_ns,
            'target_width_m': self._capture_target_width_m,
            'speed_m_s': self._capture_speed_m_s,
            'max_width_m': self._max_width,
            'valid': self._capture_valid,
            'status': self._capture_status,
            'requested_width_m': self._capture_requested_width_m,
            'requested_speed_m_s': self._capture_requested_speed_m_s,
            'error': self._capture_error,
            'result_sequence': self._capture_result_sequence,
            'result_status': self._capture_result_status,
            'result_success': self._capture_result_success,
            'result_error': self._capture_result_error,
        }


def main(args=None):
    rclpy.init(args=args)
    gripper_client = GripperClient()
    rclpy.spin(gripper_client)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
