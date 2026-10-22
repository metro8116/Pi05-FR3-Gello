#!/usr/bin/env python3
"""Local-only ROS2 execution bridge; no model or camera dependencies.

Run through ros2/pi05_fr3.launch.py in the existing ROS container. --help and
imports work without ROS, so CI can exercise the pure execution gate.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import socketserver
import threading
import time
import uuid

from bridge_core import ExecutionGate, SafetyError
from config import load_config
from protocol import ProtocolError, recv_packet, send_packet


# Native gripper Stop has been measured at ~2.4 s. This bounded hardware
# handshake is independent of controller deactivation (1.2 s), Move timeout,
# observation TTL, state freshness, and heartbeat deadlines.
GRIPPER_STOP_TIMEOUT_S = 3.0


def source_time(stamp_ns, previous_stamp_ns, now_ros_ns, now_monotonic, max_age):
    """Map a real ROS source stamp to the same host's monotonic clock."""
    if stamp_ns <= 0 or (previous_stamp_ns is not None and stamp_ns <= previous_stamp_ns):
        raise SafetyError('Source stamp is missing, duplicated, or out of order')
    age = (now_ros_ns - stamp_ns) / 1e9
    if not 0 <= age <= max_age:
        raise SafetyError('Source ROS stamp is stale or in the future')
    return now_monotonic - age


def move_result(status, success, error, *, closing=False, actual_width=None,
                target_width=None, measurement_fresh=False, deadband=.002):
    """Keep ordinary not-reached distinct from transport/controller faults."""
    if status == 4 and success is True and error == '':
        return 'completed'
    if (status == 6 and success is False
            and error in ('', 'libfranka gripper: Command failed!')
            and closing and measurement_fresh
            and isinstance(actual_width, (int, float)) and not isinstance(actual_width, bool)
            and 0 <= actual_width <= .08 and target_width is not None
            and actual_width > target_width + deadband):
        return 'not_reached'
    raise SafetyError(f'Move failed (status={status}, success={success}): {error}')


def create_node(config):
    # Deliberately lazy: these packages belong to the container, not AutoDL.
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from controller_manager_msgs.srv import SwitchController, ListControllers
    from franka_msgs.action import Move
    from std_srvs.srv import Trigger

    class Bridge(Node):
        def __init__(self):
            super().__init__('pi05_fr3_bridge')
            self.lock = threading.RLock()
            self.config = config
            self.gate = ExecutionGate(config['execution'])
            self.bridge = config['bridge']
            self.active = False
            self.switch_pending = False
            self.stop_pending = False
            self.stop_started = None
            self.list_pending = False
            self.controller_state = None
            self.arm_header = self.gripper_header = None
            self.last_accepted_width = None
            self.move_pending = False
            self.move_ack_pending = False
            self.move_ack_generation = None
            self.gripper_stop_pending = False
            self.gripper_stop_again = False
            self.gripper_stop_started = None
            self.move_generation = 0
            self.move_started = None
            self.move_handle = None
            self.gripper_result = None
            self.move_preemption = None
            self.post_preempt_open_required = None
            self.latest_requested_width = None
            self.last_pending_target_logged = None
            self.manual_open = None
            self.cancelled_manual_ids = OrderedDict()
            self.exit_error = None
            self.joint_names = [config['robot']['arm_id'] + '_joint' + str(i) for i in range(1, 8)]
            self.finger_names = [config['robot']['arm_id'] + '_finger_joint' + str(i) for i in (1, 2)]
            self.publisher = self.create_publisher(JointState, self.bridge['command_topic'], 1)
            self.create_subscription(JointState, self.bridge['state_topic'], self.arm_state, qos_profile_sensor_data)
            self.create_subscription(JointState, self.bridge['gripper_state_topic'], self.gripper_state, qos_profile_sensor_data)
            self.switch = self.create_client(SwitchController, self.bridge['controller_manager'] + '/switch_controller')
            self.list_controllers = self.create_client(ListControllers, self.bridge['controller_manager'] + '/list_controllers')
            self.move = ActionClient(self, Move, self.bridge['gripper_move_action'])
            stop_service = self.bridge.get('gripper_stop_service', self.bridge['gripper_move_action'].rsplit('/', 1)[0] + '/stop')
            self.gripper_stop = self.create_client(Trigger, stop_service)
            self.timer = self.create_timer(1 / config['execution']['interpolation_hz'], self.tick)
            self.create_timer(.25, self.poll_controllers)
            self.get_logger().info('PI05 bridge: controller remains INACTIVE until an explicit arm command')

        def source_stamp(self, msg, previous, max_age):
            stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            mono = source_time(stamp_ns, previous, self.get_clock().now().nanoseconds, time.monotonic(), max_age)
            return stamp_ns, mono

        def arm_state(self, msg):
            with self.lock:
                try:
                    if len(set(msg.name)) != len(msg.name):
                        raise SafetyError('Duplicate joint names in measured arm message')
                    indices = [msg.name.index(name) for name in self.joint_names]
                    stamp, mono = self.source_stamp(msg, self.arm_header, self.config['execution']['state_max_age_s'])
                    self.gate.update_arm([msg.position[i] for i in indices], [msg.velocity[i] for i in indices], mono)
                    self.arm_header = stamp
                except (ValueError, IndexError) as exc:
                    if self.manual_open_active():
                        self.manual_error(self.manual_open, 'Arm state invalid: ' + str(exc))
                    elif self.gate.armed or self.gate.arming:
                        self.fail('Arm state invalid: ' + str(exc))

        def gripper_state(self, msg):
            with self.lock:
                try:
                    if len(set(msg.name)) != len(msg.name):
                        raise SafetyError('Duplicate finger joint names')
                    indices = [msg.name.index(name) for name in self.finger_names]
                    stamp, mono = self.source_stamp(msg, self.gripper_header, self.config['execution'].get('gripper_max_age_s', .5))
                    self.gate.update_width(sum(msg.position[i] for i in indices), mono)
                    self.gripper_header = stamp
                except (ValueError, IndexError) as exc:
                    if self.manual_open_active():
                        self.manual_error(self.manual_open, 'Gripper state invalid: ' + str(exc))
                    elif self.gate.armed or self.gate.arming:
                        self.fail('Gripper state invalid: ' + str(exc))

        def poll_controllers(self):
            with self.lock:
                if self.list_pending or not self.list_controllers.service_is_ready():
                    return
                self.list_pending = True
                future = self.list_controllers.call_async(ListControllers.Request())
                future.add_done_callback(self.controller_status)

        def controller_status(self, future):
            with self.lock:
                self.list_pending = False
                try:
                    matches = [c for c in future.result().controller if c.name == self.bridge['controller_name']]
                    self.controller_state = matches[0].state if len(matches) == 1 else None
                    if self.manual_open_active() and self.controller_state != 'inactive':
                        self.manual_error(self.manual_open, 'Arm controller changed state during manual open')
                    if self.gate.armed and not self.switch_pending and self.controller_state != 'active':
                        self.fail('Arm controller ceased to be active')
                except Exception as exc:
                    if self.gate.armed:
                        self.fail('Cannot inspect controller state: ' + str(exc))

        def status(self):
            result = self.gate.state(time.monotonic())
            result['ready'] = bool(self.exit_error is None and result['ready'] and self.switch.service_is_ready()
                                   and self.move.server_is_ready() and self.gripper_stop.service_is_ready()
                                   and self.controller_state in ('inactive', 'active'))
            result['controller_state'] = self.controller_state
            result['exit_error'] = self.exit_error
            result['gripper_result'] = self.gripper_result
            result['gripper_latest_requested_width_m'] = self.latest_requested_width
            result['arm_holding_for_gripper_stop'] = bool(self.gate.armed and (
                self.move_preemption is not None or self.post_preempt_open_required is not None))
            if self.manual_open is not None:
                self.manual_open['last_width_m'] = self.gate.width
            result['manual_open'] = None if self.manual_open is None else dict(self.manual_open)
            result['manual_open_ready'] = bool(result['ready'] and self.controller_state == 'inactive'
                and not (self.gate.armed or self.gate.arming or self.active or self.switch_pending
                         or self.stop_pending or self.gripper_stop_pending or self.move_pending
                         or self.move_ack_pending or self.move_preemption is not None or self.manual_open_active())
                and self.gate.measured_velocity is not None
                and max(abs(v) for v in self.gate.measured_velocity) <= .05)
            return result

        def publish(self, q):
            if self.exit_error is not None:
                return
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = list(self.joint_names)
            msg.position = list(q)
            self.publisher.publish(msg)

        def request(self, packet):
            with self.lock:
                kind = packet.get('type')
                if packet.get('version') != 1:
                    raise ProtocolError('Bridge request version must be 1')
                if kind == 'state':
                    return self.status()
                if kind == 'stop':
                    if packet.get('manual_operation_id') is not None:
                        self.revoke_manual(self.manual_operation_id(packet['manual_operation_id']))
                    # A local stop is always allowed, even if the GUI lost its session.
                    self.fail(str(packet.get('reason', 'Stopped by user')))
                    return dict(type='stopped', version=1, **{k: self.status()[k] for k in ('armed', 'fault')})
                if self.exit_error is not None:
                    raise SafetyError('Bridge is shutting down: ' + self.exit_error)
                if kind == 'manual_open':
                    return self.start_manual_open(packet)
                try:
                    if kind == 'arm':
                        if (self.stop_pending or self.switch_pending or self.gripper_stop_pending
                                or self.move_pending or self.controller_state != 'inactive'):
                            raise SafetyError('Controller must be inactive and all switches complete before arm')
                        if not self.status()['ready']:
                            raise SafetyError('Fresh state, Move/stop service, and controller manager required')
                        q = self.gate.arm(packet.get('session_id'), packet.get('model_id'), time.monotonic())
                        self.last_accepted_width = None
                        self.gripper_result = None
                        self.move_preemption = None
                        self.post_preempt_open_required = None
                        self.latest_requested_width = None
                        self.last_pending_target_logged = None
                        self.manual_open = None
                        self.publish(q)
                        self.activate()
                        return dict(type='armed', version=1, armed=False, arming=True,
                                    session_id=self.gate.session_id)
                    if kind == 'heartbeat':
                        self.gate.heartbeat(packet.get('session_id'), packet.get('camera_ok'),
                                            packet.get('camera_capture_monotonic'), time.monotonic())
                        return dict(type='heartbeat', version=1, armed=self.gate.armed)
                    if kind == 'submit':
                        return self.gate.submit(packet, time.monotonic())
                    raise ProtocolError('Unsupported bridge request: ' + str(kind))
                except SafetyError as exc:
                    # A rejected packet belonging to the active session invalidates it.
                    if packet.get('session_id') == self.gate.session_id and (self.gate.armed or self.gate.arming):
                        self.fail(str(exc))
                    if (kind in ('heartbeat', 'submit') and packet.get('session_id') == self.gate.session_id and self.gate.fault
                            and not self.gate.armed and not self.gate.arming):
                        raise SafetyError(self.gate.fault) from exc
                    raise

        def manual_operation_id(self, value):
            if not isinstance(value, str) or len(value) > 36:
                raise ProtocolError('manual operation_id must be a UUID string')
            try:
                return str(uuid.UUID(value))
            except (ValueError, AttributeError) as exc:
                raise ProtocolError('manual operation_id must be a UUID string') from exc

        def revoke_manual(self, operation_id):
            self.cancelled_manual_ids[operation_id] = True
            self.cancelled_manual_ids.move_to_end(operation_id)
            while len(self.cancelled_manual_ids) > 128:
                self.cancelled_manual_ids.popitem(last=False)

        def manual_open_active(self):
            return self.manual_open is not None and self.manual_open['status'] in ('pending', 'accepted')

        def manual_current(self, operation, generation):
            return (self.exit_error is None and self.manual_open is operation and generation == self.move_generation
                    and operation['operation_id'] not in self.cancelled_manual_ids
                    and operation['status'] in ('pending', 'accepted'))

        def cancel_manual(self, reason):
            if self.manual_open_active():
                self.manual_open.update(status='cancelled', reason=str(reason),
                                        last_width_m=self.gate.width)
                self.revoke_manual(self.manual_open['operation_id'])
                self.log_move('manual_cancelled', operation_id=self.manual_open['operation_id'], reason=str(reason))

        def manual_error(self, operation, reason, fatal=False):
            if self.manual_open is operation:
                operation.update(status='error', error=str(reason), last_width_m=self.gate.width)
                if self.gripper_result is not None:
                    self.gripper_result['status'] = 'error'
                self.log_move('manual_error', operation_id=operation['operation_id'], error=str(reason))
                self.fail(str(reason))
            if fatal:
                self.emergency_exit(str(reason))

        def start_manual_open(self, packet):
            if self.exit_error is not None:
                raise SafetyError('Bridge is shutting down: ' + self.exit_error)
            if set(packet) != {'version', 'type', 'operation_id'}:
                raise ProtocolError('manual_open accepts only operation_id; width/speed are fixed')
            operation_id = self.manual_operation_id(packet.get('operation_id'))
            if operation_id in self.cancelled_manual_ids:
                raise SafetyError('Manual open operation was cancelled before it could start')
            if (self.gate.armed or self.gate.arming or self.active or self.controller_state != 'inactive'
                    or self.switch_pending or self.stop_pending or self.gripper_stop_pending
                    or self.move_pending or self.move_ack_pending or self.move_preemption is not None
                    or self.manual_open_active()):
                raise SafetyError('Manual open requires inactive model/controller and no pending operations')
            if self.manual_open is not None and self.manual_open['operation_id'] == operation_id:
                raise SafetyError('Manual open operation_id has already been used')
            if not self.status()['ready'] or max(abs(v) for v in self.gate.measured_velocity) > .05:
                raise SafetyError('Manual open requires fresh stationary arm and fresh measured gripper state')
            goal = Move.Goal()
            goal.width, goal.speed = .08, .04
            self.move_generation += 1
            generation = self.move_generation
            self.move_pending = self.move_ack_pending = True
            self.move_ack_generation = generation
            self.move_started = time.monotonic()
            self.move_preemption = self.post_preempt_open_required = None
            # This explicit operation has its own fault history; old model faults
            # remain in logs and must not mask a new manual-opening failure.
            self.gate.fault = None
            self.latest_requested_width = .08
            operation = dict(operation_id=operation_id, status='pending', target_width_m=.08,
                             speed_m_s=.04, requested_from_width_m=self.gate.width,
                             last_width_m=self.gate.width, requested_monotonic=self.move_started,
                             generation=generation)
            self.manual_open = operation
            self.gripper_result = dict(status='pending', width_m=.08, speed_m_s=.04,
                                       closing=False, manual_operation_id=operation_id)
            self.log_move('manual_requested', operation_id=operation_id, width_m=.08, speed_m_s=.04)
            try:
                future = self.move.send_goal_async(goal)
                future.add_done_callback(lambda done: self.manual_accepted(done, operation, generation))
            except Exception as exc:
                self.manual_error(operation, 'Manual open send failed: ' + str(exc), fatal=True)
            return dict(version=1, type='manual_open_started', operation_id=operation_id,
                        manual_open=dict(operation))

        def clear_move_ack(self, generation):
            if self.move_ack_generation == generation:
                self.move_ack_pending = False
                self.move_ack_generation = None

        def manual_accepted(self, future, operation, generation):
            with self.lock:
                try:
                    handle = future.result()
                    self.clear_move_ack(generation)
                    self.log_move('manual_accepted', operation_id=operation['operation_id'],
                                  goal_generation=generation, accepted=bool(handle.accepted))
                    if not self.manual_current(operation, generation):
                        if handle.accepted:
                            self.move_handle = handle
                            terminal = handle.get_result_async()
                            terminal.add_done_callback(lambda done: self.manual_completed(done, operation, generation))
                            self.stop_gripper()
                        elif self.move_ack_generation is None and self.move_handle is None:
                            self.move_pending = False
                        return
                    if not handle.accepted:
                        raise SafetyError('Manual open Move goal was rejected')
                    self.move_handle = handle
                    self.move_started = time.monotonic()
                    operation.update(status='accepted', accepted_monotonic=self.move_started)
                    self.gripper_result['status'] = 'accepted'
                    terminal = handle.get_result_async()
                    terminal.add_done_callback(lambda done: self.manual_completed(done, operation, generation))
                except Exception as exc:
                    if self.manual_current(operation, generation):
                        self.manual_error(operation, 'Manual open acceptance failed: ' + str(exc))

        def manual_completed(self, future, operation, generation):
            with self.lock:
                try:
                    result = future.result()
                    raw = result.result
                    self.log_move('manual_result', operation_id=operation['operation_id'],
                                  goal_generation=generation, result_status=result.status,
                                  success=raw.success, error=raw.error)
                    if self.manual_open is operation:
                        operation.update(result_status=result.status, success=raw.success,
                                         raw_error=raw.error, last_width_m=self.gate.width)
                    if not self.manual_current(operation, generation):
                        return
                    operation['error'] = raw.error
                    if result.status != 4 or raw.success is not True or raw.error != '':
                        raise SafetyError(f'Manual open failed (status={result.status}, success={raw.success}): {raw.error}')
                    self.move_pending = False
                    self.move_handle = None
                    operation.update(status='completed', completed_monotonic=time.monotonic())
                    self.gripper_result.update(status='completed', result_status=result.status,
                                               success=raw.success, error=raw.error)
                except Exception as exc:
                    if self.manual_current(operation, generation):
                        self.manual_error(operation, str(exc))

        def activate(self):
            request = SwitchController.Request()
            request.activate_controllers = [self.bridge['controller_name']]
            request.strictness = SwitchController.Request.STRICT
            request.activate_asap = True
            request.timeout.sec = 1
            self.switch_pending = True
            session = self.gate.session_id
            future = self.switch.call_async(request)
            future.add_done_callback(lambda done: self.activated(done, session))

        def activated(self, future, session):
            with self.lock:
                self.switch_pending = False
                try:
                    if not future.result().ok:
                        raise SafetyError('Controller manager refused activation')
                    self.active = True
                    self.controller_state = 'active'
                    if session != self.gate.session_id or not self.gate.arming:
                        self.deactivate()
                        return
                    self.gate.activated(time.monotonic())
                    self.get_logger().info('Explicitly armed: absolute joint control, local deadlines enabled')
                except Exception as exc:
                    self.fail('Activation failed: ' + str(exc))

        def deactivate(self):
            if self.stop_pending:
                return
            self.stop_started = time.monotonic()
            if not self.switch.service_is_ready():
                self.emergency_exit('Controller manager unavailable while stopping')
                return
            request = SwitchController.Request()
            request.deactivate_controllers = [self.bridge['controller_name']]
            request.strictness = SwitchController.Request.STRICT
            request.timeout.sec = 1
            self.stop_pending = True
            try:
                future = self.switch.call_async(request)
                future.add_done_callback(self.deactivated)
            except Exception as exc:
                self.stop_pending = False
                self.emergency_exit('Controller deactivation request failed: ' + str(exc))

        def deactivated(self, future):
            with self.lock:
                self.stop_pending = False
                try:
                    if not future.result().ok:
                        raise SafetyError('Controller manager refused deactivation')
                    self.active = False
                    self.controller_state = 'inactive'
                except Exception as exc:
                    self.emergency_exit('Cannot deactivate controller: ' + str(exc))

        def emergency_exit(self, reason):
            with self.lock:
                if self.exit_error is not None:
                    return
                self.exit_error = str(reason)
                self.cancel_manual('Fatal bridge shutdown: ' + self.exit_error)
                self.gate.stop(self.gate.fault or self.exit_error)
                self.move_preemption = self.post_preempt_open_required = None
                self.get_logger().fatal(self.exit_error + '; shutting down entire deployment launch')
                # Humble context shutdown waits for callbacks, so invoking it
                # here could wait for this very callback while holding lock.
                # The main spin_once loop observes this latch after we return.

        def stop_gripper(self):
            # A global stop revokes an opening preemption. Its old callbacks may
            # help finish stopping, but can never release another Move command.
            self.cancel_manual('Global gripper stop')
            self.move_preemption = None
            self.post_preempt_open_required = None
            self.move_generation += 1
            # Explicit native Stop is the sole physical stop path. A ROS action
            # cancellation also calls gripper.stop() inside the native driver;
            # sending both can overlap two hardware stops.
            if self.gripper_stop_pending:
                self.gripper_stop_again = True
                return
            if self.gripper_stop.service_is_ready():
                self.gripper_stop_pending = True
                self.gripper_stop_started = time.monotonic()
                try:
                    future = self.gripper_stop.call_async(Trigger.Request())
                    future.add_done_callback(self.gripper_stopped)
                except Exception as exc:
                    self.gripper_stop_pending = False
                    self.emergency_exit('Native gripper stop request failed: ' + str(exc))
            elif self.move_pending:
                self.emergency_exit('Gripper stop service unavailable during in-flight Move')

        def gripper_stopped(self, future):
            with self.lock:
                self.gripper_stop_pending = False
                try:
                    result = future.result()
                    if not result.success:
                        raise SafetyError(result.message)
                    # A goal request whose acceptance is still unknown can arrive
                    # after this stop. Keep re-arm blocked until that reply arrives.
                    self.move_pending = self.move_ack_pending
                    self.move_handle = None
                    if self.gripper_stop_again:
                        self.gripper_stop_again = False
                        self.stop_gripper()
                except Exception as exc:
                    self.emergency_exit('Native gripper stop failed: ' + str(exc))

        def fail(self, reason):
            was_active = self.gate.armed or self.gate.arming or self.active
            self.cancel_manual(reason)
            # A late heartbeat/stop packet must not replace the actual cause.
            first_reason = self.gate.fault or str(reason)
            self.gate.stop(first_reason)
            self.get_logger().error(str(reason))
            if was_active or self.move_pending:
                self.stop_gripper()
            if self.active or self.switch_pending or was_active:
                self.deactivate()

        def log_move(self, event, **fields):
            packet = dict(event='gripper_lifecycle', stage=event,
                          monotonic=time.monotonic(), session_id=self.gate.session_id,
                          generation=self.move_generation, actual_width_m=self.gate.width,
                          latest_requested_width_m=self.latest_requested_width)
            packet.update(fields)
            self.get_logger().info(json.dumps(packet, ensure_ascii=False, allow_nan=False))

        def begin_open_preemption(self):
            if (self.exit_error is not None or not self.gate.armed or self.move_preemption is not None
                    or self.move_ack_pending or self.move_handle is None):
                return
            preemption = dict(generation=self.move_generation, session_id=self.gate.session_id,
                              stop_done=False, terminal=False, started=time.monotonic())
            self.move_preemption = preemption
            self.gripper_result['status'] = 'preempting_for_open'
            self.log_move('preempt', reason='fresh model opening target supersedes accepted closing Move',
                          stop_deadline_s=GRIPPER_STOP_TIMEOUT_S,
                          arm_mode='rate_limited_deceleration_and_hold')
            try:
                if not self.gripper_stop.service_is_ready():
                    raise SafetyError('Native gripper stop unavailable during opening preemption')
                self.gripper_stop_pending = True
                self.gripper_stop_started = time.monotonic()
                future = self.gripper_stop.call_async(Trigger.Request())
                future.add_done_callback(lambda done: self.open_preemption_stopped(done, preemption))
            except Exception as exc:
                self.gripper_stop_pending = False
                self.fail('Opening preemption failed: ' + str(exc))

        def open_preemption_stopped(self, future, preemption):
            with self.lock:
                if (self.move_preemption is not preemption
                        or preemption['generation'] != self.move_generation
                        or preemption['session_id'] != self.gate.session_id or not self.gate.armed):
                    # The global-stop path owns this outstanding native stop now.
                    self.gripper_stopped(future)
                    return
                self.gripper_stop_pending = False
                try:
                    result = future.result()
                    if not result.success:
                        raise SafetyError(result.message)
                    preemption['stop_done'] = True
                    preemption['stop_confirmed_monotonic'] = time.monotonic()
                    self.log_move('preempt_stop_confirmed')
                    self.finish_open_preemption(preemption)
                except Exception as exc:
                    self.fail('Native gripper stop during opening preemption failed: ' + str(exc))

        def finish_open_preemption(self, preemption):
            if (self.move_preemption is not preemption or not self.gate.armed
                    or preemption['generation'] != self.move_generation
                    or preemption['session_id'] != self.gate.session_id):
                return
            if not (preemption['stop_done'] and preemption['terminal']):
                return
            self.move_preemption = None
            self.post_preempt_open_required = dict(generation=self.move_generation,
                                                   session_id=self.gate.session_id,
                                                   measurement_after=preemption['stop_confirmed_monotonic'])
            self.move_pending = False
            self.move_handle = None
            self.gripper_result['status'] = 'preempted_for_open'
            self.log_move('preempt_complete', reason='stop and old goal terminal confirmed; await a fresh tick target')
            # Deliberately no send_move here: no cached opening target is replayed.

        def send_move(self, width):
            if self.exit_error is not None or not self.gate.armed:
                return
            deadband = self.config['execution']['gripper_deadband_m']
            self.latest_requested_width = float(width)
            if self.gripper_result is not None:
                self.gripper_result['latest_requested_width_m'] = float(width)
            now = time.monotonic()
            fresh_measurement = (self.gate.width_stamp is not None
                                 and 0 <= now-self.gate.width_stamp <= self.config['execution'].get('gripper_max_age_s', .5))
            fresh_target = (self.gate.capture_stamp is not None and self.gate.execution_end is not None
                            and 0 <= now-self.gate.capture_stamp <= self.config['execution']['max_observation_age_s']
                            and now < self.gate.execution_end)
            if self.move_pending:
                if (self.last_pending_target_logged is None
                        or abs(width-self.last_pending_target_logged) >= deadband):
                    self.last_pending_target_logged = float(width)
                    self.log_move('target_while_pending', width_m=float(width),
                                  acceptance_pending=self.move_ack_pending)
                if (fresh_measurement and fresh_target and self.gripper_result is not None
                        and self.gripper_result.get('closing') and width > self.gate.width+deadband):
                    self.begin_open_preemption()
                return
            if self.post_preempt_open_required is not None:
                latch = self.post_preempt_open_required
                if (latch['generation'] != self.move_generation or latch['session_id'] != self.gate.session_id
                        or not fresh_measurement or self.gate.width_stamp <= latch['measurement_after']
                        or not fresh_target or width <= self.gate.width+deadband):
                    return
            if self.last_accepted_width is not None and abs(width - self.last_accepted_width) < deadband:
                return
            if not self.move.server_is_ready():
                self.fail('Native Move action server unavailable')
                return
            goal = Move.Goal()
            goal.width = float(width)
            goal.speed = self.config['execution']['gripper_speed_m_s']
            opening_after_preemption = self.post_preempt_open_required is not None
            self.move_generation += 1
            generation = self.move_generation
            self.move_pending = True
            self.move_ack_pending = True
            self.move_ack_generation = generation
            self.move_started = time.monotonic()
            self.gripper_result = dict(status='pending', width_m=width, speed_m_s=goal.speed,
                                       requested_from_width_m=self.gate.width,
                                       closing=width < self.gate.width - deadband,
                                       latest_requested_width_m=float(width))
            self.last_pending_target_logged = float(width)
            self.log_move('requested', width_m=float(width), speed_m_s=goal.speed,
                          closing=self.gripper_result['closing'])
            try:
                future = self.move.send_goal_async(goal)
                self.post_preempt_open_required = None
                if opening_after_preemption:
                    # Even a very fast Stop can finish before joint velocity is
                    # zero. Discard the final held q waypoint rather than pull
                    # back toward it; no target makes the limiter decelerate,
                    # and only a later submit can install new joint waypoints.
                    # Source timestamps and watchdog activity are not extended.
                    self.gate.actions = None
                future.add_done_callback(lambda done: self.move_accepted(done, generation, width))
            except Exception as exc:
                self.fail('Move send failed: ' + str(exc))

        def move_accepted(self, future, generation, width):
            with self.lock:
                try:
                    handle = future.result()
                    self.clear_move_ack(generation)
                    self.log_move('accepted', goal_generation=generation, width_m=float(width),
                                  accepted=bool(handle.accepted))
                    if generation != self.move_generation or not self.gate.armed:
                        if handle.accepted:
                            self.move_handle = handle
                            self.stop_gripper()
                        else:
                            self.move_pending = False
                        return
                    if not handle.accepted:
                        raise SafetyError('Native Move goal was rejected')
                    self.move_handle = handle
                    self.last_accepted_width = width
                    self.gripper_result['status'] = 'accepted'
                    future = handle.get_result_async()
                    future.add_done_callback(lambda done: self.move_completed(done, generation))
                except Exception as exc:
                    self.fail('Move acceptance failed: ' + str(exc))

        def move_completed(self, future, generation):
            with self.lock:
                if generation != self.move_generation:
                    try:
                        result = future.result()
                        self.log_move('late_result', goal_generation=generation, result_status=result.status,
                                      success=result.result.success, error=result.result.error)
                    except Exception as exc:
                        self.log_move('late_result_error', goal_generation=generation, error=str(exc))
                    return
                try:
                    result = future.result()
                    raw = result.result
                    self.gripper_result.update(result_status=result.status, success=raw.success, error=raw.error)
                    self.log_move('result', goal_generation=generation, result_status=result.status,
                                  success=raw.success, error=raw.error,
                                  width_m=self.gripper_result['width_m'])
                    preemption = self.move_preemption
                    if preemption is not None and preemption['generation'] == generation:
                        # Native Stop can abort the accepted closing Move. This
                        # exact result is only a terminal candidate in its own
                        # active opening handshake; Stop must still succeed, and
                        # the fresh post-stop opening gate must still pass.
                        native_stop_abort = (result.status in (5, 6) and raw.success is False
                            and raw.error == 'libfranka gripper: Command aborted!'
                            and self.gate.armed and self.move_pending and not self.move_ack_pending
                            and self.move_handle is not None
                            and preemption['session_id'] == self.gate.session_id
                            and self.gripper_result.get('closing') is True)
                        expected_terminal = ((result.status == 4 and raw.success is True and raw.error == '')
                            or (result.status in (5, 6) and raw.success is False
                                and raw.error in ('', 'libfranka gripper: Command failed!'))
                            or native_stop_abort)
                        if not expected_terminal:
                            raise SafetyError(f'Unexpected Move result during opening preemption: '
                                              f'status={result.status}, success={raw.success}, error={raw.error}')
                        preemption['terminal'] = True
                        self.log_move('preempt_goal_terminal', result_status=result.status)
                        self.finish_open_preemption(preemption)
                        return
                    self.move_pending = False
                    self.move_handle = None
                    now = time.monotonic()
                    fresh = (self.gate.width_stamp is not None
                             and 0 <= now - self.gate.width_stamp <= self.config['execution'].get('gripper_max_age_s', .5))
                    self.gripper_result['status'] = move_result(
                        result.status, raw.success, raw.error,
                        closing=self.gripper_result['closing'], actual_width=self.gate.width,
                        target_width=self.gripper_result['width_m'], measurement_fresh=fresh,
                        deadband=self.config['execution']['gripper_deadband_m'])
                    self.log_move('status', status=self.gripper_result['status'])
                    if self.gripper_result['status'] == 'not_reached':
                        self.get_logger().warning('Move target not reached; raw result retained, no retry or opening')
                except Exception as exc:
                    self.gripper_result['status'] = 'error'
                    self.fail(str(exc))

        def tick(self):
            with self.lock:
                if self.exit_error is not None:
                    return
                now = time.monotonic()
                if self.stop_pending and now - self.stop_started > 1.2:
                    self.emergency_exit('Controller deactivation timed out')
                    return
                if self.gripper_stop_pending and now - self.gripper_stop_started > GRIPPER_STOP_TIMEOUT_S:
                    self.emergency_exit(f'Native gripper stop timed out after {GRIPPER_STOP_TIMEOUT_S:.1f}s')
                    return
                if (self.move_preemption is not None
                        and now-self.move_preemption['started'] > GRIPPER_STOP_TIMEOUT_S):
                    self.fail('Opening preemption did not confirm stop and old goal terminal '
                              f'within {GRIPPER_STOP_TIMEOUT_S:.1f}s')
                    return
                if self.move_ack_pending and now - self.move_started > self.config['execution']['gripper_action_timeout_s']:
                    if self.manual_open_active():
                        self.manual_error(self.manual_open, 'Manual open acceptance timed out; ownership uncertain', fatal=True)
                    else:
                        self.emergency_exit('Native Move acceptance timed out; ownership uncertain')
                    return
                if (self.move_pending and (self.gate.armed or self.manual_open_active()) and self.move_preemption is None
                        and now - self.move_started > self.config['execution']['gripper_action_timeout_s']):
                    if self.manual_open_active():
                        self.manual_error(self.manual_open, 'Manual open Move timed out')
                    else:
                        self.fail('Native Move timed out')
                    return
                if self.manual_open_active():
                    if (not self.gate.state(now)['ready'] or self.controller_state != 'inactive'
                            or max(abs(v) for v in self.gate.measured_velocity) > .05):
                        self.manual_error(self.manual_open, 'Manual open lost fresh stationary arm/gripper state')
                    return
                try:
                    if (self.gate.armed and self.gate.actions is not None
                            and (self.move_preemption is not None or self.post_preempt_open_required is not None)):
                        # Preserve fresh timing and width intent, but discard all
                        # joint waypoints received while stopping the gripper.
                        # Distance=0 asks the existing limiter to decelerate to
                        # rest; never jump to measured q or replay these targets
                        # once the opening handshake releases. A later fresh
                        # submit is required to resume joint motion.
                        self.gate.actions = [list(self.gate.q) + [row[7]] for row in self.gate.actions]
                    command = self.gate.tick(now)
                    if command is not None:
                        self.publish(command.q)
                        if self.gate.armed and command.width_m is not None:
                            self.send_move(command.width_m)
                except SafetyError as exc:
                    self.fail(str(exc))

    return Bridge()


class BridgeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    import rclpy
    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=[])
    node = create_node(config)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(1.0)
            try:
                packet = recv_packet(self.request)
                result = node.request(packet)
            except Exception as exc:
                result = dict(type='error', version=1, error=str(exc))
            try:
                send_packet(self.request, result)
            except (OSError, ProtocolError):
                pass

    server = BridgeServer((config['bridge']['host'], config['bridge']['port']), Handler)
    worker = threading.Thread(target=server.serve_forever, name='pi05-bridge-rpc', daemon=True)
    worker.start()
    try:
        while rclpy.ok() and node.exit_error is None:
            rclpy.spin_once(node, timeout_sec=.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            with node.lock:
                node.fail('Bridge process exiting')
        server.shutdown()
        server.server_close()
        node.destroy_node()
        rclpy.try_shutdown()
    return 1 if node.exit_error else 0


if __name__ == '__main__':
    raise SystemExit(main())
