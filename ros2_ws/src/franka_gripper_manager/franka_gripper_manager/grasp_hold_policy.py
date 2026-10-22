"""Serial, testable Franka Hand grasp/hold/release action policy.

This module has no ROS imports. The caller supplies action futures and clocks;
all goal/cancel/result callbacks are guarded by the operation's object identity.
A successful Grasp means the gripper action succeeded, not task success.
"""

import math
import time
from types import MappingProxyType


POLICY_NAME = 'grasp_hold_v2'
_ABORTED_ERROR = 'libfranka gripper: Command aborted!'


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def measured_finger_width(names, positions, max_width):
    """Read both named fingers, never assume position ordering or full opening."""
    if len(names) != len(positions):
        raise ValueError('JointState name/position lengths differ')
    values = []
    for suffix in ('finger_joint1', 'finger_joint2'):
        matches = [index for index, name in enumerate(names)
                   if name == suffix or name.endswith('_' + suffix)]
        if len(matches) != 1:
            raise ValueError('JointState must contain one ' + suffix)
        position = positions[matches[0]]
        if not _finite_number(position) or position < 0.0:
            raise ValueError('Invalid measured finger position')
        values.append(float(position))
    width = sum(values)
    # Encoder noise at a mechanical endpoint is allowed; clamp only that noise.
    if width > max_width + 0.001:
        raise ValueError('Measured gripper width exceeds configured travel')
    return min(width, max_width)


class GraspHoldPolicy:
    """Hold a successful full-close grasp until an explicit fresh open request.

    ``send_goal(kind, parameters)`` returns an ActionClient-compatible Future.
    Only one operation may be in flight, including its cancellation handshake.
    """

    def __init__(self, send_goal, *, max_width=0.08, speed=0.0,
                 deadband_percent=0.02, grasp_force_n=0.0,
                 grasp_force_limit_n=0.0, command_speed_limit_m_s=0.0,
                 grasp_epsilon_inner_m=0.005, grasp_epsilon_outer_m=0.08,
                 close_trigger_width_m=0.03, release_trigger_width_m=0.06,
                 measurement_max_age_s=0.5, now=time.monotonic,
                 stamp_ns=time.time_ns, notify=lambda: None):
        values = {
            'max_width': max_width, 'speed': speed,
            'deadband_percent': deadband_percent,
            'grasp_force_n': grasp_force_n,
            'grasp_force_limit_n': grasp_force_limit_n,
            'command_speed_m_s': speed,
            'command_speed_limit_m_s': command_speed_limit_m_s,
            'grasp_epsilon_inner_m': grasp_epsilon_inner_m,
            'grasp_epsilon_outer_m': grasp_epsilon_outer_m,
            'close_trigger_width_m': close_trigger_width_m,
            'release_trigger_width_m': release_trigger_width_m,
            'measurement_max_age_s': measurement_max_age_s,
        }
        for name, value in values.items():
            if not _finite_number(value):
                raise ValueError(name + ' must be a finite number')
        if not 0.0 < max_width <= 0.08:
            raise ValueError('Franka Hand max_width must be in (0, 0.08] m')
        if not 0.0 < speed <= command_speed_limit_m_s <= 0.1:
            raise ValueError('Require explicit 0 < command speed <= configured speed limit <= 0.1 m/s')
        if not 0.0 < deadband_percent < 1.0:
            raise ValueError('command_deadband_percent must be in (0, 1)')
        # These are software ceilings, not an accessory rating or proof that
        # the actuator supports the requested force. The ROS client separately
        # requires explicit verification and respects the hardware damage lock.
        if not 0.0 < grasp_force_n <= grasp_force_limit_n <= 70.0:
            raise ValueError('Require explicit 0 < grasp force <= configured force limit <= 70 N')
        if not (0 <= grasp_epsilon_inner_m <= max_width
                and 0 <= grasp_epsilon_outer_m <= max_width):
            raise ValueError('Grasp tolerances must be within [0, max_width]')
        if not 0.0 <= close_trigger_width_m < release_trigger_width_m <= max_width:
            raise ValueError('Require 0 <= close threshold < release threshold <= max_width')
        if not 0.0 < measurement_max_age_s <= 0.5:
            raise ValueError('measurement_max_age_s must be in (0, 0.5]')
        self.max_width = float(max_width)
        self.speed = float(speed)
        self.deadband = float(deadband_percent) * self.max_width
        self.config = MappingProxyType({name: float(values[name]) for name in (
            'grasp_force_n', 'grasp_epsilon_inner_m', 'grasp_epsilon_outer_m',
            'close_trigger_width_m', 'release_trigger_width_m', 'measurement_max_age_s',
            'grasp_force_limit_n', 'command_speed_m_s', 'command_speed_limit_m_s')})
        self._send_goal = send_goal
        self._now = now
        self._stamp_ns = stamp_ns
        self._notify = notify
        self._measured_width = None
        self._measurement_time = None
        self._active = None
        self._queued_open = None
        self._last_dispatched_width = None
        self._close_latched = False
        self._state = 'idle'
        self._sequence = 0
        self._requested_width = None
        self._accepted = None
        self._status = 'pending'
        self._error = None
        self._result = None
        self._result_cancellation_reason = None

    def set_measurement(self, width, received_at=None):
        if not _finite_number(width) or not 0.0 <= width <= self.max_width:
            self._measurement_time = None
            self._measured_width = None
            return False
        measured_at = self._now() if received_at is None else received_at
        if not _finite_number(measured_at) or measured_at > self._now() + 0.001:
            self._measurement_time = None
            self._measured_width = None
            return False
        self._measured_width = float(width)
        self._measurement_time = measured_at
        return True

    def _measurement_fresh(self):
        return (self._measured_width is not None and self._measurement_time is not None
                and 0.0 <= self._now() - self._measurement_time
                <= self.config['measurement_max_age_s'])

    def _is_explicit_open(self, width):
        if width < self.config['release_trigger_width_m']:
            return False
        # Full opening also releases an existing grasp at the physical endpoint.
        return (width >= self.max_width - 1e-9
                or width > self._measured_width + self.deadband)

    def request(self, width):
        """Process the latest GELLO intent without interpreting NaN as opening."""
        if not _finite_number(width) or not 0.0 <= width <= self.max_width:
            self._fault('Non-finite or out-of-range GELLO gripper command')
            return
        width = float(width)
        if not self._measurement_fresh():
            # Never guess startup width or send a queued release on old feedback.
            if self._active is None and self._accepted is not None:
                self._fault('Measured gripper opening is missing or stale')
            return
        explicit_open = self._is_explicit_open(width)
        if self._active is not None:
            active = self._active
            if active['kind'] == 'grasp' and explicit_open:
                self._queue_release(width)
                if active['handle'] is not None and not active['cancel_sent']:
                    self._cancel_active_grasp(active)
            elif active['cancel_intent'] and not explicit_open:
                # Cancel cannot be unsent. Do not execute an outdated opening or
                # automatically retry closing once the old goal terminates.
                self._queued_open = None
                active['open_withdrawn'] = True
                self._fault('Open request withdrawn during grasp cancellation')
            return
        if self._state == 'holding':
            if explicit_open:
                self._dispatch('move', width)
            return
        if self._state == 'fault':
            if explicit_open:
                self._dispatch('move', width)
            return
        if (self._last_dispatched_width is not None
                and abs(width - self._last_dispatched_width) < self.deadband):
            return
        # Grasp.width is an expected object width, not a proportional motion
        # setpoint. Do not turn a first small GELLO closing movement into a
        # wide-width Grasp (or into a contact-closing Move). The middle band
        # deliberately holds the current mode until an explicit close/open.
        if width <= self.config['close_trigger_width_m'] and not self._close_latched:
            self._dispatch('grasp', width)
        elif explicit_open:
            self._dispatch('move', width)

    def _new_request(self, width):
        self._sequence += 1
        self._requested_width = width
        return self._sequence

    def _queue_release(self, width):
        if (self._queued_open is None
                or abs(width - self._queued_open[1]) >= self.deadband):
            self._queued_open = (self._new_request(width), width)
        active = self._active
        active['cancel_intent'] = True
        active['open_withdrawn'] = False
        self._state = 'canceling'
        # A terminal failure is never hidden by a later pending request.
        if self._status != 'error':
            self._status = 'pending'
            self._error = None
        self._notify()

    def _dispatch(self, kind, width, sequence=None):
        if self._active is not None:
            raise RuntimeError('Refusing concurrent Franka Hand goals')
        if not self._measurement_fresh():
            self._fault('Measured gripper opening expired before goal dispatch')
            return
        if kind == 'move' and width < self._measured_width - self.deadband:
            self._fault('Refusing a release target below the measured opening')
            return
        if sequence is None:
            sequence = self._new_request(width)
        else:
            self._requested_width = width
        parameters = {'width': width, 'speed': self.speed}
        if kind == 'grasp':
            parameters.update(force=self.config['grasp_force_n'],
                              epsilon_inner=self.config['grasp_epsilon_inner_m'],
                              epsilon_outer=self.config['grasp_epsilon_outer_m'])
        try:
            self.validate_goal(kind, parameters)
        except ValueError as error:
            self._fault(str(error))
            return
        if kind == 'grasp' and width <= self.config['close_trigger_width_m']:
            self._close_latched = True
        active = {
            'sequence': sequence, 'kind': kind, 'parameters': parameters,
            'handle': None, 'cancel_intent': False, 'cancel_sent': False,
            'cancel_done': False, 'cancel_ok': False, 'cancel_return_code': None,
            'cancel_started': None, 'terminal': None, 'open_withdrawn': False,
            'cancel_failed': False, 'fault_latched': False, 'fault_message': None,
        }
        self._active = active
        self._state = ('canceling' if self._result_cancellation_reason == 'operator_open'
                       else ('grasping' if kind == 'grasp' else 'moving'))
        self._last_dispatched_width = width
        if self._status != 'error':
            self._status = 'pending'
            self._error = None
        self._notify()
        try:
            future = self._send_goal(kind, parameters)
            future.add_done_callback(lambda done: self._on_response(active, done))
        except Exception as error:
            # A send exception is not proof that the remote endpoint is idle.
            # Keep this operation latched; no second goal may be issued.
            self._fault(kind + ' goal request failed: ' + str(error))

    def _on_response(self, active, future):
        if self._active is not active:
            return
        try:
            handle = future.result()
            if not handle.accepted:
                self._active = None
                self._queued_open = None
                self._fault(active['kind'] + ' goal rejected')
                return
            active['handle'] = handle
            self._accepted = {
                'sequence': active['sequence'], 'stamp_ns': int(self._stamp_ns()),
                'kind': active['kind'], **active['parameters'],
            }
            self._result = None
            self._result_cancellation_reason = None
            if not active['fault_latched']:
                self._error = None
                self._status = 'pending' if active['cancel_intent'] else 'accepted'
                self._state = 'canceling' if active['cancel_intent'] else (
                    'grasping' if active['kind'] == 'grasp' else 'moving')
            self._notify()
            result_future = handle.get_result_async()
            result_future.add_done_callback(lambda done: self._on_result(active, done))
            if (self._active is active and active['cancel_intent']
                    and not active['cancel_sent']):
                self._cancel_active_grasp(active)
        except Exception as error:
            # Keep a possibly accepted/running goal latched after an API error.
            self._fault(active['kind'] + ' goal response/result request failed: ' + str(error))

    def _cancel_active_grasp(self, active):
        if self._active is not active or active['cancel_sent']:
            return
        active['cancel_sent'] = True
        active['cancel_started'] = self._now()
        try:
            future = active['handle'].cancel_goal_async()
            future.add_done_callback(lambda done: self._on_cancel(active, done))
        except Exception as error:
            active['cancel_done'] = True
            active['cancel_failed'] = True
            self._queued_open = None
            self._fault('Grasp cancellation request failed: ' + str(error))
            self._finish_if_terminal(active)

    @staticmethod
    def _cancel_includes_handle(response, handle):
        goals = getattr(response, 'goals_canceling', [])
        goal_id = getattr(handle, 'goal_id', None)
        if goal_id is None:
            return bool(goals)
        expected = bytes(goal_id.uuid)
        return any(bytes(item.goal_id.uuid) == expected for item in goals)

    def _on_cancel(self, active, future):
        if self._active is not active:
            return
        active['cancel_done'] = True
        try:
            response = future.result()
            code = getattr(response, 'return_code', None)
            active['cancel_return_code'] = code
            active['cancel_ok'] = (code == 0 and self._cancel_includes_handle(
                response, active['handle']))
            # ERROR_GOAL_TERMINATED is safe only after a normal success terminal;
            # no assumption that a cancel ACK itself stopped the physical action.
            if not active['cancel_ok'] and code != 3:
                active['cancel_failed'] = True
                self._queued_open = None
                self._fault('Grasp cancellation was not acknowledged for this goal')
        except Exception as error:
            active['cancel_failed'] = True
            self._queued_open = None
            self._fault('Grasp cancellation response failed: ' + str(error))
        self._finish_if_terminal(active)

    def _on_result(self, active, future):
        if self._active is not active:
            return
        try:
            response = future.result()
            result = response.result
            terminal = {
                'sequence': active['sequence'], 'status': getattr(response, 'status', None),
                'success': getattr(result, 'success', None),
                'error': getattr(result, 'error', None),
            }
        except Exception as error:
            # No known terminal means no second physical command is safe.
            self._fault(active['kind'] + ' result failed: ' + str(error))
            return
        active['terminal'] = terminal
        self._finish_if_terminal(active)

    def _finish_if_terminal(self, active):
        if self._active is not active or active['terminal'] is None:
            return
        if active['cancel_sent'] and not active['cancel_done']:
            return
        terminal = active['terminal']
        # Do not expose a canceled terminal as normal pending telemetry until
        # its own cancel ACK has arrived and provenance can be classified.
        self._result = terminal
        success = (terminal['status'] == 4 and terminal['success'] is True
                   and terminal['error'] == '')
        ordinary_move_false = (active['kind'] == 'move' and terminal['status'] == 6
                               and terminal['success'] is False and terminal['error'] == '')
        operator_cancel = (
            active['kind'] == 'grasp' and active['cancel_intent']
            and active['cancel_sent'] and active['cancel_ok']
            and terminal['status'] == 5 and type(terminal['success']) is bool
            and terminal['error'] in ('', _ABORTED_ERROR))
        self._result_cancellation_reason = 'operator_open' if operator_cancel else None
        self._active = None
        queued = self._queued_open
        self._queued_open = None
        if active['fault_latched']:
            self._fault(active['fault_message'])
            return
        if active['cancel_failed'] or active['open_withdrawn']:
            self._fault('Grasp cancellation failed or opening intent was withdrawn')
            return
        if active['cancel_sent'] and not active['cancel_ok'] and not (
                active['cancel_return_code'] == 3 and success):
            self._fault('Grasp cancellation did not confirm this goal')
            return
        if not (success or ordinary_move_false or operator_cancel):
            self._fault(
                active['kind'].title() + ' failed (status=' + str(terminal['status'])
                + '): ' + str(terminal['error']))
            return
        if operator_cancel and queued is None:
            self._fault('Canceled grasp has no current explicit opening request')
            return
        if queued is not None:
            self._status = 'pending'
            self._error = None
            self._state = 'canceling'
            self._notify()
            # Re-check fresh measured opening at the actual send boundary.
            if not self._measurement_fresh() or not self._is_explicit_open(queued[1]):
                self._fault('Queued opening no longer has fresh compatible feedback')
                return
            self._dispatch('move', queued[1], sequence=queued[0])
            return
        self._status = 'completed' if success else 'not_reached'
        self._error = None
        if (active['kind'] == 'grasp'
                and active['parameters']['width'] <= self.config['close_trigger_width_m']):
            self._state = 'holding'
        else:
            self._state = 'idle'
        if active['kind'] == 'move':
            self._close_latched = False
        self._notify()

    def validate_goal(self, kind, parameters):
        """Check the exact native command before sending; never clamp or retry.

        A command within these configured bounds still needs independently
        verified actuator support and accessory strength. Move has no force
        field; only outward release is issued by this policy.
        """
        fields = {'width', 'speed'}
        if kind == 'grasp':
            fields |= {'force', 'epsilon_inner', 'epsilon_outer'}
        if (kind not in ('move', 'grasp') or set(parameters) != fields
                or not all(_finite_number(value) for value in parameters.values())):
            raise ValueError('Invalid gripper command fields or non-finite values')
        if not 0.0 <= parameters['width'] <= self.max_width:
            raise ValueError('Gripper command width exceeds configured travel')
        if not (0.0 < parameters['speed'] <= self.config['command_speed_limit_m_s']
                and parameters['speed'] == self.config['command_speed_m_s']):
            raise ValueError('Gripper command speed differs from the configured bounded speed')
        if kind == 'grasp':
            for field, setting in (('force', 'grasp_force_n'),
                                   ('epsilon_inner', 'grasp_epsilon_inner_m'),
                                   ('epsilon_outer', 'grasp_epsilon_outer_m')):
                if parameters[field] != self.config[setting]:
                    raise ValueError('Grasp command differs from configuration: ' + field)
            if not 0.0 < parameters['force'] <= self.config['grasp_force_limit_n']:
                raise ValueError('Grasp command exceeds configured force limit')

    def tick(self):
        """A stalled cancellation faults without dispatching a concurrent goal."""
        active = self._active
        if (active is not None and active['cancel_sent']
                and not active['cancel_failed']
                and self._now() - active['cancel_started'] > 2.0
                and (not active['cancel_done'] or active['terminal'] is None)):
            active['cancel_failed'] = True
            self._queued_open = None
            self._fault('Grasp cancellation timed out; waiting for old goal terminal')

    def _fault(self, message):
        if self._active is not None:
            self._active['fault_latched'] = True
            self._active['fault_message'] = message
            self._queued_open = None
        self._state = 'fault'
        self._status = 'error'
        self._error = message
        self._notify()

    def snapshot(self):
        accepted = self._accepted or {}
        result = self._result or {}
        is_grasp = accepted.get('kind') == 'grasp'
        return {
            'schema_version': 2, 'control_policy': POLICY_NAME,
            'control_state': self._state, 'sequence': self._sequence,
            'accepted_sequence': accepted.get('sequence'), 'stamp_ns': int(self._stamp_ns()),
            'command_stamp_ns': accepted.get('stamp_ns'),
            'target_width_m': accepted.get('width'), 'speed_m_s': accepted.get('speed'),
            'max_width_m': self.max_width, 'valid': self._accepted is not None,
            'status': self._status, 'requested_width_m': self._requested_width,
            'requested_speed_m_s': self.speed if self._requested_width is not None else None,
            'error': self._error, 'result_sequence': result.get('sequence'),
            'result_status': result.get('status'), 'result_success': result.get('success'),
            'result_error': result.get('error'),
            'result_cancellation_reason': self._result_cancellation_reason,
            'command_type': accepted.get('kind'),
            'force_n': accepted.get('force') if is_grasp else None,
            'epsilon_inner_m': accepted.get('epsilon_inner') if is_grasp else None,
            'epsilon_outer_m': accepted.get('epsilon_outer') if is_grasp else None,
            'policy_config': dict(self.config),
        }
