"""CPU-only execution gate. No ROS, camera, model, or network side effects.

Clock arguments are *this computer's* time.monotonic(), never the cloud clock.
The gate limits trajectories; it is not a collision-avoidance planner.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


class SafetyError(ValueError):
    pass


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SafetyError(f"{name} must be a finite number")
    return float(value)


def _vector(values, length, name):
    if not isinstance(values, (tuple, list)) or len(values) != length:
        raise SafetyError(f"{name} must contain exactly {length} numbers")
    return [_number(x, name) for x in values]


@dataclass(frozen=True)
class Command:
    q: tuple
    width_m: float | None


class ExecutionGate:
    def __init__(self, settings):
        self.cfg = dict(settings)
        self.fps = _number(settings['fps'], 'fps')
        self.hz = _number(settings['interpolation_hz'], 'interpolation_hz')
        self.lower = _vector(settings['joint_lower_rad'], 7, 'joint_lower_rad')
        self.upper = _vector(settings['joint_upper_rad'], 7, 'joint_upper_rad')
        for k in ('max_observation_age_s', 'heartbeat_timeout_s', 'action_timeout_s',
                  'state_max_age_s', 'max_start_delta_rad', 'max_tracking_error_rad',
                  'max_joint_velocity_rad_s', 'max_joint_acceleration_rad_s2',
                  'gripper_speed_m_s', 'gripper_deadband_m', 'gripper_action_timeout_s'):
            if _number(settings[k], k) <= 0:
                raise SafetyError(f'{k} must be positive')
        # Reaching a smoothly planned point can require a larger raw goal error
        # than a single model time step may jump. Keep these bounds independent.
        self.waypoint_delta = _number(settings.get('max_waypoint_delta_rad', .15),
                                      'max_waypoint_delta_rad')
        if self.waypoint_delta <= 0:
            raise SafetyError('max_waypoint_delta_rad must be positive')
        if self.fps != 15 or not 50 <= self.hz <= 200:
            raise SafetyError('Use training FPS=15 and interpolation_hz in [50, 200]')
        self.prefix = settings['prefix_steps']
        if type(self.prefix) is not int or not 1 <= self.prefix <= 6:
            raise SafetyError('prefix_steps must be 1..6 (at most 0.4 seconds)')
        if not (0 <= settings['width_min_m'] < settings['width_max_m'] <= .08):
            raise SafetyError('Gripper width must be within [0, 0.08] meters')
        # Legacy configs may retain width_clip_tolerance_m; it is ignored.
        # Actuator width uses physical projection, independently of a gross
        # model/unit sanity envelope derived from one complete configured stroke.
        if settings['gripper_speed_m_s'] != .04:
            raise SafetyError('This Move deployment uses the training speed 0.04 m/s')
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise SafetyError('Joint bounds are unordered')
        if (settings['action_timeout_s'] > 1.0 or settings['heartbeat_timeout_s'] > 1.0
                or settings['max_observation_age_s'] > 1.0):
            raise SafetyError('Watchdog timeouts may not exceed 1 second')
        self.measurement = None
        self.measured_velocity = None
        self.arm_stamp = self.width_stamp = None
        self.measurement_sequence = 0
        self.width = None
        self.session_id = self.model_id = None
        self.armed = self.arming = False
        self.fault = None
        self.q = self.velocity = None
        self.last_tick = self.arm_time = None
        self.last_heartbeat = self.camera_stamp = None
        self.last_submit = self.capture_stamp = None
        self.actions = None
        self.first_action_index = 0
        self.execution_end = None
        self.last_sequence = self.last_request_id = -1
        self.last_capture = float('-inf')

    def _check_q(self, q):
        for i, (x, lo, hi) in enumerate(zip(q, self.lower, self.upper)):
            if not lo <= x <= hi:
                raise SafetyError(f'joint {i + 1} target {x:.5f} outside [{lo}, {hi}]')

    def update_arm(self, q, velocity, stamp):
        q, velocity = _vector(q, 7, 'q'), _vector(velocity, 7, 'dq')
        self._check_q(q)
        stamp = _number(stamp, 'arm source time')
        if self.arm_stamp is not None and stamp <= self.arm_stamp:
            raise SafetyError('Arm source time did not advance')
        self.measurement, self.measured_velocity, self.arm_stamp = q, velocity, stamp
        self.measurement_sequence += 1

    def update_width(self, width, stamp):
        width, stamp = _number(width, 'width'), _number(stamp, 'gripper source time')
        if not self.cfg['width_min_m'] <= width <= self.cfg['width_max_m']:
            raise SafetyError('Measured total gripper width outside configured range')
        if self.width_stamp is not None and stamp <= self.width_stamp:
            raise SafetyError('Gripper source time did not advance')
        self.width, self.width_stamp = width, stamp

    def state(self, now):
        aa = math.inf if self.arm_stamp is None else now - self.arm_stamp
        ga = math.inf if self.width_stamp is None else now - self.width_stamp
        ready = (self.measurement is not None and self.width is not None
                 and 0 <= aa <= self.cfg['state_max_age_s']
                 and 0 <= ga <= self.cfg.get('gripper_max_age_s', .5))
        return dict(type='state', version=1,
                    state=None if self.measurement is None or self.width is None
                    else list(self.measurement) + [self.width],
                    state_age_s=aa if math.isfinite(aa) else None,
                    gripper_age_s=ga if math.isfinite(ga) else None,
                    ready=bool(ready), armed=self.armed, arming=self.arming,
                    fault=self.fault, session_id=self.session_id,
                    state_sequence=self.measurement_sequence,
                    armed_monotonic=self.arm_time if self.armed else None)

    def _fresh_state(self, now):
        if not self.state(now)['ready']:
            raise SafetyError('Fresh arm AND measured gripper state are required')

    def _session(self, session_id):
        if not isinstance(session_id, str) or session_id != self.session_id:
            raise SafetyError('Wrong or missing deployment session_id')

    def arm(self, session_id, model_id, now):
        if self.armed or self.arming:
            raise SafetyError('Stop the current session before arming another')
        if not isinstance(session_id, str) or not 8 <= len(session_id) <= 128:
            raise SafetyError('A new session_id string is required')
        if session_id == self.session_id:
            raise SafetyError('Re-arm requires a NEW session_id')
        if not isinstance(model_id, str) or not model_id or len(model_id) > 256:
            raise SafetyError('A pinned model_id is required')
        self._fresh_state(now)
        if max(abs(x) for x in self.measured_velocity) > .05:
            raise SafetyError('Arm must be stationary before activation (|dq| <= 0.05 rad/s)')
        self.session_id, self.model_id = session_id, model_id
        self.fault = None
        self.arming = True
        self.q, self.velocity = list(self.measurement), [0.] * 7
        self.arm_time = self.last_tick = self.last_heartbeat = now
        self.camera_stamp = None
        self.actions = self.last_submit = self.capture_stamp = None
        self.first_action_index, self.execution_end = 0, None
        self.last_sequence = self.last_request_id = -1
        self.last_capture = float('-inf')
        return tuple(self.q)

    def activated(self, now):
        if not self.arming or self.fault:
            raise SafetyError('Activation is no longer authorized')
        self._fresh_state(now)
        if max(abs(a - b) for a, b in zip(self.q, self.measurement)) > .03:
            raise SafetyError('Arm moved while activating; stop and explicitly re-arm')
        self.arming, self.armed = False, True
        self.arm_time = self.last_tick = self.last_heartbeat = now

    def heartbeat(self, session_id, camera_ok, camera_capture_monotonic, now):
        self._session(session_id)
        if not self.armed and not self.arming:
            raise SafetyError('Session is not armed')
        if camera_ok is not True:
            raise SafetyError('Camera capture is unhealthy')
        stamp = _number(camera_capture_monotonic, 'camera_capture_monotonic')
        age = now - stamp
        if age < 0 or age > self.cfg.get('camera_max_age_s', .20):
            raise SafetyError('Camera capture is stale or in the future')
        if self.camera_stamp is not None and stamp < self.camera_stamp:
            raise SafetyError('Camera capture time went backwards')
        self.last_heartbeat, self.camera_stamp = now, stamp

    def submit(self, packet, now):
        self._session(packet.get('session_id'))
        if not self.armed or self.fault:
            raise SafetyError('Explicit successful arm is required before submit')
        self._fresh_state(now)
        if packet.get('model_id') != self.model_id or packet.get('version') != 1 or packet.get('fps') != 15:
            raise SafetyError('Action model_id/version/FPS does not match armed session')
        sequence, request_id = packet.get('capture_seq'), packet.get('request_id')
        if type(sequence) is not int or sequence <= self.last_sequence:
            raise SafetyError('Duplicate or out-of-order capture_seq')
        if type(request_id) is not int or request_id <= self.last_request_id:
            raise SafetyError('Duplicate or out-of-order request_id')
        capture = _number(packet.get('capture_monotonic'), 'capture_monotonic')
        if capture < self.arm_time or capture <= self.last_capture or not 0 <= now - capture <= self.cfg['max_observation_age_s']:
            raise SafetyError('Observation is stale, before arm, reordered, or in the future')
        if self.camera_stamp is None:
            raise SafetyError('Fresh camera heartbeat required before actions')
        self._check_watchdogs(now)
        actions = packet.get('actions')
        if not isinstance(actions, list) or len(actions) != 50:
            raise SafetyError('PI05 response must contain exactly 50 actions')
        actions = [_vector(row, 8, 'action') for row in actions]
        # Determine the committed window before validation/reporting, while
        # preserving all 50 raw model rows for diagnostics and validation.
        first = math.ceil(max(0., now - capture) * self.fps - 1e-9)
        if first >= len(actions):
            raise SafetyError('Entire predicted trajectory has already elapsed')
        prefix_stop = min(first+self.prefix, len(actions))
        lower, upper = self.cfg['width_min_m'], self.cfg['width_max_m']
        span = upper-lower
        raw_width_min = min(row[7] for row in actions)
        raw_width_max = max(row[7] for row in actions)
        width_out_of_range_steps = []
        for index, row in enumerate(actions):
            self._check_q(row[:7])
            in_window = first <= index < prefix_stop
            # Reject scale/unit/model extremes anywhere in the raw chunk. This
            # is a plausibility check, not a contact-force or collision limit.
            if not lower-span <= row[7] <= upper+span:
                raise SafetyError(
                    'Gross gripper target outside stroke-derived sanity envelope: '
                    f'request_id={request_id}, index={index}, value={row[7]:.9g}m, '
                    f'raw_min={raw_width_min:.9g}, raw_max={raw_width_max:.9g}, '
                    f'physical_min={lower:.9g}, physical_max={upper:.9g}, '
                    f'sanity_min={lower-span:.9g}, sanity_max={upper+span:.9g}, '
                    f'start_index={first}, in_execution_window={in_window}'
                )
            if not lower <= row[7] <= upper:
                width_out_of_range_steps.append(dict(
                    index=index, raw=row[7], overrun_m=max(lower-row[7], row[7]-upper),
                    in_execution_window=in_window))
        # Model timestamps are relative to capture, not to cloud response receipt.
        # Start at the next original 15 Hz waypoint; never replay action zero after
        # transport/inference latency. The scalar width follows the same index.
        start_deltas = [a - b for a, b in zip(actions[first][:7], self.measurement)]
        joint = max(range(7), key=lambda i: abs(start_deltas[i]))
        if abs(start_deltas[joint]) > self.cfg['max_start_delta_rad']:
            raise SafetyError(
                'First executed target jumps too far from measured arm: '
                f'J{joint + 1} delta={start_deltas[joint]:.5f}rad, '
                f'limit={self.cfg["max_start_delta_rad"]:.5f}rad, '
                f'target={actions[first][joint]:.5f}, measured={self.measurement[joint]:.5f}, '
                f'filtered={self.q[joint]:.5f}, observation_age={now-capture:.3f}s, '
                f'start_index={first}, request_id={request_id}, '
                f'all_deltas={[round(value, 5) for value in start_deltas]}'
            )
        # A gross discontinuity is a contract failure, not a target to silently clamp.
        for previous, row in zip(actions, actions[1:]):
            if max(abs(a - b) for a, b in zip(previous[:7], row[:7])) > self.waypoint_delta:
                raise SafetyError('Discontinuous predicted joint trajectory')
        # Only the actually committed prefix becomes actuator commands. Copy
        # rows again so the raw 50-step model chunk and caller packet stay intact.
        prefix_actions = [list(row) for row in actions[first:prefix_stop]]
        width_clipped_steps, width_saturation = [], []
        for index, row in enumerate(prefix_actions, start=first):
            raw = row[7]
            applied = max(lower, min(upper, raw))
            if applied != raw:
                row[7] = applied
                width_clipped_steps.append(index)
                width_saturation.append(dict(index=index, raw=raw, applied=applied,
                    overrun_m=abs(applied-raw), request_id=request_id,
                    model_id=self.model_id, capture_monotonic=capture))
        execution_end = min(now + self.prefix / self.fps,
                            capture + (first + len(prefix_actions)) / self.fps,
                            capture + self.cfg['max_observation_age_s'])
        if execution_end <= now:
            raise SafetyError('No unexpired execution horizon remains')
        self.actions = prefix_actions
        self.first_action_index = first
        self.last_submit, self.capture_stamp, self.execution_end = now, capture, execution_end
        self.last_sequence, self.last_request_id, self.last_capture = sequence, request_id, capture
        return dict(type='accepted', version=1, capture_seq=sequence,
                    start_index=first, execute_steps=len(self.actions),
                    horizon_s=self.execution_end - now,
                    width_clipped_steps=width_clipped_steps,
                    width_saturation=width_saturation,
                    raw_width_min=raw_width_min, raw_width_max=raw_width_max,
                    width_out_of_range_steps=width_out_of_range_steps)

    def _check_watchdogs(self, now):
        self._fresh_state(now)
        if self.last_heartbeat is None or not 0 <= now - self.last_heartbeat <= self.cfg['heartbeat_timeout_s']:
            raise SafetyError('Local client heartbeat timed out')
        if self.camera_stamp is not None and not 0 <= now - self.camera_stamp <= self.cfg.get('camera_max_age_s', .20):
            raise SafetyError('Camera capture stopped or became stale')
        activity = self.arm_time if self.last_submit is None else self.last_submit
        if now - activity > self.cfg['action_timeout_s']:
            raise SafetyError('No fresh model action within local action timeout')
        if self.capture_stamp is not None and now - self.capture_stamp > self.cfg['max_observation_age_s']:
            raise SafetyError('Accepted observation/action deadline expired')

    def stop(self, reason='Stopped by user'):
        self.armed = self.arming = False
        self.actions = None
        self.execution_end = None
        self.fault = str(reason)

    def tick(self, now):
        if self.arming:
            self._fresh_state(now)
            if now - self.arm_time > 2.0:
                raise SafetyError('Controller activation timed out')
            return Command(tuple(self.q), None)
        if not self.armed:
            return None
        self._check_watchdogs(now)
        dt = now - self.last_tick
        if dt < 0 or dt > .05:
            raise SafetyError('Local interpolation timer stalled (>50 ms)')
        if dt == 0:
            return Command(tuple(self.q), None)
        if max(abs(a - b) for a, b in zip(self.q, self.measurement)) > self.cfg['max_tracking_error_rad']:
            raise SafetyError('Measured arm is not tracking the commanded trajectory')
        target = None
        if self.actions is not None and now < self.execution_end:
            original_index = math.floor((now - self.capture_stamp) * self.fps + 1e-9)
            index = max(0, original_index - self.first_action_index)
            if index < len(self.actions):
                target = self.actions[index]
        vmax, amax = self.cfg['max_joint_velocity_rad_s'], self.cfg['max_joint_acceleration_rad_s2']
        new_q, new_v = [], []
        for i in range(7):
            distance = target[i] - self.q[i] if target is not None else 0.
            desired = math.copysign(min(vmax, math.sqrt(2 * amax * abs(distance))), distance) if distance else 0.
            # Trapezoidal position integration: both speed and acceleration are bounded.
            v = self.velocity[i] + max(-amax * dt, min(amax * dt, desired - self.velocity[i]))
            q = self.q[i] + (self.velocity[i] + v) * .5 * dt
            new_q.append(q)
            new_v.append(v)
        self._check_q(new_q)
        self.q, self.velocity, self.last_tick = new_q, new_v, now
        return Command(tuple(new_q), None if target is None else target[7])
