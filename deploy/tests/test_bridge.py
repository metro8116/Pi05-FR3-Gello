"""Execution invariants, CPU only: never initializes ROS or contacts hardware."""
import math
import json
from pathlib import Path
import sys
import threading
import types
import unittest
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge_core import ExecutionGate, SafetyError
from config import load_config
from fr3_policy_bridge import GRIPPER_STOP_TIMEOUT_S, create_node, main, move_result, source_time
from protocol import ProtocolError


POSE = [0., -.5, 0., -1.5, 0., 1.5, 0.]
SESSION = '9c032a3b-4998-49fb-8c18-d9065b854e18'


class GateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()['execution']
        # Keep the original boundary tests independent of deployment defaults.
        self.cfg.update(max_start_delta_rad=.15, max_waypoint_delta_rad=.15,
                        max_observation_age_s=.8, action_timeout_s=.8)
        self.gate = ExecutionGate(self.cfg)
        self.refresh(100.)

    def refresh(self, now, q=None):
        self.gate.update_arm(POSE if q is None else q, [0.] * 7, now - .0001)
        self.gate.update_width(.04, now - .0001)

    def arm(self):
        hold = self.gate.arm(SESSION, 'model-sha', 100.)
        self.assertEqual(hold, tuple(POSE))
        self.assertFalse(self.gate.armed)
        self.assertTrue(self.gate.arming)
        self.gate.activated(100.01)
        self.gate.heartbeat(SESSION, True, 100.02, 100.02)

    def packet(self, **changes):
        p = dict(type='submit', version=1, fps=15, model_id='model-sha',
                 session_id=SESSION, capture_seq=1, request_id=1,
                 capture_monotonic=100.02, actions=[POSE + [.02] for _ in range(50)])
        p.update(changes)
        return p

    def test_starts_inactive_and_no_command(self):
        self.assertFalse(self.gate.armed)
        self.assertIsNone(self.gate.tick(100.01))
        self.assertTrue(self.gate.state(100.01)['ready'])

    def test_explicit_arming_holds_measured_pose_without_gripper_move(self):
        q = self.gate.arm(SESSION, 'model-sha', 100.)
        c = self.gate.tick(100.01)
        self.assertEqual(c.q, q)
        self.assertIsNone(c.width_m)
        self.assertFalse(self.gate.armed)

    def test_no_implicit_reposition_while_activating(self):
        self.gate.arm(SESSION, 'model-sha', 100.)
        q = list(POSE)
        q[0] += .04
        self.refresh(100.01, q)
        with self.assertRaisesRegex(SafetyError, 'moved while activating'):
            self.gate.activated(100.02)

    def test_stale_or_moving_arm_cannot_arm(self):
        with self.assertRaisesRegex(SafetyError, 'Fresh'):
            self.gate.arm(SESSION, 'model-sha', 101.)
        self.gate.update_arm(POSE, [.06] * 7, 100.001)
        with self.assertRaisesRegex(SafetyError, 'stationary'):
            self.gate.arm(SESSION, 'model-sha', 100.01)

    def test_gripper_actual_measurement_required(self):
        gate = ExecutionGate(self.cfg)
        gate.update_arm(POSE, [0.] * 7, 100.)
        with self.assertRaisesRegex(SafetyError, 'gripper'):
            gate.arm(SESSION, 'model-sha', 100.01)

    def test_submit_full_chunk_executes_only_small_prefix(self):
        self.arm()
        result = self.gate.submit(self.packet(), 100.03)
        self.assertEqual(result['execute_steps'], 6)
        self.assertAlmostEqual(result['horizon_s'], .4)
        self.assertEqual(result['width_clipped_steps'], [])
        self.assertEqual(len(self.gate.actions), 6)

    def test_delayed_response_starts_at_original_capture_time_index(self):
        self.arm()
        self.refresh(100.22)
        self.gate.heartbeat(SESSION, True, 100.22, 100.22)
        p = self.packet()
        for i, row in enumerate(p['actions']):
            row[7] = i * .001
        result = self.gate.submit(p, 100.22)
        self.assertEqual(result['start_index'], 3)
        self.assertLessEqual(result['horizon_s'], .4 + 1e-9)
        # Keep timer current in this independent timestamp alignment test.
        self.gate.last_tick = 100.21
        first = self.gate.tick(100.22)
        self.assertEqual(first.width_m, .003)
        self.refresh(100.24)
        self.gate.heartbeat(SESSION, True, 100.24, 100.24)
        self.gate.tick(100.24)
        self.refresh(100.26)
        self.gate.heartbeat(SESSION, True, 100.26, 100.26)
        # Another 40 ms later: still action 3, not replaying 0 at receipt + tick.
        self.assertEqual(self.gate.tick(100.26).width_m, .003)
        self.refresh(100.29)
        self.gate.heartbeat(SESSION, True, 100.29, 100.29)
        self.assertEqual(self.gate.tick(100.29).width_m, .004)

    def test_first_executed_target_jump_checked_after_latency_skip(self):
        self.arm()
        self.refresh(100.22)
        self.gate.heartbeat(SESSION, True, 100.22, 100.22)
        p = self.packet()
        for i, row in enumerate(p['actions']):
            row[0] = min(i * .06, .18)
        self.assertEqual(p['actions'][0][0], 0.)
        with self.assertRaisesRegex(SafetyError, 'First executed'):
            self.gate.submit(p, 100.22)

    def test_reject_nonfinite_or_wrong_shape_even_outside_prefix(self):
        self.arm()
        for bad in ([], [[0.] * 8] * 49, [[0.] * 7] * 50):
            with self.assertRaises(SafetyError):
                self.gate.submit(self.packet(actions=bad), 100.03)
        for value in (math.nan, math.inf, True, '0'):
            p = self.packet()
            p['actions'][49][0] = value
            with self.assertRaises(SafetyError):
                self.gate.submit(p, 100.03)

    def test_all_prediction_rows_checked_for_joint_bounds_and_gross_width_errors(self):
        self.arm()
        for index, value in ((0, 3.), (3, 0.), (5, 0.), (7, 80.), (7, -.08001), (7, .16001), (7, math.nan)):
            p = self.packet()
            p['actions'][49][index] = value
            with self.assertRaises(SafetyError):
                self.gate.submit(p, 100.03)

    def test_tiny_predicted_width_overshoots_limited_and_reported_without_mutating_packet(self):
        self.arm()
        p = self.packet(capture_monotonic=100.03)
        p['actions'][0][7] = -.0005
        p['actions'][1][7] = .0805
        p['actions'][49][7] = -.001
        accepted = self.gate.submit(p, 100.03)
        self.assertEqual(accepted['width_clipped_steps'], [0, 1])
        self.assertEqual(self.gate.actions[0][7], 0.)
        self.assertEqual(self.gate.actions[1][7], .08)
        self.assertEqual(p['actions'][0][7], -.0005)
        self.assertEqual(p['actions'][1][7], .0805)
        self.assertEqual(p['actions'][49][7], -.001)
        self.assertEqual([entry['index'] for entry in accepted['width_out_of_range_steps']], [0, 1, 49])
        self.assertFalse(accepted['width_out_of_range_steps'][-1]['in_execution_window'])
        self.assertEqual(accepted['raw_width_min'], -.001)
        self.assertEqual(accepted['raw_width_max'], .0805)

    def test_width_clipping_never_applies_to_actual_measurement_or_joint_targets(self):
        self.arm()
        for width in (-.0005, .0805):
            with self.assertRaisesRegex(SafetyError, 'Measured'):
                self.gate.update_width(width, 100.02)
        p = self.packet()
        p['actions'][49][0] = self.cfg['joint_upper_rad'][0] + .0005
        with self.assertRaisesRegex(SafetyError, 'joint 1'):
            self.gate.submit(p, 100.03)

    def test_legacy_width_tolerance_is_ignored_by_physical_projection(self):
        for tolerance in (None, 0., .002, .01):
            with self.subTest(legacy_tolerance=tolerance):
                cfg = dict(self.cfg)
                if tolerance is None:
                    cfg.pop('width_clip_tolerance_m', None)
                else:
                    cfg['width_clip_tolerance_m'] = tolerance
                self.gate = ExecutionGate(cfg)
                self.refresh(100.)
                self.arm()
                p = self.packet(capture_monotonic=100.03)
                p['actions'][0][7], p['actions'][1][7] = .09, -.01
                result = self.gate.submit(p, 100.03)
                self.assertEqual(result['width_clipped_steps'], [0, 1])
                self.assertEqual(self.gate.actions[0][7], .08)
                self.assertEqual(self.gate.actions[1][7], 0.)

    def test_width_projection_only_touches_latency_aligned_prefix_with_raw_provenance(self):
        self.arm()
        self.refresh(100.22)
        self.gate.heartbeat(SESSION, True, 100.22, 100.22)
        p = self.packet()
        for index, width in ((0, -.005), (2, .087), (3, -.004), (8, .084), (9, .09), (49, -.006)):
            p['actions'][index][7] = width
        original = [list(row) for row in p['actions']]
        result = self.gate.submit(p, 100.22)
        self.assertEqual(result['start_index'], 3)
        self.assertEqual(result['width_clipped_steps'], [3, 8])
        self.assertEqual(len(self.gate.actions), 6)
        self.assertEqual(self.gate.actions[0][7], 0.)
        self.assertEqual(self.gate.actions[-1][7], .08)
        self.assertEqual(p['actions'], original)
        reported = result['width_out_of_range_steps']
        self.assertEqual([entry['index'] for entry in reported], [0, 2, 3, 8, 9, 49])
        self.assertEqual([entry['index'] for entry in reported if entry['in_execution_window']], [3, 8])
        self.assertEqual(result['raw_width_min'], -.006)
        self.assertEqual(result['raw_width_max'], .09)
        for entry, index in zip(result['width_saturation'], (3, 8)):
            self.assertEqual(entry['index'], index)
            self.assertEqual(entry['raw'], original[index][7])
            self.assertAlmostEqual(entry['overrun_m'], .004)
            self.assertEqual(entry['request_id'], p['request_id'])
            self.assertEqual(entry['model_id'], p['model_id'])
            self.assertEqual(entry['capture_monotonic'], p['capture_monotonic'])
        self.assertEqual([entry['applied'] for entry in result['width_saturation']], [0., .08])

    def test_unexecuted_future_width_overshoot_is_reported_without_rejecting_or_saturating(self):
        self.arm()
        p = self.packet()
        p['actions'][49][7] = .083
        result = self.gate.submit(p, 100.03)
        self.assertEqual(result['width_clipped_steps'], [])
        self.assertEqual(result['width_saturation'], [])
        self.assertEqual(result['width_out_of_range_steps'][0]['index'], 49)
        self.assertFalse(result['width_out_of_range_steps'][0]['in_execution_window'])
        self.assertEqual(p['actions'][49][7], .083)

    def test_gross_width_rejection_has_original_index_and_execution_window_context(self):
        self.arm()
        p = self.packet()
        p['actions'][49][7] = 80.
        with self.assertRaises(SafetyError) as error:
            self.gate.submit(p, 100.03)
        message = str(error.exception)
        for fragment in ('Gross gripper', 'request_id=1', 'index=49', 'value=80m',
                         'raw_min=', 'raw_max=80', 'sanity_min=-0.08', 'sanity_max=0.16',
                         'start_index=1', 'in_execution_window=False'):
            self.assertIn(fragment, message)
        self.assertIsNone(self.gate.actions)
        self.assertEqual(p['actions'][49][7], 80.)

    def test_width_nonfinite_and_nonnumeric_are_rejected_even_in_unexecuted_future(self):
        self.arm()
        for value in (math.nan, math.inf, -math.inf, True, '0.03'):
            with self.subTest(value=value):
                p = self.packet()
                p['actions'][49][7] = value
                with self.assertRaises(SafetyError):
                    self.gate.submit(p, 100.03)
                self.assertIsNone(self.gate.actions)

    def test_large_initial_jump_and_future_discontinuity_fail(self):
        self.arm()
        p = self.packet(capture_monotonic=100.03)
        for row in p['actions']:
            row[0] = .2
        with self.assertRaisesRegex(SafetyError, 'jumps'):
            self.gate.submit(p, 100.03)
        p = self.packet()
        p['actions'][49][0] = .2
        with self.assertRaisesRegex(SafetyError, 'Discontinuous'):
            self.gate.submit(p, 100.03)

    def test_030_start_limit_accepts_smooth_020_offset(self):
        self.cfg.update(max_start_delta_rad=.30, max_waypoint_delta_rad=.15)
        self.gate = ExecutionGate(self.cfg)
        self.refresh(100.)
        self.arm()
        p = self.packet()
        for index, row in enumerate(p['actions']):
            row[0] = .20 + index * .0005
        result = self.gate.submit(p, 100.03)
        self.assertEqual(result['start_index'], 1)
        self.assertEqual(result['execute_steps'], 6)
        self.assertAlmostEqual(self.gate.actions[0][0], .2005)

    def test_030_start_limit_still_rejects_target_above_030(self):
        self.cfg.update(max_start_delta_rad=.30, max_waypoint_delta_rad=.15)
        self.gate = ExecutionGate(self.cfg)
        self.refresh(100.)
        self.arm()
        p = self.packet()
        for row in p['actions']:
            row[0] = .3001
        with self.assertRaisesRegex(SafetyError, 'jumps'):
            self.gate.submit(p, 100.03)
        self.assertIsNone(self.gate.actions)

    def test_030_start_limit_does_not_relax_015_adjacent_waypoint_limit(self):
        self.cfg.update(max_start_delta_rad=.30, max_waypoint_delta_rad=.15)
        self.gate = ExecutionGate(self.cfg)
        self.refresh(100.)
        self.arm()
        p = self.packet()
        # The executed first point is exactly measured pose. Only a later
        # adjacent pair jumps by .20, isolating the waypoint check from start.
        for row in p['actions'][20:]:
            row[0] = .20
        with self.assertRaisesRegex(SafetyError, 'Discontinuous'):
            self.gate.submit(p, 100.03)
        self.assertIsNone(self.gate.actions)

    def test_040_start_limit_accepts_smooth_0346_offset(self):
        self.cfg.update(max_start_delta_rad=.40, max_waypoint_delta_rad=.15,
                        max_observation_age_s=1., action_timeout_s=.8)
        self.gate = ExecutionGate(self.cfg)
        self.refresh(100.)
        self.arm()
        p = self.packet()
        for index, row in enumerate(p['actions']):
            row[0] = .346 + index*.0005
        accepted = self.gate.submit(p, 100.03)
        self.assertEqual(accepted['start_index'], 1)
        self.assertAlmostEqual(self.gate.actions[0][0], .3465)

    def test_040_start_limit_still_rejects_above_040(self):
        self.cfg.update(max_start_delta_rad=.40, max_waypoint_delta_rad=.15,
                        max_observation_age_s=1., action_timeout_s=.8)
        self.gate = ExecutionGate(self.cfg)
        self.refresh(100.)
        self.arm()
        p = self.packet()
        for row in p['actions']:
            row[0] = .4001
        with self.assertRaisesRegex(SafetyError, 'jumps'):
            self.gate.submit(p, 100.03)
        self.assertIsNone(self.gate.actions)

    def test_040_start_limit_preserves_015_adjacent_waypoint_check(self):
        self.cfg.update(max_start_delta_rad=.40, max_waypoint_delta_rad=.15,
                        max_observation_age_s=1., action_timeout_s=.8)
        self.gate = ExecutionGate(self.cfg)
        self.refresh(100.)
        self.arm()
        p = self.packet()
        for row in p['actions'][20:]:
            row[0] = .20
        with self.assertRaisesRegex(SafetyError, 'Discontinuous'):
            self.gate.submit(p, 100.03)
        self.assertIsNone(self.gate.actions)

    def test_one_second_ttl_accepts_fresh_replacement_when_old_capture_age_is_0815(self):
        for deadline in (.8, 1.):
            with self.subTest(observation_deadline=deadline):
                self.cfg.update(max_start_delta_rad=.40, max_waypoint_delta_rad=.15,
                                max_observation_age_s=deadline, action_timeout_s=.8)
                self.gate = ExecutionGate(self.cfg)
                self.refresh(100.)
                self.arm()
                # First reply arrived at capture age .485s; replacement arrives
                # .330s later, when that accepted capture is .815s old.
                self.refresh(100.505)
                self.gate.heartbeat(SESSION, True, 100.505, 100.505)
                self.gate.submit(self.packet(capture_monotonic=100.02), 100.505)
                self.refresh(100.835)
                self.gate.heartbeat(SESSION, True, 100.835, 100.835)
                fresh = self.packet(capture_seq=2, request_id=2, capture_monotonic=100.8)
                self.assertAlmostEqual(100.835-self.gate.capture_stamp, .815)
                self.assertAlmostEqual(100.835-self.gate.last_submit, .330)
                if deadline == .8:
                    with self.assertRaisesRegex(SafetyError, 'Accepted observation/action deadline expired'):
                        self.gate.submit(fresh, 100.835)
                else:
                    result = self.gate.submit(fresh, 100.835)
                    self.assertEqual(result['capture_seq'], 2)
                    self.assertAlmostEqual(self.gate.capture_stamp, 100.8)

    def test_one_second_ttl_still_rejects_expired_old_capture_before_replacement(self):
        self.cfg.update(max_start_delta_rad=.40, max_waypoint_delta_rad=.15,
                        max_observation_age_s=1., action_timeout_s=.8)
        self.gate = ExecutionGate(self.cfg)
        self.refresh(100.)
        self.arm()
        self.refresh(100.505)
        self.gate.heartbeat(SESSION, True, 100.505, 100.505)
        self.gate.submit(self.packet(capture_monotonic=100.02), 100.505)
        self.refresh(101.025)
        self.gate.heartbeat(SESSION, True, 101.025, 101.025)
        self.assertGreater(101.025-self.gate.capture_stamp, 1.)
        self.assertLess(101.025-self.gate.last_submit, .8)
        p = self.packet(capture_seq=2, request_id=2, capture_monotonic=101.01)
        with self.assertRaisesRegex(SafetyError, 'Accepted observation/action deadline expired'):
            self.gate.submit(p, 101.025)

    def test_one_second_capture_ttl_does_not_extend_08_action_submission_timeout(self):
        self.cfg.update(max_start_delta_rad=.40, max_waypoint_delta_rad=.15,
                        max_observation_age_s=1., action_timeout_s=.8)
        self.gate = ExecutionGate(self.cfg)
        self.refresh(100.)
        self.arm()
        self.gate.submit(self.packet(capture_monotonic=100.02), 100.03)
        self.refresh(100.835)
        self.gate.heartbeat(SESSION, True, 100.835, 100.835)
        self.assertLess(100.835-self.gate.capture_stamp, 1.)
        self.assertGreater(100.835-self.gate.last_submit, .8)
        p = self.packet(capture_seq=2, request_id=2, capture_monotonic=100.82)
        with self.assertRaisesRegex(SafetyError, 'No fresh model action within local action timeout'):
            self.gate.submit(p, 100.835)

    def test_session_model_version_fps_contract(self):
        self.arm()
        for fields in ({'session_id': 'wrong'}, {'model_id': 'other'}, {'version': 2}, {'fps': 30}):
            with self.assertRaises(SafetyError):
                self.gate.submit(self.packet(**fields), 100.03)

    def test_out_of_order_and_replayed_packets_rejected(self):
        self.arm()
        self.gate.submit(self.packet(), 100.03)
        for fields in ({}, {'capture_seq': 2}, {'request_id': 2},
                       {'capture_seq': 2, 'request_id': 2, 'capture_monotonic': 100.02}):
            with self.assertRaises(SafetyError):
                self.gate.submit(self.packet(**fields), 100.04)

    def test_source_age_is_local_not_cloud_clock(self):
        self.arm()
        for stamp in (99.99, 100.04, -1., math.nan):
            with self.assertRaises(SafetyError):
                self.gate.submit(self.packet(capture_monotonic=stamp), 100.03)
        self.refresh(100.90)
        self.gate.heartbeat(SESSION, True, 100.90, 100.90)
        with self.assertRaises(SafetyError):
            self.gate.submit(self.packet(capture_monotonic=100.02), 100.90)

    def test_camera_health_cannot_be_fabricated_by_heartbeat_only(self):
        self.arm()
        for args in ((False, 100.02), (True, 99.), (True, 101.)):
            with self.assertRaises(SafetyError):
                self.gate.heartbeat(SESSION, *args, now=100.03)
        self.refresh(100.23)
        with self.assertRaisesRegex(SafetyError, 'Camera'):
            self.gate.tick(100.23)

    def test_heartbeat_cannot_extend_old_model_actions(self):
        self.arm()
        self.gate.submit(self.packet(), 100.03)
        for i in range(1, 81):
            now = 100.01 + .01 * i
            self.refresh(now, self.gate.q)
            self.gate.heartbeat(SESSION, True, now, now)
            self.gate.tick(now)
        self.refresh(100.83, self.gate.q)
        self.gate.heartbeat(SESSION, True, 100.83, 100.83)
        with self.assertRaisesRegex(SafetyError, 'deadline|timeout'):
            self.gate.tick(100.83)

    def test_prefix_end_does_not_start_seventh_action(self):
        self.arm()
        p = self.packet(capture_monotonic=100.03)
        p['actions'][6] = POSE + [.07]
        self.gate.submit(p, 100.03)
        for i in range(1, 46):
            now = 100.01 + .01 * i
            self.refresh(now)
            self.gate.heartbeat(SESSION, True, now, now)
            cmd = self.gate.tick(now)
            self.assertNotEqual(cmd.width_m, .07)
        self.assertIsNone(cmd.width_m)
        self.assertEqual(cmd.q, tuple(POSE))

    def test_rate_limiter_velocity_and_acceleration_bounded_on_reversal(self):
        self.arm()
        old_v = list(self.gate.velocity)
        old_q = list(self.gate.q)
        seq = 0
        for i in range(1, 100):
            now = 100.01 + .01 * i
            self.refresh(now, old_q)
            self.gate.heartbeat(SESSION, True, now, now)
            if i in (1, 31, 61, 91):
                seq += 1
                p = self.packet(capture_seq=seq, request_id=seq, capture_monotonic=now)
                for row in p['actions']:
                    row[0] = .10 if i < 60 else -.03
                self.gate.submit(p, now)
            cmd = self.gate.tick(now)
            for q, prev, v, pv in zip(cmd.q, old_q, self.gate.velocity, old_v):
                self.assertLessEqual(abs(v), self.cfg['max_joint_velocity_rad_s'] + 1e-10)
                self.assertLessEqual(abs(v - pv), self.cfg['max_joint_acceleration_rad_s2'] * .01 + 1e-10)
                self.assertLessEqual(abs(q - prev), self.cfg['max_joint_velocity_rad_s'] * .01 + 1e-10)
            old_q, old_v = list(cmd.q), list(self.gate.velocity)

    def test_tracking_error_and_local_timer_stall_fail(self):
        self.arm()
        q = list(POSE)
        q[0] -= .26
        self.refresh(100.02, q)
        with self.assertRaisesRegex(SafetyError, 'tracking'):
            self.gate.tick(100.03)
        self.refresh(100.07)
        with self.assertRaisesRegex(SafetyError, 'timer stalled'):
            self.gate.tick(100.07)

    def test_stop_clears_commands_and_requires_new_session(self):
        self.arm()
        self.gate.submit(self.packet(), 100.03)
        self.gate.stop('user stop')
        self.assertIsNone(self.gate.tick(100.04))
        self.assertEqual(self.gate.state(100.04)['fault'], 'user stop')
        with self.assertRaisesRegex(SafetyError, 'NEW'):
            self.gate.arm(SESSION, 'model-sha', 100.04)
        self.gate.arm('0a037346-abb5-44c9-9677-671d2e727334', 'model-sha', 100.04)
        self.assertIsNone(self.gate.fault)

    def test_bad_configuration_cannot_extend_prefix_or_watchdogs(self):
        for field, value in (('prefix_steps', 50), ('action_timeout_s', 5.),
                             ('gripper_speed_m_s', .1), ('fps', 30), ('max_joint_velocity_rad_s', 0.)):
            cfg = dict(self.cfg)
            cfg[field] = value
            with self.assertRaises(SafetyError):
                ExecutionGate(cfg)


class SourceAndGripperTests(unittest.TestCase):
    def test_source_timestamp_mapping_and_replay_checks(self):
        self.assertAlmostEqual(source_time(1_000_000_000, None, 1_010_000_000, 100., .2), 99.99)
        for stamp, previous in ((0, None), (1_000, 1_000), (999, 1_000), (1_100_000_000, None)):
            with self.assertRaises(SafetyError):
                source_time(stamp, previous, 1_010_000_000, 100., .2)

    def test_move_success_and_exact_closing_not_reached(self):
        self.assertEqual(move_result(4, True, ''), 'completed')
        geometry = dict(closing=True, actual_width=.03, target_width=0., measurement_fresh=True)
        for error in ('', 'libfranka gripper: Command failed!'):
            self.assertEqual(move_result(6, False, error, **geometry), 'not_reached')

    def test_move_error_whitelist_requires_closing_and_fresh_geometry(self):
        geometry = dict(closing=True, actual_width=.03, target_width=0., measurement_fresh=True)
        for change in ({'closing': False}, {'measurement_fresh': False}, {'actual_width': 0.},
                       {'actual_width': math.nan}, {'actual_width': .09}, {'target_width': .05}):
            args = dict(geometry, **change)
            with self.assertRaises(SafetyError):
                move_result(6, False, 'libfranka gripper: Command failed!', **args)
        for status, success, error in ((6, False, 'network error'), (5, False, ''), (4, False, ''), (6, True, '')):
            with self.assertRaises(SafetyError):
                move_result(status, success, error, **geometry)


class FakeFuture:
    def __init__(self):
        self.callbacks = []
        self.value = None

    def add_done_callback(self, cb):
        self.callbacks.append(cb)

    def result(self):
        return self.value

    def complete(self, value):
        self.value = value
        for cb in list(self.callbacks):
            cb(self)


class FakeClient:
    def __init__(self, *args):
        self.requests = []
        self.futures = []

    def service_is_ready(self):
        return True

    server_is_ready = service_is_ready

    def call_async(self, request):
        f = FakeFuture()
        self.requests.append(request)
        self.futures.append(f)
        return f

    send_goal_async = call_async


class FakeHandle:
    accepted = True

    def __init__(self):
        self.cancelled = 0
        self.result = FakeFuture()

    def cancel_goal_async(self):
        self.cancelled += 1
        return FakeFuture()

    def get_result_async(self):
        return self.result


class FakeMessage:
    def __init__(self):
        self.header = types.SimpleNamespace(stamp=None)


class FakeNode:
    def __init__(self, name):
        self.published = []
        self.clients = {}
        self.messages = []

    def create_publisher(self, *args):
        return types.SimpleNamespace(publish=self.published.append)

    def create_subscription(self, *args):
        return None

    def create_timer(self, *args):
        return None

    def create_client(self, service_type, name):
        client = FakeClient()
        self.clients[name] = client
        return client

    def get_logger(self):
        return types.SimpleNamespace(**{name: self.messages.append for name in ('info', 'warning', 'error', 'fatal')})

    def get_clock(self):
        return types.SimpleNamespace(now=lambda: types.SimpleNamespace(
            nanoseconds=100_000_000_000, to_msg=lambda: 'real-source-stamp'))


def fake_ros_modules():
    request = type('Request', (), {'STRICT': 2,
                                   '__init__': lambda self: setattr(self, 'timeout', types.SimpleNamespace(sec=0))})
    service = type('Service', (), {'Request': request})
    move = type('Move', (), {'Goal': type('Goal', (), {})})
    modules = {'rclpy': types.SimpleNamespace(try_shutdown=lambda: None),
               'rclpy.node': types.SimpleNamespace(Node=FakeNode),
               'rclpy.action': types.SimpleNamespace(ActionClient=FakeClient),
               'rclpy.qos': types.SimpleNamespace(qos_profile_sensor_data=None),
               'sensor_msgs': types.ModuleType('sensor_msgs'),
               'sensor_msgs.msg': types.SimpleNamespace(JointState=FakeMessage),
               'controller_manager_msgs': types.ModuleType('controller_manager_msgs'),
               'controller_manager_msgs.srv': types.SimpleNamespace(SwitchController=service, ListControllers=service),
               'franka_msgs': types.ModuleType('franka_msgs'),
               'franka_msgs.action': types.SimpleNamespace(Move=move),
               'std_srvs': types.ModuleType('std_srvs'),
               'std_srvs.srv': types.SimpleNamespace(Trigger=service)}
    return modules


class RosAdapterMockTests(unittest.TestCase):
    """Exercise async activation, contact diagnostics, and late goal cancellation."""
    def setUp(self):
        self.modules = patch.dict(sys.modules, fake_ros_modules())
        self.modules.start()
        self.clock = patch('fr3_policy_bridge.time.monotonic', return_value=100.)
        self.now = self.clock.start()
        self.node = create_node(load_config())
        self.node.controller_state = 'inactive'
        self.node.gate.update_arm(POSE, [0.] * 7, 99.999)
        self.node.gate.update_width(.04, 99.999)

    def tearDown(self):
        self.clock.stop()
        self.modules.stop()

    def arm(self):
        result = self.node.request(dict(type='arm', version=1, session_id=SESSION, model_id='model-sha'))
        self.assertTrue(result['arming'])
        self.assertFalse(self.node.gate.armed)
        self.assertEqual(self.node.published[-1].position, POSE)
        self.assertEqual(self.node.switch.requests[-1].strictness, 2)
        self.node.switch.futures[-1].complete(types.SimpleNamespace(ok=True))
        self.assertTrue(self.node.gate.armed)

    def test_async_arm_requires_successful_switch_and_preserves_current_pose(self):
        self.arm()
        self.assertEqual(len(self.node.move.requests), 0)
        self.assertEqual(self.node.switch.requests[-1].activate_controllers, ['joint_impedance_controller'])

    def manual_packet(self, operation_id=None, **extra):
        return dict(version=1, type='manual_open', operation_id=operation_id or str(uuid.uuid4()), **extra)

    def test_manual_open_fixed_goal_completes_without_activating_or_publishing_arm(self):
        self.assertTrue(self.node.status()['manual_open_ready'])
        packet = self.manual_packet()
        reply = self.node.request(packet)
        self.assertEqual(reply['type'], 'manual_open_started')
        self.assertEqual(reply['operation_id'], packet['operation_id'])
        self.assertEqual(self.node.move.requests[-1].width, .08)
        self.assertEqual(self.node.move.requests[-1].speed, .04)
        self.assertFalse(self.node.status()['manual_open_ready'])
        handle = FakeHandle()
        self.node.move.futures[-1].complete(handle)
        self.assertEqual(self.node.status()['manual_open']['status'], 'accepted')
        handle.result.complete(types.SimpleNamespace(status=4,
            result=types.SimpleNamespace(success=True, error='')))
        self.node.gate.update_width(.079999, 100.)
        manual = self.node.status()['manual_open']
        self.assertEqual(manual['status'], 'completed')
        self.assertEqual(manual['result_status'], 4)
        self.assertTrue(manual['success'])
        self.assertEqual(manual['error'], '')
        self.assertAlmostEqual(manual['last_width_m'], .079999)
        self.assertEqual(self.node.switch.requests, [])
        self.assertEqual(self.node.published, [])
        self.assertFalse(self.node.gate.armed)
        self.assertTrue(self.node.status()['manual_open_ready'])

    def test_manual_open_rejects_active_controller_and_custom_parameters(self):
        for field in ('width', 'speed', 'target_width_m', 'speed_m_s'):
            with self.subTest(field=field), self.assertRaises(ProtocolError):
                self.node.request(self.manual_packet(**{field: .01}))
        self.arm()
        with self.assertRaisesRegex(SafetyError, 'inactive'):
            self.node.request(self.manual_packet())
        self.assertEqual(self.node.move.requests, [])

    def test_manual_open_requires_fresh_stationary_actual_state(self):
        self.node.gate.arm_stamp = 99.
        with self.assertRaisesRegex(SafetyError, 'fresh stationary'):
            self.node.request(self.manual_packet())
        self.node.gate.arm_stamp = 99.999
        self.node.gate.width_stamp = 99.
        with self.assertRaisesRegex(SafetyError, 'fresh stationary'):
            self.node.request(self.manual_packet())
        self.node.gate.width_stamp = 99.999
        self.node.gate.measured_velocity = [.051] + [0.] * 6
        with self.assertRaisesRegex(SafetyError, 'fresh stationary'):
            self.node.request(self.manual_packet())
        self.assertEqual(self.node.move.requests, [])

    def test_manual_open_pending_blocks_model_arm_and_another_manual_goal(self):
        self.node.request(self.manual_packet())
        with self.assertRaises(SafetyError):
            self.node.request(dict(type='arm', version=1, session_id=SESSION, model_id='model-sha'))
        with self.assertRaises(SafetyError):
            self.node.request(self.manual_packet())
        self.assertEqual(len(self.node.move.requests), 1)
        self.assertEqual(self.node.switch.requests, [])

    def test_stop_before_manual_request_revokes_uuid_even_after_new_model_arm(self):
        packet = self.manual_packet()
        self.node.request(dict(type='stop', version=1, manual_operation_id=packet['operation_id']))
        with self.assertRaisesRegex(SafetyError, 'cancelled'):
            self.node.request(packet)
        self.assertEqual(self.node.move.requests, [])
        self.arm()
        self.assertIn(packet['operation_id'], self.node.cancelled_manual_ids)
        with self.assertRaisesRegex(SafetyError, 'cancelled'):
            self.node.request(packet)

    def test_stop_manual_unknown_acceptance_stops_late_goal_and_keeps_cancelled_result(self):
        packet = self.manual_packet()
        self.node.request(packet)
        acceptance = self.node.move.futures[-1]
        self.node.request(dict(type='stop', version=1, manual_operation_id=packet['operation_id']))
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.assertTrue(self.node.move_pending)
        self.assertEqual(self.node.status()['manual_open']['status'], 'cancelled')
        handle = FakeHandle()
        acceptance.complete(handle)
        self.assertEqual(handle.cancelled, 0)
        self.assertEqual(len(self.node.gripper_stop.requests), 2)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        handle.result.complete(types.SimpleNamespace(status=4,
            result=types.SimpleNamespace(success=True, error='')))
        self.assertEqual(self.node.status()['manual_open']['status'], 'cancelled')
        self.assertEqual(self.node.status()['manual_open']['result_status'], 4)
        self.assertFalse(self.node.move_pending)
        self.assertFalse(self.node.gate.armed)
        self.assertEqual(len(self.node.move.requests), 1)
        self.assertEqual(self.node.switch.requests, [])

    def test_late_old_manual_result_cannot_clear_new_manual_pending_or_replace_its_state(self):
        first = self.manual_packet()
        self.node.request(first)
        old = FakeHandle()
        self.node.move.futures[-1].complete(old)
        self.node.request(dict(type='stop', version=1, manual_operation_id=first['operation_id']))
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        second = self.manual_packet()
        self.node.request(second)
        old.result.complete(types.SimpleNamespace(status=4,
            result=types.SimpleNamespace(success=True, error='')))
        self.assertTrue(self.node.move_pending)
        self.assertTrue(self.node.move_ack_pending)
        manual = self.node.status()['manual_open']
        self.assertEqual(manual['operation_id'], second['operation_id'])
        self.assertEqual(manual['status'], 'pending')

    def test_manual_open_ack_timeout_stops_gripper_and_marks_error_while_arm_inactive(self):
        self.node.request(self.manual_packet())
        self.now.return_value = 105.01
        self.node.tick()
        self.assertEqual(self.node.status()['manual_open']['status'], 'error')
        self.assertIn('acceptance timed out', self.node.status()['manual_open']['error'])
        self.assertIn('ownership uncertain', self.node.exit_error)
        self.assertEqual(len(self.node.gripper_stop.requests), 1)
        self.assertEqual(self.node.switch.requests, [])

    def test_manual_open_accepted_result_has_its_own_five_second_timeout(self):
        self.node.request(self.manual_packet())
        self.now.return_value = 102.5
        handle = FakeHandle()
        self.node.move.futures[-1].complete(handle)
        self.now.return_value = 103.01
        self.node.gate.update_arm(POSE, [0.] * 7, 103.009)
        self.node.gate.update_width(.07, 103.009)
        self.node.tick()
        self.assertEqual(self.node.status()['manual_open']['status'], 'accepted')
        self.now.return_value = 107.51
        self.node.tick()
        self.assertEqual(self.node.status()['manual_open']['status'], 'error')
        self.assertIn('Move timed out', self.node.status()['manual_open']['error'])
        self.assertEqual(len(self.node.gripper_stop.requests), 1)
        self.assertEqual(self.node.switch.requests, [])

    def test_manual_open_failed_raw_result_stops_and_never_uses_contact_exception(self):
        self.node.request(self.manual_packet())
        handle = FakeHandle()
        self.node.move.futures[-1].complete(handle)
        handle.result.complete(types.SimpleNamespace(status=6,
            result=types.SimpleNamespace(success=False, error='libfranka gripper: Command failed!')))
        manual = self.node.status()['manual_open']
        self.assertEqual(manual['status'], 'error')
        self.assertEqual(manual['result_status'], 6)
        self.assertFalse(manual['success'])
        self.assertEqual(manual['raw_error'], 'libfranka gripper: Command failed!')
        self.assertEqual(len(self.node.gripper_stop.requests), 1)
        self.assertEqual(self.node.switch.requests, [])

    def test_manual_open_state_staleness_during_execution_stops_only_gripper(self):
        self.node.request(self.manual_packet())
        self.node.move.futures[-1].complete(FakeHandle())
        self.now.return_value = 100.3
        self.node.tick()
        self.assertEqual(self.node.status()['manual_open']['status'], 'error')
        self.assertIn('fresh stationary', self.node.status()['manual_open']['error'])
        self.assertEqual(len(self.node.gripper_stop.requests), 1)
        self.assertEqual(self.node.switch.requests, [])

    def test_manual_cancel_uuid_collection_is_bounded(self):
        for _ in range(160):
            self.node.revoke_manual(str(uuid.uuid4()))
        self.assertEqual(len(self.node.cancelled_manual_ids), 128)

    def test_activation_failure_stops_instead_of_submitting_actions(self):
        self.node.request(dict(type='arm', version=1, session_id=SESSION, model_id='model-sha'))
        self.node.switch.futures[-1].complete(types.SimpleNamespace(ok=False))
        self.assertFalse(self.node.gate.armed)
        self.assertIn('Activation failed', self.node.gate.fault)
        self.assertEqual(self.node.switch.requests[-1].deactivate_controllers, ['joint_impedance_controller'])
        self.assertEqual(len(self.node.move.requests), 0)

    def test_contact_keeps_raw_result_and_never_retries_same_close(self):
        self.arm()
        self.node.send_move(0.)
        goal = self.node.move.requests[-1]
        self.assertEqual(goal.width, 0.)
        self.assertEqual(goal.speed, .04)
        handle = FakeHandle()
        self.node.move.futures[-1].complete(handle)
        error = 'libfranka gripper: Command failed!'
        handle.result.complete(types.SimpleNamespace(status=6,
            result=types.SimpleNamespace(success=False, error=error)))
        self.assertEqual(self.node.gripper_result['status'], 'not_reached')
        self.assertEqual(self.node.gripper_result['error'], error)
        self.assertTrue(self.node.gate.armed)
        self.node.send_move(0.)
        self.assertEqual(len(self.node.move.requests), 1)
        self.assertEqual(len(self.node.gripper_stop.requests), 0)

    def test_opening_error_is_fatal_not_contact(self):
        self.arm()
        self.node.send_move(.07)
        handle = FakeHandle()
        self.node.move.futures[-1].complete(handle)
        handle.result.complete(types.SimpleNamespace(status=6,
            result=types.SimpleNamespace(success=False, error='libfranka gripper: Command failed!')))
        self.assertFalse(self.node.gate.armed)
        self.assertEqual(self.node.gripper_result['status'], 'error')
        self.assertEqual(len(self.node.gripper_stop.requests), 1)
        self.assertEqual(len(self.node.move.requests), 1)

    def accepted_closing_for_preemption(self):
        self.arm()
        self.node.send_move(.02)
        handle = FakeHandle()
        self.node.move.futures[-1].complete(handle)
        self.node.gate.capture_stamp = 99.95
        self.node.gate.execution_end = 100.35
        return handle

    def canceled_result(self, handle, status=5, error=''):
        handle.result.complete(types.SimpleNamespace(status=status,
            result=types.SimpleNamespace(success=False, error=error)))

    def test_open_preemption_stop_ack_first_waits_for_terminal_and_next_fresh_target(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.assertEqual(handle.cancelled, 0)
        self.assertEqual(len(self.node.gripper_stop.requests), 1)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.assertTrue(self.node.move_pending)
        self.assertEqual(len(self.node.move.requests), 1)
        self.canceled_result(handle)
        self.assertFalse(self.node.move_pending)
        self.assertTrue(self.node.gate.armed)
        self.assertEqual(len(self.node.move.requests), 1)  # Callback never replays .07.
        self.now.return_value = 100.01
        self.node.gate.update_width(.04, 100.001)
        self.node.send_move(.06)
        self.assertEqual(len(self.node.move.requests), 2)
        self.assertEqual(self.node.move.requests[-1].width, .06)
        self.assertEqual(self.node.status()['gripper_latest_requested_width_m'], .06)

    def test_open_preemption_terminal_first_waits_for_native_stop_confirmation(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.canceled_result(handle, status=6, error='libfranka gripper: Command failed!')
        self.assertTrue(self.node.move_pending)
        self.assertTrue(self.node.gate.armed)
        self.assertIsNone(self.node.gate.fault)
        self.node.send_move(.075)
        self.assertEqual(len(self.node.move.requests), 1)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.assertFalse(self.node.move_pending)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_native_stop_aborted_close_result_first_waits_for_stop_ack_and_fresh_open(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.canceled_result(handle, status=6, error='libfranka gripper: Command aborted!')
        self.assertTrue(self.node.move_preemption['terminal'])
        self.assertFalse(self.node.move_preemption['stop_done'])
        self.assertTrue(self.node.move_pending)
        self.assertTrue(self.node.gate.armed)
        self.assertIsNone(self.node.gate.fault)
        self.assertEqual(self.node.gripper_result['error'], 'libfranka gripper: Command aborted!')
        self.node.send_move(.075)
        self.assertEqual(len(self.node.move.requests), 1)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.assertFalse(self.node.move_pending)
        self.assertIsNotNone(self.node.post_preempt_open_required)
        self.node.send_move(.07)  # Recent measurement still predates native Stop.
        self.assertEqual(len(self.node.move.requests), 1)
        self.now.return_value = 100.01
        self.node.gate.update_width(.04, 100.001)
        self.node.send_move(.06)
        self.assertEqual(len(self.node.move.requests), 2)
        self.assertEqual(self.node.move.requests[-1].width, .06)
        self.assertEqual(self.node.move.requests[-1].speed, .04)

    def test_native_stop_ack_first_still_waits_for_aborted_close_result(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.assertTrue(self.node.move_preemption['stop_done'])
        self.assertFalse(self.node.move_preemption['terminal'])
        self.node.send_move(.075)
        self.assertEqual(len(self.node.move.requests), 1)
        self.canceled_result(handle, status=6, error='libfranka gripper: Command aborted!')
        self.assertTrue(self.node.gate.armed)
        self.assertIsNone(self.node.gate.fault)
        self.assertFalse(self.node.move_pending)
        self.assertEqual(len(self.node.move.requests), 1)
        self.now.return_value = 100.01
        self.node.gate.update_width(.056, 100.001)
        self.node.send_move(.049)  # A stale opening intent is now a closing target.
        self.assertEqual(len(self.node.move.requests), 1)
        self.node.send_move(.08)
        self.assertEqual(len(self.node.move.requests), 2)
        self.assertEqual(self.node.move.requests[-1].width, .08)

    def test_native_cancelled_aborted_close_result_first_still_waits_for_stop_ack(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.canceled_result(handle, status=5, error='libfranka gripper: Command aborted!')
        self.assertTrue(self.node.gate.armed)
        self.assertIsNone(self.node.gate.fault)
        self.assertTrue(self.node.move_preemption['terminal'])
        self.assertFalse(self.node.move_preemption['stop_done'])
        self.node.send_move(.075)
        self.assertEqual(len(self.node.move.requests), 1)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.assertFalse(self.node.move_pending)
        self.node.send_move(.075)
        self.assertEqual(len(self.node.move.requests), 1)  # Post-stop source sample still missing.
        self.now.return_value = 100.01
        self.node.gate.update_width(.04, 100.001)
        self.node.send_move(.06)
        self.assertEqual(len(self.node.move.requests), 2)
        self.assertEqual(self.node.move.requests[-1].width, .06)

    def test_native_stop_ack_first_accepts_cancelled_aborted_close_terminal_only(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.assertTrue(self.node.move_pending)
        self.node.send_move(.075)
        self.assertEqual(len(self.node.move.requests), 1)
        self.canceled_result(handle, status=5, error='libfranka gripper: Command aborted!')
        self.assertTrue(self.node.gate.armed)
        self.assertIsNone(self.node.gate.fault)
        self.assertFalse(self.node.move_pending)
        self.assertIsNotNone(self.node.post_preempt_open_required)
        self.assertEqual(len(self.node.move.requests), 1)
        self.now.return_value = 100.01
        self.node.gate.update_width(.04, 100.001)
        self.node.send_move(.08)
        self.assertEqual(len(self.node.move.requests), 2)
        self.assertEqual(self.node.move.requests[-1].width, .08)

    def test_native_stop_aborted_candidate_does_not_override_failed_stop_ack(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.canceled_result(handle, status=6, error='libfranka gripper: Command aborted!')
        self.node.gripper_stop.futures[-1].complete(
            types.SimpleNamespace(success=False, message='stop failure'))
        self.assertFalse(self.node.gate.armed)
        self.assertIn('stop failure', self.node.gate.fault)
        self.assertIsNone(self.node.move_preemption)
        self.assertIsNone(self.node.post_preempt_open_required)
        self.node.send_move(.08)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_failed_preemption_stop_ack_fences_later_aborted_goal_result(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.node.gripper_stop.futures[-1].complete(
            types.SimpleNamespace(success=False, message='stop refused'))
        first_fault = self.node.gate.fault
        self.canceled_result(handle, status=6, error='libfranka gripper: Command aborted!')
        self.assertFalse(self.node.gate.armed)
        self.assertEqual(self.node.gate.fault, first_fault)
        self.assertIn('stop refused', first_fault)
        self.assertIsNone(self.node.move_preemption)
        self.assertIsNone(self.node.post_preempt_open_required)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_native_aborted_close_without_open_preemption_remains_fatal(self):
        self.arm()
        self.node.send_move(.02)
        handle = FakeHandle()
        self.node.move.futures[-1].complete(handle)
        self.canceled_result(handle, status=6, error='libfranka gripper: Command aborted!')
        self.assertFalse(self.node.gate.armed)
        self.assertIn('Command aborted!', self.node.gate.fault)
        self.assertEqual(self.node.gripper_result['status'], 'error')
        self.assertEqual(len(self.node.gripper_stop.requests), 1)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_manual_open_native_aborted_result_remains_fatal(self):
        self.node.request(self.manual_packet())
        handle = FakeHandle()
        self.node.move.futures[-1].complete(handle)
        self.canceled_result(handle, status=6, error='libfranka gripper: Command aborted!')
        manual = self.node.status()['manual_open']
        self.assertEqual(manual['status'], 'error')
        self.assertEqual(manual['raw_error'], 'libfranka gripper: Command aborted!')
        self.assertEqual(len(self.node.gripper_stop.requests), 1)
        self.assertEqual(self.node.switch.requests, [])

    def test_native_aborted_open_preemption_requires_exact_aborted_status(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.canceled_result(handle, status=4, error='libfranka gripper: Command aborted!')
        self.assertFalse(self.node.gate.armed)
        self.assertIn('Unexpected Move result', self.node.gate.fault)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_native_aborted_open_preemption_requires_matching_session(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.node.move_preemption['session_id'] = 'different-session'
        self.canceled_result(handle, status=6, error='libfranka gripper: Command aborted!')
        self.assertFalse(self.node.gate.armed)
        self.assertIn('Unexpected Move result', self.node.gate.fault)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_global_stop_and_rearm_fence_late_native_aborted_preempt_result(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        preemption_stop = self.node.gripper_stop.futures[-1]
        self.node.request(dict(type='stop', version=1, reason='operator stop'))
        self.assertIsNone(self.node.move_preemption)
        preemption_stop.complete(types.SimpleNamespace(success=True))
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.node.switch.futures[-1].complete(types.SimpleNamespace(ok=True))
        self.assertEqual(self.node.gate.fault, 'operator stop')
        new_session = '0a037346-abb5-44c9-9677-671d2e727334'
        self.node.request(dict(type='arm', version=1, session_id=new_session, model_id='model-sha'))
        self.node.switch.futures[-1].complete(types.SimpleNamespace(ok=True))
        self.node.send_move(.07)
        generation = self.node.move_generation
        self.canceled_result(handle, status=6, error='libfranka gripper: Command aborted!')
        self.assertTrue(self.node.gate.armed)
        self.assertEqual(self.node.gate.session_id, new_session)
        self.assertIsNone(self.node.gate.fault)
        self.assertTrue(self.node.move_pending)
        self.assertTrue(self.node.move_ack_pending)
        self.assertEqual(self.node.move_generation, generation)
        self.assertEqual(self.node.gripper_result['status'], 'pending')
        self.assertEqual(len(self.node.move.requests), 2)

    def test_after_open_preemption_elastic_reopening_cannot_trigger_a_new_close(self):
        handle = self.accepted_closing_for_preemption()
        self.node.gate.update_width(.045, 100.)
        self.node.send_move(.049)
        self.canceled_result(handle)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.assertFalse(self.node.move_pending)
        self.assertIsNotNone(self.node.post_preempt_open_required)
        self.now.return_value = 100.01
        self.node.gate.update_width(.056, 100.001)
        self.node.send_move(.049)  # Former opening target is now below measured width.
        self.assertEqual(len(self.node.move.requests), 1)
        self.node.send_move(.08)
        self.assertEqual(len(self.node.move.requests), 2)
        self.assertEqual(self.node.move.requests[-1].width, .08)
        self.assertIsNone(self.node.post_preempt_open_required)

    def test_post_preemption_waits_for_measurement_captured_after_native_stop(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.canceled_result(handle)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.node.send_move(.07)  # Old .04 measurement is recent, but predates stop.
        self.assertEqual(len(self.node.move.requests), 1)
        self.assertIsNotNone(self.node.post_preempt_open_required)
        self.now.return_value = 100.01
        self.node.gate.update_width(.04, 100.001)
        self.node.send_move(.07)
        self.assertEqual(len(self.node.move.requests), 2)

    def test_global_stop_during_open_preemption_fences_both_late_callbacks(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        preemption_stop = self.node.gripper_stop.futures[-1]
        self.node.request(dict(type='stop', version=1, reason='operator stop'))
        self.assertIsNone(self.node.move_preemption)
        self.canceled_result(handle)
        preemption_stop.complete(types.SimpleNamespace(success=True))
        self.assertEqual(len(self.node.gripper_stop.requests), 2)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.node.send_move(.07)
        self.assertFalse(self.node.gate.armed)
        self.assertEqual(len(self.node.move.requests), 1)
        self.assertEqual(self.node.gate.fault, 'operator stop')

    def test_open_intent_during_unknown_acceptance_is_recorded_without_sending(self):
        self.arm()
        self.node.send_move(.02)
        acceptance = self.node.move.futures[-1]
        self.node.gate.capture_stamp, self.node.gate.execution_end = 99.95, 100.35
        self.node.send_move(.07)
        self.assertEqual(len(self.node.move.requests), 1)
        self.assertEqual(len(self.node.gripper_stop.requests), 0)
        self.assertEqual(self.node.latest_requested_width, .07)
        handle = FakeHandle()
        acceptance.complete(handle)
        self.assertEqual(len(self.node.gripper_stop.requests), 0)  # No cached intent acted on.
        self.node.send_move(.075)
        self.assertEqual(handle.cancelled, 0)
        self.assertEqual(len(self.node.gripper_stop.requests), 1)

    def test_open_preemption_requires_fresh_measured_width_and_target(self):
        handle = self.accepted_closing_for_preemption()
        self.node.gate.width_stamp = 99.0
        self.node.send_move(.07)
        self.assertEqual(handle.cancelled, 0)
        self.assertEqual(len(self.node.gripper_stop.requests), 0)
        self.node.gate.width_stamp = 99.999
        self.node.gate.execution_end = 99.99
        self.node.send_move(.07)
        self.assertEqual(handle.cancelled, 0)
        self.assertEqual(len(self.node.gripper_stop.requests), 0)

    def test_open_preemption_does_not_swallow_unexpected_hardware_result(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.canceled_result(handle, status=6, error='network/hardware failure')
        self.assertFalse(self.node.gate.armed)
        self.assertIn('Unexpected Move result', self.node.gate.fault)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_preemption_near_move_deadline_uses_bounded_stop_handshake(self):
        self.accepted_closing_for_preemption()
        self.node.move_started = 100. - self.node.config['execution']['gripper_action_timeout_s'] + .05
        self.node.send_move(.07)
        self.now.return_value = 100.1
        with patch.object(self.node.gate, 'tick', return_value=None):
            self.node.tick()
        self.assertTrue(self.node.gate.armed)
        self.assertIsNotNone(self.node.move_preemption)
        self.assertIsNone(self.node.gate.fault)

    def test_preemption_stop_ack_without_goal_terminal_has_a_short_deadline(self):
        self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.now.return_value = 103.01
        self.node.tick()
        self.assertFalse(self.node.gate.armed)
        self.assertIn('within 3.0s', self.node.gate.fault)
        self.assertEqual(len(self.node.move.requests), 1)

    def refresh_hold_inputs(self, now, sequence=None):
        self.now.return_value = now
        gate = self.node.gate
        gate.update_arm(list(gate.q), list(gate.velocity), now - .0001)
        gate.update_width(.04, now - .0001)
        gate.heartbeat(SESSION, True, now - .0001, now)
        if sequence is None:
            return None
        target = list(gate.measurement)
        target[0] += .1
        packet = dict(type='submit', version=1, session_id=SESSION, model_id='model-sha', fps=15,
                      request_id=sequence, capture_seq=sequence, capture_monotonic=now,
                      actions=[list(target) + [.07] for _ in range(50)])
        self.node.request(packet)
        return packet

    def test_native_stop_has_own_three_second_deadline_without_changing_controller_timeout(self):
        self.assertEqual(GRIPPER_STOP_TIMEOUT_S, 3.)
        self.assertEqual(self.node.config['execution']['gripper_action_timeout_s'], 5.)
        self.node.stop_gripper()
        self.now.return_value = 102.9
        self.node.tick()
        self.assertIsNone(self.node.exit_error)
        self.assertTrue(self.node.gripper_stop_pending)
        self.now.return_value = 103.01
        self.node.tick()
        self.assertIn('Native gripper stop timed out after 3.0s', self.node.exit_error)

    def test_open_preemption_waits_up_to_three_seconds_for_old_terminal_after_stop_ack(self):
        self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.now.return_value = 102.9
        with patch.object(self.node.gate, 'tick', return_value=None):
            self.node.tick()
        self.assertTrue(self.node.gate.armed)
        self.assertIsNone(self.node.gate.fault)
        self.now.return_value = 103.01
        self.node.tick()
        self.assertFalse(self.node.gate.armed)
        self.assertIn('within 3.0s', self.node.gate.fault)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_2390ms_stop_holds_arm_discards_joint_waypoints_and_resumes_only_new_submit(self):
        handle = self.accepted_closing_for_preemption()
        self.node.move_started = 97.2  # A closing Move can already be old when opening supersedes it.
        self.node.send_move(.07)
        sequence = 0
        latest_packet = None
        for i in range(1, 240):
            now = 100. + i * .01
            if i % 10 == 1:
                sequence += 1
                latest_packet = self.refresh_hold_inputs(now, sequence)
            else:
                self.refresh_hold_inputs(now)
            capture_before = self.node.gate.capture_stamp
            submit_before = self.node.gate.last_submit
            self.node.tick()
            self.assertTrue(self.node.gate.armed)
            self.assertIsNone(self.node.exit_error)
            self.assertEqual(self.node.gate.q, POSE)
            self.assertEqual(self.node.gate.velocity, [0.] * 7)
            self.assertEqual(self.node.gate.capture_stamp, capture_before)
            self.assertEqual(self.node.gate.last_submit, submit_before)
            self.assertEqual(len(self.node.move.requests), 1)
        self.assertEqual(latest_packet['actions'][0][0], .1)  # Raw submitted predictions remain intact.
        self.assertEqual(self.node.gate.actions[0][0], 0.)
        self.assertTrue(self.node.status()['arm_holding_for_gripper_stop'])
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.canceled_result(handle, status=6, error='libfranka gripper: Command aborted!')
        self.assertIsNotNone(self.node.post_preempt_open_required)
        self.node.send_move(.07)  # No post-stop sample yet; no cached opening is sent.
        self.assertEqual(len(self.node.move.requests), 1)
        self.refresh_hold_inputs(102.4)
        self.node.tick()
        self.assertEqual(len(self.node.move.requests), 2)
        self.assertEqual(self.node.move.requests[-1].width, .07)
        self.assertFalse(self.node.status()['arm_holding_for_gripper_stop'])
        self.assertEqual(self.node.gate.q, POSE)
        self.refresh_hold_inputs(102.41)
        self.node.tick()
        self.assertEqual(self.node.gate.q, POSE)  # Opening completion does not replay hold-era q targets.
        sequence += 1
        self.refresh_hold_inputs(102.42, sequence)
        self.node.tick()
        self.assertGreater(self.node.gate.q[0], POSE[0])
        self.assertLessEqual(self.node.gate.velocity[0], .008 + 1e-10)

    def test_arm_hold_uses_existing_acceleration_limiter_to_decelerate_without_position_jump(self):
        self.accepted_closing_for_preemption()
        self.node.gate.velocity[0] = .2
        self.node.send_move(.07)
        previous_q, previous_v = list(self.node.gate.q), list(self.node.gate.velocity)
        stopped_q = None
        sequence = 0
        for i in range(1, 41):
            now = 100. + i * .01
            if i % 10 == 1:
                sequence += 1
                self.refresh_hold_inputs(now, sequence)
            else:
                self.refresh_hold_inputs(now)
            self.node.tick()
            q, velocity = self.node.gate.q, self.node.gate.velocity
            self.assertLessEqual(abs(velocity[0] - previous_v[0]), .8 * .01 + 1e-10)
            self.assertLessEqual(abs(q[0] - previous_q[0]), .35 * .01 + 1e-10)
            self.assertGreaterEqual(velocity[0], -1e-10)
            if i == 26:
                stopped_q = q[0]
            if i >= 26:
                self.assertAlmostEqual(velocity[0], 0.)
                self.assertAlmostEqual(q[0], stopped_q)
            previous_q, previous_v = list(q), list(velocity)
        self.assertAlmostEqual(self.node.gate.q[0] - POSE[0], .2 ** 2 / (2 * .8))
        self.assertTrue(self.node.gate.armed)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_fast_stop_with_nonzero_velocity_discards_last_hold_target_and_never_reverses(self):
        handle = self.accepted_closing_for_preemption()
        self.node.gate.velocity[0] = .2
        self.node.send_move(.07)
        self.refresh_hold_inputs(100.01, 1)
        self.node.tick()
        self.assertGreater(self.node.gate.velocity[0], 0.)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.canceled_result(handle, status=6, error='libfranka gripper: Command aborted!')
        self.refresh_hold_inputs(100.02)
        self.node.tick()
        self.assertEqual(len(self.node.move.requests), 2)
        self.assertGreater(self.node.gate.velocity[0], 0.)
        self.assertIsNone(self.node.gate.actions)
        capture, last_submit = self.node.gate.capture_stamp, self.node.gate.last_submit
        previous_q = self.node.gate.q[0]
        stopped_q = None
        for i in range(3, 43):
            self.refresh_hold_inputs(100. + i * .01)
            self.node.tick()
            self.assertIsNone(self.node.gate.actions)
            self.assertGreaterEqual(self.node.gate.velocity[0], -1e-10)
            self.assertGreaterEqual(self.node.gate.q[0], previous_q - 1e-10)
            if i == 26:
                stopped_q = self.node.gate.q[0]
            if i >= 26:
                self.assertAlmostEqual(self.node.gate.velocity[0], 0.)
                self.assertAlmostEqual(self.node.gate.q[0], stopped_q)
            self.assertEqual(self.node.gate.capture_stamp, capture)
            self.assertEqual(self.node.gate.last_submit, last_submit)
            previous_q = self.node.gate.q[0]
        self.assertAlmostEqual(stopped_q, .2 ** 2 / (2 * .8))
        self.refresh_hold_inputs(100.43, 2)
        self.node.tick()
        self.assertGreater(self.node.gate.q[0], stopped_q)

    def test_gripper_freshness_still_faults_during_stop_handshake(self):
        self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.now.return_value = 100.51
        self.node.gate.update_arm(POSE, [0.] * 7, 100.5099)
        self.node.gate.heartbeat(SESSION, True, 100.5099, 100.51)
        self.node.tick()
        self.assertFalse(self.node.gate.armed)
        self.assertIn('Fresh arm AND measured gripper state', self.node.gate.fault)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_heartbeat_timeout_still_faults_during_stop_handshake(self):
        self.accepted_closing_for_preemption()
        self.node.gate.heartbeat(SESSION, True, 100., 100.)
        self.node.send_move(.07)
        self.now.return_value = 100.51
        self.node.gate.update_arm(POSE, [0.] * 7, 100.5099)
        self.node.gate.update_width(.04, 100.5099)
        self.node.tick()
        self.assertFalse(self.node.gate.armed)
        self.assertIn('heartbeat timed out', self.node.gate.fault)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_first_fault_survives_subsequent_stop_request(self):
        self.arm()
        self.node.fail('Native Move timed out')
        self.node.request(dict(type='stop', version=1, reason='Session is not armed'))
        self.assertEqual(self.node.status()['fault'], 'Native Move timed out')

    def test_late_same_session_heartbeat_reports_root_fault_without_exposing_to_wrong_session(self):
        self.arm()
        self.node.fail('Native Move timed out')
        with self.assertRaisesRegex(SafetyError, 'Native Move timed out'):
            self.node.request(dict(type='heartbeat', version=1, session_id=SESSION,
                                   camera_ok=True, camera_capture_monotonic=100.))
        with self.assertRaisesRegex(SafetyError, 'Wrong or missing'):
            self.node.request(dict(type='heartbeat', version=1, session_id='different-session',
                                   camera_ok=True, camera_capture_monotonic=100.))

    def test_preemption_stop_submission_exception_explicitly_shuts_down(self):
        self.accepted_closing_for_preemption()
        with patch.object(self.node.gripper_stop, 'call_async', side_effect=RuntimeError('stop RPC unavailable')):
            self.node.send_move(.07)
        self.assertFalse(self.node.gate.armed)
        self.assertIn('stop request failed', self.node.exit_error)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_deactivation_submission_exception_explicitly_shuts_down(self):
        self.arm()
        with patch.object(self.node.switch, 'call_async', side_effect=RuntimeError('switch RPC unavailable')):
            self.node.fail('test fault')
        self.assertFalse(self.node.gate.armed)
        self.assertIn('deactivation request failed', self.node.exit_error)

    def test_gripper_json_lifecycle_records_release_intent_and_raw_cancel_result(self):
        handle = self.accepted_closing_for_preemption()
        self.node.send_move(.07)
        self.node.send_move(.0701)  # Below deadband: no repeated pending-target log.
        self.canceled_result(handle, status=6, error='libfranka gripper: Command failed!')
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        events = [json.loads(message) for message in self.node.messages if message.startswith('{')]
        stages = [event['stage'] for event in events]
        self.assertEqual(stages.count('target_while_pending'), 1)
        for stage in ('requested', 'accepted', 'preempt', 'result', 'preempt_complete'):
            self.assertIn(stage, stages)
        result = next(event for event in events if event['stage'] == 'result')
        self.assertEqual(result['result_status'], 6)
        self.assertEqual(result['error'], 'libfranka gripper: Command failed!')
        target = next(event for event in events if event['stage'] == 'target_while_pending')
        self.assertEqual(target['width_m'], .07)

    def test_stop_during_pending_acceptance_stops_late_goal_before_rearm(self):
        self.arm()
        self.node.send_move(0.)
        pending = self.node.move.futures[-1]
        self.node.request(dict(type='stop', version=1))
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.node.switch.futures[-1].complete(types.SimpleNamespace(ok=True))
        with self.assertRaises(SafetyError):
            self.node.request(dict(type='arm', version=1,
                session_id='0a037346-abb5-44c9-9677-671d2e727334', model_id='model-sha'))
        handle = FakeHandle()
        pending.complete(handle)
        self.assertEqual(handle.cancelled, 0)
        self.assertEqual(len(self.node.gripper_stop.requests), 2)
        self.assertTrue(self.node.gripper_stop_pending)
        self.node.gripper_stop.futures[-1].complete(types.SimpleNamespace(success=True))
        self.assertFalse(self.node.move_pending)
        self.assertFalse(self.node.gate.armed)
        self.assertEqual(len(self.node.move.requests), 1)

    def test_late_goal_while_stop_response_pending_requires_second_stop(self):
        self.arm()
        self.node.send_move(0.)
        pending = self.node.move.futures[-1]
        self.node.request(dict(type='stop', version=1))
        first_stop = self.node.gripper_stop.futures[-1]
        pending.complete(FakeHandle())
        self.assertTrue(self.node.gripper_stop_again)
        first_stop.complete(types.SimpleNamespace(success=True))
        self.assertEqual(len(self.node.gripper_stop.requests), 2)

    def test_controller_stop_timeout_shuts_down_launch_owner(self):
        self.arm()
        self.node.request(dict(type='stop', version=1))
        self.now.return_value = 101.3
        self.node.tick()
        self.assertIn('deactivation timed out', self.node.exit_error)

    def test_stopping_accepted_move_uses_only_native_stop_not_action_cancel(self):
        handle = self.accepted_closing_for_preemption()
        with patch.object(handle, 'cancel_goal_async', side_effect=AssertionError('must not double-stop')) as cancel:
            self.node.request(dict(type='stop', version=1, reason='operator stop'))
        cancel.assert_not_called()
        self.assertEqual(len(self.node.gripper_stop.requests), 1)
        self.assertFalse(self.node.gate.armed)
        self.assertEqual(self.node.gate.fault, 'operator stop')

    def test_open_preemption_uses_only_native_stop_while_waiting_for_old_terminal(self):
        handle = self.accepted_closing_for_preemption()
        with patch.object(handle, 'cancel_goal_async', side_effect=AssertionError('must not double-stop')) as cancel:
            self.node.send_move(.07)
        cancel.assert_not_called()
        self.assertEqual(len(self.node.gripper_stop.requests), 1)
        self.assertTrue(self.node.move_preemption is not None)
        self.assertFalse(self.node.move_preemption['stop_done'])
        self.assertFalse(self.node.move_preemption['terminal'])
        self.assertEqual(len(self.node.move.requests), 1)

    def test_fatal_in_locked_callback_returns_without_context_shutdown_and_releases_lock(self):
        self.arm()
        finished, acquired = threading.Event(), threading.Event()
        errors = []
        def callback():
            try:
                with self.node.lock:
                    self.node.emergency_exit('fatal in callback')
                finished.set()
            except Exception as exc:
                errors.append(exc)
        with patch.object(sys.modules['rclpy'], 'try_shutdown',
                          side_effect=AssertionError('context shutdown would wait for this callback')) as shutdown:
            worker = threading.Thread(target=callback, daemon=True)
            worker.start()
            self.assertTrue(finished.wait(.5))
            worker.join(.5)
            shutdown.assert_not_called()
        self.assertEqual(errors, [])
        def other_thread():
            with self.node.lock:
                acquired.set()
        check = threading.Thread(target=other_thread, daemon=True)
        check.start()
        self.assertTrue(acquired.wait(.5))
        check.join(.5)
        self.assertFalse(self.node.gate.armed)
        self.assertEqual(self.node.exit_error, 'fatal in callback')

    def test_fatal_latches_first_error_and_blocks_new_motion_rpc_and_direct_commands(self):
        self.arm()
        self.node.fail('first state fault')
        self.node.emergency_exit('native stop timed out')
        self.node.emergency_exit('later deactivation fault')
        self.assertEqual(self.node.exit_error, 'native stop timed out')
        self.assertEqual(self.node.gate.fault, 'first state fault')
        self.assertFalse(self.node.status()['ready'])
        self.assertFalse(self.node.status()['manual_open_ready'])
        for packet in (
            dict(type='arm', version=1, session_id=str(uuid.uuid4()), model_id='model-sha'),
            self.manual_packet(), dict(type='submit', version=1, session_id=SESSION),
            dict(type='heartbeat', version=1, session_id=SESSION),
        ):
            with self.subTest(kind=packet['type']), self.assertRaisesRegex(SafetyError, 'shutting down'):
                self.node.request(packet)
        with self.assertRaisesRegex(SafetyError, 'shutting down'):
            self.node.start_manual_open(self.manual_packet())
        previous_publish, previous_move = len(self.node.published), len(self.node.move.requests)
        self.node.publish(POSE)
        self.node.send_move(.08)
        self.node.tick()
        self.assertEqual(len(self.node.published), previous_publish)
        self.assertEqual(len(self.node.move.requests), previous_move)
        self.assertFalse(self.node.request(dict(type='state', version=1))['armed'])

    def test_main_exits_fatal_spin_once_then_shuts_down_context_outside_callback(self):
        self.arm()
        events = []
        callback_active = [False]
        context_alive = [True]
        server_stopped = threading.Event()
        class FakeServer:
            def __init__(self, address, handler):
                events.append('server_started')

            def serve_forever(self):
                server_stopped.wait(1)

            def shutdown(self):
                events.append('server_shutdown')
                server_stopped.set()

            def server_close(self):
                events.append('server_closed')
        def spin_once(node, timeout_sec):
            self.assertEqual(timeout_sec, .1)
            callback_active[0] = True
            try:
                with node.lock:
                    node.emergency_exit('native stop deadline')
            finally:
                callback_active[0] = False
            events.append('callback_returned')
        def shutdown():
            self.assertFalse(callback_active[0])
            events.append('context_shutdown')
            context_alive[0] = False
        self.node.destroy_node = lambda: events.append('node_destroyed')
        fake_rclpy = types.SimpleNamespace(init=lambda **kwargs: events.append('context_init'),
            ok=lambda: context_alive[0], spin_once=spin_once, try_shutdown=shutdown)
        modules = {'rclpy': fake_rclpy,
                   'rclpy.executors': types.SimpleNamespace(ExternalShutdownException=type('ExternalShutdownException', (Exception,), {}))}
        with patch.dict(sys.modules, modules), patch('fr3_policy_bridge.create_node', return_value=self.node), \
                patch('fr3_policy_bridge.BridgeServer', FakeServer):
            result = main([])
        self.assertEqual(result, 1)
        self.assertEqual(self.node.exit_error, 'native stop deadline')
        self.assertFalse(self.node.gate.armed)
        self.assertLess(events.index('callback_returned'), events.index('context_shutdown'))
        self.assertLess(events.index('server_closed'), events.index('context_shutdown'))
        self.assertIn('node_destroyed', events)

    def test_stop_during_activation_when_activation_reply_arrives_first(self):
        self.node.request(dict(type='arm', version=1, session_id=SESSION, model_id='model-sha'))
        activation = self.node.switch.futures[-1]
        self.node.request(dict(type='stop', version=1))
        deactivation = self.node.switch.futures[-1]
        activation.complete(types.SimpleNamespace(ok=True))
        self.assertFalse(self.node.gate.armed)
        self.assertTrue(self.node.stop_pending)
        deactivation.complete(types.SimpleNamespace(ok=True))
        self.assertFalse(self.node.active)
        self.assertFalse(self.node.gate.armed)
        self.assertEqual(len(self.node.move.requests), 0)

    def test_stop_during_activation_when_deactivation_reply_arrives_first(self):
        self.node.request(dict(type='arm', version=1, session_id=SESSION, model_id='model-sha'))
        activation = self.node.switch.futures[-1]
        self.node.request(dict(type='stop', version=1))
        self.node.switch.futures[-1].complete(types.SimpleNamespace(ok=True))
        activation.complete(types.SimpleNamespace(ok=True))
        self.assertFalse(self.node.gate.armed)
        self.assertEqual(len(self.node.switch.requests), 3)
        self.assertEqual(self.node.switch.requests[-1].deactivate_controllers, ['joint_impedance_controller'])
        self.node.switch.futures[-1].complete(types.SimpleNamespace(ok=True))
        self.assertFalse(self.node.active)
        self.assertEqual(len(self.node.move.requests), 0)


if __name__ == '__main__':
    unittest.main()
