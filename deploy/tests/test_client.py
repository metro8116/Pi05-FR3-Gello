"""Pure CPU mocks: these tests never connect to ROS, AutoDL, cameras, or FR3."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config
from client_runtime import CameraTimestampMapper, ClientError, DeploymentRuntime, check_cameras, check_state
from protocol import ACTION_SEMANTICS, CAMERA_KEYS


def metadata(**updates):
    return dict({'version': 1, 'type': 'metadata', 'state_dim': 8, 'action_dim': 8,
                 'camera_keys': list(CAMERA_KEYS), 'fps': 15, 'chunk_size': 50,
                 'model_id': 'test-sha256', 'checkpoint': '/not-a-real-checkpoint',
                 'mock': False, 'action_semantics': ACTION_SEMANTICS,
                 'task': 'Put the avocado in the basket.'}, **updates)


def state(**updates):
    return dict({'version': 1, 'type': 'state', 'state': [0.0] * 8,
                 'state_age_s': 0.0, 'gripper_age_s': 0.0,
                 'ready': True, 'armed': False, 'fault': None, 'session_id': None}, **updates)


class Cameras:
    def __init__(self):
        self.started = self.closed = False
        self.listeners = []

    def start(self):
        self.started = True

    def snapshot(self):
        now = time.monotonic()
        return {key: {'rgb': 'test-image', 'monotonic': now, 'number': 1} for key in CAMERA_KEYS}, {}

    def close(self):
        self.closed = True

    def add_frame_listener(self, listener):
        self.listeners.append(listener)

    def remove_frame_listener(self, listener):
        if listener in self.listeners:
            self.listeners.remove(listener)


class Recording:
    def __init__(self, *args, **kwargs):
        self.options = kwargs
        self.done = threading.Event()
        self.status, self.reason = 'starting', None
        self.session = self.frames = None

    def start(self, session, frames):
        self.session, self.frames = session, frames
        self.status = 'recording'

    def submit(self, key, frame):
        pass

    def stop(self, reason):
        if self.reason is None:
            self.reason = reason
        self.status = 'saved'
        self.done.set()

    def snapshot(self):
        return dict(status=self.status, directory='mock-recordings/' + str(self.session),
                    error=None, frames={}, dropped={})

    def wait(self, timeout=None):
        return self.done.wait(timeout)


class Bridge:
    def __init__(self):
        self.token = None
        self.calls = []
        self.submissions = []
        self.stopped = threading.Event()

    def request(self, packet):
        self.calls.append(copy.deepcopy(packet))
        kind = packet['type']
        if kind == 'state':
            return state(armed=self.token is not None, session_id=self.token)
        if kind == 'arm':
            self.token = packet['session_id']
            return {'version': 1, 'type': 'armed', 'armed': True, 'session_id': self.token}
        if kind == 'stop':
            if packet['session_id'] == self.token:
                self.token = None
            self.stopped.set()
            return {'version': 1, 'type': 'stopped'}
        if kind == 'heartbeat':
            return {'version': 1, 'type': 'heartbeat'}
        if kind == 'submit':
            if packet['session_id'] != self.token:
                return {'version': 1, 'type': 'accepted', 'accepted': False, 'fault': 'wrong session'}
            self.submissions.append(copy.deepcopy(packet))
            return {'version': 1, 'type': 'accepted', 'accepted': True}
        raise AssertionError(kind)


class Model:
    def __init__(self, **meta_updates):
        self.meta = metadata(**meta_updates)
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls = []
        self.cancelled = False
        self.reply_override = {}

    def request(self, packet):
        self.calls.append(copy.deepcopy(packet))
        if packet['type'] == 'hello':
            return self.meta
        self.entered.set()
        if not self.release.wait(2):
            raise RuntimeError('test inference deliberately blocked')
        return dict({'version': 1, 'type': 'actions',
                     'session_id': packet['session_id'], 'request_id': packet['request_id'],
                     'capture_seq': packet['capture_seq'], 'capture_monotonic': packet['capture_monotonic'],
                     'model_id': self.meta['model_id'], 'fps': 15,
                     'actions': [[0.0] * 8 for _ in range(50)], 'inference_s': 0.001}, **self.reply_override)

    def cancel(self):
        # Do not release the response: tests exercise replies arriving after stop.
        self.cancelled = True


class ManualOpenBridge(Bridge):
    """Manual Move adapter only; no model session is created by opening."""
    def __init__(self, status='completed', width=.08, gripper_age_s=0.):
        super().__init__()
        self.manual_status, self.width, self.gripper_age_s = status, width, gripper_age_s
        self.manual = None
        self.revoked = set()

    def request(self, packet):
        kind = packet['type']
        if kind == 'manual_open':
            self.calls.append(copy.deepcopy(packet))
            operation_id = packet['operation_id']
            if operation_id in self.revoked:
                raise ClientError('manual operation was revoked')
            self.manual = dict(operation_id=operation_id, status=self.manual_status,
                               target_width_m=.08, speed_m_s=.04, error='')
            return dict(version=1, type='manual_open_started', operation_id=operation_id)
        if kind == 'state':
            self.calls.append(copy.deepcopy(packet))
            return state(state=[0.] * 7 + [self.width], gripper_age_s=self.gripper_age_s,
                         manual_open=copy.deepcopy(self.manual))
        if kind == 'stop':
            operation_id = packet.get('manual_operation_id')
            if operation_id is not None:
                self.revoked.add(operation_id)
                if self.manual and self.manual['operation_id'] == operation_id:
                    self.manual['status'] = 'cancelled'
            return super().request(packet)
        return super().request(packet)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.runtime = None
        self.recordings = []
        self.encoder = patch('client_runtime.encode_image', return_value={})
        self.encoder.start()

    def tearDown(self):
        if self.runtime:
            self.runtime.close()
            self.model.release.set()
            for worker in self.runtime._workers:
                worker.join(timeout=1)
        self.encoder.stop()

    def make_runtime(self, model=None, bridge=None, config=None, recorder_factory=None):
        self.model = model or Model()
        self.bridge = bridge or Bridge()
        def factory(*args, **kwargs):
            recording = (recorder_factory or Recording)(*args, **kwargs)
            self.recordings.append(recording)
            return recording
        self.runtime = DeploymentRuntime(config or load_config(), cameras=Cameras(),
                                         model=self.model, bridge=self.bridge, recorder_factory=factory)
        return self.runtime

    def wait_until(self, test, timeout=1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if test():
                return
            time.sleep(0.005)
        self.fail('expected event did not occur')

    def test_connect_is_preview_only(self):
        runtime = self.make_runtime()
        runtime.connect()
        time.sleep(0.12)
        self.assertTrue(runtime.snapshot()['connected'])
        self.assertFalse(runtime.snapshot()['armed'])
        self.assertEqual([p['type'] for p in self.model.calls], ['hello'])
        self.assertNotIn('arm', [p['type'] for p in self.bridge.calls])
        self.assertEqual(self.recordings, [])

    def test_video_ready_before_arm_and_stop_detaches_capture_without_waiting(self):
        runtime = self.make_runtime()
        runtime.connect()
        original = self.bridge.request
        def request(packet):
            if packet['type'] == 'arm':
                self.assertEqual(self.recordings[-1].status, 'recording')
                self.assertEqual(set(self.recordings[-1].frames), set(CAMERA_KEYS))
            return original(packet)
        self.bridge.request = request
        runtime.arm()
        self.assertEqual(len(runtime.cameras.listeners), 1)
        started = time.monotonic()
        runtime.stop('用户停止录像测试')
        self.assertLess(time.monotonic() - started, .1)
        self.assertTrue(self.bridge.stopped.wait(1))
        self.assertEqual(runtime.cameras.listeners, [])
        self.assertEqual(self.recordings[-1].reason, '用户停止录像测试')
        self.assertTrue(runtime.wait_recording(0))

    def test_video_start_failure_never_sends_arm(self):
        class BrokenRecording(Recording):
            def start(self, session, frames):
                raise RuntimeError('video disk is read only')
        runtime = self.make_runtime(recorder_factory=BrokenRecording)
        runtime.connect()
        with self.assertRaisesRegex(RuntimeError, 'video disk is read only'):
            runtime.arm()
        self.assertNotIn('arm', [packet['type'] for packet in self.bridge.calls])
        self.assertFalse(runtime.snapshot()['armed'])
        self.assertIsNone(runtime.snapshot()['session_id'])
        self.assertEqual(runtime.cameras.listeners, [])
        self.assertIn('video disk is read only', self.recordings[-1].reason)

    def test_video_writer_error_stops_model_but_old_recording_error_cannot_stop_new_session(self):
        runtime = self.make_runtime()
        runtime.connect()
        runtime.arm()
        old = self.recordings[-1]
        old.options['on_error']('disk full')
        self.assertFalse(runtime.snapshot()['armed'])
        self.assertTrue(self.bridge.stopped.wait(1))
        self.assertIn('disk full', old.reason)
        runtime.arm()
        new_token = runtime.snapshot()['session_id']
        self.assertNotEqual(self.recordings[-1].session, old.session)
        old.options['on_error']('late old writer failure')
        self.assertTrue(runtime.snapshot()['armed'])
        self.assertEqual(runtime.snapshot()['session_id'], new_token)

    def test_camera_protection_and_fault_window_close_both_finalize_recording(self):
        runtime = self.make_runtime()
        runtime.connect()
        runtime.arm()
        runtime.cameras.snapshot = lambda: ({}, {'camera_1': 'USB disappeared'})
        self.wait_until(lambda: runtime.snapshot()['session_id'] is None)
        self.assertTrue(runtime.wait_recording(0))
        self.assertIn('USB disappeared', self.recordings[-1].reason)
        first_reason = self.recordings[-1].reason
        runtime.close('supervisor fault')
        self.assertEqual(self.recordings[-1].reason, first_reason)
        self.assertEqual(runtime.cameras.listeners, [])

    def test_manual_open_failure_cannot_restart_or_extend_an_execution_video(self):
        runtime = self.make_runtime()
        runtime.connect()
        runtime.arm()
        runtime.stop('本次模型执行结束')
        runtime.bridge = ManualOpenBridge(status='error')
        self.assertIsNone(runtime.open_gripper())
        self.assertFalse(runtime.snapshot()['armed'])
        self.assertEqual(len(self.recordings), 1)
        self.assertEqual(self.recordings[0].reason, '本次模型执行结束')
        self.assertTrue(runtime.wait_recording(0))

    def test_video_saving_does_not_delay_stop_and_blocks_rearm_until_finalized(self):
        class SlowSaveRecording(Recording):
            def stop(self, reason):
                self.reason = self.reason or reason
                self.status = 'saving'
        runtime = self.make_runtime(recorder_factory=SlowSaveRecording)
        runtime.connect()
        runtime.arm()
        started = time.monotonic()
        runtime.stop('test async video finalize')
        self.assertLess(time.monotonic() - started, .1)
        self.assertTrue(self.bridge.stopped.wait(1))
        self.assertFalse(runtime.wait_recording(0))
        with self.assertRaisesRegex(ClientError, '录像正在保存'):
            runtime.arm()
        self.recordings[-1].done.set()
        runtime.arm()
        self.assertEqual(len(self.recordings), 2)

    def test_stop_while_video_start_is_pending_cannot_later_arm(self):
        entered, release = threading.Event(), threading.Event()
        class SlowStartRecording(Recording):
            def start(self, session, frames):
                entered.set()
                release.wait(1)
                super().start(session, frames)
        runtime = self.make_runtime(recorder_factory=SlowStartRecording)
        runtime.connect()
        worker = threading.Thread(target=runtime.arm)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            runtime.stop('取消录像准备')
        finally:
            release.set()
            worker.join(1)
        self.assertNotIn('arm', [packet['type'] for packet in self.bridge.calls])
        self.assertFalse(runtime.snapshot()['armed'])
        self.assertEqual(runtime.cameras.listeners, [])

    def test_close_during_connection_cannot_later_start_cameras(self):
        class SlowHelloModel(Model):
            def request(self, packet):
                if packet['type'] == 'hello':
                    self.entered.set()
                    self.release.wait(1)
                return super().request(packet)
        model = SlowHelloModel()
        runtime = self.make_runtime(model=model)
        worker = threading.Thread(target=runtime.connect)
        worker.start()
        self.assertTrue(model.entered.wait(1))
        runtime.close()
        model.release.set()
        worker.join(1)
        self.assertFalse(runtime.cameras.started)
        self.assertFalse(runtime.snapshot()['connected'])

    def test_fault_close_revokes_control_and_prevents_old_window_reconnect(self):
        runtime = self.make_runtime()
        runtime.connect()
        runtime._arm_pending = True
        runtime.close('Native gripper stop timed out')
        snapshot = runtime.snapshot()
        self.assertFalse(snapshot['connected'])
        self.assertFalse(snapshot['armed'])
        self.assertFalse(snapshot['arming'])
        self.assertIsNone(snapshot['session_id'])
        self.assertFalse(snapshot['state']['ready'])
        self.assertFalse(snapshot['state']['armed'])
        self.assertEqual(snapshot['state']['fault'], 'Native gripper stop timed out')
        self.wait_until(lambda: runtime.cameras.closed)
        requests = len(self.bridge.calls)
        with self.assertRaises(ClientError):
            runtime.connect()
        with self.assertRaises(ClientError):
            runtime.arm()
        self.assertEqual(len(self.bridge.calls), requests)

    def test_late_state_reply_after_fault_close_cannot_restore_ready(self):
        class SlowMonitorBridge(Bridge):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()
                self.states = 0

            def request(self, packet):
                if packet['type'] == 'state':
                    self.states += 1
                    if self.states > 1:
                        self.entered.set()
                        self.release.wait(1)
                        return state(ready=True, armed=True)
                return super().request(packet)

        bridge = SlowMonitorBridge()
        runtime = self.make_runtime(bridge=bridge)
        runtime.connect()
        try:
            self.assertTrue(bridge.entered.wait(1))
            runtime.close('ROS launch exited')
        finally:
            bridge.release.set()
        for worker in runtime._workers:
            worker.join(1)
        self.assertFalse(runtime.snapshot()['connected'])
        self.assertFalse(runtime.snapshot()['state']['ready'])
        self.assertFalse(runtime.snapshot()['state']['armed'])
        self.assertEqual(runtime.snapshot()['state']['fault'], 'ROS launch exited')

    def test_stop_does_not_wait_for_cloud_and_late_response_is_discarded(self):
        runtime = self.make_runtime()
        runtime.connect()
        runtime.arm()
        self.assertTrue(self.model.entered.wait(1))
        started = time.monotonic()
        runtime.stop('test stop')
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertFalse(runtime.snapshot()['armed'])
        self.assertIsNone(runtime.snapshot()['session_id'])
        self.assertTrue(self.bridge.stopped.wait(1))
        self.model.release.set()
        time.sleep(0.05)
        self.assertEqual(self.bridge.submissions, [])
        self.assertTrue(self.model.cancelled)

    def test_mock_server_cannot_arm(self):
        runtime = self.make_runtime(model=Model(mock=True))
        runtime.connect()
        with self.assertRaisesRegex(ClientError, '模拟服务'):
            runtime.arm()
        self.assertNotIn('arm', [p['type'] for p in self.bridge.calls])

    def test_previous_stop_reason_does_not_prevent_explicit_new_arm(self):
        class PreviousFaultBridge(Bridge):
            def request(self, packet):
                result = super().request(packet)
                if packet['type'] == 'state' and self.token is None:
                    result['fault'] = 'previous user stop'
                return result
        runtime = self.make_runtime(bridge=PreviousFaultBridge())
        runtime.connect()
        runtime.arm()
        self.assertTrue(runtime.snapshot()['armed'])
        self.assertTrue(self.model.entered.wait(1))

    def test_waits_for_camera_frames_newer_than_controller_activation(self):
        class FrozenCameras(Cameras):
            def __init__(self):
                super().__init__()
                self.freeze = True
                self.captured = time.monotonic()

            def snapshot(self):
                captured = self.captured if self.freeze else time.monotonic()
                return {key: {'rgb': 'test-image', 'monotonic': captured, 'number': 1} for key in CAMERA_KEYS}, {}
        runtime = self.make_runtime()
        runtime.cameras = FrozenCameras()
        runtime.connect()
        runtime.arm()
        self.assertFalse(self.model.entered.wait(0.04))
        runtime.cameras.freeze = False
        self.assertTrue(self.model.entered.wait(1))

    def test_accepted_response_keeps_absolute_8d_actions_and_session(self):
        runtime = self.make_runtime()
        self.model.release.set()
        runtime.connect()
        runtime.arm()
        self.wait_until(lambda: bool(self.bridge.submissions))
        submitted = self.bridge.submissions[0]
        self.assertEqual(submitted['session_id'], runtime.snapshot()['session_id'])
        self.assertEqual(submitted['type'], 'submit')
        self.assertEqual(submitted['actions'], [[0.0] * 8 for _ in range(50)])
        self.assertEqual(submitted['fps'], 15)

    def test_first_action_waits_for_initial_camera_heartbeat(self):
        class RequiresHeartbeatBridge(Bridge):
            def __init__(self):
                super().__init__()
                self.camera_health = False

            def request(self, packet):
                if packet['type'] == 'heartbeat':
                    self.camera_health = True
                if packet['type'] == 'submit' and not self.camera_health:
                    raise ClientError('Fresh camera heartbeat required before actions')
                return super().request(packet)
        runtime = self.make_runtime(bridge=RequiresHeartbeatBridge())
        self.model.release.set()
        runtime.connect()
        runtime.arm()
        self.wait_until(lambda: bool(self.bridge.submissions))
        kinds = [p['type'] for p in self.bridge.calls]
        self.assertLess(kinds.index('heartbeat'), kinds.index('submit'))
        self.assertTrue(runtime.snapshot()['armed'])

    def test_incompatible_metadata_rejected_before_camera_start(self):
        runtime = self.make_runtime(model=Model(fps=30))
        with self.assertRaises(ValueError):
            runtime.connect()
        self.assertFalse(runtime.cameras.started)

    def test_changed_model_reply_disarms(self):
        model = Model()
        model.reply_override = {'model_id': 'different-checkpoint'}
        model.release.set()
        runtime = self.make_runtime(model=model)
        runtime.connect()
        runtime.arm()
        self.wait_until(lambda: runtime.snapshot()['session_id'] is None)
        self.assertEqual(self.bridge.submissions, [])
        self.assertIn('Checkpoint changed', runtime.snapshot()['fault'])

    def test_stale_cloud_observation_is_never_submitted(self):
        config = load_config()
        config['execution']['max_observation_age_s'] = 0.02
        runtime = self.make_runtime(config=config)
        runtime.connect()
        runtime.arm()
        self.assertTrue(self.model.entered.wait(1))
        time.sleep(0.04)
        self.model.release.set()
        self.wait_until(lambda: runtime.snapshot()['session_id'] is None)
        self.assertEqual(self.bridge.submissions, [])
        self.assertIn('过期', runtime.snapshot()['fault'])
        self.assertTrue(runtime.wait_recording(0))
        self.assertIn('过期', self.recordings[-1].reason)
        self.assertEqual(runtime.cameras.listeners, [])

    def test_stop_during_arm_preflight_cancels_pending_enable(self):
        class SlowStateBridge(Bridge):
            def __init__(self):
                super().__init__()
                self.slow = False
                self.entered, self.release = threading.Event(), threading.Event()

            def request(self, packet):
                if self.slow and packet['type'] == 'state' and threading.current_thread().name == 'test-arm':
                    self.entered.set()
                    self.release.wait(1)
                return super().request(packet)
        bridge = SlowStateBridge()
        runtime = self.make_runtime(bridge=bridge)
        runtime.connect()
        bridge.slow = True
        errors = []
        def arm():
            try:
                runtime.arm()
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=arm, name='test-arm')
        worker.start()
        self.assertTrue(bridge.entered.wait(1))
        runtime.stop('cancel preflight')
        bridge.release.set()
        worker.join(1)
        self.assertTrue(errors)
        self.assertNotIn('arm', [p['type'] for p in bridge.calls])
        self.assertFalse(runtime.snapshot()['armed'])

    def test_stop_during_delayed_arm_sends_another_stop_after_reply(self):
        class SlowArmBridge(Bridge):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def request(self, packet):
                if packet['type'] == 'arm':
                    self.entered.set()
                    self.release.wait(1)
                return super().request(packet)
        bridge = SlowArmBridge()
        runtime = self.make_runtime(bridge=bridge)
        runtime.connect()
        worker = threading.Thread(target=runtime.arm)
        worker.start()
        self.assertTrue(bridge.entered.wait(1))
        runtime.stop('cancel arm')
        self.assertTrue(bridge.stopped.wait(1))
        bridge.release.set()
        worker.join(1)
        self.wait_until(lambda: bridge.token is None)
        self.assertFalse(runtime.snapshot()['armed'])
        self.assertEqual(self.bridge.submissions, [])

    def test_camera_ages_and_skew(self):
        config = load_config()['cameras']
        frames = {key: {'monotonic': 10.0} for key in CAMERA_KEYS}
        self.assertEqual(check_cameras(frames, {}, config, 10.1), 10.0)
        with self.assertRaisesRegex(ClientError, '过期'):
            check_cameras(frames, {}, config, 10.21)
        frames[CAMERA_KEYS[1]]['monotonic'] = 10.1
        with self.assertRaisesRegex(ClientError, '时间差'):
            check_cameras(frames, {}, config, 10.11)

    def make_preview_runtime(self, model=None, bridge=None):
        # Existing connected cameras/state, with no background workers needed.
        runtime = self.make_runtime(model=model, bridge=bridge)
        runtime._connected, runtime._metadata = True, self.model.meta
        self.events = []
        runtime.on_event = self.events.append
        return runtime

    def test_preview_inference_reports_raw_actions_and_only_reads_bridge_state(self):
        runtime = self.make_preview_runtime()
        self.model.release.set()
        actions = [[.01 * i, 0., 0., 0., 0., 0., 0., .081] for i in range(50)]
        self.model.reply_override['actions'] = actions
        report = runtime.preview_inference()
        self.assertIsNotNone(report)
        self.assertEqual([packet['type'] for packet in self.bridge.calls], ['state'])
        self.assertEqual(report['actions'], actions)  # Width is deliberately not clamped.
        self.assertEqual(report['aligned_target'], actions[report['aligned_index']])
        self.assertEqual(report['max_joint_delta_joint'], 1)
        self.assertEqual(report['model_id'], self.model.meta['model_id'])
        self.assertEqual(report['checkpoint'], self.model.meta['checkpoint'])
        self.assertFalse(runtime.snapshot()['armed'])
        self.assertIsNone(runtime.snapshot()['session_id'])
        self.assertEqual([event['kind'] for event in self.events], ['preview_inference'])
        self.assertIs(self.events[0]['report'], report)

    def test_preview_requires_connected_unarmed_and_no_activation_preflight(self):
        runtime = self.make_preview_runtime()
        for connected, armed, session, pending in ((False, False, None, False),
                                                   (True, True, 'session', False),
                                                   (True, False, 'session', False),
                                                   (True, False, None, True)):
            with self.subTest(connected=connected, armed=armed, session=session, pending=pending):
                runtime._connected, runtime._armed = connected, armed
                runtime._session, runtime._arm_pending = session, pending
                self.assertIsNone(runtime.preview_inference())
                self.assertEqual(self.events[-1]['kind'], 'operation_error')
        runtime._session = None
        runtime._arm_pending = runtime._armed = False
        self.assertEqual(self.bridge.calls, [])
        self.assertEqual(self.model.calls, [])

    def test_preview_rejects_an_externally_armed_bridge_without_model_request(self):
        class ArmedBridge(Bridge):
            def request(self, packet):
                result = super().request(packet)
                result['armed'] = True
                return result
        runtime = self.make_preview_runtime(bridge=ArmedBridge())
        self.assertIsNone(runtime.preview_inference())
        self.assertEqual([packet['type'] for packet in self.bridge.calls], ['state'])
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.events[-1]['kind'], 'operation_error')

    def test_preview_late_response_after_stop_is_discarded_without_bridge_commands(self):
        runtime = self.make_preview_runtime()
        reports = []
        worker = threading.Thread(target=lambda: reports.append(runtime.preview_inference()))
        worker.start()
        self.assertTrue(self.model.entered.wait(1))
        runtime.stop('cancel read-only prediction')
        self.model.release.set()
        worker.join(1)
        self.assertEqual(reports, [None])
        self.assertEqual([packet['type'] for packet in self.bridge.calls], ['state'])
        self.assertEqual([event['kind'] for event in self.events], ['stopped'])
        self.assertFalse(runtime._preview_pending)

    def test_preview_model_failure_reports_operation_error_without_stopping_or_arming(self):
        class FailingModel(Model):
            def request(self, packet):
                self.calls.append(copy.deepcopy(packet))
                raise RuntimeError('diagnostic-only model failure')
        runtime = self.make_preview_runtime(model=FailingModel())
        self.assertIsNone(runtime.preview_inference())
        self.assertEqual([packet['type'] for packet in self.bridge.calls], ['state'])
        self.assertEqual(self.events[0]['kind'], 'operation_error')
        self.assertIn('diagnostic-only', self.events[0]['message'])
        self.assertIsNone(runtime.snapshot()['fault'])
        self.assertIsNone(runtime.snapshot()['session_id'])

    def test_preview_cancelled_during_state_read_never_requests_model(self):
        class SlowStateBridge(Bridge):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def request(self, packet):
                self.entered.set()
                self.release.wait(1)
                return super().request(packet)
        bridge = SlowStateBridge()
        runtime = self.make_preview_runtime(bridge=bridge)
        reports = []
        worker = threading.Thread(target=lambda: reports.append(runtime.preview_inference()))
        worker.start()
        self.assertTrue(bridge.entered.wait(1))
        runtime.stop('cancel preview state read')
        bridge.release.set()
        worker.join(1)
        self.assertEqual(reports, [None])
        self.assertEqual(self.model.calls, [])
        self.assertEqual([packet['type'] for packet in bridge.calls], ['state'])
        self.assertEqual([event['kind'] for event in self.events], ['stopped'])

    def test_preview_cannot_start_during_actual_activation_preflight(self):
        class SlowStateBridge(Bridge):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def request(self, packet):
                self.entered.set()
                self.release.wait(1)
                return super().request(packet)
        bridge = SlowStateBridge()
        runtime = self.make_preview_runtime(bridge=bridge)
        errors = []
        def activate():
            try:
                runtime.arm()
            except ClientError as exc:
                errors.append(str(exc))
        worker = threading.Thread(target=activate)
        worker.start()
        self.assertTrue(bridge.entered.wait(1))
        self.assertTrue(runtime.snapshot()['arming'])
        self.assertIsNone(runtime.preview_inference())
        runtime.stop('cancel test activation')
        bridge.release.set()
        worker.join(1)
        self.assertTrue(errors)
        self.assertEqual(self.model.calls, [])
        self.assertEqual([packet['type'] for packet in bridge.calls], ['state'])

    def test_preview_requires_full_fifty_action_block_without_submitting(self):
        runtime = self.make_preview_runtime()
        self.model.release.set()
        self.model.reply_override['actions'] = [[0.] * 8 for _ in range(49)]
        self.assertIsNone(runtime.preview_inference())
        self.assertEqual(self.events[-1]['kind'], 'operation_error')
        self.assertIn('50', self.events[-1]['message'])
        self.assertEqual([packet['type'] for packet in self.bridge.calls], ['state'])

    def test_arm_and_second_preview_are_rejected_while_preview_request_is_pending(self):
        runtime = self.make_preview_runtime()
        worker = threading.Thread(target=runtime.preview_inference)
        worker.start()
        self.assertTrue(self.model.entered.wait(1))
        with self.assertRaisesRegex(ClientError, '只预测诊断'):
            runtime.arm()
        self.assertIsNone(runtime.preview_inference())
        self.model.release.set()
        worker.join(1)
        self.assertEqual([packet['type'] for packet in self.bridge.calls], ['state'])
        self.assertEqual(len(self.model.calls), 1)

    def test_preview_rejects_incompatible_model_response_without_robot_commands(self):
        runtime = self.make_preview_runtime()
        self.model.release.set()
        self.model.reply_override['model_id'] = 'unexpected-model'
        self.assertIsNone(runtime.preview_inference())
        self.assertEqual([packet['type'] for packet in self.bridge.calls], ['state'])
        self.assertEqual(self.events[0]['kind'], 'operation_error')
        self.assertFalse(runtime.snapshot()['armed'])

    def test_preview_uses_oldest_source_for_latency_skip_and_exact_jpeg_size(self):
        clock = [100.]
        class FixedCameras(Cameras):
            def snapshot(self):
                return {key: {'rgb': 'test-image', 'monotonic': 100., 'number': 123,
                              'source_timestamp_domain': 'global_time'} for key in CAMERA_KEYS}, {}
        class SourceAgesBridge(Bridge):
            def request(self, packet):
                self.calls.append(copy.deepcopy(packet))
                return state(state_age_s=.05, gripper_age_s=.04)
        class TimedModel(Model):
            def request(self, packet):
                result = super().request(packet)
                clock[0] += .2
                return result
        runtime = self.make_preview_runtime(model=TimedModel(), bridge=SourceAgesBridge())
        runtime.clock, runtime.cameras = lambda: clock[0], FixedCameras()
        self.model.release.set()
        actions = [[.01*i, 0., 0., 0., 0., 0., 0., .081] for i in range(50)]
        self.model.reply_override['actions'] = actions
        with patch('client_runtime.encode_image', return_value={'data': 'YWJjZA=='}):
            report = runtime.preview_inference()
        self.assertAlmostEqual(report['capture_monotonic'], 99.95)
        self.assertAlmostEqual(report['observation_age_s'], .25)
        self.assertAlmostEqual(report['rpc_s'], .2)
        self.assertEqual(report['aligned_index'], 4)
        self.assertEqual(report['aligned_target'], actions[4])
        self.assertAlmostEqual(report['max_joint_delta_rad'], .04)
        self.assertEqual(report['encoded_jpeg_bytes'], {key: 4 for key in CAMERA_KEYS})
        self.assertEqual(report['camera_clock_domains'], {key: 'global_time' for key in CAMERA_KEYS})
        self.assertEqual([packet['type'] for packet in self.bridge.calls], ['state'])

    def start_inference_only(self, runtime):
        # Isolate model execution from the monitor to deterministically exercise
        # each cloud response/rejection race, without a real bridge or sockets.
        token = '8a4a1d0f-df64-4b14-bda1-648efcf13f37'
        with runtime._lock:
            runtime._session, runtime._armed = token, True
            runtime._arm_monotonic = None
        self.bridge.token = token
        worker = threading.Thread(target=runtime._inference_loop, daemon=True)
        runtime._workers.append(worker)
        worker.start()
        return worker

    def test_rejected_action_event_keeps_full_raw_chunk_without_request_images_after_stop(self):
        class RecordingModel(Model):
            def request(self, packet):
                reply = super().request(packet)
                self.raw_reply = copy.deepcopy(reply)
                return reply
        class RejectWidth(Bridge):
            def request(self, packet):
                if packet['type'] == 'submit':
                    self.calls.append(copy.deepcopy(packet))
                    raise ClientError('Gripper target outside configured meter range')
                return super().request(packet)
        model, bridge = RecordingModel(), RejectWidth()
        runtime = self.make_preview_runtime(model=model, bridge=bridge)
        original_actions = [[i * .001] * 7 + [.084 if i == 27 else .08] for i in range(50)]
        model.reply_override['actions'] = copy.deepcopy(original_actions)
        model.reply_override['raw_model_diagnostic'] = {'kept': True}
        model.release.set()
        self.start_inference_only(runtime)
        self.wait_until(lambda: any(event['kind'] == 'action_rejected' for event in self.events))
        kinds = [event['kind'] for event in self.events]
        self.assertLess(kinds.index('stopped'), kinds.index('action_rejected'))
        rejected = next(event for event in self.events if event['kind'] == 'action_rejected')
        self.assertEqual(rejected['response'], model.raw_reply)
        self.assertEqual(rejected['response']['actions'], original_actions)
        self.assertEqual(len(rejected['response']['actions']), 50)
        self.assertTrue(all(len(row) == 8 for row in rejected['response']['actions']))
        self.assertEqual(rejected['response']['raw_model_diagnostic'], {'kept': True})
        self.assertTrue(rejected['session_current'])
        self.assertFalse(rejected['stale'])
        self.assertEqual(rejected['session_id'], model.raw_reply['session_id'])
        context = rejected['request_context']
        self.assertEqual(set(context), {'session_id', 'request_id', 'capture_seq',
                                       'capture_monotonic', 'state', 'task', 'model',
                                       'request_s', 'observation_age_s', 'bridge_submit_monotonic'})
        self.assertEqual(context['request_id'], model.raw_reply['request_id'])
        self.assertEqual(context['state'], [0.] * 8)
        self.assertEqual(context['task'], runtime.config['task'])
        self.assertEqual(context['model']['checkpoint'], model.meta['checkpoint'])
        self.assertGreaterEqual(context['request_s'], 0.)
        self.assertGreaterEqual(context['observation_age_s'], 0.)
        self.assertGreaterEqual(context['bridge_submit_monotonic'], context['capture_monotonic'])
        self.assertNotIn('images', str(context))
        self.assertNotIn('test-image', str(context))
        self.assertFalse(runtime.snapshot()['armed'])
        self.assertTrue(bridge.stopped.wait(1))

    def test_accepted_inference_transfers_width_diagnostics_without_changing_raw_reply(self):
        class OneModelReply(Model):
            def request(self, packet):
                if self.calls:
                    raise RuntimeError('network failure in next iteration')
                self.returned_reply = super().request(packet)
                self.original_reply = copy.deepcopy(self.returned_reply)
                return self.returned_reply
        class WidthDiagnostics(Bridge):
            def __init__(self):
                super().__init__()
                self.diagnostics = dict(width_clipped_steps=[0, 3, 49],
                                        width_out_of_range_steps=[0, 3, 49],
                                        width_saturation={'low': 1, 'high': 2},
                                        raw_width_min_m=-.0001, raw_width_max_m=.081,
                                        raw_width_min=-.0001, raw_width_max=.081)

            def request(self, packet):
                reply = super().request(packet)
                if packet['type'] == 'submit':
                    reply.update(copy.deepcopy(self.diagnostics))
                return reply
        model, bridge = OneModelReply(), WidthDiagnostics()
        runtime = self.make_preview_runtime(model=model, bridge=bridge)
        model.reply_override['actions'] = [[0.] * 7 + [.081] for _ in range(50)]
        model.release.set()
        self.start_inference_only(runtime)
        self.wait_until(lambda: runtime.snapshot()['session_id'] is None)
        accepted = next(event for event in self.events if event['kind'] == 'inference')
        for key, value in bridge.diagnostics.items():
            self.assertEqual(accepted[key], value)
        self.assertEqual(model.returned_reply, model.original_reply)
        self.assertEqual(bridge.submissions[0]['actions'], model.original_reply['actions'])
        self.assertNotIn('action_rejected', [event['kind'] for event in self.events])
        self.assertIn('network failure', runtime.snapshot()['fault'])

    def test_camera_failure_in_new_iteration_does_not_reuse_previous_action_reply(self):
        class OnceCameras(Cameras):
            def __init__(self):
                super().__init__()
                self.snapshots = 0

            def snapshot(self):
                self.snapshots += 1
                if self.snapshots > 1:
                    raise ClientError('new camera iteration failed')
                return super().snapshot()
        runtime = self.make_preview_runtime()
        runtime.cameras = OnceCameras()
        self.model.release.set()
        self.start_inference_only(runtime)
        self.wait_until(lambda: runtime.snapshot()['session_id'] is None)
        self.assertEqual(len(self.bridge.submissions), 1)
        self.assertEqual([event['kind'] for event in self.events], ['inference', 'stopped'])
        self.assertIn('new camera iteration', runtime.snapshot()['fault'])

    def test_cancelled_late_cloud_reply_is_not_logged_as_an_action_rejection(self):
        runtime = self.make_preview_runtime()
        self.start_inference_only(runtime)
        self.assertTrue(self.model.entered.wait(1))
        runtime.stop('operator cancelled cloud request')
        self.model.release.set()
        self.assertTrue(self.bridge.stopped.wait(1))
        time.sleep(.04)
        self.assertEqual([event['kind'] for event in self.events], ['stopped'])
        self.assertEqual(self.bridge.submissions, [])

    def test_late_submitted_bridge_rejection_is_retained_as_stale_without_second_stop(self):
        class SlowReject(Bridge):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def request(self, packet):
                if packet['type'] == 'submit':
                    self.calls.append(copy.deepcopy(packet))
                    self.entered.set()
                    self.release.wait(1)
                    raise ClientError('late bridge rejection')
                return super().request(packet)
        bridge = SlowReject()
        runtime = self.make_preview_runtime(bridge=bridge)
        self.model.release.set()
        self.start_inference_only(runtime)
        self.assertTrue(bridge.entered.wait(1))
        runtime.stop('operator cancelled pending submit')
        self.assertTrue(bridge.stopped.wait(1))
        bridge.release.set()
        self.wait_until(lambda: any(event['kind'] == 'action_rejected' for event in self.events))
        self.assertEqual([event['kind'] for event in self.events], ['stopped', 'action_rejected'])
        rejected = self.events[-1]
        self.assertTrue(rejected['stale'])
        self.assertFalse(rejected['session_current'])
        self.assertIsNone(rejected['current_session_id'])
        self.assertEqual(rejected['session_id'], rejected['request_context']['session_id'])
        self.assertEqual(len(rejected['response']['actions']), 50)
        self.assertEqual(len([p for p in bridge.calls if p['type'] == 'stop']), 1)
        self.assertEqual(runtime.snapshot()['fault'], 'operator cancelled pending submit')

    def test_monitor_stop_before_submit_rejection_logs_raw_without_stopping_new_session(self):
        class SlowReject(Bridge):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def request(self, packet):
                if packet['type'] == 'submit':
                    self.calls.append(copy.deepcopy(packet))
                    self.entered.set()
                    self.release.wait(1)
                    raise ClientError('Gripper target outside configured meter range')
                return super().request(packet)
        bridge = SlowReject()
        runtime = self.make_preview_runtime(bridge=bridge)
        self.model.release.set()
        self.start_inference_only(runtime)
        self.assertTrue(bridge.entered.wait(1))
        old_token = runtime.snapshot()['session_id']
        runtime.stop('monitor observed bridge fault first')
        self.assertTrue(bridge.stopped.wait(1))
        new_token = 'new-arming-session-0001'
        with runtime._lock:
            runtime._session = new_token
            runtime._arm_pending = True
            runtime._last_fault = None
        bridge.release.set()
        self.wait_until(lambda: any(event['kind'] == 'action_rejected' for event in self.events))
        rejected = self.events[-1]
        self.assertEqual(rejected['kind'], 'action_rejected')
        self.assertTrue(rejected['stale'])
        self.assertFalse(rejected['session_current'])
        self.assertEqual(rejected['session_id'], old_token)
        self.assertEqual(rejected['current_session_id'], new_token)
        self.assertEqual(len(rejected['response']['actions']), 50)
        self.assertEqual(runtime.snapshot()['session_id'], new_token)
        self.assertTrue(runtime._arm_pending)
        self.assertIsNone(runtime.snapshot()['fault'])
        self.assertEqual(len([p for p in bridge.calls if p['type'] == 'stop']), 1)

    def test_late_accepted_bridge_reply_is_discarded_after_stop_or_new_session(self):
        class SlowAccept(Bridge):
            def __init__(self):
                super().__init__()
                self.entered, self.release, self.returned = threading.Event(), threading.Event(), threading.Event()

            def request(self, packet):
                reply = super().request(packet)
                if packet['type'] == 'submit':
                    reply.update(width_clipped_steps=[3], raw_width_max=.081)
                    self.entered.set()
                    self.release.wait(1)
                    self.returned.set()
                return reply
        bridge = SlowAccept()
        runtime = self.make_preview_runtime(bridge=bridge)
        self.model.release.set()
        self.start_inference_only(runtime)
        self.assertTrue(bridge.entered.wait(1))
        runtime.stop('operator stopped while acceptance reply pending')
        self.assertTrue(bridge.stopped.wait(1))
        with runtime._lock:
            runtime._session = 'new-session-waiting-for-activation'
            runtime._arm_pending = True
            runtime._last_fault = None
        bridge.release.set()
        self.assertTrue(bridge.returned.wait(1))
        time.sleep(.04)
        self.assertEqual([event['kind'] for event in self.events], ['stopped'])
        self.assertEqual(runtime.snapshot()['session_id'], 'new-session-waiting-for-activation')
        self.assertTrue(runtime._arm_pending)
        self.assertIsNone(runtime.snapshot()['fault'])

    def test_manual_open_only_uses_state_and_manual_command_and_verifies_actual_width(self):
        bridge = ManualOpenBridge(width=.079)
        runtime = self.make_preview_runtime(bridge=bridge)
        result = runtime.open_gripper()
        self.assertIsNotNone(result)
        self.assertEqual([p['type'] for p in bridge.calls], ['state', 'manual_open', 'state'])
        command = bridge.calls[1]
        self.assertEqual(set(command), {'version', 'type', 'operation_id'})
        self.assertEqual(len(command['operation_id']), 36)
        self.assertEqual([event['kind'] for event in self.events], ['manual_gripper_opened'])
        self.assertEqual(self.events[0]['width_m'], .079)
        self.assertEqual(self.events[0]['operation_id'], command['operation_id'])
        self.assertIs(self.events[0]['state'], result)
        self.assertEqual(self.model.calls, [])
        self.assertFalse(runtime.snapshot()['manual_opening'])
        self.assertFalse(runtime.snapshot()['armed'])
        self.assertIsNone(runtime.snapshot()['session_id'])

    def test_manual_open_requires_connected_and_no_other_operation(self):
        runtime = self.make_preview_runtime(bridge=ManualOpenBridge())
        cases = ({'_connected': False}, {'_armed': True}, {'_session': 'other'},
                 {'_arm_pending': True}, {'_preview_pending': True}, {'_manual_pending': True})
        for fields in cases:
            with self.subTest(fields=fields):
                for key, value in fields.items():
                    setattr(runtime, key, value)
                self.assertIsNone(runtime.open_gripper())
                self.assertEqual(self.events[-1]['kind'], 'operation_error')
                for key in fields:
                    setattr(runtime, key, True if key == '_connected' else None if key == '_session' else False)
        self.assertEqual(self.bridge.calls, [])
        self.assertEqual(self.model.calls, [])

    def test_manual_open_cannot_overlap_arm_or_preview_or_second_open(self):
        class SlowManual(ManualOpenBridge):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def request(self, packet):
                result = super().request(packet)
                if packet['type'] == 'manual_open':
                    self.entered.set()
                    self.release.wait(1)
                return result
        bridge = SlowManual()
        runtime = self.make_preview_runtime(bridge=bridge)
        worker = threading.Thread(target=runtime.open_gripper)
        worker.start()
        self.assertTrue(bridge.entered.wait(1))
        self.assertTrue(runtime.snapshot()['manual_opening'])
        with self.assertRaisesRegex(ClientError, '手动开爪'):
            runtime.arm()
        self.assertIsNone(runtime.preview_inference())
        self.assertIsNone(runtime.open_gripper())
        self.assertTrue(runtime.snapshot()['manual_opening'])  # Rejected duplicate does not clear owner.
        bridge.release.set()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual([p['type'] for p in bridge.calls], ['state', 'manual_open', 'state'])
        self.assertEqual(self.model.calls, [])

    def test_manual_open_stop_during_preflight_revokes_id_without_sending_move(self):
        class SlowState(ManualOpenBridge):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def request(self, packet):
                if packet['type'] == 'state':
                    self.entered.set()
                    self.release.wait(1)
                return super().request(packet)
        bridge = SlowState()
        runtime = self.make_preview_runtime(bridge=bridge)
        worker = threading.Thread(target=runtime.open_gripper)
        worker.start()
        self.assertTrue(bridge.entered.wait(1))
        operation_id = runtime._manual_operation_id
        runtime.stop('cancel before manual send')
        self.assertTrue(bridge.stopped.wait(1))
        stop = next(p for p in bridge.calls if p['type'] == 'stop')
        self.assertEqual(stop['manual_operation_id'], operation_id)
        self.assertIsNone(stop['session_id'])
        bridge.release.set()
        worker.join(1)
        self.assertNotIn('manual_open', [p['type'] for p in bridge.calls])
        self.assertEqual([event['kind'] for event in self.events], ['stopped'])
        self.assertFalse(runtime.snapshot()['manual_opening'])

    def test_manual_open_stop_while_start_reply_waits_rejects_late_completion(self):
        class SlowReply(ManualOpenBridge):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def request(self, packet):
                result = super().request(packet)
                if packet['type'] == 'manual_open':
                    self.entered.set()
                    self.release.wait(1)
                return result
        bridge = SlowReply()
        runtime = self.make_preview_runtime(bridge=bridge)
        worker = threading.Thread(target=runtime.open_gripper)
        worker.start()
        self.assertTrue(bridge.entered.wait(1))
        operation_id = runtime._manual_operation_id
        runtime.stop('cancel manual Move')
        self.assertTrue(bridge.stopped.wait(1))
        self.assertIn(operation_id, bridge.revoked)
        self.assertEqual(bridge.manual['status'], 'cancelled')
        bridge.release.set()
        worker.join(1)
        self.assertEqual([event['kind'] for event in self.events], ['stopped'])
        self.assertEqual([p['type'] for p in bridge.calls], ['state', 'manual_open', 'stop'])
        self.assertFalse(runtime.snapshot()['manual_opening'])

    def test_close_during_manual_open_independently_sends_revoke_without_session(self):
        class SlowReply(ManualOpenBridge):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def request(self, packet):
                result = super().request(packet)
                if packet['type'] == 'manual_open':
                    self.entered.set()
                    self.release.wait(1)
                return result
        bridge = SlowReply()
        runtime = self.make_preview_runtime(bridge=bridge)
        worker = threading.Thread(target=runtime.open_gripper)
        worker.start()
        self.assertTrue(bridge.entered.wait(1))
        operation_id = runtime._manual_operation_id
        runtime.close()
        self.assertTrue(bridge.stopped.wait(1))
        self.assertIn(operation_id, bridge.revoked)
        bridge.release.set()
        worker.join(1)
        self.assertFalse(runtime.snapshot()['connected'])
        self.assertFalse(runtime.snapshot()['manual_opening'])
        self.assertNotIn('manual_gripper_opened', [event['kind'] for event in self.events])

    def test_manual_completed_without_79mm_actual_width_times_out_and_stops(self):
        clock = [100.]
        class NarrowComplete(ManualOpenBridge):
            def __init__(self):
                super().__init__(width=.078)

            def request(self, packet):
                result = super().request(packet)
                if packet['type'] == 'state' and self.manual is not None:
                    clock[0] += 1.55
                return result
        bridge = NarrowComplete()
        runtime = self.make_preview_runtime(bridge=bridge)
        runtime.clock = lambda: clock[0]
        self.assertIsNone(runtime.open_gripper())
        self.assertTrue(bridge.stopped.wait(1))
        self.assertIn('未确认实际全开', self.events[-1]['message'])
        self.assertNotIn('manual_gripper_opened', [event['kind'] for event in self.events])
        self.assertLessEqual(clock[0] - 100., 7.)
        stop = next(p for p in bridge.calls if p['type'] == 'stop')
        self.assertEqual(stop['manual_operation_id'], bridge.manual['operation_id'])
        self.assertTrue(set(p['type'] for p in bridge.calls) <= {'state', 'manual_open', 'stop'})

    def test_manual_completed_with_stale_gripper_width_never_claims_full_open(self):
        class StaleComplete(ManualOpenBridge):
            def request(self, packet):
                result = super().request(packet)
                if packet['type'] == 'state' and self.manual is not None:
                    result['gripper_age_s'] = .51
                return result
        bridge = StaleComplete()
        runtime = self.make_preview_runtime(bridge=bridge)
        self.assertIsNone(runtime.open_gripper())
        self.assertTrue(bridge.stopped.wait(1))
        self.assertEqual(self.events[-1]['kind'], 'operation_error')
        self.assertIn('gripper_age_s', self.events[-1]['message'])
        self.assertNotIn('manual_gripper_opened', [event['kind'] for event in self.events])

    def test_manual_pending_and_accepted_wait_for_completed_actual_open(self):
        class Progress(ManualOpenBridge):
            def __init__(self):
                super().__init__(status='pending')
                self.polls = 0

            def request(self, packet):
                result = super().request(packet)
                if packet['type'] == 'state' and self.manual is not None:
                    self.polls += 1
                    result['manual_open']['status'] = ('pending', 'accepted', 'completed')[min(self.polls-1, 2)]
                return result
        bridge = Progress()
        runtime = self.make_preview_runtime(bridge=bridge)
        self.assertIsNotNone(runtime.open_gripper())
        self.assertEqual(bridge.polls, 3)
        self.assertEqual([event['kind'] for event in self.events], ['manual_gripper_opened'])

    def test_manual_error_or_cancelled_result_requests_stop(self):
        for status in ('error', 'cancelled'):
            with self.subTest(status=status):
                bridge = ManualOpenBridge(status=status)
                runtime = self.make_preview_runtime(bridge=bridge)
                self.assertIsNone(runtime.open_gripper())
                self.assertTrue(bridge.stopped.wait(1))
                self.assertEqual(self.events[-1]['kind'], 'operation_error')
                self.assertIn(status, self.events[-1]['message'])
                self.assertFalse(runtime.snapshot()['manual_opening'])
                self.assertNotIn('manual_gripper_opened', [event['kind'] for event in self.events])

    def test_manual_wrong_operation_reply_and_externally_armed_state_are_rejected(self):
        class WrongId(ManualOpenBridge):
            def request(self, packet):
                result = super().request(packet)
                if packet['type'] == 'manual_open':
                    result['operation_id'] = 'wrong-id'
                return result
        class ArmedState(ManualOpenBridge):
            def request(self, packet):
                result = super().request(packet)
                if packet['type'] == 'state':
                    result['armed'] = True
                return result
        for bridge in (WrongId(), ArmedState()):
            with self.subTest(bridge=type(bridge).__name__):
                runtime = self.make_preview_runtime(bridge=bridge)
                self.assertIsNone(runtime.open_gripper())
                self.assertTrue(bridge.stopped.wait(1))
                self.assertEqual(self.events[-1]['kind'], 'operation_error')
                self.assertFalse(runtime.snapshot()['manual_opening'])
                self.assertTrue(set(p['type'] for p in bridge.calls) <= {'state', 'manual_open', 'stop'})

    def test_manual_late_error_after_stop_is_discarded_and_cannot_clear_new_owner(self):
        class LateFailure(ManualOpenBridge):
            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def request(self, packet):
                if packet['type'] == 'manual_open':
                    self.entered.set()
                    self.release.wait(1)
                    raise RuntimeError('late transport failure')
                return super().request(packet)
        bridge = LateFailure()
        runtime = self.make_preview_runtime(bridge=bridge)
        worker = threading.Thread(target=runtime.open_gripper)
        worker.start()
        self.assertTrue(bridge.entered.wait(1))
        runtime.stop('cancel late opening')
        self.assertTrue(bridge.stopped.wait(1))
        with runtime._lock:
            runtime._manual_operation_id = 'a-new-owner'
            runtime._manual_pending = True
        bridge.release.set()
        worker.join(1)
        self.assertEqual([event['kind'] for event in self.events], ['stopped'])
        self.assertEqual(runtime._manual_operation_id, 'a-new-owner')
        self.assertTrue(runtime.snapshot()['manual_opening'])
        with runtime._lock:
            runtime._manual_operation_id, runtime._manual_pending = None, False

    def test_state_age_and_units(self):
        self.assertEqual(len(check_state(state())), 8)
        for packet in (state(state_age_s=0.21), state(gripper_age_s=0.51),
                       state(state=[0.0] * 7 + [0.09]), state(state=[float('nan')] * 8),
                       state(state_age_s=True)):
            with self.assertRaises(ClientError):
                check_state(packet)


class CameraTimestampTests(unittest.TestCase):
    def test_global_source_age_is_preserved_in_local_monotonic_time(self):
        mapper = CameraTimestampMapper()
        capture = mapper.map(1_000_000., 'timestamp_domain.global_time', 1, 50., 1000.08)
        self.assertAlmostEqual(capture, 49.92)
        frames = {key: {'monotonic': capture} for key in CAMERA_KEYS}
        self.assertAlmostEqual(check_cameras(frames, {}, load_config()['cameras'], 50.), capture)

    def test_buffered_global_frame_is_rejected_even_with_fresh_receipt(self):
        mapper = CameraTimestampMapper()
        with self.assertRaisesRegex(ClientError, '过期'):
            mapper.map(1_000_000., 'global_time', 1, 50., 1000.3)

    def test_frame_source_future_nonfinite_and_regressions_are_rejected(self):
        for source in (float('nan'), float('inf')):
            with self.assertRaises(ClientError):
                CameraTimestampMapper().map(source, 'global_time', 1, 50., 1000.)
        with self.assertRaisesRegex(ClientError, '未来'):
            CameraTimestampMapper().map(1_000_100., 'system_time', 1, 50., 1000.)
        mapper = CameraTimestampMapper()
        mapper.map(1_000_000., 'global_time', 10, 50., 1000.01)
        with self.assertRaisesRegex(ClientError, '帧号'):
            mapper.map(1_000_010., 'global_time', 10, 50.01, 1000.02)
        with self.assertRaisesRegex(ClientError, '时间戳'):
            mapper.map(999_999., 'global_time', 11, 50.01, 1000.02)

    def test_hardware_calibration_uses_minimum_receive_offset(self):
        mapper = CameraTimestampMapper(hardware_warmup_frames=3)
        self.assertIsNone(mapper.map(1000., 'hardware_clock', 1, 100.10, 1000.))
        self.assertIsNone(mapper.map(1033., 'hardware_clock', 2, 100.183, 1000.033))
        self.assertAlmostEqual(mapper.map(1066., 'hardware_clock', 3, 100.116, 1000.066), 100.116)
        self.assertAlmostEqual(mapper.map(1099., 'hardware_clock', 4, 100.149, 1000.099), 100.149)
        capture = mapper.map(1132., 'hardware_clock', 5, 100.212, 1000.132)
        self.assertAlmostEqual(capture, 100.182)
        self.assertAlmostEqual(100.212 - capture, .030)
        with self.assertRaisesRegex(ClientError, '过期'):
            mapper.map(1165., 'hardware_clock', 6, 100.5, 1000.165)

    def test_global_source_clock_jump_cannot_move_capture_backwards(self):
        mapper = CameraTimestampMapper()
        mapper.map(1_000_000., 'global_time', 1, 50., 1000.01)
        with self.assertRaisesRegex(ClientError, '回退'):
            mapper.map(1_000_001., 'global_time', 2, 50.01, 1000.11)


if __name__ == '__main__':
    unittest.main()
