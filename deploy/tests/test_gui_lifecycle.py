"""Fault-window behavior without Tk, cameras, ROS, SSH, or network access."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy_fr3 import DeploymentWindow


class Variable:
    def __init__(self):
        self.value = ''

    def set(self, value):
        self.value = value


class FaultWindowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'session_status.json'
        self.window = window = DeploymentWindow.__new__(DeploymentWindow)
        window.session_status, window.session_failure = self.path, None
        window.root, window.runtime = MagicMock(), MagicMock()
        window.status_label, window.session_label = MagicMock(), MagicMock()
        window.runtime.snapshot.return_value = dict(connected=False, armed=False, arming=False,
                                                    manual_opening=False, state=None, metadata=None)
        window.runtime.cameras.snapshot.return_value = ({}, {})
        window.events = queue.Queue()
        window.closing = window.busy = False
        window.quit_requested = threading.Event()
        window.runtime.wait_recording.return_value = True
        window.messagebox = MagicMock()
        window.photo_refs, window.preview_labels = [], []
        for name in ('status', 'session_info', 'model_info', 'robot_info', 'camera_info', 'timing', 'recording_info'):
            setattr(window, name, Variable())
        for name in ('connect_button', 'predict_button', 'open_button', 'arm_button', 'stop_button'):
            setattr(window, name, MagicMock())

    def write_failure(self, **changes):
        record = dict(version=1, phase='stopping', reason='Native gripper stop timed out',
                      source='ros.log', source_line='[FATAL] Native gripper stop timed out',
                      logs='/tmp/session/logs')
        record.update(changes)
        self.path.write_text(json.dumps(record))

    def test_fault_retains_window_with_first_reason_and_disables_all_controls(self):
        self.write_failure()
        self.window.events.put(dict(kind='status_error', message='Connection refused'))
        self.window.update()
        self.window.runtime.close.assert_called_once_with('部署会话故障: Native gripper stop timed out')
        self.window.root.destroy.assert_not_called()
        for name in ('connect_button', 'predict_button', 'open_button', 'arm_button', 'stop_button'):
            getattr(self.window, name).configure.assert_called_with(state='disabled')
        self.assertIn('Native gripper stop timed out', self.window.session_info.value)
        self.assertIn('/tmp/session/logs', self.window.session_info.value)
        self.assertNotIn('Connection refused', self.window.session_info.value)
        self.write_failure(phase='stopped', reason='Connection refused')
        self.window.update()
        self.window.runtime.close.assert_called_once()
        self.assertEqual(self.window.session_failure['reason'], 'Native gripper stop timed out')
        self.assertIn('ROS 与隧道已停止', self.window.status.value)
        self.window.connect()
        self.window.predict()
        self.window.open_gripper()
        self.window.arm()
        self.window.runtime.connect.assert_not_called()
        self.window.runtime.arm.assert_not_called()
        self.window.runtime.preview_inference.assert_not_called()
        self.window.runtime.open_gripper.assert_not_called()
        self.window.messagebox.askokcancel.assert_not_called()

    def test_explicit_close_still_schedules_window_destruction_after_fault(self):
        self.write_failure()
        self.window.check_session_failure()
        self.window.close()
        self.assertTrue(self.window.closing)
        self.window.root.after.assert_called_with(450, self.window._finish_close)

    def test_close_and_supervisor_signal_wait_for_video_without_delaying_control_stop(self):
        self.window.runtime.wait_recording.return_value = False
        self.window.quit_requested.set()
        self.window.update()
        self.assertTrue(self.window.closing)
        self.window.runtime.close.assert_called_once()
        self.window.root.destroy.assert_not_called()
        self.window._finish_close()
        self.window.root.after.assert_called_with(100, self.window._finish_close)
        self.window.root.destroy.assert_not_called()
        self.window.runtime.wait_recording.return_value = True
        self.window._finish_close()
        self.window.root.destroy.assert_called_once()

    def test_video_saved_path_and_drop_count_are_visible_and_saving_blocks_rearm(self):
        self.window.runtime.snapshot.return_value = dict(
            connected=True, armed=False, arming=False, manual_opening=False, metadata={'mock': False},
            state=dict(ready=True, armed=False),
            recording=dict(status='saving', directory='/tmp/recordings/session',
                           frames={'observation.images.camera_1': 7, 'observation.images.camera_2': 8},
                           dropped={'observation.images.camera_1': 2, 'observation.images.camera_2': 0}))
        self.window.update()
        self.window.arm_button.configure.assert_called_with(state='disabled')
        self.assertIn('/tmp/recordings/session', self.window.recording_info.value)
        self.assertIn('丢帧 [2, 0]', self.window.recording_info.value)

    def test_missing_partial_or_unrelated_status_file_does_not_close_runtime(self):
        self.window.check_session_failure()
        for text in ('{', '[]', '{"version":1,"phase":"running"}'):
            self.path.write_text(text)
            self.window.check_session_failure()
        self.assertIsNone(self.window.session_failure)
        self.window.runtime.close.assert_not_called()

    def test_gripper_stop_wait_is_visible_and_clears_when_execution_resumes(self):
        snapshot = dict(connected=True, armed=True, arming=False, manual_opening=False,
                        metadata={'mock': False},
                        state=dict(ready=True, armed=True, arm_holding_for_gripper_stop=True))
        self.window.runtime.snapshot.return_value = snapshot
        self.window.update()
        self.assertIn('等待夹爪停止', self.window.status.value)
        snapshot['state']['arm_holding_for_gripper_stop'] = False
        self.window.update()
        self.assertIn('模型执行中', self.window.status.value)


if __name__ == '__main__':
    unittest.main()
