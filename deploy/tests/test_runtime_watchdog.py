"""Host supervisor RPC watchdog regressions with mocked deployment services.

No Docker, SSH, ROS, camera, or robot process is started. A controlled monotonic
clock exercises the complete start/finally path without waiting in real time.
"""
from __future__ import annotations

import json
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_session
from config import load_config


class RuntimeWatchdogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.now = 100.
        self.gui_exit_at = 110.
        self.ros_exit_at = self.tunnel_exit_at = self.stop_signal_at = None
        self.gui = None
        self.ros_log = None
        self.retained_samples = []
        self.handlers = {}
        self.probes = []
        self.processes = []
        self.stop_times = []
        self.failed_stop_attempts = 0
        self.health_response = lambda: self.state(ready=False, fault='Stopped by user')
        self.config = load_config()
        self.lease = MagicMock()
        self.lease.handle.fileno.return_value = 42
        lease_class = MagicMock()
        lease_class.return_value.acquire.return_value = self.lease
        fake_time = types.SimpleNamespace(monotonic=lambda: self.now, sleep=self.advance)
        patches = [
            patch.object(local_session, 'ROOT', self.root),
            patch.object(local_session, 'RUNTIME_DIR', self.root / 'runtime'),
            patch.object(local_session, 'FAULT_LOCK', self.root / 'fault.lock'),
            patch.object(local_session, 'ControlLease', lease_class),
            patch.object(local_session.signal, 'signal', side_effect=self.register_signal),
            patch.object(local_session, 'inspect_host_python', return_value='fake-python'),
            patch.object(local_session, 'check_overlay'),
            patch.object(local_session, 'assert_no_legacy_control'),
            patch.object(local_session, 'wait_metadata', return_value={'mock': False}),
            patch.object(local_session, 'bundle_for_container', return_value='/tmp/pi05-deploy-test'),
            patch.object(local_session, 'docker_base', return_value=['fake-docker']),
            patch.object(local_session, 'container_name', return_value='fake-container'),
            patch.object(local_session, 'ssh_command', return_value=['fake-ssh']),
            patch.object(local_session, 'process_start', return_value='start'),
            patch.object(local_session, 'refresh_child', return_value=[]),
            patch.object(local_session, 'packet_call', side_effect=self.packet_call),
            patch.object(local_session, 'stop_bridge'),
            patch.object(local_session, 'stop_container_bundle', side_effect=self.stop_bundle),
            patch.object(local_session, 'stop_child'),
            patch.object(local_session.subprocess, 'Popen', side_effect=self.launch),
            patch.object(local_session, 'time', fake_time),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    @staticmethod
    def state(**changes):
        result = dict(type='state', version=1, ready=True, armed=False, fault=None)
        result.update(changes)
        return result

    def advance(self, seconds):
        self.now += seconds
        manifest = self.root / 'runtime/session.json'
        if manifest.is_file():
            record = json.loads(manifest.read_text())
            if record['state'] == 'faulted':
                self.lease.release.assert_not_called()
                self.assertNotIn(self.gui, [call.args[0] for call in local_session.stop_child.call_args_list])
                status = json.loads((Path(record['logs']) / 'session_status.json').read_text())
                self.assertEqual(status['phase'], 'stopped')
                self.retained_samples.append(status)
                if self.stop_signal_at is not None and self.now >= self.stop_signal_at:
                    self.handlers[signal.SIGTERM](signal.SIGTERM, None)

    def register_signal(self, number, handler):
        previous = self.handlers.get(number, signal.SIG_DFL)
        self.handlers[number] = handler
        return previous

    def launch(self, argv, **kwargs):
        role = 'tunnel' if argv[0] == 'fake-ssh' else ('gui' if argv[0] == 'fake-python' else 'ros')
        process = types.SimpleNamespace(pid=12345 + len(self.processes), role=role)
        def poll():
            deadline = dict(gui=self.gui_exit_at, ros=self.ros_exit_at, tunnel=self.tunnel_exit_at)[role]
            return 0 if deadline is not None and self.now >= deadline else None
        process.poll = poll
        if role == 'gui':
            self.gui = process
            if self.ros_log is not None:
                (Path(argv[argv.index('--session-status') + 1]).parent / 'ros.log').write_text(self.ros_log)
        self.processes.append(process)
        return process

    def packet_call(self, host, port, packet, timeout):
        self.assertEqual(packet, {'version': 1, 'type': 'state'})
        if self.gui is None:
            # Startup readiness succeeds before the preview GUI is launched.
            return self.state()
        self.probes.append((self.now, timeout))
        return self.health_response()

    def stop_bundle(self, bundle):
        self.assertEqual(bundle, '/tmp/pi05-deploy-test')
        self.lease.release.assert_not_called()
        record = json.loads((self.root / 'runtime/session.json').read_text())
        self.assertEqual(record['state'], 'stopping')
        self.stop_times.append(self.now)
        if self.failed_stop_attempts:
            self.failed_stop_attempts -= 1
            raise RuntimeError('ROS stop is not yet confirmed')

    def run_session(self, no_tunnel=True):
        local_session.start(self.config, self.root / 'config.yaml', no_tunnel=no_tunnel)

    def assert_probe_limits(self):
        self.assertTrue(self.probes)
        self.assertTrue(all(timeout == .3 for _, timeout in self.probes))
        for first, second in zip(self.probes, self.probes[1:]):
            self.assertGreaterEqual(second[0] - first[0], .5 - 1e-8)

    def assert_cleaned(self):
        self.lease.release.assert_called_once()
        self.assertFalse((self.root / 'runtime/session.json').exists())
        self.assertEqual(local_session.stop_child.call_count, len(self.processes))

    def test_unresponsive_bridge_stops_live_processes_and_retains_lease_until_confirmed(self):
        def timeout():
            self.assertTrue(all(process.poll() is None for process in self.processes))
            self.advance(.3)
            raise TimeoutError('bridge handler is blocked')

        self.health_response = timeout
        self.failed_stop_attempts = 1
        with self.assertRaisesRegex(RuntimeError, 'Bridge state RPC unresponsive'):
            self.run_session()
        self.assertEqual(len(self.processes), 2)
        self.assert_probe_limits()
        self.assertEqual(len(self.stop_times), 2)
        self.assertLessEqual(self.stop_times[0] - 100., 2.4)
        self.assertGreaterEqual(self.stop_times[1] - self.stop_times[0], 2.)
        self.assertTrue(self.retained_samples)
        self.assertIn('bridge handler is blocked', self.retained_samples[0]['reason'])
        self.assert_cleaned()

    def test_one_timeout_then_valid_state_recovers_without_ending_preview(self):
        self.gui_exit_at = 103.

        def once():
            self.health_response = lambda: self.state(ready=False, fault='Stopped by user')
            self.advance(.3)
            raise TimeoutError('brief delay')

        self.health_response = once
        self.run_session()
        self.assert_probe_limits()
        self.assertGreaterEqual(len(self.probes), 4)
        self.assertGreaterEqual(self.stop_times[0], self.gui_exit_at)
        self.assertEqual(self.retained_samples, [])
        self.assert_cleaned()

    def test_responsive_fault_and_not_ready_are_not_transport_failures(self):
        self.gui_exit_at = 104.
        self.run_session()
        self.assert_probe_limits()
        self.assertGreaterEqual(len(self.probes), 6)
        self.assertGreaterEqual(self.stop_times[0], self.gui_exit_at)
        self.assert_cleaned()

    def test_malformed_state_replies_do_not_keep_an_unusable_bridge_alive(self):
        self.health_response = lambda: dict(type='error', version=1, error='invalid state')
        with self.assertRaisesRegex(RuntimeError, 'Bridge did not return a version 1 state'):
            self.run_session()
        self.assert_probe_limits()
        self.assertLessEqual(self.stop_times[0] - 100., 2.2)
        self.assert_cleaned()

    def test_ros_exit_keeps_gui_and_first_fatal_until_user_closes_window(self):
        self.ros_exit_at = 100.6
        self.ros_log = ('[python3-6] [ERROR] [1792639475.4] [pi05_fr3_bridge]: stale gripper state\n'
                        '[python3-6] [FATAL] [1792639476.0] [pi05_fr3_bridge]: Native gripper stop timed out\n'
                        '[python3-6] [ERROR] [1792639477.0] [pi05_fr3_bridge]: Connection refused\n')
        with self.assertRaisesRegex(RuntimeError, 'ROS launch exited unexpectedly'):
            self.run_session()
        self.assertTrue(self.retained_samples)
        first = self.retained_samples[0]
        self.assertEqual(first['reason'], 'Native gripper stop timed out')
        self.assertEqual(first['source'], 'ros.log')
        self.assertIn('[FATAL]', first['source_line'])
        self.assertLess(self.stop_times[0], self.gui_exit_at)
        self.assertGreaterEqual(self.now, self.gui_exit_at)
        self.assert_cleaned()

    def test_explicit_stop_signal_closes_retained_fault_window(self):
        self.ros_exit_at = 100.6
        self.gui_exit_at = 200.
        self.stop_signal_at = 104.
        with self.assertRaisesRegex(RuntimeError, 'ROS launch exited unexpectedly'):
            self.run_session()
        self.assertTrue(self.retained_samples)
        self.assertIsNone(self.gui.poll(), 'GUI should be explicitly stopped, not naturally exited')
        self.assertLess(self.now, 105.)
        self.assert_cleaned()

    def test_tunnel_failure_stops_other_services_but_retains_gui(self):
        self.tunnel_exit_at = 100.6
        with self.assertRaisesRegex(RuntimeError, 'SSH tunnel exited unexpectedly'):
            self.run_session(no_tunnel=False)
        self.assertTrue(self.retained_samples)
        self.assertIn('SSH tunnel exited unexpectedly', self.retained_samples[0]['reason'])
        self.assertEqual([call.args[0].role for call in local_session.stop_child.call_args_list],
                         ['ros', 'tunnel', 'gui'])
        self.assert_cleaned()

    def test_terminal_bridge_error_enters_fault_display_before_process_exit(self):
        self.health_response = lambda: self.state(exit_error='Native gripper stop timed out')
        with self.assertRaisesRegex(RuntimeError, 'Native gripper stop timed out'):
            self.run_session()
        self.assertTrue(self.retained_samples)
        self.assertEqual(self.retained_samples[0]['reason'], 'Native gripper stop timed out')
        self.assertLess(self.stop_times[0] - 100., 1.)
        self.assert_cleaned()


class DeploymentFailureReasonTests(unittest.TestCase):
    def test_real_driver_error_is_not_replaced_by_cleanup_or_transport_noise(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            (logs / 'ros.log').write_text(
                '[driver-2] [ERROR] [1792640504.1] [pi05_gripper]: actual driver failure\n'
                '[python3-6] [ERROR] [1792640504.2] [pi05_fr3_bridge]: Bridge process exiting\n'
                '[python3-6] [ERROR] [1792640504.3] [pi05_fr3_bridge]: [Errno 111] Connection refused\n')
            result = local_session.deployment_failure(logs, 'ROS launch exited unexpectedly')
            self.assertEqual(result['reason'], 'actual driver failure')
            self.assertEqual(result['source'], 'ros.log')
            self.assertIn('[driver-2]', result['source_line'])

    def test_only_cleanup_noise_uses_the_meaningful_supervisor_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            logs = Path(temporary)
            (logs / 'ros.log').write_text(
                '[python3-6] [ERROR] [1792640504.2] [pi05_fr3_bridge]: Bridge process exiting\n'
                '[python3-6] [ERROR] [1792640504.3] [pi05_fr3_bridge]: Connection closed\n')
            result = local_session.deployment_failure(logs, 'SSH tunnel exited unexpectedly')
            self.assertEqual(result['reason'], 'SSH tunnel exited unexpectedly')
            self.assertEqual(result['source'], 'supervisor')


class RetainedSessionStopTests(unittest.TestCase):
    def test_stop_script_signals_a_retained_fault_session_and_removes_its_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            manifest = runtime / 'session.json'
            manifest.write_text(json.dumps(dict(pid=12345, start='owned', state='faulted',
                                                config=load_config(), bundle='/tmp/pi05-deploy-test')))
            exited = False

            def identity(pid):
                self.assertEqual(pid, 12345)
                return None if exited else 'owned'

            def signal_stop(pid, number):
                nonlocal exited
                self.assertEqual((pid, number), (12345, signal.SIGTERM))
                exited = True

            with patch.object(local_session, 'RUNTIME_DIR', runtime), \
                    patch.object(local_session, 'process_start', side_effect=identity), \
                    patch.object(Path, 'read_bytes', return_value=b'python\0local_session.py\0'), \
                    patch.object(local_session, 'stop_bridge') as stop_bridge, \
                    patch.object(local_session, 'stop_container_bundle') as stop_bundle, \
                    patch.object(local_session.os, 'kill', side_effect=signal_stop) as kill:
                local_session.stop()
            kill.assert_called_once_with(12345, signal.SIGTERM)
            stop_bridge.assert_called_once()
            stop_bundle.assert_not_called()
            self.assertFalse(manifest.exists())


if __name__ == '__main__':
    unittest.main()
