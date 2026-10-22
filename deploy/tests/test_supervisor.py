"""Ownership/startup/stop regressions using temporary locks and mocked processes.

No Docker command, SSH process, ROS node, camera or robot is started by tests.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from itertools import count
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import container_runner
import control_ownership
import local_session
import prepare_local
from config import load_config


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lock = Path(self.tmp.name) / "control.lock"
        self.path_patch = patch.object(control_ownership, "default_lock_path", return_value=self.lock)
        self.path_patch.start()

    def tearDown(self):
        self.path_patch.stop()
        self.tmp.cleanup()

    def test_gello_and_pi05_cannot_own_the_robot_simultaneously(self):
        with control_ownership.ControlLease("pi05"):
            with self.assertRaises(control_ownership.ControlBusy):
                control_ownership.ControlLease("gello").acquire()
        with control_ownership.ControlLease("gello"):
            with self.assertRaises(control_ownership.ControlBusy):
                control_ownership.ControlLease("pi05").acquire()

    def test_adopted_gui_descriptor_closing_preserves_supervisor_ownership(self):
        with control_ownership.ControlLease("pi05") as parent:
            with patch.dict(os.environ, {"FR3_CONTROL_FD": str(parent.handle.fileno())}):
                gui = control_ownership.ControlLease.from_environment("pi05")
            gui.release()
            with self.assertRaises(control_ownership.ControlBusy):
                control_ownership.assert_available()
        control_ownership.assert_available()

    def test_adopted_gui_descriptor_preserves_lock_if_parent_crashes(self):
        parent = control_ownership.ControlLease("pi05").acquire()
        with patch.dict(os.environ, {"FR3_CONTROL_FD": str(parent.handle.fileno())}):
            gui = control_ownership.ControlLease.from_environment("pi05")
        # A process crash closes its fd, rather than explicitly LOCK_UN-ing the
        # shared open file description still held by its surviving child.
        parent.handle.close()
        parent.handle = None
        with self.assertRaises(control_ownership.ControlBusy):
            control_ownership.assert_available()
        gui.release()
        control_ownership.assert_available()

    def test_inherited_mode_mismatch_fails(self):
        with control_ownership.ControlLease("gello") as parent:
            with patch.dict(os.environ, {"FR3_CONTROL_FD": str(parent.handle.fileno())}):
                with self.assertRaises(control_ownership.ControlBusy):
                    control_ownership.ControlLease.from_environment("pi05")


class SessionStartupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = load_config()
        self.lease = MagicMock()
        self.lease_class = MagicMock(return_value=MagicMock(acquire=MagicMock(return_value=self.lease)))
        self.patches = [patch.object(local_session, "ROOT", self.root),
                        patch.object(local_session, "RUNTIME_DIR", self.root / "runtime"),
                        patch.object(local_session, "FAULT_LOCK", self.root / "fault.lock"),
                        patch.object(local_session, "ControlLease", self.lease_class),
                        patch.object(local_session.signal, "signal"),
                        patch.object(local_session, "inspect_host_python", return_value="fake-python"),
                        patch.object(local_session, "check_overlay", return_value={}),
                        patch.object(local_session, "assert_no_legacy_control"),
                        patch.object(local_session, "wait_metadata", return_value={"mock": True}),
                        patch.object(local_session, "bundle_for_container"),
                        patch.object(local_session.subprocess, "Popen")]
        self.active = [item.start() for item in self.patches]

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def test_network_only_never_acquires_robot_or_launches_hardware(self):
        local_session.start(self.config, self.root / "config.yaml", no_tunnel=True, network_only=True)
        self.lease_class.assert_not_called()
        local_session.inspect_host_python.assert_not_called()
        local_session.bundle_for_container.assert_not_called()
        local_session.subprocess.Popen.assert_not_called()

    def test_mock_metadata_blocks_before_container_or_gui_start(self):
        with self.assertRaisesRegex(RuntimeError, "Mock"):
            local_session.start(self.config, self.root / "config.yaml", no_tunnel=True)
        local_session.bundle_for_container.assert_not_called()
        local_session.subprocess.Popen.assert_not_called()
        self.lease.release.assert_called_once()

    def test_hardware_fault_lock_blocks_before_ssh_or_container_start(self):
        (self.root / "fault.lock").touch()
        with self.assertRaisesRegex(RuntimeError, "fault lock"):
            local_session.start(self.config, self.root / "config.yaml")
        local_session.wait_metadata.assert_not_called()
        local_session.subprocess.Popen.assert_not_called()
        self.lease.release.assert_called_once()

    def test_unprepared_overlay_blocks_all_control_startup(self):
        local_session.check_overlay.side_effect = RuntimeError("overlay not prepared")
        with self.assertRaisesRegex(RuntimeError, "overlay not prepared"):
            local_session.start(self.config, self.root / "config.yaml")
        local_session.wait_metadata.assert_not_called()
        local_session.bundle_for_container.assert_not_called()
        local_session.subprocess.Popen.assert_not_called()
        self.lease.release.assert_called_once()

    def test_startup_failure_retains_journal_and_lease_until_ros_stop_confirmed(self):
        local_session.wait_metadata.return_value = {"mock": False}
        local_session.bundle_for_container.return_value = "/tmp/pi05-deploy-test-session"
        manifest = self.root / "runtime/session.json"
        process = types.SimpleNamespace(pid=12345, poll=lambda: 1, wait=MagicMock())

        def launch(*args, **kwargs):
            # This journal must exist before Docker can start ROS, rather than
            # appearing only after the GUI successfully opens.
            self.assertTrue(manifest.is_file())
            self.assertEqual(json.loads(manifest.read_text())["bundle"], "/tmp/pi05-deploy-test-session")
            return process

        local_session.subprocess.Popen.side_effect = launch
        checks = []

        def stop_container(bundle):
            self.lease.release.assert_not_called()
            checks.append(json.loads(manifest.read_text())["state"])
            if len(checks) == 1:
                raise RuntimeError("first shutdown cannot be confirmed")

        def retry_pause(seconds):
            self.assertEqual(json.loads(manifest.read_text())["state"], "stop_unconfirmed")
            self.lease.release.assert_not_called()

        with patch.object(local_session, "process_start", return_value="start"), \
                patch.object(local_session, "refresh_child", return_value=[]), \
                patch.object(local_session, "stop_bridge"), \
                patch.object(local_session, "stop_container_bundle", side_effect=stop_container), \
                patch.object(local_session, "stop_child"), \
                patch.object(local_session.time, "sleep", side_effect=retry_pause):
            with self.assertRaisesRegex(RuntimeError, "ROS launch exited"):
                local_session.start(self.config, self.root / "config.yaml", no_tunnel=True)
        self.assertEqual(checks, ["stopping", "stopping"])
        self.lease.release.assert_called_once()
        self.assertFalse(manifest.exists())


class RecordedProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "ros_process.json").write_text(json.dumps({"pid": 12345, "start": "old"}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_reused_pid_does_not_signal_unrelated_process(self):
        with patch.object(container_runner, "process_start", return_value="new"), \
                patch.object(container_runner, "process_info", return_value=None), \
                patch.object(container_runner.os, "kill") as kill:
            container_runner.stop(self.root)
        kill.assert_not_called()

    def test_recorded_pid_must_still_have_exact_session_bundle_in_argv(self):
        with patch.object(container_runner, "process_start", return_value="old"), \
                patch.object(Path, "read_bytes", return_value=b"python3\0unrelated.py\0"), \
                patch.object(container_runner.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "unrelated"):
                container_runner.stop(self.root)
        kill.assert_not_called()

    def test_dead_launch_leader_does_not_skip_living_recorded_descendants(self):
        first = {"pid": 23456, "start": "first", "ppid": 1, "pgrp": 12345}
        second = {"pid": 23457, "start": "second", "ppid": 1, "pgrp": 12345}
        members = {item["pid"]: item for item in (first, second)}
        record = {"pid": 12345, "start": "old", "members": [first, second]}
        (self.root / "ros_process.json").write_text(json.dumps(record))

        def send_signal(pid, sig):
            if (pid == first["pid"] and sig == signal.SIGTERM) or sig == signal.SIGKILL:
                members.pop(pid, None)

        with patch.object(container_runner, "process_info", side_effect=lambda pid: members.get(int(pid))), \
                patch.object(container_runner.os, "kill", side_effect=send_signal) as kill, \
                patch.object(container_runner.time, "monotonic", side_effect=count()), \
                patch.object(container_runner.time, "sleep"):
            container_runner.stop(self.root)
        sent = [(call.args[0], call.args[1]) for call in kill.call_args_list]
        self.assertIn((first["pid"], signal.SIGTERM), sent)
        self.assertIn((second["pid"], signal.SIGKILL), sent)
        self.assertNotIn(12345, [pid for pid, sig in sent])
        self.assertFalse(members)
        self.assertEqual(json.loads((self.root / "ros_process.json").read_text())["state"], "stopped")

    def test_reused_member_pid_is_never_signaled(self):
        record = {"pid": 12345, "start": "old", "members": [{"pid": 23456, "start": "earlier"}]}
        with patch.object(container_runner, "process_info", return_value={"pid": 23456, "start": "reused"}), \
                patch.object(container_runner.os, "kill") as kill:
            container_runner.terminate_members(self.root, record)
        kill.assert_not_called()

    def test_live_scan_finds_adopted_orphans_and_nested_children_only(self):
        graph = {10: {"pid": 10, "start": "supervisor", "ppid": 1, "pgrp": 10},
                 11: {"pid": 11, "start": "launch", "ppid": 10, "pgrp": 10},
                 12: {"pid": 12, "start": "nested", "ppid": 11, "pgrp": 10},
                 13: {"pid": 13, "start": "adopted", "ppid": 10, "pgrp": 10},
                 14: {"pid": 14, "start": "reused", "ppid": 1, "pgrp": 14},
                 20: {"pid": 20, "start": "unrelated", "ppid": 1, "pgrp": 20}}
        record = {"start": "supervisor", "members": [{"pid": 14, "start": "earlier"}]}
        with patch.object(container_runner, "process_info", side_effect=lambda pid: graph.get(int(pid))), \
                patch.object(Path, "glob", return_value=[Path(f"/proc/{pid}") for pid in graph]):
            found = container_runner.live_members(record, 10)
        self.assertEqual({item["pid"] for item in found}, {11, 12, 13})

    def test_stop_before_any_manifest_prevents_delayed_launch(self):
        (self.root / "ros_process.json").unlink()
        container_runner.stop(self.root)
        self.assertTrue((self.root / "stop.requested").is_file())
        library = MagicMock()
        library.prctl.return_value = 0
        with patch.object(container_runner.ctypes, "CDLL", return_value=library), \
                patch.object(container_runner.os, "getpgrp", return_value=os.getpid()), \
                patch.object(container_runner.signal, "signal"), \
                patch.object(container_runner.subprocess, "Popen") as launch:
            self.assertEqual(container_runner.run(self.root), 0)
        launch.assert_not_called()


class OverlayTests(unittest.TestCase):
    def prepared_overlay(self, root, source='new'):
        overlay = root / 'pi05_deploy_overlay'
        setup = overlay / 'install/setup.bash'
        setup.parent.mkdir(parents=True)
        setup.write_text('# Mock overlay setup; never sourced by these tests.\n')
        driver = overlay / 'install/pi05_gripper_driver/lib/pi05_gripper_driver/pi05_gripper_node'
        driver.parent.mkdir(parents=True)
        driver.write_text('# Mock isolated gripper driver; never executed.\n')
        driver.chmod(0o700)
        stamp = overlay / 'controller_build.json'
        stamp.write_text(json.dumps({'source_sha256': source, 'tests_passed': True,
                                     'gripper_driver_tests_passed': True}))
        return stamp, setup, driver

    def test_source_changes_or_missing_setup_invalidate_success_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stamp, setup, _ = self.prepared_overlay(root, source='old')
            with patch.object(prepare_local, "HOST_WS", root), \
                    patch.object(prepare_local, "source_digest", return_value="new"), \
                    patch.object(prepare_local.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    prepare_local.check_overlay()
                record = json.loads(stamp.read_text())
                record['source_sha256'] = 'new'
                stamp.write_text(json.dumps(record))
                self.assertEqual(prepare_local.check_overlay(), record)
                setup.unlink()
                with self.assertRaisesRegex(RuntimeError, "setup.bash"):
                    prepare_local.check_overlay()
                run.assert_not_called()

    def test_missing_isolated_gripper_driver_invalidates_success_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stamp, _, driver = self.prepared_overlay(root)
            with patch.object(prepare_local, 'HOST_WS', root), \
                    patch.object(prepare_local, 'source_digest', return_value='new'), \
                    patch.object(prepare_local.subprocess, 'run') as run:
                self.assertEqual(prepare_local.check_overlay(), json.loads(stamp.read_text()))
                driver.unlink()
                with self.assertRaisesRegex(RuntimeError, 'Isolated gripper driver is missing'):
                    prepare_local.check_overlay()
                run.assert_not_called()

    def test_missing_or_failed_gripper_driver_test_result_rejects_prepared_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stamp, _, _ = self.prepared_overlay(root)
            valid = json.loads(stamp.read_text())
            with patch.object(prepare_local, 'HOST_WS', root), \
                    patch.object(prepare_local, 'source_digest', return_value='new'), \
                    patch.object(prepare_local.subprocess, 'run') as run:
                self.assertEqual(prepare_local.check_overlay(), valid)
                for value in (None, False):
                    with self.subTest(gripper_driver_tests_passed=value):
                        record = dict(valid)
                        if value is None:
                            record.pop('gripper_driver_tests_passed')
                        else:
                            record['gripper_driver_tests_passed'] = value
                        stamp.write_text(json.dumps(record))
                        with self.assertRaisesRegex(RuntimeError, 'tests did not pass'):
                            prepare_local.check_overlay()
                run.assert_not_called()

    def test_failed_rebuild_invalidates_previous_success_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stamp, _, _ = self.prepared_overlay(root, source='old')
            results = [types.SimpleNamespace(stdout="true\n"), subprocess.CalledProcessError(1, "build")]
            with patch.object(prepare_local, "HOST_WS", root), \
                    patch.object(prepare_local.subprocess, "run", side_effect=results):
                with self.assertRaises(subprocess.CalledProcessError):
                    prepare_local.prepare(build=True)
            self.assertFalse(stamp.exists())

    def test_stopped_container_is_rejected_without_build_or_hardware(self):
        with patch.object(prepare_local.subprocess, "run", return_value=types.SimpleNamespace(stdout="false\n")) as run:
            with self.assertRaisesRegex(RuntimeError, "not running"):
                prepare_local.prepare(build=True)
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
