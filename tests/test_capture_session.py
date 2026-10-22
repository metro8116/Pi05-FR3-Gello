"""CPU-only capture migration checks. All Docker, camera and signal actions are mocked."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("capture_session_stop", REPO / "capture/session_stop.py")
stop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stop)
SESSION = "a" * 32


class CaptureSessionTests(unittest.TestCase):
    def test_missing_state_does_not_invoke_docker(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"PI05_CAPTURE_SESSION_FILE": temp + "/missing.json"}), \
                patch.object(stop.subprocess, "run", side_effect=AssertionError("Docker must not run")):
            self.assertEqual(stop.main([]), 0)

    def test_invalid_or_foreign_state_is_not_stopped(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp) / "state.json"
            for payload in ({"owner": "other", "session_id": SESSION},
                            {"owner": stop.OWNER, "session_id": "invalid"}):
                state.write_text(json.dumps(payload))
                with patch.dict(os.environ, {"PI05_CAPTURE_SESSION_FILE": str(state)}), \
                        patch.object(stop.subprocess, "run", side_effect=AssertionError("Docker must not run")):
                    self.assertEqual(stop.main([]), 2)
                self.assertTrue(state.exists())

    def test_stop_targets_recorded_container_and_session(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp) / "state.json"
            state.write_text(json.dumps({"owner": stop.OWNER, "session_id": SESSION, "container": "new-capture-container"}))
            calls = []
            def run(argv, **kwargs):
                calls.append((argv, kwargs))
                return subprocess.CompletedProcess(argv, 0, "true\n" if argv[1] == "inspect" else "", "")
            with patch.dict(os.environ, {"PI05_CAPTURE_SESSION_FILE": str(state)}), patch.object(stop.subprocess, "run", run):
                self.assertEqual(stop.main([]), 0)
            self.assertEqual(calls[0][0][-1], "new-capture-container")
            self.assertEqual(calls[1][0][-1], SESSION)
            self.assertEqual(calls[1][1]["input"], stop.STOP_CODE)
            self.assertFalse(state.exists())

    def test_failed_stop_preserves_state(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp) / "state.json"
            state.write_text(json.dumps({"owner": stop.OWNER, "session_id": SESSION, "container": "capture"}))
            responses = [subprocess.CompletedProcess([], 0, "true\n", ""), subprocess.CompletedProcess([], 2)]
            with patch.dict(os.environ, {"PI05_CAPTURE_SESSION_FILE": str(state)}), patch.object(stop.subprocess, "run", side_effect=responses):
                self.assertEqual(stop.main([]), 2)
            self.assertTrue(state.exists())

    def test_remote_filter_and_pid_reuse_without_real_signals(self):
        with tempfile.TemporaryDirectory() as temp:
            proc = Path(temp)
            def process(pid, owner, session, start="100"):
                directory = proc / str(pid)
                directory.mkdir(exist_ok=True)
                (directory / "environ").write_bytes((f"PI05_CAPTURE_OWNER={owner}\0LOCAL_ROS2_SESSION_ID={session}\0").encode())
                fields = ["S"] + ["0"] * 19
                fields[19] = start
                (directory / "stat").write_text(f"{pid} (test) " + " ".join(fields))
                (directory / "cmdline").write_bytes(b"ros2_control_node\0")
            process(101, stop.OWNER, SESSION)
            process(102, stop.OWNER, "b" * 32)
            process(103, "legacy", SESSION)
            sent = []
            namespace = {"Path": lambda p: proc if p == "/proc" else Path(p),
                         "os": types.SimpleNamespace(getpid=lambda: 999, kill=lambda *args: sent.append(args)),
                         "owner": b"PI05_CAPTURE_OWNER=pi05-fr3-gello",
                         "session_entry": ("LOCAL_ROS2_SESSION_ID=" + SESSION).encode()}
            functions = [node for node in ast.parse(stop.STOP_CODE).body if isinstance(node, ast.FunctionDef)
                         and node.name in ("processes", "send")]
            exec(compile(ast.Module(body=functions, type_ignores=[]), "<isolated-stop-functions>", "exec"), namespace)
            targets = namespace["processes"]()
            self.assertEqual(set(targets), {101})
            namespace["send"](targets, signal.SIGINT)
            self.assertEqual(sent, [(101, signal.SIGINT)])
            process(101, stop.OWNER, SESSION, start="200")
            namespace["send"](targets, signal.SIGKILL)
            self.assertEqual(sent, [(101, signal.SIGINT)])

    def test_start_exports_new_paths_and_preview_without_running_gui(self):
        with tempfile.TemporaryDirectory() as temp:
            fake = Path(temp) / "fake_python"
            output = Path(temp) / "export.json"
            fake.write_text("#!" + sys.executable + "\nimport os,json,sys\nkeys=['TAKE_DATA_OUTPUT_DIR','TAKE_DATA_AUTOSTART_ROBOT','LOCAL_ROS2_WS','LOCAL_ROS2_GRIPPER_CFG','PYTHONPATH','PI05_CONTAINER']\njson.dump({'argv':sys.argv[1:],'env':{k:os.environ.get(k) for k in keys}},open(os.environ['CAPTURE_TEST_EXPORT'],'w'))\n")
            fake.chmod(0o755)
            env = dict(os.environ, CAPTURE_PYTHON=str(fake), CAPTURE_TEST_EXPORT=str(output))
            result = subprocess.run(["bash", str(REPO / "capture/start_capture.sh"), "--output-dir", temp + "/new raw", "--no-camera"], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            exported = json.loads(output.read_text())
            self.assertEqual(exported["argv"], [str(REPO / "capture/video_action_recorder.py")])
            self.assertEqual(exported["env"]["TAKE_DATA_OUTPUT_DIR"], temp + "/new raw")
            self.assertEqual(exported["env"]["TAKE_DATA_AUTOSTART_ROBOT"], "0")
            self.assertEqual(exported["env"]["LOCAL_ROS2_WS"], str(REPO / "ros2_ws"))
            self.assertEqual(exported["env"]["LOCAL_ROS2_GRIPPER_CFG"], "example_fr3_config_franka_hand_move.yaml")
            self.assertTrue(exported["env"]["PYTHONPATH"].startswith(str(REPO / "common") + ":" + str(REPO / "src")))
            self.assertFalse((Path(temp) / "new raw").exists())

    def test_ros_sources_have_no_old_absolute_paths_or_global_kill(self):
        for folder in (REPO / "capture", REPO / "ros2_ws/src"):
            for path in folder.rglob("*"):
                if path.suffix not in (".py", ".sh", ".yaml", ".yml"):
                    continue
                source = path.read_text()
                self.assertNotIn("/home/zezhi", source, str(path))
                self.assertNotIn("pkill", source, str(path))
        for path in (REPO / "ros2_ws/.clang-format", REPO / "ros2_ws/.clang-tidy"):
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
