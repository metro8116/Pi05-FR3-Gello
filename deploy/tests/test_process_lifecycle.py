"""Actual Linux process cleanup with a fake ros2 executable and harmless sleepers.

No ROS packages are imported, no network connection is opened, and no hardware
is addressed. Only PID/start-time verified processes created by this test are
ever signaled, including in failure cleanup.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

DEPLOY_ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start": fields[19], "ppid": int(fields[1])}
    except (OSError, ValueError, IndexError):
        return None


def read_json_if_ready(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@unittest.skipUnless(sys.platform == "linux" and Path("/proc/self/stat").is_file(),
                     "Linux /proc and child-subreaper support are required")
class RealDummyProcessLifecycleTests(unittest.TestCase):
    def test_orphan_ignoring_interrupt_and_term_is_reaped_from_owned_session(self):
        with tempfile.TemporaryDirectory(prefix="pi05-deploy-lifecycle-", dir="/tmp") as temporary:
            directory = Path(temporary)
            fake_bin = directory / "fakebin"
            fake_bin.mkdir()
            child_path = directory / "harmless_child.py"
            child_path.write_text('''import json, os, signal, time
from pathlib import Path
directory = Path(__file__).resolve().parent
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pid = os.getpid()
fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
(directory / "dummy_child.json").write_text(json.dumps({"pid": pid, "start": fields[19]}))
observed = []
while True:
    parent = os.getppid()
    if parent not in observed:
        observed.append(parent)
        temp = directory / "observed_parents.tmp"
        temp.write_text(json.dumps(observed))
        temp.replace(directory / "observed_parents.json")
    time.sleep(0.01)
''')
            ros2 = fake_bin / "ros2"
            ros2.write_text(f"#!{sys.executable}\n" + '''import os, signal, subprocess, sys, time
from pathlib import Path
directory = Path(__file__).resolve().parents[1]
def leave(signum, frame):
    raise SystemExit(0)
signal.signal(signal.SIGINT, leave)
signal.signal(signal.SIGTERM, leave)
subprocess.Popen([sys.executable, str(directory / "harmless_child.py")])
while True:
    time.sleep(0.05)
''')
            ros2.chmod(0o700)
            env = os.environ.copy()
            env["PATH"] = str(fake_bin) + os.pathsep + env.get("PATH", "")
            known = {}
            supervisor = None
            with (directory / "dummy_supervisor.log").open("w+") as log:
                try:
                    supervisor = subprocess.Popen(
                        [sys.executable, str(DEPLOY_ROOT / "container_runner.py"), str(directory)],
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                    initial = process_identity(supervisor.pid)
                    if initial:
                        known[initial["pid"]] = initial["start"]
                    deadline = time.monotonic() + 8
                    child = None
                    record = None
                    while time.monotonic() < deadline:
                        record = read_json_if_ready(directory / "ros_process.json")
                        if record:
                            for member in record.get("members", []):
                                known[member["pid"]] = member["start"]
                        child = read_json_if_ready(directory / "dummy_child.json")
                        if child:
                            known[child["pid"]] = child["start"]
                        if record and child and record.get("state") == "running":
                            break
                        if supervisor.poll() is not None:
                            log.seek(0)
                            self.fail("Dummy supervisor exited before startup: " + log.read())
                        time.sleep(.02)
                    self.assertIsNotNone(child, "Harmless dummy child did not become ready")
                    self.assertIsNotNone(record, "Supervisor did not write its process journal")
                    self.assertEqual(record["pid"], supervisor.pid)
                    self.assertEqual(record["start"], process_identity(supervisor.pid)["start"])
                    # Ensure the initial parent has been observed before asking
                    # the supervisor to orphan/adopt the intentionally stubborn child.
                    deadline = time.monotonic() + 2
                    while not read_json_if_ready(directory / "observed_parents.json") and time.monotonic() < deadline:
                        time.sleep(.01)
                    completed = subprocess.run(
                        [sys.executable, str(DEPLOY_ROOT / "container_runner.py"), str(directory), "--stop"],
                        env=env, capture_output=True, text=True, timeout=26)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    supervisor.wait(timeout=4)
                    self.assertEqual(supervisor.returncode, 0)
                    self.assertIsNone(process_identity(child["pid"]), "Owned dummy child is still executing")
                    final = read_json_if_ready(directory / "ros_process.json")
                    self.assertEqual(final["state"], "stopped")
                    self.assertEqual(final["members"], [])
                    observed = read_json_if_ready(directory / "observed_parents.json")
                    self.assertIn(supervisor.pid, observed,
                                  "The orphaned dummy child was not adopted by the Linux subreaper")
                    self.assertGreaterEqual(len(observed), 2)
                finally:
                    # Collect the session's exact current journal before its
                    # directory disappears; never signal by executable name or PGID.
                    record = read_json_if_ready(directory / "ros_process.json")
                    if record:
                        for member in record.get("members", []):
                            known[member["pid"]] = member["start"]
                    child = read_json_if_ready(directory / "dummy_child.json")
                    if child:
                        known[child["pid"]] = child["start"]
                    for pid, stamp in list(known.items()):
                        if supervisor is not None and pid == supervisor.pid:
                            continue
                        current = process_identity(pid)
                        if current and current["start"] == stamp:
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    if supervisor is not None and supervisor.poll() is None:
                        current = process_identity(supervisor.pid)
                        if current and current["start"] == known.get(supervisor.pid):
                            try:
                                os.kill(supervisor.pid, signal.SIGINT)
                            except ProcessLookupError:
                                pass
                        try:
                            supervisor.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            current = process_identity(supervisor.pid)
                            if current and current["start"] == known.get(supervisor.pid):
                                os.kill(supervisor.pid, signal.SIGKILL)
                            supervisor.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
