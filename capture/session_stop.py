"""Stop only a tagged capture session, with PID-reuse and delayed-launch guards.

Adapted from data-collection/scripts/stop_gello_teleoperation.sh. No broad name
matching, service shutdown, or process killing occurs without an exact session.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

OWNER = "pi05-fr3-gello"
REPO_ROOT = Path(__file__).resolve().parent.parent

STOP_CODE = r'''
import fcntl, os, signal, sys, time
from pathlib import Path
session = sys.argv[1]
if len(session) != 32 or any(c not in 'abcdef0123456789' for c in session):
    sys.exit('Invalid capture session')
owner = b'PI05_CAPTURE_OWNER=pi05-fr3-gello'
session_entry = ('LOCAL_ROS2_SESSION_ID=' + session).encode()
lock_path = '/tmp/pi05_capture_session_' + session + '.lock'
with os.fdopen(os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600), 'r+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    Path('/tmp/pi05_capture_cancel_' + session).touch(mode=0o600)

def processes():
    result = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            environment = (entry / 'environ').read_bytes().split(b'\0')
            if owner not in environment or session_entry not in environment:
                continue
            stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            argv = (entry / 'cmdline').read_bytes()
            result[int(entry.name)] = (stat[19], argv)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result

def send(targets, sig):
    current = processes()
    for pid, identity in targets.items():
        if current.get(pid) == identity:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

def wait_empty(seconds):
    deadline = time.monotonic() + seconds
    while True:
        remaining = processes()
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(0.1)

remaining = processes()
for sig, delay in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 1)):
    if not remaining:
        break
    send(remaining, sig)
    remaining = wait_empty(delay)
if remaining:
    sys.exit('Could not confirm capture session stopped: ' + repr(sorted(remaining)))
print('Capture session stopped: ' + session)
'''


def load_state(path):
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("owner") != OWNER:
        raise ValueError("Capture state is not owned by this repository")
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session")
    parser.add_argument("--container")
    args = parser.parse_args(argv)
    state_path = Path(os.environ.get("PI05_CAPTURE_SESSION_FILE", REPO_ROOT / "runs/capture/session.json"))
    try:
        state = load_state(state_path)
        session = args.session or (state or {}).get("session_id")
        if not session:
            print("No tracked capture session; no process scan or stop was attempted.")
            return 0
        if not isinstance(session, str) or not re.fullmatch(r"[a-f0-9]{32}", session):
            raise ValueError("Invalid capture session identifier")
        container = args.container or (state or {}).get("container") or os.environ.get("PI05_CONTAINER", "pi05-fr3-gello-ros2-1")
        if not isinstance(container, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", container):
            raise ValueError("Invalid container identifier")
        env = dict(os.environ, DOCKER_HOST=os.environ.get("LOCAL_ROS2_DOCKER_HOST", "unix:///var/run/docker.sock"))
        inspected = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", container], env=env,
                                   capture_output=True, text=True, check=False, timeout=10)
        if inspected.returncode != 0:
            raise RuntimeError("Cannot inspect the recorded container; stop state retained: " + inspected.stderr.strip())
        if inspected.stdout.strip() == "true":
            result = subprocess.run(["docker", "exec", "-i", container, "/usr/bin/python3", "-", session],
                                    env=env, input=STOP_CODE, text=True, check=False, timeout=15)
            if result.returncode:
                return result.returncode
        else:
            print("Recorded capture container is not running.")
        # A late stop must not erase a newer session's state.
        current = load_state(state_path)
        if current and current.get("session_id") == session and current.get("container") == container:
            state_path.unlink()
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print("Capture stop failed: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
