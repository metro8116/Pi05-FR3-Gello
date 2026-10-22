"""One local owner of FR3 control; kernel flock automatically expires on exit."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys


def default_lock_path():
    return Path('/tmp') / f'fr3-control-{os.getuid()}.lock'


class ControlBusy(RuntimeError):
    pass


class ControlLease:
    def __init__(self, mode, path=None):
        if mode not in ('gello', 'pi05'):
            raise ValueError('Control mode must be gello or pi05')
        self.mode = mode
        self.path = Path(path or default_lock_path())
        self.handle = None
        self._adopted = False

    def acquire(self):
        if self.handle is not None:
            return self
        handle = self.path.open('a+', encoding='utf-8')
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            owner = handle.read(2048).strip() or 'another FR3 control session'
            handle.close()
            raise ControlBusy(f'FR3 control is already owned: {owner}. Close that session first.')
        handle.seek(0)
        handle.truncate()
        json.dump({'mode': self.mode, 'pid': os.getpid()}, handle)
        handle.flush()
        self.handle = handle
        return self

    def release(self):
        if self.handle is not None:
            if not self._adopted:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    @classmethod
    def from_environment(cls, mode):
        """The launcher passes the SAME locked open file description to the GUI.

        Closing the GUI's descriptor cannot unlock a still-running supervisor;
        if the supervisor dies, the GUI retains ownership until it also exits.
        """
        value = os.environ.get("FR3_CONTROL_FD")
        if value is None:
            return cls(mode).acquire()
        fd = int(value)
        expected = default_lock_path().stat()
        actual = os.fstat(fd)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ControlBusy("Inherited control descriptor does not match the shared lock")
        handle = os.fdopen(os.dup(fd), 'r+', encoding='utf-8')
        handle.seek(0)
        owner = json.loads(handle.read(2048))
        if owner.get('mode') != mode:
            handle.close()
            raise ControlBusy("Inherited control mode does not match")
        # The descriptor must really own a lock; a plain unlocked file is invalid.
        try:
            assert_available()
        except ControlBusy:
            pass
        else:
            handle.close()
            raise ControlBusy("Inherited FR3 control lock is no longer held")
        result = cls(mode)
        result.handle = handle
        result._adopted = True
        return result

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


def assert_available(path=None):
    with ControlLease('gello', path):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description='Check the shared local FR3 control lock without touching hardware')
    ap.add_argument('--check', action='store_true', required=True)
    args = ap.parse_args(argv)
    try:
        assert_available()
    except ControlBusy as exc:
        print(str(exc), file=sys.stderr)
        return 4
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
