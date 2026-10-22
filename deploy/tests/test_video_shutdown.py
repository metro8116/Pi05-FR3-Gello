"""Video save grace using owned harmless Linux children, never cameras or ROS."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_session
from config import load_config


class VideoShutdownTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'linux' and Path('/proc/self/stat').is_file(), 'requires Linux /proc')
    def test_gui_receives_term_first_and_encoder_finishes_without_being_signaled(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            encoder = directory / 'fake_encoder.py'
            encoder.write_text('''import json, os, signal, sys, time
from pathlib import Path
d = Path(sys.argv[1])
def interrupted(signum, frame):
    (d / 'encoder_terminated').touch()
    raise SystemExit(2)
signal.signal(signal.SIGTERM, interrupted)
fields = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
(d / 'encoder.json').write_text(json.dumps({'pid': os.getpid(), 'start': fields[19]}))
while not (d / 'finalize_requested').exists(): time.sleep(.01)
time.sleep(.2)
(d / 'video_saved').touch()
''')
            gui_source = directory / 'fake_gui.py'
            gui_source.write_text('''import signal, subprocess, sys, time
from pathlib import Path
d = Path(sys.argv[1])
requested = False
def close(signum, frame):
    global requested
    requested = True
signal.signal(signal.SIGTERM, close)
child = subprocess.Popen([sys.executable, str(d / 'fake_encoder.py'), str(d)])
(d / 'gui_ready').touch()
while not requested: time.sleep(.01)
(d / 'finalize_requested').touch()
raise SystemExit(child.wait(timeout=3))
''')
            gui = subprocess.Popen([sys.executable, '-B', str(gui_source), str(directory)], start_new_session=True)
            gui.fr3_start, gui.fr3_name = local_session.process_start(gui.pid), 'client'
            gui.fr3_members = {gui.pid: gui.fr3_start}
            try:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if (directory / 'encoder.json').is_file() and (directory / 'gui_ready').exists():
                        try:
                            child = json.loads((directory / 'encoder.json').read_text())
                            gui.fr3_members[child['pid']] = child['start']
                            break
                        except ValueError:
                            pass
                    time.sleep(.01)
                self.assertTrue((directory / 'encoder.json').is_file())
                local_session.stop_child(gui, timeout=2)
                self.assertEqual(gui.returncode, 0)
                self.assertTrue((directory / 'video_saved').exists())
                self.assertFalse((directory / 'encoder_terminated').exists())
                self.assertIsNone(local_session.process_start(child['pid']))
            finally:
                for pid, stamp in gui.fr3_members.items():
                    if local_session.process_start(pid) == stamp:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                gui.wait(timeout=3)

    def test_crashed_supervisor_recovery_uses_the_same_gui_save_grace(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            gui = dict(pid=23456, start='gui-start', name='client',
                       members=[dict(pid=23456, start='gui-start'), dict(pid=23457, start='encoder-start')])
            (runtime / 'session.json').write_text(json.dumps(dict(
                pid=12345, start='dead-supervisor', config=load_config(),
                bundle='/tmp/pi05-deploy-test', children=[gui])))
            with patch.object(local_session, 'RUNTIME_DIR', runtime), \
                    patch.object(local_session, 'process_start', return_value=None), \
                    patch.object(local_session, 'stop_bridge'), \
                    patch.object(local_session, 'stop_container_bundle'), \
                    patch.object(local_session, 'stop_gui_members') as stop_gui, \
                    patch.object(local_session, 'stop_host_members') as stop_members:
                local_session.stop()
            process, timeout = stop_gui.call_args.args
            self.assertEqual((process.pid, process.fr3_start, timeout), (23456, 'gui-start', 12))
            self.assertEqual(process.fr3_members[23457], 'encoder-start')
            stop_members.assert_not_called()


if __name__ == '__main__':
    unittest.main()
