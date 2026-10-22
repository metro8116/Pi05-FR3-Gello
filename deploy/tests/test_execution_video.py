"""Execution recording tests: tiny CPU arrays, fake writers and synthetic FFmpeg.

No camera, ROS, GPU, model, SSH or existing deployment process is used.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import execution_video as video

try:
    import numpy as np
except ImportError:
    np = None

CONFIG = dict(serials=["synthetic-a", "synthetic-b"], width=32, height=24, capture_fps=15)


def frame(camera=0, number=1, stamp=None):
    return dict(rgb=np.full((24, 32, 3), (number * 10) % 256, dtype=np.uint8),
                monotonic=time.monotonic() if stamp is None else stamp,
                number=number, serial=CONFIG["serials"][camera],
                source_timestamp_ms=number * 1000 / 15, source_timestamp_domain="synthetic",
                received_monotonic=time.monotonic())


class FakeWriter:
    def __init__(self, path, width, height, fps):
        self.path, self.frames, self.closed = path, [], False
        self.close_timeout = None

    def write(self, rgb, *, timeout_s):
        self.frames.append(rgb.copy())

    def close(self, *, timeout_s):
        self.closed, self.close_timeout = True, timeout_s


@unittest.skipIf(np is None, "requires existing NumPy for synthetic video frames")
class ExecutionRecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="execution-video-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "recordings"
        self.writers = []
        self.errors = []
        self.error_event = threading.Event()
        self.recorders = []

    def tearDown(self):
        for recorder in self.recorders:
            recorder.stop("test cleanup")
            self.assertTrue(recorder.wait(3), "owned fake encoder failed to finish")

    def factory(self, cls=FakeWriter):
        def create(*args):
            result = cls(*args)
            self.writers.append(result)
            return result
        return create

    def make(self, **kwargs):
        def on_error(message):
            self.errors.append(message)
            self.error_event.set()
        recorder = video.ExecutionVideoRecorder(CONFIG, output_root=self.root,
                                                on_error=on_error,
                                                writer_factory=kwargs.pop("writer_factory", self.factory()), **kwargs)
        self.recorders.append(recorder)
        return recorder

    def seeds(self):
        now = time.monotonic()
        return {key: frame(index, stamp=now) for index, key in enumerate(video.CAMERA_KEYS)}

    def test_constructor_and_stop_before_start_do_no_io(self):
        recorder = self.make()
        recorder.stop("before start")
        self.assertTrue(recorder.wait(.01))
        self.assertFalse(self.root.exists())
        self.assertEqual(recorder.snapshot()["status"], "idle")
        with self.assertRaisesRegex(video.RecordingError, "cancelled before startup"):
            recorder.start("late-enable-after-stop", self.seeds())
        self.assertFalse(self.root.exists())

    def test_start_requires_both_writers_and_real_initial_frames(self):
        recorder = self.make()
        state = recorder.start("session-one", self.seeds())
        self.assertEqual(state["status"], "recording")
        self.assertEqual(state["frames"], {key: 1 for key in video.CAMERA_KEYS})
        self.assertEqual(len(self.writers), 2)
        self.assertTrue(all(len(writer.frames) == 1 for writer in self.writers))
        self.assertFalse(recorder.worker.daemon)

    def test_missing_or_invalid_initial_frame_fails_without_files(self):
        for mutation in ("missing", "dtype", "serial", "shape"):
            with self.subTest(mutation=mutation):
                seeds = self.seeds()
                first = seeds[video.CAMERA_KEYS[0]]
                if mutation == "missing":
                    del seeds[video.CAMERA_KEYS[1]]
                elif mutation == "dtype":
                    first["rgb"] = first["rgb"].astype(np.float32)
                elif mutation == "serial":
                    first["serial"] = "wrong camera"
                else:
                    first["rgb"] = first["rgb"][:10]
                with self.assertRaises(video.RecordingError):
                    self.make().start("bad-frame", seeds)
                self.assertFalse(self.root.exists())

    def test_second_writer_failure_aborts_start_and_closes_first(self):
        good_factory = self.factory()
        def create(*args):
            if self.writers:
                raise video.RecordingError("injected encoder startup failure")
            return good_factory(*args)
        recorder = self.make(writer_factory=create)
        with self.assertRaisesRegex(video.RecordingError, "startup failure"):
            recorder.start("broken", self.seeds())
        self.assertTrue(recorder.wait(1))
        self.assertTrue(self.writers[0].closed)
        self.assertTrue(self.error_event.wait(1))
        self.assertEqual(len(self.errors), 1)

    def test_seed_listener_duplicates_are_not_recorded_twice(self):
        recorder, seeds = self.make(), self.seeds()
        recorder.start("duplicate-check", seeds)
        for key in video.CAMERA_KEYS:
            self.assertFalse(recorder.submit(key, seeds[key]))
        self.assertTrue(recorder.submit(video.CAMERA_KEYS[0], frame(0, 2)))
        recorder.stop("operator stop")
        self.assertFalse(recorder.submit(video.CAMERA_KEYS[0], frame(0, 3)))
        self.assertTrue(recorder.wait(1))
        state = recorder.snapshot()
        self.assertEqual(state["frames"], {video.CAMERA_KEYS[0]: 2, video.CAMERA_KEYS[1]: 1})
        self.assertEqual(state["duplicates"], {key: 1 for key in video.CAMERA_KEYS})
        metadata = json.loads((Path(state["directory"]) / "metadata.json").read_text())
        rows = [json.loads(line) for line in (Path(state["directory"]) / "frames.jsonl").read_text().splitlines()]
        self.assertEqual(metadata["stop_reason"], "operator stop")
        self.assertEqual(metadata["camera_serials"], dict(zip(video.CAMERA_KEYS, CONFIG["serials"])))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]["source_timestamp_domain"], "synthetic")
        self.assertNotIn("rgb", rows[-1])

    def test_submit_and_stop_do_not_wait_for_encoder_io_and_queue_drops_are_counted(self):
        entered, release = threading.Event(), threading.Event()
        class BlockingWriter(FakeWriter):
            block = False
            def write(self, rgb, *, timeout_s):
                if self.block:
                    entered.set()
                    if not release.wait(timeout_s):
                        raise video.RecordingError("injected frame deadline")
                super().write(rgb, timeout_s=timeout_s)
        recorder = self.make(queue_size=2, writer_factory=self.factory(BlockingWriter))
        recorder.start("queue", self.seeds())
        self.writers[0].block = True
        recorder.submit(video.CAMERA_KEYS[0], frame(0, 2))
        self.assertTrue(entered.wait(1))
        start = time.monotonic()
        for number in range(3, 20):
            recorder.submit(video.CAMERA_KEYS[0], frame(0, number))
        recorder.stop("stop without waiting")
        self.assertLess(time.monotonic() - start, .1)
        self.assertFalse(recorder.wait(.01))
        self.assertGreater(recorder.snapshot()["dropped"][video.CAMERA_KEYS[0]], 0)
        release.set()
        self.assertTrue(recorder.wait(1))

    def test_stop_during_start_never_reports_recording_ready(self):
        entered, release = threading.Event(), threading.Event()
        class StartingWriter(FakeWriter):
            def write(self, rgb, *, timeout_s):
                entered.set()
                release.wait(timeout_s)
                super().write(rgb, timeout_s=timeout_s)
        recorder = self.make(writer_factory=self.factory(StartingWriter))
        outcomes = []
        def start():
            try:
                recorder.start("cancelled-start", self.seeds())
                outcomes.append("unexpected success")
            except video.RecordingError:
                outcomes.append("cancelled")
        worker = threading.Thread(target=start)
        worker.start()
        self.assertTrue(entered.wait(1))
        recorder.stop("cancel enable")
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(outcomes, ["cancelled"])
        self.assertTrue(recorder.wait(1))
        self.assertNotEqual(recorder.snapshot()["status"], "recording")

    def test_frame_failure_preserves_first_error_and_calls_back_without_internal_lock(self):
        class FailingWriter(FakeWriter):
            fail = False
            def write(self, rgb, *, timeout_s):
                if self.fail:
                    raise video.RecordingError("disk full simulation")
                super().write(rgb, timeout_s=timeout_s)
        recorder = self.make(writer_factory=self.factory(FailingWriter))
        callback = threading.Event()
        def on_error(message):
            recorder.stop("error callback")  # Would deadlock if invoked under lock.
            recorder.snapshot()
            self.errors.append(message)
            callback.set()
        recorder.on_error = on_error
        recorder.start("disk-fault", self.seeds())
        self.writers[0].fail = True
        recorder.submit(video.CAMERA_KEYS[0], frame(0, 2))
        self.assertTrue(callback.wait(1))
        self.assertTrue(recorder.wait(1))
        self.assertEqual(self.errors, ["disk full simulation"])
        self.assertEqual(recorder.snapshot()["status"], "error")
        self.assertEqual(recorder.snapshot()["dropped"][video.CAMERA_KEYS[0]], 1)

    def test_encoder_readiness_failure_never_allows_start_success(self):
        class NotReadyWriter(FakeWriter):
            def wait_ready(self, *, timeout_s):
                raise video.RecordingError("encoder startup missing codec")
        recorder = self.make(writer_factory=self.factory(NotReadyWriter))
        with self.assertRaisesRegex(video.RecordingError, "missing codec"):
            recorder.start("not-really-ready", self.seeds())
        self.assertTrue(recorder.wait(1))
        self.assertEqual(recorder.snapshot()["status"], "error")

    def test_timestamp_journal_failure_counts_unindexed_delivered_frame(self):
        recorder = self.make()
        original_open = Path.open
        class BrokenJournal:
            def write(self, value):
                raise OSError("timestamp disk write failed")
            def close(self):
                pass
        def open_file(path, *args, **kwargs):
            if path.name == "frames.jsonl":
                return BrokenJournal()
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", open_file):
            with self.assertRaisesRegex(video.RecordingError, "timestamp disk write failed"):
                recorder.start("journal-failed", self.seeds())
            self.assertTrue(recorder.wait(1))
        state = recorder.snapshot()
        self.assertEqual(state["frames"][video.CAMERA_KEYS[0]], 1)
        self.assertEqual(state["unindexed_frames"][video.CAMERA_KEYS[0]], 1)
        self.assertEqual(state["dropped"][video.CAMERA_KEYS[0]], 0)
        self.assertEqual(state["dropped"][video.CAMERA_KEYS[1]], 1)

    def test_repeated_session_id_creates_new_directory_without_overwriting(self):
        recorder = self.make()
        first = recorder.start("same-session", self.seeds())["directory"]
        recorder.stop("first complete"); self.assertTrue(recorder.wait(1))
        before = (Path(first) / "metadata.json").read_bytes()
        second = recorder.start("same-session", self.seeds())["directory"]
        recorder.stop("second complete"); self.assertTrue(recorder.wait(1))
        self.assertNotEqual(first, second)
        self.assertEqual((Path(first) / "metadata.json").read_bytes(), before)

    def test_shared_drain_and_finalize_budget_drops_backlog_instead_of_waiting_forever(self):
        class SlowWriter(FakeWriter):
            slow = False
            def write(self, rgb, *, timeout_s):
                if self.slow:
                    time.sleep(min(.025, timeout_s))
                super().write(rgb, timeout_s=timeout_s)
        recorder = self.make(writer_factory=self.factory(SlowWriter))
        with patch.object(video, "DRAIN_TIMEOUT_S", .06), patch.object(video, "FINALIZE_TIMEOUT_S", .06):
            recorder.start("bounded-drain", self.seeds())
            self.writers[0].slow = True
            for number in range(2, 40):
                recorder.submit(video.CAMERA_KEYS[0], frame(0, number))
            start = time.monotonic()
            recorder.stop("bounded shutdown")
            self.assertTrue(recorder.wait(.5))
            self.assertLess(time.monotonic() - start, .3)
            self.assertGreater(recorder.snapshot()["dropped"][video.CAMERA_KEYS[0]], 0)
            self.assertTrue(all(0 <= writer.close_timeout <= .061 for writer in self.writers))


class FFmpegResolverTests(unittest.TestCase):
    def test_explicit_executable_wins_without_probing_other_environments(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "existing ffmpeg"
            executable.write_text("#!/bin/sh\nexit 0\n"); executable.chmod(0o700)
            with patch.dict(os.environ, {"PI05_FFMPEG": str(executable)}), patch.object(video.subprocess, "run") as run:
                self.assertEqual(video.resolve_ffmpeg(), str(executable))
            run.assert_not_called()

    def test_explicit_converter_fallback_is_bounded_and_only_returns_existing_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "ffmpeg"
            executable.write_text("#!/bin/sh\nexit 0\n"); executable.chmod(0o700)
            env = {"PI05_FFMPEG": "", "PI05_CONVERTER_PYTHON": "/explicit/converter/python"}
            with patch.dict(os.environ, env), patch.object(video.shutil, "which", return_value=None), \
                    patch.dict(sys.modules, {"imageio_ffmpeg": None}), \
                    patch.object(video.subprocess, "run", return_value=types.SimpleNamespace(stdout=str(executable) + "\n")) as run:
                self.assertEqual(video.resolve_ffmpeg(), str(executable))
            self.assertEqual(run.call_args.args[0][0], "/explicit/converter/python")
            self.assertEqual(run.call_args.kwargs["timeout"], 3.)

    def test_missing_dependency_does_not_install_or_download(self):
        with patch.dict(os.environ, {"PI05_FFMPEG": "", "PI05_CONVERTER_PYTHON": ""}), \
                patch.object(video.shutil, "which", return_value=None), \
                patch.dict(sys.modules, {"imageio_ffmpeg": None}), patch.object(video.subprocess, "run") as run:
            with self.assertRaisesRegex(video.RecordingError, "Nothing was downloaded"):
                video.resolve_ffmpeg()
        run.assert_not_called()


@unittest.skipIf(np is None or os.name != "posix", "requires NumPy and POSIX test subprocesses")
class EncoderDeadlineTest(unittest.TestCase):
    def test_hung_owned_encoder_is_rejected_before_start_and_killed_within_budget(self):
        # Only this deliberately inert test subprocess is signaled or killed.
        with tempfile.TemporaryDirectory(prefix="execution-video-stalled-") as temporary:
            executable = Path(temporary) / "inert-ffmpeg"
            executable.write_text("#!/bin/sh\nexec sleep 30\n")
            executable.chmod(0o700)
            with patch.dict(os.environ, {"PI05_FFMPEG": str(executable)}), \
                    patch.object(video, "WRITE_TIMEOUT_S", .06), \
                    patch.object(video, "FINALIZE_TIMEOUT_S", .15), \
                    patch.object(video, "START_TIMEOUT_S", .5):
                recorder = video.ExecutionVideoRecorder(CONFIG, output_root=Path(temporary) / "videos")
                start = time.monotonic()
                with self.assertRaisesRegex(video.RecordingError, "initialization header"):
                    recorder.start("inert-encoder-test", {key: frame(index) for index, key in enumerate(video.CAMERA_KEYS)})
                self.assertTrue(recorder.wait(1))
                self.assertLess(time.monotonic() - start, .7)
                self.assertEqual(recorder.snapshot()["status"], "error")


@unittest.skipIf(np is None, "requires existing NumPy")
class SyntheticFFmpegTest(unittest.TestCase):
    def test_real_two_camera_h264_mp4_encode_and_decode(self):
        try:
            import cv2
            executable = video.resolve_ffmpeg()
        except (ImportError, video.RecordingError) as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory(prefix="execution-video-real-") as temporary, \
                patch.dict(os.environ, {"PI05_FFMPEG": executable}):
            recorder = video.ExecutionVideoRecorder(CONFIG, output_root=temporary)
            seeds = {key: frame(index) for index, key in enumerate(video.CAMERA_KEYS)}
            try:
                recorder.start("synthetic-only", seeds)
                for number in range(2, 11):
                    for camera, key in enumerate(video.CAMERA_KEYS):
                        self.assertTrue(recorder.submit(key, frame(camera, number)))
            finally:
                recorder.stop("synthetic done")
                self.assertTrue(recorder.wait(8))
            snapshot = recorder.snapshot()
            self.assertEqual(snapshot["status"], "saved", snapshot)
            self.assertEqual(snapshot["frames"], {key: 10 for key in video.CAMERA_KEYS})
            for index in (1, 2):
                capture = cv2.VideoCapture(str(Path(snapshot["directory"]) / f"camera_{index}.mp4"))
                try:
                    count = 0
                    while True:
                        ok, image = capture.read()
                        if not ok:
                            break
                        self.assertEqual(image.shape, (24, 32, 3))
                        count += 1
                    self.assertEqual(count, 10)
                finally:
                    capture.release()


if __name__ == "__main__":
    unittest.main()
