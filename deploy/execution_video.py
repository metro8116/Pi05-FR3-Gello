"""Bounded, asynchronous two-camera execution videos; no hardware on import.

Camera arrays are already owned immutable copies from RealSensePair. submit()
only queues their references and metadata; encoding and disk I/O stay off the
camera/control threads. This recording is for reviewing execution, not training.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import re
import select
import shutil
import subprocess
import threading
import time
import uuid

CAMERA_KEYS = ("observation.images.camera_1", "observation.images.camera_2")
START_TIMEOUT_S = 3.0
WRITE_TIMEOUT_S = .75
DRAIN_TIMEOUT_S = 5.0
FINALIZE_TIMEOUT_S = 2.0


class RecordingError(RuntimeError):
    pass


def resolve_ffmpeg():
    """Resolve an existing executable only; never install or download one."""
    explicit = os.environ.get("PI05_FFMPEG")
    if explicit:
        path = shutil.which(explicit) or str(Path(explicit).expanduser())
        if Path(path).is_file() and os.access(path, os.X_OK):
            return path
        raise RecordingError(f"PI05_FFMPEG is not an executable file: {explicit}")
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if Path(path).is_file() and os.access(path, os.X_OK):
            return path
    except (ImportError, OSError, RuntimeError):
        pass
    converter = os.environ.get("PI05_CONVERTER_PYTHON")
    if converter:
        try:
            result = subprocess.run(
                [converter, "-B", "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
                check=True, capture_output=True, text=True, timeout=3.)
            path = result.stdout.strip()
            if Path(path).is_file() and os.access(path, os.X_OK):
                return path
        except (OSError, subprocess.SubprocessError):
            pass
    raise RecordingError("Video recording needs an existing FFmpeg executable: install system ffmpeg, "
                         "provide imageio-ffmpeg, or set PI05_FFMPEG to an existing binary. Nothing was downloaded.")


class _FFmpegWriter:
    def __init__(self, path, width, height, fps):
        self.path = Path(path)
        self.log_path = self.path.with_suffix(".ffmpeg.log")
        self.log = self.log_path.open("xb")
        self.process = None
        self.closed = False
        try:
            self.process = subprocess.Popen([
                resolve_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "1",
                "-g", str(max(1, round(fps))),
                "-movflags", "+frag_keyframe+empty_moov+default_base_moof", str(self.path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log, bufsize=0)
            os.set_blocking(self.process.stdin.fileno(), False)
        except Exception:
            self.log.close()
            if self.process is not None:
                self.process.kill()
                self.process.wait(timeout=.5)
            raise

    def _failure(self, message):
        try:
            detail = self.log_path.read_text(errors="replace")[-1500:].strip()
        except OSError:
            detail = ""
        return RecordingError(message + (": " + detail if detail else ""))

    def write(self, rgb, *, timeout_s):
        if self.closed:
            raise RecordingError("Encoder is already closed")
        data = memoryview(rgb.tobytes())
        deadline = time.monotonic() + timeout_s
        fd = self.process.stdin.fileno()
        while data:
            if self.process.poll() is not None:
                raise self._failure(f"FFmpeg exited while recording (code {self.process.returncode})")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._failure("FFmpeg frame write timed out")
            if not select.select([], [fd], [], min(.05, remaining))[1]:
                continue
            try:
                count = os.write(fd, data[:65536])
            except BlockingIOError:
                continue
            except (BrokenPipeError, OSError) as exc:
                raise self._failure(f"FFmpeg pipe failed: {exc}") from exc
            if count <= 0:
                raise self._failure("FFmpeg pipe accepted no bytes")
            data = data[count:]

    def close(self, *, timeout_s):
        if self.closed:
            return
        self.closed = True
        deadline = time.monotonic() + max(0., timeout_s)
        try:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=max(0., deadline - time.monotonic() - .2))
            except subprocess.TimeoutExpired as exc:
                self.process.kill()
                try:
                    self.process.wait(timeout=max(.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass
                raise self._failure("FFmpeg finalization timed out; encoder was killed") from exc
            if self.process.returncode != 0:
                raise self._failure(f"FFmpeg finalization failed (code {self.process.returncode})")
            if not self.path.is_file() or self.path.stat().st_size == 0:
                raise RecordingError(f"Encoder produced no video: {self.path}")
        finally:
            self.log.close()

    def wait_ready(self, *, timeout_s):
        """Confirm the encoder opened a real MP4, not merely a writable pipe."""
        deadline = time.monotonic() + timeout_s
        while True:
            if self.process.poll() is not None:
                raise self._failure(f"FFmpeg failed during startup (code {self.process.returncode})")
            try:
                with self.path.open("rb") as handle:
                    header = handle.read(12)
                if len(header) >= 12 and header[4:8] == b"ftyp" and self.process.poll() is None:
                    return
            except FileNotFoundError:
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._failure("FFmpeg did not create an MP4 initialization header before startup deadline")
            time.sleep(min(.01, remaining))


class ExecutionVideoRecorder:
    def __init__(self, camera_config, *, output_root=None, on_error=None, on_event=None,
                 fps=None, queue_size=64, writer_factory=None,
                 clock=time.monotonic, wall_clock=time.time):
        self.config = dict(camera_config)
        self.width, self.height = self.config.get("width", 640), self.config.get("height", 480)
        self.fps = fps if fps is not None else self.config.get("capture_fps", 30)
        serials = self.config.get("serials", [])
        if (len(serials) != 2 or len(set(serials)) != 2 or any(not isinstance(s, str) or not s for s in serials)
                or type(self.width) is not int or type(self.height) is not int
                or self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2
                or isinstance(self.fps, bool) or not isinstance(self.fps, (int, float))
                or not math.isfinite(self.fps) or self.fps <= 0
                or type(queue_size) is not int or queue_size < 2):
            raise RecordingError("Invalid two-camera video configuration")
        self.serials = dict(zip(CAMERA_KEYS, serials))
        self.output_root = Path(output_root if output_root is not None else Path(__file__).resolve().parent / "recordings").expanduser()
        self.on_error, self.on_event = on_error, on_event
        self.writer_factory = writer_factory or _FFmpegWriter
        self.clock, self.wall_clock, self.queue_size = clock, wall_clock, queue_size
        self.lock = threading.Lock()
        self.worker = None
        self.done = threading.Event()
        self.done.set()
        self.ready = threading.Event()
        self.stop_event = threading.Event()
        self.status, self.directory, self.error = "idle", None, None
        self.accepting = False
        self.cancelled_before_start = False
        self.session_id = None
        self.frames = {key: 0 for key in CAMERA_KEYS}
        self.dropped = {key: 0 for key in CAMERA_KEYS}
        self.duplicates = {key: 0 for key in CAMERA_KEYS}
        self.unindexed_frames = {key: 0 for key in CAMERA_KEYS}

    def _notify(self, callback, *args, **kwargs):
        if callback is None:
            return
        # User callbacks must not block the encoder or hold its lock. Runtime
        # callbacks revoke model control; this thread owns no encoder resources.
        def invoke():
            try:
                callback(*args, **kwargs)
            except Exception:
                pass
        threading.Thread(target=invoke, name="execution-video-notify", daemon=True).start()

    def _fail(self, message):
        with self.lock:
            first = self.error is None
            if first:
                self.error = str(message)
            self.accepting = False
            self.status = "error"
            self._request_stop_locked("recording_error")
            error, session = self.error, self.session_id
        if first:
            self._notify(self.on_error, error)
            self._notify(self.on_event, "error", session_id=session, error=error)

    def _frame(self, key, frame):
        import numpy as np
        if key not in CAMERA_KEYS or not isinstance(frame, dict):
            raise RecordingError("Unknown camera or invalid video frame")
        rgb, stamp, number = frame.get("rgb"), frame.get("monotonic"), frame.get("number")
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                or rgb.shape != (self.height, self.width, 3)
                or isinstance(stamp, bool) or not isinstance(stamp, (int, float))
                or not math.isfinite(stamp) or stamp < 0 or type(number) is not int or number < 0
                or frame.get("serial") != self.serials[key]):
            raise RecordingError(f"Invalid RGB/source metadata for {key}")
        metadata = {name: value for name, value in frame.items() if name in {
            "monotonic", "number", "serial", "source_timestamp_ms", "source_timestamp_domain",
            "received_monotonic", "global_time_enabled", "device_timestamp_ms", "timestamp_domain"}}
        # Check only the tiny metadata object. Pixel bytes stay in the camera's
        # immutable owned array until the worker encodes them.
        try:
            json.dumps(metadata, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise RecordingError(f"Invalid video timestamp metadata: {exc}") from exc
        return rgb, metadata

    def start(self, session_id, initial_frames):
        if not isinstance(session_id, str) or not session_id or not isinstance(initial_frames, dict):
            raise RecordingError("Recording needs a session_id and two real initial camera frames")
        seeds = [(key, *self._frame(key, initial_frames.get(key))) for key in CAMERA_KEYS]
        with self.lock:
            if self.cancelled_before_start:
                raise RecordingError("Video recording start was cancelled before startup")
            if self.worker is not None and not self.done.is_set():
                raise RecordingError("Previous video recording is still active or saving")
            self.session_id, self.status, self.error = session_id, "starting", None
            self.accepting = True
            self.frames = {key: 0 for key in CAMERA_KEYS}
            self.dropped = {key: 0 for key in CAMERA_KEYS}
            self.duplicates = {key: 0 for key in CAMERA_KEYS}
            self.unindexed_frames = {key: 0 for key in CAMERA_KEYS}
            self.seen = {key: (metadata["number"], metadata["monotonic"]) for key, _, metadata in seeds}
            self.queue = queue.Queue(maxsize=self.queue_size)
            for item in seeds:
                self.queue.put_nowait(item)
            self.ready, self.done, self.stop_event = threading.Event(), threading.Event(), threading.Event()
            self.started_mono, self.started_wall = self.clock(), self.wall_clock()
            self.stop_real = self.stop_mono = self.stop_wall = self.stop_reason = None
            self.ended_mono = self.ended_wall = None
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)[:64]
            name = datetime.fromtimestamp(self.started_wall).strftime("%Y%m%d_%H%M%S")
            self.directory = self.output_root / f"{name}_{safe}_{uuid.uuid4().hex[:8]}"
            self.worker = threading.Thread(target=self._run, name="execution-video-encoder", daemon=False)
            self.worker.start()
            ready = self.ready
        if not ready.wait(START_TIMEOUT_S):
            self._fail("Video recording startup timed out")
            raise RecordingError("Video recording startup timed out")
        with self.lock:
            if self.status != "recording":
                raise RecordingError(self.error or "Video recording start was cancelled")
        return self.snapshot()

    def submit(self, camera_key, frame):
        with self.lock:
            if not self.accepting:
                return False
        try:
            rgb, metadata = self._frame(camera_key, frame)
        except RecordingError as exc:
            self._fail(str(exc))
            return False
        with self.lock:
            if not self.accepting:
                return False
            number, stamp = metadata["number"], metadata["monotonic"]
            previous_number, previous_stamp = self.seen.get(camera_key, (-1, -1.))
            if number <= previous_number or stamp <= previous_stamp:
                self.duplicates[camera_key] += 1
                return False
            self.seen[camera_key] = (number, stamp)
            try:
                self.queue.put_nowait((camera_key, rgb, metadata))
            except queue.Full:
                self.dropped[camera_key] += 1
                return False
            return True

    def _request_stop_locked(self, reason):
        if getattr(self, "stop_real", None) is None:
            self.stop_real, self.stop_mono, self.stop_wall = time.monotonic(), self.clock(), self.wall_clock()
            self.stop_reason = str(reason)
        self.stop_event.set()

    def stop(self, reason="stopped"):
        with self.lock:
            self.accepting = False
            if self.worker is None:
                # Runtime can revoke the recorder after assigning it but before
                # its arm worker calls start(). A late start must stay revoked.
                self.cancelled_before_start = True
                self.stop_event.set()
                return
            if self.done.is_set():
                return
            self._request_stop_locked(reason)
            if self.status != "error":
                self.status = "saving"

    def wait(self, timeout=None):
        return self.done.wait(timeout)

    def snapshot(self):
        with self.lock:
            return dict(status=self.status, directory=None if self.directory is None else str(self.directory),
                        error=self.error, session_id=self.session_id, fps=self.fps,
                        frames=dict(self.frames), dropped=dict(self.dropped), duplicates=dict(self.duplicates),
                        unindexed_frames=dict(self.unindexed_frames))

    def _metadata(self):
        snapshot = self.snapshot()
        with self.lock:
            snapshot.update(schema_version=1, purpose="execution_review_video_not_training_data",
                            started_wall=self.started_wall, started_monotonic=self.started_mono,
                            stop_requested_wall=self.stop_wall, stop_requested_monotonic=self.stop_mono,
                            ended_wall=self.ended_wall, ended_monotonic=self.ended_mono,
                            stop_reason=self.stop_reason, camera_serials=dict(self.serials),
                            width=self.width, height=self.height, pixel_format="rgb24", codec="h264",
                            frame_timing="one encoded frame per accepted capture; no duplication or resampling; nominal video FPS",
                            frame_count_semantics="frames counts complete raw frames delivered to encoder; unindexed_frames counts delivered frames whose timestamp journal write failed; such recordings are errors",
                            files={key: f"camera_{index}.mp4" for index, key in enumerate(CAMERA_KEYS, 1)},
                            frame_metadata="frames.jsonl")
        temporary = self.directory / "metadata.json.tmp"
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.directory / "metadata.json")

    def _discard_queue(self):
        while True:
            try:
                key, _, _ = self.queue.get_nowait()
            except queue.Empty:
                return
            with self.lock:
                self.dropped[key] += 1

    def _run(self):
        writers, journal = {}, None
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            journal = (self.directory / "frames.jsonl").open("x", encoding="utf-8")
            self._metadata()
            for index, key in enumerate(CAMERA_KEYS, 1):
                if self.stop_event.is_set():
                    raise RecordingError("Video recording start was cancelled")
                writers[key] = self.writer_factory(self.directory / f"camera_{index}.mp4", self.width, self.height, self.fps)
            while True:
                with self.lock:
                    stop_real, error = self.stop_real, self.error
                if error:
                    break
                if stop_real is not None and time.monotonic() - stop_real >= DRAIN_TIMEOUT_S:
                    self._discard_queue()
                    break
                try:
                    key, rgb, metadata = self.queue.get(timeout=.05)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue
                budget = WRITE_TIMEOUT_S
                if stop_real is not None:
                    budget = min(budget, max(.01, stop_real + DRAIN_TIMEOUT_S - time.monotonic()))
                try:
                    writers[key].write(rgb, timeout_s=budget)
                except Exception:
                    with self.lock:
                        self.dropped[key] += 1
                    raise
                with self.lock:
                    index = self.frames[key]
                    self.frames[key] += 1
                try:
                    journal.write(json.dumps(dict(camera_key=key, frame_index=index, **metadata), allow_nan=False) + "\n")
                    journal.flush()
                except Exception:
                    with self.lock:
                        self.unindexed_frames[key] += 1
                    raise
                with self.lock:
                    initial_frames_written = (self.status == "starting" and not self.stop_event.is_set()
                                              and all(self.frames[name] > 0 for name in CAMERA_KEYS))
                if initial_frames_written:
                    for writer in writers.values():
                        wait_ready = getattr(writer, "wait_ready", None)
                        if callable(wait_ready):
                            wait_ready(timeout_s=WRITE_TIMEOUT_S)
                with self.lock:
                    started = (self.status == "starting" and not self.stop_event.is_set()
                               and all(self.frames[name] > 0 for name in CAMERA_KEYS))
                    if started:
                        self.status = "recording"
                        self.ready.set()
                if started:
                    self._notify(self.on_event, "started", **self.snapshot())
        except Exception as exc:
            self._fail(str(exc))
        finally:
            with self.lock:
                self.accepting = False
                if self.stop_real is None:
                    self._request_stop_locked("recording_finished")
                stop_real = self.stop_real
                if self.status != "error":
                    self.status = "saving"
            self._discard_queue()
            # A single shared deadline bounds both encoders, rather than giving
            # each one the entire finalization allowance.
            deadline = min(time.monotonic() + FINALIZE_TIMEOUT_S,
                           stop_real + DRAIN_TIMEOUT_S + FINALIZE_TIMEOUT_S)
            for writer in writers.values():
                try:
                    writer.close(timeout_s=max(0., deadline - time.monotonic()))
                except Exception as exc:
                    self._fail(str(exc))
            if journal is not None:
                try:
                    journal.close()
                except OSError as exc:
                    self._fail(str(exc))
            with self.lock:
                self.ended_mono, self.ended_wall = self.clock(), self.wall_clock()
                self.status = "error" if self.error else "saved"
            try:
                if self.directory.is_dir():
                    self._metadata()
            except Exception as exc:
                self._fail(f"Cannot save video metadata: {exc}")
            self.ready.set()
            self.done.set()
            if self.error is None:
                self._notify(self.on_event, "saved", **self.snapshot())
