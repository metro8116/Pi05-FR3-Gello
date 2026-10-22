"""Local camera/client runtime. No Torch, LeRobot, ROS, or hardware side effects on import.

The camera and inference workers are independent of the Tk event loop. Each RPC
uses its own socket, so a slow cloud request cannot block a local stop request.
Only the ROS bridge can own execution; reconnecting never restores an arm token.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
import uuid

from config import ROOT
from execution_video import ExecutionVideoRecorder
from protocol import (CAMERA_KEYS, PROTOCOL_VERSION, encode_image,
                      recv_packet, send_packet, validate_actions_response,
                      validate_metadata)


class ClientError(RuntimeError):
    pass


class CameraTimestampMapper:
    """Map actual SDK capture timestamps, never just assign the receipt time.

    Global/system domains use epoch timestamps. Hardware-only clocks use the
    minimum observed receive-minus-device offset; a short calibration period
    avoids immediately treating a buffered first frame as a fresh capture.
    Hardware fallback cannot estimate a constant transport delay, so metadata
    preserves that clock mode for the operator and diagnostics.
    """

    def __init__(self, max_age_s=0.2, hardware_warmup_frames=5):
        self.max_age_s = float(max_age_s)
        self.hardware_warmup_frames = int(hardware_warmup_frames)
        self.domain = self.last_timestamp_ms = self.last_frame_number = None
        self.minimum_offset = self.last_capture = None
        self.hardware_samples = 0

    def map(self, timestamp_ms, domain, frame_number, received_monotonic, received_wall):
        for value in (timestamp_ms, received_monotonic, received_wall):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ClientError('相机源时间戳不是有限数值')
        if timestamp_ms < 0 or type(frame_number) is not int or frame_number < 0:
            raise ClientError('相机源时间或帧号无效')
        domain = str(domain).rsplit('.', 1)[-1]
        if domain not in ('global_time', 'system_time', 'hardware_clock'):
            raise ClientError(f'不支持的相机时间域: {domain}')
        if self.last_frame_number is not None and frame_number <= self.last_frame_number:
            raise ClientError('相机帧号重复或回退')
        if domain == self.domain and self.last_timestamp_ms is not None and timestamp_ms <= self.last_timestamp_ms:
            raise ClientError('相机源时间戳未更新或回退')
        if domain != self.domain:
            self.minimum_offset, self.hardware_samples = None, 0
        self.domain, self.last_timestamp_ms, self.last_frame_number = domain, timestamp_ms, frame_number
        device_s = timestamp_ms / 1000.0
        if domain in ('global_time', 'system_time'):
            age = received_wall - device_s
            if age < -0.001:
                raise ClientError('相机源时间戳在本机时钟未来')
            # Sub-millisecond conversion/clock-read rounding must not create a
            # future monotonic capture timestamp.
            capture = received_monotonic - max(age, 0.0)
        else:
            candidate = received_monotonic - device_s
            self.minimum_offset = candidate if self.minimum_offset is None else min(self.minimum_offset, candidate)
            self.hardware_samples += 1
            if self.hardware_samples < self.hardware_warmup_frames:
                return None
            capture = device_s + self.minimum_offset
        age = received_monotonic - capture
        if not 0 <= age <= self.max_age_s:
            raise ClientError(f'相机源帧已过期: {age:.3f}s')
        if self.last_capture is not None and capture <= self.last_capture:
            raise ClientError('映射后的相机采集时刻未更新或回退')
        self.last_capture = capture
        return capture


class RPCClient:
    """Short-lived framed requests; cancellation affects only this transport."""

    def __init__(self, host, port, timeout_s=1.0, connect_timeout_s=None):
        self.address = (host, int(port))
        self.timeout_s = float(timeout_s)
        self.connect_timeout_s = float(connect_timeout_s or timeout_s)
        self._lock = threading.Lock()
        self._sockets = set()

    def request(self, packet):
        sock = socket.create_connection(self.address, timeout=self.connect_timeout_s)
        with self._lock:
            self._sockets.add(sock)
        try:
            sock.settimeout(self.timeout_s)
            send_packet(sock, packet)
            reply = recv_packet(sock)
            if reply.get('type') == 'error' or reply.get('ok') is False:
                raise ClientError(str(reply.get('error') or reply.get('message') or reply))
            return reply
        finally:
            with self._lock:
                self._sockets.discard(sock)
            sock.close()

    def cancel(self):
        with self._lock:
            sockets = tuple(self._sockets)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()


class RealSensePair:
    """Keep the latest RGB frame from each fixed serial, without queued old frames."""

    def __init__(self, config, clock=time.monotonic):
        self.config, self.clock = config, clock
        self._lock = threading.Lock()
        self._frames = {}
        self._errors = {}
        self._threads = []
        self._stop = threading.Event()
        self._frame_listeners = []

    def start(self):
        if self._threads:
            raise ClientError('相机已启动')
        try:
            import pyrealsense2 as rs
            import numpy as np
        except ImportError as exc:
            raise ClientError('本机环境需要 pyrealsense2 和 numpy；请使用原采集环境') from exc
        self._stop.clear()
        for key, serial in zip(CAMERA_KEYS, self.config['serials']):
            worker = threading.Thread(target=self._capture, args=(rs, np, key, serial), daemon=True)
            self._threads.append(worker)
            worker.start()

    def _capture(self, rs, np, key, serial):
        pipeline = rs.pipeline()
        started = False
        try:
            options = rs.config()
            options.enable_device(serial)
            options.enable_stream(rs.stream.color, self.config['width'], self.config['height'],
                                  rs.format.rgb8, self.config['capture_fps'])
            # Prefer a source clock comparable across both cameras. Unsupported
            # sensors/firmware use the explicit hardware-clock fallback below.
            global_time_enabled = False
            try:
                profile = options.resolve(rs.pipeline_wrapper(pipeline))
                for sensor in profile.get_device().query_sensors():
                    is_color = any(stream.stream_type() == rs.stream.color for stream in sensor.get_stream_profiles())
                    if is_color and sensor.supports(rs.option.global_time_enabled):
                        sensor.set_option(rs.option.global_time_enabled, 1.0)
                        global_time_enabled = True
            except (RuntimeError, AttributeError):
                pass
            pipeline.start(options)
            started = True
            mapper = CameraTimestampMapper(self.config['max_age_s'])
            # SDK frames are never retained: copy once, latest-only storage.
            while not self._stop.is_set():
                frame = pipeline.wait_for_frames(timeout_ms=500).get_color_frame()
                if not frame:
                    continue
                number = frame.get_frame_number()
                received_wall, received = time.time(), self.clock()
                timestamp_ms = frame.get_timestamp()
                domain = str(frame.get_frame_timestamp_domain())
                try:
                    captured = mapper.map(timestamp_ms, domain, number, received, received_wall)
                    if captured is None:
                        continue
                except ClientError as exc:
                    with self._lock:
                        self._errors[key] = str(exc)
                    continue
                rgb = np.asanyarray(frame.get_data()).copy()
                packet = {'rgb': rgb, 'monotonic': captured, 'number': number, 'serial': serial,
                          'source_timestamp_ms': timestamp_ms, 'source_timestamp_domain': domain,
                          'received_monotonic': received, 'global_time_enabled': global_time_enabled}
                with self._lock:
                    self._frames[key] = packet
                    self._errors.pop(key, None)
                    listeners = tuple(self._frame_listeners)
                # Subscribers only enqueue immutable RGB frames. Encoding and
                # disk I/O never run in a capture or control thread.
                for listener in listeners:
                    try:
                        listener(key, packet)
                    except Exception as exc:
                        with self._lock:
                            self._errors[key] = '录像帧接收失败: ' + str(exc)
        except Exception as exc:
            with self._lock:
                self._errors[key] = str(exc)
        finally:
            if started:
                try:
                    pipeline.stop()
                except Exception:
                    pass

    def snapshot(self):
        # Arrays are immutable after insertion; callers need not make a GUI copy.
        with self._lock:
            return {key: dict(value) for key, value in self._frames.items()}, dict(self._errors)

    def add_frame_listener(self, listener):
        with self._lock:
            if listener not in self._frame_listeners:
                self._frame_listeners.append(listener)

    def remove_frame_listener(self, listener):
        with self._lock:
            if listener in self._frame_listeners:
                self._frame_listeners.remove(listener)

    def close(self):
        self._stop.set()
        for worker in self._threads:
            worker.join(timeout=0.7)
        self._threads.clear()
        with self._lock:
            self._frames.clear()


def check_cameras(frames, errors, config, now):
    """Validate mapped source ages/skew; never compare an AutoDL clock to this clock."""
    if errors:
        raise ClientError('相机错误: ' + '; '.join(f'{k}: {v}' for k, v in errors.items()))
    if set(frames) != set(CAMERA_KEYS):
        raise ClientError('等待两台相机的新帧')
    times = [float(frames[key]['monotonic']) for key in CAMERA_KEYS]
    ages = [now - value for value in times]
    if any(not math.isfinite(age) or age < -0.001 or age > config['max_age_s'] for age in ages):
        raise ClientError(f'相机帧过期，当前年龄 {[round(a, 3) for a in ages]} 秒')
    if max(times) - min(times) > config['max_skew_s']:
        raise ClientError(f'两路相机采集时间差超过 {config["max_skew_s"]} 秒')
    return min(times)


def check_state(packet, max_age_s=0.2):
    if packet.get('version') != PROTOCOL_VERSION or packet.get('type') != 'state':
        raise ClientError('本机数据桥状态协议不匹配')
    state = packet.get('state')
    if not isinstance(state, list) or len(state) != 8 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in state):
        raise ClientError('本机实际状态必须为有限的 8 维关节角/夹爪总开口')
    if not 0 <= state[7] <= 0.080001:
        raise ClientError('夹爪实际总开口超出 0..0.08 m')
    for key, limit in (('state_age_s', max_age_s), ('gripper_age_s', 0.5)):
        age = packet.get(key)
        if isinstance(age, bool) or not isinstance(age, (int, float)) or not math.isfinite(age) or not 0 <= age <= limit:
            raise ClientError(f'{key} 过期或无效: {age}')
    return list(state)


class DeploymentRuntime:
    """Session-fenced execution; stop clears the local session before doing I/O."""

    def __init__(self, config, on_event=None, cameras=None, model=None, bridge=None, clock=time.monotonic,
                 recorder_factory=None):
        self.config, self.clock = config, clock
        self.on_event = on_event or (lambda _event: None)
        self.cameras = cameras or RealSensePair(config['cameras'], clock)
        mc, bc = config['model'], config['bridge']
        self.model = model or RPCClient(mc['host'], mc['port'], mc['request_timeout_s'], mc['connect_timeout_s'])
        self.bridge = bridge or RPCClient(bc['host'], bc['port'], bc['timeout_s'])
        self._lock = threading.RLock()
        self._stop_all = threading.Event()
        self._connected = False
        self._session = None
        self._armed = False
        self._arm_monotonic = None
        self._heartbeat_ready = False
        self._generation = 0
        self._metadata = None
        self._state = None
        self._capture_seq = 0
        self._request_id = 0
        self._workers = []
        self._last_fault = None
        self._arm_pending = False
        self._preview_pending = False
        self._manual_pending = False
        self._manual_operation_id = None
        self._recorder_factory = recorder_factory or ExecutionVideoRecorder
        self._recording = self._recording_listener = None

    def _emit(self, kind, **fields):
        self.on_event({'kind': kind, **fields})

    def snapshot(self):
        with self._lock:
            return {'connected': self._connected, 'armed': self._armed,
                    'arming': not self._armed and (self._arm_pending or self._session is not None),
                    'manual_opening': self._manual_pending,
                    'session_id': self._session, 'metadata': self._metadata,
                    'state': self._state, 'fault': self._last_fault,
                    'recording': None if self._recording is None else self._recording.snapshot()}

    def _recording_event(self, token, kind, **fields):
        fields.setdefault('session_id', token)
        name = kind if kind.startswith('recording_') else 'recording_' + kind
        if name == 'recording_error':
            fields.setdefault('message', str(fields.get('error', '录像失败')))
            fields.setdefault('stale', not self._is_session(token))
        self._emit(name, **fields)

    def _recording_error(self, token, message):
        current = self.stop('录像失败，已停用模型: ' + str(message), expected_session=token)
        self._emit('recording_error', session_id=token, message=str(message), stale=not current)

    def _start_recording(self, token, frames):
        options = self.config.get('recording', {})
        directory = Path(os.environ.get('PI05_RECORDING_DIR', options.get('directory', 'recordings'))).expanduser()
        if not directory.is_absolute():
            directory = ROOT / directory
        recorder = self._recorder_factory(
            self.config['cameras'], output_root=directory,
            fps=self.config['cameras']['capture_fps'], queue_size=options.get('queue_size', 64),
            on_error=lambda message: self._recording_error(token, message),
            on_event=lambda kind, **fields: self._recording_event(token, kind, **fields))
        listener = recorder.submit
        with self._lock:
            if not self._is_session(token):
                recorder.stop('启用前已取消')
                return
            self._recording, self._recording_listener = recorder, listener
            self.cameras.add_frame_listener(listener)
        recorder.start(token, frames)
        if not self._is_session(token):
            recorder.stop('启用录像过程中已取消')

    def wait_recording(self, timeout=0):
        """Only GUI close/finalization waits; Stop itself never waits for disk."""
        with self._lock:
            recorder = self._recording
        return recorder is None or recorder.wait(timeout)

    def connect(self):
        """Run in a worker; connect performs no bridge arm or model action."""
        if self._stop_all.is_set():
            raise ClientError('窗口已关闭')
        with self._lock:
            if self._connected:
                raise ClientError('已经连接；停止不会自动重新启用执行')
        metadata = self.model.request({'version': PROTOCOL_VERSION, 'type': 'hello'})
        validate_metadata(metadata)
        if metadata.get('fps') != self.config['execution']['fps']:
            raise ClientError('服务器动作频率与训练 15Hz 不一致')
        state = self.bridge.request({'version': PROTOCOL_VERSION, 'type': 'state'})
        # In preview, report stale/unavailable states without permitting arm.
        with self._lock:
            if self._stop_all.is_set():
                return
            # close() fences this section with the same lock. A delayed hello
            # reply can never start cameras after the GUI has relinquished them.
            self.cameras.start()
            self._metadata, self._state = metadata, state
            self._connected, self._last_fault = True, None
        for target in (self._monitor_loop, self._inference_loop):
            worker = threading.Thread(target=target, daemon=True)
            self._workers.append(worker)
            worker.start()
        self._emit('connected', metadata=metadata)

    def arm(self):
        """Only called after the operator explicitly presses the execution button."""
        # A read-only prediction and activation cannot overlap, including the
        # activation preflight before a bridge session token has been allocated.
        with self._lock:
            if self._arm_pending or self._preview_pending or self._manual_pending:
                raise ClientError('启用、只预测诊断或手动开爪正在处理，请等待完成')
            self._arm_pending = True
        try:
            return self._arm_impl()
        finally:
            with self._lock:
                self._arm_pending = False

    def _arm_impl(self):
        with self._lock:
            if not self._connected or self._stop_all.is_set():
                raise ClientError('先连接推理服务、数据桥和相机')
            if self._session is not None:
                raise ClientError('已有执行/启用请求')
            if self._recording is not None and not self._recording.wait(0):
                raise ClientError('上一段录像正在保存，请稍候再启用')
            metadata = self._metadata
            generation = self._generation
            if metadata.get('mock') is not False:
                raise ClientError('模拟服务或未声明真实模型的服务禁止启用 FR3')
        frames, errors = self.cameras.snapshot()
        check_cameras(frames, errors, self.config['cameras'], self.clock())
        state = self.bridge.request({'version': PROTOCOL_VERSION, 'type': 'state'})
        check_state(state, self.config['execution']['state_max_age_s'])
        # Bridge keeps its previous stop reason as history. An explicit new
        # session may recover when ready=True; the bridge checks controller
        # inactivity and cleanup completion before accepting arm.
        if not state.get('ready'):
            raise ClientError(f'机械臂尚未就绪: {state.get("fault") or "ready=false"}')
        if state.get('armed'):
            raise ClientError('数据桥已由其他执行会话占用')
        token = str(uuid.uuid4())
        with self._lock:
            if self._session is not None or self._stop_all.is_set() or generation != self._generation:
                raise ClientError('启用请求已取消或另一个请求正在处理')
            self._session, self._last_fault = token, None
        try:
            # Prepare both real RGB writers before the first arm request. This
            # runs in the GUI operation worker, never in the control loop.
            self._start_recording(token, frames)
            if not self._is_session(token):
                return
            frames, errors = self.cameras.snapshot()
            check_cameras(frames, errors, self.config['cameras'], self.clock())
            if not self._is_session(token):
                return
            recording = self._recording.snapshot()
            if recording.get('status') != 'recording':
                raise ClientError('双相机录像未就绪: ' + str(recording.get('error') or recording.get('status')))
            result = self.bridge.request({'version': PROTOCOL_VERSION, 'type': 'arm',
                                          'session_id': token, 'model_id': metadata['model_id']})
            if result.get('fault'):
                raise ClientError(str(result['fault']))
            with self._lock:
                if self._session == token:
                    self._heartbeat_ready = True
            deadline = self.clock() + 3.0
            while self.clock() < deadline:
                if not self._is_session(token):
                    self._send_stop(token, '启用过程中已取消')
                    return
                status = self.bridge.request({'version': PROTOCOL_VERSION, 'type': 'state'})
                if status.get('fault'):
                    raise ClientError(str(status['fault']))
                if status.get('armed') and status.get('session_id') == token:
                    # The first reply can be faster than the monitor's 100 ms
                    # period. Establish camera health BEFORE inference may run.
                    frames, errors = self.cameras.snapshot()
                    captured = check_cameras(frames, errors, self.config['cameras'], self.clock())
                    heartbeat = self.bridge.request({'version': PROTOCOL_VERSION, 'type': 'heartbeat',
                                                     'session_id': token, 'camera_ok': True,
                                                     'camera_capture_monotonic': captured})
                    if heartbeat.get('fault'):
                        raise ClientError(str(heartbeat['fault']))
                    with self._lock:
                        cancelled = self._session != token
                        if not cancelled:
                            self._state, self._armed = status, True
                            self._arm_monotonic = status.get('armed_monotonic') or self.clock()
                    if cancelled:
                        self._send_stop(token, '启用过程中已取消')
                        return
                    self._emit('armed', session_id=token)
                    return
                self._stop_all.wait(0.05)
            raise ClientError('本机控制器未能在 3 秒内完成启用')
        except Exception as exc:
            if self._is_session(token):
                self.stop('启用失败: ' + str(exc))
            else:
                self._send_stop(token, '取消延迟启用')
            raise

    def _preview_generation_current(self, generation):
        with self._lock:
            return (generation == self._generation and self._connected
                    and not self._stop_all.is_set() and not self._armed
                    and self._session is None and not self._arm_pending
                    and not self._manual_pending)

    def preview_inference(self):
        """Predict from current images/state without issuing any robot commands.

        Run in a GUI worker. This path requests only bridge ``state`` and model
        ``infer``; it never arms, heartbeats or submits actions. Stop/close and
        activation fence late responses with the existing generation counter.
        Targets are reported exactly as predicted, including out-of-range values.
        """
        generation = None
        owns_preview = False
        try:
            with self._lock:
                if not self._connected or self._stop_all.is_set():
                    raise ClientError('请先连接并预览，再运行只预测诊断')
                if self._armed or self._session is not None or self._arm_pending or self._manual_pending:
                    raise ClientError('只预测诊断仅可在未启用/未启用中的预览状态运行')
                if self._preview_pending:
                    raise ClientError('只预测诊断已在处理，请等待完成')
                self._preview_pending = owns_preview = True
                generation = self._generation
                metadata = dict(self._metadata)
                self._capture_seq += 1
                self._request_id += 1
                sequence, request_id = self._capture_seq, self._request_id
            frames, errors = self.cameras.snapshot()
            sample_time, sample_wall = self.clock(), time.time()
            captured = check_cameras(frames, errors, self.config['cameras'], sample_time)
            packet = self.bridge.request({'version': PROTOCOL_VERSION, 'type': 'state'})
            measured = check_state(packet, self.config['execution']['state_max_age_s'])
            if packet.get('armed') or packet.get('arming'):
                raise ClientError('本机数据桥已启用或正在启用，禁止只预测诊断')
            state_received = self.clock()
            captured = min(captured, state_received - float(packet['state_age_s']),
                           state_received - float(packet['gripper_age_s']))
            if not self._preview_generation_current(generation):
                return None
            encode_started = self.clock()
            images = {key: encode_image(frames[key]['rgb'], quality=self.config['cameras']['jpeg_quality'])
                      for key in CAMERA_KEYS}
            request = {'type': 'infer', 'version': PROTOCOL_VERSION, 'session_id': str(uuid.uuid4()),
                       'request_id': request_id, 'capture_seq': sequence, 'capture_monotonic': captured,
                       'observation': {'state': measured, 'task': self.config['task'], 'images': images}}
            if not self._preview_generation_current(generation):
                return None
            started = self.clock()
            response = self.model.request(request)
            completed = self.clock()
            if not self._preview_generation_current(generation):
                return None
            validate_actions_response(response, request=request, model_id=metadata['model_id'])
            actions = response['actions']
            if len(actions) != 50:
                raise ClientError('PI05 只预测诊断需要完整的 50×8 动作')
            age = completed - captured
            aligned = math.ceil(max(0., age) * self.config['execution']['fps'] - 1e-9)
            target = list(actions[aligned]) if aligned < len(actions) else None
            deltas = [a-b for a, b in zip(target[:7], measured[:7])] if target is not None else None
            joint = max(range(7), key=lambda i: abs(deltas[i])) if deltas is not None else None
            jpeg_sizes = {}
            for key, envelope in images.items():
                data = envelope.get('data', '')
                jpeg_sizes[key] = len(data) * 3 // 4 - (len(data) - len(data.rstrip('=')))
            report = {
                'mode': 'preview_only_no_robot_commands',
                'sample_monotonic': sample_time, 'sample_wall_time': sample_wall,
                'capture_monotonic': captured, 'completed_monotonic': completed,
                'request_id': request_id, 'capture_seq': sequence,
                'state': measured, 'q_rad': measured[:7], 'gripper_width_m': measured[7],
                'camera_source_age_s': {key: sample_time-float(frames[key]['monotonic']) for key in CAMERA_KEYS},
                'camera_clock_domains': {key: frames[key].get('source_timestamp_domain') for key in CAMERA_KEYS},
                'camera_frame_numbers': {key: frames[key].get('number') for key in CAMERA_KEYS},
                'state_age_s': packet['state_age_s'], 'gripper_age_s': packet['gripper_age_s'],
                'encoded_jpeg_bytes': jpeg_sizes, 'rpc_s': completed-started,
                'encode_plus_rpc_s': completed-encode_started, 'inference_s': response.get('inference_s'),
                'observation_age_s': age, 'aligned_index': aligned, 'aligned_target': target,
                'aligned_target_delta_rad': deltas,
                'max_joint_delta_rad': None if joint is None else abs(deltas[joint]),
                'max_joint_delta_joint': None if joint is None else joint+1,
                'actions': actions, 'model_id': metadata['model_id'], 'checkpoint': metadata['checkpoint'],
            }
            with self._lock:
                if not self._preview_generation_current(generation):
                    return None
                self._emit('preview_inference', report=report)
            return report
        except Exception as exc:
            with self._lock:
                if generation is None or self._preview_generation_current(generation):
                    self._emit('operation_error', operation='preview_inference', message=str(exc))
            return None
        finally:
            if owns_preview:
                with self._lock:
                    self._preview_pending = False

    def _manual_generation_current(self, generation, operation_id):
        with self._lock:
            return (generation == self._generation and self._manual_pending
                    and self._manual_operation_id == operation_id and self._connected
                    and not self._stop_all.is_set() and not self._armed
                    and self._session is None and not self._arm_pending
                    and not self._preview_pending)

    def open_gripper(self):
        """Worker-only explicit opening; never enable arm control or inference.

        The UUID is allocated before any I/O. Stop independently revokes that
        UUID even when the manual request has not reached the bridge yet, so a
        delayed request cannot start a Move after Stop. A completed action is
        only reported as fully open after fresh actual width reaches 79 mm.
        """
        generation = operation_id = None
        owns_manual = False
        try:
            with self._lock:
                if not self._connected or self._stop_all.is_set():
                    raise ClientError('请先连接并预览，再手动打开夹爪')
                if (self._armed or self._session is not None or self._arm_pending
                        or self._preview_pending or self._manual_pending):
                    raise ClientError('手动开爪仅可在未启用、无其他操作的预览状态运行')
                operation_id = str(uuid.uuid4())
                generation = self._generation
                self._manual_pending = owns_manual = True
                self._manual_operation_id = operation_id
            # Leave time for a bounded local RPC (normally <= 2 x 0.4 s for
            # connect and receive) beyond this six-second polling deadline.
            deadline = self.clock() + 6.0
            state = self.bridge.request({'version': PROTOCOL_VERSION, 'type': 'state'})
            if not self._manual_generation_current(generation, operation_id):
                return None
            check_state(state, self.config['execution']['state_max_age_s'])
            if state.get('armed') or state.get('arming'):
                raise ClientError('本机数据桥已启用或正在启用，禁止手动开爪')
            if self.clock() >= deadline:
                raise ClientError('手动开爪等待超过 6 秒')
            if not self._manual_generation_current(generation, operation_id):
                return None
            # Do not hold _lock around transport: Stop must send its independent
            # revoke request while this request is still waiting for its reply.
            reply = self.bridge.request({'version': PROTOCOL_VERSION, 'type': 'manual_open',
                                         'operation_id': operation_id})
            if not self._manual_generation_current(generation, operation_id):
                return None
            if self.clock() >= deadline:
                raise ClientError('手动开爪等待超过 6 秒')
            if (reply.get('version') != PROTOCOL_VERSION
                    or reply.get('type') != 'manual_open_started'
                    or reply.get('operation_id') != operation_id):
                raise ClientError('手动开爪启动回复的协议或操作编号不匹配')
            while self.clock() < deadline:
                if not self._manual_generation_current(generation, operation_id):
                    return None
                state = self.bridge.request({'version': PROTOCOL_VERSION, 'type': 'state'})
                if not self._manual_generation_current(generation, operation_id):
                    return None
                if self.clock() >= deadline:
                    raise ClientError('手动开爪等待超过 6 秒；未确认实际全开')
                measured = check_state(state, self.config['execution']['state_max_age_s'])
                if state.get('armed') or state.get('arming'):
                    raise ClientError('手动开爪过程中机械臂控制被启用')
                manual = state.get('manual_open')
                if manual is not None:
                    if not isinstance(manual, dict) or manual.get('operation_id') != operation_id:
                        raise ClientError('手动开爪操作已被其他操作替换')
                    target = manual.get('target_width_m')
                    if (isinstance(target, bool) or not isinstance(target, (int, float))
                            or not math.isfinite(target) or abs(target - .08) > 1e-6):
                        raise ClientError('手动开爪目标必须为总开口 0.08 m')
                    status = manual.get('status')
                    if status in ('error', 'cancelled'):
                        raise ClientError(f'手动开爪{status}: {manual.get("error") or manual}')
                    if status not in ('pending', 'accepted', 'completed'):
                        raise ClientError(f'未知手动开爪状态: {status}')
                    if status == 'completed' and measured[7] >= .079:
                        with self._lock:
                            if not self._manual_generation_current(generation, operation_id):
                                return None
                            self._state = state
                            self._emit('manual_gripper_opened', state=state, width_m=measured[7],
                                       operation_id=operation_id)
                        return state
                with self._lock:
                    if not self._manual_generation_current(generation, operation_id):
                        return None
                    self._state = state
                self._stop_all.wait(min(.05, max(0., deadline - self.clock())))
            raise ClientError('手动开爪等待超过 6 秒；未确认实际全开')
        except Exception as exc:
            with self._lock:
                if generation is None or self._manual_generation_current(generation, operation_id):
                    if owns_manual:
                        self.stop('手动开爪失败: ' + str(exc))
                    self._emit('operation_error', operation='open_gripper', message=str(exc),
                               operation_id=operation_id)
            return None
        finally:
            if owns_manual:
                with self._lock:
                    # A late old worker must not clear a newer operation's lock.
                    if self._manual_operation_id == operation_id:
                        self._manual_pending = False
                        self._manual_operation_id = None

    def _is_session(self, token, armed=False):
        with self._lock:
            return self._session == token and (self._armed or not armed) and not self._stop_all.is_set()

    def _send_stop(self, token, reason, manual_operation_id=None):
        try:
            packet = {'version': PROTOCOL_VERSION, 'type': 'stop',
                      'session_id': token, 'reason': reason}
            if manual_operation_id is not None:
                packet['manual_operation_id'] = manual_operation_id
            self.bridge.request(packet)
        except Exception as exc:
            self._emit('stop_error', message=f'本机停止请求未获确认: {exc}；请使用现场停止开关')

    def stop(self, reason='用户停止', *, expected_session=None):
        # This happens immediately even if a cloud request is currently blocked.
        with self._lock:
            if expected_session is not None and self._session != expected_session:
                return False
            token, self._session, self._armed = self._session, None, False
            manual_operation_id = self._manual_operation_id if self._manual_pending else None
            self._heartbeat_ready = False
            self._arm_monotonic = None
            self._generation += 1
            self._last_fault = reason
            recording, listener = self._recording, self._recording_listener
            self._recording_listener = None
        if listener is not None:
            self.cameras.remove_frame_listener(listener)
        if recording is not None:
            recording.stop(reason)
        self.model.cancel()
        if token is not None or manual_operation_id is not None:
            threading.Thread(target=self._send_stop, args=(token, reason, manual_operation_id), daemon=True).start()
        self._emit('stopped', reason=reason)
        return True

    def _monitor_loop(self):
        period = min(float(self.config['execution'].get('heartbeat_period_s', 0.1)), 0.1)
        while not self._stop_all.wait(period):
            with self._lock:
                token, armed = self._session, self._armed
                heartbeat_ready = self._heartbeat_ready
            try:
                if token is not None and heartbeat_ready:
                    frames, errors = self.cameras.snapshot()
                    captured = check_cameras(frames, errors, self.config['cameras'], self.clock())
                    reply = self.bridge.request({'version': PROTOCOL_VERSION, 'type': 'heartbeat',
                                                 'session_id': token, 'camera_ok': True,
                                                 'camera_capture_monotonic': captured})
                    if reply.get('fault'):
                        raise ClientError(str(reply['fault']))
                state = self.bridge.request({'version': PROTOCOL_VERSION, 'type': 'state'})
                with self._lock:
                    if self._stop_all.is_set():
                        return
                    self._state = state
                if token is not None and heartbeat_ready and self._is_session(token):
                    check_state(state, self.config['execution']['state_max_age_s'])
                    if state.get('fault'):
                        raise ClientError(str(state['fault']))
                    if armed and (not state.get('armed') or state.get('session_id') != token):
                        raise ClientError('本机执行会话已停止或被替换')
            except Exception as exc:
                if self._stop_all.is_set():
                    return
                if token is not None and heartbeat_ready and self._is_session(token):
                    self.stop(str(exc))
                self._emit('status_error', message=str(exc))

    def _inference_loop(self):
        while not self._stop_all.wait(0.01):
            with self._lock:
                token = self._session if self._armed else None
                metadata = self._metadata
                armed_monotonic = self._arm_monotonic
            if token is None:
                continue
            # Diagnostics belong to one observation only. A subsequent camera or
            # transport failure must never retain the preceding chunk's reply.
            request = reply = None
            duration = age = bridge_submit_monotonic = None
            try:
                frames, errors = self.cameras.snapshot()
                captured = check_cameras(frames, errors, self.config['cameras'], self.clock())
                state_packet = self.bridge.request({'version': PROTOCOL_VERSION, 'type': 'state'})
                state = check_state(state_packet, self.config['execution']['state_max_age_s'])
                if state_packet.get('session_id') != token or not state_packet.get('armed'):
                    raise ClientError('本机执行会话未启用')
                # Measured state and both images belong to this same host clock.
                captured = min(captured, self.clock() - float(state_packet['state_age_s']),
                               self.clock() - float(state_packet['gripper_age_s']))
                if armed_monotonic is not None and captured < armed_monotonic:
                    # All source samples must be newer than controller activation.
                    # Keep heartbeating while waiting; never send a pre-arm image.
                    continue
                with self._lock:
                    self._capture_seq += 1
                    self._request_id += 1
                    sequence, request_id = self._capture_seq, self._request_id
                request = {'type': 'infer', 'version': PROTOCOL_VERSION, 'session_id': token,
                           'request_id': request_id, 'capture_seq': sequence,
                           'capture_monotonic': captured,
                           'observation': {'state': state, 'task': self.config['task'],
                                           'images': {key: encode_image(frames[key]['rgb'],
                                                      quality=self.config['cameras']['jpeg_quality'])
                                                      for key in CAMERA_KEYS}}}
                if not self._is_session(token, armed=True):
                    continue
                started = self.clock()
                reply = self.model.request(request)
                duration = self.clock() - started
                # A late reply from an old session can never be sent to the bridge.
                if not self._is_session(token, armed=True):
                    continue
                validate_actions_response(reply, request=request, model_id=metadata['model_id'])
                age = self.clock() - captured
                if age > self.config['execution']['max_observation_age_s']:
                    raise ClientError(f'远程动作已过期: 观测年龄 {age:.3f}s，网络/推理耗时 {duration:.3f}s')
                if not self._is_session(token, armed=True):
                    continue
                submission = dict(reply, type='submit')
                bridge_submit_monotonic = self.clock()
                accepted = self.bridge.request(submission)
                if accepted.get('fault') or accepted.get('accepted') is False:
                    raise ClientError(str(accepted.get('fault') or accepted.get('error') or '动作被数据桥拒绝'))
                width_diagnostics = {key: copy.deepcopy(accepted[key]) for key in (
                    'width_clipped_steps', 'width_out_of_range_steps', 'width_saturation', 'raw_width_min_m', 'raw_width_max_m',
                    'raw_width_min', 'raw_width_max') if key in accepted}
                with self._lock:
                    if not self._is_session(token, armed=True):
                        continue
                    self._emit('inference', request_s=duration, observation_age_s=age,
                               inference_s=reply.get('inference_s'), request_id=request_id,
                               **width_diagnostics)
            except Exception as exc:
                with self._lock:
                    session_current = self._is_session(token)
                    current_session_id = self._session
                    if session_current:
                        # Revoke execution before queueing any diagnostic event.
                        self.stop(f'执行停止: {exc}')
                    # The monitor may observe the bridge's fault and stop before
                    # this submitted RPC returns its rejection. Keep that raw
                    # evidence labelled stale; never stop or alter a new session.
                    # A cancelled cloud response that never reached submit is
                    # still discarded entirely.
                    if (request is not None and reply is not None
                            and (session_current or bridge_submit_monotonic is not None)):
                        context = {key: request[key] for key in (
                            'session_id', 'request_id', 'capture_seq', 'capture_monotonic')}
                        context.update(state=list(request['observation']['state']),
                                       task=request['observation']['task'],
                                       model={key: copy.deepcopy(metadata[key]) for key in (
                                           'model_id', 'checkpoint', 'fps', 'chunk_size',
                                           'state_dim', 'action_dim', 'action_semantics') if key in metadata})
                        context.update({key: value for key, value in (
                            ('request_s', duration), ('observation_age_s', age),
                            ('bridge_submit_monotonic', bridge_submit_monotonic)) if value is not None})
                        self._emit('action_rejected', error=str(exc), request_context=context,
                                   response=copy.deepcopy(reply), session_id=token,
                                   session_current=session_current, stale=not session_current,
                                   current_session_id=current_session_id)

    def close(self, reason='关闭部署窗口'):
        self._stop_all.set()
        self.stop(reason)
        with self._lock:
            self._connected = False
            self._arm_pending = self._preview_pending = self._manual_pending = False
            self._manual_operation_id = None
            if self._state is not None:
                self._state = dict(self._state, ready=False, armed=False, arming=False,
                                   manual_open_ready=False, fault=reason,
                                   state_age_s=None, gripper_age_s=None)
        # Camera SDK shutdown is bounded and never performed in the Tk event loop.
        threading.Thread(target=self.cameras.close, daemon=True).start()


def replay_observation(dataset_root, episode_index=0, frame_index=0):
    """Read one official v3 observation, without touching ROS or live cameras."""
    try:
        import cv2
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ClientError('回放检查需要 opencv-python 和 pyarrow；也可在原训练 uv 环境运行') from exc
    root = Path(dataset_root).expanduser().resolve()
    info = json.loads((root / 'meta/info.json').read_text(encoding='utf-8'))
    if info.get('codebase_version') != 'v3.0' or info.get('fps') != 15:
        raise ClientError('回放需要 15 FPS 的 LeRobot v3.0 数据集')
    if not isinstance(episode_index, int) or episode_index < 0 or not isinstance(frame_index, int) or frame_index < 0:
        raise ClientError('episode 和 frame 必须为非负整数')
    episodes = [row for path in sorted((root / 'meta/episodes').glob('chunk-*/file-*.parquet'))
                for row in pq.read_table(path).to_pylist() if row.get('episode_index') == episode_index]
    if len(episodes) != 1:
        raise ClientError(f'找不到唯一 episode {episode_index}')
    episode = episodes[0]
    if frame_index >= episode['length']:
        raise ClientError(f'frame 超出 episode 长度 {episode["length"]}')
    relative = info['data_path'].format(chunk_index=episode['data/chunk_index'], file_index=episode['data/file_index'])
    rows = pq.read_table(root / relative, columns=['episode_index', 'frame_index', 'observation.state']).to_pylist()
    matches = [row for row in rows if row['episode_index'] == episode_index and row['frame_index'] == frame_index]
    if len(matches) != 1:
        raise ClientError('数据索引不唯一')
    images = {}
    for key in CAMERA_KEYS:
        relative = info['video_path'].format(video_key=key, chunk_index=episode[f'videos/{key}/chunk_index'],
                                             file_index=episode[f'videos/{key}/file_index'])
        video = cv2.VideoCapture(str(root / relative))
        try:
            absolute_frame = round(episode[f'videos/{key}/from_timestamp'] * info['fps']) + frame_index
            if not video.isOpened() or not video.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame):
                raise ClientError(f'无法定位视频帧: {relative}:{absolute_frame}')
            success, bgr = video.read()
            if not success:
                raise ClientError(f'无法解码视频帧: {relative}:{absolute_frame}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape != (480, 640, 3):
                raise ClientError(f'相机分辨率与训练不匹配: {rgb.shape}')
            images[key] = rgb
        finally:
            video.release()
    return {'state': matches[0]['observation.state'], 'images': images, 'task': 'Put the avocado in the basket.'}


def infer_replay(config, dataset_root, episode_index=0, frame_index=0):
    observation = replay_observation(dataset_root, episode_index, frame_index)
    mc = config['model']
    # Read-only diagnosis may measure a slow WAN/model call. Live execution
    # retains its configured short RPC and observation deadlines independently.
    client = RPCClient(mc['host'], mc['port'], max(mc['request_timeout_s'], 30.0), mc['connect_timeout_s'])
    metadata = client.request({'type': 'hello', 'version': PROTOCOL_VERSION})
    validate_metadata(metadata)
    encode_started = time.monotonic()
    request = {'version': PROTOCOL_VERSION, 'type': 'infer', 'session_id': str(uuid.uuid4()),
               'request_id': 1, 'capture_seq': 1, 'capture_monotonic': time.monotonic(),
               'observation': {**observation,
                               'images': {key: encode_image(rgb, quality=config['cameras']['jpeg_quality'])
                                          for key, rgb in observation['images'].items()}}}
    rpc_started = time.monotonic()
    result = client.request(request)
    completed = time.monotonic()
    validate_actions_response(result, request=request, model_id=metadata['model_id'])
    return {'mode': 'replay_only_no_robot_connection', 'metadata': metadata,
            'episode': episode_index, 'frame': frame_index,
            'timing': {'rpc_s': completed - rpc_started,
                       'encode_plus_rpc_s': completed - encode_started},
            'response': result}
