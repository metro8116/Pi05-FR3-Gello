#!/usr/bin/env python3
"""FR3 local preview/execution GUI, or robot-free recorded-observation inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import signal
import sys
import threading
import time

from config import load_config
from client_runtime import DeploymentRuntime, infer_replay
from protocol import CAMERA_KEYS


class DeploymentWindow:
    def __init__(self, root, config, session_status=None):
        import tkinter as tk
        from tkinter import messagebox
        self.tk, self.messagebox, self.root = tk, messagebox, root
        self.events, self.config = queue.Queue(), config
        self.runtime = DeploymentRuntime(config, self._event)
        self.session_status = session_status
        self.session_failure = None
        self.closing, self.busy, self.photo_refs = False, False, []
        self.quit_requested = threading.Event()
        root.title('FR3 / PI05 远程部署 — 默认仅预览')
        width = min(1140, root.winfo_screenwidth() - 60)
        height = min(840, root.winfo_screenheight() - 80)
        # Give Tk a stable outer size. Live status text must not resize the
        # window whenever a float changes length or a line starts wrapping.
        root.geometry(f'{width}x{height}')
        root.minsize(min(1060, width), min(760, height))
        self.status = tk.StringVar(value='未连接。先启动 AutoDL 推理服务、SSH 隧道和本机模型控制桥。')
        self.session_info = tk.StringVar(value='')
        self.model_info = tk.StringVar(value='模型：未连接')
        self.robot_info = tk.StringVar(value='FR3 实际状态：未连接')
        self.camera_info = tk.StringVar(value='相机：未连接')
        self.timing = tk.StringVar(value='尚未请求在线推理')
        self.recording_info = tk.StringVar(value='录像：启用模型执行时自动录制双相机；停止时保存。')
        self.status_label = tk.Label(root, textvariable=self.status, anchor='nw', height=2,
                                     wraplength=width-24, fg='#154777')
        self.status_label.pack(fill='x', padx=12, pady=8)
        self.session_label = tk.Label(root, textvariable=self.session_info, anchor='nw', height=4,
                                      wraplength=width-24, justify='left', fg='#a32626')
        for variable, lines in ((self.model_info, 3), (self.robot_info, 4),
                                (self.camera_info, 3), (self.timing, 1), (self.recording_info, 2)):
            tk.Label(root, textvariable=variable, anchor='nw', height=lines,
                     wraplength=width-24, justify='left').pack(fill='x', padx=12)
        previews = tk.Frame(root)
        previews.pack(padx=12, pady=8)
        self.preview_labels = []
        for i, serial in enumerate(config['cameras']['serials']):
            column = tk.Frame(previews)
            column.grid(row=0, column=i, padx=4)
            tk.Label(column, text=f'camera_{i+1} / {serial}').pack()
            viewport = tk.Frame(column, width=480, height=360, bg='#20252a')
            viewport.pack()
            viewport.pack_propagate(False)
            label = tk.Label(viewport, text='等待图像', bg='#20252a', fg='white')
            label.pack(fill='both', expand=True)
            self.preview_labels.append(label)
        buttons = tk.Frame(root)
        buttons.pack(pady=10)
        self.connect_button = tk.Button(buttons, text='连接并预览', command=self.connect, width=20)
        self.connect_button.grid(row=0, column=0, padx=5)
        self.predict_button = tk.Button(buttons, text='仅预测（不运动）', command=self.predict,
                                        width=20, state='disabled')
        self.predict_button.grid(row=0, column=1, padx=5)
        self.open_button = tk.Button(buttons, text='打开夹爪（80 mm）', command=self.open_gripper,
                                     width=20, state='disabled')
        self.open_button.grid(row=0, column=2, padx=5)
        self.arm_button = tk.Button(buttons, text='启用模型执行', command=self.arm, width=20, state='disabled')
        self.arm_button.grid(row=0, column=3, padx=5)
        self.stop_button = tk.Button(buttons, text='停止模型执行', command=self.stop, width=20,
                                     bg='#b52f31', fg='white')
        self.stop_button.grid(row=0, column=4, padx=5)
        tk.Label(root, text='任务：' + config['task'] + '    |    夹爪：Move，0.04 m/s    |    动作：15 Hz\n'
                 '“打开夹爪”会释放当前物体；仅在模型停用时可用。', anchor='w', justify='left').pack(fill='x', padx=12, pady=8)
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.update)

    def _event(self, event):
        # Keep robot-free predictions and execution failures inspectable from
        # client.log; no image bytes or credentials are included in events.
        print(json.dumps({'client_wall_time': time.time(), **event}, ensure_ascii=False), flush=True)
        self.events.put(event)

    def _run(self, fn):
        if self.busy or self.session_failure is not None:
            return
        self.busy = True
        def worker():
            try:
                fn()
            except Exception as exc:
                self._event({'kind': 'operation_error', 'message': str(exc)})
            finally:
                self._event({'kind': 'operation_done'})
        threading.Thread(target=worker, daemon=True).start()

    def connect(self):
        self.status.set('正在连接；此操作不会启用机械臂运动。')
        self._run(self.runtime.connect)

    def predict(self):
        self.status.set('正在对当前画面和状态做单次预测；不会启用控制器或发送动作。')
        self._run(self.runtime.preview_inference)

    def open_gripper(self):
        self.status.set('正在将夹爪打开到 80 mm，速度 0.04 m/s；机械臂控制器保持停用。')
        self._run(self.runtime.open_gripper)

    def arm(self):
        if self.busy or self.session_failure is not None:
            return
        accepted = self.messagebox.askokcancel(
            '启用 FR3 模型执行',
            '确认当前场地已清空、你在现场且可操作停止开关；GELLO 遥操已退出。\n\n'
            '模型将控制机械臂和夹爪。点击“确定”开始本次执行会话。', parent=self.root)
        if accepted:
            self.status.set('正在准备双相机录像；录像就绪后启用本次模型执行会话…')
            self._run(self.runtime.arm)

    def stop(self):
        if self.session_failure is not None:
            return
        self.runtime.stop('用户点击停止')
        self.status.set('已取消本次执行；正在向本机控制桥发送停止请求。')

    def check_session_failure(self):
        if self.session_status is None:
            return
        try:
            record = json.loads(Path(self.session_status).read_text())
        except (OSError, ValueError):
            return
        if (not isinstance(record, dict) or record.get('version') != 1
                or record.get('phase') not in ('stopping', 'stopped')
                or not isinstance(record.get('reason'), str)):
            return
        if self.session_failure is None:
            self.session_failure = dict(record)
            # Terminal fence: late network replies cannot reconnect cameras or
            # arm this old window after the supervisor tears its services down.
            self.runtime.close('部署会话故障: ' + record['reason'])
            self.session_label.pack(fill='x', padx=12, after=self.status_label)
            self.root.title('FR3 / PI05 — 故障已停用')
        self.session_failure['phase'] = record['phase']

    def update(self):
        if self.closing:
            return
        if self.quit_requested.is_set():
            self.close()
            return
        self.check_session_failure()
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event['kind']
            if kind == 'connected':
                meta = event['metadata']
                self.model_info.set(f'模型：{meta.get("model_id")} | mock={meta.get("mock")}\ncheckpoint：{meta.get("checkpoint")}')
                self.status.set('已连接，仅预览。按“启用模型执行”才会开始控制。')
            elif kind == 'armed':
                self.status.set('模型执行中；现场停止开关始终可用。')
            elif kind == 'inference':
                self.timing.set(f'请求 #{event["request_id"]}：往返 {event["request_s"]:.3f}s | '
                                f'观测年龄 {event["observation_age_s"]:.3f}s | 模型推理 {event.get("inference_s")}s')
            elif kind == 'preview_inference':
                report = event['report']
                self.timing.set(f'仅预测：往返 {report["rpc_s"]:.3f}s | '
                                f'观测年龄 {report["observation_age_s"]:.3f}s | '
                                f'对齐目标最大偏差 J{report["max_joint_delta_joint"]} '
                                f'{report["max_joint_delta_rad"]:.3f} rad')
                self.status.set('单次预测完成，机械臂保持停用；详细目标与偏差已写入 client.log。')
            elif kind == 'manual_gripper_opened':
                self.status.set(f'夹爪已打开，实际开口 {event["width_m"] * 1000:.1f} mm；模型仍停用。')
            elif kind == 'operation_done':
                self.busy = False
            elif kind == 'stopped':
                self.status.set('模型已停用：' + event['reason'])
            elif kind in ('operation_error', 'status_error', 'stop_error'):
                self.status.set(event['message'])
            elif kind == 'recording_error' and not event.get('stale'):
                self.status.set('录像失败，模型已停用：' + event['message'])
        snapshot = self.runtime.snapshot()
        terminal = self.session_failure is not None
        self.connect_button.configure(state='disabled' if terminal or self.busy or snapshot['connected'] else 'normal')
        self.predict_button.configure(state='normal' if not terminal and snapshot['connected'] and not self.busy and
                                       not snapshot['armed'] and not snapshot['arming'] and
                                       not snapshot.get('manual_opening') else 'disabled')
        self.open_button.configure(state='normal' if not terminal and snapshot['connected'] and not self.busy and
                                   not snapshot['armed'] and not snapshot['arming'] and
                                   not snapshot.get('manual_opening') and
                                   (snapshot.get('recording') or {}).get('status') not in ('starting', 'saving') and
                                   (snapshot.get('state') or {}).get('manual_open_ready') else 'disabled')
        self.arm_button.configure(state='normal' if not terminal and snapshot['connected'] and not self.busy and
                                   not snapshot['armed'] and not snapshot['arming'] and
                                   not snapshot.get('manual_opening') and
                                   (snapshot.get('recording') or {}).get('status') not in ('starting', 'saving') and
                                   (snapshot.get('state') or {}).get('ready') and
                                   snapshot['metadata'].get('mock') is False else 'disabled')
        self.stop_button.configure(state='disabled' if terminal else 'normal')
        recording = snapshot.get('recording')
        if recording:
            counts = [(recording.get('frames') or {}).get(key, 0) for key in CAMERA_KEYS]
            dropped = [(recording.get('dropped') or {}).get(key, 0) for key in CAMERA_KEYS]
            names = {'starting': '准备中', 'recording': '录像中', 'saving': '保存中',
                     'saved': '已保存', 'error': '录像失败'}
            title = names.get(recording.get('status'), recording.get('status'))
            self.recording_info.set(f'录像：{title} | 两路帧数 {counts} | 丢帧 {dropped}\n'
                                    f'{recording.get("error") or recording.get("directory", "")}')
        state = snapshot['state']
        if state:
            if not terminal and snapshot['armed']:
                self.status.set('等待夹爪停止，机械臂正在减速保持。' if state.get('arm_holding_for_gripper_stop')
                                else '模型执行中；现场停止开关始终可用。')
            vector = state.get('state')
            vector_text = (', '.join(f'{value:.4f}' for value in vector)
                           if vector is not None else '等待状态')
            def age_text(value):
                return f'{value:.3f}' if isinstance(value, (int, float)) else '—'
            self.robot_info.set(f'FR3：ready={state.get("ready")} | armed={state.get("armed")} | '
                                f'关节年龄={age_text(state.get("state_age_s"))}s | '
                                f'夹爪年龄={age_text(state.get("gripper_age_s"))}s\n'
                                f'实际状态(rad × 7, m)：[{vector_text}]\n故障：{state.get("fault")}')
        frames, errors = self.runtime.cameras.snapshot()
        ages = [(time.monotonic() - frames[key]['monotonic']) if key in frames else None for key in CAMERA_KEYS]
        skew = abs(frames[CAMERA_KEYS[0]]['monotonic'] - frames[CAMERA_KEYS[1]]['monotonic']) if len(frames) == 2 else None
        domains = [frames[key].get('source_timestamp_domain') if key in frames else None for key in CAMERA_KEYS]
        self.camera_info.set(f'相机帧年龄：{[round(a,3) if a is not None else None for a in ages]}s | '
                             f'双相机采集时间差：{round(skew,3) if skew is not None else None}s | '
                             f'源时钟：{domains} | {errors or ""}')
        if frames:
            try:
                import cv2
                photos = []
                for key, label in zip(CAMERA_KEYS, self.preview_labels):
                    if key not in frames:
                        continue
                    rgb = cv2.resize(frames[key]['rgb'], (480, 360))
                    ppm = b'P6\n480 360\n255\n' + rgb.tobytes()
                    photo = self.tk.PhotoImage(data=ppm, format='PPM')
                    label.configure(image=photo, text='')
                    photos.append(photo)
                self.photo_refs = photos
            except Exception as exc:
                self.camera_info.set('预览显示错误: ' + str(exc))
        if terminal:
            failure = self.session_failure
            phase = 'ROS 与隧道已停止' if failure['phase'] == 'stopped' else '正在停止 ROS 与隧道'
            self.status.set('部署已故障停用；' + phase + '。窗口保留供查看。')
            self.session_info.set(f'原因：{failure["reason"]}\n'
                                  f'日志：{failure.get("logs", "")}（{failure.get("source", "supervisor")}）\n'
                                  '关闭此窗口后可重新启动；本窗口不会重新连接或启用控制。')
            self.camera_info.set('相机已停用，正在释放本次相机资源。')
        self.root.after(100, self.update)

    def close(self):
        if not self.closing:
            self.closing = True
            self.runtime.close()
            self._close_deadline = time.monotonic() + 9.
            # Revoke control immediately; only window destruction waits for the
            # asynchronous video trailer/metadata, never the native Stop path.
            self.root.after(450, self._finish_close)

    def _finish_close(self):
        if self.runtime.wait_recording(0):
            self.root.destroy()
        elif time.monotonic() >= self._close_deadline:
            self._event(dict(kind='recording_error', message='录像收尾超时；请检查录像目录与metadata'))
            self.root.destroy()
        else:
            self.status.set('模型已停用，正在保存双相机录像…')
            self.root.after(100, self._finish_close)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--session-status', type=Path, help='监督进程写入的本次会话故障状态')
    parser.add_argument('--replay-dataset', type=Path, help='只对录制数据做远程推理，不连接相机/ROS/机械臂')
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--replay-output', type=Path, help='保存只读回放推理结果 JSON')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.replay_dataset:
        result = infer_replay(config, args.replay_dataset, args.episode, args.frame)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        print(rendered)
        if args.replay_output:
            args.replay_output.expanduser().write_text(rendered + '\n', encoding='utf-8')
        return 0
    # A preview window owns local camera/control resources until it closes.
    # The read-only replay path deliberately never acquires this hardware lease.
    from control_ownership import ControlLease
    lease = ControlLease.from_environment('pi05')
    import tkinter as tk
    window = None
    previous_signals = {}
    try:
        root = tk.Tk()
        window = DeploymentWindow(root, config, session_status=args.session_status)
        def request_close(signum, frame):
            window.quit_requested.set()
        previous_signals = {number: signal.signal(number, request_close)
                            for number in (signal.SIGINT, signal.SIGTERM)}
        root.mainloop()
    finally:
        if window is not None:
            window.runtime.close()
            window.runtime.cameras.close()
            window.runtime.wait_recording(1.)
        lease.release()
        for number, handler in previous_signals.items():
            signal.signal(number, handler)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
